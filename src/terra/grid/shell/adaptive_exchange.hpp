#pragma once

// Domain-level assembly exchange + hanging-node constraint for AMR (single communicator, all-local).
//
// After a local FE operator apply, every subdomain holds PARTIAL values on shared DoFs. This header
// makes them consistent, for faces, block edges and block vertices alike, via GLOBAL NODE CLASSES:
// every node gets a global coordinate (per axis, in units of 1/(nodes-1) of a finest cell), and all
// coincident copies -- shared by 2 (face), 4 (edge) or 8 (vertex) subdomains, same-level or 2:1 --
// fall into one class. Assembly is then uniform, with correct multiplicity by construction:
//
//   1. asm     : each HANGING fine copy's partial is P^T-scattered to its coarse parents (weights from
//                the 2:1 face correspondence) -- hanging DoFs carry no genuine value of their own.
//   2. classes : each genuine class is summed into a canonical copy (its first member), which then
//                also receives the asm contributions.
//   3. broadcast: every class member is overwritten with the canonical (assembled) value.
//   4. constraint (separate call): every hanging copy = interpolation of its (assembled) parents,
//                making the 2:1 interface conforming.
//
// Templated on a field accessor with Grid4D's exact (subdomain, x, y, r) indexing, so it is faithful
// to the real storage yet host-testable; the Kokkos kernels in adaptive_2to1_kokkos.hpp consume the
// SAME tables, so host and device semantics are identical by construction. All appliers are two-phase
// (gather then scatter) -> results independent of entry order.
//
// Diamond seams ARE handled, including 2:1 seams (refined blocks at diamond boundaries): coincident
// seam nodes are identified node-by-node in global coordinates via the diamond adjacency
// (diamond_seam: same-pole forward, cross-pole reversed) and merged into the classes with a
// union-find; pole corners group transitively around the face-adjacency cycle. Hanging nodes across a
// reversed seam get their fine-side lateral index flipped into the fine block's own frame. The only
// remaining restriction: refining a pole/corner-touching block throws (the 5-way corner cycle is
// untested at 2:1). Deferred: MPI transport (all neighbors local).

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "adaptive_face_ops.hpp"
#include "adaptive_neighborhood.hpp"
#include "spherical_shell.hpp"

namespace terra::grid::shell::amr {

// Which node plane a face occupies and its two in-face axes, given the block node counts.
struct FaceAxes
{
    int axis;   // fixed axis: 0=x,1=y,2=r
    int fixed;  // node index on the fixed axis (0 or extent-1)
    int a1axis; // first in-face axis
    int a2axis; // second in-face axis
    int n1;     // node count along a1
    int n2;     // node count along a2
};

inline FaceAxes face_axes( Face f, int nx, int ny, int nr )
{
    switch ( f )
    {
    case Face::XLOW:  return { 0, 0,      1, 2, ny, nr };
    case Face::XHIGH: return { 0, nx - 1, 1, 2, ny, nr };
    case Face::YLOW:  return { 1, 0,      0, 2, nx, nr };
    case Face::YHIGH: return { 1, ny - 1, 0, 2, nx, nr };
    case Face::RLOW:  return { 2, 0,      0, 1, nx, ny };
    default:          return { 2, nr - 1, 0, 1, nx, ny }; // RHIGH
    }
}

// Gather a face slab (row-major a1*n2 + a2) out of `field` for subdomain `sub`.
template < typename FieldT >
inline std::vector< double > extract_slab( const FieldT& field, int sub, const FaceAxes& fa )
{
    std::vector< double > slab( static_cast< std::size_t >( fa.n1 ) * fa.n2 );
    int                   c[3];
    c[fa.axis] = fa.fixed;
    for ( int a1 = 0; a1 < fa.n1; ++a1 )
        for ( int a2 = 0; a2 < fa.n2; ++a2 )
        {
            c[fa.a1axis]          = a1;
            c[fa.a2axis]          = a2;
            slab[a1 * fa.n2 + a2] = field( sub, c[0], c[1], c[2] );
        }
    return slab;
}

// Scatter a face slab back into `field`.
template < typename FieldT >
inline void insert_slab( FieldT& field, int sub, const FaceAxes& fa, const std::vector< double >& slab )
{
    int c[3];
    c[fa.axis] = fa.fixed;
    for ( int a1 = 0; a1 < fa.n1; ++a1 )
        for ( int a2 = 0; a2 < fa.n2; ++a2 )
        {
            c[fa.a1axis]                   = a1;
            c[fa.a2axis]                   = a2;
            field( sub, c[0], c[1], c[2] ) = slab[a1 * fa.n2 + a2];
        }
}

// Local index of a subdomain (by its finest-frame anchor) in a domain.
inline int local_index( const DistributedDomain& dom, const SubdomainInfo& anchor )
{
    return std::get< 0 >( dom.subdomains().at( anchor ) );
}

// (subdomain, x, y, r) address of one field DoF -- the unit all tables are expressed in.
struct Idx4
{
    int s, x, y, r;
};

// Node (a1, a2) of a face, as a full 4-index.
inline Idx4 face_node( int sub, const FaceAxes& fa, int a1, int a2 )
{
    int c[3];
    c[fa.axis]   = fa.fixed;
    c[fa.a1axis] = a1;
    c[fa.a2axis] = a2;
    return Idx4{ sub, c[0], c[1], c[2] };
}

// Flat assembly/constraint tables. Built once per mesh (cache and reuse; rebuilding per call is
// test-only convenience).
struct TwoToOneTables
{
    // P^T scatter of hanging partials: dst(canonical genuine copy) += w * src(hanging copy). Several
    // entries may share a dst (atomic on device); sources are read before any write (two-phase).
    std::vector< Idx4 >   asm_dst, asm_src;
    std::vector< double > asm_w;

    // coincident-copy classes (CSR): members of class c are cls_members[cls_offsets[c] ..
    // cls_offsets[c+1]). Canonical copy = first member. Only genuine classes with >= 2 members are
    // stored (hanging locations and unshared nodes need no summation).
    std::vector< int >  cls_offsets; // size = n_classes + 1
    std::vector< Idx4 > cls_members;

    // hanging-node constraint: dst(hanging copy) = sum_p w[p] * src[p](canonical parents). One entry
    // per hanging copy.
    std::vector< Idx4 >                    con_dst;
    std::vector< std::array< Idx4, 4 > >   con_src;
    std::vector< std::array< double, 4 > > con_w;
    std::vector< int >                     con_np;
};

inline TwoToOneTables build_2to1_tables( const DistributedDomain& dom, int nx, int ny, int nr )
{
    static constexpr Face kFaces[6] = { Face::XLOW, Face::XHIGH, Face::YLOW,
                                        Face::YHIGH, Face::RLOW, Face::RHIGH };
    TwoToOneTables t;

    using Key4 = std::array< int, 4 >; // copy identity (s,x,y,r)
    auto key_of = []( const Idx4& i ) { return Key4{ i.s, i.x, i.y, i.r }; };

    // ---- pass 1: hanging copies + raw constraint rows (from the 2:1 face correspondence) ----------
    std::set< Key4 > hanging; // all hanging copies (constraint destinations)
    for ( const auto& [anchor, tup] : dom.subdomains() )
    {
        const int   sub = std::get< 0 >( tup );
        const auto& nbh = dom.adaptive_neighborhood( anchor );
        for ( Face fc : kFaces )
        {
            const FaceAxes fa = face_axes( fc, nx, ny, nr );
            for ( const auto& nb : nbh.faces )
            {
                if ( nb.my_face != fc || nb.rel_level != +1 )
                    continue;
                const int      fsub = local_index( dom, nb.anchor );
                const FaceAxes nfa  = face_axes( nb.neighbor_face, nx, ny, nr );
                for ( const auto& nd : face_correspondence( fa.n1, fa.n2, nb.sub_octant ) )
                {
                    if ( nd.coincident )
                        continue;
                    // for reversed seams the fine block's in-face LATERAL index runs backwards
                    // relative to my frame (nd.a1 is in my frame; a2 is radial, never reversed)
                    const int  fa1  = nb.seam_reversed ? nfa.n1 - 1 - nd.a1 : nd.a1;
                    const Idx4 fine = face_node( fsub, nfa, fa1, nd.a2 );
                    if ( !hanging.insert( key_of( fine ) ).second )
                        continue; // a face-edge hanging copy seen via a second coarse face: same row
                    std::array< Idx4, 4 >   srcs{};
                    std::array< double, 4 > ws{};
                    for ( int p = 0; p < nd.n_parents; ++p )
                    {
                        srcs[p] = face_node( sub, fa, nd.pa1[p], nd.pa2[p] );
                        ws[p]   = nd.w[p];
                    }
                    t.con_dst.push_back( fine );
                    t.con_src.push_back( srcs );
                    t.con_w.push_back( ws );
                    t.con_np.push_back( nd.n_parents );
                }
            }
        }
    }

    // ---- pass 2: global node classes (within-diamond coincidence) ---------------------------------
    // global coordinate per axis in units of 1/(nodes-1) of a finest cell:
    //   g = anchor_finest * (nodes-1) + node_idx * finest_span(subdivision)
    const int M = dom.max_subdivision();
    std::map< std::array< long, 4 >, std::vector< Idx4 > > classes; // (diamond,gx,gy,gr) -> copies
    for ( const auto& [anchor, tup] : dom.subdomains() )
    {
        const int  sub  = std::get< 0 >( tup );
        const int  k    = dom.subdivision_of( anchor );
        const long span = 1L << ( M - k );
        for ( int x = 0; x < nx; ++x )
            for ( int y = 0; y < ny; ++y )
                for ( int r = 0; r < nr; ++r )
                {
                    const std::array< long, 4 > g{ anchor.diamond_id(),
                                                   anchor.subdomain_x() * ( nx - 1 ) + x * span,
                                                   anchor.subdomain_y() * ( ny - 1 ) + y * span,
                                                   anchor.subdomain_r() * ( nr - 1 ) + r * span };
                    classes[g].push_back( Idx4{ sub, x, y, r } );
                }
    }

    // provisional class id per copy (all copies, singletons included)
    std::vector< std::vector< Idx4 > > cls;
    std::map< Key4, int >              class_of;
    std::map< std::array< long, 4 >, int > gid; // gcoord -> class id (for seam lookups)
    cls.reserve( classes.size() );
    for ( const auto& [g, members] : classes )
    {
        const int id = static_cast< int >( cls.size() );
        for ( const auto& m : members )
            class_of[key_of( m )] = id;
        gid[g] = id;
        cls.push_back( members );
    }

    // ---- pass 3: diamond-seam identifications ------------------------------------------------------
    // Node-level pairing in GLOBAL coordinates, valid at ANY subdivision mix: a seam-face node's
    // in-face lateral global coordinate u maps to the neighbor diamond as (reversed ? G - u : u),
    // radial unchanged (adjacency + orientation from diamond_seam, transcribed from the uniform pole
    // wiring). If the mapped position holds a node on the other side, the two classes merge
    // (union-find); if not, the node is seam-hanging and pass 1 has already given it a constraint row.
    // Only pole/corner-touching blocks are still forbidden to refine.
    std::vector< int > uf( cls.size() );
    for ( std::size_t i = 0; i < uf.size(); ++i )
        uf[i] = static_cast< int >( i );
    auto find = [&]( int a ) {
        while ( uf[a] != a )
            a = uf[a] = uf[uf[a]];
        return a;
    };
    auto unite = [&]( int a, int b ) {
        a = find( a );
        b = find( b );
        if ( a != b )
            uf[b] = a;
    };

    const int  S_lat = dom.domain_info().num_subdomains_per_diamond_side();
    const long G_lat = static_cast< long >( S_lat << M ) * ( nx - 1 ); // lateral node-coordinate range

    for ( const auto& [anchor, tup] : dom.subdomains() )
    {
        const int sub = std::get< 0 >( tup );
        const int k   = dom.subdivision_of( anchor );
        const int sh  = M - k;
        const int ox  = anchor.subdomain_x() >> sh;
        const int oy  = anchor.subdomain_y() >> sh;
        const int nblk = S_lat << k;

        const bool xext = ( ox == 0 || ox == nblk - 1 );
        const bool yext = ( oy == 0 || oy == nblk - 1 );
        if ( k > 0 && xext && yext )
            throw std::runtime_error(
                "AMR seam assembly: refining pole/corner-touching blocks is not supported." );

        Face seam_faces[2];
        int  n_seam = 0;
        if ( ox == 0 )
            seam_faces[n_seam++] = Face::XLOW;
        else if ( ox == nblk - 1 )
            seam_faces[n_seam++] = Face::XHIGH;
        if ( oy == 0 )
            seam_faces[n_seam++] = Face::YLOW;
        else if ( oy == nblk - 1 )
            seam_faces[n_seam++] = Face::YHIGH;

        const long span = 1L << sh;
        for ( int sf = 0; sf < n_seam; ++sf )
        {
            const Face        f    = seam_faces[sf];
            const DiamondSeam seam = diamond_seam( anchor.diamond_id(), f );
            const FaceAxes    fa   = face_axes( f, nx, ny, nr );
            for ( int a1 = 0; a1 < fa.n1; ++a1 )
                for ( int a2 = 0; a2 < fa.n2; ++a2 )
                {
                    const Idx4 mine = face_node( sub, fa, a1, a2 );
                    const long gx   = anchor.subdomain_x() * ( long ) ( nx - 1 ) + mine.x * span;
                    const long gy   = anchor.subdomain_y() * ( long ) ( ny - 1 ) + mine.y * span;
                    const long gr   = anchor.subdomain_r() * ( long ) ( nr - 1 ) + mine.r * span;

                    // my in-face lateral global coordinate, reversed if the seam flips
                    long u = ( fa.a1axis == 0 ) ? gx : gy;
                    if ( seam.reversed )
                        u = G_lat - u;
                    const int  nb_axis  = ( seam.nb_face == Face::XLOW || seam.nb_face == Face::XHIGH ) ? 0 : 1;
                    const long nb_fixed = ( seam.nb_face == Face::XLOW || seam.nb_face == Face::YLOW ) ? 0 : G_lat;

                    std::array< long, 4 > ng{ seam.nb_diamond, 0, 0, gr };
                    ng[1 + nb_axis]       = nb_fixed;
                    ng[1 + ( 1 - nb_axis )] = u;

                    const auto it = gid.find( ng );
                    if ( it != gid.end() )
                        unite( class_of.at( key_of( mine ) ), it->second );
                    // not found -> no coincident node on the other side (seam-hanging); constraint
                    // handles it via pass 1.
                }
        }
    }

    // ---- pass 4: merged classes -> canonical map + CSR (genuine classes with >= 2 members) --------
    std::map< int, std::vector< Idx4 > > merged;
    for ( std::size_t c = 0; c < cls.size(); ++c )
    {
        auto& v = merged[find( static_cast< int >( c ) )];
        v.insert( v.end(), cls[c].begin(), cls[c].end() );
    }

    std::map< Key4, Idx4 > canonical;
    t.cls_offsets.push_back( 0 );
    for ( auto& [root, members] : merged )
    {
        std::sort( members.begin(), members.end(), []( const Idx4& a, const Idx4& b ) {
            return std::tie( a.s, a.x, a.y, a.r ) < std::tie( b.s, b.x, b.y, b.r );
        } );
        bool is_hanging = false;
        for ( const auto& m : members )
            if ( hanging.count( key_of( m ) ) )
                is_hanging = true;
        if ( is_hanging )
            continue; // hanging locations get values from the constraint, not from summation
        for ( const auto& m : members )
            canonical[key_of( m )] = members.front();
        if ( members.size() >= 2 )
        {
            for ( const auto& m : members )
                t.cls_members.push_back( m );
            t.cls_offsets.push_back( static_cast< int >( t.cls_members.size() ) );
        }
    }

    auto canon = [&]( const Idx4& i ) {
        const auto it = canonical.find( key_of( i ) );
        return it == canonical.end() ? i : it->second;
    };

    // ---- pass 5: derive the P^T assembly from the constraint rows (parents canonicalized) --------
    for ( std::size_t i = 0; i < t.con_np.size(); ++i )
        for ( int p = 0; p < t.con_np[i]; ++p )
        {
            t.asm_dst.push_back( canon( t.con_src[i][p] ) );
            t.asm_src.push_back( t.con_dst[i] );
            t.asm_w.push_back( t.con_w[i][p] );
        }

    // NOTE: con_src deliberately reads the owning coarse face's OWN copies (not canonicalized): after
    // the exchange all class copies are equal anyway, and this keeps the constraint correct even when
    // applied in isolation (e.g. after a plain interpolation instead of an assembly).

    return t;
}

// Host application of the assembly exchange: class sums -> hanging P^T -> broadcast.
template < typename FieldT >
inline void apply_exchange_tables( const TwoToOneTables& t, FieldT& field )
{
    // gather hanging scatters (reads hanging copies -- untouched by every later phase)
    std::vector< double > asm_tmp( t.asm_w.size() );
    for ( std::size_t i = 0; i < t.asm_w.size(); ++i )
    {
        const Idx4& s = t.asm_src[i];
        asm_tmp[i]    = t.asm_w[i] * field( s.s, s.x, s.y, s.r );
    }

    // gather class sums (reads coincident copies before any write)
    const std::size_t     ncls = t.cls_offsets.size() - 1;
    std::vector< double > cls_tmp( ncls );
    for ( std::size_t c = 0; c < ncls; ++c )
    {
        double v = 0.0;
        for ( int m = t.cls_offsets[c]; m < t.cls_offsets[c + 1]; ++m )
        {
            const Idx4& i = t.cls_members[m];
            v += field( i.s, i.x, i.y, i.r );
        }
        cls_tmp[c] = v;
    }

    // scatter: canonical = class sum, then += hanging contributions
    for ( std::size_t c = 0; c < ncls; ++c )
    {
        const Idx4& i = t.cls_members[t.cls_offsets[c]];
        field( i.s, i.x, i.y, i.r ) = cls_tmp[c];
    }
    for ( std::size_t i = 0; i < t.asm_w.size(); ++i )
    {
        const Idx4& d = t.asm_dst[i];
        field( d.s, d.x, d.y, d.r ) += asm_tmp[i];
    }

    // broadcast the assembled canonical value to all other copies
    for ( std::size_t c = 0; c < ncls; ++c )
    {
        const Idx4&  can = t.cls_members[t.cls_offsets[c]];
        const double v   = field( can.s, can.x, can.y, can.r );
        for ( int m = t.cls_offsets[c] + 1; m < t.cls_offsets[c + 1]; ++m )
        {
            const Idx4& i = t.cls_members[m];
            field( i.s, i.x, i.y, i.r ) = v;
        }
    }
}

// Host application of the hanging-node constraint: overwrite each hanging copy with the interpolation
// of its (assembled) coarse parents. Idempotent; two-phase makes it order-independent.
template < typename FieldT >
inline void apply_constraint_tables( const TwoToOneTables& t, FieldT& field )
{
    std::vector< double > tmp( t.con_np.size() );
    for ( std::size_t i = 0; i < t.con_np.size(); ++i )
    {
        double v = 0.0;
        for ( int p = 0; p < t.con_np[i]; ++p )
        {
            const Idx4& s = t.con_src[i][p];
            v += t.con_w[i][p] * field( s.s, s.x, s.y, s.r );
        }
        tmp[i] = v;
    }
    for ( std::size_t i = 0; i < t.con_np.size(); ++i )
    {
        const Idx4& d = t.con_dst[i];
        field( d.s, d.x, d.y, d.r ) = tmp[i];
    }
}

// Perform the additive assembly exchange (faces, edges, vertices; same-level and 2:1) over all local
// subdomains. Convenience wrapper: builds the tables and applies them. For repeated use (every
// operator apply), build the tables once with build_2to1_tables and call apply_exchange_tables.
template < typename FieldT >
inline void exchange_faces_2to1( const DistributedDomain& dom, FieldT& field, int nx, int ny, int nr )
{
    const auto t = build_2to1_tables( dom, nx, ny, nr );
    apply_exchange_tables( t, field );
}

// Apply the hanging-node constraint over all local subdomains: every hanging fine DoF is overwritten
// with the interpolation of its coarse parents, making the interface conforming. Run AFTER the
// exchange (the parents must hold assembled values). Same caching advice as above.
template < typename FieldT >
inline void constrain_hanging_faces( const DistributedDomain& dom, FieldT& field, int nx, int ny, int nr )
{
    const auto t = build_2to1_tables( dom, nx, ny, nr );
    apply_constraint_tables( t, field );
}

} // namespace terra::grid::shell::amr
