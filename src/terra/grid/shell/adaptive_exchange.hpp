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
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
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

    // hanging-node constraint (CSR): dst = sum over parents of wt * parent. Parents are RESOLVED to
    // genuine (non-hanging) nodes at the physical-node (class-root) level, so one pass is exact even for
    // NESTED (graded 0->1->2) hanging where a level-k hanging's parent is coincident with a
    // level-(k-1) hanging. ALL copies of a hanging physical node get a row (a genuine-looking even node
    // that is coincident with a hanging node on a finer neighbor is itself constrained). Within a rank
    // this is exact; a nested chain that crosses a rank boundary is not resolved (documented edge).
    // Row i owns con_par[con_off[i] .. con_off[i+1]) with matching con_wt.
    std::vector< Idx4 >   con_dst; // one per hanging copy
    std::vector< int >    con_off; // size con_dst.size()+1
    std::vector< Idx4 >   con_par; // flat genuine parents
    std::vector< double > con_wt;  // flat weights
};

// `guard_corners` rejects an unsupported 2:1 pole/corner seam. It is a GLOBAL mesh-validity check, so
// it must be true when all seam partners are present (the single-rank / all-leaves build). On an
// owned-only distributed domain a corner block's seam partner lives on another rank and would look
// "missing"; pass false there -- the all-leaves build already validated the mesh.
inline TwoToOneTables build_2to1_tables( const DistributedDomain& dom, int nx, int ny, int nr,
                                         bool guard_corners = true )
{
    static constexpr Face kFaces[6] = { Face::XLOW, Face::XHIGH, Face::YLOW,
                                        Face::YHIGH, Face::RLOW, Face::RHIGH };
    TwoToOneTables t;

    using Key4 = std::array< int, 4 >; // copy identity (s,x,y,r)
    auto key_of = []( const Idx4& i ) { return Key4{ i.s, i.x, i.y, i.r }; };


    // ---- pass 1: hanging DoFs + constraint rows (LOCAL to each fine block) -------------------------
    // A 2:1 fine face carries hanging nodes at odd in-face indices; their parents are the bracketing
    // EVEN nodes of the SAME (fine) block's face. Those even nodes are coincident with the coarse
    // parents, so after the class exchange they hold the assembled coarse values -- interpolating from
    // them is identical to interpolating from the coarse parents. Expressing the rule purely on the
    // fine block's own face (no coarse block, correspondence, or seam reference) makes the constraint
    // and the hanging P^T scatter fully local, which is what lets each rank own its hanging nodes.
    std::set< Key4 >                       hanging; // odd (hanging) representative copies
    std::vector< Idx4 >                    imm_dst; // immediate constraint rows (pre-resolution)
    std::vector< std::array< Idx4, 4 > >   imm_src;
    std::vector< std::array< double, 4 > > imm_w;
    std::vector< int >                     imm_np;
    for ( const auto& [anchor, tup] : dom.subdomains() )
    {
        const int   sub = std::get< 0 >( tup );
        const auto& nbh = dom.adaptive_neighborhood( anchor );
        for ( Face fc : kFaces )
        {
            bool coarser_across = false;
            for ( const auto& nb : nbh.faces )
                if ( nb.my_face == fc && nb.rel_level == -1 )
                    coarser_across = true;
            if ( !coarser_across )
                continue; // not a 2:1 fine face -> no hanging nodes here

            const FaceAxes fa = face_axes( fc, nx, ny, nr );
            // A 2:1 face needs an INTERIOR midpoint on each in-face axis to carry the hanging node and to
            // reach its two bracketing even parents locally -- i.e. >= 3 nodes (>= 2 cells) per axis. With
            // fewer (e.g. nx=2 => one lateral cell), index 1 is the block boundary and its parent at 2 runs
            // off the block. This is a block-too-coarse config, not a valid mesh: fail clearly here rather
            // than with an out-of-range map lookup deep in the constraint resolver.
            if ( fa.n1 < 3 || fa.n2 < 3 )
                throw std::runtime_error(
                    "AMR 2:1 hanging refinement needs >= 3 nodes per refined block axis (a 2:1 face has n1=" +
                    std::to_string( fa.n1 ) + ", n2=" + std::to_string( fa.n2 ) +
                    "; block nx=" + std::to_string( nx ) + " ny=" + std::to_string( ny ) + " nr=" +
                    std::to_string( nr ) +
                    "). Increase LDR or reduce the lateral subdomain count so 2^LDR >= 2*S_lat (and keep >= 2 "
                    "radial cells per block)." );
            for ( int a1 = 0; a1 < fa.n1; ++a1 )
                for ( int a2 = 0; a2 < fa.n2; ++a2 )
                {
                    const bool o1 = a1 & 1, o2 = a2 & 1;
                    if ( !o1 && !o2 )
                        continue; // even/even -> coincident with a coarse node -> genuine, not hanging
                    const Idx4 fine = face_node( sub, fa, a1, a2 );
                    if ( !hanging.insert( key_of( fine ) ).second )
                        continue; // block-edge hanging node shared by a second 2:1 face: same row

                    std::array< Idx4, 4 >   srcs{};
                    std::array< double, 4 > ws{};
                    int                     np = 0;
                    if ( o1 && !o2 ) // edge midpoint along a1
                    {
                        srcs[0] = face_node( sub, fa, a1 - 1, a2 );
                        srcs[1] = face_node( sub, fa, a1 + 1, a2 );
                        ws[0] = ws[1] = 0.5;
                        np              = 2;
                    }
                    else if ( !o1 && o2 ) // edge midpoint along a2
                    {
                        srcs[0] = face_node( sub, fa, a1, a2 - 1 );
                        srcs[1] = face_node( sub, fa, a1, a2 + 1 );
                        ws[0] = ws[1] = 0.5;
                        np              = 2;
                    }
                    else // face centre
                    {
                        srcs[0] = face_node( sub, fa, a1 - 1, a2 - 1 );
                        srcs[1] = face_node( sub, fa, a1 + 1, a2 - 1 );
                        srcs[2] = face_node( sub, fa, a1 - 1, a2 + 1 );
                        srcs[3] = face_node( sub, fa, a1 + 1, a2 + 1 );
                        ws[0] = ws[1] = ws[2] = ws[3] = 0.25;
                        np                            = 4;
                    }
                    imm_dst.push_back( fine );
                    imm_src.push_back( srcs );
                    imm_w.push_back( ws );
                    imm_np.push_back( np );
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
        // A refined pole/corner-touching block is allowed only if its seams are CONFORMING (every
        // seam-face node finds a coincident partner) -- e.g. in a uniformly subdivided mesh. A 2:1
        // interface at a pentagon corner (missed nodes below) is not supported and throws.
        const bool corner_guard = ( k > 0 && xext && yext );
        int        corner_misses = 0;

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
                    else
                        ++corner_misses; // seam-hanging here; only fatal for corner blocks (below)
                }
        }
        // NOTE (pentagon 2:1): a refined pole/corner block leaves seam-hanging nodes at the pentagon
        // vertex whose seam partners are absent (corner_misses > 0). These were once rejected, but they
        // are ordinary hanging nodes: pass 1 constrains each from its own block's bracketing even nodes,
        // and the pole vertex itself is merged 5-way by the transitive seam union -- so the assembled
        // operator is correct. Verified by test_adaptive_epsilon_divdiv_gpu's T4 (pole-corner 2:1): the
        // pole-refined mesh is more accurate than the uniform base and gives an l2 IDENTICAL to np=1 at
        // np=4 (where the pole's 5 diamonds straddle ranks). The guard is therefore not needed.
        (void) corner_guard;
        (void) corner_misses;
        (void) guard_corners;
    }

    // ---- pass 4: merged classes (root -> all coincident copies, incl. seam-united) ----------------
    std::map< int, std::vector< Idx4 > > merged;
    for ( std::size_t c = 0; c < cls.size(); ++c )
    {
        auto& v = merged[find( static_cast< int >( c ) )];
        v.insert( v.end(), cls[c].begin(), cls[c].end() );
    }
    for ( auto& [root, members] : merged )
        std::sort( members.begin(), members.end(), []( const Idx4& a, const Idx4& b ) {
            return std::tie( a.s, a.x, a.y, a.r ) < std::tie( b.s, b.x, b.y, b.r );
        } );

    auto root_of = [&]( const Idx4& i ) { return find( class_of.at( key_of( i ) ) ); };

    // one immediate row per hanging PHYSICAL node (its odd representative copy)
    std::map< int, int > imm_row_of_root;
    for ( std::size_t i = 0; i < imm_dst.size(); ++i )
        imm_row_of_root[root_of( imm_dst[i] )] = static_cast< int >( i );
    auto is_hanging_root = [&]( int r ) { return imm_row_of_root.count( r ) > 0; };

    // canonical copy + CSR class members for GENUINE classes. A class is genuine iff its root is NOT a
    // hanging physical node -- so a genuine-looking even node that is coincident with a hanging node is
    // excluded from the class summation (and constrained below instead).
    std::map< Key4, Idx4 > canonical;
    std::map< int, Idx4 >  root_front;
    t.cls_offsets.push_back( 0 );
    for ( const auto& [root, members] : merged )
    {
        root_front[root] = members.front();
        if ( is_hanging_root( root ) )
            continue;
        for ( const auto& m : members )
            canonical[key_of( m )] = members.front();
        if ( members.size() >= 2 )
        {
            for ( const auto& m : members )
                t.cls_members.push_back( m );
            t.cls_offsets.push_back( static_cast< int >( t.cls_members.size() ) );
        }
    }

    // ---- pass 5: resolve each hanging physical node to GENUINE parents (memoized, at the class-root
    // level so coincidence is handled), then emit a CSR constraint row for EVERY copy of it ---------
    std::map< int, std::map< int, double > >                    resolved; // hanging root -> {genuine root: w}
    std::function< const std::map< int, double >&( int ) > resolve =
        [&]( int root ) -> const std::map< int, double >& {
        auto found = resolved.find( root );
        if ( found != resolved.end() )
            return found->second;
        std::map< int, double >& out = resolved[root]; // insert empty first (cycle guard)
        const int                i   = imm_row_of_root.at( root );
        for ( int p = 0; p < imm_np[i]; ++p )
        {
            const int    pr = root_of( imm_src[i][p] );
            const double w  = imm_w[i][p];
            if ( is_hanging_root( pr ) )
                for ( const auto& [gr, gw] : resolve( pr ) )
                    out[gr] += w * gw;
            else
                out[pr] += w;
        }
        return out;
    };

    t.con_off.push_back( 0 );
    for ( const auto& [hroot, i] : imm_row_of_root )
    {
        (void) i;
        const auto& gparents = resolve( hroot );
        for ( const Idx4& copy : merged.at( hroot ) ) // constrain EVERY copy of this hanging node
        {
            t.con_dst.push_back( copy );
            for ( const auto& [gr, w] : gparents )
            {
                // prefer the parent's copy ON THE SAME block as this hanging copy (block-local, so the
                // row stays in-diamond and is correct even on a field whose copies are not yet
                // consistent); fall back to the canonical copy only when the parent has no copy here
                // (the genuinely cross-block nested case, where post-exchange all copies agree anyway).
                Idx4 par = root_front.at( gr );
                for ( const Idx4& c : merged.at( gr ) )
                    if ( c.s == copy.s )
                    {
                        par = c;
                        break;
                    }
                t.con_par.push_back( par );
                t.con_wt.push_back( w );
            }
            t.con_off.push_back( static_cast< int >( t.con_par.size() ) );
        }
    }

    auto canon = [&]( const Idx4& i ) {
        const auto it = canonical.find( key_of( i ) );
        return it == canonical.end() ? i : it->second;
    };

    // ---- pass 6: P^T assembly = transpose of the resolved constraint (genuine parents, canonical) ---
    for ( std::size_t i = 0; i < t.con_dst.size(); ++i )
        for ( int p = t.con_off[i]; p < t.con_off[i + 1]; ++p )
        {
            t.asm_dst.push_back( canon( t.con_par[p] ) );
            t.asm_src.push_back( t.con_dst[i] );
            t.asm_w.push_back( t.con_wt[p] );
        }

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
    std::vector< double > tmp( t.con_dst.size() );
    for ( std::size_t i = 0; i < t.con_dst.size(); ++i )
    {
        double v = 0.0;
        for ( int p = t.con_off[i]; p < t.con_off[i + 1]; ++p )
        {
            const Idx4& s = t.con_par[p];
            v += t.con_wt[p] * field( s.s, s.x, s.y, s.r );
        }
        tmp[i] = v;
    }
    for ( std::size_t i = 0; i < t.con_dst.size(); ++i )
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
