#pragma once

// Domain-level 2:1 face exchange (additive assembly) for AMR, single communicator / all-local neighbors.
//
// Drives the per-interface kernels in adaptive_face_ops.hpp across a whole adaptive DistributedDomain:
// for every subdomain and face it pulls the shared face slab(s) out of the field, runs the matching
// kernel, and writes the assembled values back. Templated on a field accessor with Grid4D's exact
// (subdomain, x, y, r) indexing, so it is faithful to the real storage yet runs on the host (no Kokkos)
// for testing; a device version applies the same logic to Grid4D slices.
//
// Handles the interior cases only (Milestone A keeps 2:1 interfaces off diamond seams, so no index
// inversion). Face-interior assembly; edge/vertex DoFs need the separate edge/vertex exchange.
//
//   rel_level 0  : matched same-level faces -> node-wise sum on face-INTERIOR nodes (edge/vertex DoFs
//                  are shared by >2 subdomains and belong to the separate edge/vertex exchange)
//   rel_level +1 : this (coarse) subdomain gathers its up-to-4 finer neighbours (P^T assembly +
//                  broadcast of the assembled values to the fine coincident nodes)
//   rel_level -1 : this (fine) subdomain's coarse-facing face is handled by the coarse side; skipped
//
// The operations are materialized as FLAT TABLES (build_2to1_tables): plain (subdomain,x,y,r) index
// tuples + weights, built once per mesh. The same tables drive the host appliers here and the Kokkos
// device kernels in adaptive_2to1_kokkos.hpp, so host and device semantics are identical by
// construction. All appliers are two-phase (gather then scatter), so results do not depend on entry
// order even when a node is source in one interface and destination in another.

#include <array>
#include <map>
#include <set>
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
            c[fa.a1axis]         = a1;
            c[fa.a2axis]         = a2;
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
            c[fa.a1axis]                  = a1;
            c[fa.a2axis]                  = a2;
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

// Flat 2:1 operation tables. Built once per mesh (cache and reuse; rebuilding per call is test-only).
struct TwoToOneTables
{
    // 2:1 additive assembly: dst(coarse) += w * src(fine). Several entries may share a dst (atomic on
    // device); sources are read before any write (two-phase).
    std::vector< Idx4 >   asm_dst, asm_src;
    std::vector< double > asm_w;

    // post-assembly broadcast: dst(fine coincident) = src(coarse assembled). Unique dst per entry.
    std::vector< Idx4 > bc_dst, bc_src;

    // same-level matched sum over face-interior nodes: a and b both become a+b. Disjoint pairs.
    std::vector< Idx4 > pair_a, pair_b;

    // hanging-node constraint: dst(fine hanging) = sum_p w[p] * src[p](coarse). Unique dst per entry
    // (a fine face-edge node hanging on two coarse faces keeps its first entry; the interpolations
    // agree once the edge exchange has made the coarse edges consistent).
    std::vector< Idx4 >                con_dst;
    std::vector< std::array< Idx4, 4 > >  con_src;
    std::vector< std::array< double, 4 > > con_w;
    std::vector< int >                 con_np;
};

inline TwoToOneTables build_2to1_tables( const DistributedDomain& dom, int nx, int ny, int nr )
{
    static constexpr Face kFaces[6] = { Face::XLOW, Face::XHIGH, Face::YLOW,
                                        Face::YHIGH, Face::RLOW, Face::RHIGH };
    TwoToOneTables                   t;
    std::set< std::array< int, 4 > > seen_bc, seen_con;

    for ( const auto& [anchor, tup] : dom.subdomains() )
    {
        const int   sub = std::get< 0 >( tup );
        const auto& nbh = dom.adaptive_neighborhood( anchor );

        for ( Face fc : kFaces )
        {
            const FaceAxes fa = face_axes( fc, nx, ny, nr );
            for ( const auto& nb : nbh.faces )
            {
                if ( nb.my_face != fc )
                    continue;

                if ( nb.rel_level == +1 )
                {
                    const int      fsub = local_index( dom, nb.anchor );
                    const FaceAxes nfa  = face_axes( nb.neighbor_face, nx, ny, nr );
                    for ( const auto& nd : face_correspondence( fa.n1, fa.n2, nb.sub_octant ) )
                    {
                        const Idx4                 fine = face_node( fsub, nfa, nd.a1, nd.a2 );
                        const std::array< int, 4 > key{ fine.s, fine.x, fine.y, fine.r };

                        for ( int p = 0; p < nd.n_parents; ++p )
                        {
                            t.asm_dst.push_back( face_node( sub, fa, nd.pa1[p], nd.pa2[p] ) );
                            t.asm_src.push_back( fine );
                            t.asm_w.push_back( nd.w[p] );
                        }
                        if ( nd.coincident )
                        {
                            if ( seen_bc.insert( key ).second )
                            {
                                t.bc_dst.push_back( fine );
                                t.bc_src.push_back( face_node( sub, fa, nd.pa1[0], nd.pa2[0] ) );
                            }
                        }
                        else if ( seen_con.insert( key ).second )
                        {
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
                else if ( nb.rel_level == 0 && anchor < nb.anchor ) // each pair processed once
                {
                    const int      osub = local_index( dom, nb.anchor );
                    const FaceAxes ofa  = face_axes( nb.neighbor_face, nx, ny, nr );
                    for ( int a1 = 1; a1 < fa.n1 - 1; ++a1 )
                        for ( int a2 = 1; a2 < fa.n2 - 1; ++a2 )
                        {
                            t.pair_a.push_back( face_node( sub, fa, a1, a2 ) );
                            t.pair_b.push_back( face_node( osub, ofa, a1, a2 ) );
                        }
                }
            }
        }
    }
    return t;
}

// Host application of the exchange tables (assembly -> broadcast -> same-level sums).
template < typename FieldT >
inline void apply_exchange_tables( const TwoToOneTables& t, FieldT& field )
{
    std::vector< double > tmp( t.asm_w.size() );
    for ( std::size_t i = 0; i < t.asm_w.size(); ++i )
    {
        const Idx4& s = t.asm_src[i];
        tmp[i]        = t.asm_w[i] * field( s.s, s.x, s.y, s.r );
    }
    for ( std::size_t i = 0; i < t.asm_w.size(); ++i )
    {
        const Idx4& d = t.asm_dst[i];
        field( d.s, d.x, d.y, d.r ) += tmp[i];
    }
    for ( std::size_t i = 0; i < t.bc_dst.size(); ++i )
    {
        const Idx4& d = t.bc_dst[i];
        const Idx4& s = t.bc_src[i];
        field( d.s, d.x, d.y, d.r ) = field( s.s, s.x, s.y, s.r );
    }
    for ( std::size_t i = 0; i < t.pair_a.size(); ++i )
    {
        const Idx4&  a   = t.pair_a[i];
        const Idx4&  b   = t.pair_b[i];
        const double sum = field( a.s, a.x, a.y, a.r ) + field( b.s, b.x, b.y, b.r );
        field( a.s, a.x, a.y, a.r ) = sum;
        field( b.s, b.x, b.y, b.r ) = sum;
    }
}

// Host application of the hanging-node constraint: overwrite each hanging fine DoF with the
// interpolation of its coarse parents. Idempotent (destinations are never sources of other entries in
// a 2:1-balanced mesh without staircases; two-phase makes it order-independent regardless).
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

// Perform the additive 2:1 face exchange over all local subdomains of an adaptive domain.
// Convenience wrapper: builds the tables and applies them. For repeated use (every operator apply),
// build the tables once with build_2to1_tables and call apply_exchange_tables directly.
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
