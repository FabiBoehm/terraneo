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
//   rel_level 0  : matched same-level faces -> node-wise sum, written to both copies (processed once)
//   rel_level +1 : this (coarse) subdomain gathers its up-to-4 finer neighbours -> assemble_face_2to1
//   rel_level -1 : this (fine) subdomain's coarse-facing face is handled by the coarse side; skipped

#include <map>
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

// Perform the additive 2:1 face exchange over all local subdomains of an adaptive domain.
template < typename FieldT >
inline void exchange_faces_2to1( const DistributedDomain& dom, FieldT& field, int nx, int ny, int nr )
{
    static constexpr Face kFaces[6] = { Face::XLOW, Face::XHIGH, Face::YLOW,
                                        Face::YHIGH, Face::RLOW, Face::RHIGH };

    for ( const auto& [anchor, tup] : dom.subdomains() )
    {
        const int   sub = std::get< 0 >( tup );
        const auto& nbh = dom.adaptive_neighborhood( anchor );

        for ( Face f : kFaces )
        {
            const FaceAxes fa = face_axes( f, nx, ny, nr );

            // gather this face's neighbors of each kind
            std::map< int, std::vector< double > > fine_by_octant; // rel +1
            std::vector< const AdaptiveFaceNeighbor* > same;       // rel  0

            for ( const auto& nb : nbh.faces )
            {
                if ( nb.my_face != f )
                    continue;
                if ( nb.rel_level == +1 )
                {
                    const int      fsub = local_index( dom, nb.anchor );
                    const FaceAxes nfa  = face_axes( nb.neighbor_face, nx, ny, nr );
                    fine_by_octant[nb.sub_octant] = extract_slab( field, fsub, nfa );
                }
                else if ( nb.rel_level == 0 )
                {
                    same.push_back( &nb );
                }
                // rel_level == -1 is handled from the coarse side; skip here.
            }

            // 2:1 assembly (coarse side)
            if ( !fine_by_octant.empty() )
            {
                auto coarse = extract_slab( field, sub, fa );
                assemble_face_2to1( coarse, fine_by_octant, fa.n1, fa.n2 );
                insert_slab( field, sub, fa, coarse );
                for ( const auto& nb : nbh.faces )
                    if ( nb.my_face == f && nb.rel_level == +1 )
                    {
                        const FaceAxes nfa = face_axes( nb.neighbor_face, nx, ny, nr );
                        insert_slab( field, local_index( dom, nb.anchor ), nfa,
                                     fine_by_octant[nb.sub_octant] );
                    }
            }

            // same-level matched sum, processed once per edge (lower anchor drives it)
            for ( const AdaptiveFaceNeighbor* nb : same )
            {
                if ( !( anchor < nb->anchor ) )
                    continue;
                const int      osub = local_index( dom, nb->anchor );
                const FaceAxes ofa  = face_axes( nb->neighbor_face, nx, ny, nr );
                auto           a    = extract_slab( field, sub, fa );
                auto           b    = extract_slab( field, osub, ofa );
                for ( std::size_t i = 0; i < a.size(); ++i )
                    a[i] = b[i] = a[i] + b[i];
                insert_slab( field, sub, fa, a );
                insert_slab( field, osub, ofa, b );
            }
        }
    }
}

} // namespace terra::grid::shell::amr
