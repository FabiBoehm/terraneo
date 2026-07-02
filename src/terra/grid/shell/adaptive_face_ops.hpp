#pragma once

// 2:1 face transfer kernels for the AMR halo, built on the face-node correspondence.
//
// These operate on a single shared face as a flat n1*n2 slab (row-major: idx = a1*n2 + a2, in-face
// axes a1,a2). They are the per-interface kernels the device halo will run; kept as pure host functions
// on plain arrays so the numerics are login-node testable (the device version applies the identical
// correspondence to Grid4D face slices).
//
//   prolongate_face : coarse slab -> fine slab      (P;   fine = interpolation of coarse)
//   restrict_face   : fine slab  -> coarse slab     (P^T; accumulate, the transpose of P)
//   assemble_face_2to1 : additive 2:1 assembly of one coarse face against its fine neighbours
//                        (coarse += sum_octant P^T(fine); then broadcast assembled genuine DoFs to the
//                         fine coincident nodes; fine hanging nodes are left to the constraint step)
//
// P and P^T are exact transposes by construction, so a V-cycle built on them stays symmetric.

#include <map>
#include <vector>

#include "adaptive_face_transfer.hpp"

namespace terra::grid::shell::amr {

// fine = P * coarse. fine[n] = sum_p w(n,p) * coarse[parent_p], for one 2:1 quadrant `sub_octant`.
inline std::vector< double > prolongate_face( const std::vector< double >& coarse, int n1, int n2,
                                              int sub_octant )
{
    std::vector< double > fine( static_cast< std::size_t >( n1 ) * n2, 0.0 );
    for ( const auto& nd : face_correspondence( n1, n2, sub_octant ) )
    {
        double v = 0.0;
        for ( int p = 0; p < nd.n_parents; ++p )
            v += nd.w[p] * coarse[nd.pa1[p] * n2 + nd.pa2[p]];
        fine[nd.a1 * n2 + nd.a2] = v;
    }
    return fine;
}

// coarse += P^T * fine (accumulate into an existing coarse slab). Transpose of prolongate_face.
inline void restrict_face_add( const std::vector< double >& fine, std::vector< double >& coarse, int n1,
                               int n2, int sub_octant )
{
    for ( const auto& nd : face_correspondence( n1, n2, sub_octant ) )
    {
        const double v = fine[nd.a1 * n2 + nd.a2];
        for ( int p = 0; p < nd.n_parents; ++p )
            coarse[nd.pa1[p] * n2 + nd.pa2[p]] += nd.w[p] * v;
    }
}

// Additive 2:1 assembly of one coarse face against its (up to 4) fine neighbours, keyed by sub_octant.
// In place: `coarse` starts as the coarse subdomain's partial and ends as the assembled genuine DoFs;
// each fine slab's coincident nodes are overwritten with the assembled value (hanging nodes untouched).
inline void assemble_face_2to1( std::vector< double >& coarse,
                                std::map< int, std::vector< double > >& fine_by_octant, int n1, int n2 )
{
    // 1. accumulate P^T of every fine slab into the coarse slab (coarse now holds the assembled DoFs)
    for ( const auto& [oct, fine] : fine_by_octant )
        restrict_face_add( fine, coarse, n1, n2, oct );

    // 2. broadcast assembled genuine DoFs back to the fine coincident nodes
    for ( auto& [oct, fine] : fine_by_octant )
        for ( const auto& nd : face_correspondence( n1, n2, oct ) )
            if ( nd.coincident )
                fine[nd.a1 * n2 + nd.a2] = coarse[nd.pa1[0] * n2 + nd.pa2[0]];
}

} // namespace terra::grid::shell::amr
