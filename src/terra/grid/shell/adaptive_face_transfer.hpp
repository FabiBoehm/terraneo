#pragma once

// 2:1 face-node correspondence for AMR: the shared numerical core of the adaptive halo (assembly) and
// the hanging-node constraint.
//
// A coarse block and its 2:1-finer neighbor keep the SAME node count; the fine block covers one quadrant
// of the coarse face. On that shared face, every fine node either coincides with a coarse node or hangs
// between coarse nodes. This dependency-free host helper enumerates, for each fine face-node, its coarse
// parents and weights:
//   * both in-face indices even        -> coincident      (1 parent,  weight 1)
//   * one index odd (edge midpoint)    -> edge-hanging     (2 parents, weight 1/2 each)
//   * both indices odd (face centre)   -> face-centre      (4 parents, weight 1/4 each)
//
// Weights here are TOPOLOGICAL: exact for the linear (P1) lateral direction and for uniform radial
// spacing. Geometry-exact radial weights (norm-based, for non-uniform / tanh radii) come from
// fe/wedge/shell/grid_transfer_linear.hpp:prolongation_linear_weights when this is wired to real fields;
// the index correspondence (which parents) is unchanged by that.

#include <array>
#include <vector>

namespace terra::grid::shell::amr {

// One fine face-node and its coarse-parent stencil (indices in the coarse face's in-face node grid).
struct FineFaceNode
{
    int                    a1, a2;    // this fine node's in-face indices (axes a1, a2)
    bool                   coincident; // true: sits exactly on a coarse node
    int                    n_parents;  // 1, 2, or 4
    std::array< int, 4 >   pa1{};      // coarse parent indices along a1
    std::array< int, 4 >   pa2{};      // coarse parent indices along a2
    std::array< double, 4 > w{};       // weights (sum to 1)
};

// Correspondence for one 2:1 quadrant. n1, n2 = block node counts along the two in-face axes (same for
// coarse and fine blocks). sub_octant in [0,4): q1 = octant&1 selects the a1 half, q2 = (octant>>1)&1
// the a2 half. Requires (n1-1) and (n2-1) even so the 2:1 split is integral.
inline std::vector< FineFaceNode > face_correspondence( int n1, int n2, int sub_octant )
{
    const int H1 = ( n1 - 1 ) / 2; // coarse cells (= coarse node offset) the quadrant spans in a1
    const int H2 = ( n2 - 1 ) / 2;
    const int q1 = sub_octant & 1;
    const int q2 = ( sub_octant >> 1 ) & 1;

    std::vector< FineFaceNode > out;
    out.reserve( static_cast< std::size_t >( n1 ) * n2 );
    for ( int jf = 0; jf < n1; ++jf )
        for ( int kf = 0; kf < n2; ++kf )
        {
            const bool odd1 = jf & 1;
            const bool odd2 = kf & 1;
            const int  jc   = q1 * H1 + jf / 2; // lower coarse node in a1
            const int  kc   = q2 * H2 + kf / 2; // lower coarse node in a2

            FineFaceNode nd;
            nd.a1 = jf;
            nd.a2 = kf;
            nd.coincident = ( !odd1 && !odd2 );

            if ( !odd1 && !odd2 )
            {
                nd.n_parents = 1;
                nd.pa1 = { jc, 0, 0, 0 };
                nd.pa2 = { kc, 0, 0, 0 };
                nd.w   = { 1.0, 0.0, 0.0, 0.0 };
            }
            else if ( odd1 && !odd2 )
            {
                nd.n_parents = 2;
                nd.pa1 = { jc, jc + 1, 0, 0 };
                nd.pa2 = { kc, kc, 0, 0 };
                nd.w   = { 0.5, 0.5, 0.0, 0.0 };
            }
            else if ( !odd1 && odd2 )
            {
                nd.n_parents = 2;
                nd.pa1 = { jc, jc, 0, 0 };
                nd.pa2 = { kc, kc + 1, 0, 0 };
                nd.w   = { 0.5, 0.5, 0.0, 0.0 };
            }
            else
            {
                nd.n_parents = 4;
                nd.pa1 = { jc, jc + 1, jc, jc + 1 };
                nd.pa2 = { kc, kc, kc + 1, kc + 1 };
                nd.w   = { 0.25, 0.25, 0.25, 0.25 };
            }
            out.push_back( nd );
        }
    return out;
}

} // namespace terra::grid::shell::amr
