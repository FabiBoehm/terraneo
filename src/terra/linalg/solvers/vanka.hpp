#pragma once

#include "solver.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/integrands.hpp"
#include "terra/fe/wedge/kernel_helpers.hpp"
#include "terra/fe/wedge/quadrature/quadrature.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/linalg/vector_q1isoq2_q1.hpp"

namespace terra::linalg::solvers {

/// @brief Vertex-centered additive Vanka smoother for Stokes-type saddle-point systems.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Iterates over fine grid vertices. For each vertex, assembles the fully assembled
/// diagonal block from ALL cells sharing that vertex (up to 8 for interior nodes),
/// solving a small local system:
/// - 3x3 for velocity-only vertices
/// - 4x4 saddle-point at coarse-grid-aligned vertices (velocity + pressure DOF)
///
/// Since each vertex is visited exactly once, corrections are applied directly
/// without atomic operations or overlap weighting.
///
/// @tparam StokesOperatorT The Stokes operator type (must satisfy Block2x2OperatorLike).
template < Block2x2OperatorLike StokesOperatorT >
class Vanka
{
  public:
    using OperatorType       = StokesOperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;
    using ScalarType         = typename SolutionVectorType::ScalarType;

    static_assert(
        Block2VectorLike< SolutionVectorType >,
        "Vanka requires block 2-vectors (velocity + pressure)." );

  private:
    static constexpr int VecDim = 3;

    grid::shell::DistributedDomain       domain_fine_;
    grid::Grid3DDataVec< ScalarType, 3 > grid_;
    grid::Grid2DDataScalar< ScalarType > radii_;

    int        iterations_;
    ScalarType omega_;
    bool       treat_boundary_;

    SolutionVectorType tmp_;

    // Grid dimensions
    int cells_x_;
    int cells_y_;
    int cells_r_;

    // Kernel views (set before each kernel launch)
    grid::Grid4DDataVec< ScalarType, VecDim > res_vel_;
    grid::Grid4DDataScalar< ScalarType >      res_pre_;
    grid::Grid4DDataVec< ScalarType, VecDim > sol_vel_;
    grid::Grid4DDataScalar< ScalarType >      sol_pre_;

  public:
    Vanka(
        const grid::shell::DistributedDomain&        domain_fine,
        const grid::Grid3DDataVec< ScalarType, 3 >&  grid,
        const grid::Grid2DDataScalar< ScalarType >&  radii,
        int                                          iterations,
        SolutionVectorType&                          tmp,
        ScalarType                                   omega          = 0.5,
        bool                                         treat_boundary = true )
    : domain_fine_( domain_fine )
    , grid_( grid )
    , radii_( radii )
    , iterations_( iterations )
    , omega_( omega )
    , treat_boundary_( treat_boundary )
    , tmp_( tmp )
    {
        cells_x_ = domain_fine.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
        cells_y_ = cells_x_;
        cells_r_ = domain_fine.domain_info().subdomain_num_nodes_radially() - 1;
    }

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iter = 0; iter < iterations_; ++iter )
        {
            // Compute global residual: tmp = b - A*x
            apply( A, x, tmp_ );
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );

            // Store views for kernel access
            res_vel_ = tmp_.block_1().grid_data();
            res_pre_ = tmp_.block_2().grid_data();
            sol_vel_ = x.block_1().grid_data();
            sol_pre_ = x.block_2().grid_data();

            // Launch kernel over fine grid VERTICES (not cells)
            Kokkos::parallel_for(
                "vanka_vertex", grid::shell::local_domain_md_range_policy_nodes( domain_fine_ ), *this );
            Kokkos::fence();
        }
    }

    /// @brief Kokkos kernel: per-vertex Vanka local solve.
    KOKKOS_INLINE_FUNCTION void
        operator()( const int sd, const int x, const int y, const int r ) const
    {
        using namespace fe::wedge;

        // Wedge-to-hex node mapping
        constexpr int w2h[2][6] = { { 0, 1, 2, 4, 5, 6 }, { 3, 2, 1, 7, 6, 5 } };

        // Skip boundary nodes (Dirichlet BCs - correction is zero)
        if ( treat_boundary_ && ( r == 0 || r == cells_r_ ) )
            return;

        const bool is_coarse = ( x % 2 == 0 ) && ( y % 2 == 0 ) && ( r % 2 == 0 );

        // ===== 1. Assemble velocity diagonal from all cells sharing this vertex =====
        // For vector Laplacian, K_vel(d1,d2) = a * delta(d1,d2), so we just compute scalar a.

        ScalarType a = 0.0;      // assembled velocity diagonal
        ScalarType b[VecDim];    // assembled coupling velocity->pressure (only for coarse nodes)
        for ( int d = 0; d < VecDim; ++d )
            b[d] = 0.0;

        // Loop over all cells containing vertex (x, y, r)
        for ( int dx = -1; dx <= 0; ++dx )
        {
            const int cx = x + dx;
            if ( cx < 0 || cx >= cells_x_ )
                continue;

            for ( int dy = -1; dy <= 0; ++dy )
            {
                const int cy = y + dy;
                if ( cy < 0 || cy >= cells_y_ )
                    continue;

                // Gather lateral surface geometry for cell (cx, cy)
                dense::Vec< ScalarType, 3 > phy_surf[2][num_nodes_per_wedge_surface] = {};
                wedge_surface_physical_coords( phy_surf, grid_, sd, cx, cy );

                for ( int dr = -1; dr <= 0; ++dr )
                {
                    const int cr = r + dr;
                    if ( cr < 0 || cr >= cells_r_ )
                        continue;

                    // Hex node index of (x,y,r) within cell (cx, cy, cr)
                    const int hi = 4 * ( r - cr ) + 2 * ( y - cy ) + ( x - cx );

                    const ScalarType r_1 = radii_( sd, cr );
                    const ScalarType r_2 = radii_( sd, cr + 1 );

                    // --- Velocity diagonal contribution (quad_felippa_3x2) ---
                    {
                        constexpr auto              nq = quadrature::quad_felippa_3x2_num_quad_points;
                        dense::Vec< ScalarType, 3 > qp[nq];
                        ScalarType                  qw[nq];
                        quadrature::quad_felippa_3x2_quad_points( qp );
                        quadrature::quad_felippa_3x2_quad_weights( qw );

                        for ( int wedge = 0; wedge < 2; ++wedge )
                        {
                            // Find local index of hi in this wedge
                            int local_i = -1;
                            for ( int k = 0; k < 6; ++k )
                            {
                                if ( w2h[wedge][k] == hi )
                                {
                                    local_i = k;
                                    break;
                                }
                            }
                            if ( local_i < 0 )
                                continue;

                            for ( int q = 0; q < nq; ++q )
                            {
                                const auto J       = jac( phy_surf[wedge], r_1, r_2, qp[q] );
                                const auto det_val = J.det();
                                const auto abs_det = Kokkos::abs( det_val );
                                const auto J_inv_t = J.inv_transposed( det_val );

                                const auto grad_phy = J_inv_t * grad_shape( local_i, qp[q] );
                                a += qw[q] * grad_phy.dot( grad_phy ) * abs_det;
                            }
                        }
                    }

                    // --- Coupling contribution (quad_felippa_1x1) ---
                    if ( is_coarse )
                    {
                        const int px = x / 2;
                        const int py = y / 2;
                        const int pr = r / 2;

                        // Pressure hex node within coarse cell (cx/2, cy/2, cr/2)
                        const int hj = 4 * ( pr - cr / 2 ) + 2 * ( py - cy / 2 ) + ( px - cx / 2 );

                        // Check hj is valid (0-7)
                        if ( hj < 0 || hj > 7 )
                            continue;

                        constexpr auto              nq = quadrature::quad_felippa_1x1_num_quad_points;
                        dense::Vec< ScalarType, 3 > qp[nq];
                        ScalarType                  qw[nq];
                        quadrature::quad_felippa_1x1_quad_points( qp );
                        quadrature::quad_felippa_1x1_quad_weights( qw );

                        const int fine_radial_wedge_index = cr % 2;

                        for ( int wedge = 0; wedge < 2; ++wedge )
                        {
                            // Both hi and hj must be in this wedge
                            int local_i = -1, local_j = -1;
                            for ( int k = 0; k < 6; ++k )
                            {
                                if ( w2h[wedge][k] == hi )
                                    local_i = k;
                                if ( w2h[wedge][k] == hj )
                                    local_j = k;
                            }
                            if ( local_i < 0 || local_j < 0 )
                                continue;

                            const int fine_lat_wedge_index = fine_lateral_wedge_idx( cx, cy, wedge );

                            for ( int q = 0; q < nq; ++q )
                            {
                                const auto J       = jac( phy_surf[wedge], r_1, r_2, qp[q] );
                                const auto det_val = Kokkos::abs( J.det() );
                                const auto J_inv_t = J.inv().transposed();

                                const auto grad_i = grad_shape( local_i, qp[q] );
                                const auto shape_j = shape_coarse(
                                    local_j, fine_radial_wedge_index, fine_lat_wedge_index, qp[q] );

                                for ( int d = 0; d < VecDim; ++d )
                                {
                                    b[d] += qw[q] * ( -( J_inv_t * grad_i )( d ) * shape_j ) * det_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ===== 2. Gather residual =====
        ScalarType rhs_vel[VecDim];
        for ( int d = 0; d < VecDim; ++d )
            rhs_vel[d] = res_vel_( sd, x, y, r, d );

        // ===== 3. Solve and apply correction =====

        if ( a <= 0.0 )
            return; // degenerate node, skip

        if ( is_coarse )
        {
            // 4x4 saddle-point system:
            // [aI   b ] [du]   [r_u]
            // [b^T -ε ] [dp] = [r_p]
            //
            // where aI is a*I_{3x3}, b is 3x1 coupling vector

            const ScalarType rhs_pre = res_pre_( sd, x / 2, y / 2, r / 2 );
            const ScalarType eps     = 1e-2 * a;
            const ScalarType a_inv   = 1.0 / a;

            // Schur complement: S = b^T * (aI)^{-1} * b + eps = (b.b)/a + eps
            ScalarType b_dot_b = 0.0;
            for ( int d = 0; d < VecDim; ++d )
                b_dot_b += b[d] * b[d];

            const ScalarType S = b_dot_b * a_inv + eps;

            // dp = (b^T * (aI)^{-1} * r_u - r_p) / S
            ScalarType bT_ainv_ru = 0.0;
            for ( int d = 0; d < VecDim; ++d )
                bT_ainv_ru += b[d] * rhs_vel[d] * a_inv;

            const ScalarType dp = ( bT_ainv_ru - rhs_pre ) / S;

            // du_d = (r_u_d - b_d * dp) / a
            for ( int d = 0; d < VecDim; ++d )
            {
                const ScalarType du = ( rhs_vel[d] - b[d] * dp ) * a_inv;
                sol_vel_( sd, x, y, r, d ) += omega_ * du;
            }
            sol_pre_( sd, x / 2, y / 2, r / 2 ) += omega_ * dp;
        }
        else
        {
            // 3x3 diagonal system: du_d = r_u_d / a
            const ScalarType a_inv = 1.0 / a;
            for ( int d = 0; d < VecDim; ++d )
            {
                sol_vel_( sd, x, y, r, d ) += omega_ * rhs_vel[d] * a_inv;
            }
        }
    }
};

} // namespace terra::linalg::solvers
