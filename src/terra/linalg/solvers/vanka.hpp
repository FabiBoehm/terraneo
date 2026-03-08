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
/// solving a small 11x11 local saddle-point system:
/// - 3 velocity DOFs (diagonal due to vector Laplacian structure)
/// - 8 pressure DOFs from the containing coarse cell
///
/// Pressure corrections are only applied at coarse-grid-aligned vertices to
/// avoid double-counting.
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
    static constexpr int VecDim   = 3;
    static constexpr int SYS_SIZE = 11; // 3 velocity + 8 pressure

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

    /// @brief Kokkos kernel: per-vertex Vanka local solve with 11x11 saddle-point system.
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

        // Determine containing coarse cell for pressure DOFs
        const int ccx = Kokkos::min( x / 2, cells_x_ / 2 - 1 );
        const int ccy = Kokkos::min( y / 2, cells_y_ / 2 - 1 );
        const int ccr = Kokkos::min( r / 2, cells_r_ / 2 - 1 );

        // ===== 1. Assemble local system entries =====
        // Velocity diagonal: K_vel(d1,d2) = a * delta(d1,d2)
        ScalarType a = 0.0;

        // Coupling matrix B(d, pk) for d=0..2, pk=0..7
        ScalarType B[VecDim][8] = {};

        // Loop over all fine cells containing vertex (x, y, r)
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
                    {
                        const int cell_ccx = cx / 2;
                        const int cell_ccy = cy / 2;
                        const int cell_ccr = cr / 2;

                        constexpr auto              nq = quadrature::quad_felippa_1x1_num_quad_points;
                        dense::Vec< ScalarType, 3 > qp[nq];
                        ScalarType                  qw[nq];
                        quadrature::quad_felippa_1x1_quad_points( qp );
                        quadrature::quad_felippa_1x1_quad_weights( qw );

                        const int fine_radial_wedge_index = cr % 2;

                        // Loop over pressure nodes of this cell's coarse cell
                        for ( int pox = 0; pox <= 1; ++pox )
                        for ( int poy = 0; poy <= 1; ++poy )
                        for ( int por = 0; por <= 1; ++por )
                        {
                            // Pressure node in global coarse coords
                            const int px = cell_ccx + pox;
                            const int py = cell_ccy + poy;
                            const int pr = cell_ccr + por;

                            // Map to local pressure index in our 8-DOF block
                            const int lox = px - ccx;
                            const int loy = py - ccy;
                            const int lor = pr - ccr;
                            if ( lox < 0 || lox > 1 || loy < 0 || loy > 1 || lor < 0 || lor > 1 )
                                continue;
                            const int pk = 4 * lor + 2 * loy + lox;

                            // Hex node of pressure DOF within fine cell
                            const int hj = 4 * por + 2 * poy + pox;

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

                                const int fine_lat_wedge_index =
                                    fine_lateral_wedge_idx( cx, cy, wedge );

                                for ( int q = 0; q < nq; ++q )
                                {
                                    const auto J = jac( phy_surf[wedge], r_1, r_2, qp[q] );
                                    const auto det_val = Kokkos::abs( J.det() );
                                    const auto J_inv_t = J.inv().transposed();

                                    const auto grad_i  = grad_shape( local_i, qp[q] );
                                    const auto shape_j = shape_coarse(
                                        local_j, fine_radial_wedge_index, fine_lat_wedge_index,
                                        qp[q] );

                                    for ( int d = 0; d < VecDim; ++d )
                                    {
                                        B[d][pk] +=
                                            qw[q] *
                                            ( -( J_inv_t * grad_i )( d ) * shape_j ) * det_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ===== 2. Build and solve 11x11 local system =====
        if ( a <= 0.0 )
            return; // degenerate node, skip

        dense::Mat< ScalarType, SYS_SIZE, SYS_SIZE > M;
        dense::Vec< ScalarType, SYS_SIZE >           rhs;
        M.fill( 0.0 );

        // Velocity block: M(d,d) = a
        for ( int d = 0; d < VecDim; ++d )
            M( d, d ) = a;

        // Coupling blocks: M(d, 3+pk) = B(d,pk) and M(3+pk, d) = B(d,pk)
        for ( int d = 0; d < VecDim; ++d )
            for ( int pk = 0; pk < 8; ++pk )
            {
                M( d, 3 + pk )     = B[d][pk];
                M( 3 + pk, d ) = B[d][pk];
            }

        // Pressure stabilization: M(3+pk, 3+pk) = -eps
        const ScalarType eps = 1e-2 * a;
        for ( int pk = 0; pk < 8; ++pk )
            M( 3 + pk, 3 + pk ) = -eps;

        // Gather residual
        for ( int d = 0; d < VecDim; ++d )
            rhs( d ) = res_vel_( sd, x, y, r, d );

        for ( int pk = 0; pk < 8; ++pk )
        {
            const int lox = pk & 1;
            const int loy = ( pk >> 1 ) & 1;
            const int lor = ( pk >> 2 ) & 1;
            rhs( 3 + pk ) = res_pre_( sd, ccx + lox, ccy + loy, ccr + lor );
        }

        // Solve the 11x11 system in-place
        dense::lu_solve( M, rhs );

        // ===== 3. Apply corrections =====

        // Velocity correction at ALL vertices
        for ( int d = 0; d < VecDim; ++d )
            sol_vel_( sd, x, y, r, d ) += omega_ * rhs( d );

        // Pressure correction ONLY at coarse-aligned vertices (one pressure node)
        if ( is_coarse )
        {
            const int lox = x / 2 - ccx;
            const int loy = y / 2 - ccy;
            const int lor = r / 2 - ccr;
            const int pk  = 4 * lor + 2 * loy + lox;
            sol_pre_( sd, x / 2, y / 2, r / 2 ) += omega_ * rhs( 3 + pk );
        }
    }
};

} // namespace terra::linalg::solvers
