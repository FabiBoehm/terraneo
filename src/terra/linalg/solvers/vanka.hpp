#pragma once

#include "solver.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/integrands.hpp"
#include "terra/fe/wedge/kernel_helpers.hpp"
#include "terra/fe/wedge/quadrature/quadrature.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/linalg/vector_q1isoq2_q1.hpp"

namespace terra::linalg::solvers {

/// @brief Additive Vanka smoother for Stokes-type saddle-point systems.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Solves small local saddle-point systems per fine hex cell, coupling velocity and pressure DOFs.
///
/// For each fine hex cell, the local system has:
/// - 8 velocity nodes x VecDim components = VecDim*8 velocity DOFs
/// - 8 coarse pressure nodes = 8 pressure DOFs
/// - Total: VecDim*8 + 8 DOFs (32 for VecDim=3)
///
/// The smoother assembles and solves these local systems via LU factorization,
/// then applies corrections additively (in parallel) with relaxation.
///
/// @tparam StokesOperatorT The Stokes operator type (must satisfy Block2x2OperatorLike).
template < Block2x2OperatorLike StokesOperatorT >
class Vanka
{
  public:
    using OperatorType      = StokesOperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;
    using ScalarType         = typename SolutionVectorType::ScalarType;

    static_assert(
        Block2VectorLike< SolutionVectorType >,
        "Vanka requires block 2-vectors (velocity + pressure)." );

  private:
    static constexpr int VecDim      = 3;
    static constexpr int num_vel_dofs = 8 * VecDim;
    static constexpr int num_pre_dofs = 8;
    static constexpr int N           = num_vel_dofs + num_pre_dofs;

    grid::shell::DistributedDomain     domain_fine_;
    grid::Grid3DDataVec< ScalarType, 3 > grid_;
    grid::Grid2DDataScalar< ScalarType > radii_;

    int        iterations_;
    ScalarType omega_;
    bool       treat_boundary_;

    SolutionVectorType tmp_;

    // Kernel views (set before each kernel launch)
    grid::Grid4DDataVec< ScalarType, VecDim > res_vel_;
    grid::Grid4DDataScalar< ScalarType >      res_pre_;
    grid::Grid4DDataVec< ScalarType, VecDim > sol_vel_;
    grid::Grid4DDataScalar< ScalarType >      sol_pre_;

  public:
    /// @brief Construct a Vanka smoother.
    ///
    /// @param domain_fine Fine grid distributed domain (velocity level).
    /// @param grid Fine grid unit sphere coordinates.
    /// @param radii Fine grid radial node positions.
    /// @param iterations Number of Vanka smoothing iterations.
    /// @param tmp Temporary vector for residual computation (must be allocated).
    /// @param omega Relaxation parameter (typically 0.3-0.7 for additive Vanka).
    /// @param treat_boundary If true, enforce Dirichlet boundary conditions.
    Vanka(
        const grid::shell::DistributedDomain&       domain_fine,
        const grid::Grid3DDataVec< ScalarType, 3 >& grid,
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
    {}

    /// @brief Solve (smooth) the Stokes system using additive Vanka iteration.
    ///
    /// For each iteration:
    /// 1. Compute global residual r = b - A*x
    /// 2. For each fine hex cell (in parallel):
    ///    a. Assemble local 32x32 saddle-point system
    ///    b. Solve via LU factorization
    ///    c. Scatter correction to global solution with relaxation omega
    ///
    /// @param A Stokes operator.
    /// @param x Solution vector (velocity + pressure, updated in-place).
    /// @param b Right-hand side vector.
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

            // Launch Vanka kernel over fine hex cells
            Kokkos::parallel_for(
                "vanka_smooth", grid::shell::local_domain_md_range_policy_cells( domain_fine_ ), *this );
            Kokkos::fence();
        }
    }

    /// @brief Kokkos kernel: per-cell Vanka local solve.
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        using namespace fe::wedge;

        constexpr int num_wedges = num_wedges_per_hex_cell;
        constexpr int num_nodes  = num_nodes_per_wedge;

        // Wedge-to-hex node mapping
        constexpr int w2h[2][6] = { { 0, 1, 2, 4, 5, 6 }, { 3, 2, 1, 7, 6, 5 } };
        constexpr int hex_ox[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
        constexpr int hex_oy[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
        constexpr int hex_or[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

        // Local system matrix and RHS
        dense::Mat< ScalarType, N, N > K;
        dense::Vec< ScalarType, N >    rhs;
        K.fill( 0.0 );
        rhs.fill( 0.0 );

        // ===== 1. Gather geometry =====

        dense::Vec< ScalarType, 3 > wedge_phy_surf[num_wedges][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        const ScalarType r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarType r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // ===== 2. Assemble A_uu (velocity-velocity block, VectorLaplace) =====

        {
            constexpr auto              num_quad = quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > qp[num_quad];
            ScalarType                  qw[num_quad];
            quadrature::quad_felippa_3x2_quad_points( qp );
            quadrature::quad_felippa_3x2_quad_weights( qw );

            for ( int wedge = 0; wedge < num_wedges; ++wedge )
            {
                for ( int q = 0; q < num_quad; ++q )
                {
                    const auto J       = jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                    const auto det_val = J.det();
                    const auto abs_det = Kokkos::abs( det_val );
                    const auto J_inv_t = J.inv_transposed( det_val );

                    dense::Vec< ScalarType, 3 > grad_phy[num_nodes];
                    for ( int k = 0; k < num_nodes; ++k )
                    {
                        grad_phy[k] = J_inv_t * grad_shape( k, qp[q] );
                    }

                    for ( int i = 0; i < num_nodes; ++i )
                    {
                        const int hi = w2h[wedge][i];
                        for ( int j = 0; j < num_nodes; ++j )
                        {
                            const int        hj  = w2h[wedge][j];
                            const ScalarType val = qw[q] * grad_phy[i].dot( grad_phy[j] ) * abs_det;
                            for ( int d = 0; d < VecDim; ++d )
                            {
                                K( hi * VecDim + d, hj * VecDim + d ) += val;
                            }
                        }
                    }
                }
            }
        }

        // ===== 3. Assemble A_up and A_pu (gradient/divergence coupling) =====

        {
            constexpr auto              num_quad = quadrature::quad_felippa_1x1_num_quad_points;
            dense::Vec< ScalarType, 3 > qp[num_quad];
            ScalarType                  qw[num_quad];
            quadrature::quad_felippa_1x1_quad_points( qp );
            quadrature::quad_felippa_1x1_quad_weights( qw );

            const int fine_radial_wedge_index = r_cell % 2;

            for ( int q = 0; q < num_quad; ++q )
            {
                for ( int wedge = 0; wedge < num_wedges; ++wedge )
                {
                    const int fine_lat_wedge_index = fine_lateral_wedge_idx( x_cell, y_cell, wedge );

                    const auto J       = jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                    const auto det_val = Kokkos::abs( J.det() );
                    const auto J_inv_t = J.inv().transposed();

                    for ( int i = 0; i < num_nodes; ++i )
                    {
                        const int  hi     = w2h[wedge][i];
                        const auto grad_i = grad_shape( i, qp[q] );

                        for ( int j = 0; j < num_nodes; ++j )
                        {
                            const int        hj      = w2h[wedge][j];
                            const ScalarType shape_j = shape_coarse(
                                j, fine_radial_wedge_index, fine_lat_wedge_index, qp[q] );

                            for ( int d = 0; d < VecDim; ++d )
                            {
                                const ScalarType val =
                                    qw[q] * ( -( J_inv_t * grad_i )( d ) * shape_j ) * det_val;
                                K( hi * VecDim + d, num_vel_dofs + hj ) += val; // A_up (gradient)
                                K( num_vel_dofs + hj, hi * VecDim + d ) += val; // A_pu (divergence)
                            }
                        }
                    }
                }
            }
        }

        // ===== 4. Extract local residual =====

        // Velocity residual from fine grid
        for ( int i = 0; i < 8; ++i )
        {
            for ( int d = 0; d < VecDim; ++d )
            {
                rhs( i * VecDim + d ) = res_vel_(
                    local_subdomain_id, x_cell + hex_ox[i], y_cell + hex_oy[i], r_cell + hex_or[i], d );
            }
        }

        // Pressure residual from coarse grid
        for ( int i = 0; i < 8; ++i )
        {
            rhs( num_vel_dofs + i ) = res_pre_(
                local_subdomain_id, x_cell / 2 + hex_ox[i], y_cell / 2 + hex_oy[i], r_cell / 2 + hex_or[i] );
        }

        // ===== 5. Apply boundary treatment =====

        if ( treat_boundary_ )
        {
            const bool is_inner = ( r_cell == 0 );
            const bool is_outer = ( r_cell + 1 == static_cast< int >( radii_.extent( 1 ) ) - 1 );

            if ( is_inner || is_outer )
            {
                for ( int i = 0; i < 8; ++i )
                {
                    const bool is_boundary_node =
                        ( is_inner && hex_or[i] == 0 ) || ( is_outer && hex_or[i] == 1 );

                    if ( is_boundary_node )
                    {
                        for ( int d = 0; d < VecDim; ++d )
                        {
                            const int dof = i * VecDim + d;
                            // Zero out row and column, set diagonal to 1
                            for ( int k = 0; k < N; ++k )
                            {
                                K( dof, k ) = 0.0;
                                K( k, dof ) = 0.0;
                            }
                            K( dof, dof ) = 1.0;
                            rhs( dof )    = 0.0;
                        }
                    }
                }
            }
        }

        // ===== 6. Solve local system via LU =====

        dense::lu_solve( K, rhs );

        // ===== 7. Scatter corrections =====

        // Velocity corrections to fine grid
        for ( int i = 0; i < 8; ++i )
        {
            for ( int d = 0; d < VecDim; ++d )
            {
                Kokkos::atomic_add(
                    &sol_vel_(
                        local_subdomain_id, x_cell + hex_ox[i], y_cell + hex_oy[i], r_cell + hex_or[i], d ),
                    omega_ * rhs( i * VecDim + d ) );
            }
        }

        // Pressure corrections to coarse grid
        for ( int i = 0; i < 8; ++i )
        {
            Kokkos::atomic_add(
                &sol_pre_(
                    local_subdomain_id, x_cell / 2 + hex_ox[i], y_cell / 2 + hex_oy[i], r_cell / 2 + hex_or[i] ),
                omega_ * rhs( num_vel_dofs + i ) );
        }
    }
};

} // namespace terra::linalg::solvers
