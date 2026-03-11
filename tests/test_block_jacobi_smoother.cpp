
/// @brief Compares convergence of block Jacobi vs point Jacobi as stand-alone smoothers
///        on the EpsilonDivDiv (viscous) operator.
///
/// Setup:
///   - EpsilonDivDiv operator on a single-level spherical shell domain with stored local matrices.
///   - Random initial guess, zero RHS  =>  pure error reduction.
///   - Run N iterations of each smoother and track ||r||_2 at each step.
///   - Block Jacobi should converge faster (smaller spectral radius of the iteration matrix)
///     because it captures cross-component coupling (A_xy, A_yx, ...) within each node's block.

#include "fe/wedge/operators/shell/epsilon_divdiv.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/block_jacobi.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using linalg::DiagonallyScaledOperator;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::solvers::power_iteration;

using ScalarType = double;

/// @brief Simple coefficient field: k(x) = 2 + sin(z).
struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords =
            grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        data_( local_subdomain_id, x, y, r ) = 2.0 + Kokkos::sin( coords( 2 ) );
    }
};

/// @brief Initialize a velocity field with some smooth non-trivial function.
struct InitialGuessInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords =
            grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        // Non-trivial smooth initial guess (satisfies homogeneous Dirichlet approximately).
        data_( local_subdomain_id, x, y, r, 0 ) = Kokkos::sin( 3 * cx ) * Kokkos::cos( 2 * cy );
        data_( local_subdomain_id, x, y, r, 1 ) = Kokkos::cos( 4 * cy ) * Kokkos::sin( 1 * cz );
        data_( local_subdomain_id, x, y, r, 2 ) = Kokkos::sin( 2 * cz ) * Kokkos::cos( 3 * cx );
    }
};

/// @brief Zero out boundary nodes.
struct ZeroBoundary
{
    Grid4DDataVec< double, 3 > data_;
    int                        num_shells_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        if ( r == 0 || r == num_shells_ - 1 )
        {
            for ( int d = 0; d < 3; ++d )
            {
                data_( local_subdomain_id, x, y, r, d ) = 0.0;
            }
        }
    }
};

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int level = 3;

    // --- Domain setup ---

    const ScalarType r_min = 0.5;
    const ScalarType r_max = 1.0;

    auto domain      = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );
    auto coords      = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    auto radii       = grid::shell::subdomain_shell_radii< ScalarType >( domain );
    auto mask_data   = grid::setup_node_ownership_mask_data( domain );
    auto bdry_mask   = grid::shell::setup_boundary_mask_data( domain );

    // --- Coefficient field ---

    VectorQ1Scalar< ScalarType > k( "k", domain, mask_data );
    Kokkos::parallel_for(
        "k_interpolation",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KInterpolator{ coords, radii, k.grid_data() } );

    // --- Operator with stored local matrices (needed for block diagonal extraction) ---

    using Viscous = fe::wedge::operators::shell::EpsilonDivDiv< ScalarType >;

    Viscous A( domain, coords, radii, bdry_mask, k.grid_data(), true, false );

    // Note: get_local_matrix() falls back to assemble_local_matrix() when storage mode is Off,
    // so no need to enable stored matrix mode for block diagonal extraction.

    // --- Vectors ---

    VectorQ1Vec< ScalarType > x_point( "x_point", domain, mask_data );
    VectorQ1Vec< ScalarType > x_block( "x_block", domain, mask_data );
    VectorQ1Vec< ScalarType > b( "b", domain, mask_data );
    VectorQ1Vec< ScalarType > tmp( "tmp", domain, mask_data );
    VectorQ1Vec< ScalarType > residual( "residual", domain, mask_data );

    // Zero RHS: we're doing pure error smoothing (Ax = 0, initial guess != 0).
    linalg::assign( b, 0.0 );

    // Set initial guess.
    Kokkos::parallel_for(
        "initial_guess",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        InitialGuessInterpolator{ coords, radii, x_point.grid_data() } );

    // Zero out boundary DOFs (homogeneous Dirichlet).
    const int num_shells = domain.domain_info().subdomain_num_nodes_radially();
    Kokkos::parallel_for(
        "zero_boundary",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        ZeroBoundary{ x_point.grid_data(), num_shells } );

    // Copy same initial guess for block Jacobi.
    linalg::assign( x_block, x_point );

    // --- Point Jacobi setup ---

    VectorQ1Vec< ScalarType > inv_diag( "inv_diag", domain, mask_data );
    {
        VectorQ1Vec< ScalarType > ones( "ones", domain, mask_data );
        linalg::assign( ones, 1.0 );
        A.set_diagonal( true );
        linalg::apply( A, ones, inv_diag );
        A.set_diagonal( false );
        linalg::invert_entries( inv_diag );
    }

    // Estimate spectral radius for relaxation parameter.
    VectorQ1Vec< ScalarType > tmp_pi_0( "tmp_pi_0", domain, mask_data );
    VectorQ1Vec< ScalarType > tmp_pi_1( "tmp_pi_1", domain, mask_data );
    DiagonallyScaledOperator< Viscous > inv_diag_A( A, inv_diag );
    const double max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
    const double omega  = 2.0 / ( 1.5 * max_ev );

    std::cout << "Spectral radius estimate: " << max_ev << std::endl;
    std::cout << "Relaxation parameter omega: " << omega << std::endl;

    VectorQ1Vec< ScalarType > tmp_point( "tmp_point", domain, mask_data );
    linalg::solvers::Jacobi< Viscous > point_jacobi( inv_diag, 1, tmp_point, omega );

    // --- Block Jacobi setup ---

    auto inv_block_diag = linalg::solvers::compute_inverse_block_diagonal< Viscous, 3 >( A, domain );

    VectorQ1Vec< ScalarType > tmp_block( "tmp_block", domain, mask_data );
    linalg::solvers::BlockJacobi< Viscous, 3 > block_jacobi( inv_block_diag, 1, tmp_block, omega );

    // --- Run smoothing iterations and compare convergence ---

    const int    num_iterations = 50;
    auto         table          = std::make_shared< util::Table >();

    // Compute initial residual norm (same for both since same initial guess).
    linalg::apply( A, x_point, residual );
    linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
    const double r0 = linalg::norm_2( residual );
    std::cout << "Initial residual norm: " << r0 << std::endl;

    double prev_r_point = r0;
    double prev_r_block = r0;

    for ( int iter = 1; iter <= num_iterations; ++iter )
    {
        // One point Jacobi step.
        linalg::solvers::solve( point_jacobi, A, x_point, b );

        // One block Jacobi step.
        linalg::solvers::solve( block_jacobi, A, x_block, b );

        // Compute residuals.
        linalg::apply( A, x_point, residual );
        linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
        const double r_point = linalg::norm_2( residual );

        linalg::apply( A, x_block, residual );
        linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
        const double r_block = linalg::norm_2( residual );

        const double rate_point = r_point / prev_r_point;
        const double rate_block = r_block / prev_r_block;

        table->add_row(
            { { "iteration", iter },
              { "r_point", r_point },
              { "r_block", r_block },
              { "rate_point", rate_point },
              { "rate_block", rate_block },
              { "rel_point", r_point / r0 },
              { "rel_block", r_block / r0 } } );

        prev_r_point = r_point;
        prev_r_block = r_block;
    }

    // --- Print results ---

    std::cout << "\n=== Smoother Convergence Comparison ===" << std::endl;
    std::cout << "Operator: EpsilonDivDiv (viscous block of Stokes)" << std::endl;
    std::cout << "Level: " << level << std::endl;
    std::cout << "Relaxation omega: " << omega << std::endl;
    std::cout << "Iterations: " << num_iterations << "\n" << std::endl;

    table->select_columns( { "iteration", "r_point", "r_block", "rate_point", "rate_block" } ).print_pretty();

    // --- Check that block Jacobi converges faster ---

    const double final_r_point = prev_r_point;
    const double final_r_block = prev_r_block;

    std::cout << "\nFinal residuals after " << num_iterations << " iterations:" << std::endl;
    std::cout << "  Point Jacobi: " << final_r_point << " (relative: " << final_r_point / r0 << ")" << std::endl;
    std::cout << "  Block Jacobi: " << final_r_block << " (relative: " << final_r_block / r0 << ")" << std::endl;

    if ( final_r_block < final_r_point )
    {
        std::cout << "\nBlock Jacobi converged faster by factor " << final_r_point / final_r_block << std::endl;
    }
    else
    {
        std::cout << "\nWARNING: Block Jacobi did NOT converge faster than point Jacobi." << std::endl;
        std::cout << "This may indicate a bug in the block diagonal extraction." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
