
/// @brief Compares convergence of block Jacobi vs point Jacobi as stand-alone smoothers
///        on the EpsilonDivDiv (viscous) operator.
///
/// Runs the comparison for three viscosity profiles:
///   1. Constant k=1 (baseline)
///   2. Lin et al. 2022 radial viscosity profile (contrast ~1000)
///   3. Stotz et al. 2017 radial viscosity profile (contrast ~12000)

#include "fe/wedge/operators/shell/epsilon_divdiv.hpp"
#include "geophysics/viscosity/viscosity_interpolation.hpp"
#include "linalg/solvers/block_jacobi.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "shell/radial_profiles.hpp"
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

/// @brief Run smoothing comparison for a given coefficient field.
/// @return (final_r_point, final_r_block) — final residual norms.
std::pair< double, double > run_comparison(
    const std::string&                                                              label,
    DistributedDomain&                                                              domain,
    const Grid3DDataVec< double, 3 >&                                               coords,
    const Grid2DDataScalar< double >&                                               radii,
    const Grid4DDataScalar< grid::NodeOwnershipFlag >&                              mask_data,
    const Grid4DDataScalar< double >&                                               k_data,
    const int                                                                       num_iterations )
{
    using Viscous = fe::wedge::operators::shell::EpsilonDivDiv< ScalarType >;

    Viscous A( domain, coords, radii, k_data, true, false );

    // --- Vectors ---

    VectorQ1Vec< ScalarType > x_point( "x_point", domain, mask_data );
    VectorQ1Vec< ScalarType > x_block( "x_block", domain, mask_data );
    VectorQ1Vec< ScalarType > b( "b", domain, mask_data );
    VectorQ1Vec< ScalarType > residual( "residual", domain, mask_data );

    linalg::assign( b, 0.0 );

    // Set initial guess.
    Kokkos::parallel_for(
        "initial_guess",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        InitialGuessInterpolator{ coords, radii, x_point.grid_data() } );

    // Zero boundary DOFs.
    const int num_shells = domain.domain_info().subdomain_num_nodes_radially();
    Kokkos::parallel_for(
        "zero_boundary",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        ZeroBoundary{ x_point.grid_data(), num_shells } );

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

    VectorQ1Vec< ScalarType > tmp_pi_0( "tmp_pi_0", domain, mask_data );
    VectorQ1Vec< ScalarType > tmp_pi_1( "tmp_pi_1", domain, mask_data );
    DiagonallyScaledOperator< Viscous > inv_diag_A( A, inv_diag );
    const double max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
    const double omega  = 2.0 / ( 1.5 * max_ev );

    VectorQ1Vec< ScalarType > tmp_point( "tmp_point", domain, mask_data );
    linalg::solvers::Jacobi< Viscous > point_jacobi( inv_diag, 1, tmp_point, omega );

    // --- Block Jacobi setup ---

    auto inv_block_diag = linalg::solvers::compute_inverse_block_diagonal< Viscous, 3 >( A, domain );

    VectorQ1Vec< ScalarType > tmp_block( "tmp_block", domain, mask_data );
    linalg::solvers::BlockJacobi< Viscous, 3 > block_jacobi( inv_block_diag, 1, tmp_block, omega );

    // --- Run smoothing iterations ---

    auto table = std::make_shared< util::Table >();

    linalg::apply( A, x_point, residual );
    linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
    const double r0 = linalg::norm_2( residual );

    double prev_r_point = r0;
    double prev_r_block = r0;

    for ( int iter = 1; iter <= num_iterations; ++iter )
    {
        linalg::solvers::solve( point_jacobi, A, x_point, b );
        linalg::solvers::solve( block_jacobi, A, x_block, b );

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
              { "rate_block", rate_block } } );

        prev_r_point = r_point;
        prev_r_block = r_block;
    }

    // --- Print results ---

    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << "omega: " << omega << ", spectral radius est.: " << max_ev
              << ", initial residual: " << r0 << std::endl;

    table->select_columns( { "iteration", "r_point", "r_block", "rate_point", "rate_block" } ).print_pretty();

    const double ratio = prev_r_block / prev_r_point;
    std::cout << "Final:  point=" << prev_r_point << "  block=" << prev_r_block
              << "  ratio(block/point)=" << ratio << std::endl;

    return { prev_r_point, prev_r_block };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int level = 3;

    // --- Domain setup ---

    const ScalarType r_min = 0.5;
    const ScalarType r_max = 1.0;

    auto domain    = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );
    auto coords    = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    auto radii     = grid::shell::subdomain_shell_radii< ScalarType >( domain );
    auto mask_data = grid::setup_node_ownership_mask_data( domain );

    const int num_iterations = 50;

    // --- 1. Constant viscosity k=1 (baseline) ---
    {
        VectorQ1Scalar< ScalarType > k( "k_const", domain, mask_data );
        linalg::assign( k, 1.0 );

        auto [rp, rb] = run_comparison( "Constant k=1", domain, coords, radii, mask_data, k.grid_data(), num_iterations );
    }

    // --- 2. Lin et al. 2022 radial viscosity profile ---
    {
        auto profile_2d = shell::interpolate_radial_profile_into_subdomains_from_csv< ScalarType >(
            TERRA_DATA_DIR "/radialprofiles/ViscosityProfile_Lin_et_al_2022.csv",
            "radius_normalized_0p5_1p0",
            "viscosity_scaled_by_min",
            radii );

        VectorQ1Scalar< ScalarType > k( "k_lin", domain, mask_data );
        geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType > interp( profile_2d );
        interp.interpolate( k.grid_data() );

        auto [rp, rb] =
            run_comparison( "Lin et al. 2022 (contrast ~1000)", domain, coords, radii, mask_data, k.grid_data(), num_iterations );
    }

    // --- 3. Stotz et al. 2017 radial viscosity profile ---
    {
        auto profile_2d = shell::interpolate_radial_profile_into_subdomains_from_csv< ScalarType >(
            TERRA_DATA_DIR "/radialprofiles/ViscosityProfile_Stotz_et_al_2017.csv",
            "radius_normalized_0p5_1p0",
            "viscosity_scaled_by_min",
            radii );

        VectorQ1Scalar< ScalarType > k( "k_stotz", domain, mask_data );
        geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType > interp( profile_2d );
        interp.interpolate( k.grid_data() );

        auto [rp, rb] =
            run_comparison( "Stotz et al. 2017 (contrast ~12000)", domain, coords, radii, mask_data, k.grid_data(), num_iterations );
    }

    return EXIT_SUCCESS;
}
