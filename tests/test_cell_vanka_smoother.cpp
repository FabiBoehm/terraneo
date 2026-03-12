
/// @brief Compares convergence of cell Vanka vs block Jacobi vs point Jacobi as smoothers
///        on the EpsilonDivDiv (viscous) operator.
///
/// Runs the comparison for four viscosity profiles:
///   1. Constant k=1 (baseline)
///   2. Lin et al. 2022 radial viscosity profile (contrast ~1000)
///   3. Stotz et al. 2017 radial viscosity profile (contrast ~12000)
///   4. Laterally varying viscosity: k(x,y,z) = 1 + 1000 * sin^2(3x) * cos^2(2y)
///
/// For each profile, runs:
///   - Naked smoother comparison (point Jacobi, block Jacobi, cell Vanka)
///   - Multigrid V-cycle comparison (block Jacobi vs cell Vanka as smoother)

#include "fe/wedge/operators/shell/epsilon_divdiv.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "geophysics/viscosity/viscosity_interpolation.hpp"
#include "linalg/solvers/block_jacobi.hpp"
#include "linalg/solvers/cell_vanka.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
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

/// @brief Laterally varying viscosity: k(x,y,z) = 1 + contrast * sin^2(3x) * cos^2(2y)
struct LateralViscosityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    double                     contrast_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords =
            grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );

        const double s = Kokkos::sin( 3 * cx );
        const double c = Kokkos::cos( 2 * cy );

        data_( local_subdomain_id, x, y, r ) = 1.0 + contrast_ * s * s * c * c;
    }
};

// ============================================================================
// Stand-alone smoother comparison
// ============================================================================

void run_smoother_comparison(
    const std::string&                                 label,
    DistributedDomain&                                 domain,
    const Grid3DDataVec< double, 3 >&                  coords,
    const Grid2DDataScalar< double >&                  radii,
    const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data,
    const Grid4DDataScalar< double >&                  k_data,
    const int                                          num_iterations )
{
    using Viscous = fe::wedge::operators::shell::EpsilonDivDiv< ScalarType >;

    Viscous A( domain, coords, radii, k_data, true, false );

    // --- Vectors ---

    VectorQ1Vec< ScalarType > x_point( "x_point", domain, mask_data );
    VectorQ1Vec< ScalarType > x_block( "x_block", domain, mask_data );
    VectorQ1Vec< ScalarType > x_vanka( "x_vanka", domain, mask_data );
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
    linalg::assign( x_vanka, x_point );

    // --- Point Jacobi setup ---

    Kokkos::Timer timer_setup_point;

    VectorQ1Vec< ScalarType > inv_diag( "inv_diag", domain, mask_data );
    {
        VectorQ1Vec< ScalarType > ones( "ones", domain, mask_data );
        linalg::assign( ones, 1.0 );
        A.set_diagonal( true );
        linalg::apply( A, ones, inv_diag );
        A.set_diagonal( false );
        linalg::invert_entries( inv_diag );
    }

    VectorQ1Vec< ScalarType >          tmp_pi_0( "tmp_pi_0", domain, mask_data );
    VectorQ1Vec< ScalarType >          tmp_pi_1( "tmp_pi_1", domain, mask_data );
    DiagonallyScaledOperator< Viscous > inv_diag_A( A, inv_diag );
    const double max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
    const double omega  = 2.0 / ( 1.1 * max_ev );

    VectorQ1Vec< ScalarType >          tmp_point( "tmp_point", domain, mask_data );
    linalg::solvers::Jacobi< Viscous > point_jacobi( inv_diag, 1, tmp_point, omega );

    Kokkos::fence();
    const double time_setup_point = timer_setup_point.seconds();

    // --- Block Jacobi setup ---

    Kokkos::Timer timer_setup_block;

    auto inv_block_diag = linalg::solvers::compute_inverse_block_diagonal< Viscous, 3 >( A, domain );

    VectorQ1Vec< ScalarType >                  tmp_block( "tmp_block", domain, mask_data );
    linalg::solvers::BlockJacobi< Viscous, 3 > block_jacobi( inv_block_diag, 1, tmp_block, omega );

    Kokkos::fence();
    const double time_setup_block = timer_setup_block.seconds();

    // --- Cell Vanka setup ---

    Kokkos::Timer timer_setup_vanka;

    auto inv_cell_matrices = linalg::solvers::compute_cell_vanka_matrices< Viscous, 3 >( A, domain );

    VectorQ1Vec< ScalarType >                tmp_vanka( "tmp_vanka", domain, mask_data );
    VectorQ1Vec< ScalarType >                corr_vanka( "corr_vanka", domain, mask_data );
    linalg::solvers::CellVanka< Viscous, 3 > cell_vanka( inv_cell_matrices, 1, tmp_vanka, corr_vanka );

    Kokkos::fence();
    const double time_setup_vanka = timer_setup_vanka.seconds();

    // --- Run smoothing iterations ---

    auto table = std::make_shared< util::Table >();

    linalg::apply( A, x_point, residual );
    linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
    const double r0 = linalg::norm_2( residual );

    double prev_r_point = r0;
    double prev_r_block = r0;
    double prev_r_vanka = r0;

    double time_apply_point = 0.0;
    double time_apply_block = 0.0;
    double time_apply_vanka = 0.0;

    for ( int iter = 1; iter <= num_iterations; ++iter )
    {
        Kokkos::Timer t_point;
        linalg::solvers::solve( point_jacobi, A, x_point, b );
        Kokkos::fence();
        time_apply_point += t_point.seconds();

        Kokkos::Timer t_block;
        linalg::solvers::solve( block_jacobi, A, x_block, b );
        Kokkos::fence();
        time_apply_block += t_block.seconds();

        Kokkos::Timer t_vanka;
        linalg::solvers::solve( cell_vanka, A, x_vanka, b );
        Kokkos::fence();
        time_apply_vanka += t_vanka.seconds();

        linalg::apply( A, x_point, residual );
        linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
        const double r_point = linalg::norm_2( residual );

        linalg::apply( A, x_block, residual );
        linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
        const double r_block = linalg::norm_2( residual );

        linalg::apply( A, x_vanka, residual );
        linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
        const double r_vanka = linalg::norm_2( residual );

        const double rate_point = r_point / prev_r_point;
        const double rate_block = r_block / prev_r_block;
        const double rate_vanka = r_vanka / prev_r_vanka;

        table->add_row(
            { { "iteration", iter },
              { "r_point", r_point },
              { "r_block", r_block },
              { "r_vanka", r_vanka },
              { "rate_point", rate_point },
              { "rate_block", rate_block },
              { "rate_vanka", rate_vanka } } );

        prev_r_point = r_point;
        prev_r_block = r_block;
        prev_r_vanka = r_vanka;
    }

    // --- Print results ---

    std::cout << "\n=== Smoother: " << label << " ===" << std::endl;
    std::cout << "omega: " << omega << ", spectral radius est.: " << max_ev
              << ", initial residual: " << r0 << std::endl;

    table->select_columns(
              { "iteration", "r_point", "r_block", "r_vanka", "rate_point", "rate_block", "rate_vanka" } )
        .print_pretty();

    std::cout << "Final:  point=" << prev_r_point << "  block=" << prev_r_block
              << "  vanka=" << prev_r_vanka
              << "  ratio(block/point)=" << prev_r_block / prev_r_point
              << "  ratio(vanka/point)=" << prev_r_vanka / prev_r_point << std::endl;

    std::cout << "\nTiming (setup):  point=" << time_setup_point << "s  block=" << time_setup_block
              << "s  vanka=" << time_setup_vanka << "s" << std::endl;
    std::cout << "Timing (apply, " << num_iterations << " iters):  point=" << time_apply_point
              << "s  block=" << time_apply_block << "s  vanka=" << time_apply_vanka << "s" << std::endl;
    std::cout << "Timing (per iter):  point=" << time_apply_point / num_iterations
              << "s  block=" << time_apply_block / num_iterations
              << "s  vanka=" << time_apply_vanka / num_iterations << "s" << std::endl;
}

// ============================================================================
// Multigrid V-cycle comparison
// ============================================================================

template < typename SmootherT >
void run_multigrid_vcycles(
    const std::string& label,
    std::vector< fe::wedge::operators::shell::EpsilonDivDiv< ScalarType > >& A_levels,
    std::vector< DistributedDomain >&                                        domains,
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >&              mask_data_levels,
    std::vector< SmootherT >&                                                smoothers,
    VectorQ1Vec< ScalarType >&                                               x,
    VectorQ1Vec< ScalarType >&                                               b,
    const int                                                                num_cycles )
{
    using Viscous      = fe::wedge::operators::shell::EpsilonDivDiv< ScalarType >;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    const auto num_levels = domains.size();

    // Transfer operators
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;
    for ( size_t level = 0; level < num_levels - 1; ++level )
    {
        P.emplace_back( linalg::OperatorApplyMode::Add );
        R.emplace_back( domains[level] );
    }

    // Temporary vectors
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg, tmp_mg_r, tmp_mg_e;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data_levels[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data_levels[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data_levels[level] );
        }
    }

    // Coarse grid solver (PCG on level 0)
    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;
    auto                                     cg_table = std::make_shared< util::Table >();
    std::vector< VectorQ1Vec< ScalarType > > cg_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        cg_tmps.emplace_back( "tmp_cg", domains[0], mask_data_levels[0] );
    }
    CoarseGridSolver coarse_solver(
        linalg::solvers::IterativeSolverParameters{ 200, 1e-10, 1e-16 }, cg_table, cg_tmps );

    // Coarse operators (levels 0..num_levels-2)
    std::vector< Viscous > A_coarse( A_levels.begin(), A_levels.begin() + static_cast< long >( num_levels - 1 ) );

    using MG = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, SmootherT, CoarseGridSolver >;

    auto mg_table = std::make_shared< util::Table >();
    MG   mg( P, R, A_coarse, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_solver, num_cycles, 1e-6 );
    mg.collect_statistics( mg_table );
    mg.set_tag( label );

    Kokkos::Timer timer_mg;
    linalg::solvers::solve( mg, A_levels.back(), x, b );
    Kokkos::fence();
    const double time_mg = timer_mg.seconds();

    mg_table->query_rows_equals( "tag", label )
        .select_columns( { "cycle", "absolute_residual", "relative_residual", "residual_convergence_rate" } )
        .print_pretty();

    std::cout << "MG solve time: " << time_mg << "s" << std::endl;
}

void run_multigrid_comparison(
    const std::string&                                                          label,
    const int                                                                   min_level,
    const int                                                                   max_level,
    const std::function< void( DistributedDomain&,
                               const Grid3DDataVec< double, 3 >&,
                               const Grid2DDataScalar< double >&,
                               Grid4DDataScalar< double >& ) >&                k_setup,
    const int                                                                   num_cycles )
{
    using Viscous = fe::wedge::operators::shell::EpsilonDivDiv< ScalarType >;

    const auto num_levels = static_cast< size_t >( max_level - min_level + 1 );

    // Build domain hierarchy.
    std::vector< DistributedDomain >                             domains;
    std::vector< Grid3DDataVec< double, 3 > >                    coords_shell;
    std::vector< Grid2DDataScalar< double > >                    coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >   mask_data;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const auto idx = static_cast< size_t >( level - min_level );
        domains.push_back( DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
    }

    // Build operators.
    std::vector< Viscous >                       A_levels;
    std::vector< VectorQ1Scalar< ScalarType > >  k_vecs;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        k_vecs.emplace_back( "k_" + std::to_string( level ), domains[level], mask_data[level] );
        k_setup( domains[level], coords_shell[level], coords_radii[level], k_vecs.back().grid_data() );
        A_levels.emplace_back( domains[level], coords_shell[level], coords_radii[level], k_vecs[level].grid_data(), true, false );
    }

    // Build initial guess and rhs on finest level.
    VectorQ1Vec< ScalarType > x_point( "x_mg_point", domains.back(), mask_data.back() );
    VectorQ1Vec< ScalarType > x_block( "x_mg_block", domains.back(), mask_data.back() );
    VectorQ1Vec< ScalarType > x_vanka( "x_mg_vanka", domains.back(), mask_data.back() );
    VectorQ1Vec< ScalarType > b( "b_mg", domains.back(), mask_data.back() );
    linalg::assign( b, 0.0 );

    Kokkos::parallel_for(
        "initial_guess",
        grid::shell::local_domain_md_range_policy_nodes( domains.back() ),
        InitialGuessInterpolator{ coords_shell.back(), coords_radii.back(), x_point.grid_data() } );
    const int num_shells = domains.back().domain_info().subdomain_num_nodes_radially();
    Kokkos::parallel_for(
        "zero_boundary",
        grid::shell::local_domain_md_range_policy_nodes( domains.back() ),
        ZeroBoundary{ x_point.grid_data(), num_shells } );
    linalg::assign( x_block, x_point );
    linalg::assign( x_vanka, x_point );

    // --- Point Jacobi smoothers ---
    using PointSmoother = linalg::solvers::Jacobi< Viscous >;
    std::vector< PointSmoother > point_smoothers;

    // --- Block Jacobi smoothers ---
    using BlockSmoother = linalg::solvers::BlockJacobi< Viscous, 3 >;
    std::vector< BlockSmoother > block_smoothers;

    // --- Cell Vanka smoothers ---
    using VankaSmoother = linalg::solvers::CellVanka< Viscous, 3 >;
    std::vector< VankaSmoother > vanka_smoothers;

    std::vector< VectorQ1Vec< ScalarType > >                point_tmps;
    std::vector< VectorQ1Vec< ScalarType > >                block_tmps;
    std::vector< VectorQ1Vec< ScalarType > >                vanka_tmps;
    std::vector< VectorQ1Vec< ScalarType > >                vanka_corrs;
    std::vector< VectorQ1Vec< ScalarType > >                inv_diags;
    using InvBlockDiagType = typename BlockSmoother::InverseBlockDiagonalType;
    std::vector< InvBlockDiagType > inv_block_diags;
    using InvCellType = typename VankaSmoother::InverseCellMatricesType;
    std::vector< InvCellType > inv_cell_mats;

    double time_setup_point_total = 0.0;
    double time_setup_block_total = 0.0;
    double time_setup_vanka_total = 0.0;

    for ( size_t level = 0; level < num_levels; ++level )
    {
        // Compute point inverse diagonal for omega estimation.
        Kokkos::Timer timer_point;
        inv_diags.emplace_back( "inv_diag_" + std::to_string( level ), domains[level], mask_data[level] );
        {
            VectorQ1Vec< ScalarType > ones( "ones", domains[level], mask_data[level] );
            linalg::assign( ones, 1.0 );
            A_levels[level].set_diagonal( true );
            linalg::apply( A_levels[level], ones, inv_diags.back() );
            A_levels[level].set_diagonal( false );
            linalg::invert_entries( inv_diags.back() );
        }

        VectorQ1Vec< ScalarType >          tmp0( "tmp0", domains[level], mask_data[level] );
        VectorQ1Vec< ScalarType >          tmp1( "tmp1", domains[level], mask_data[level] );
        DiagonallyScaledOperator< Viscous > dA( A_levels[level], inv_diags.back() );
        const double max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( dA, tmp0, tmp1, 100 );
        const double omega  = 2.0 / ( 1.1 * max_ev );

        std::cout << "  level " << level << ": spectral_radius=" << max_ev << ", omega=" << omega << std::endl;

        constexpr int smoother_steps = 3;

        // Point Jacobi smoother.
        point_tmps.emplace_back( "pt_tmp_" + std::to_string( level ), domains[level], mask_data[level] );
        point_smoothers.emplace_back( inv_diags.back(), smoother_steps, point_tmps.back(), omega );
        Kokkos::fence();
        time_setup_point_total += timer_point.seconds();

        // Block Jacobi smoother.
        Kokkos::Timer timer_block;
        inv_block_diags.push_back(
            linalg::solvers::compute_inverse_block_diagonal< Viscous, 3 >( A_levels[level], domains[level] ) );
        block_tmps.emplace_back( "bk_tmp_" + std::to_string( level ), domains[level], mask_data[level] );
        block_smoothers.emplace_back( inv_block_diags.back(), smoother_steps, block_tmps.back(), omega );
        Kokkos::fence();
        time_setup_block_total += timer_block.seconds();

        // Cell Vanka smoother.
        Kokkos::Timer timer_vanka;
        inv_cell_mats.push_back(
            linalg::solvers::compute_cell_vanka_matrices< Viscous, 3 >( A_levels[level], domains[level] ) );
        vanka_tmps.emplace_back( "vk_tmp_" + std::to_string( level ), domains[level], mask_data[level] );
        vanka_corrs.emplace_back( "vk_corr_" + std::to_string( level ), domains[level], mask_data[level] );
        vanka_smoothers.emplace_back( inv_cell_mats.back(), smoother_steps, vanka_tmps.back(), vanka_corrs.back() );
        Kokkos::fence();
        time_setup_vanka_total += timer_vanka.seconds();
    }

    std::cout << "\nSmoother setup time (all levels):  point=" << time_setup_point_total
              << "s  block=" << time_setup_block_total
              << "s  vanka=" << time_setup_vanka_total << "s" << std::endl;

    std::cout << "\n=== MG Point Jacobi: " << label << " ===" << std::endl;
    run_multigrid_vcycles< PointSmoother >(
        "mg_point_" + label, A_levels, domains, mask_data, point_smoothers, x_point, b, num_cycles );

    VectorQ1Vec< ScalarType > residual( "residual", domains.back(), mask_data.back() );
    linalg::apply( A_levels.back(), x_point, residual );
    linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
    const double final_r_point = linalg::norm_2( residual );

    std::cout << "\n=== MG Block Jacobi: " << label << " ===" << std::endl;
    run_multigrid_vcycles< BlockSmoother >(
        "mg_block_" + label, A_levels, domains, mask_data, block_smoothers, x_block, b, num_cycles );

    linalg::apply( A_levels.back(), x_block, residual );
    linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
    const double final_r_block = linalg::norm_2( residual );

    std::cout << "\n=== MG Cell Vanka: " << label << " ===" << std::endl;
    run_multigrid_vcycles< VankaSmoother >(
        "mg_vanka_" + label, A_levels, domains, mask_data, vanka_smoothers, x_vanka, b, num_cycles );

    linalg::apply( A_levels.back(), x_vanka, residual );
    linalg::lincomb( residual, { 1.0, -1.0 }, { b, residual } );
    const double final_r_vanka = linalg::norm_2( residual );

    std::cout << "MG Final:  point=" << final_r_point << "  block=" << final_r_block << "  vanka=" << final_r_vanka
              << "\n  ratio(block/point)=" << final_r_block / final_r_point
              << "  ratio(vanka/point)=" << final_r_vanka / final_r_point << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int level     = 3;
    const int min_level = 1;
    const int max_level = 3;

    // --- Domain setup for smoother tests ---

    const ScalarType r_min = 0.5;
    const ScalarType r_max = 1.0;

    auto domain    = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );
    auto coords    = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    auto radii     = grid::shell::subdomain_shell_radii< ScalarType >( domain );
    auto mask_data = grid::setup_node_ownership_mask_data( domain );

    const int num_smoother_iterations = 50;
    const int num_mg_cycles           = 100;

    // ================================================================
    // Part 1: Naked smoother comparison
    // ================================================================

    std::cout << "\n################################################################" << std::endl;
    std::cout << "# Part 1: Naked smoother comparison (point vs block vs vanka)" << std::endl;
    std::cout << "################################################################" << std::endl;

    // --- 1. Constant viscosity k=1 (baseline) ---
    {
        VectorQ1Scalar< ScalarType > k( "k_const", domain, mask_data );
        linalg::assign( k, 1.0 );
        run_smoother_comparison( "Constant k=1", domain, coords, radii, mask_data, k.grid_data(), num_smoother_iterations );
    }

    // --- 2. Lin et al. 2022 radial viscosity profile ---
    {
        auto profile_2d = shell::interpolate_radial_profile_into_subdomains_from_csv< ScalarType >(
            TERRA_DATA_DIR "/radialprofiles/ViscosityProfile_Lin_et_al_2022.csv",
            "radius_normalized_0p5_1p0",
            "viscosity_scaled_by_min",
            radii );

        VectorQ1Scalar< ScalarType >                                          k( "k_lin", domain, mask_data );
        geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType > interp( profile_2d );
        interp.interpolate( k.grid_data() );

        run_smoother_comparison(
            "Lin et al. 2022 (contrast ~1000)", domain, coords, radii, mask_data, k.grid_data(), num_smoother_iterations );
    }

    // --- 3. Stotz et al. 2017 radial viscosity profile ---
    {
        auto profile_2d = shell::interpolate_radial_profile_into_subdomains_from_csv< ScalarType >(
            TERRA_DATA_DIR "/radialprofiles/ViscosityProfile_Stotz_et_al_2017.csv",
            "radius_normalized_0p5_1p0",
            "viscosity_scaled_by_min",
            radii );

        VectorQ1Scalar< ScalarType >                                          k( "k_stotz", domain, mask_data );
        geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType > interp( profile_2d );
        interp.interpolate( k.grid_data() );

        run_smoother_comparison(
            "Stotz et al. 2017 (contrast ~12000)", domain, coords, radii, mask_data, k.grid_data(), num_smoother_iterations );
    }

    // --- 4. Laterally varying viscosity ---
    {
        VectorQ1Scalar< ScalarType > k( "k_lateral", domain, mask_data );
        Kokkos::parallel_for(
            "k_lateral",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            LateralViscosityInterpolator{ coords, radii, k.grid_data(), 1000.0 } );

        run_smoother_comparison(
            "Lateral k = 1 + 1000*sin^2(3x)*cos^2(2y)",
            domain, coords, radii, mask_data, k.grid_data(), num_smoother_iterations );
    }

    // ================================================================
    // Part 2: Multigrid V-cycle comparison
    // ================================================================

    std::cout << "\n################################################################" << std::endl;
    std::cout << "# Part 2: Multigrid V-cycle comparison (point vs block vs vanka, tol=1e-6)" << std::endl;
    std::cout << "################################################################" << std::endl;

    // --- 1. Constant k=1 ---
    {
        auto k_setup = []( DistributedDomain& /*dom*/,
                           const Grid3DDataVec< double, 3 >& /*cs*/,
                           const Grid2DDataScalar< double >& /*cr*/,
                           Grid4DDataScalar< double >& k_data ) {
            Kokkos::deep_copy( k_data, 1.0 );
        };
        run_multigrid_comparison( "Constant k=1", min_level, max_level, k_setup, num_mg_cycles );
    }

    // --- 2. Lin et al. 2022 ---
    {
        auto k_setup = []( DistributedDomain& dom,
                           const Grid3DDataVec< double, 3 >& /*cs*/,
                           const Grid2DDataScalar< double >& cr,
                           Grid4DDataScalar< double >& k_data ) {
            auto profile_2d = shell::interpolate_radial_profile_into_subdomains_from_csv< ScalarType >(
                TERRA_DATA_DIR "/radialprofiles/ViscosityProfile_Lin_et_al_2022.csv",
                "radius_normalized_0p5_1p0",
                "viscosity_scaled_by_min",
                cr );
            geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType > interp( profile_2d );
            interp.interpolate( k_data );
        };
        run_multigrid_comparison( "Lin et al. 2022 (contrast ~1000)", min_level, max_level, k_setup, num_mg_cycles );
    }

    // --- 3. Stotz et al. 2017 ---
    {
        auto k_setup = []( DistributedDomain& dom,
                           const Grid3DDataVec< double, 3 >& /*cs*/,
                           const Grid2DDataScalar< double >& cr,
                           Grid4DDataScalar< double >& k_data ) {
            auto profile_2d = shell::interpolate_radial_profile_into_subdomains_from_csv< ScalarType >(
                TERRA_DATA_DIR "/radialprofiles/ViscosityProfile_Stotz_et_al_2017.csv",
                "radius_normalized_0p5_1p0",
                "viscosity_scaled_by_min",
                cr );
            geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType > interp( profile_2d );
            interp.interpolate( k_data );
        };
        run_multigrid_comparison( "Stotz et al. 2017 (contrast ~12000)", min_level, max_level, k_setup, num_mg_cycles );
    }

    // --- 4. Laterally varying viscosity ---
    {
        auto k_setup = []( DistributedDomain& dom,
                           const Grid3DDataVec< double, 3 >& cs,
                           const Grid2DDataScalar< double >& cr,
                           Grid4DDataScalar< double >& k_data ) {
            Kokkos::parallel_for(
                "k_lateral",
                grid::shell::local_domain_md_range_policy_nodes( dom ),
                LateralViscosityInterpolator{ cs, cr, k_data, 1000.0 } );
            Kokkos::fence();
        };
        run_multigrid_comparison(
            "Lateral k = 1 + 1000*sin^2(3x)*cos^2(2y)", min_level, max_level, k_setup, num_mg_cycles );
    }

    return EXIT_SUCCESS;
}
