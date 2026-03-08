

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/identity.hpp"
#include "fe/wedge/operators/shell/prolongation_stokes.hpp"
#include "fe/wedge/operators/shell/restriction_stokes.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/vanka.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

struct SolutionVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_u_;
    bool                       only_boundary_;

    SolutionVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data_u,
        const bool                        only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        dense::Vec< double, 3 > u;
        u( 0 ) = -4 * Kokkos::cos( 4 * cz );
        u( 1 ) = 8 * Kokkos::cos( 8 * cx );
        u( 2 ) = -2 * Kokkos::cos( 2 * cy );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            for ( int d = 0; d < 3; d++ )
            {
                data_u_( local_subdomain_id, x, y, r, d ) = u( d );
            }
        }
    }
};

struct SolutionPressureInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_p_;
    bool                       only_boundary_;

    SolutionPressureInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data_p,
        const bool                        only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_p_( data_p )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        double p = Kokkos::sin( 4 * cx ) * Kokkos::sin( 8 * cy ) * Kokkos::sin( 2 * cz );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_p_( local_subdomain_id, x, y, r ) = p;
        }
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_u_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data_u )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        dense::Vec< double, 3 > u;
        u( 0 ) = 4 * Kokkos::sin( 8 * cy ) * Kokkos::sin( 2 * cz ) * Kokkos::cos( 4 * cx ) -
                 64 * Kokkos::cos( 4 * cz );
        u( 1 ) = 8 * Kokkos::sin( 4 * cx ) * Kokkos::sin( 2 * cz ) * Kokkos::cos( 8 * cy ) +
                 512 * Kokkos::cos( 8 * cx );
        u( 2 ) = 2 * Kokkos::sin( 4 * cx ) * Kokkos::sin( 8 * cy ) * Kokkos::cos( 2 * cz ) -
                 8 * Kokkos::cos( 2 * cy );

        for ( int d = 0; d < 3; d++ )
        {
            data_u_( local_subdomain_id, x, y, r, d ) = u( d );
        }
    }
};

std::pair< double, double > test( int min_level, int max_level, const std::shared_ptr< util::Table >& table )
{
    using ScalarType = double;

    // Set up domains for all levels.
    // Domain level k will be used as:
    //   - Velocity grid for MG level k-1
    //   - Pressure grid for MG level k
    // So MG level l has: vel @ domain[l+1], pre @ domain[l]

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = min_level; level <= max_level; level++ )
    {
        const int idx = level - min_level;

        domains.push_back( DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto num_domain_levels = domains.size();
    const auto velocity_level    = num_domain_levels - 1; // finest velocity domain index
    const auto pressure_level    = num_domain_levels - 2; // finest pressure domain index

    // Number of Stokes MG levels = max_level - min_level (= num_domain_levels - 1)
    // MG level l (0-indexed): vel @ domain[l+1], pre @ domain[l]
    // Finest MG level = num_mg_levels - 1, coarsest = 0
    const auto num_mg_levels = num_domain_levels - 1;

    // ===== Set up Stokes vectors for finest grid =====

    using Stokes      = fe::wedge::operators::shell::Stokes< ScalarType >;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { "u", "f", "solution", "error" };

    // FGMRES needs 2*m+4 vectors for restart m. With m=30, that's 64 vectors.
    constexpr int fgmres_restart    = 30;
    constexpr int num_fgmres_tmps   = 2 * fgmres_restart + 4;

    for ( int i = 0; i < num_fgmres_tmps; i++ )
    {
        stok_vec_names.push_back( "fgmres_tmp_" + std::to_string( i ) );
    }

    for ( const auto& name : stok_vec_names )
    {
        stok_vecs[name] = VectorQ1IsoQ2Q1< ScalarType >(
            name,
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    auto& u        = stok_vecs["u"];
    auto& f        = stok_vecs["f"];
    auto& solution = stok_vecs["solution"];
    auto& error    = stok_vecs["error"];

    // ===== Counting DoFs =====

    const auto num_dofs_velocity =
        3 * kernels::common::count_masked< long >( mask_data[velocity_level], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( mask_data[pressure_level], grid::NodeOwnershipFlag::OWNED );

    // ===== Set up Stokes operator on finest level =====

    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        true,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        false,
        false );

    Stokes K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        false,
        true );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // ===== Set up Stokes MG hierarchy =====

    using ProlongationS = fe::wedge::operators::shell::ProlongationStokes< ScalarType >;
    using RestrictionS  = fe::wedge::operators::shell::RestrictionStokes< ScalarType >;
    using VankaSmoother = linalg::solvers::Vanka< Stokes >;

    // Coarse Stokes operators: A_c[l] is the operator at MG level l, for l=0..num_mg_levels-2
    std::vector< Stokes > A_c;
    for ( int mg_level = 0; mg_level < num_mg_levels - 1; mg_level++ )
    {
        // MG level mg_level: vel @ domain[mg_level+1], pre @ domain[mg_level]
        A_c.emplace_back(
            domains[mg_level + 1],
            domains[mg_level],
            coords_shell[mg_level + 1],
            coords_radii[mg_level + 1],
            true,
            false );
    }

    // Prolongation and restriction between MG levels
    std::vector< ProlongationS > P;
    std::vector< RestrictionS >  R;
    for ( int mg_level = 0; mg_level < num_mg_levels - 1; mg_level++ )
    {
        // Prolongation from MG level mg_level to mg_level+1
        P.emplace_back( linalg::OperatorApplyMode::Add );

        // Restriction from MG level mg_level+1 to mg_level
        // Coarse vel domain = domain[mg_level+1], coarse pre domain = domain[mg_level]
        R.emplace_back( domains[mg_level + 1], domains[mg_level] );
    }

    // Temporary vectors at each MG level
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_mg;
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_mg_r;
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_mg_e;

    for ( int mg_level = 0; mg_level < num_mg_levels; mg_level++ )
    {
        // MG level mg_level: vel @ domain[mg_level+1], pre @ domain[mg_level]
        tmp_mg.emplace_back(
            "tmp_mg_" + std::to_string( mg_level ),
            domains[mg_level + 1],
            domains[mg_level],
            mask_data[mg_level + 1],
            mask_data[mg_level] );

        if ( mg_level < num_mg_levels - 1 )
        {
            tmp_mg_r.emplace_back(
                "tmp_mg_r_" + std::to_string( mg_level ),
                domains[mg_level + 1],
                domains[mg_level],
                mask_data[mg_level + 1],
                mask_data[mg_level] );

            tmp_mg_e.emplace_back(
                "tmp_mg_e_" + std::to_string( mg_level ),
                domains[mg_level + 1],
                domains[mg_level],
                mask_data[mg_level + 1],
                mask_data[mg_level] );
        }
    }

    // Vanka smoothers at each MG level
    // Each Vanka smoother needs a tmp vector for residual computation
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > vanka_tmps;
    std::vector< VankaSmoother >                  smoothers;

    for ( int mg_level = 0; mg_level < num_mg_levels; mg_level++ )
    {
        vanka_tmps.emplace_back(
            "vanka_tmp_" + std::to_string( mg_level ),
            domains[mg_level + 1],
            domains[mg_level],
            mask_data[mg_level + 1],
            mask_data[mg_level] );

        smoothers.emplace_back(
            domains[mg_level + 1],          // velocity domain at this MG level
            coords_shell[mg_level + 1],     // coords at velocity level
            coords_radii[mg_level + 1],     // radii at velocity level
            3,                              // iterations per smoothing step
            vanka_tmps.back(),
            0.5,                            // omega
            true );                         // treat_boundary
    }

    // Coarse grid solver: Vanka with many iterations on the coarsest level
    VectorQ1IsoQ2Q1< ScalarType > coarse_vanka_tmp(
        "coarse_vanka_tmp",
        domains[1],
        domains[0],
        mask_data[1],
        mask_data[0] );

    using CoarseGridSolver = VankaSmoother;
    CoarseGridSolver coarse_grid_solver(
        domains[1],
        coords_shell[1],
        coords_radii[1],
        50,                 // many iterations for coarse solve
        coarse_vanka_tmp,
        0.5,
        true );

    // Assemble full Stokes multigrid
    constexpr auto num_mg_cycles = 2;

    using StokesMG =
        linalg::solvers::Multigrid< Stokes, ProlongationS, RestrictionS, VankaSmoother, CoarseGridSolver >;
    StokesMG mg( P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, num_mg_cycles, 1e-8 );
    mg.collect_statistics( table );
    mg.set_tag( "stokes_vanka_mg" );

    // ===== Set up solution data =====

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            stok_vecs["solution"].block_1().grid_data(),
            false ) );

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domains[pressure_level] ),
        SolutionPressureInterpolator(
            coords_shell[pressure_level],
            coords_radii[pressure_level],
            stok_vecs["solution"].block_2().grid_data(),
            false ) );

    // Set up rhs data.

    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        RHSVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level], stok_vecs["fgmres_tmp_1"].block_1().grid_data() ) );

    linalg::apply( M, stok_vecs["fgmres_tmp_1"].block_1(), stok_vecs["f"].block_1() );

    // Set up boundary data.

    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            stok_vecs["fgmres_tmp_0"].block_1().grid_data(),
            true ) );

    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann,
        K_neumann_diag,
        stok_vecs["fgmres_tmp_0"],
        stok_vecs["fgmres_tmp_1"],
        stok_vecs["f"],
        boundary_mask_data[velocity_level],
        grid::shell::ShellBoundaryFlag::BOUNDARY );

    // ===== Set up FGMRES outer solver with Stokes-MG preconditioner =====

    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < num_fgmres_tmps; i++ )
    {
        tmp_fgmres.push_back( stok_vecs["fgmres_tmp_" + std::to_string( i )] );
    }

    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                    = fgmres_restart;
    fgmres_options.relative_residual_tolerance = 1e-8;
    fgmres_options.absolute_residual_tolerance = 1e-12;
    fgmres_options.max_iterations              = 200;

    auto solver_table = std::make_shared< util::Table >();

    linalg::solvers::FGMRES< Stokes, StokesMG > fgmres( tmp_fgmres, fgmres_options, solver_table, mg );

    linalg::assign( u, 0.0 );
    linalg::solvers::solve( fgmres, K, u, f );

    solver_table->query_rows_equals( "tag", "fgmres_solver" )
        .select_columns( { "absolute_residual", "relative_residual", "iteration" } )
        .print_pretty();

    // ===== Compute errors =====

    const double avg_pressure_solution =
        kernels::common::masked_sum(
            solution.block_2().grid_data(), solution.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;
    const double avg_pressure_approximation =
        kernels::common::masked_sum(
            u.block_2().grid_data(), u.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;

    linalg::lincomb( solution.block_2(), { 1.0 }, { solution.block_2() }, -avg_pressure_solution );
    linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

    linalg::apply( K, u, stok_vecs["fgmres_tmp_0"] );
    linalg::lincomb( stok_vecs["fgmres_tmp_1"], { 1.0, -1.0 }, { f, stok_vecs["fgmres_tmp_0"] } );
    const auto inf_residual_vel = linalg::norm_inf( stok_vecs["fgmres_tmp_1"].block_1() );
    const auto inf_residual_pre = linalg::norm_inf( stok_vecs["fgmres_tmp_1"].block_2() );

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error_velocity =
        std::sqrt( dot( error.block_1(), error.block_1() ) / static_cast< double >( num_dofs_velocity ) );
    const auto l2_error_pressure =
        std::sqrt( dot( error.block_2(), error.block_2() ) / static_cast< double >( num_dofs_pressure ) );

    table->add_row(
        { { "level", max_level },
          { "dofs_vel", num_dofs_velocity },
          { "l2_error_vel", l2_error_velocity },
          { "dofs_pre", num_dofs_pressure },
          { "l2_error_pre", l2_error_pressure },
          { "inf_res_vel", inf_residual_vel },
          { "inf_res_pre", inf_residual_pre } } );

    return { l2_error_velocity, l2_error_pressure };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    double prev_l2_error_vel = 1.0;
    double prev_l2_error_pre = 1.0;

    const int min_level = 1;
    for ( int level = min_level + 2; level < 5; ++level )
    {
        std::cout << "level = " << level << std::endl;
        Kokkos::Timer timer;
        timer.reset();
        const auto [l2_error_vel, l2_error_pre] = test( min_level, level, table );
        const auto time_total                   = timer.seconds();
        table->add_row( { { "level", level }, { "time_total", time_total } } );

        if ( level > 2 )
        {
            const double order_vel = prev_l2_error_vel / l2_error_vel;
            const double order_pre = prev_l2_error_pre / l2_error_pre;

            std::cout << "order_vel = " << order_vel << std::endl;
            std::cout << "order_pre = " << order_pre << std::endl;

            table->add_row( { { "level", level }, { "order_vel", order_vel }, { "order_pre", order_pre } } );
        }
        prev_l2_error_vel = l2_error_vel;
        prev_l2_error_pre = l2_error_pre;
    }

    table->query_rows_not_none( "order_vel" ).select_columns( { "level", "order_vel", "order_pre" } ).print_pretty();
    table->query_rows_not_none( "dofs_vel" )
        .select_columns( { "level", "dofs_vel", "l2_error_vel", "l2_error_pre" } )
        .print_pretty();

    return 0;
}
