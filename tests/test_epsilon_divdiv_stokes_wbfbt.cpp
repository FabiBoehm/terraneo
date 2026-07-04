// Variant of test_epsilon_divdiv_stokes.cpp that verifies the w-BFBT Schur
// preconditioner (from apps/mantlecirculation/src/wbfbt_schur_preconditioner.hpp)
// drives outer FGMRES to the same analytical solution as the existing
// lumped-pressure-mass Schur PC.  Same manufactured solution:
//
//   k(x)      = 2 + sin(z)
//   u_ex(x)   = (-4 cos(4 z), 8 cos(8 x), -2 cos(2 y))
//   p_ex(x)   = sin(4 x) sin(8 y) sin(2 z)
//   f         = K(k) · [u_ex; p_ex]                     (analytical RHS)
//   BCs       = Dirichlet on CMB + SURFACE
//
// Runs the solver twice — once with the mass-Schur PC and once with WBFBT —
// going through the same outer FGMRES via PolymorphicSchurPreconditioner, and
// asserts both L2 errors stay below a generous threshold.  This pins down
// correctness of the WBFBT path (not just its iteration count) at variable
// viscosity.

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp"

#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"

#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"

#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"

#include "../apps/mantlecirculation/src/polymorphic_schur_preconditioner.hpp"
#include "../apps/mantlecirculation/src/wbfbt_pressure_poisson_explicit_kw.hpp"
#include "../apps/mantlecirculation/src/wbfbt_schur_preconditioner.hpp"
#include "../apps/mantlecirculation/src/wbfbt_weighted_lumped_velocity_mass.hpp"

#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::local_domain_md_range_policy_nodes;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::DiagonallyScaledOperator;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::solvers::DiagonalSolver;
using linalg::solvers::power_iteration;
using terra::grid::shell::BoundaryConditions;

namespace mc = terra::mantlecirculation;

// ---------------------------------------------------------------------------
// Manufactured solution (lifted verbatim from test_epsilon_divdiv_stokes.cpp)
// ---------------------------------------------------------------------------

struct SolutionVelocityInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_u_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const bool on_boundary = util::has_flag( mask_( sd, x, y, r ), BOUNDARY );
        if ( only_boundary_ && !on_boundary )
            return;
        data_u_( sd, x, y, r, 0 ) = -4.0 * Kokkos::cos( 4.0 * c( 2 ) );
        data_u_( sd, x, y, r, 1 ) =  8.0 * Kokkos::cos( 8.0 * c( 0 ) );
        data_u_( sd, x, y, r, 2 ) = -2.0 * Kokkos::cos( 2.0 * c( 1 ) );
    }
};

struct SolutionPressureInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_p_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        data_p_( sd, x, y, r ) =
            Kokkos::sin( 4.0 * c( 0 ) ) * Kokkos::sin( 8.0 * c( 1 ) ) * Kokkos::sin( 2.0 * c( 2 ) );
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c  = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double                  x0 = 4.0 * c( 2 );

        data_( sd, x, y, r, 0 ) =
            -64.0 * ( Kokkos::sin( c( 2 ) ) + 2.0 ) * Kokkos::cos( x0 )
            - 16.0 * Kokkos::sin( x0 ) * Kokkos::cos( c( 2 ) )
            + 4.0 * Kokkos::sin( 8.0 * c( 1 ) ) * Kokkos::sin( 2.0 * c( 2 ) ) * Kokkos::cos( 4.0 * c( 0 ) );

        data_( sd, x, y, r, 1 ) =
            512.0 * ( Kokkos::sin( c( 2 ) ) + 2.0 ) * Kokkos::cos( 8.0 * c( 0 ) )
            + 8.0 * Kokkos::sin( 4.0 * c( 0 ) ) * Kokkos::sin( 2.0 * c( 2 ) ) * Kokkos::cos( 8.0 * c( 1 ) )
            - 4.0 * Kokkos::sin( 2.0 * c( 1 ) ) * Kokkos::cos( c( 2 ) );

        data_( sd, x, y, r, 2 ) =
            -8.0 * ( Kokkos::sin( c( 2 ) ) + 2.0 ) * Kokkos::cos( 2.0 * c( 1 ) )
            + 2.0 * Kokkos::sin( 4.0 * c( 0 ) ) * Kokkos::sin( 8.0 * c( 1 ) ) * Kokkos::cos( 2.0 * c( 2 ) );
    }
};

struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        data_( sd, x, y, r ) = 2.0 + Kokkos::sin( c( 2 ) );
    }
};

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

struct RunResult
{
    double l2_error_velocity;
    double l2_error_pressure;
    int    iterations;
};

enum class SchurChoice
{
    MASS,
    WBFBT,
};

RunResult run_test( int min_level, int max_level, SchurChoice schur,
                    const std::shared_ptr< util::Table >& table )
{
    using ScalarType = double;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    const ScalarType r_min = 0.5;
    const ScalarType r_max = 1.0;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const int idx = level - min_level;
        domains.push_back( DistributedDomain::create_uniform( level, level, r_min, r_max, 0, 0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const size_t num_levels     = domains.size();
    const size_t velocity_level = num_levels - 1;
    const size_t pressure_level = num_levels - 2;

    BoundaryConditions bcs         = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } };
    BoundaryConditions bcs_neumann = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };

    using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using Gradient    = Stokes::Block12Type;
    using Divergence  = Stokes::Block21Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;
    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;

    // Viscosity (variable) on every level — same KInterpolator as the parent test.
    std::vector< VectorQ1Scalar< ScalarType > > k_levels;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        k_levels.emplace_back( "k_" + std::to_string( level ), domains[level], mask_data[level] );
        Kokkos::parallel_for(
            "k_interp_" + std::to_string( level ),
            local_domain_md_range_policy_nodes( domains[level] ),
            KInterpolator{ coords_shell[level], coords_radii[level], k_levels[level].grid_data() } );
        Kokkos::fence();
    }

    // Fine-level Stokes operator with the user BCs (Dirichlet) and the Neumann
    // copy used to assemble the symmetric K_w (gradient/divergence) pieces of
    // the w-BFBT preconditioner.  K_neumann_diag matches the construction in
    // test_epsilon_divdiv_stokes.cpp so the Dirichlet enforcement on the RHS
    // mirrors the production path.
    Stokes K(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], k_levels[velocity_level].grid_data(),
        bcs, false );

    Stokes K_neumann(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], k_levels[velocity_level].grid_data(),
        bcs_neumann, false );

    Stokes K_neumann_diag(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], k_levels[velocity_level].grid_data(),
        bcs_neumann, true );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // MG hierarchy on the viscous block (Jacobi smoother + PCG coarse).
    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg, tmp_mg_r, tmp_mg_e;

    for ( size_t level = 0; level < num_levels; ++level )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );

            A_c.emplace_back(
                domains[level], coords_shell[level], coords_radii[level],
                boundary_mask_data[level], k_levels[level].grid_data(),
                bcs, false );

            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;
    using Smoother = linalg::solvers::Jacobi< Viscous >;
    std::vector< Smoother > smoothers;

    for ( size_t level = 0; level < num_levels; ++level )
    {
        inverse_diagonals.emplace_back(
            "inv_diag_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > ones(
            "inv_diag_ones_" + std::to_string( level ), domains[level], mask_data[level] );
        linalg::assign( ones, 1.0 );

        if ( level == velocity_level )
        {
            K.block_11().set_diagonal( true );
            linalg::apply( K.block_11(), ones, inverse_diagonals.back() );
            K.block_11().set_diagonal( false );
        }
        else
        {
            A_c[level].set_diagonal( true );
            linalg::apply( A_c[level], ones, inverse_diagonals.back() );
            A_c[level].set_diagonal( false );
        }
        linalg::invert_entries( inverse_diagonals.back() );

        VectorQ1Vec< ScalarType > pi0( "pi0_" + std::to_string( level ), domains[level], mask_data[level] );
        VectorQ1Vec< ScalarType > pi1( "pi1_" + std::to_string( level ), domains[level], mask_data[level] );
        double max_ev = 0.0;
        if ( level == velocity_level )
        {
            DiagonallyScaledOperator< Viscous > inv_diag_A( K.block_11(), inverse_diagonals[level] );
            max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, pi0, pi1, 100 );
        }
        else
        {
            DiagonallyScaledOperator< Viscous > inv_diag_A( A_c[level], inverse_diagonals[level] );
            max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, pi0, pi1, 100 );
        }
        const auto omega_opt = 2.0 / ( 1.3 * max_ev );
        constexpr int smoother_prepost = 3;
        smoothers.emplace_back( inverse_diagonals[level], smoother_prepost, tmp_mg[level], omega_opt );
    }

    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;
    std::vector< VectorQ1Vec< ScalarType > > coarse_tmps;
    for ( int i = 0; i < 4; ++i )
        coarse_tmps.emplace_back( "coarse_tmp", domains[0], mask_data[0] );
    CoarseGridSolver coarse_solver(
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-6, 1e-16 }, table, coarse_tmps );

    using PrecVisc = linalg::solvers::Multigrid<
        Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    constexpr int num_mg_cycles = 1;
    PrecVisc prec_visc(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_solver, num_mg_cycles, 1e-8 );

    // PressureMass on the pressure level — needed both as the (2,2)-block
    // OperatorType slot for `PrecStokes` and (with k = 1/eta) as the lumped
    // mass-Schur preconditioner in the MASS branch.
    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[pressure_level], mask_data[pressure_level] );
    linalg::assign( k_pm, k_levels[pressure_level] );
    linalg::invert_entries( k_pm );

    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level],
        k_pm.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    VectorQ1Scalar< ScalarType > lumped_diagonal_pmass(
        "lumped_diagonal_pmass", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > ones_p( "ones_p", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( ones_p, 1.0 );
        linalg::apply( pmass, ones_p, lumped_diagonal_pmass );
    }

    // Build the Schur preconditioner of the requested kind, wrapped in
    // PolymorphicSchurPreconditioner so `PrecStokes` has a single static type.
    using PrecSchur = mc::PolymorphicSchurPreconditioner< PressureMass >;
    PrecSchur prec_schur;

    // sqrt(eta) on velocity level + C_w + K_w solver — only constructed on the
    // WBFBT branch but declared here so their storage outlives PrecSchur.
    using WBFBTCw = mc::WBFBTWeightedLumpedVelocityMass< ScalarType, 3 >;
    std::unique_ptr< WBFBTCw > c_w;
    VectorQ1Scalar< ScalarType > sqrt_eta_velocity(
        "sqrt_eta_velocity", domains[velocity_level], mask_data[velocity_level] );

    using WBFBTKw = mc::ExplicitKwPressurePoissonSolver< ScalarType, Gradient, Divergence >;
    std::shared_ptr< mc::WBFBTPressurePoissonSolver< ScalarType > > kw_solver;

    if ( schur == SchurChoice::MASS )
    {
        util::logroot << "Schur PC: lumped pressure mass M_p(1/eta)\n";
        DiagonalSolver< PressureMass > diag_solver( lumped_diagonal_pmass );
        prec_schur = PrecSchur::make( std::move( diag_solver ) );
    }
    else
    {
        util::logroot << "Schur PC: w-BFBT (explicit K_w + lumped C_w(sqrt eta))\n";

        // sqrt(eta) on velocity level — eta = 2 + sin(z) is strictly positive.
        linalg::assign( sqrt_eta_velocity, k_levels[velocity_level] );
        Kokkos::parallel_for(
            "sqrt_eta_velocity",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                auto& v          = sqrt_eta_velocity.grid_data()( sd, x, y, r );
                v                = Kokkos::sqrt( v );
            } );
        Kokkos::fence();

        c_w = std::make_unique< WBFBTCw >(
            domains[velocity_level], coords_shell[velocity_level],
            coords_radii[velocity_level], mask_data[velocity_level] );
        c_w->refresh( sqrt_eta_velocity );

        kw_solver = std::make_shared< WBFBTKw >(
            K_neumann.block_12(),
            K_neumann.block_21(),
            c_w->inv_diag_velocity(),
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level],
            /*max_iterations=*/200,
            /*relative_tol=*/static_cast< ScalarType >( 1e-6 ),
            table );

        using WBFBTSchur =
            mc::WBFBTSchurPreconditioner< Viscous, Gradient, Divergence, PressureMass >;
        WBFBTSchur wbfbt(
            K.block_11(),
            K_neumann.block_12(),
            K_neumann.block_21(),
            kw_solver,
            c_w->inv_diag_velocity(),
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level] );

        prec_schur = PrecSchur::make( std::move( wbfbt ) );
    }

    using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
        Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

    VectorQ1IsoQ2Q1< ScalarType > triangular_prec_tmp(
        "triangular_prec_tmp",
        domains[velocity_level], domains[pressure_level],
        mask_data[velocity_level], mask_data[pressure_level] );

    PrecStokes prec_stokes( K.block_11(), pmass, K.block_12(), triangular_prec_tmp, prec_visc, prec_schur );

    // RHS: interpolate the analytical body force on velocity nodes, lift to the
    // FE coefficient vector via M, then strong-enforce the analytical velocity
    // values at Dirichlet boundary nodes (mirroring test_epsilon_divdiv_stokes).
    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    for ( const auto& name : { "u", "f", "solution", "error", "tmp_0", "tmp_1" } )
    {
        stok_vecs[name] = VectorQ1IsoQ2Q1< ScalarType >(
            name,
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level] );
    }

    Kokkos::parallel_for(
        "sol_velocity_interp_full",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["solution"].block_1().grid_data(),
            boundary_mask_data[velocity_level], /*only_boundary=*/false } );

    Kokkos::parallel_for(
        "sol_pressure_interp_full",
        local_domain_md_range_policy_nodes( domains[pressure_level] ),
        SolutionPressureInterpolator{
            coords_shell[pressure_level], coords_radii[pressure_level],
            stok_vecs["solution"].block_2().grid_data() } );

    Kokkos::parallel_for(
        "rhs_velocity_interp",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        RHSVelocityInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["tmp_1"].block_1().grid_data() } );

    linalg::apply( M, stok_vecs["tmp_1"].block_1(), stok_vecs["f"].block_1() );

    // Boundary-only velocity values for the Dirichlet enforcement on RHS.
    Kokkos::parallel_for(
        "sol_velocity_interp_boundary",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["tmp_0"].block_1().grid_data(),
            boundary_mask_data[velocity_level], /*only_boundary=*/true } );

    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann,
        K_neumann_diag,
        stok_vecs["tmp_0"],
        stok_vecs["tmp_1"],
        stok_vecs["f"],
        boundary_mask_data[velocity_level],
        BOUNDARY );

    // Outer FGMRES — restart=100, max_iters=300 gives the test enough budget to
    // converge to 1e-10 at level 5 while keeping FGMRES tmps (2·restart+4 =
    // 204 IsoQ2Q1 vectors) well within single-GPU memory.
    constexpr int fgmres_restart = 100;
    constexpr int fgmres_max     = 300;
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < 2 * fgmres_restart + 4; ++i )
    {
        tmp_fgmres.emplace_back(
            "tmp_fgmres_" + std::to_string( i ),
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level] );
    }
    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                     = fgmres_restart;
    fgmres_options.max_iterations              = fgmres_max;
    fgmres_options.relative_residual_tolerance = 1e-10;
    fgmres_options.absolute_residual_tolerance = 1e-16;

    auto solver_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( tmp_fgmres, fgmres_options, solver_table, prec_stokes );
    fgmres.set_tag( "stokes_fgmres" );

    linalg::assign( stok_vecs["u"], 0.0 );
    linalg::solvers::solve( fgmres, K, stok_vecs["u"], stok_vecs["f"] );

    solver_table->query_rows_equals( "tag", "stokes_fgmres" )
        .select_columns( { "absolute_residual", "relative_residual", "iteration" } )
        .print_pretty();

    const auto num_dofs_velocity =
        3 * kernels::common::count_masked< long >( mask_data[velocity_level], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( mask_data[pressure_level], grid::NodeOwnershipFlag::OWNED );

    auto& u        = stok_vecs["u"];
    auto& solution = stok_vecs["solution"];
    auto& error    = stok_vecs["error"];

    const double avg_p_sol = kernels::common::masked_sum(
        solution.block_2().grid_data(), solution.block_2().mask_data(),
        grid::NodeOwnershipFlag::OWNED ) / num_dofs_pressure;
    const double avg_p_app = kernels::common::masked_sum(
        u.block_2().grid_data(), u.block_2().mask_data(),
        grid::NodeOwnershipFlag::OWNED ) / num_dofs_pressure;
    linalg::lincomb( solution.block_2(), { 1.0 }, { solution.block_2() }, -avg_p_sol );
    linalg::lincomb( u.block_2(),        { 1.0 }, { u.block_2() },        -avg_p_app );

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const double l2_err_u = std::sqrt(
        dot( error.block_1(), error.block_1() ) / static_cast< double >( num_dofs_velocity ) );
    const double l2_err_p = std::sqrt(
        dot( error.block_2(), error.block_2() ) / static_cast< double >( num_dofs_pressure ) );

    const int iters =
        static_cast< int >( solver_table->query_rows_equals( "tag", "stokes_fgmres" ).rows().size() );

    util::logroot << "  l2_error_u = " << l2_err_u
                  << "  l2_error_p = " << l2_err_p
                  << "  outer iters = " << iters << "\n";

    return { l2_err_u, l2_err_p, iters };
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    // h-convergence over refinement levels 3, 4, 5.  Both PCs must (a) reach
    // the analytic solution within a generous absolute threshold and (b)
    // exhibit an L2-error reduction factor consistent with the discretization
    // order (Q1-iso-Q2 velocity, Q1 pressure).
    constexpr int          min_level   = 2;
    std::vector< int >     max_levels  = { 3, 4, 5 };

    auto table = std::make_shared< util::Table >();

    util::logroot << "=== test_epsilon_divdiv_stokes_wbfbt  "
                  << "(refinement levels " << max_levels.front()
                  << ".." << max_levels.back() << ") ===\n";

    std::vector< double > err_u_mass,  err_p_mass;
    std::vector< double > err_u_wbfbt, err_p_wbfbt;
    std::vector< int >    iters_mass, iters_wbfbt;

    bool ok = true;

    for ( int lvl : max_levels )
    {
        util::logroot << "\n--- level " << lvl << " : mass-Schur PC ---\n";
        const auto mass  = run_test( min_level, lvl, SchurChoice::MASS,  table );
        util::logroot <<  "\n--- level " << lvl << " : w-BFBT Schur PC ---\n";
        const auto wbfbt = run_test( min_level, lvl, SchurChoice::WBFBT, table );

        err_u_mass.push_back( mass.l2_error_velocity );
        err_p_mass.push_back( mass.l2_error_pressure );
        err_u_wbfbt.push_back( wbfbt.l2_error_velocity );
        err_p_wbfbt.push_back( wbfbt.l2_error_pressure );
        iters_mass.push_back( mass.iterations );
        iters_wbfbt.push_back( wbfbt.iterations );

        // Both PCs solve the same discrete saddle-point system to the same
        // outer FGMRES tolerance — their L2 errors against the manufactured
        // solution must agree to several digits (limited by the FGMRES tol).
        constexpr double cross_pc_tol = 1e-6;
        if ( std::abs( mass.l2_error_velocity - wbfbt.l2_error_velocity ) > cross_pc_tol
             || std::abs( mass.l2_error_pressure - wbfbt.l2_error_pressure ) > cross_pc_tol )
        {
            util::logroot << "FAIL: mass vs wbfbt errors disagree at level " << lvl
                          << " by more than " << cross_pc_tol << "\n";
            ok = false;
        }
    }

    // h-convergence rates between consecutive levels.  Q1-iso-Q2 velocity =
    // O(h^2) → ratio err(L+1)/err(L) → 0.25.  Q1 pressure = O(h) → ratio → 0.5.
    // We accept anything below 0.7 (loose bound that catches stagnation /
    // divergence with refinement).
    constexpr double rate_threshold_u = 0.5;
    constexpr double rate_threshold_p = 0.7;

    util::logroot << "\n--- summary ---\n";
    for ( size_t i = 0; i < max_levels.size(); ++i )
    {
        util::logroot << "level=" << max_levels[i]
                      << "  err_u(mass)=" << err_u_mass[i]
                      << "  err_p(mass)=" << err_p_mass[i]
                      << "  iters(mass)=" << iters_mass[i]
                      << "  err_u(wbfbt)=" << err_u_wbfbt[i]
                      << "  err_p(wbfbt)=" << err_p_wbfbt[i]
                      << "  iters(wbfbt)=" << iters_wbfbt[i] << "\n";
    }

    for ( size_t i = 1; i < max_levels.size(); ++i )
    {
        const double r_u_m = err_u_mass[i]  / std::max( err_u_mass[i - 1],  1e-30 );
        const double r_p_m = err_p_mass[i]  / std::max( err_p_mass[i - 1],  1e-30 );
        const double r_u_w = err_u_wbfbt[i] / std::max( err_u_wbfbt[i - 1], 1e-30 );
        const double r_p_w = err_p_wbfbt[i] / std::max( err_p_wbfbt[i - 1], 1e-30 );
        util::logroot << "ratio  L" << max_levels[i - 1] << "->L" << max_levels[i]
                      << "  u(mass)="  << r_u_m  << "  p(mass)="  << r_p_m
                      << "  u(wbfbt)=" << r_u_w  << "  p(wbfbt)=" << r_p_w << "\n";

        if ( r_u_w > rate_threshold_u || r_p_w > rate_threshold_p )
        {
            util::logroot << "FAIL: wbfbt convergence rate too slow at L"
                          << max_levels[i] << " (u_ratio=" << r_u_w
                          << " p_ratio=" << r_p_w << ")\n";
            ok = false;
        }
        if ( r_u_m > rate_threshold_u || r_p_m > rate_threshold_p )
        {
            util::logroot << "FAIL: mass convergence rate too slow at L"
                          << max_levels[i] << " (u_ratio=" << r_u_m
                          << " p_ratio=" << r_p_m << ")\n";
            ok = false;
        }
    }

    if ( !ok )
        Kokkos::abort( "test_epsilon_divdiv_stokes_wbfbt failed" );

    return 0;
}
