// Stokes solver convergence test set up to mirror the A7 benchmark's INITIAL
// state (timestep 0, before any Stokes solve has run): Ra=7e3, Frank-Kamenetskii
// viscosity with rmu=1e5 (~1e5 contrast top↔bottom), free-slip on both shells,
// radii from the A7 config (kRm=1.22, kRp=2.22).  The temperature field is
// the same conductive profile + Y_3^2 lateral perturbation (eps=0.01) used as
// the A7 initial condition.
//
// The test runs the same solver structure mc::StokesContext uses:
//   - Cheby-Jacobi smoothed MG on the viscous block,
//   - FGMRES on the coarse level with inverse-diagonal preconditioning,
//   - outer FGMRES on the Stokes saddle-point system,
// once with the lumped pressure-mass Schur PC (M_p(1/eta)) and once with the
// w-BFBT Schur PC (explicit K_w via FGMRES, lumped C_w(sqrt eta), Neumann B/B^T).
// Both PCs go through the same outer FGMRES instance via
// PolymorphicSchurPreconditioner.
//
// There is no analytical solution to compare against — variable-viscosity
// Stokes has no closed form — so the test only asserts that both PCs drive the
// outer relative residual below a generous tolerance within the iteration
// budget.  The recorded iteration counts are the actual diagnostic.

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/strong_algebraic_freeslip_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp"

#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "terra/linalg/solvers/chebyshev.hpp"

#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"

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
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::local_domain_md_range_policy_nodes;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using terra::grid::shell::BoundaryConditions;

namespace mc = terra::mantlecirculation;

// A7 benchmark geometry + viscosity parameters.  Matches
// config_bench_A7_freeslip_wbfbt_pic2_prod.toml.
constexpr double kRm        = 1.22;
constexpr double kRp        = 2.22;
constexpr double kRmu       = 1.0e5;  // FK contrast (eta_max / eta_min)
constexpr double kRayleigh  = 7.0e3;
constexpr double kPerturbDefault = 0.01;  // initial-temperature-sph-epsilon

constexpr auto BOUNDARY = static_cast< grid::shell::ShellBoundaryFlag >(
    static_cast< uint8_t >( CMB ) | static_cast< uint8_t >( SURFACE ) );

// Real spherical harmonic Y_3^2(θ, φ) = sqrt(105/(32π)) sin²θ cosθ cos(2φ).
// Matches the (sph-degree=3, sph-order=2) initial-T perturbation in A7.
KOKKOS_INLINE_FUNCTION
double Y32( const double theta, const double phi )
{
    constexpr double c  = 1.0219854764332823;
    const double     st = Kokkos::sin( theta );
    const double     ct = Kokkos::cos( theta );
    return c * st * st * ct * Kokkos::cos( 2.0 * phi );
}

// Returns A7 initial T(r, θ, φ) clamped to [0, 1].
KOKKOS_INLINE_FUNCTION
double a7_initial_temperature( const double cx, const double cy, const double cz, double perturb_amp )
{
    const double rad = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    double T_lateral = 0.0;
    if ( perturb_amp != 0.0 && rad > 0.0 )
    {
        const double theta = Kokkos::acos( cz / rad );
        const double phi   = Kokkos::atan2( cy, cx );
        T_lateral          = perturb_amp * Y32( theta, phi );
    }
    return Kokkos::clamp( ( kRp - rad ) / ( kRp - kRm ) + T_lateral, 0.0, 1.0 );
}

// eta(T) = rmu^(0.5 - T), same as apps/mantlecirculation/src/interpolators.hpp.
struct FrankKamenetskiiViscosityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    grid::Grid4DDataScalar< double > data_;
    double                     rmu_;
    double                     perturb_amp_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double T = a7_initial_temperature( c( 0 ), c( 1 ), c( 2 ), perturb_amp_ );
        data_( sd, x, y, r ) = Kokkos::pow( rmu_, 0.5 - T );
    }
};

// Radial buoyancy body force f(x) = -Ra · T(x) · r̂.  Uses the full A7 T
// (conductive + Y_3^2 perturbation), matching what the mc app feeds into
// EpsDivDivStokes at the initial timestep.
struct RadialBuoyancyRHSInterpolator
{
    Grid3DDataVec< double, 3 >       grid_;
    Grid2DDataScalar< double >       radii_;
    grid::Grid4DDataVec< double, 3 > data_;
    double                           rayleigh_;
    double                           perturb_amp_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double rad = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
        if ( rad <= 0.0 )
        {
            for ( int d = 0; d < 3; ++d )
                data_( sd, x, y, r, d ) = 0.0;
            return;
        }
        const double T = a7_initial_temperature( c( 0 ), c( 1 ), c( 2 ), perturb_amp_ );
        const double f_radial = -rayleigh_ * T;
        for ( int d = 0; d < 3; ++d )
            data_( sd, x, y, r, d ) = f_radial * c( d ) / rad;
    }
};

// Inverse-diagonal-as-preconditioner SolverLike wrapper for the coarse FGMRES.
// Same pattern as the one in apps/mantlecirculation/src/stokes_solver.hpp.
template < linalg::OperatorLike OperatorT >
struct InverseDiagonalPreconditioner
{
    using OperatorType       = OperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;

    SolutionVectorType inv_diag_;

    explicit InverseDiagonalPreconditioner( const SolutionVectorType& d )
    : inv_diag_( d )
    {}

    void solve_impl( OperatorType& /*A*/, SolutionVectorType& x, const RHSVectorType& b )
    {
        linalg::assign( x, b );
        linalg::scale_in_place( x, inv_diag_ );
    }
};

enum class SchurChoice { MASS, WBFBT };

struct RunResult
{
    int    iters;
    double rel_residual_final;
};

RunResult run_test( int    min_level,
                    int    max_level,
                    int    cheby_order,
                    double perturb_amp,
                    SchurChoice schur,
                    const std::shared_ptr< util::Table >& solver_table )
{
    using ScalarType = double;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    util::logroot << "Allocating domains (levels " << min_level << ".." << max_level << ") ...\n";
    for ( int level = min_level; level <= max_level; ++level )
    {
        const int idx = level - min_level;
        domains.push_back( DistributedDomain::create_uniform( level, level, kRm, kRp, 0, 0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const size_t num_levels     = domains.size();
    const size_t velocity_level = num_levels - 1;
    const size_t pressure_level = num_levels - 2;

    BoundaryConditions bcs = {
        { CMB, FREESLIP },
        { SURFACE, FREESLIP },
    };
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using Gradient    = Stokes::Block12Type;
    using Divergence  = Stokes::Block21Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;
    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;

    // Per-level viscosity fields.
    std::vector< VectorQ1Scalar< ScalarType > > eta;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        eta.emplace_back( "eta_" + std::to_string( level ), domains[level], mask_data[level] );
        Kokkos::parallel_for(
            "fk_viscosity_level_" + std::to_string( level ),
            local_domain_md_range_policy_nodes( domains[level] ),
            FrankKamenetskiiViscosityInterpolator{
                coords_shell[level], coords_radii[level], eta[level].grid_data(), kRmu, perturb_amp } );
        Kokkos::fence();
    }

    util::logroot << "  rmu = " << kRmu
                  << "  eta range = [" << std::pow( kRmu, -0.5 )
                  << ", " << std::pow( kRmu, 0.5 ) << "]\n";

    // Fine outer Stokes operator (free-slip), and the Neumann copy used to
    // build the symmetric B / B^T inside K_w on the WBFBT path.
    Stokes K_op(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], eta[velocity_level].grid_data(),
        bcs, false );

    Stokes K_op_neumann(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], eta[velocity_level].grid_data(),
        bcs_neumann, false );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // Coarse viscous operators + prolongations/restrictions for MG.
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
                boundary_mask_data[level], eta[level].grid_data(), bcs, false );

            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // Inverse diagonals per level.
    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        inverse_diagonals.emplace_back(
            "inv_diag_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > ones(
            "inv_diag_ones_" + std::to_string( level ), domains[level], mask_data[level] );
        linalg::assign( ones, 1.0 );

        if ( level == velocity_level )
        {
            K_op.block_11().set_diagonal( true );
            linalg::apply( K_op.block_11(), ones, inverse_diagonals.back() );
            K_op.block_11().set_diagonal( false );
        }
        else
        {
            A_c[level].set_diagonal( true );
            linalg::apply( A_c[level], ones, inverse_diagonals.back() );
            A_c[level].set_diagonal( false );
        }
        linalg::invert_entries( inverse_diagonals.back() );
    }

    // Chebyshev smoothers per level.
    using Smoother = linalg::solvers::Chebyshev< Viscous >;
    std::vector< Smoother > smoothers;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        std::vector< VectorQ1Vec< ScalarType > > cheby_tmps;
        cheby_tmps.emplace_back( "cheby_tmp_0_" + std::to_string( level ), domains[level], mask_data[level] );
        cheby_tmps.emplace_back( "cheby_tmp_1_" + std::to_string( level ), domains[level], mask_data[level] );

        constexpr int chebyshev_prepost = 3;
        smoothers.emplace_back( cheby_order, inverse_diagonals[level], cheby_tmps, chebyshev_prepost );
    }

    // Coarse FGMRES (matches the mc app's coarse solver in stokes_solver.hpp).
    using CoarseGridPrec   = InverseDiagonalPreconditioner< Viscous >;
    using CoarseGridSolver = linalg::solvers::FGMRES< Viscous, CoarseGridPrec >;
    constexpr int coarse_restart = 30;
    constexpr int coarse_max     = 50;
    std::vector< VectorQ1Vec< ScalarType > > coarse_tmps;
    for ( int i = 0; i < 2 * coarse_restart + 4; ++i )
        coarse_tmps.emplace_back( "coarse_tmp", domains[0], mask_data[0] );
    linalg::solvers::FGMRESOptions< ScalarType > coarse_opts;
    coarse_opts.restart                     = coarse_restart;
    coarse_opts.max_iterations              = coarse_max;
    coarse_opts.relative_residual_tolerance = 1e-6;
    coarse_opts.absolute_residual_tolerance = 1e-16;
    CoarseGridSolver coarse_solver( coarse_tmps, coarse_opts, solver_table, CoarseGridPrec( inverse_diagonals[0] ) );
    coarse_solver.set_tag( "coarse_grid_fgmres" );

    using PrecVisc = linalg::solvers::Multigrid<
        Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    constexpr int num_mg_cycles = 1;
    PrecVisc prec_visc(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_solver, num_mg_cycles, 1e-6 );

    // PressureMass operator on the pressure level — slotted into PrecStokes as
    // the (2,2)-block OperatorType, and (with k=1/eta) used by the MASS path
    // as the lumped Schur preconditioner.
    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[pressure_level], mask_data[pressure_level] );
    linalg::assign( k_pm, eta[pressure_level] );
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

    using PrecSchur = mc::PolymorphicSchurPreconditioner< PressureMass >;
    PrecSchur prec_schur;

    // Pieces for the WBFBT branch — declared at outer scope so storage outlives
    // PrecSchur (which only holds references via the type-erased holder).
    using WBFBTCw = mc::WBFBTWeightedLumpedVelocityMass< ScalarType, 3 >;
    std::unique_ptr< WBFBTCw > c_w;
    VectorQ1Scalar< ScalarType > sqrt_eta_velocity(
        "sqrt_eta_velocity", domains[velocity_level], mask_data[velocity_level] );

    using WBFBTKw = mc::ExplicitKwPressurePoissonSolver< ScalarType, Gradient, Divergence >;
    std::shared_ptr< mc::WBFBTPressurePoissonSolver< ScalarType > > kw_solver;

    if ( schur == SchurChoice::MASS )
    {
        util::logroot << "Schur PC: lumped pressure mass M_p(1/eta)\n";
        linalg::solvers::DiagonalSolver< PressureMass > diag_solver( lumped_diagonal_pmass );
        prec_schur = PrecSchur::make( std::move( diag_solver ) );
    }
    else
    {
        util::logroot << "Schur PC: w-BFBT (explicit K_w + lumped C_w(sqrt eta))\n";

        linalg::assign( sqrt_eta_velocity, eta[velocity_level] );
        Kokkos::parallel_for(
            "sqrt_eta_velocity",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                auto& v = sqrt_eta_velocity.grid_data()( sd, x, y, r );
                v       = Kokkos::sqrt( v );
            } );
        Kokkos::fence();

        c_w = std::make_unique< WBFBTCw >(
            domains[velocity_level], coords_shell[velocity_level],
            coords_radii[velocity_level], mask_data[velocity_level] );
        c_w->refresh( sqrt_eta_velocity );

        kw_solver = std::make_shared< WBFBTKw >(
            K_op_neumann.block_12(),
            K_op_neumann.block_21(),
            c_w->inv_diag_velocity(),
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level],
            /*max_iterations=*/200,
            /*relative_tol=*/static_cast< ScalarType >( 1e-6 ),
            solver_table );

        using WBFBTSchur =
            mc::WBFBTSchurPreconditioner< Viscous, Gradient, Divergence, PressureMass >;
        WBFBTSchur wbfbt(
            K_op.block_11(),
            K_op_neumann.block_12(),
            K_op_neumann.block_21(),
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
    PrecStokes prec_stokes(
        K_op.block_11(), pmass, K_op.block_12(), triangular_prec_tmp, prec_visc, prec_schur );

    // RHS: M · (-Ra · T · r̂), strong-enforced for the free-slip boundary.
    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    for ( const auto& name : { "u", "f", "tmp" } )
    {
        stok_vecs[name] = VectorQ1IsoQ2Q1< ScalarType >(
            name,
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level] );
    }

    Kokkos::parallel_for(
        "radial_buoyancy_rhs",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        RadialBuoyancyRHSInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["tmp"].block_1().grid_data(), kRayleigh, perturb_amp } );
    Kokkos::fence();

    linalg::apply( M, stok_vecs["tmp"].block_1(), stok_vecs["f"].block_1() );
    fe::strong_algebraic_freeslip_enforcement_in_place(
        stok_vecs["f"], coords_shell[velocity_level], boundary_mask_data[velocity_level], BOUNDARY );

    // Outer FGMRES.
    constexpr int fgmres_restart   = 50;
    constexpr int fgmres_max_iters = 100;
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
    fgmres_options.max_iterations              = fgmres_max_iters;
    fgmres_options.relative_residual_tolerance = 1e-6;
    fgmres_options.absolute_residual_tolerance = 1e-16;

    auto outer_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres(
        tmp_fgmres, fgmres_options, outer_table, prec_stokes );
    fgmres.set_tag( "stokes_fgmres" );

    linalg::assign( stok_vecs["u"], 0.0 );
    linalg::solvers::solve( fgmres, K_op, stok_vecs["u"], stok_vecs["f"] );

    outer_table->query_rows_equals( "tag", "stokes_fgmres" )
        .select_columns( { "absolute_residual", "relative_residual", "iteration" } )
        .print_pretty();

    const auto rows = outer_table->query_rows_equals( "tag", "stokes_fgmres" ).rows();
    const int iters = static_cast< int >( rows.size() );
    const double rel_residual_final = rows.empty() ? 1.0 :
        std::get< double >( rows.back().at( "relative_residual" ) );

    return { iters, rel_residual_final };
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    // Single-level run: this is a regression / sanity check against the A7
    // initial state, not a convergence study.  Level 5 matches the mc app's
    // benchmark refinement and fits comfortably on a single H100.
    int    min_level   = 2;
    int    max_level   = 5;
    int    cheby_order = 4;
    double perturb_amp = kPerturbDefault;
    for ( int i = 1; i + 1 < argc; ++i )
    {
        const std::string a = argv[i];
        if ( a == "--min-level" )         min_level   = std::atoi( argv[i + 1] );
        else if ( a == "--max-level" )    max_level   = std::atoi( argv[i + 1] );
        else if ( a == "--cheby-order" )  cheby_order = std::atoi( argv[i + 1] );
        else if ( a == "--perturb-amp" )  perturb_amp = std::atof( argv[i + 1] );
    }

    util::logroot << "=== A7-initial Stokes Schur-PC test  (level " << max_level
                  << ", fs/fs, FK rmu=" << kRmu
                  << ", perturb_amp=" << perturb_amp << ") ===\n";

    auto table = std::make_shared< util::Table >();

    util::logroot << "--- run with mass-Schur PC ---\n";
    const auto mass = run_test( min_level, max_level, cheby_order, perturb_amp, SchurChoice::MASS, table );

    util::logroot << "--- run with w-BFBT Schur PC ---\n";
    const auto wbfbt = run_test( min_level, max_level, cheby_order, perturb_amp, SchurChoice::WBFBT, table );

    // Only the w-BFBT PC is asserted to converge: mass-Schur is known to
    // stagnate at A7 contrast 1e5 (see notes in apps/mantlecirculation/) and
    // its iteration count is the diagnostic point of the comparison.
    constexpr double residual_tol = 1.0e-5;  // loose vs the 1e-6 FGMRES tol

    util::logroot << "Summary  mass: iters=" << mass.iters
                  << " rel_res=" << mass.rel_residual_final
                  << "   wbfbt: iters=" << wbfbt.iters
                  << " rel_res=" << wbfbt.rel_residual_final << "\n";

    if ( wbfbt.rel_residual_final > residual_tol )
    {
        util::logroot << "FAIL: w-BFBT did not converge (rel_res="
                      << wbfbt.rel_residual_final << ")\n";
        Kokkos::abort( "test_epsilon_divdiv_stokes_viscosity_contrast failed" );
    }

    return 0;
}
