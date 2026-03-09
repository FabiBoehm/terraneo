
// Stokes Schur complement preconditioner comparison test.
//
// Compares iteration counts for different Schur complement approximations
// in a block preconditioner for the Stokes system:
//
//   1. Block-diagonal:   MG + Identity        (no pressure preconditioning)
//   2. Block-diagonal:   MG + lumped M_p(1/mu) (inverse-viscosity-weighted pressure mass)
//   3. Block-triangular: MG + lumped M_p(1/mu) (current default approach)
//   4. Block-diagonal:   MG + diag(L_p)        (pressure Laplacian diagonal)
//   5. Block-triangular: MG + diag(L_p)        (pressure Laplacian diagonal)
//   6. Block-diagonal:   MG + L_p via PCG(10)   (pressure Laplacian PCG)
//   7. Block-triangular: MG + L_p via PCG(10)   (pressure Laplacian PCG)
//   8. Block-diagonal:   MG + diag(div(1/mu grad)) (viscosity-weighted pressure Laplacian)
//   9. Block-triangular: MG + diag(div(1/mu grad)) (viscosity-weighted pressure Laplacian)
//
// Each configuration is tested for:
//   - Constant viscosity (k=1)
//   - Radial profile: Lin et al. 2022
//   - Radial profile: Stotz et al. 2017

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/identity.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/div_k_grad.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp"

#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/pminres.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "util/info.hpp"

#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/geophysics/viscosity/viscosity_interpolation.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"
#include "terra/linalg/solvers/identity_solver.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "terra/shell/radial_profiles.hpp"

#include "util/init.hpp"
#include "util/table.hpp"

#include <algorithm>
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

// ---------------------------------------------------------------------------
// Viscosity mode
// ---------------------------------------------------------------------------

enum class ViscosityMode
{
    Constant,
    LinEtAl2022,
    StotzEtAl2017
};

std::string viscosity_mode_label( ViscosityMode mode )
{
    switch ( mode )
    {
    case ViscosityMode::Constant:     return "constant (k=1)";
    case ViscosityMode::LinEtAl2022:  return "Lin et al. 2022";
    case ViscosityMode::StotzEtAl2017: return "Stotz et al. 2017";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Interpolators (for RHS / boundary data)
// ---------------------------------------------------------------------------

struct SolutionVelocityInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_u_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    SolutionVelocityInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataVec< double, 3 >&                         data_u,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const bool                                                only_boundary )
    : grid_( grid ), radii_( radii ), data_u_( data_u ), mask_( mask ), only_boundary_( only_boundary ) {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const bool on_boundary = util::has_flag( mask_( local_subdomain_id, x, y, r ), BOUNDARY );
        if ( !only_boundary_ || on_boundary )
        {
            data_u_( local_subdomain_id, x, y, r, 0 ) = -4 * Kokkos::cos( 4 * coords( 2 ) );
            data_u_( local_subdomain_id, x, y, r, 1 ) =  8 * Kokkos::cos( 8 * coords( 0 ) );
            data_u_( local_subdomain_id, x, y, r, 2 ) = -2 * Kokkos::cos( 2 * coords( 1 ) );
        }
    }
};

struct SolutionPressureInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataScalar< double >                         data_p_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    SolutionPressureInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataScalar< double >&                         data_p,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const bool                                                only_boundary )
    : grid_( grid ), radii_( radii ), data_p_( data_p ), mask_( mask ), only_boundary_( only_boundary ) {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const bool on_boundary = util::has_flag( mask_( local_subdomain_id, x, y, r ), BOUNDARY );
        if ( !only_boundary_ || on_boundary )
        {
            data_p_( local_subdomain_id, x, y, r ) =
                Kokkos::sin( 4 * coords( 0 ) ) * Kokkos::sin( 8 * coords( 1 ) ) * Kokkos::sin( 2 * coords( 2 ) );
        }
    }
};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

using ScalarType = double;

using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
using Viscous     = Stokes::Block11Type;
using Gradient    = Stokes::Block12Type;
using Divergence  = Stokes::Block21Type;
using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

using PressureMass       = fe::wedge::operators::shell::KMass< ScalarType >;
using PressureLaplace    = fe::wedge::operators::shell::LaplaceSimple< ScalarType >;
using PressureDivKGrad   = fe::wedge::operators::shell::DivKGrad< ScalarType >;

using Smoother         = linalg::solvers::Jacobi< Viscous >;
using CoarseGridSolver = linalg::solvers::PCG< Viscous >;
using PrecVisc         = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;

// ---------------------------------------------------------------------------
// Interpolate viscosity coefficient for a given level
// ---------------------------------------------------------------------------

void interpolate_viscosity(
    ViscosityMode                                                  mode,
    const DistributedDomain&                                       domain,
    const Grid3DDataVec< double, 3 >&                              coords_shell,
    const Grid2DDataScalar< double >&                              coords_radii,
    const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&      boundary_mask_data,
    VectorQ1Scalar< ScalarType >&                                  k )
{
    switch ( mode )
    {
    case ViscosityMode::Constant:
    {
        linalg::assign( k, 1.0 );
        break;
    }
    case ViscosityMode::LinEtAl2022:
    {
        const std::string csv_path =
            std::string( TERRANEO_SOURCE_DIR ) + "/data/radialprofiles/ViscosityProfile_Lin_et_al_2022.csv";
        auto radial_profile = shell::interpolate_radial_profile_into_subdomains_from_csv(
            csv_path, "radius_normalized_0p5_1p0", "viscosity_scaled_by_min", coords_radii );
        geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType > interpolator( radial_profile );
        interpolator.interpolate( k.grid_data() );
        break;
    }
    case ViscosityMode::StotzEtAl2017:
    {
        const std::string csv_path =
            std::string( TERRANEO_SOURCE_DIR ) + "/data/radialprofiles/ViscosityProfile_Stotz_et_al_2017.csv";
        auto radial_profile = shell::interpolate_radial_profile_into_subdomains_from_csv(
            csv_path, "radius_normalized_0p5_1p0", "viscosity_scaled_by_min", coords_radii );
        geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType > interpolator( radial_profile );
        interpolator.interpolate( k.grid_data() );
        break;
    }
    }
}

// ---------------------------------------------------------------------------
// Run a single FGMRES solve and return iteration count
// ---------------------------------------------------------------------------

template < typename PrecT >
int run_fgmres_solve(
    Stokes&                                          K,
    PrecT&                                           prec,
    VectorQ1IsoQ2Q1< ScalarType >&                   u,
    const VectorQ1IsoQ2Q1< ScalarType >&             f,
    std::vector< VectorQ1IsoQ2Q1< ScalarType > >&    tmp_fgmres,
    int                                              max_iters,
    const std::string&                               label )
{
    linalg::solvers::FGMRESOptions< ScalarType > opts;
    opts.restart                     = max_iters;
    opts.max_iterations              = max_iters;
    opts.relative_residual_tolerance = 1e-10;

    auto solver_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, PrecT > fgmres( tmp_fgmres, opts, solver_table, prec );

    linalg::assign( u, 0 );
    linalg::solvers::solve( fgmres, K, u, f );

    const auto rows = solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows();
    const int  iterations = static_cast< int >( rows.size() );

    util::logroot << "  " << label << ": " << iterations << " iterations\n";

    return iterations;
}

// ---------------------------------------------------------------------------
// Main test function
// ---------------------------------------------------------------------------

void test(
    ViscosityMode                          viscosity_mode,
    int                                    min_level,
    int                                    max_level,
    const std::shared_ptr< util::Table >&  results_table )
{
    const std::string viscosity_label = viscosity_mode_label( viscosity_mode );

    util::logroot << "\n========================================\n";
    util::logroot << "Viscosity: " << viscosity_label << "\n";
    util::logroot << "Levels: " << min_level << " - " << max_level << "\n";
    util::logroot << "========================================\n";

    // ---- Domain setup ----

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    const ScalarType r_min = 0.5;
    const ScalarType r_max = 1.0;

    util::logroot << "Allocating domains ...\n";
    for ( int level = min_level; level <= max_level; level++ )
    {
        const int idx = level - min_level;
        domains.push_back( DistributedDomain::create_uniform( level, level, r_min, r_max, 0, 0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    // ---- Vectors ----

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    for ( const auto& name : { "u", "f", "solution", "tmp_0", "tmp_1" } )
    {
        stok_vecs[std::string( name )] = VectorQ1IsoQ2Q1< ScalarType >(
            name,
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level] );
    }

    auto& u = stok_vecs["u"];
    auto& f = stok_vecs["f"];

    std::vector< VectorQ1Vec< ScalarType > > tmp_mg;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_r;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_e;

    for ( size_t level = 0; level < num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    const auto num_dofs_velocity =
        3 * kernels::common::count_masked< long >( mask_data[num_levels - 1], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( mask_data[num_levels - 2], grid::NodeOwnershipFlag::OWNED );

    // ---- Boundary conditions ----

    BoundaryConditions bcs         = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } };
    BoundaryConditions bcs_neumann = { { CMB, NEUMANN },   { SURFACE, NEUMANN } };

    // ---- Coefficient field (viscosity) ----

    VectorQ1Scalar< ScalarType > k( "k", domains[velocity_level], mask_data[velocity_level] );
    interpolate_viscosity(
        viscosity_mode, domains[velocity_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], k );

    // ---- Stokes operators ----

    util::logroot << "Setting operators ...\n";

    Stokes K_op(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], k.grid_data(), bcs, false );

    Stokes K_neumann(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], k.grid_data(), bcs_neumann, false );

    Stokes K_neumann_diag(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], k.grid_data(), bcs_neumann, true );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // ---- Multigrid hierarchy for velocity block ----

    util::logroot << "MG hierarchy ...\n";

    std::vector< Viscous >      A_diag;
    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;
    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;

    for ( size_t level = 0; level < num_levels; level++ )
    {
        VectorQ1Scalar< ScalarType > k_c( "k_c", domains[level], mask_data[level] );
        interpolate_viscosity(
            viscosity_mode, domains[level],
            coords_shell[level], coords_radii[level],
            boundary_mask_data[level], k_c );

        A_diag.emplace_back(
            domains[level], coords_shell[level], coords_radii[level],
            boundary_mask_data[level], k_c.grid_data(), bcs, true );

        if ( level < num_levels - 1 )
        {
            A_c.emplace_back(
                domains[level], coords_shell[level], coords_radii[level],
                boundary_mask_data[level], k_c.grid_data(), bcs, false );
            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // ---- Smoother setup with power iteration ----

    std::vector< Smoother > smoothers;
    for ( size_t level = 0; level < num_levels; level++ )
    {
        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > tmp( "inv_diag_tmp", domains[level], mask_data[level] );
        linalg::assign( tmp, 1.0 );

        if ( level == num_levels - 1 )
        {
            K_op.block_11().set_diagonal( true );
            linalg::apply( K_op.block_11(), tmp, inverse_diagonals.back() );
            K_op.block_11().set_diagonal( false );
        }
        else
        {
            A_c[level].set_diagonal( true );
            linalg::apply( A_c[level], tmp, inverse_diagonals.back() );
            A_c[level].set_diagonal( false );
        }

        linalg::invert_entries( inverse_diagonals.back() );

        constexpr int smoother_prepost = 3;
        VectorQ1Vec< ScalarType > tmp_pi_0( "tmp_pi_0", domains[level], mask_data[level] );
        VectorQ1Vec< ScalarType > tmp_pi_1( "tmp_pi_1", domains[level], mask_data[level] );

        double max_ev = 0.0;
        if ( level == num_levels - 1 )
        {
            DiagonallyScaledOperator< Viscous > inv_diag_A( K_op.block_11(), inverse_diagonals[level] );
            max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
        }
        else
        {
            DiagonallyScaledOperator< Viscous > inv_diag_A( A_c[level], inverse_diagonals[level] );
            max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
        }

        const auto omega_opt = 2.0 / ( 1.3 * max_ev );
        smoothers.emplace_back( inverse_diagonals[level], smoother_prepost, tmp_mg[level], omega_opt );
    }

    // ---- Coarse grid solver ----

    std::vector< VectorQ1Vec< ScalarType > > coarse_grid_tmps;
    for ( int i = 0; i < 4; i++ )
        coarse_grid_tmps.emplace_back( "tmp_cg", domains[0], mask_data[0] );

    auto cg_table = std::make_shared< util::Table >();
    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-6, 1e-16 }, cg_table, coarse_grid_tmps );

    constexpr int num_mg_cycles = 1;
    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, num_mg_cycles, 1e-8 );

    // ---- RHS assembly ----
    // We compute f = K * u_exact (applying the Stokes operator to a smooth function).
    // This gives a consistent RHS for any viscosity profile.

    util::logroot << "Assembling RHS ...\n";

    Kokkos::parallel_for(
        "solution interpolation (velocity)",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["solution"].block_1().grid_data(), boundary_mask_data[velocity_level], false ) );

    Kokkos::parallel_for(
        "solution interpolation (pressure)",
        local_domain_md_range_policy_nodes( domains[pressure_level] ),
        SolutionPressureInterpolator(
            coords_shell[pressure_level], coords_radii[pressure_level],
            stok_vecs["solution"].block_2().grid_data(), boundary_mask_data[pressure_level], false ) );

    // f = K_neumann * u_exact (RHS from applying operator to known solution)
    linalg::apply( K_neumann, stok_vecs["solution"], f );

    // Apply Dirichlet BCs
    Kokkos::parallel_for(
        "boundary interpolation (velocity)",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["tmp_0"].block_1().grid_data(), boundary_mask_data[velocity_level], true ) );

    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann, K_neumann_diag,
        stok_vecs["tmp_0"], stok_vecs["tmp_1"],
        f, boundary_mask_data[velocity_level], BOUNDARY );

    // ---- FGMRES temporaries (shared across all solves) ----

    const int max_iters = 200;
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < 2 * max_iters + 4; ++i )
    {
        tmp_fgmres.emplace_back(
            "tmp_fgmres_" + std::to_string( i ),
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level] );
    }

    // ---- Pressure Schur complement operators ----

    // Inverse-viscosity-weighted pressure mass (lumped)
    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[pressure_level], mask_data[pressure_level] );
    {
        // Interpolate viscosity on pressure level and invert
        interpolate_viscosity(
            viscosity_mode, domains[pressure_level],
            coords_shell[pressure_level], coords_radii[pressure_level],
            boundary_mask_data[pressure_level], k_pm );
        linalg::invert_entries( k_pm );
    }

    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level],
        k_pm.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    VectorQ1Scalar< ScalarType > lumped_diag_pmass( "lumped_diag_pmass", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > tmp( "tmp_pmass", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( tmp, 1.0 );
        linalg::apply( pmass, tmp, lumped_diag_pmass );
    }

    // Pressure Laplacian (on pressure grid, with boundary treatment)
    PressureLaplace pressure_laplace(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level],
        true, false );

    VectorQ1Scalar< ScalarType > diag_pressure_laplace( "diag_pL", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > tmp( "tmp_pL", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( tmp, 1.0 );
        pressure_laplace.set_diagonal( true );
        linalg::apply( pressure_laplace, tmp, diag_pressure_laplace );
        pressure_laplace.set_diagonal( false );
    }

    // Viscosity-weighted pressure Laplacian: div(1/mu * grad p)
    PressureDivKGrad pressure_divkgrad(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level],
        boundary_mask_data[pressure_level], k_pm.grid_data(), true, false );

    VectorQ1Scalar< ScalarType > diag_pressure_divkgrad( "diag_pDKG", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > tmp( "tmp_pDKG", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( tmp, 1.0 );
        pressure_divkgrad.set_diagonal( true );
        linalg::apply( pressure_divkgrad, tmp, diag_pressure_divkgrad );
        pressure_divkgrad.set_diagonal( false );
    }

    // Viscosity-weighted pressure Laplacian with Neumann BCs: div(1/mu * grad p)
    PressureDivKGrad pressure_divkgrad_neumann(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level],
        boundary_mask_data[pressure_level], k_pm.grid_data(), false, false );

    VectorQ1Scalar< ScalarType > diag_pressure_divkgrad_neumann( "diag_pDKG_N", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > tmp( "tmp_pDKG_N", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( tmp, 1.0 );
        pressure_divkgrad_neumann.set_diagonal( true );
        linalg::apply( pressure_divkgrad_neumann, tmp, diag_pressure_divkgrad_neumann );
        pressure_divkgrad_neumann.set_diagonal( false );
    }

    // Temporary block vector for triangular preconditioners
    VectorQ1IsoQ2Q1< ScalarType > tri_prec_tmp(
        "tri_prec_tmp",
        domains[velocity_level], domains[pressure_level],
        mask_data[velocity_level], mask_data[pressure_level] );

    util::logroot << "\nRunning Schur complement comparisons ...\n";
    util::logroot << "  velocity dofs: " << num_dofs_velocity << "\n";
    util::logroot << "  pressure dofs: " << num_dofs_pressure << "\n\n";

    // ====================================================================
    // 1. Block-triangular: MG + lumped pressure mass (1/mu weighted)
    // ====================================================================
    {
        using PrecSchur  = DiagonalSolver< PressureMass >;
        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

        VectorQ1Scalar< ScalarType > diag_copy( "diag_copy_pmass2", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, lumped_diag_pmass );

        PrecSchur  prec_22( diag_copy );
        PrecStokes prec( K_op.block_11(), pmass, K_op.block_12(), tri_prec_tmp, prec_11, prec_22 );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, "BlockTri + PressureMass(1/mu)" );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", std::string( "BlockTri_PressureMass" ) },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    }

    // ====================================================================
    // 2. Block-triangular: MG + pressure Laplacian diagonal
    // ====================================================================
    {
        using PrecSchur  = DiagonalSolver< PressureLaplace >;
        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureLaplace, Gradient, PrecVisc, PrecSchur >;

        VectorQ1Scalar< ScalarType > diag_copy( "diag_copy_pL2", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, diag_pressure_laplace );

        PrecSchur  prec_22( diag_copy );
        PrecStokes prec( K_op.block_11(), pressure_laplace, K_op.block_12(), tri_prec_tmp, prec_11, prec_22 );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, "BlockTri + PressureLapl(diag)" );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", std::string( "BlockTri_PressureLaplace" ) },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    }

    // ====================================================================
    // 3. Block-triangular: MG + pressure Laplacian PCG (few iterations)
    // ====================================================================
    {
        using PrecSchurInner = DiagonalSolver< PressureLaplace >;
        using PrecSchur      = linalg::solvers::PCG< PressureLaplace, PrecSchurInner >;
        using PrecStokes     = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureLaplace, Gradient, PrecVisc, PrecSchur >;

        VectorQ1Scalar< ScalarType > diag_copy( "diag_copy_pL4", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, diag_pressure_laplace );

        PrecSchurInner prec_inner( diag_copy );

        std::vector< VectorQ1Scalar< ScalarType > > pcg_tmps;
        for ( int i = 0; i < 4; i++ )
            pcg_tmps.emplace_back( "pcg_tmp_" + std::to_string( i ), domains[pressure_level], mask_data[pressure_level] );

        auto pcg_table = std::make_shared< util::Table >();
        PrecSchur prec_22(
            linalg::solvers::IterativeSolverParameters{ 10, 1e-2, 1e-16 },
            pcg_table, pcg_tmps, prec_inner );

        PrecStokes prec( K_op.block_11(), pressure_laplace, K_op.block_12(), tri_prec_tmp, prec_11, prec_22 );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, "BlockTri + PressureLapl(PCG-10)" );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", std::string( "BlockTri_PressureLaplacePCG" ) },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    }

    // ====================================================================
    // 4. Block-triangular: MG + div(1/mu grad) diagonal
    // ====================================================================
    {
        using PrecSchur  = DiagonalSolver< PressureDivKGrad >;
        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureDivKGrad, Gradient, PrecVisc, PrecSchur >;

        VectorQ1Scalar< ScalarType > diag_copy( "diag_copy_dkg2", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, diag_pressure_divkgrad );

        PrecSchur  prec_22( diag_copy );
        PrecStokes prec( K_op.block_11(), pressure_divkgrad, K_op.block_12(), tri_prec_tmp, prec_11, prec_22 );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, "BlockTri + DivKGrad(1/mu,diag)" );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", std::string( "BlockTri_DivKGrad_diag" ) },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    }

    // ====================================================================
    // 5. Block-triangular: MG + div(1/mu grad) PCG (10 iterations)
    // ====================================================================
    {
        using PrecSchurInner = DiagonalSolver< PressureDivKGrad >;
        using PrecSchur      = linalg::solvers::PCG< PressureDivKGrad, PrecSchurInner >;
        using PrecStokes     = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureDivKGrad, Gradient, PrecVisc, PrecSchur >;

        VectorQ1Scalar< ScalarType > diag_copy( "diag_copy_dkg_pcg", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, diag_pressure_divkgrad );

        PrecSchurInner prec_inner( diag_copy );

        std::vector< VectorQ1Scalar< ScalarType > > pcg_tmps;
        for ( int i = 0; i < 4; i++ )
            pcg_tmps.emplace_back( "pcg_dkg_tmp_" + std::to_string( i ), domains[pressure_level], mask_data[pressure_level] );

        auto pcg_table = std::make_shared< util::Table >();
        PrecSchur prec_22(
            linalg::solvers::IterativeSolverParameters{ 10, 1e-2, 1e-16 },
            pcg_table, pcg_tmps, prec_inner );

        PrecStokes prec( K_op.block_11(), pressure_divkgrad, K_op.block_12(), tri_prec_tmp, prec_11, prec_22 );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, "BlockTri + DivKGrad(1/mu,PCG-10)" );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", std::string( "BlockTri_DivKGrad_PCG" ) },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    }

    // ====================================================================
    // 6. Block-triangular: MG + div(1/mu grad) Neumann, diagonal
    // ====================================================================
    {
        using PrecSchur  = DiagonalSolver< PressureDivKGrad >;
        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureDivKGrad, Gradient, PrecVisc, PrecSchur >;

        VectorQ1Scalar< ScalarType > diag_copy( "diag_copy_dkg_n", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, diag_pressure_divkgrad_neumann );

        PrecSchur  prec_22( diag_copy );
        PrecStokes prec( K_op.block_11(), pressure_divkgrad_neumann, K_op.block_12(), tri_prec_tmp, prec_11, prec_22 );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, "BlockTri + DivKGrad(1/mu,N,diag)" );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", std::string( "BlockTri_DivKGrad_N_diag" ) },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    }

    // ====================================================================
    // 7. Block-triangular: MG + div(1/mu grad) Neumann, PCG-10
    // ====================================================================
    {
        using PrecSchurInner = DiagonalSolver< PressureDivKGrad >;
        using PrecSchur      = linalg::solvers::PCG< PressureDivKGrad, PrecSchurInner >;
        using PrecStokes     = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureDivKGrad, Gradient, PrecVisc, PrecSchur >;

        VectorQ1Scalar< ScalarType > diag_copy( "diag_copy_dkg_n_pcg", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, diag_pressure_divkgrad_neumann );

        PrecSchurInner prec_inner( diag_copy );

        std::vector< VectorQ1Scalar< ScalarType > > pcg_tmps;
        for ( int i = 0; i < 4; i++ )
            pcg_tmps.emplace_back( "pcg_dkg_n_tmp_" + std::to_string( i ), domains[pressure_level], mask_data[pressure_level] );

        auto pcg_table = std::make_shared< util::Table >();
        PrecSchur prec_22(
            linalg::solvers::IterativeSolverParameters{ 10, 1e-2, 1e-16 },
            pcg_table, pcg_tmps, prec_inner );

        PrecStokes prec( K_op.block_11(), pressure_divkgrad_neumann, K_op.block_12(), tri_prec_tmp, prec_11, prec_22 );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, "BlockTri + DivKGrad(1/mu,N,PCG-10)" );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", std::string( "BlockTri_DivKGrad_N_PCG" ) },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    const int min_level = 2;
    const int max_level = 4;

    auto results = std::make_shared< util::Table >();

    // Constant viscosity
    test( ViscosityMode::Constant, min_level, max_level, results );

    // Radial profile: Lin et al. 2022
    test( ViscosityMode::LinEtAl2022, min_level, max_level, results );

    // Radial profile: Stotz et al. 2017
    test( ViscosityMode::StotzEtAl2017, min_level, max_level, results );

    // Print summary
    util::logroot << "\n\n";
    util::logroot << "======================================================\n";
    util::logroot << "  Schur Complement Comparison Summary\n";
    util::logroot << "======================================================\n\n";

    results->select_columns( { "viscosity", "preconditioner", "iterations", "dofs_vel", "dofs_pre" } )
        .print_pretty();

    return 0;
}
