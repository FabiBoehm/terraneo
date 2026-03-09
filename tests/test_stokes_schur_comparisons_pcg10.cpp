
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
#include "fe/wedge/operators/shell/vector_kmass.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp"

#include "linalg/composed_pressure_operator.hpp"
#include "linalg/solvers/bfbt_preconditioner.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/gca/gca.hpp"
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
#include <chrono>
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

using ViscousKMass = fe::wedge::operators::shell::VectorKMass< ScalarType >;

using PressureMass    = fe::wedge::operators::shell::KMass< ScalarType >;
using PressureLaplace = fe::wedge::operators::shell::LaplaceSimple< ScalarType >;
using PressureDKG     = fe::wedge::operators::shell::DivKGrad< ScalarType >;

using Smoother         = linalg::solvers::Jacobi< Viscous >;
using CoarseGridSolver = linalg::solvers::PCG< Viscous >;
using PrecVisc         = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;

// Scalar (pressure) MG types for DivKGrad inner solves
using ScalarProlongation    = fe::wedge::operators::shell::ProlongationConstant< ScalarType >;
using ScalarRestriction     = fe::wedge::operators::shell::RestrictionConstant< ScalarType >;
using DKGSmoother           = linalg::solvers::Chebyshev< PressureDKG >;
using DKGCoarseGridSolver   = linalg::solvers::PCG< PressureDKG >;
using PrecDKG               = linalg::solvers::Multigrid< PressureDKG, ScalarProlongation, ScalarRestriction, DKGSmoother, DKGCoarseGridSolver >;

// BFBT composed operator types
using BDinvBT      = linalg::ComposedBDinvBT< Gradient, Divergence >;
using BDinvADinvBT = linalg::ComposedBDinvADinvBT< Gradient, Divergence, Viscous >;

// ---------------------------------------------------------------------------
// Adapter: wraps a solver for operator A as a preconditioner for operator B
// (same vector types, different operator types — ignores passed operator)
// ---------------------------------------------------------------------------

template < linalg::OperatorLike TargetOperatorT, linalg::solvers::SolverLike InnerSolverT >
class SolverAdapter
{
  public:
    using OperatorType      = TargetOperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType     = linalg::DstOf< OperatorType >;

    using InnerOperatorType = typename InnerSolverT::OperatorType;

    static_assert( std::is_same_v< SolutionVectorType, linalg::SrcOf< InnerOperatorType > > );
    static_assert( std::is_same_v< RHSVectorType, linalg::DstOf< InnerOperatorType > > );

    SolverAdapter( InnerSolverT& inner_solver, InnerOperatorType& inner_operator )
    : inner_solver_( inner_solver )
    , inner_operator_( inner_operator )
    {}

    void solve_impl( OperatorType& /* ignored */, SolutionVectorType& x, const RHSVectorType& b )
    {
        linalg::solvers::solve( inner_solver_, inner_operator_, x, b );
    }

  private:
    InnerSolverT&     inner_solver_;
    InnerOperatorType& inner_operator_;
};

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
    Kokkos::fence();
    MPI_Barrier( MPI_COMM_WORLD );
    auto t_start = std::chrono::high_resolution_clock::now();

    linalg::solvers::solve( fgmres, K, u, f );

    Kokkos::fence();
    MPI_Barrier( MPI_COMM_WORLD );
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration< double >( t_end - t_start ).count();

    const auto rows = solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows();
    const int  iterations = static_cast< int >( rows.size() );

    util::logroot << "  " << label << ": " << iterations << " iterations, " << elapsed_s << " s\n";

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

    // Temporary block vector for triangular preconditioners
    VectorQ1IsoQ2Q1< ScalarType > tri_prec_tmp(
        "tri_prec_tmp",
        domains[velocity_level], domains[pressure_level],
        mask_data[velocity_level], mask_data[pressure_level] );

    util::logroot << "\nRunning Schur complement comparisons ...\n";
    util::logroot << "  velocity dofs: " << num_dofs_velocity << "\n";
    util::logroot << "  pressure dofs: " << num_dofs_pressure << "\n\n";

    // Weighted pressure mass (block-triangular)
    {
        using PrecSchur  = DiagonalSolver< PressureMass >;
        PrecSchur prec_schur( lumped_diag_pmass );

        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

        PrecStokes prec( K_op.block_11(), pmass, K_op.block_12(), tri_prec_tmp, prec_11, prec_schur );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, "BlockTri + M_p(1/mu)" );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", "BlockTri_Mp_inv_mu" },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    }

    // ====================================================================
    // BFBT preconditioner variants
    //
    // S_BFBT^{-1} = (B D^{-1} B^T)^{-1} (B D^{-1} A D^{-1} B^T) (B D^{-1} B^T)^{-1}
    //
    // Three choices for D:
    //   (a) diag(M_u)         — lumped velocity mass
    //   (b) diag(A)           — diagonal of the viscous operator
    //   (c) diag(M_u(mu))     — lumped viscosity-weighted velocity mass
    // ====================================================================

    // ---- Velocity-space inverse diagonals for BFBT ----

    // (a) inv_diag(A) — reuse the smoother diagonal from the finest level
    VectorQ1Vec< ScalarType > inv_diag_A( "inv_diag_A", domains[velocity_level], mask_data[velocity_level] );
    linalg::assign( inv_diag_A, inverse_diagonals[velocity_level] );

    // (b) inv_diag(M_u(mu)) — lumped diagonal
    VectorQ1Vec< ScalarType > inv_diag_MuK( "inv_diag_MuK", domains[velocity_level], mask_data[velocity_level] );
    {
        ViscousKMass MK_diag(
            domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
            k.grid_data(), true );
        VectorQ1Vec< ScalarType > ones( "ones_vel2", domains[velocity_level], mask_data[velocity_level] );
        linalg::assign( ones, 1.0 );
        linalg::apply( MK_diag, ones, inv_diag_MuK );
        linalg::invert_entries( inv_diag_MuK );
    }

    // (c) inv_diag(M_u(sqrt(mu))) — lumped diagonal with sqrt viscosity
    VectorQ1Scalar< ScalarType > k_sqrt( "k_sqrt", domains[velocity_level], mask_data[velocity_level] );
    linalg::assign( k_sqrt, k );
    {
        auto data = k_sqrt.grid_data();
        Kokkos::parallel_for( "sqrt_visc",
            Kokkos::RangePolicy< Kokkos::DefaultExecutionSpace >( 0, data.size() ),
            KOKKOS_LAMBDA( const int i ) {
                data.data()[i] = Kokkos::sqrt( data.data()[i] );
            } );
    }

    VectorQ1Vec< ScalarType > inv_diag_MuK_sqrt( "inv_diag_MuK_sqrt", domains[velocity_level], mask_data[velocity_level] );
    {
        ViscousKMass MK_sqrt(
            domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
            k_sqrt.grid_data(), true );
        VectorQ1Vec< ScalarType > ones( "ones_vel3", domains[velocity_level], mask_data[velocity_level] );
        linalg::assign( ones, 1.0 );
        linalg::apply( MK_sqrt, ones, inv_diag_MuK_sqrt );
        linalg::invert_entries( inv_diag_MuK_sqrt );
    }

    // ---- DivKGrad(1/mu) MG hierarchy for inner BFBT solves ----

    // Build DivKGrad operators and Chebyshev smoothers on each pressure level
    // Pressure levels: indices 0 to pressure_level in the domains array
    const auto num_pressure_levels = pressure_level + 1; // e.g., levels 2,3 -> indices 0,1

    std::vector< PressureDKG >                    dkg_ops;
    std::vector< PressureDKG >                    dkg_ops_diag;
    std::vector< ScalarProlongation >             dkg_P;
    std::vector< ScalarRestriction >              dkg_R;
    std::vector< VectorQ1Scalar< ScalarType > >   dkg_inv_diags;
    std::vector< VectorQ1Scalar< ScalarType > >   dkg_diags;
    std::vector< VectorQ1Scalar< ScalarType > >   dkg_tmp;
    std::vector< VectorQ1Scalar< ScalarType > >   dkg_tmp_r;
    std::vector< VectorQ1Scalar< ScalarType > >   dkg_tmp_e;

    for ( size_t level = 0; level <= pressure_level; level++ )
    {
        // Inverse viscosity on this level
        VectorQ1Scalar< ScalarType > k_inv_lev( "k_inv_dkg", domains[level], mask_data[level] );
        interpolate_viscosity(
            viscosity_mode, domains[level],
            coords_shell[level], coords_radii[level],
            boundary_mask_data[level], k_inv_lev );
        linalg::invert_entries( k_inv_lev );

        // DivKGrad operator (Dirichlet BCs, non-diagonal)
        dkg_ops.emplace_back(
            domains[level], coords_shell[level], coords_radii[level],
            boundary_mask_data[level], k_inv_lev.grid_data(), true, false );

        // Diagonal version for smoother setup
        dkg_ops_diag.emplace_back(
            domains[level], coords_shell[level], coords_radii[level],
            boundary_mask_data[level], k_inv_lev.grid_data(), true, true );

        // Diagonal and inverse diagonal
        dkg_diags.emplace_back( "dkg_diag_" + std::to_string( level ), domains[level], mask_data[level] );
        dkg_inv_diags.emplace_back( "dkg_inv_diag_" + std::to_string( level ), domains[level], mask_data[level] );
        {
            VectorQ1Scalar< ScalarType > ones( "ones_dkg", domains[level], mask_data[level] );
            linalg::assign( ones, 1.0 );
            linalg::apply( dkg_ops_diag.back(), ones, dkg_diags.back() );
            linalg::assign( dkg_inv_diags.back(), dkg_diags.back() );
            linalg::invert_entries( dkg_inv_diags.back() );
        }

        dkg_tmp.emplace_back( "dkg_tmp_" + std::to_string( level ), domains[level], mask_data[level] );

        if ( level < pressure_level )
        {
            // Coarse-level operators and grid transfer
            dkg_P.emplace_back( linalg::OperatorApplyMode::Add );
            dkg_R.emplace_back( domains[level] );
            dkg_tmp_r.emplace_back( "dkg_tmp_r_" + std::to_string( level ), domains[level], mask_data[level] );
            dkg_tmp_e.emplace_back( "dkg_tmp_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    // GCA assembly for DKG MG hierarchy
    {
        using TwoGridGCA = linalg::solvers::TwoGridGCA< ScalarType, PressureDKG >;

        VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", domains[0], mask_data[0] );
        linalg::assign( GCAElements, 1 );

        // Set stored matrix mode on all coarse operators
        for ( size_t level = 0; level < pressure_level; level++ )
        {
            dkg_ops[level].set_stored_matrix_mode(
                linalg::OperatorStoredMatrixMode::Full, level, GCAElements.grid_data() );
        }

        // Assemble GCA from fine to coarse
        for ( int level = pressure_level - 1; level >= 0; level-- )
        {
            if ( level == static_cast< int >( pressure_level ) - 1 )
            {
                TwoGridGCA( dkg_ops[pressure_level], dkg_ops[level], level, GCAElements.grid_data() );
            }
            else
            {
                TwoGridGCA( dkg_ops[level + 1], dkg_ops[level], level, GCAElements.grid_data() );
            }
        }
    }

    // Chebyshev smoothers for each level
    std::vector< DKGSmoother > dkg_smoothers;
    for ( size_t level = 0; level <= pressure_level; level++ )
    {
        std::vector< VectorQ1Scalar< ScalarType > > cheb_tmps;
        cheb_tmps.emplace_back( "cheb_tmp_0_" + std::to_string( level ), domains[level], mask_data[level] );
        cheb_tmps.emplace_back( "cheb_tmp_1_" + std::to_string( level ), domains[level], mask_data[level] );

        constexpr int cheb_order      = 3;
        constexpr int cheb_iterations = 3;
        dkg_smoothers.emplace_back( cheb_order, dkg_inv_diags[level], cheb_tmps, cheb_iterations );
    }

    // Coarse grid solver for DivKGrad
    std::vector< VectorQ1Scalar< ScalarType > > dkg_cg_tmps;
    for ( int i = 0; i < 4; i++ )
        dkg_cg_tmps.emplace_back( "dkg_cg_tmp_" + std::to_string( i ), domains[0], mask_data[0] );

    auto dkg_cg_table = std::make_shared< util::Table >();
    DKGCoarseGridSolver dkg_coarse_solver(
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-6, 1e-16 }, dkg_cg_table, dkg_cg_tmps );

    // Coarse-level DivKGrad operators (all except finest pressure level)
    std::vector< PressureDKG > dkg_A_c( dkg_ops.begin(), dkg_ops.begin() + pressure_level );

    // MG with 1 V-cycle (1/mu weighting)
    PrecDKG prec_dkg_1v(
        dkg_P, dkg_R, dkg_A_c, dkg_tmp_r, dkg_tmp_e, dkg_tmp,
        dkg_smoothers, dkg_smoothers, dkg_coarse_solver, 1, 1e-8 );

    // ---- BFBT temporaries (velocity-space) ----

    VectorQ1Vec< ScalarType > bfbt_tmp_vel_1( "bfbt_tmp_vel_1", domains[velocity_level], mask_data[velocity_level] );
    VectorQ1Vec< ScalarType > bfbt_tmp_vel_2( "bfbt_tmp_vel_2", domains[velocity_level], mask_data[velocity_level] );
    VectorQ1Vec< ScalarType > bfbt_tmp_vel_3( "bfbt_tmp_vel_3", domains[velocity_level], mask_data[velocity_level] );
    VectorQ1Vec< ScalarType > bfbt_tmp_vel_4( "bfbt_tmp_vel_4", domains[velocity_level], mask_data[velocity_level] );

    // ---- BFBT temporaries (pressure-space) ----

    VectorQ1Scalar< ScalarType > bfbt_t1( "bfbt_t1", domains[pressure_level], mask_data[pressure_level] );
    VectorQ1Scalar< ScalarType > bfbt_t2( "bfbt_t2", domains[pressure_level], mask_data[pressure_level] );

    std::vector< VectorQ1Scalar< ScalarType > > bfbt_pcg_tmps;
    for ( int i = 0; i < 4; i++ )
        bfbt_pcg_tmps.emplace_back( "bfbt_pcg_tmp_" + std::to_string( i ), domains[pressure_level], mask_data[pressure_level] );

    // Helper lambda: BFBT with PCG + diagonal inner preconditioner + relaxation
    auto run_bfbt_pcg = [&]( VectorQ1Vec< ScalarType >&    inv_diag,
                             VectorQ1Scalar< ScalarType >& inner_prec_diag,
                             int                           inner_iters,
                             double                        omega,
                             const std::string& label,
                             const std::string& table_label )
    {
        BDinvBT bdinvbt( K_neumann.block_12(), K_neumann.block_21(), inv_diag, bfbt_tmp_vel_1 );
        BDinvADinvBT bdinvadinvbt( K_neumann.block_12(), K_neumann.block_21(), K_neumann.block_11(),
                                   inv_diag, bfbt_tmp_vel_3, bfbt_tmp_vel_4 );

        VectorQ1Scalar< ScalarType > diag_copy( "diag_inner_bfbt", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, inner_prec_diag );

        using InnerPrec   = linalg::solvers::DiagonalSolver< BDinvBT >;
        using InnerSolver = linalg::solvers::PCG< BDinvBT, InnerPrec >;

        InnerPrec inner_prec( diag_copy );

        auto inner_pcg_table = std::make_shared< util::Table >();
        InnerSolver inner_solver(
            linalg::solvers::IterativeSolverParameters{ inner_iters, 1e-6, 1e-16 },
            inner_pcg_table, bfbt_pcg_tmps, inner_prec );

        using BFBTPrec = linalg::solvers::BFBTPreconditioner< BDinvBT, BDinvBT, BDinvADinvBT, InnerSolver >;
        BFBTPrec bfbt_prec( bdinvbt, bdinvadinvbt, inner_solver, bfbt_t1, bfbt_t2, omega );

        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, BDinvBT, Gradient, PrecVisc, BFBTPrec >;

        PrecStokes prec( K_op.block_11(), bdinvbt, K_op.block_12(), tri_prec_tmp, prec_11, bfbt_prec );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, label );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", table_label },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    };

    // Helper lambda: BFBT with naked MG V-cycles as inner solver (no PCG wrapper)
    using MGAdapter = SolverAdapter< BDinvBT, PrecDKG >;

    auto run_bfbt_naked_mg = [&]( VectorQ1Vec< ScalarType >& inv_diag,
                                  PrecDKG&                   mg,
                                  double                     omega,
                                  const std::string& label,
                                  const std::string& table_label )
    {
        BDinvBT bdinvbt( K_neumann.block_12(), K_neumann.block_21(), inv_diag, bfbt_tmp_vel_1 );
        BDinvADinvBT bdinvadinvbt( K_neumann.block_12(), K_neumann.block_21(), K_neumann.block_11(),
                                   inv_diag, bfbt_tmp_vel_3, bfbt_tmp_vel_4 );

        MGAdapter mg_adapter( mg, dkg_ops[pressure_level] );

        using BFBTPrec = linalg::solvers::BFBTPreconditioner< BDinvBT, BDinvBT, BDinvADinvBT, MGAdapter >;
        BFBTPrec bfbt_prec( bdinvbt, bdinvadinvbt, mg_adapter, bfbt_t1, bfbt_t2, omega );

        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, BDinvBT, Gradient, PrecVisc, BFBTPrec >;

        PrecStokes prec( K_op.block_11(), bdinvbt, K_op.block_12(), tri_prec_tmp, prec_11, bfbt_prec );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, label );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", table_label },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    };

    // Helper lambda: BFBT with PCG + DivKGrad MG V-cycle inner preconditioner
    using InnerSolverMG = linalg::solvers::PCG< BDinvBT, MGAdapter >;

    auto run_bfbt_mg_pcg = [&]( VectorQ1Vec< ScalarType >& inv_diag,
                                PrecDKG&                   mg,
                                PressureDKG&               mg_fine_op,
                                int                        inner_iters,
                                double                     omega,
                                const std::string& label,
                                const std::string& table_label )
    {
        BDinvBT bdinvbt( K_neumann.block_12(), K_neumann.block_21(), inv_diag, bfbt_tmp_vel_1 );
        BDinvADinvBT bdinvadinvbt( K_neumann.block_12(), K_neumann.block_21(), K_neumann.block_11(),
                                   inv_diag, bfbt_tmp_vel_3, bfbt_tmp_vel_4 );

        MGAdapter mg_adapter( mg, mg_fine_op );

        auto inner_pcg_table = std::make_shared< util::Table >();
        InnerSolverMG inner_solver(
            linalg::solvers::IterativeSolverParameters{ inner_iters, 1e-6, 1e-16 },
            inner_pcg_table, bfbt_pcg_tmps, mg_adapter );

        using BFBTPrec = linalg::solvers::BFBTPreconditioner< BDinvBT, BDinvBT, BDinvADinvBT, InnerSolverMG >;
        BFBTPrec bfbt_prec( bdinvbt, bdinvadinvbt, inner_solver, bfbt_t1, bfbt_t2, omega );

        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, BDinvBT, Gradient, PrecVisc, BFBTPrec >;

        PrecStokes prec( K_op.block_11(), bdinvbt, K_op.block_12(), tri_prec_tmp, prec_11, bfbt_prec );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, label );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", table_label },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    };

    // Helper lambda: BFBT with DKG as substitute subsystem operator + diagonal inner preconditioner
    using DKGDiagPrec     = DiagonalSolver< PressureDKG >;
    using DKGInnerSolver  = linalg::solvers::PCG< PressureDKG, DKGDiagPrec >;
    using DKGSolverAdapter = SolverAdapter< BDinvBT, DKGInnerSolver >;

    auto run_bfbt_dkg_pcg = [&]( VectorQ1Vec< ScalarType >& inv_diag,
                                 int                        inner_iters,
                                 double                     omega,
                                 const std::string& label,
                                 const std::string& table_label )
    {
        BDinvBT bdinvbt( K_neumann.block_12(), K_neumann.block_21(), inv_diag, bfbt_tmp_vel_1 );
        BDinvADinvBT bdinvadinvbt( K_neumann.block_12(), K_neumann.block_21(), K_neumann.block_11(),
                                   inv_diag, bfbt_tmp_vel_3, bfbt_tmp_vel_4 );

        VectorQ1Scalar< ScalarType > diag_copy( "diag_inner_dkg", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( diag_copy, dkg_diags[pressure_level] );

        DKGDiagPrec dkg_diag_prec( diag_copy );

        auto inner_pcg_table = std::make_shared< util::Table >();
        DKGInnerSolver dkg_inner_solver(
            linalg::solvers::IterativeSolverParameters{ inner_iters, 1e-6, 1e-16 },
            inner_pcg_table, bfbt_pcg_tmps, dkg_diag_prec );

        DKGSolverAdapter dkg_adapter( dkg_inner_solver, dkg_ops[pressure_level] );

        using BFBTPrec = linalg::solvers::BFBTPreconditioner< BDinvBT, BDinvBT, BDinvADinvBT, DKGSolverAdapter >;
        BFBTPrec bfbt_prec( bdinvbt, bdinvadinvbt, dkg_adapter, bfbt_t1, bfbt_t2, omega );

        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, BDinvBT, Gradient, PrecVisc, BFBTPrec >;

        PrecStokes prec( K_op.block_11(), bdinvbt, K_op.block_12(), tri_prec_tmp, prec_11, bfbt_prec );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, label );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", table_label },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    };

    // Helper lambda: BFBT with DKG as substitute subsystem operator + MG inner preconditioner
    using DKGMGPrec       = SolverAdapter< PressureDKG, PrecDKG >;
    using DKGMGInnerSolver = linalg::solvers::PCG< PressureDKG, DKGMGPrec >;
    using DKGMGSolverAdapter = SolverAdapter< BDinvBT, DKGMGInnerSolver >;

    auto run_bfbt_dkg_mg_pcg = [&]( VectorQ1Vec< ScalarType >& inv_diag,
                                    int                        inner_iters,
                                    double                     omega,
                                    const std::string& label,
                                    const std::string& table_label )
    {
        BDinvBT bdinvbt( K_neumann.block_12(), K_neumann.block_21(), inv_diag, bfbt_tmp_vel_1 );
        BDinvADinvBT bdinvadinvbt( K_neumann.block_12(), K_neumann.block_21(), K_neumann.block_11(),
                                   inv_diag, bfbt_tmp_vel_3, bfbt_tmp_vel_4 );

        DKGMGPrec dkg_mg_prec( prec_dkg_1v, dkg_ops[pressure_level] );

        auto inner_pcg_table = std::make_shared< util::Table >();
        DKGMGInnerSolver dkg_inner_solver(
            linalg::solvers::IterativeSolverParameters{ inner_iters, 1e-6, 1e-16 },
            inner_pcg_table, bfbt_pcg_tmps, dkg_mg_prec );

        DKGMGSolverAdapter dkg_adapter( dkg_inner_solver, dkg_ops[pressure_level] );

        using BFBTPrec = linalg::solvers::BFBTPreconditioner< BDinvBT, BDinvBT, BDinvADinvBT, DKGMGSolverAdapter >;
        BFBTPrec bfbt_prec( bdinvbt, bdinvadinvbt, dkg_adapter, bfbt_t1, bfbt_t2, omega );

        using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, BDinvBT, Gradient, PrecVisc, BFBTPrec >;

        PrecStokes prec( K_op.block_11(), bdinvbt, K_op.block_12(), tri_prec_tmp, prec_11, bfbt_prec );

        int iters = run_fgmres_solve( K_op, prec, u, f, tmp_fgmres, max_iters, label );
        results_table->add_row( {
            { "viscosity", viscosity_label },
            { "preconditioner", table_label },
            { "iterations", iters },
            { "dofs_vel", num_dofs_velocity },
            { "dofs_pre", num_dofs_pressure } } );
    };

    // =====================================================================
    // Exact BD⁻¹Bᵀ subsystem — Neumann BCs, 100 inner iterations
    // =====================================================================

    for ( double omega : { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 } )
    {
        std::string om_str = std::to_string( omega ).substr( 0, 3 );
        run_bfbt_pcg( inv_diag_A, dkg_diags[pressure_level], 25, omega,
                      "neum(diagA,PCG25,w=" + om_str + ")", "neum_diagA_PCG25_w" + om_str );
        run_bfbt_pcg( inv_diag_MuK, dkg_diags[pressure_level], 25, omega,
                      "neum(MuK,PCG25,w=" + om_str + ")", "neum_MuK_PCG25_w" + om_str );
        run_bfbt_pcg( inv_diag_MuK_sqrt, dkg_diags[pressure_level], 25, omega,
                      "neum(MuK_sqrt,PCG25,w=" + om_str + ")", "neum_MuK_sqrt_PCG25_w" + om_str );
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
    const int max_level = 5;

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
