
/// @brief Tests convergence of velocity-only cell Vanka as a smoother
///        in the velocity MG preconditioner of a Stokes solver (FGMRES + block triangular preconditioner).
///
/// Uses the EpsDivDivStokes operator (variable viscosity Stokes with iso-Q2/Q1 elements).
///
/// Runs the comparison for four viscosity profiles:
///   1. Constant k=1 (baseline)
///   2. Lin et al. 2022 radial viscosity profile (contrast ~1000)
///   3. Stotz et al. 2017 radial viscosity profile (contrast ~12000)
///   4. Laterally varying viscosity: k(x,y,z) = 1 + 1000 * sin^2(3x) * cos^2(2y)

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/prolongation_linear.hpp"
#include "fe/wedge/operators/shell/restriction_linear.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "geophysics/viscosity/viscosity_interpolation.hpp"
#include "linalg/solvers/block_jacobi.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/cell_vanka.hpp"
#include "linalg/solvers/cell_vanka_stokes.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "shell/radial_profiles.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::DiagonallyScaledOperator;
using linalg::solvers::DiagonalSolver;
using linalg::solvers::power_iteration;

using ScalarType = double;

/// @brief Initialize a velocity field with some smooth non-trivial function.
struct InitialGuessVelocityInterpolator
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

/// @brief Initialize a pressure field with some smooth non-trivial function.
struct InitialGuessPressureInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords =
            grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        data_( local_subdomain_id, x, y, r ) =
            Kokkos::sin( 2 * cx ) * Kokkos::cos( 3 * cy ) * Kokkos::sin( 1 * cz );
    }
};

/// @brief Set boundary velocity DOFs from src to dst.
struct SetOnBoundary
{
    Grid4DDataVec< double, 3 > src_;
    Grid4DDataVec< double, 3 > dst_;
    int                        num_shells_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        if ( r == 0 || r == num_shells_ - 1 )
        {
            for ( int d = 0; d < 3; ++d )
            {
                dst_( local_subdomain_id, x, y, r, d ) = src_( local_subdomain_id, x, y, r, d );
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
// Wrapper operator for Vanka-preconditioned spectral radius estimation.
// Applies V^{-1} A x, where V^{-1} is one Vanka sweep (omega=1).
// ============================================================================

template < typename OperatorT, int BlockSize, typename VankaSmootherT >
class VankaScaledOperator
{
  public:
    using SrcVectorType = linalg::SrcOf< OperatorT >;
    using DstVectorType = linalg::DstOf< OperatorT >;
    using ScalarType    = typename SrcVectorType::ScalarType;

    VankaScaledOperator(
        OperatorT&      op,
        VankaSmootherT& vanka,
        SrcVectorType&  tmp )
    : op_( op )
    , vanka_( vanka )
    , tmp_( tmp )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        // tmp = A * src
        linalg::apply( op_, src, tmp_ );

        // dst = V^{-1} * tmp (one Vanka sweep with omega=1, starting from dst=0)
        linalg::assign( dst, 0.0 );
        linalg::solvers::solve( vanka_, op_, dst, tmp_ );
    }

  private:
    OperatorT&      op_;
    VankaSmootherT& vanka_;
    SrcVectorType&  tmp_;
};

// ============================================================================
// Stokes FGMRES solve with a given velocity MG smoother type
// ============================================================================

template < typename SmootherT >
int run_stokes_fgmres(
    const std::string&                                                          label,
    const int                                                                   min_level,
    const int                                                                   max_level,
    const std::function< void( DistributedDomain&,
                               const Grid3DDataVec< double, 3 >&,
                               const Grid2DDataScalar< double >&,
                               Grid4DDataScalar< double >& ) >&                k_setup,
    const int                                                                   max_fgmres_iters,
    const int                                                                   num_mg_cycles,
    const int                                                                   smoother_steps_override,
    double&                                                                     solve_time,
    double&                                                                     setup_time )
{
    using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous     = typename Stokes::Block11Type;
    using Gradient    = typename Stokes::Block12Type;
    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecLinear< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecLinear< ScalarType >;

    const auto num_levels     = static_cast< size_t >( max_level - min_level + 1 );
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    // Build domain hierarchy.
    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const auto idx = static_cast< size_t >( level - min_level );
        domains.push_back( DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    // Build viscosity on all levels.
    std::vector< VectorQ1Scalar< ScalarType > > k_vecs;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        k_vecs.emplace_back( "k_" + std::to_string( level ), domains[level], mask_data[level] );
        k_setup( domains[level], coords_shell[level], coords_radii[level], k_vecs.back().grid_data() );
    }

    // Build Stokes operator on finest level (with Dirichlet BCs).
    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vecs[velocity_level].grid_data(),
        true,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vecs[velocity_level].grid_data(),
        false,
        false );

    Stokes K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vecs[velocity_level].grid_data(),
        false,
        true );

    // Build coarse viscous operators for MG.
    std::vector< Viscous > A_c;
    for ( size_t level = 0; level < num_levels - 1; ++level )
    {
        A_c.emplace_back(
            domains[level],
            coords_shell[level],
            coords_radii[level],
            boundary_mask_data[level],
            k_vecs[level].grid_data(),
            true,
            false );
    }

    // Transfer operators.
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;
    for ( size_t level = 0; level < num_levels - 1; ++level )
    {
        P.emplace_back( coords_shell[level + 1], coords_radii[level + 1], linalg::OperatorApplyMode::Add );
        R.emplace_back( domains[level], coords_shell[level + 1], coords_radii[level + 1] );
    }

    // MG temporary vectors.
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg, tmp_mg_r, tmp_mg_e;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    // Build smoothers on all levels.
    std::vector< SmootherT >                 smoothers;
    std::vector< VectorQ1Vec< ScalarType > > smoother_tmps;

    // Extra storage for vanka smoother data (kept alive).
    using VankaSmoother     = linalg::solvers::CellVanka< Viscous, 3 >;
    using InvCellType      = typename VankaSmoother::InverseCellMatricesType;
    std::vector< InvCellType >               inv_cell_mats;
    std::vector< VectorQ1Vec< ScalarType > > vanka_corrs;

    Kokkos::Timer timer_smoother_setup;

    for ( size_t level = 0; level < num_levels; ++level )
    {
        const int smoother_steps = smoother_steps_override;
        smoother_tmps.emplace_back( "sm_tmp_" + std::to_string( level ), domains[level], mask_data[level] );

        if ( level == velocity_level )
        {
            inv_cell_mats.push_back(
                linalg::solvers::compute_cell_vanka_matrices< Viscous, 3 >( K.block_11(), domains[level] ) );
        }
        else
        {
            inv_cell_mats.push_back(
                linalg::solvers::compute_cell_vanka_matrices< Viscous, 3 >( A_c[level], domains[level] ) );
        }
        vanka_corrs.emplace_back( "vk_corr_" + std::to_string( level ), domains[level], mask_data[level] );

        // Compute Vanka-specific spectral radius: rho(V^{-1}A).
        VectorQ1Vec< ScalarType > tmp0( "tmp0", domains[level], mask_data[level] );
        VectorQ1Vec< ScalarType > tmp1( "tmp1", domains[level], mask_data[level] );
        VectorQ1Vec< ScalarType > vk_pi_tmp( "vk_pi_tmp", domains[level], mask_data[level] );
        SmootherT vanka_unit( inv_cell_mats.back(), 1, vk_pi_tmp, vanka_corrs.back(), 1.0, &domains[level] );

        VectorQ1Vec< ScalarType > vk_pi_op_tmp( "vk_pi_op_tmp", domains[level], mask_data[level] );
        using VankaOp = VankaScaledOperator< Viscous, 3, SmootherT >;
        double vanka_max_ev = 0.0;
        if ( level == velocity_level )
        {
            VankaOp vanka_op( K.block_11(), vanka_unit, vk_pi_op_tmp );
            vanka_max_ev = power_iteration< VankaOp >( vanka_op, tmp0, tmp1, 100 );
        }
        else
        {
            VankaOp vanka_op( A_c[level], vanka_unit, vk_pi_op_tmp );
            vanka_max_ev = power_iteration< VankaOp >( vanka_op, tmp0, tmp1, 100 );
        }
        const double omega_vanka = 2.0 / ( 1.1 * std::abs( vanka_max_ev ) );
        std::cout << "  Vanka level " << level << ": rho(V^{-1}A) = " << vanka_max_ev
                  << ", omega = " << omega_vanka << std::endl;

        smoothers.emplace_back(
            inv_cell_mats.back(), smoother_steps, smoother_tmps.back(), vanka_corrs.back(), omega_vanka,
            &domains[level] );
    }

    Kokkos::fence();
    const double time_smoother_setup = timer_smoother_setup.seconds();
    setup_time = time_smoother_setup;

    // Coarse grid solver (PCG on level 0).
    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;
    auto                                     cg_table = std::make_shared< util::Table >();
    std::vector< VectorQ1Vec< ScalarType > > cg_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        cg_tmps.emplace_back( "tmp_cg", domains[0], mask_data[0] );
    }
    CoarseGridSolver coarse_solver(
        linalg::solvers::IterativeSolverParameters{ 200, 1e-10, 1e-16 }, cg_table, cg_tmps );

    // Velocity MG preconditioner.
    using PrecVisc = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, SmootherT, CoarseGridSolver >;
    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_solver, num_mg_cycles, 1e-10 );

    // Schur complement preconditioner: lumped inverse diagonal of (1/k)-weighted pressure mass.
    VectorQ1Scalar< ScalarType > k_inv( "k_inv", domains[pressure_level], mask_data[pressure_level] );
    linalg::assign( k_inv, k_vecs[pressure_level] );
    linalg::invert_entries( k_inv );

    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level], k_inv.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    VectorQ1Scalar< ScalarType > lumped_diag_pmass(
        "lumped_diag_pmass", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > ones( "ones_p", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( ones, 1.0 );
        linalg::apply( pmass, ones, lumped_diag_pmass );
    }

    using PrecSchur = DiagonalSolver< PressureMass >;
    PrecSchur inv_lumped_pmass( lumped_diag_pmass );

    // Block triangular preconditioner.
    using PrecStokes = linalg::solvers::
        BlockTriangularPreconditioner2x2< Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

    VectorQ1IsoQ2Q1< ScalarType > triangular_prec_tmp(
        "tri_prec_tmp",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    PrecStokes prec_stokes( K.block_11(), pmass, K.block_12(), triangular_prec_tmp, prec_11, inv_lumped_pmass );

    // FGMRES outer solver.
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < 2 * max_fgmres_iters + 4; ++i )
    {
        tmp_fgmres.emplace_back(
            "tmp_fgmres_" + std::to_string( i ),
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                      = max_fgmres_iters;
    fgmres_options.max_iterations               = max_fgmres_iters;
    fgmres_options.relative_residual_tolerance  = 1e-10;
    fgmres_options.absolute_residual_tolerance  = 1e-16;

    auto solver_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( tmp_fgmres, fgmres_options, solver_table, prec_stokes );

    // Set up Stokes vectors.
    VectorQ1IsoQ2Q1< ScalarType > u(
        "u",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > f(
        "f",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > tmp_bc_0(
        "tmp_bc_0",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > tmp_bc_1(
        "tmp_bc_1",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    // RHS = 0 (homogeneous), initial guess is a smooth function.
    linalg::assign( f, 0.0 );

    // Set non-trivial initial guess for velocity.
    Kokkos::parallel_for(
        "initial_guess_vel",
        grid::shell::local_domain_md_range_policy_nodes( domains[velocity_level] ),
        InitialGuessVelocityInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level], u.block_1().grid_data() } );

    // Set non-trivial initial guess for pressure.
    Kokkos::parallel_for(
        "initial_guess_pre",
        grid::shell::local_domain_md_range_policy_nodes( domains[pressure_level] ),
        InitialGuessPressureInterpolator{
            coords_shell[pressure_level], coords_radii[pressure_level], u.block_2().grid_data() } );

    // Enforce Dirichlet BCs: zero velocity on boundary.
    linalg::assign( tmp_bc_0, 0.0 );
    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann,
        K_neumann_diag,
        tmp_bc_0,
        tmp_bc_1,
        f,
        boundary_mask_data[velocity_level],
        grid::shell::ShellBoundaryFlag::BOUNDARY );

    // Zero out velocity boundary DOFs in the initial guess.
    const int num_shells = domains[velocity_level].domain_info().subdomain_num_nodes_radially();
    Kokkos::parallel_for(
        "zero_boundary_u",
        grid::shell::local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SetOnBoundary{ tmp_bc_0.block_1().grid_data(), u.block_1().grid_data(), num_shells } );

    // Solve.
    std::cout << "\n=== FGMRES + " << label << " ===" << std::endl;
    std::cout << "Smoother setup time: " << time_smoother_setup << "s" << std::endl;

    Kokkos::Timer timer_solve;
    linalg::solvers::solve( fgmres, K, u, f );
    Kokkos::fence();
    solve_time = timer_solve.seconds();

    solver_table->query_rows_equals( "tag", "fgmres_solver" )
        .select_columns( { "iteration", "absolute_residual", "relative_residual" } )
        .print_pretty();

    const int iterations =
        static_cast< int >( solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows().size() );

    std::cout << "FGMRES iterations: " << iterations << ", solve time: " << solve_time << "s" << std::endl;

    return iterations;
}

// ============================================================================
// Stokes FGMRES solve with block Jacobi as velocity MG smoother
// ============================================================================

int run_stokes_fgmres_block_jacobi(
    const std::string&                                                          label,
    const int                                                                   min_level,
    const int                                                                   max_level,
    const std::function< void( DistributedDomain&,
                               const Grid3DDataVec< double, 3 >&,
                               const Grid2DDataScalar< double >&,
                               Grid4DDataScalar< double >& ) >&                k_setup,
    const int                                                                   max_fgmres_iters,
    const int                                                                   num_mg_cycles,
    const int                                                                   smoother_steps_override,
    double&                                                                     solve_time,
    double&                                                                     setup_time )
{
    using Stokes       = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous      = typename Stokes::Block11Type;
    using Gradient     = typename Stokes::Block12Type;
    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecLinear< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecLinear< ScalarType >;

    using BlockSmoother = linalg::solvers::BlockJacobi< Viscous, 3 >;

    const auto num_levels     = static_cast< size_t >( max_level - min_level + 1 );
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    // Build domain hierarchy.
    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const auto idx = static_cast< size_t >( level - min_level );
        domains.push_back( DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    // Build viscosity on all levels.
    std::vector< VectorQ1Scalar< ScalarType > > k_vecs;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        k_vecs.emplace_back( "k_" + std::to_string( level ), domains[level], mask_data[level] );
        k_setup( domains[level], coords_shell[level], coords_radii[level], k_vecs.back().grid_data() );
    }

    // Build Stokes operator on finest level (with Dirichlet BCs).
    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vecs[velocity_level].grid_data(),
        true,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vecs[velocity_level].grid_data(),
        false,
        false );

    Stokes K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vecs[velocity_level].grid_data(),
        false,
        true );

    // Build coarse viscous operators for MG.
    std::vector< Viscous > A_c;
    for ( size_t level = 0; level < num_levels - 1; ++level )
    {
        A_c.emplace_back(
            domains[level],
            coords_shell[level],
            coords_radii[level],
            boundary_mask_data[level],
            k_vecs[level].grid_data(),
            true,
            false );
    }

    // Transfer operators.
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;
    for ( size_t level = 0; level < num_levels - 1; ++level )
    {
        P.emplace_back( coords_shell[level + 1], coords_radii[level + 1], linalg::OperatorApplyMode::Add );
        R.emplace_back( domains[level], coords_shell[level + 1], coords_radii[level + 1] );
    }

    // MG temporary vectors.
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg, tmp_mg_r, tmp_mg_e;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    // Build block Jacobi smoothers on all levels.
    std::vector< BlockSmoother >                smoothers;
    std::vector< VectorQ1Vec< ScalarType > >    smoother_tmps;
    std::vector< VectorQ1Vec< ScalarType > >    inv_diags;
    using InvBlockDiagType = typename BlockSmoother::InverseBlockDiagonalType;
    std::vector< InvBlockDiagType >             inv_block_diags;

    Kokkos::Timer timer_smoother_setup;

    for ( size_t level = 0; level < num_levels; ++level )
    {
        smoother_tmps.emplace_back( "sm_tmp_" + std::to_string( level ), domains[level], mask_data[level] );

        // Compute inverse block diagonal.
        if ( level == velocity_level )
        {
            inv_block_diags.push_back(
                linalg::solvers::compute_inverse_block_diagonal< Viscous, 3 >( K.block_11(), domains[level] ) );
        }
        else
        {
            inv_block_diags.push_back(
                linalg::solvers::compute_inverse_block_diagonal< Viscous, 3 >( A_c[level], domains[level] ) );
        }

        // Compute omega via power iteration on D^{-1}A.
        // We use the point diagonal for spectral radius estimation.
        inv_diags.emplace_back( "inv_diag_" + std::to_string( level ), domains[level], mask_data[level] );
        {
            VectorQ1Vec< ScalarType > ones( "ones", domains[level], mask_data[level] );
            linalg::assign( ones, 1.0 );
            if ( level == velocity_level )
            {
                K.block_11().set_diagonal( true );
                linalg::apply( K.block_11(), ones, inv_diags.back() );
                K.block_11().set_diagonal( false );
            }
            else
            {
                A_c[level].set_diagonal( true );
                linalg::apply( A_c[level], ones, inv_diags.back() );
                A_c[level].set_diagonal( false );
            }
            linalg::invert_entries( inv_diags.back() );
        }

        VectorQ1Vec< ScalarType > tmp0( "tmp0", domains[level], mask_data[level] );
        VectorQ1Vec< ScalarType > tmp1( "tmp1", domains[level], mask_data[level] );
        double max_ev = 0.0;
        if ( level == velocity_level )
        {
            DiagonallyScaledOperator< Viscous > dA( K.block_11(), inv_diags.back() );
            max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( dA, tmp0, tmp1, 100 );
        }
        else
        {
            DiagonallyScaledOperator< Viscous > dA( A_c[level], inv_diags.back() );
            max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( dA, tmp0, tmp1, 100 );
        }
        const double omega = 2.0 / ( 1.1 * std::abs( max_ev ) );
        std::cout << "  BlockJacobi level " << level << ": rho(D^{-1}A) = " << max_ev
                  << ", omega = " << omega << std::endl;

        smoothers.emplace_back( inv_block_diags.back(), smoother_steps_override, smoother_tmps.back(), omega );
    }

    Kokkos::fence();
    const double time_smoother_setup = timer_smoother_setup.seconds();
    setup_time = time_smoother_setup;

    // Coarse grid solver (PCG on level 0).
    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;
    auto                                     cg_table = std::make_shared< util::Table >();
    std::vector< VectorQ1Vec< ScalarType > > cg_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        cg_tmps.emplace_back( "tmp_cg", domains[0], mask_data[0] );
    }
    CoarseGridSolver coarse_solver(
        linalg::solvers::IterativeSolverParameters{ 200, 1e-10, 1e-16 }, cg_table, cg_tmps );

    // Velocity MG preconditioner.
    using PrecVisc = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, BlockSmoother, CoarseGridSolver >;
    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_solver, num_mg_cycles, 1e-10 );

    // Schur complement preconditioner: lumped inverse diagonal of (1/k)-weighted pressure mass.
    VectorQ1Scalar< ScalarType > k_inv( "k_inv", domains[pressure_level], mask_data[pressure_level] );
    linalg::assign( k_inv, k_vecs[pressure_level] );
    linalg::invert_entries( k_inv );

    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level], k_inv.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    VectorQ1Scalar< ScalarType > lumped_diag_pmass(
        "lumped_diag_pmass", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > ones( "ones_p", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( ones, 1.0 );
        linalg::apply( pmass, ones, lumped_diag_pmass );
    }

    using PrecSchur = DiagonalSolver< PressureMass >;
    PrecSchur inv_lumped_pmass( lumped_diag_pmass );

    // Block triangular preconditioner.
    using PrecStokes = linalg::solvers::
        BlockTriangularPreconditioner2x2< Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

    VectorQ1IsoQ2Q1< ScalarType > triangular_prec_tmp(
        "tri_prec_tmp",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    PrecStokes prec_stokes( K.block_11(), pmass, K.block_12(), triangular_prec_tmp, prec_11, inv_lumped_pmass );

    // FGMRES outer solver.
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < 2 * max_fgmres_iters + 4; ++i )
    {
        tmp_fgmres.emplace_back(
            "tmp_fgmres_" + std::to_string( i ),
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                      = max_fgmres_iters;
    fgmres_options.max_iterations               = max_fgmres_iters;
    fgmres_options.relative_residual_tolerance  = 1e-10;
    fgmres_options.absolute_residual_tolerance  = 1e-16;

    auto solver_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( tmp_fgmres, fgmres_options, solver_table, prec_stokes );

    // Set up Stokes vectors.
    VectorQ1IsoQ2Q1< ScalarType > u(
        "u",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > f(
        "f",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > tmp_bc_0(
        "tmp_bc_0",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > tmp_bc_1(
        "tmp_bc_1",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    // RHS = 0 (homogeneous), initial guess is a smooth function.
    linalg::assign( f, 0.0 );

    // Set non-trivial initial guess for velocity.
    Kokkos::parallel_for(
        "initial_guess_vel",
        grid::shell::local_domain_md_range_policy_nodes( domains[velocity_level] ),
        InitialGuessVelocityInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level], u.block_1().grid_data() } );

    // Set non-trivial initial guess for pressure.
    Kokkos::parallel_for(
        "initial_guess_pre",
        grid::shell::local_domain_md_range_policy_nodes( domains[pressure_level] ),
        InitialGuessPressureInterpolator{
            coords_shell[pressure_level], coords_radii[pressure_level], u.block_2().grid_data() } );

    // Enforce Dirichlet BCs: zero velocity on boundary.
    linalg::assign( tmp_bc_0, 0.0 );
    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann,
        K_neumann_diag,
        tmp_bc_0,
        tmp_bc_1,
        f,
        boundary_mask_data[velocity_level],
        grid::shell::ShellBoundaryFlag::BOUNDARY );

    // Zero out velocity boundary DOFs in the initial guess.
    const int num_shells = domains[velocity_level].domain_info().subdomain_num_nodes_radially();
    Kokkos::parallel_for(
        "zero_boundary_u",
        grid::shell::local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SetOnBoundary{ tmp_bc_0.block_1().grid_data(), u.block_1().grid_data(), num_shells } );

    // Solve.
    std::cout << "\n=== FGMRES + " << label << " ===" << std::endl;
    std::cout << "Smoother setup time: " << time_smoother_setup << "s" << std::endl;

    Kokkos::Timer timer_solve;
    linalg::solvers::solve( fgmres, K, u, f );
    Kokkos::fence();
    solve_time = timer_solve.seconds();

    solver_table->query_rows_equals( "tag", "fgmres_solver" )
        .select_columns( { "iteration", "absolute_residual", "relative_residual" } )
        .print_pretty();

    const int iterations =
        static_cast< int >( solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows().size() );

    std::cout << "FGMRES iterations: " << iterations << ", solve time: " << solve_time << "s" << std::endl;

    return iterations;
}

// ============================================================================
// Run comparison for one viscosity profile
// ============================================================================

void run_stokes_smoother_comparison(
    const std::string&                                                          label,
    const int                                                                   min_level,
    const int                                                                   max_level,
    const std::function< void( DistributedDomain&,
                               const Grid3DDataVec< double, 3 >&,
                               const Grid2DDataScalar< double >&,
                               Grid4DDataScalar< double >& ) >&                k_setup,
    const int                                                                   max_fgmres_iters,
    const int                                                                   num_mg_cycles )
{
    using Stokes  = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous = typename Stokes::Block11Type;

    using VankaSmoother    = linalg::solvers::CellVanka< Viscous, 3 >;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Stokes smoother comparison (Block Jacobi vs Vanka): " << label << std::endl;
    std::cout << "================================================================" << std::endl;

    double time_jacobi_3 = 0.0, time_jacobi_6 = 0.0;
    double setup_jacobi_3 = 0.0, setup_jacobi_6 = 0.0;
    double time_vanka_3 = 0.0, time_vanka_6 = 0.0;
    double setup_vanka_3 = 0.0, setup_vanka_6 = 0.0;

    const int iters_jacobi_3 =
        run_stokes_fgmres_block_jacobi( "Block Jacobi (3 steps)", min_level, max_level, k_setup, max_fgmres_iters, num_mg_cycles, 3, time_jacobi_3, setup_jacobi_3 );

    const int iters_jacobi_6 =
        run_stokes_fgmres_block_jacobi( "Block Jacobi (6 steps)", min_level, max_level, k_setup, max_fgmres_iters, num_mg_cycles, 6, time_jacobi_6, setup_jacobi_6 );

    const int iters_vanka_3 =
        run_stokes_fgmres< VankaSmoother >( "Cell Vanka (3 steps)", min_level, max_level, k_setup, max_fgmres_iters, num_mg_cycles, 3, time_vanka_3, setup_vanka_3 );

    const int iters_vanka_6 =
        run_stokes_fgmres< VankaSmoother >( "Cell Vanka (6 steps)", min_level, max_level, k_setup, max_fgmres_iters, num_mg_cycles, 6, time_vanka_6, setup_vanka_6 );

    std::cout << "\n--- Summary: " << label << " ---" << std::endl;
    std::cout << "FGMRES iterations:  jacobi(3)=" << iters_jacobi_3 << "  jacobi(6)=" << iters_jacobi_6
              << "  vanka(3)=" << iters_vanka_3 << "  vanka(6)=" << iters_vanka_6 << std::endl;
    std::cout << "Setup time:  jacobi(3)=" << setup_jacobi_3 << "s  jacobi(6)=" << setup_jacobi_6
              << "s  vanka(3)=" << setup_vanka_3 << "s  vanka(6)=" << setup_vanka_6 << "s" << std::endl;
    std::cout << "Solve time:  jacobi(3)=" << time_jacobi_3 << "s  jacobi(6)=" << time_jacobi_6
              << "s  vanka(3)=" << time_vanka_3 << "s  vanka(6)=" << time_vanka_6 << "s" << std::endl;
}

// ============================================================================
// FGMRES with Stokes Vanka preconditioner (velocity + pressure DOFs)
// ============================================================================

int run_stokes_fgmres_with_stokes_vanka(
    const std::string&                                                          label,
    const int                                                                   min_level,
    const int                                                                   max_level,
    const std::function< void( DistributedDomain&,
                               const Grid3DDataVec< double, 3 >&,
                               const Grid2DDataScalar< double >&,
                               Grid4DDataScalar< double >& ) >&                k_setup,
    const int                                                                   max_fgmres_iters,
    const int                                                                   smoother_steps,
    double&                                                                     solve_time,
    double&                                                                     setup_time )
{
    using Stokes       = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous      = typename Stokes::Block11Type;
    using StokesVanka  = linalg::solvers::CellVankaStokes< Stokes >;

    const auto num_levels     = static_cast< size_t >( max_level - min_level + 1 );
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    // Build domain hierarchy (only need finest two levels for Stokes).
    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const auto idx = static_cast< size_t >( level - min_level );
        domains.push_back( DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    // Build viscosity on finest level.
    VectorQ1Scalar< ScalarType > k_vec( "k_finest", domains[velocity_level], mask_data[velocity_level] );
    k_setup( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], k_vec.grid_data() );

    // Build Stokes operator on finest level (with Dirichlet BCs).
    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vec.grid_data(),
        true,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vec.grid_data(),
        false,
        false );

    Stokes K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k_vec.grid_data(),
        false,
        true );

    // Build Stokes Vanka matrices.
    Kokkos::Timer timer_vanka_setup;
    auto inv_stokes_cells = linalg::solvers::compute_stokes_cell_vanka_matrices< Viscous, ScalarType >(
        K.block_11(),
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        true );
    Kokkos::fence();
    const double time_vanka_setup = timer_vanka_setup.seconds();
    setup_time = time_vanka_setup;

    // Temporary vectors for Stokes Vanka.
    VectorQ1IsoQ2Q1< ScalarType > vanka_tmp(
        "vanka_tmp",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > vanka_corr(
        "vanka_corr",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    // Estimate spectral radius for omega.
    // Use a simple fixed omega for now.
    const double omega_stokes_vanka = 1.0 / 8.0;

    StokesVanka stokes_vanka(
        inv_stokes_cells,
        smoother_steps,
        vanka_tmp,
        vanka_corr,
        omega_stokes_vanka,
        &domains[velocity_level],
        &domains[pressure_level] );

    // FGMRES outer solver with Stokes Vanka as preconditioner.
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < 2 * max_fgmres_iters + 4; ++i )
    {
        tmp_fgmres.emplace_back(
            "tmp_fgmres_" + std::to_string( i ),
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                      = max_fgmres_iters;
    fgmres_options.max_iterations               = max_fgmres_iters;
    fgmres_options.relative_residual_tolerance  = 1e-10;
    fgmres_options.absolute_residual_tolerance  = 1e-16;

    auto solver_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, StokesVanka > fgmres(
        tmp_fgmres, fgmres_options, solver_table, stokes_vanka );

    // Set up Stokes vectors.
    VectorQ1IsoQ2Q1< ScalarType > u(
        "u",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > f(
        "f",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > tmp_bc_0(
        "tmp_bc_0",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    VectorQ1IsoQ2Q1< ScalarType > tmp_bc_1(
        "tmp_bc_1",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    linalg::assign( f, 0.0 );

    // Non-trivial initial guess.
    Kokkos::parallel_for(
        "initial_guess_vel",
        grid::shell::local_domain_md_range_policy_nodes( domains[velocity_level] ),
        InitialGuessVelocityInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level], u.block_1().grid_data() } );

    Kokkos::parallel_for(
        "initial_guess_pre",
        grid::shell::local_domain_md_range_policy_nodes( domains[pressure_level] ),
        InitialGuessPressureInterpolator{
            coords_shell[pressure_level], coords_radii[pressure_level], u.block_2().grid_data() } );

    // Enforce Dirichlet BCs.
    linalg::assign( tmp_bc_0, 0.0 );
    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann,
        K_neumann_diag,
        tmp_bc_0,
        tmp_bc_1,
        f,
        boundary_mask_data[velocity_level],
        grid::shell::ShellBoundaryFlag::BOUNDARY );

    const int num_shells = domains[velocity_level].domain_info().subdomain_num_nodes_radially();
    Kokkos::parallel_for(
        "zero_boundary_u",
        grid::shell::local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SetOnBoundary{ tmp_bc_0.block_1().grid_data(), u.block_1().grid_data(), num_shells } );

    // Solve.
    std::cout << "\n=== FGMRES + " << label << " ===" << std::endl;
    std::cout << "Stokes Vanka setup time: " << time_vanka_setup << "s" << std::endl;

    Kokkos::Timer timer_solve;
    linalg::solvers::solve( fgmres, K, u, f );
    Kokkos::fence();
    solve_time = timer_solve.seconds();

    solver_table->query_rows_equals( "tag", "fgmres_solver" )
        .select_columns( { "iteration", "absolute_residual", "relative_residual" } )
        .print_pretty();

    const int iterations =
        static_cast< int >( solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows().size() );

    std::cout << "FGMRES iterations: " << iterations << ", solve time: " << solve_time << "s" << std::endl;

    return iterations;
}

// ============================================================================
// Main
// ============================================================================

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int min_level        = 1;
    const int max_level        = 3;
    const int max_fgmres_iters = 100;
    const int num_mg_cycles    = 2;

    std::cout << "\n################################################################" << std::endl;
    std::cout << "# Stokes smoother comparison (FGMRES + block triangular prec)" << std::endl;
    std::cout << "# Velocity MG smoother: Block Jacobi vs Cell Vanka (coloring)" << std::endl;
    std::cout << "# min_level=" << min_level << ", max_level=" << max_level
              << ", MG V-cycles=" << num_mg_cycles << ", FGMRES tol=1e-10" << std::endl;
    std::cout << "################################################################" << std::endl;

    // --- 1. Constant k=1 ---
    {
        auto k_setup = []( DistributedDomain& /*dom*/,
                           const Grid3DDataVec< double, 3 >& /*cs*/,
                           const Grid2DDataScalar< double >& /*cr*/,
                           Grid4DDataScalar< double >& k_data ) {
            Kokkos::deep_copy( k_data, 1.0 );
        };
        run_stokes_smoother_comparison( "Constant k=1", min_level, max_level, k_setup, max_fgmres_iters, num_mg_cycles );
    }

    // --- 2. Lin et al. 2022 ---
    {
        auto k_setup = []( DistributedDomain& /*dom*/,
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
        run_stokes_smoother_comparison(
            "Lin et al. 2022 (contrast ~1000)", min_level, max_level, k_setup, max_fgmres_iters, num_mg_cycles );
    }

    // --- 3. Stotz et al. 2017 ---
    {
        auto k_setup = []( DistributedDomain& /*dom*/,
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
        run_stokes_smoother_comparison(
            "Stotz et al. 2017 (contrast ~12000)", min_level, max_level, k_setup, max_fgmres_iters, num_mg_cycles );
    }

    // --- 4. Laterally varying viscosity (needs more iterations) ---
    {
        const int max_fgmres_iters_lateral = 150;
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
        run_stokes_smoother_comparison(
            "Lateral k = 1 + 1000*sin^2(3x)*cos^2(2y)", min_level, max_level, k_setup, max_fgmres_iters_lateral, num_mg_cycles );
    }

    // ================================================================
    // Stokes Vanka: coupled velocity-pressure cell Vanka preconditioner
    // ================================================================

    std::cout << "\n################################################################" << std::endl;
    std::cout << "# Stokes Vanka: FGMRES + coupled velocity-pressure Vanka prec" << std::endl;
    std::cout << "################################################################" << std::endl;

    {
        auto k_setup = []( DistributedDomain& /*dom*/,
                           const Grid3DDataVec< double, 3 >& /*cs*/,
                           const Grid2DDataScalar< double >& /*cr*/,
                           Grid4DDataScalar< double >& k_data ) {
            Kokkos::deep_copy( k_data, 1.0 );
        };
        double solve_time = 0.0, setup_time = 0.0;
        const int iters = run_stokes_fgmres_with_stokes_vanka(
            "Stokes Vanka (k=1, 3 steps)", min_level, max_level, k_setup, max_fgmres_iters, 3,
            solve_time, setup_time );
        std::cout << "Stokes Vanka (k=1): " << iters << " iters, setup=" << setup_time
                  << "s, solve=" << solve_time << "s" << std::endl;
    }

    return EXIT_SUCCESS;
}
