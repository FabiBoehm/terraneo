/// @file adjoint_taylor_test.cpp
///
/// Adjoint gradient check for the initial-condition problem of mantle convection, set up like the HyTeG reference
/// test (Ponsuganth Ilangovan, apps/2024-convbench/adjoint-stuff/SphereInverse.cpp on branch ponsuganth/adjoint-stuff).
///
/// Forward model (nondimensional Boussinesq convection in a spherical shell, constant viscosity):
///   Stokes:  -div(2 eps(u)) + grad p = Ra T r_hat,  div u = 0,  u = 0 on both boundaries (no-slip)
///   Energy:  (M + dt (K + C(u) + G(u))) T^{n+1} = M T^n   (implicit SUPG step, kappa = 1),
///            T = 1 at the CMB, T = 0 at the surface (strong Dirichlet).
///   Step order as in the reference: u^0 = S(T^0); then for n = 0..N-1: T^{n+1} = E(u^n) T^n, u^{n+1} = S(T^{n+1}).
///   All T^n and u^n are kept in memory (the reference writes ADIOS2 checkpoints).
///
/// Objective: J(T^0) = 1/2 (T^N - T_obs)^T M (T^N - T_obs).
///
/// Adjoint (discrete, velocity held fixed as in the reference, whose adjoint Stokes solve is switched off):
///   lambda^N = M (T^N - T_obs); for n = N-1..0: solve L_n^T y = lambda^{n+1}, lambda^n = M y,
///   with L_n the eliminated SUPG matrix of step n and all Dirichlet entries set to zero. dJ/dT^0 = lambda^0.
///   Because the SUPG step is a linear system with an assembled matrix, its transpose is applied exactly
///   (UnsteadyAdvectionDiffusionSUPG::set_transposed), so the gradient is the exact gradient of the discrete
///   frozen-velocity model.
///
/// Test (as in the reference): T_obs = forward(T_true) after N steps; the base guess is T^0 = T_obs; the Taylor
/// remainder |J(T^0 + h dT) - J(T^0) - h g.dT| is evaluated for h, h/2, h/4, ... and must fall at O(h^2).
///   --frozen-velocity 1: the perturbed runs reuse the base run's velocities. The adjoint is then exact: rate 2.
///   --frozen-velocity 0: the perturbed runs re-solve Stokes (the reference's setting). The adjoint ignores the
///                        buoyancy feedback, so the remainder only falls at O(h) once that error dominates.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "grid/shell/bit_masks.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pbicgstab.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector_w1_iso_w2_w1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "util/cli11_helper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::ShellBoundaryFlag;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorW1IsoW2W1;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;

using Real = double;

struct Parameters
{
    int    min_level      = 2;
    int    max_level      = 4;
    int    lat_sdr        = 0;
    int    rad_sdr        = 0;
    double r_min          = 1.22;
    double r_max          = 2.22;
    double Ra             = 1e5;
    double kappa          = 1.0;
    double dt             = 2e-5;
    int    steps          = 80;
    double h0             = 1e-2;
    int    halvings       = 6;
    int    frozen         = 1;
    double stokes_tol     = 1e-10;
    double energy_tol     = 1e-13;
    std::string dump_dir;          // if set: write fields for visualisation (raw float64, one file per field)
    int    dump_every     = 20;    // snapshot interval in time steps
    int    taylor         = 1;     // run the Taylor test
};

// ---------------------------------------------------------------------------------------------------------------
// Field initialisation kernels (free functions: nvcc does not allow extended lambdas inside other lambdas).
// ---------------------------------------------------------------------------------------------------------------

/// Reference initial condition: conductive profile plus Y_3^2 / 10, clipped to [0, 1].
void initial_temperature( const DistributedDomain& domain, const Grid3DDataVec< Real, 3 >& coords,
                          const Grid2DDataScalar< Real >& radii, VectorW1Scalar< Real >& T, Real r_min, Real r_max )
{
    auto       data = T.grid_data();
    const Real c32  = std::sqrt( 105.0 / ( 16.0 * M_PI ) );
    Kokkos::parallel_for(
        "init_T", local_domain_md_range_policy_nodes( domain ), KOKKOS_LAMBDA( int s, int x, int y, int r ) {
            const auto p  = grid::shell::coords( s, x, y, r, coords, radii );
            const Real rr = p.norm();
            // real spherical harmonic l = 3, m = 2:  c * z (x^2 - y^2) / r^3
            const Real y32 = c32 * p( 2 ) * ( p( 0 ) * p( 0 ) - p( 1 ) * p( 1 ) ) / ( rr * rr * rr );
            const Real ref = ( r_max - rr ) / ( r_max - r_min );
            data( s, x, y, r ) = Kokkos::min( Real( 1 ), Kokkos::max( Real( 0 ), ref + y32 / 10.0 ) );
        } );
    Kokkos::fence();
}

/// Smooth perturbation that vanishes on both boundaries (consistent on duplicated interface nodes, unlike random).
void perturbation( const DistributedDomain& domain, const Grid3DDataVec< Real, 3 >& coords,
                   const Grid2DDataScalar< Real >& radii, VectorW1Scalar< Real >& dT, Real r_min, Real r_max )
{
    auto data = dT.grid_data();
    Kokkos::parallel_for(
        "init_dT", local_domain_md_range_policy_nodes( domain ), KOKKOS_LAMBDA( int s, int x, int y, int r ) {
            const auto p    = grid::shell::coords( s, x, y, r, coords, radii );
            const Real rr   = p.norm();
            const Real bump = 4.0 * ( rr - r_min ) * ( r_max - rr ) / ( ( r_max - r_min ) * ( r_max - r_min ) );
            data( s, x, y, r ) =
                bump * ( Kokkos::sin( 5.0 * p( 0 ) + 0.3 ) * Kokkos::cos( 3.0 * p( 1 ) - 0.7 ) +
                         0.5 * Kokkos::sin( 4.0 * p( 2 ) + 1.1 ) );
        } );
    Kokkos::fence();
}

/// Buoyancy force density Ra T r_hat at the velocity nodes (coords are unit-sphere directions).
void buoyancy( const DistributedDomain& domain, const Grid3DDataVec< Real, 3 >& coords, const VectorW1Scalar< Real >& T,
               VectorW1Vec< Real >& out, Real Ra )
{
    auto t = T.grid_data();
    auto o = out.grid_data();
    Kokkos::parallel_for(
        "buoyancy", local_domain_md_range_policy_nodes( domain ), KOKKOS_LAMBDA( int s, int x, int y, int r ) {
            for ( int d = 0; d < 3; ++d )
                o( s, x, y, r, d ) = Ra * t( s, x, y, r ) * coords( s, x, y, d );
        } );
    Kokkos::fence();
}

/// Dirichlet data for T: 1 at the CMB, 0 at the surface, 0 in the interior.
void temperature_boundary_values( VectorW1Scalar< Real >& g, const Grid4DDataScalar< ShellBoundaryFlag >& bmask )
{
    linalg::assign( g, 0.0 );
    kernels::common::assign_masked_else_keep_old( g.grid_data(), Real( 1 ), bmask, CMB );
}

/// Node positions (x, y, z) for the field dumps.
void node_positions( const DistributedDomain& domain, const Grid3DDataVec< Real, 3 >& coords,
                     const Grid2DDataScalar< Real >& radii, VectorW1Vec< Real >& xyz )
{
    auto o = xyz.grid_data();
    Kokkos::parallel_for(
        "xyz", local_domain_md_range_policy_nodes( domain ), KOKKOS_LAMBDA( int s, int x, int y, int r ) {
            const auto p = grid::shell::coords( s, x, y, r, coords, radii );
            for ( int d = 0; d < 3; ++d )
                o( s, x, y, r, d ) = p( d );
        } );
    Kokkos::fence();
}

/// Writes a scalar grid field as raw float64 (same node order for every field, incl. duplicated interface nodes).
void write_raw( const std::string& path, const Grid4DDataScalar< Real >& v )
{
    auto          h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, v );
    std::ofstream f( path, std::ios::binary );
    f.write( reinterpret_cast< const char* >( h.data() ), static_cast< std::streamsize >( h.size() * sizeof( Real ) ) );
}

/// a <- a / b entrywise (dual vector -> density with the lumped mass).
void divide_entries( const DistributedDomain& domain, VectorW1Scalar< Real >& a, const VectorW1Scalar< Real >& b )
{
    auto d = a.grid_data();
    auto m = b.grid_data();
    Kokkos::parallel_for(
        "divide_entries", local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( int s, int x, int y, int r ) { d( s, x, y, r ) /= m( s, x, y, r ); } );
    Kokkos::fence();
}

void zero_on_boundary( VectorW1Scalar< Real >& v, const Grid4DDataScalar< ShellBoundaryFlag >& bmask )
{
    kernels::common::assign_masked_else_keep_old( v.grid_data(), Real( 0 ), bmask, BOUNDARY );
}

// ---------------------------------------------------------------------------------------------------------------

int run( const Parameters& prm )
{
    using Stokes       = fe::wedge::operators::shell::EpsDivDivStokes< Real >;
    using Viscous      = Stokes::Block11Type;
    using Gradient     = Stokes::Block12Type;
    using VecMass      = fe::wedge::operators::shell::VectorMass< Real >;
    using Mass         = fe::wedge::operators::shell::Mass< Real >;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< Real >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< Real >;
    using Energy       = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< Real >;
    using PressureMass = fe::wedge::operators::shell::KMass< Real >;

    // --- grid hierarchy (velocity and temperature on the finest level, pressure one level coarser) ---------------
    std::vector< DistributedDomain >                       domains;
    std::vector< Grid3DDataVec< Real, 3 > >                coords;
    std::vector< Grid2DDataScalar< Real > >                radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > > owner;
    std::vector< Grid4DDataScalar< ShellBoundaryFlag > >   bmask;
    for ( int level = prm.min_level; level <= prm.max_level; ++level )
    {
        domains.push_back(
            DistributedDomain::create_uniform( level, level, prm.r_min, prm.r_max, prm.lat_sdr, prm.rad_sdr ) );
        coords.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< Real >( domains.back() ) );
        radii.push_back( grid::shell::subdomain_shell_radii< Real >( domains.back() ) );
        owner.push_back( grid::setup_node_ownership_mask_data( domains.back() ) );
        bmask.push_back( grid::shell::setup_boundary_mask_data( domains.back() ) );
    }
    const int  nl = static_cast< int >( domains.size() );
    const int  vl = nl - 1; // velocity / temperature level
    const int  pl = nl - 2; // pressure level
    const auto& dom = domains[vl];

    const long n_T = kernels::common::count_masked< long >( owner[vl], grid::NodeOwnershipFlag::OWNED );
    util::logroot << "Shell r = [" << prm.r_min << ", " << prm.r_max << "], level " << prm.max_level
                  << ", temperature DoFs " << n_T << ", Ra " << prm.Ra << ", dt " << prm.dt << ", steps "
                  << prm.steps << "\n";

    auto table = std::make_shared< util::Table >();

    // --- Stokes operator and solver (as in tests/test_epsilon_divdiv_stokes.cpp, viscosity 1) ------------------
    BoundaryConditions bcs = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } }; // array type: must not be const

    VectorW1Scalar< Real > visc( "visc", dom, owner[vl] );
    linalg::assign( visc, 1.0 );
    Stokes  K( dom, domains[pl], coords[vl], radii[vl], bmask[vl], visc.grid_data(), bcs, false );
    VecMass Mu( dom, coords[vl], radii[vl], false );

    std::vector< VectorW1Scalar< Real > > visc_c;
    std::vector< Viscous >                A_c;
    std::vector< Prolongation >           P;
    std::vector< Restriction >            R;
    std::vector< VectorW1Vec< Real > >    mg_tmp, mg_r, mg_e, inv_diag;
    for ( int l = 0; l < nl; ++l )
    {
        mg_tmp.emplace_back( "mg_tmp" + std::to_string( l ), domains[l], owner[l] );
        if ( l < nl - 1 )
        {
            visc_c.emplace_back( "visc_c" + std::to_string( l ), domains[l], owner[l] );
            linalg::assign( visc_c.back(), 1.0 );
            A_c.emplace_back( domains[l], coords[l], radii[l], bmask[l], visc_c.back().grid_data(), bcs, false );
            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[l] );
            mg_r.emplace_back( "mg_r" + std::to_string( l ), domains[l], owner[l] );
            mg_e.emplace_back( "mg_e" + std::to_string( l ), domains[l], owner[l] );
        }
    }

    using Smoother = linalg::solvers::Jacobi< Viscous >;
    std::vector< Smoother > smoothers;
    for ( int l = 0; l < nl; ++l )
    {
        inv_diag.emplace_back( "inv_diag" + std::to_string( l ), domains[l], owner[l] );
        VectorW1Vec< Real > ones( "ones", domains[l], owner[l] ), pi0( "pi0", domains[l], owner[l] ),
            pi1( "pi1", domains[l], owner[l] );
        linalg::assign( ones, 1.0 );
        Viscous& A = ( l == nl - 1 ) ? K.block_11() : A_c[l];
        A.set_diagonal( true );
        linalg::apply( A, ones, inv_diag.back() );
        A.set_diagonal( false );
        linalg::invert_entries( inv_diag.back() );
        linalg::DiagonallyScaledOperator< Viscous > DA( A, inv_diag[l] );
        const double max_ev = linalg::solvers::power_iteration< linalg::DiagonallyScaledOperator< Viscous > >(
            DA, pi0, pi1, 100 );
        smoothers.emplace_back( inv_diag[l], 3, mg_tmp[l], 2.0 / ( 1.3 * max_ev ) );
    }

    using Coarse = linalg::solvers::PCG< Viscous >;
    std::vector< VectorW1Vec< Real > > coarse_tmp;
    for ( int i = 0; i < 4; ++i )
        coarse_tmp.emplace_back( "coarse_tmp", domains[0], owner[0] );
    Coarse coarse( linalg::solvers::IterativeSolverParameters{ 1000, 1e-8, 1e-16 }, table, coarse_tmp );

    using PrecVisc = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, Coarse >;
    PrecVisc prec_11( P, R, A_c, mg_r, mg_e, mg_tmp, smoothers, smoothers, coarse, 1, 1e-8 );

    VectorW1Scalar< Real > inv_visc_p( "inv_visc_p", domains[pl], owner[pl] );
    linalg::assign( inv_visc_p, 1.0 );
    PressureMass pmass( domains[pl], coords[pl], radii[pl], inv_visc_p.grid_data(), false );
    pmass.set_lumped_diagonal( true );
    VectorW1Scalar< Real > pmass_diag( "pmass_diag", domains[pl], owner[pl] );
    {
        VectorW1Scalar< Real > ones( "ones_p", domains[pl], owner[pl] );
        linalg::assign( ones, 1.0 );
        linalg::apply( pmass, ones, pmass_diag );
    }
    using PrecSchur = linalg::solvers::DiagonalSolver< PressureMass >;
    PrecSchur prec_22( pmass_diag );

    using PrecStokes =
        linalg::solvers::BlockTriangularPreconditioner2x2< Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;
    VectorW1IsoW2W1< Real > prec_tmp( "prec_tmp", dom, domains[pl], owner[vl], owner[pl] );
    PrecStokes              prec_stokes( K.block_11(), pmass, K.block_12(), prec_tmp, prec_11, prec_22 );

    const int                              restart = 50;
    std::vector< VectorW1IsoW2W1< Real > > fgmres_tmp;
    for ( int i = 0; i < 2 * restart + 4; ++i )
        fgmres_tmp.emplace_back( "fgmres_tmp", dom, domains[pl], owner[vl], owner[pl] );
    linalg::solvers::FGMRESOptions< Real > fopt;
    fopt.restart                     = restart;
    fopt.max_iterations              = 500;
    fopt.relative_residual_tolerance = prm.stokes_tol;
    linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( fgmres_tmp, fopt, table, prec_stokes );

    VectorW1IsoW2W1< Real > up( "up", dom, domains[pl], owner[vl], owner[pl] );
    VectorW1IsoW2W1< Real > f_stokes( "f_stokes", dom, domains[pl], owner[vl], owner[pl] );
    VectorW1Vec< Real >     force( "force", dom, owner[vl] );

    /// u <- velocity of the Stokes solution with buoyancy Ra T r_hat (no-slip, so homogeneous velocity BCs).
    auto solve_stokes = [&]( const VectorW1Scalar< Real >& T, VectorW1Vec< Real >& u ) {
        buoyancy( dom, coords[vl], T, force, prm.Ra );
        linalg::apply( Mu, force, f_stokes.block_1() );
        linalg::assign( f_stokes.block_2(), 0.0 );
        fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like( f_stokes, bmask[vl], BOUNDARY );
        linalg::assign( up, 0.0 );
        linalg::solvers::solve( fgmres, K, up, f_stokes );
        linalg::assign( u, up.block_1() );
        table->clear();
    };

    // --- energy step (implicit SUPG) and its transpose ----------------------------------------------------------
    VectorW1Vec< Real > vel( "vel", dom, owner[vl] ); // the operators below read this field
    Energy E( dom, coords[vl], radii[vl], bmask[vl], vel, prm.kappa, prm.dt, true );               // eliminated
    Energy E_neu( dom, coords[vl], radii[vl], bmask[vl], vel, prm.kappa, prm.dt, false );          // no BC rows
    Energy E_neu_diag( dom, coords[vl], radii[vl], bmask[vl], vel, prm.kappa, prm.dt, false, true ); // diag
    Mass   M( dom, coords[vl], radii[vl], false );

    VectorW1Scalar< Real > g_bc( "g_bc", dom, owner[vl] ), rhs( "rhs", dom, owner[vl] ), tmp( "tmp", dom, owner[vl] );
    temperature_boundary_values( g_bc, bmask[vl] );

    std::vector< VectorW1Scalar< Real > > bicg_tmp;
    for ( int i = 0; i < 2 * ( 2 + 1 ) + 2; ++i )
        bicg_tmp.emplace_back( "bicg_tmp", dom, owner[vl] );
    linalg::solvers::PBiCGStab< Energy > bicg(
        2, linalg::solvers::IterativeSolverParameters{ 500, prm.energy_tol, 1e-30 }, table, bicg_tmp );

    /// T <- E(vel) T : one implicit step with T = 1 / 0 imposed at CMB / surface.
    auto energy_step = [&]( VectorW1Scalar< Real >& T ) {
        linalg::apply( M, T, rhs );
        fe::strong_algebraic_dirichlet_enforcement_poisson_like( E_neu, E_neu_diag, g_bc, tmp, rhs, bmask[vl], BOUNDARY );
        linalg::solvers::solve( bicg, E, T, rhs );
        table->clear();
    };

    /// lambda <- M L^-T lambda with Dirichlet entries removed (one step of the discrete adjoint).
    VectorW1Scalar< Real > y( "y", dom, owner[vl] );
    auto adjoint_step = [&]( VectorW1Scalar< Real >& lambda ) {
        zero_on_boundary( lambda, bmask[vl] );
        E.set_transposed( true );
        linalg::assign( y, 0.0 );
        linalg::solvers::solve( bicg, E, y, lambda );
        E.set_transposed( false );
        zero_on_boundary( y, bmask[vl] );
        linalg::apply( M, y, lambda );
        zero_on_boundary( lambda, bmask[vl] );
        table->clear();
    };

    // --- storage of the forward trajectory ----------------------------------------------------------------------
    const int                             N = prm.steps;
    std::vector< VectorW1Vec< Real > >    u_store;
    for ( int n = 0; n < N; ++n )
        u_store.emplace_back( "u" + std::to_string( n ), dom, owner[vl] );

    /// Runs the forward model from T0 (T0 is overwritten with T^N). store: keep the velocities in u_store.
    /// frozen: do not solve Stokes, use u_store (filled by an earlier stored run) instead.
    std::function< void( int, const VectorW1Scalar< Real >&, const VectorW1Vec< Real >& ) > snapshot;
    auto forward = [&]( VectorW1Scalar< Real >& T, bool store, bool frozen ) {
        for ( int n = 0; n < N; ++n )
        {
            if ( frozen )
                linalg::assign( vel, u_store[n] );
            else
                solve_stokes( T, vel ); // u^n = S(T^n)
            if ( store )
                linalg::assign( u_store[n], vel );
            if ( snapshot && n % prm.dump_every == 0 )
                snapshot( n, T, vel );
            energy_step( T ); // T^{n+1} = E(u^n) T^n
        }
        if ( snapshot )
            snapshot( N, T, vel );
    };

    VectorW1Scalar< Real > T_obs( "T_obs", dom, owner[vl] ), T0( "T0", dom, owner[vl] ), T( "T", dom, owner[vl] ),
        dT( "dT", dom, owner[vl] ), lam( "lambda", dom, owner[vl] ), misfit( "misfit", dom, owner[vl] );

    auto objective = [&]( const VectorW1Scalar< Real >& TN ) {
        linalg::lincomb( misfit, { 1.0, -1.0 }, { TN, T_obs } );
        linalg::apply( M, misfit, tmp );
        return 0.5 * linalg::dot( misfit, tmp );
    };

    // --- optional dumps for visualisation ----------------------------------------------------------------------
    int rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    const std::string rsuf = "_r" + std::to_string( rank ) + ".bin";
    VectorW1Scalar< Real > mlump( "mlump", dom, owner[vl] ), dens( "dens", dom, owner[vl] );
    {
        VectorW1Scalar< Real > ones( "ones_T", dom, owner[vl] );
        linalg::assign( ones, 1.0 );
        linalg::apply( M, ones, mlump ); // lumped mass: turns dual vectors (adjoint, gradient) into densities
    }
    auto dump = [&]( const std::string& name, const VectorW1Scalar< Real >& v ) {
        if ( !prm.dump_dir.empty() )
            write_raw( prm.dump_dir + "/" + name + rsuf, v.grid_data() );
    };
    auto dump_dual = [&]( const std::string& name, const VectorW1Scalar< Real >& v ) {
        if ( prm.dump_dir.empty() )
            return;
        linalg::assign( dens, v );
        divide_entries( dom, dens, mlump );
        write_raw( prm.dump_dir + "/" + name + rsuf, dens.grid_data() );
    };
    auto dump_vec = [&]( const std::string& name, const VectorW1Vec< Real >& v ) {
        if ( prm.dump_dir.empty() )
            return;
        for ( int d = 0; d < 3; ++d )
            write_raw( prm.dump_dir + "/" + name + "_" + std::to_string( d ) + rsuf, v.grid_data().comp_[d] );
    };
    if ( !prm.dump_dir.empty() )
    {
        VectorW1Vec< Real > xyz( "xyz", dom, owner[vl] );
        node_positions( dom, coords[vl], radii[vl], xyz );
        dump_vec( "xyz", xyz );
    }

    // 1) synthetic observation: T_obs = forward(T_true)
    initial_temperature( dom, coords[vl], radii[vl], T_obs, prm.r_min, prm.r_max );
    kernels::common::assign_masked_else_keep_old( T_obs.grid_data(), g_bc.grid_data(), bmask[vl], BOUNDARY );
    util::logroot << "Forward run from the true initial condition ...\n";
    snapshot = [&]( int n, const VectorW1Scalar< Real >& Tn, const VectorW1Vec< Real >& un ) {
        dump( "true_T_" + std::to_string( n ), Tn );
        if ( n < N )
            dump_vec( "true_u_" + std::to_string( n ), un );
    };
    forward( T_obs, false, false );
    dump( "T_obs", T_obs );

    // 2) base run from the guess T^0 = T_obs, storing the trajectory
    linalg::assign( T0, T_obs );
    linalg::assign( T, T0 );
    util::logroot << "Base forward run from T^0 = T_obs ...\n";
    snapshot = [&]( int n, const VectorW1Scalar< Real >& Tn, const VectorW1Vec< Real >& un ) {
        dump( "base_T_" + std::to_string( n ), Tn );
        if ( n < N )
            dump_vec( "base_u_" + std::to_string( n ), un );
    };
    forward( T, true, false );
    snapshot = nullptr;
    const double J0 = objective( T );
    VectorW1Scalar< Real > TN_base( "TN_base", dom, owner[vl] );
    linalg::assign( TN_base, T );
    dump( "misfit", misfit );

    // 3) adjoint: lambda^N = M (T^N - T_obs), then N transposed steps backward through the stored velocities
    util::logroot << "Adjoint run ...\n";
    linalg::assign( lam, tmp ); // tmp = M (T^N - T_obs) from objective()
    zero_on_boundary( lam, bmask[vl] );
    dump_dual( "adj_" + std::to_string( N ), lam );
    for ( int n = N - 1; n >= 0; --n )
    {
        linalg::assign( vel, u_store[n] );
        adjoint_step( lam );
        if ( n % prm.dump_every == 0 )
            dump_dual( "adj_" + std::to_string( n ), lam );
    }
    // lam = dJ/dT^0 (a dual vector: dJ = lam . dT)

    perturbation( dom, coords[vl], radii[vl], dT, prm.r_min, prm.r_max );
    const double gdT = linalg::dot( lam, dT );
    dump( "dT", dT );
    dump_dual( "gradient", lam );
    if ( !prm.dump_dir.empty() )
    {
        // Response of the final temperature to the perturbation, (T^N(T0 + h dT) - T^N(T0)) / h, with the base
        // velocities (what the adjoint sees) and with re-solved Stokes (the real model).
        const double hv = 1e-3;
        for ( int fr = 1; fr >= 0; --fr )
        {
            linalg::lincomb( T, { 1.0, hv }, { T0, dT } );
            forward( T, false, fr == 1 );
            linalg::lincomb( T, { 1.0 / hv, -1.0 / hv }, { T, TN_base } );
            dump( fr ? "response_frozen" : "response_coupled", T );
        }
    }
    util::logroot << "J(T^0) = " << J0 << ",  g . dT = " << gdT << ",  |g|_inf = " << linalg::norm_inf( lam ) << "\n";

    // 4) Taylor remainder test
    util::logroot << "Taylor test, " << ( prm.frozen ? "perturbed runs reuse the base velocities (exact adjoint)"
                                                      : "perturbed runs re-solve Stokes (adjoint ignores buoyancy)" )
                  << "\n";
    util::logroot << "      h            J(T0+h dT)        |dJ|           rate1   |dJ - h g.dT|   rate2\n";
    double h = prm.h0, prev1 = 0, prev2 = 0;
    for ( int k = 0; prm.taylor && k <= prm.halvings; ++k, h *= 0.5 )
    {
        linalg::lincomb( T, { 1.0, h }, { T0, dT } );
        forward( T, false, prm.frozen != 0 );
        const double Jh = objective( T );
        const double r1 = std::abs( Jh - J0 );
        const double r2 = std::abs( Jh - J0 - h * gdT );
        char line[256];
        if ( k == 0 )
            std::snprintf( line, sizeof( line ), "  %.4e   %.10e   %.4e     -     %.4e      -\n", h, Jh, r1, r2 );
        else
            std::snprintf( line, sizeof( line ), "  %.4e   %.10e   %.4e   %5.2f   %.4e   %5.2f\n", h, Jh, r1,
                           std::log2( prev1 / r1 ), r2, std::log2( prev2 / r2 ) );
        util::logroot << line;
        prev1 = r1;
        prev2 = r2;
    }
    util::logroot << "Expected: rate1 = 1, rate2 = 2 if the gradient is exact.\n";
    return 0;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    Parameters prm;
    CLI::App   app{ "Adjoint Taylor test for the initial-condition problem (reference: HyTeG SphereInverse)" };
    util::add_option_with_default( app, "--min-level", prm.min_level, "Coarsest multigrid level." );
    util::add_option_with_default( app, "--max-level", prm.max_level, "Finest level (velocity and temperature)." );
    util::add_option_with_default( app, "--lat-sdr", prm.lat_sdr, "Lateral subdomain refinement." );
    util::add_option_with_default( app, "--rad-sdr", prm.rad_sdr, "Radial subdomain refinement." );
    util::add_option_with_default( app, "--Ra", prm.Ra, "Rayleigh number." );
    util::add_option_with_default( app, "--kappa", prm.kappa, "Thermal diffusivity." );
    util::add_option_with_default( app, "--dt", prm.dt, "Fixed time step." );
    util::add_option_with_default( app, "--steps", prm.steps, "Number of time steps." );
    util::add_option_with_default( app, "--h0", prm.h0, "Largest Taylor-test perturbation size." );
    util::add_option_with_default( app, "--halvings", prm.halvings, "Number of halvings of h." );
    util::add_option_with_default( app, "--frozen-velocity", prm.frozen,
                                   "1: perturbed runs reuse the base velocities; 0: they re-solve Stokes." );
    util::add_option_with_default( app, "--stokes-tol", prm.stokes_tol, "FGMRES relative tolerance." );
    util::add_option_with_default( app, "--energy-tol", prm.energy_tol, "BiCGStab relative tolerance." );
    util::add_option_with_default( app, "--dump-dir", prm.dump_dir, "Write fields for visualisation into this directory." );
    util::add_option_with_default( app, "--dump-every", prm.dump_every, "Snapshot interval in time steps." );
    util::add_option_with_default( app, "--taylor", prm.taylor, "Run the Taylor test (0: only the fields)." );
    CLI11_PARSE( app, argc, argv );

    return run( prm );
}
