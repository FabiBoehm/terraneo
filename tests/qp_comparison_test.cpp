// qp_comparison_test
//
// Convergence comparison of quadrature variants of the eps-div-div (deviatoric
// viscous) A-block operator, solved as a STANDALONE vector operator (no pressure
// coupling) with Dirichlet BCs and a manufactured solution:
//
//   1. wedge 1-point  (Felippa 1x1, matrix-free fast Dirichlet/Neumann path)
//   2. wedge 6-point  (Felippa 3x2, assembled slow path)
//   3. hex   8-point  (2x2x2 Gauss, portable hex path -- runs on CUDA/host)
//
// All three solve the SAME manufactured problem (eta = 2 + sin z) and report the
// L2 error across refinement levels so the discretizations can be compared.
//
// The manufactured solution and the precomputed RHS f = -div(2 eta (eps(u) -
// 1/3 div(u) I)) are taken verbatim from test_epsilon_divdiv_cg.cpp.

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "grid/shell/bit_masks.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/cli11_helper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

#include <algorithm>
#include <map>
#include <string>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::DiagonallyScaledOperator;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::solvers::power_iteration;

// Viscosity profile selector -- identical to test_epsilon_divdiv_stokes.cpp.
//   profile 0 = constant : eta = base
//   profile 1 = smooth   : eta = 2 + sin(z)
//   profile 2 = tanh     : eta = base ... base*contrast, sharp radial jump at r_mid
enum class ViscProfileKind
{
    Constant = 0,
    Smooth   = 1,
    Tanh     = 2,
};

struct ViscConfig
{
    int    profile  = static_cast< int >( ViscProfileKind::Smooth );
    double base     = 1.0;
    double contrast = 1000.0;
    double r_mid    = 0.75;
    double width    = 0.05;

    KOKKOS_INLINE_FUNCTION
    double eta( const double x, const double y, const double z ) const
    {
        if ( profile == static_cast< int >( ViscProfileKind::Constant ) )
        {
            return base;
        }
        if ( profile == static_cast< int >( ViscProfileKind::Tanh ) )
        {
            const double rad      = Kokkos::sqrt( x * x + y * y + z * z );
            const double eta_low  = base;
            const double eta_high = base * contrast;
            return eta_low + ( eta_high - eta_low ) * 0.5 * ( 1.0 + Kokkos::tanh( ( rad - r_mid ) / width ) );
        }
        return 2.0 + Kokkos::sin( z ); // Smooth
    }

    KOKKOS_INLINE_FUNCTION
    void eta_grad( const double x, const double y, const double z, double& gx, double& gy, double& gz ) const
    {
        gx = 0.0;
        gy = 0.0;
        gz = 0.0;
        if ( profile == static_cast< int >( ViscProfileKind::Constant ) )
        {
            return;
        }
        if ( profile == static_cast< int >( ViscProfileKind::Tanh ) )
        {
            const double rad = Kokkos::sqrt( x * x + y * y + z * z );
            if ( rad > 0.0 )
            {
                const double eta_low  = base;
                const double eta_high = base * contrast;
                const double t        = Kokkos::tanh( ( rad - r_mid ) / width );
                const double deta_dr  = ( eta_high - eta_low ) * 0.5 * ( 1.0 - t * t ) / width;
                gx                    = deta_dr * x / rad;
                gy                    = deta_dr * y / rad;
                gz                    = deta_dr * z / rad;
            }
            return;
        }
        gz = Kokkos::cos( z ); // Smooth: eta = 2 + sin(z)
    }
};

// Divergence-free manufactured velocity (same as the Stokes test):
//   u = (-4 cos 4z, 8 cos 8x, -2 cos 2y)
struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataVec< double, 3 >&                         data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        bool                                                      only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_( mask )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
        {
            data_( local_subdomain_id, x, y, r, 0 ) = -4 * Kokkos::cos( 4 * coords( 2 ) );
            data_( local_subdomain_id, x, y, r, 1 ) = 8 * Kokkos::cos( 8 * coords( 0 ) );
            data_( local_subdomain_id, x, y, r, 2 ) = -2 * Kokkos::cos( 2 * coords( 1 ) );
        }
    }
};

struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    ViscConfig                 cfg_;

    KInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const ViscConfig&                 cfg )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , cfg_( cfg )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        data_( local_subdomain_id, x, y, r )  = cfg_.eta( coords( 0 ), coords( 1 ), coords( 2 ) );
    }
};

// Pure-viscous manufactured RHS (no pressure term). The velocity is
// divergence-free, so f = -div(2 eta eps(u)) = -eta Lap(u) - 2 (grad eta . eps(u)).
// This is the Stokes-test RHS with the grad-p term removed.
struct RHSInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    ViscConfig                 cfg_;

    RHSInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data,
        const ViscConfig&                 cfg )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , cfg_( cfg )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  cx     = coords( 0 );
        const double                  cy     = coords( 1 );
        const double                  cz     = coords( 2 );

        const double eta = cfg_.eta( cx, cy, cz );
        double       gx, gy, gz;
        cfg_.eta_grad( cx, cy, cz, gx, gy, gz );

        const double lap_u0 = 64.0 * Kokkos::cos( 4 * cz );
        const double lap_u1 = -512.0 * Kokkos::cos( 8 * cx );
        const double lap_u2 = 8.0 * Kokkos::cos( 2 * cy );

        const double eps01 = -32.0 * Kokkos::sin( 8 * cx );
        const double eps02 = 8.0 * Kokkos::sin( 4 * cz );
        const double eps12 = 2.0 * Kokkos::sin( 2 * cy );

        data_( local_subdomain_id, x, y, r, 0 ) = -eta * lap_u0 - 2.0 * ( gy * eps01 + gz * eps02 );
        data_( local_subdomain_id, x, y, r, 1 ) = -eta * lap_u1 - 2.0 * ( gx * eps01 + gz * eps12 );
        data_( local_subdomain_id, x, y, r, 2 ) = -eta * lap_u2 - 2.0 * ( gx * eps02 + gy * eps12 );
    }
};

enum class QpConfig
{
    WedgeQp1,
    WedgeQp6,
    HexQp1,
    HexQp8,
};

using Epsilon = fe::wedge::operators::shell::EpsilonDivDivKerngen< double, 3 >;

static const char* config_name( QpConfig c )
{
    switch ( c )
    {
        case QpConfig::WedgeQp1:
            return "wedge-1pt";
        case QpConfig::WedgeQp6:
            return "wedge-6pt";
        case QpConfig::HexQp1:
            return "hex-1pt";
        case QpConfig::HexQp8:
            return "hex-8pt";
    }
    return "?";
}

// Apply the chosen quadrature / kernel-path configuration to an A-block operator.
static void configure_op( Epsilon& A, QpConfig cfg )
{
    using KP = Epsilon::KernelPath;
    switch ( cfg )
    {
        case QpConfig::WedgeQp1:
            A.set_single_quadpoint( true ); // matrix-free fast DN path is intrinsically 1-point
            break;
        case QpConfig::WedgeQp6:
            A.set_single_quadpoint( false );
            A.set_kernel_path( KP::Slow ); // 6-point only takes effect on the assembled path
            break;
        case QpConfig::HexQp1:
            A.set_single_quadpoint( true ); // single cell-centre point (rank-deficient / hourglass)
            A.set_kernel_path( KP::HexPortable );
            break;
        case QpConfig::HexQp8:
            A.set_single_quadpoint( false ); // 2x2x2 Gauss on the hex (full rank)
            A.set_kernel_path( KP::HexPortable );
            break;
    }
}

double test(
    int level, int level_subdomains, QpConfig cfg, const ViscConfig& vcfg, const std::shared_ptr< util::Table >& table )
{
    using ScalarType = double;

    const int num_levels = level + 1; // MG hierarchy: domains 0..level
    const int fine       = num_levels - 1;

    // ---- per-level geometry / masks ----
    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > bmask;
    std::vector< Grid3DDataVec< double, 3 > >                         coords;
    std::vector< Grid2DDataScalar< double > >                         radii;
    std::vector< VectorQ1Scalar< ScalarType > >                       k;

    for ( int l = 0; l < num_levels; ++l )
    {
        domains.push_back( DistributedDomain::create_uniform( l, l, 0.5, 1.0, level_subdomains, level_subdomains ) );
        mask.push_back( grid::setup_node_ownership_mask_data( domains[l] ) );
        bmask.push_back( grid::shell::setup_boundary_mask_data( domains[l] ) );
        coords.push_back( terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[l] ) );
        radii.push_back( terra::grid::shell::subdomain_shell_radii< ScalarType >( domains[l] ) );
        k.emplace_back( "k_" + std::to_string( l ), domains[l], mask[l] );
        Kokkos::parallel_for(
            "coefficient interpolation",
            local_domain_md_range_policy_nodes( domains[l] ),
            KInterpolator( coords[l], radii[l], k[l].grid_data(), vcfg ) );
    }
    Kokkos::fence();

    const auto num_dofs = kernels::common::count_masked< long >( mask[fine], grid::NodeOwnershipFlag::OWNED );

    BoundaryConditions bcs         = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } };
    BoundaryConditions bcs_neumann = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };

    // ---- fine, config-specific operators (the operator under study) ----
    Epsilon A( domains[fine], coords[fine], radii[fine], bmask[fine], k[fine].grid_data(), bcs, false );
    Epsilon A_neumann(
        domains[fine], coords[fine], radii[fine], bmask[fine], k[fine].grid_data(), bcs_neumann, false );
    Epsilon A_neumann_diag(
        domains[fine], coords[fine], radii[fine], bmask[fine], k[fine].grid_data(), bcs_neumann, true );
    configure_op( A, cfg );
    configure_op( A_neumann, cfg );
    configure_op( A_neumann_diag, cfg );

    // ---- multigrid preconditioner (same structure as the Stokes A-block MG) ----
    // Coarse operators use the default wedge-1pt fast path (cheap, full-rank); the
    // fine-level operator inside the V-cycle is the config-specific `A` (passed at solve()).
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;
    using Smoother     = linalg::solvers::Jacobi< Epsilon >;
    using CoarseSolver = linalg::solvers::PCG< Epsilon >;
    using PrecMG       = linalg::solvers::Multigrid< Epsilon, Prolongation, Restriction, Smoother, CoarseSolver >;

    std::vector< Epsilon >                      A_c;
    std::vector< Prolongation >                 P;
    std::vector< Restriction >                  R;
    std::vector< VectorQ1Vec< ScalarType > >    tmp_mg, tmp_mg_r, tmp_mg_e, inv_diag;

    for ( int l = 0; l < num_levels; ++l )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( l ), domains[l], mask[l] );
        if ( l < fine )
        {
            A_c.emplace_back( domains[l], coords[l], radii[l], bmask[l], k[l].grid_data(), bcs, false );
            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[l] );
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( l ), domains[l], mask[l] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( l ), domains[l], mask[l] );
        }
    }

    std::vector< Smoother > smoothers;
    for ( int l = 0; l < num_levels; ++l )
    {
        inv_diag.emplace_back( "inv_diag_" + std::to_string( l ), domains[l], mask[l] );
        VectorQ1Vec< ScalarType > ones( "ones_" + std::to_string( l ), domains[l], mask[l] );
        linalg::assign( ones, 1.0 );

        Epsilon& Al = ( l == fine ) ? A : A_c[l];
        Al.set_diagonal( true );
        linalg::apply( Al, ones, inv_diag.back() );
        Al.set_diagonal( false );
        linalg::invert_entries( inv_diag.back() );

        VectorQ1Vec< ScalarType > pi0( "pi0_" + std::to_string( l ), domains[l], mask[l] );
        VectorQ1Vec< ScalarType > pi1( "pi1_" + std::to_string( l ), domains[l], mask[l] );
        DiagonallyScaledOperator< Epsilon > scaled( Al, inv_diag[l] );
        const double max_ev = power_iteration< DiagonallyScaledOperator< Epsilon > >( scaled, pi0, pi1, 50 );
        const double omega  = 2.0 / ( 1.3 * max_ev );

        smoothers.emplace_back( inv_diag[l], 3, tmp_mg[l], omega );
    }

    auto                                     coarse_table = std::make_shared< util::Table >();
    std::vector< VectorQ1Vec< ScalarType > > coarse_tmps;
    for ( int i = 0; i < 4; ++i )
    {
        coarse_tmps.emplace_back( "coarse_tmp_" + std::to_string( i ), domains[0], mask[0] );
    }
    CoarseSolver coarse_solver(
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-6, 1e-16 }, coarse_table, coarse_tmps );

    PrecMG prec( P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_solver, 1, 1e-8 );

    // ---- manufactured problem at the finest level ----
    VectorQ1Vec< ScalarType > u( "u", domains[fine], mask[fine] );
    VectorQ1Vec< ScalarType > g( "g", domains[fine], mask[fine] );
    VectorQ1Vec< ScalarType > rhs_tmp( "rhs_tmp", domains[fine], mask[fine] );
    VectorQ1Vec< ScalarType > solution( "solution", domains[fine], mask[fine] );
    VectorQ1Vec< ScalarType > error( "error", domains[fine], mask[fine] );
    VectorQ1Vec< ScalarType > b( "b", domains[fine], mask[fine] );

    using Mass = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;
    Mass M( domains[fine], coords[fine], radii[fine], false );

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domains[fine] ),
        SolutionInterpolator( coords[fine], radii[fine], solution.grid_data(), bmask[fine], false ) );
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domains[fine] ),
        SolutionInterpolator( coords[fine], radii[fine], g.grid_data(), bmask[fine], true ) );
    Kokkos::fence();
    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domains[fine] ),
        RHSInterpolator( coords[fine], radii[fine], rhs_tmp.grid_data(), vcfg ) );
    Kokkos::fence();

    linalg::apply( M, rhs_tmp, b );
    fe::strong_algebraic_dirichlet_enforcement_vectorlaplace_like(
        A_neumann, A_neumann_diag, g, rhs_tmp, b, bmask[fine], BOUNDARY );
    Kokkos::fence();

    // Flexible GMRES outer solver (the MG V-cycle with a tolerance-based coarse
    // solve is a variable/non-symmetric preconditioner, so FGMRES is required --
    // same outer/preconditioner pairing as the Stokes A-block).
    constexpr int                            restart = 50;
    std::vector< VectorQ1Vec< ScalarType > > fgmres_tmps;
    for ( int i = 0; i < 2 * restart + 4; ++i )
    {
        fgmres_tmps.emplace_back( "fgmres_tmp_" + std::to_string( i ), domains[fine], mask[fine] );
    }
    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                     = restart;
    fgmres_options.max_iterations              = 300;
    fgmres_options.relative_residual_tolerance = 1e-9;

    const std::string solver_tag = std::string( "fgmres_" ) + config_name( cfg ) + "_level_" + std::to_string( level );
    linalg::solvers::FGMRES< Epsilon, PrecMG > fgmres( fgmres_tmps, fgmres_options, table, prec );
    fgmres.set_tag( solver_tag );

    Kokkos::Timer timer;
    timer.reset();
    linalg::solvers::solve( fgmres, A, u, b );
    Kokkos::fence();
    const auto time_solver = timer.seconds();

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error = std::sqrt( dot( error, error ) / num_dofs );

    const auto n_iters = static_cast< int >( table->query_rows_equals( "tag", solver_tag ).rows().size() );

    table->add_row( { { "config", std::string( config_name( cfg ) ) },
                      { "level", level },
                      { "dofs", num_dofs },
                      { "l2_error", l2_error },
                      { "iters", n_iters },
                      { "time_solver", time_solver },
                      { "h", ( 1.0 - 0.5 ) / std::pow( 2, level ) } } );

    return l2_error;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int         max_level     = 4;
    int         min_level     = 2;
    std::string visc_profile  = "smooth"; // constant | smooth | tanh
    double      visc_base     = 1.0;
    double      visc_contrast = 1000.0;
    double      visc_rmid     = 0.75;
    double      visc_width    = 0.05;
    {
        CLI::App app{ "qp_comparison_test" };
        util::add_option_with_default( app, "--max-level", max_level, "Finest refinement level." );
        util::add_option_with_default( app, "--min-level", min_level, "Coarsest refinement level." );
        util::add_option_with_default(
            app, "--visc-profile", visc_profile, "Viscosity profile: constant | smooth | tanh." );
        util::add_option_with_default(
            app, "--visc-base", visc_base, "Base viscosity (constant value, or eta_low for tanh)." );
        util::add_option_with_default(
            app, "--visc-contrast", visc_contrast, "Viscosity contrast eta_high/eta_low for the tanh profile." );
        util::add_option_with_default( app, "--visc-rmid", visc_rmid, "Radius of the tanh viscosity jump." );
        util::add_option_with_default( app, "--visc-width", visc_width, "Width of the tanh viscosity transition." );
        CLI11_PARSE( app, argc, argv );
    }

    ViscConfig vcfg;
    vcfg.base     = visc_base;
    vcfg.contrast = visc_contrast;
    vcfg.r_mid    = visc_rmid;
    vcfg.width    = visc_width;
    if ( visc_profile == "constant" )
    {
        vcfg.profile = static_cast< int >( ViscProfileKind::Constant );
    }
    else if ( visc_profile == "tanh" )
    {
        vcfg.profile = static_cast< int >( ViscProfileKind::Tanh );
    }
    else
    {
        if ( visc_profile != "smooth" )
        {
            util::logroot << "Unknown --visc-profile '" << visc_profile << "', using 'smooth'.\n";
        }
        vcfg.profile = static_cast< int >( ViscProfileKind::Smooth );
    }
    util::logroot << "Viscosity profile: " << visc_profile << " (base=" << vcfg.base << ", contrast=" << vcfg.contrast
                  << ", r_mid=" << vcfg.r_mid << ", width=" << vcfg.width << ")\n";

    auto table = std::make_shared< util::Table >();

    const std::vector< QpConfig > configs = {
        QpConfig::WedgeQp1, QpConfig::WedgeQp6, QpConfig::HexQp1, QpConfig::HexQp8 };

    for ( const auto cfg : configs )
    {
        util::logroot << "\n================ config: " << config_name( cfg ) << " ================\n";
        double prev_err = 0.0;
        for ( int level = min_level; level <= max_level; ++level )
        {
            const double err = test( level, 0, cfg, vcfg, table );
            if ( level > min_level && err > 0.0 )
            {
                util::logroot << config_name( cfg ) << " level " << level << ": l2_error=" << err
                              << " order_ratio=" << ( prev_err / err ) << "\n";
                table->add_row(
                    { { "config", std::string( config_name( cfg ) ) }, { "level", level }, { "order", prev_err / err } } );
            }
            else
            {
                util::logroot << config_name( cfg ) << " level " << level << ": l2_error=" << err << "\n";
            }
            prev_err = err;
        }
    }

    util::logroot << "\n";
    table->query_rows_not_none( "l2_error" )
        .select_columns( { "config", "level", "dofs", "h", "l2_error", "iters", "time_solver" } )
        .print_pretty();
    table->query_rows_not_none( "order" ).select_columns( { "config", "level", "order" } ).print_pretty();

    return 0;
}
