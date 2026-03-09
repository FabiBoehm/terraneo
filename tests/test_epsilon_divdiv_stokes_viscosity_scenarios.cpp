
/// @brief Test comparing Stokes FGMRES convergence for EpsDivDivStokes with different
///        viscosity scenarios. Reuses the coefficient scenarios from the DivKGrad
///        interpolation modes test, but solves the full Stokes system (FGMRES +
///        block-triangular preconditioner with multigrid on Block 11).
///        Reports FGMRES iteration counts for each scenario.

#include <numeric>

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp"

#include "io/xdmf.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/gca/gca_elements_collector.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/radial_line_smoother.hpp"
#include "linalg/solvers/pcg.hpp"

#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "terra/shell/radial_profiles.hpp"
#include "terra/mpi/mpi.hpp"

#include "util/init.hpp"
#include "util/table.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::ShellBoundaryFlag;
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
using linalg::solvers::InterpolationMode;
using linalg::solvers::power_iteration;
using linalg::solvers::TwoGridGCA;
using terra::grid::shell::BoundaryConditions;

// =============================================================================
// Solution / RHS interpolators (from test_epsilon_divdiv_stokes.cpp)
// =============================================================================

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
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
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
            const double cx = coords( 0 );
            const double cy = coords( 1 );
            const double cz = coords( 2 );

            data_u_( local_subdomain_id, x, y, r, 0 ) = -4 * Kokkos::cos( 4 * cz );
            data_u_( local_subdomain_id, x, y, r, 1 ) =  8 * Kokkos::cos( 8 * cx );
            data_u_( local_subdomain_id, x, y, r, 2 ) = -2 * Kokkos::cos( 2 * cy );
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
    : grid_( grid )
    , radii_( radii )
    , data_p_( data_p )
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
            const double cx = coords( 0 );
            const double cy = coords( 1 );
            const double cz = coords( 2 );

            data_p_( local_subdomain_id, x, y, r ) =
                Kokkos::sin( 4 * cx ) * Kokkos::sin( 8 * cy ) * Kokkos::sin( 2 * cz );
        }
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const real_t x0 = 4 * coords( 2 );

        data_( local_subdomain_id, x, y, r, 0 ) =
            -64.0 * ( Kokkos::sin( coords( 2 ) ) + 2 ) * Kokkos::cos( x0 ) -
            16.0 * Kokkos::sin( x0 ) * Kokkos::cos( coords( 2 ) ) +
            4 * Kokkos::sin( 8 * coords( 1 ) ) * Kokkos::sin( 2 * coords( 2 ) ) * Kokkos::cos( 4 * coords( 0 ) );

        data_( local_subdomain_id, x, y, r, 1 ) =
            512.0 * ( Kokkos::sin( coords( 2 ) ) + 2 ) * Kokkos::cos( 8 * coords( 0 ) ) +
            8 * Kokkos::sin( 4 * coords( 0 ) ) * Kokkos::sin( 2 * coords( 2 ) ) * Kokkos::cos( 8 * coords( 1 ) ) -
            4.0 * Kokkos::sin( 2 * coords( 1 ) ) * Kokkos::cos( coords( 2 ) );

        data_( local_subdomain_id, x, y, r, 2 ) =
            -8.0 * ( Kokkos::sin( coords( 2 ) ) + 2 ) * Kokkos::cos( 2 * coords( 1 ) ) +
            2 * Kokkos::sin( 4 * coords( 0 ) ) * Kokkos::sin( 8 * coords( 1 ) ) * Kokkos::cos( 2 * coords( 2 ) );
    }
};

// =============================================================================
// Viscosity interpolators (from test_div_k_grad_gca_interpolation_modes.cpp)
// =============================================================================

/// @brief Simple k = 2 + sin(z) — the original Stokes test viscosity.
struct KInterpolatorSinZ
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KInterpolatorSinZ(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        data_( local_subdomain_id, x, y, r ) = 2 + Kokkos::sin( coords( 2 ) );
    }
};

/// @brief Radial tanh profile: k(r) = 1 + 0.5*(k_max-1)*(tanh(alpha*(r-r_mid)/(r_max-r_min)*2)+1)
struct KInterpolatorTanh
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    const double               alpha_;
    const double               r_min_;
    const double               r_max_;
    const double               k_max_;

    KInterpolatorTanh(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const double                      r_min,
        const double                      r_max,
        const double                      alpha,
        const double                      k_max )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , alpha_( alpha )
    , r_min_( r_min )
    , r_max_( r_max )
    , k_max_( k_max )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  rad    = coords.norm();
        const double                  x0     = 0.5 * r_max_;
        const double                  x1     = 0.5 * r_min_;
        data_( local_subdomain_id, x, y, r ) =
            1.0 + 0.5 * ( k_max_ - 1.0 ) * ( Kokkos::tanh( alpha_ * ( -x0 - x1 + rad ) / ( x0 - x1 ) ) + 1 );
    }
};

/// @brief Unaligned lateral step: k = k_max where 2x + 3y + 5z > 1, else 1.
struct KInterpolatorUnalignedLateralStep
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    const double               k_max_;

    KInterpolatorUnalignedLateralStep(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const double                      k_max )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , k_max_( k_max )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        data_( local_subdomain_id, x, y, r ) =
            ( 2.0 * coords( 0 ) + 3.0 * coords( 1 ) + 5.0 * coords( 2 ) > 1.0 ) ? k_max_ : 1.0;
    }
};

/// @brief Piecewise linear interpolation of a radial viscosity profile on device.
struct KInterpolatorRadialProfile
{
    Grid3DDataVec< double, 3 >    grid_;
    Grid2DDataScalar< double >    radii_;
    Grid4DDataScalar< double >    data_;
    Kokkos::View< const double* > profile_radii_;
    Kokkos::View< const double* > profile_values_;

    KInterpolatorRadialProfile(
        const Grid3DDataVec< double, 3 >&    grid,
        const Grid2DDataScalar< double >&    radii,
        const Grid4DDataScalar< double >&    data,
        const Kokkos::View< const double* >& profile_radii,
        const Kokkos::View< const double* >& profile_values )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , profile_radii_( profile_radii )
    , profile_values_( profile_values )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  rad    = coords.norm();
        const int                     n      = static_cast< int >( profile_radii_.extent( 0 ) );

        if ( rad <= profile_radii_( 0 ) )
        {
            data_( local_subdomain_id, x, y, r ) = profile_values_( 0 );
            return;
        }
        if ( rad >= profile_radii_( n - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = profile_values_( n - 1 );
            return;
        }

        int lo = 0, hi = n - 1;
        while ( lo + 1 < hi )
        {
            int mid = ( lo + hi ) / 2;
            if ( profile_radii_( mid ) <= rad )
                lo = mid;
            else
                hi = mid;
        }
        const double t                        = ( rad - profile_radii_( lo ) ) / ( profile_radii_( hi ) - profile_radii_( lo ) );
        data_( local_subdomain_id, x, y, r ) = profile_values_( lo ) + t * ( profile_values_( hi ) - profile_values_( lo ) );
    }
};

// =============================================================================
// Coefficient scenario types
// =============================================================================

enum class CoeffType
{
    SinZ,                 // k = 2 + sin(z) — original Stokes test (has analytical solution)
    Constant,             // k = 1
    Tanh,                 // radial tanh
    UnalignedLateralStep, // 2x + 3y + 5z > 1
    RadialProfile         // from CSV file
};

struct CoeffScenario
{
    std::string name;
    double      alpha;
    double      k_max;
    CoeffType   coeff_type;

    Kokkos::View< const double* > profile_radii;
    Kokkos::View< const double* > profile_values;
};

// =============================================================================
// Helper: interpolate viscosity at all nodes for a given level
// =============================================================================

void interpolate_k(
    const DistributedDomain&                    domain,
    const Grid3DDataVec< double, 3 >&           coords_shell,
    const Grid2DDataScalar< double >&           coords_radii,
    const Grid4DDataScalar< double >&           k_data,
    const CoeffScenario&                        scenario,
    double                                      r_min,
    double                                      r_max )
{
    switch ( scenario.coeff_type )
    {
    case CoeffType::SinZ:
        Kokkos::parallel_for(
            "k interpolation (sin z)",
            local_domain_md_range_policy_nodes( domain ),
            KInterpolatorSinZ( coords_shell, coords_radii, k_data ) );
        break;
    case CoeffType::Constant:
        Kokkos::parallel_for(
            "k interpolation (constant)",
            local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                k_data( sd, x, y, r ) = 1.0;
            } );
        break;
    case CoeffType::Tanh:
        Kokkos::parallel_for(
            "k interpolation (tanh)",
            local_domain_md_range_policy_nodes( domain ),
            KInterpolatorTanh( coords_shell, coords_radii, k_data, r_min, r_max, scenario.alpha, scenario.k_max ) );
        break;
    case CoeffType::UnalignedLateralStep:
        Kokkos::parallel_for(
            "k interpolation (unaligned lateral step)",
            local_domain_md_range_policy_nodes( domain ),
            KInterpolatorUnalignedLateralStep( coords_shell, coords_radii, k_data, scenario.k_max ) );
        break;
    case CoeffType::RadialProfile:
        Kokkos::parallel_for(
            "k interpolation (radial profile)",
            local_domain_md_range_policy_nodes( domain ),
            KInterpolatorRadialProfile(
                coords_shell, coords_radii, k_data, scenario.profile_radii, scenario.profile_values ) );
        break;
    }
    Kokkos::fence();
}

// =============================================================================
// Main Stokes solve
// =============================================================================

/// @tparam ChebyOrder 0 = Jacobi with 3 sweeps, >=2 = Chebyshev of given order (1 sweep).
template < int ChebyOrder >
std::pair< int, double > run_stokes_solve(
    int                                      min_level,
    int                                      max_level,
    const CoeffScenario&                     scenario,
    linalg::solvers::InterpolationMode       interp_mode )
{
    using ScalarType = double;
    const double r_min = 0.5;
    const double r_max = 1.0;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                        coords_shell;
    std::vector< Grid2DDataScalar< double > >                        coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    const int num_levels    = max_level - min_level + 1;
    const int velocity_level = num_levels - 1;
    const int pressure_level = num_levels - 2;

    util::logroot << "Allocating domains ...\n";
    for ( int level = min_level; level <= max_level; level++ )
    {
        const int idx = level - min_level;
        domains.push_back(
            DistributedDomain::create_uniform( level, level, r_min, r_max, 0, 0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    // Stokes vectors.
    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    for ( const auto& name : { "u", "f", "solution", "error", "tmp_0", "tmp_1", "tmp_5", "tmp_6" } )
    {
        stok_vecs.emplace(
            std::piecewise_construct,
            std::forward_as_tuple( name ),
            std::forward_as_tuple(
                name,
                domains[velocity_level],
                domains[pressure_level],
                mask_data[velocity_level],
                mask_data[pressure_level] ) );
    }

    auto& u     = stok_vecs["u"];
    auto& f     = stok_vecs["f"];

    std::vector< VectorQ1Vec< ScalarType > > tmp_mg;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_r;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_e;

    for ( int level = 0; level < num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    // Operators.
    using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using Gradient    = Stokes::Block12Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    // Viscosity on finest level.
    VectorQ1Scalar< ScalarType > k( "k", domains[velocity_level], mask_data[velocity_level] );
    util::logroot << "Interpolating k ...\n";
    interpolate_k( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
                   k.grid_data(), scenario, r_min, r_max );

    // GCA elements: all elements.
    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", domains[0], mask_data[0] );
    linalg::assign( GCAElements, 1 );

    // Stokes operators.
    Stokes K(
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

    // MG hierarchy.
    std::vector< Viscous >      A_diag;
    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;

    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;

    util::logroot << "MG hierarchy ...\n";
    for ( int level = 0; level < num_levels; level++ )
    {
        VectorQ1Scalar< ScalarType > k_c( "k_c", domains[level], mask_data[level] );
        interpolate_k( domains[level], coords_shell[level], coords_radii[level],
                       k_c.grid_data(), scenario, r_min, r_max );

        A_diag.emplace_back(
            domains[level], coords_shell[level], coords_radii[level],
            boundary_mask_data[level], k_c.grid_data(), bcs, true );

        if ( level < num_levels - 1 )
        {
            A_c.emplace_back(
                domains[level], coords_shell[level], coords_radii[level],
                boundary_mask_data[level], k_c.grid_data(), bcs, false );

            A_c.back().set_stored_matrix_mode(
                linalg::OperatorStoredMatrixMode::Full, level, GCAElements.grid_data() );

            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // RHS and boundary conditions.
    const bool has_analytical_solution = ( scenario.coeff_type == CoeffType::SinZ );

    if ( has_analytical_solution )
    {
        Kokkos::parallel_for(
            "solution interpolation (velocity)",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            SolutionVelocityInterpolator(
                coords_shell[velocity_level], coords_radii[velocity_level],
                stok_vecs["solution"].block_1().grid_data(),
                boundary_mask_data[velocity_level], false ) );

        Kokkos::parallel_for(
            "solution interpolation (pressure)",
            local_domain_md_range_policy_nodes( domains[pressure_level] ),
            SolutionPressureInterpolator(
                coords_shell[pressure_level], coords_radii[pressure_level],
                stok_vecs["solution"].block_2().grid_data(),
                boundary_mask_data[pressure_level], false ) );

        Kokkos::parallel_for(
            "rhs interpolation",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            RHSVelocityInterpolator(
                coords_shell[velocity_level], coords_radii[velocity_level],
                stok_vecs["tmp_1"].block_1().grid_data() ) );

        linalg::apply( M, stok_vecs["tmp_1"].block_1(), stok_vecs["f"].block_1() );

        Kokkos::parallel_for(
            "boundary interpolation (velocity)",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            SolutionVelocityInterpolator(
                coords_shell[velocity_level], coords_radii[velocity_level],
                stok_vecs["tmp_0"].block_1().grid_data(),
                boundary_mask_data[velocity_level], true ) );

        fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
            K_neumann, K_neumann_diag,
            stok_vecs["tmp_0"], stok_vecs["tmp_1"], stok_vecs["f"],
            boundary_mask_data[velocity_level], BOUNDARY );
    }
    else
    {
        // No analytical solution: use f = M * 1 as RHS, zero Dirichlet BCs.
        VectorQ1Vec< ScalarType > ones( "ones", domains[velocity_level], mask_data[velocity_level] );
        linalg::assign( ones, 1.0 );
        linalg::apply( M, ones, f.block_1() );

        // Zero Dirichlet BCs: enforce u=0 on boundary.
        Kokkos::parallel_for(
            "zero boundary",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            SolutionVelocityInterpolator(
                coords_shell[velocity_level], coords_radii[velocity_level],
                stok_vecs["tmp_0"].block_1().grid_data(),
                boundary_mask_data[velocity_level], true ) );

        // Actually set zero on boundary for tmp_0.
        linalg::assign( stok_vecs["tmp_0"], 0 );

        fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
            K_neumann, K_neumann_diag,
            stok_vecs["tmp_0"], stok_vecs["tmp_1"], stok_vecs["f"],
            boundary_mask_data[velocity_level], BOUNDARY );
    }

    // Assemble GCA coarse operators.
    for ( int level = num_levels - 2; level >= 0; level-- )
    {
        util::logroot << "Assembling GCA on level " << level << "\n";
        TwoGridGCA< ScalarType, Viscous >(
            ( level == num_levels - 2 ) ? K_neumann.block_11() : A_c[level + 1],
            A_c[level], level, GCAElements.grid_data(), true, interp_mode );
    }

    // Setup smoothers.
    using Smoother = std::conditional_t< ChebyOrder == -1,
                                         linalg::solvers::HybridJacobiRadialLineSmoother< Viscous >,
                                         std::conditional_t< ChebyOrder <= 1,
                                             linalg::solvers::Jacobi< Viscous >,
                                             linalg::solvers::Chebyshev< Viscous > > >;

    std::vector< Smoother > smoothers;
    for ( int level = 0; level < num_levels; level++ )
    {
        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > tmp_id(
            "inverse_diagonal_tmp" + std::to_string( level ), domains[level], mask_data[level] );
        linalg::assign( tmp_id, 1.0 );

        if ( level == num_levels - 1 )
        {
            K.block_11().set_diagonal( true );
            linalg::apply( K.block_11(), tmp_id, inverse_diagonals.back() );
            K.block_11().set_diagonal( false );
        }
        else
        {
            A_c[level].set_diagonal( true );
            linalg::apply( A_c[level], tmp_id, inverse_diagonals.back() );
            A_c[level].set_diagonal( false );
        }

        linalg::invert_entries( inverse_diagonals.back() );

        if constexpr ( ChebyOrder == -1 )
        {
            VectorQ1Vec< ScalarType > tmp_pi_0(
                "tmp_pi_0" + std::to_string( level ), domains[level], mask_data[level] );
            VectorQ1Vec< ScalarType > tmp_pi_1(
                "tmp_pi_1" + std::to_string( level ), domains[level], mask_data[level] );

            double max_ev = 0.0;
            if ( level == num_levels - 1 )
            {
                DiagonallyScaledOperator< Viscous > inv_diag_A( K.block_11(), inverse_diagonals[level] );
                max_ev =
                    power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
            }
            else
            {
                DiagonallyScaledOperator< Viscous > inv_diag_A( A_c[level], inverse_diagonals[level] );
                max_ev =
                    power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
            }
            const auto omega_opt = 2.0 / ( 1.3 * max_ev );

            util::logroot << "Hybrid Jacobi+RadLine on level " << level << " (omega_j=" << omega_opt << ")\n";
            if ( level == num_levels - 1 )
                smoothers.emplace_back(
                    K_neumann.block_11(), inverse_diagonals[level], 1, tmp_mg[level], omega_opt );
            else
                smoothers.emplace_back(
                    A_c[level], inverse_diagonals[level], 1, tmp_mg[level], omega_opt );
        }
        else if constexpr ( ChebyOrder <= 1 )
        {
            constexpr auto smoother_prepost = 3;
            VectorQ1Vec< ScalarType > tmp_pi_0(
                "tmp_pi_0" + std::to_string( level ), domains[level], mask_data[level] );
            VectorQ1Vec< ScalarType > tmp_pi_1(
                "tmp_pi_1" + std::to_string( level ), domains[level], mask_data[level] );

            double max_ev = 0.0;
            if ( level == num_levels - 1 )
            {
                DiagonallyScaledOperator< Viscous > inv_diag_A( K.block_11(), inverse_diagonals[level] );
                max_ev =
                    power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
            }
            else
            {
                DiagonallyScaledOperator< Viscous > inv_diag_A( A_c[level], inverse_diagonals[level] );
                max_ev =
                    power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
            }

            const auto omega_opt = 2.0 / ( 1.3 * max_ev );
            smoothers.emplace_back( inverse_diagonals[level], smoother_prepost, tmp_mg[level], omega_opt );
            util::logroot << "Jacobi omega on level " << level << ": " << omega_opt << "\n";
        }
        else
        {
            std::vector< VectorQ1Vec< ScalarType > > cheby_tmps;
            cheby_tmps.emplace_back( "cheby_tmp0_" + std::to_string( level ), domains[level], mask_data[level] );
            cheby_tmps.emplace_back( "cheby_tmp1_" + std::to_string( level ), domains[level], mask_data[level] );
            smoothers.emplace_back( ChebyOrder, inverse_diagonals[level], cheby_tmps, 1 );
            util::logroot << "Chebyshev order " << ChebyOrder << " on level " << level << "\n";
        }
    }

    // Coarse grid solver.
    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;

    auto table = std::make_shared< util::Table >();

    std::vector< VectorQ1Vec< ScalarType > > coarse_grid_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        coarse_grid_tmps.emplace_back( "tmp_coarse_grid", domains[0], mask_data[0] );
    }

    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-6, 1e-16 }, table, coarse_grid_tmps );

    constexpr auto num_mg_cycles = 1;

    using PrecVisc =
        linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;

    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, num_mg_cycles, 1e-8 );

    // Pressure mass preconditioner.
    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[max_level - min_level], mask_data[max_level - min_level] );
    linalg::assign( k_pm, k );
    linalg::invert_entries( k_pm );

    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;
    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level], k_pm.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    VectorQ1Scalar< ScalarType > lumped_diagonal_pmass(
        "lumped_diagonal_pmass", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > tmp_pm(
            "tmp_pm", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( tmp_pm, 1.0 );
        linalg::apply( pmass, tmp_pm, lumped_diagonal_pmass );
    }

    using PrecSchur = linalg::solvers::DiagonalSolver< PressureMass >;
    PrecSchur inv_lumped_pmass( lumped_diagonal_pmass );

    using PrecStokes =
        linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

    VectorQ1IsoQ2Q1< ScalarType > triangular_prec_tmp(
        "triangular_prec_tmp",
        domains[velocity_level], domains[pressure_level],
        mask_data[velocity_level], mask_data[pressure_level] );

    PrecStokes prec_stokes( K.block_11(), pmass, K.block_12(), triangular_prec_tmp, prec_11, inv_lumped_pmass );

    // FGMRES solver.
    const int iters = 200;

    constexpr auto                               num_tmps_fgmres = 200;
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < 2 * num_tmps_fgmres + 4; ++i )
    {
        tmp_fgmres.emplace_back(
            "tmp_" + std::to_string( i ),
            domains[velocity_level], domains[pressure_level],
            mask_data[velocity_level], mask_data[pressure_level] );
    }

    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                     = iters;
    fgmres_options.max_iterations              = iters;
    fgmres_options.relative_residual_tolerance = 1e-10;

    auto solver_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( tmp_fgmres, fgmres_options, solver_table, prec_stokes );

    // Solve with timing.
    util::logroot << "Solve ...\n";
    linalg::assign( u, 0 );
    Kokkos::fence();
    Kokkos::Timer timer;
    linalg::solvers::solve( fgmres, K, u, f );
    Kokkos::fence();
    const double elapsed = timer.seconds();

    solver_table->query_rows_equals( "tag", "fgmres_solver" )
        .select_columns( { "absolute_residual", "relative_residual", "iteration" } )
        .print_pretty();

    const int iterations =
        static_cast< int >( solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows().size() );

    return { iterations, elapsed };
}

// =============================================================================

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int min_level = 2;
    const int max_level = 5;

    // Load radial viscosity profiles from CSV files.
    const std::string data_dir = std::string( TERRANEO_SOURCE_DIR ) + "/data/radialprofiles/";

    auto load_profile = []( const std::string& filename, const std::string& key_radii,
                            const std::string& key_values )
        -> std::pair< Kokkos::View< const double* >, Kokkos::View< const double* > >
    {
        auto table_result = util::read_table_from_csv( filename );
        if ( table_result.is_err() )
        {
            std::cerr << "Error reading CSV: " << table_result.error() << std::endl;
            Kokkos::abort( "Failed to read radial profile CSV." );
        }
        const auto& profile_table = table_result.unwrap();

        auto radii_vec  = profile_table.column_as_vector< double >( key_radii );
        auto values_vec = profile_table.column_as_vector< double >( key_values );

        std::vector< int > idx( radii_vec.size() );
        std::iota( idx.begin(), idx.end(), 0 );
        std::sort( idx.begin(), idx.end(), [&]( int a, int b ) { return radii_vec[a] < radii_vec[b]; } );

        const int n = static_cast< int >( idx.size() );

        Kokkos::View< double* > radii_dev( "profile_radii", n );
        Kokkos::View< double* > values_dev( "profile_values", n );
        auto                    radii_host  = Kokkos::create_mirror_view( radii_dev );
        auto                    values_host = Kokkos::create_mirror_view( values_dev );

        for ( int i = 0; i < n; i++ )
        {
            radii_host( i )  = radii_vec[idx[i]];
            values_host( i ) = values_vec[idx[i]];
        }

        Kokkos::deep_copy( radii_dev, radii_host );
        Kokkos::deep_copy( values_dev, values_host );
        Kokkos::fence();

        return { radii_dev, values_dev };
    };

    auto [lin_radii, lin_values] =
        load_profile( data_dir + "ViscosityProfile_Lin_et_al_2022.csv",
                      "radius_normalized_0p5_1p0", "viscosity_scaled_by_min" );

    auto [stotz_radii, stotz_values] =
        load_profile( data_dir + "ViscosityProfile_Stotz_et_al_2017.csv",
                      "radius_normalized_0p5_1p0", "viscosity_scaled_by_min" );

    if ( mpi::rank() == 0 )
    {
        std::cout << "Loaded Lin et al. 2022 profile: " << lin_radii.extent( 0 ) << " points" << std::endl;
        std::cout << "Loaded Stotz et al. 2017 profile: " << stotz_radii.extent( 0 ) << " points" << std::endl;
    }

    std::vector< CoeffScenario > scenarios = {
        { "k = 2 + sin(z) (original Stokes)", 0.0, 0.0, CoeffType::SinZ, {}, {} },
        { "Constant k=1", 0.0, 1.0, CoeffType::Constant, {}, {} },
        { "Tanh k_max=10, alpha=1", 1.0, 10.0, CoeffType::Tanh, {}, {} },
        { "Unaligned lateral step k_max=10", 0.0, 10.0, CoeffType::UnalignedLateralStep, {}, {} },
        { "Unaligned lateral step k_max=100", 0.0, 100.0, CoeffType::UnalignedLateralStep, {}, {} },
        { "Lin et al. 2022", 0.0, 0.0, CoeffType::RadialProfile, lin_radii, lin_values },
        { "Stotz et al. 2017", 0.0, 0.0, CoeffType::RadialProfile, stotz_radii, stotz_values },
    };

    // Smoother configurations: { name, cheby_order (0=Jacobi-3) }
    struct SmootherConfig
    {
        std::string name;
        int         cheby_order;
    };

    std::vector< SmootherConfig > smoother_configs = {
        { "Jacobi-3", 0 },
        { "Cheby-4", 4 },
        { "RadLine", -1 },
    };

    // Dispatch helper: template instantiation for each Chebyshev order.
    auto dispatch_solve = []( int min_lev, int max_lev, const CoeffScenario& sc,
                              InterpolationMode im, int cheby_order ) -> std::pair< int, double >
    {
        switch ( cheby_order )
        {
        case -1: return run_stokes_solve< -1 >( min_lev, max_lev, sc, im );
        case 0:  return run_stokes_solve< 0 >( min_lev, max_lev, sc, im );
        case 2:  return run_stokes_solve< 2 >( min_lev, max_lev, sc, im );
        case 3:  return run_stokes_solve< 3 >( min_lev, max_lev, sc, im );
        case 4:  return run_stokes_solve< 4 >( min_lev, max_lev, sc, im );
        default: return { -1, 0.0 };
        }
    };

    const auto interp_mode = InterpolationMode::Constant;

    if ( mpi::rank() == 0 )
    {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "  EpsDivDivStokes: Smoother Comparison (Constant interp)" << std::endl;
        std::cout << "  levels: " << min_level << " - " << max_level << std::endl;
        std::cout << "============================================================\n" << std::endl;
    }

    // results[scenario_idx][smoother_idx] = { iterations, time }
    const size_t ns = scenarios.size();
    const size_t nm = smoother_configs.size();
    std::vector< std::vector< int > >    iter_results( ns, std::vector< int >( nm, -1 ) );
    std::vector< std::vector< double > > time_results( ns, std::vector< double >( nm, 0.0 ) );

    for ( size_t si = 0; si < ns; si++ )
    {
        const auto& scenario = scenarios[si];

        if ( mpi::rank() == 0 )
        {
            std::cout << "\n========================================================" << std::endl;
            std::cout << "  SCENARIO: " << scenario.name << std::endl;
            std::cout << "========================================================" << std::endl;
        }

        for ( size_t mi = 0; mi < nm; mi++ )
        {
            if ( mpi::rank() == 0 )
                std::cout << "\n  --- " << smoother_configs[mi].name << " ---\n" << std::endl;

            auto [iterations, elapsed] =
                dispatch_solve( min_level, max_level, scenario, interp_mode, smoother_configs[mi].cheby_order );

            if ( mpi::rank() == 0 )
                std::cout << "  => FGMRES iterations: " << iterations << "  time: " << std::fixed
                          << std::setprecision( 2 ) << elapsed << " s" << std::endl;

            iter_results[si][mi] = iterations;
            time_results[si][mi] = elapsed;
        }
    }

    // Summary tables.
    if ( mpi::rank() == 0 )
    {
        std::cout << "\n\n============================================================" << std::endl;
        std::cout << "  ITERATION COUNT SUMMARY" << std::endl;
        std::cout << "============================================================\n" << std::endl;

        std::cout << std::setw( 45 ) << std::left << "Scenario";
        for ( size_t mi = 0; mi < nm; mi++ )
            std::cout << std::setw( 12 ) << smoother_configs[mi].name;
        std::cout << std::endl;
        std::cout << std::string( 45 + 12 * nm, '-' ) << std::endl;

        for ( size_t si = 0; si < ns; si++ )
        {
            std::cout << std::setw( 45 ) << std::left << scenarios[si].name;
            for ( size_t mi = 0; mi < nm; mi++ )
                std::cout << std::setw( 12 ) << iter_results[si][mi];
            std::cout << std::endl;
        }

        std::cout << "\n\n============================================================" << std::endl;
        std::cout << "  SOLVE TIME SUMMARY (seconds)" << std::endl;
        std::cout << "============================================================\n" << std::endl;

        std::cout << std::setw( 45 ) << std::left << "Scenario";
        for ( size_t mi = 0; mi < nm; mi++ )
            std::cout << std::setw( 12 ) << smoother_configs[mi].name;
        std::cout << std::endl;
        std::cout << std::string( 45 + 12 * nm, '-' ) << std::endl;

        for ( size_t si = 0; si < ns; si++ )
        {
            std::cout << std::setw( 45 ) << std::left << scenarios[si].name;
            for ( size_t mi = 0; mi < nm; mi++ )
                std::cout << std::setw( 12 ) << std::fixed << std::setprecision( 2 ) << time_results[si][mi];
            std::cout << std::endl;
        }

        std::cout << "\nPASS: All scenarios completed." << std::endl;
    }

    return EXIT_SUCCESS;
}
