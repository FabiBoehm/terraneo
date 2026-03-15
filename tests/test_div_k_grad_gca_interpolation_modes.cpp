

/// @brief Test comparing multigrid convergence rates for all GCA interpolation modes:
///        Constant, Linear, and OperatorDependent (AMG-style).
///        Runs a DivKGrad problem with variable coefficient on a spherical shell
///        and reports convergence rates and MG cycle counts for each mode.
///        Supports both FE-interpolated and exact coefficient evaluation modes.

#include <numeric>

#include <fe/wedge/operators/shell/restriction_linear.hpp>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/div_k_grad.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/gca/gca_elements_collector.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "terra/shell/radial_profiles.hpp"
#include "terra/mpi/mpi.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::solvers::InterpolationMode;
using linalg::solvers::TwoGridGCA;
using terra::linalg::DiagonallyScaledOperator;
using terra::linalg::solvers::power_iteration;

using fe::wedge::operators::shell::NoCoeffEval;

template < std::floating_point T >
struct SolutionInterpolator
{
    Grid3DDataVec< T, 3 > grid_;
    Grid2DDataScalar< T > radii_;
    Grid4DDataScalar< T > data_;
    bool                  only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< T, 3 >& grid,
        const Grid2DDataScalar< T >& radii,
        const Grid4DDataScalar< T >& data,
        bool                         only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< T, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const T                  value   = ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

struct RHSInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    const double               alpha_;
    const double               r_min_;
    const double               r_max_;
    const double               k_max_;

    RHSInterpolator(
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
        const double                  x0     = Kokkos::sinh( coords( 1 ) );
        const double                  x1     = 2 * coords( 0 );
        const double                  x2     = Kokkos::sin( x1 );
        const double                  x3     = 0.5 * r_max_;
        const double                  x4     = 0.5 * r_min_;
        const double                  x5     = Kokkos::sqrt(
            Kokkos::pow( coords( 0 ), 2 ) + Kokkos::pow( coords( 1 ), 2 ) + Kokkos::pow( coords( 2 ), 2 ) );
        const double x6                      = alpha_ / ( x3 - x4 );
        const double x7                      = Kokkos::tanh( x6 * ( -x3 - x4 + x5 ) );
        const double km1h                    = 0.5 * ( k_max_ - 1.0 );
        const double x9                      = x6 * ( 1 - Kokkos::pow( x7, 2 ) ) / x5;
        data_( local_subdomain_id, x, y, r ) = -0.5 * km1h * x2 * x9 * coords( 1 ) * Kokkos::cosh( coords( 1 ) ) -
                                               coords( 0 ) * x0 * km1h * x9 * Kokkos::cos( x1 ) +
                                               1.5 * x0 * x2 * ( 1.0 + km1h * ( x7 + 1 ) );
    }
};

/// @brief RHS for constant-coefficient case: -laplacian(u) where u = 0.5*sin(2x)*sinh(y).
/// laplacian(u) = d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2
///              = 0.5*(-4*sin(2x)*sinh(y)) + 0.5*(sin(2x)*sinh(y)) + 0
///              = 0.5*sin(2x)*sinh(y)*(-4+1) = -1.5*sin(2x)*sinh(y)
/// So f = -laplacian(u) = 1.5*sin(2x)*sinh(y).
struct RHSInterpolatorConstantK
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    RHSInterpolatorConstantK(
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
        data_( local_subdomain_id, x, y, r ) =
            1.5 * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
    }
};

// ============================================================================
// KInterpolator structs: write coefficient values to grid nodes (for FE interpolation mode).
// ============================================================================

struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    const double               alpha_;
    const double               r_min_;
    const double               r_max_;
    const double               k_max_;

    KInterpolator(
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

        const double rad                     = coords.norm();
        const double x0                      = 0.5 * r_max_;
        const double x1                      = 0.5 * r_min_;
        data_( local_subdomain_id, x, y, r ) =
            1.0 + 0.5 * ( k_max_ - 1.0 ) * ( Kokkos::tanh( alpha_ * ( -x0 - x1 + rad ) / ( x0 - x1 ) ) + 1 );
    }
};

struct KInterpolatorRadialStep
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    const double               r_mid_;
    const double               k_max_;

    KInterpolatorRadialStep(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const double                      r_min,
        const double                      r_max,
        const double                      k_max )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , r_mid_( 0.5 * ( r_min + r_max ) )
    , k_max_( k_max )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  rad    = coords.norm();
        data_( local_subdomain_id, x, y, r ) = ( rad > r_mid_ ) ? k_max_ : 1.0;
    }
};

struct KInterpolatorLateralStep
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    const double               k_max_;

    KInterpolatorLateralStep(
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
        data_( local_subdomain_id, x, y, r ) = ( coords( 2 ) > 0.0 ) ? k_max_ : 1.0;
    }
};

struct KInterpolatorUnalignedRadialStep
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    const double               r_thresh_;
    const double               k_max_;

    KInterpolatorUnalignedRadialStep(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const double                      r_min,
        const double                      r_max,
        const double                      k_max )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , r_thresh_( r_min + ( r_max - r_min ) / M_PI )
    , k_max_( k_max )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  rad    = coords.norm();
        data_( local_subdomain_id, x, y, r ) = ( rad > r_thresh_ ) ? k_max_ : 1.0;
    }
};

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

// ============================================================================
// CoeffEval structs: evaluate coefficient exactly at physical coordinates (for exact evaluation mode).
// These are device-callable and trivially copyable for Kokkos GPU execution.
// ============================================================================

struct CoeffEvalConstant
{
    KOKKOS_INLINE_FUNCTION
    double operator()( const dense::Vec< double, 3 >& ) const { return 1.0; }
};

struct CoeffEvalTanh
{
    double alpha_;
    double r_min_;
    double r_max_;
    double k_max_;

    KOKKOS_INLINE_FUNCTION
    double operator()( const dense::Vec< double, 3 >& x ) const
    {
        const double rad = x.norm();
        const double x0  = 0.5 * r_max_;
        const double x1  = 0.5 * r_min_;
        return 1.0 + 0.5 * ( k_max_ - 1.0 ) * ( Kokkos::tanh( alpha_ * ( -x0 - x1 + rad ) / ( x0 - x1 ) ) + 1 );
    }
};

struct CoeffEvalRadialStep
{
    double r_mid_;
    double k_max_;

    KOKKOS_INLINE_FUNCTION
    double operator()( const dense::Vec< double, 3 >& x ) const { return ( x.norm() > r_mid_ ) ? k_max_ : 1.0; }
};

struct CoeffEvalLateralStep
{
    double k_max_;

    KOKKOS_INLINE_FUNCTION
    double operator()( const dense::Vec< double, 3 >& x ) const { return ( x( 2 ) > 0.0 ) ? k_max_ : 1.0; }
};

struct CoeffEvalUnalignedRadialStep
{
    double r_thresh_;
    double k_max_;

    KOKKOS_INLINE_FUNCTION
    double operator()( const dense::Vec< double, 3 >& x ) const { return ( x.norm() > r_thresh_ ) ? k_max_ : 1.0; }
};

struct CoeffEvalUnalignedLateralStep
{
    double k_max_;

    KOKKOS_INLINE_FUNCTION
    double operator()( const dense::Vec< double, 3 >& x ) const
    {
        return ( 2.0 * x( 0 ) + 3.0 * x( 1 ) + 5.0 * x( 2 ) > 1.0 ) ? k_max_ : 1.0;
    }
};

/// @brief Evaluates a radial viscosity profile via piecewise linear interpolation on device.
/// Stores sorted (ascending) radii and values in Kokkos::Views (trivially copyable).
struct CoeffEvalRadialProfile
{
    Kokkos::View< const double* > radii_;
    Kokkos::View< const double* > values_;

    KOKKOS_INLINE_FUNCTION
    double operator()( const dense::Vec< double, 3 >& x ) const
    {
        const double r   = x.norm();
        const int    n   = static_cast< int >( radii_.extent( 0 ) );

        // Clamp below / above.
        if ( r <= radii_( 0 ) )
            return values_( 0 );
        if ( r >= radii_( n - 1 ) )
            return values_( n - 1 );

        // Binary search for the interval [i, i+1] containing r.
        int lo = 0, hi = n - 1;
        while ( lo + 1 < hi )
        {
            int mid = ( lo + hi ) / 2;
            if ( radii_( mid ) <= r )
                lo = mid;
            else
                hi = mid;
        }

        // Linear interpolation.
        const double t = ( r - radii_( lo ) ) / ( radii_( hi ) - radii_( lo ) );
        return values_( lo ) + t * ( values_( hi ) - values_( lo ) );
    }
};

/// @brief Writes radial profile values to grid nodes (for FE interpolation mode).
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

// ============================================================================

enum class CoeffType
{
    Constant,
    Tanh,
    RadialStep,
    LateralStep,
    UnalignedRadialStep,
    UnalignedLateralStep,
    RadialProfile
};

/// @brief Runs a single multigrid solve for DivKGrad with GCA using the specified interpolation mode.
/// @tparam CoeffEval Coefficient evaluator type. Use NoCoeffEval for FE interpolation mode,
///                   or a device-callable functor for exact evaluation mode.
/// @return A pair of (number of MG cycles, L2 error). L2 error is -1 when no analytical solution is available.
template < std::floating_point T, typename CoeffEval = NoCoeffEval >
std::pair< int, T > run_multigrid_solve(
    int                                           min_level,
    int                                           max_level,
    const std::shared_ptr< util::Table >&         table,
    int                                           prepost_smooth,
    double                                        alpha,
    double                                        k_max,
    InterpolationMode                             interp_mode,
    CoeffType                                     coeff_type,
    CoeffEval                                     coeff_eval   = {} )
{
    using ScalarType = T;
    const double r_min = 0.5;
    const double r_max = 1.0;

    static constexpr bool exact_eval = !std::is_same_v< CoeffEval, NoCoeffEval >;

    using DivKGrad     = fe::wedge::operators::shell::DivKGrad< ScalarType, CoeffEval >;
    using Prolongation = fe::wedge::operators::shell::ProlongationConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionConstant< ScalarType >;
    using Smoother     = linalg::solvers::Jacobi< DivKGrad >;
    using CoarseGridSolver = linalg::solvers::PCG< DivKGrad >;

    std::vector< DistributedDomain >              domains;
    std::vector< Grid3DDataVec< ScalarType, 3 > > subdomain_shell_coords;
    std::vector< Grid2DDataScalar< ScalarType > > subdomain_radii;

    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    std::vector< VectorQ1Scalar< ScalarType > > tmp_r_c;
    std::vector< VectorQ1Scalar< ScalarType > > tmp_e_c;
    std::vector< VectorQ1Scalar< ScalarType > > tmp;
    std::vector< DivKGrad >                     A_c;
    std::vector< Prolongation >                 P_additive;
    std::vector< Restriction >                  R;
    std::vector< Smoother >                     smoothers;
    std::vector< VectorQ1Scalar< ScalarType > > coarse_grid_tmps;

    for ( int level = 0; level <= max_level; level++ )
    {
        auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );
        domains.push_back( domain );
        subdomain_shell_coords.push_back(
            terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain ) );
        subdomain_radii.push_back( terra::grid::shell::subdomain_shell_radii< ScalarType >( domain ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domain ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domain ) );
    }

    // Helper to construct a DivKGrad operator at a given level.
    auto make_operator = [&]( int level, bool treat_boundary, bool diagonal ) -> DivKGrad
    {
        if constexpr ( exact_eval )
        {
            return DivKGrad(
                domains[level],
                subdomain_shell_coords[level],
                subdomain_radii[level],
                boundary_mask_data[level],
                coeff_eval,
                treat_boundary,
                diagonal );
        }
        else
        {
            // FE interpolation mode: need k grid data at this level.
            // Caller must have set up k_grid before calling this.
            // This path is not currently used but kept for future use.
            Kokkos::abort( "FE interpolation path: use the overload with k grid data" );
            // Unreachable, but needed for compilation:
            return DivKGrad(
                domains[level],
                subdomain_shell_coords[level],
                subdomain_radii[level],
                boundary_mask_data[level],
                Grid4DDataScalar< ScalarType >(),
                treat_boundary,
                diagonal );
        }
    };

    // Helper to construct a DivKGrad operator with k grid data (FE interpolation mode).
    auto make_operator_fe = [&]( int level, const Grid4DDataScalar< ScalarType >& k_data,
                                 bool treat_boundary, bool diagonal ) -> DivKGrad
    {
        if constexpr ( !exact_eval )
        {
            return DivKGrad(
                domains[level],
                subdomain_shell_coords[level],
                subdomain_radii[level],
                boundary_mask_data[level],
                k_data,
                treat_boundary,
                diagonal );
        }
        else
        {
            // In exact eval mode, ignore k_data and use functor.
            (void) k_data;
            return make_operator( level, treat_boundary, diagonal );
        }
    };

    // Coefficient field on finest level (only needed for FE interpolation mode).
    VectorQ1Scalar< ScalarType > k( "k", domains.back(), mask_data.back() );
    if constexpr ( !exact_eval )
    {
        if ( coeff_type == CoeffType::Constant )
        {
            linalg::assign( k, 1.0 );
        }
        else if ( coeff_type == CoeffType::RadialStep )
        {
            Kokkos::parallel_for(
                "coefficient interpolation (radial step)",
                local_domain_md_range_policy_nodes( domains.back() ),
                KInterpolatorRadialStep(
                    subdomain_shell_coords.back(), subdomain_radii.back(), k.grid_data(), r_min, r_max, k_max ) );
            Kokkos::fence();
        }
        else if ( coeff_type == CoeffType::LateralStep )
        {
            Kokkos::parallel_for(
                "coefficient interpolation (lateral step)",
                local_domain_md_range_policy_nodes( domains.back() ),
                KInterpolatorLateralStep(
                    subdomain_shell_coords.back(), subdomain_radii.back(), k.grid_data(), k_max ) );
            Kokkos::fence();
        }
        else if ( coeff_type == CoeffType::UnalignedRadialStep )
        {
            Kokkos::parallel_for(
                "coefficient interpolation (unaligned radial step)",
                local_domain_md_range_policy_nodes( domains.back() ),
                KInterpolatorUnalignedRadialStep(
                    subdomain_shell_coords.back(), subdomain_radii.back(), k.grid_data(), r_min, r_max, k_max ) );
            Kokkos::fence();
        }
        else if ( coeff_type == CoeffType::UnalignedLateralStep )
        {
            Kokkos::parallel_for(
                "coefficient interpolation (unaligned lateral step)",
                local_domain_md_range_policy_nodes( domains.back() ),
                KInterpolatorUnalignedLateralStep(
                    subdomain_shell_coords.back(), subdomain_radii.back(), k.grid_data(), k_max ) );
            Kokkos::fence();
        }
        else
        {
            Kokkos::parallel_for(
                "coefficient interpolation (tanh)",
                local_domain_md_range_policy_nodes( domains.back() ),
                KInterpolator(
                    subdomain_shell_coords.back(), subdomain_radii.back(), k.grid_data(), r_min, r_max, alpha,
                    k_max ) );
            Kokkos::fence();
        }
    }

    // Mark all elements for GCA.
    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", domains[min_level], mask_data[min_level] );
    linalg::assign( GCAElements, 1 );

    // Fine-level operators.
    DivKGrad A         = make_operator_fe( max_level, k.grid_data(), true, false );
    DivKGrad A_neumann = make_operator_fe( max_level, k.grid_data(), false, false );
    DivKGrad A_neumann_diag = make_operator_fe( max_level, k.grid_data(), false, true );

    // Build MG hierarchy.
    for ( int level = min_level; level <= max_level; level++ )
    {
        tmp.emplace_back( "tmp_level_" + std::to_string( level ), domains[level], mask_data[level] );

        if ( level == min_level )
        {
            constexpr int num_coarse_grid_tmps = 4;
            for ( int i = 0; i < num_coarse_grid_tmps; ++i )
            {
                coarse_grid_tmps.emplace_back(
                    "coarse_grid_tmps_" + std::to_string( i ), domains[level], mask_data[level] );
            }
        }

        if ( level < max_level )
        {
            tmp_r_c.emplace_back( "tmp_r_c_level_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_e_c.emplace_back( "tmp_e_c_level_" + std::to_string( level ), domains[level], mask_data[level] );

            VectorQ1Scalar< ScalarType > k_c( "k_c", domains[level], mask_data[level] );
            if constexpr ( !exact_eval )
            {
                if ( coeff_type == CoeffType::Constant )
                {
                    linalg::assign( k_c, 1.0 );
                }
                else if ( coeff_type == CoeffType::RadialStep )
                {
                    Kokkos::parallel_for(
                        "coefficient interpolation (radial step)",
                        local_domain_md_range_policy_nodes( domains[level] ),
                        KInterpolatorRadialStep(
                            subdomain_shell_coords[level], subdomain_radii[level], k_c.grid_data(),
                            r_min, r_max, k_max ) );
                    Kokkos::fence();
                }
                else if ( coeff_type == CoeffType::LateralStep )
                {
                    Kokkos::parallel_for(
                        "coefficient interpolation (lateral step)",
                        local_domain_md_range_policy_nodes( domains[level] ),
                        KInterpolatorLateralStep(
                            subdomain_shell_coords[level], subdomain_radii[level], k_c.grid_data(), k_max ) );
                    Kokkos::fence();
                }
                else if ( coeff_type == CoeffType::UnalignedRadialStep )
                {
                    Kokkos::parallel_for(
                        "coefficient interpolation (unaligned radial step)",
                        local_domain_md_range_policy_nodes( domains[level] ),
                        KInterpolatorUnalignedRadialStep(
                            subdomain_shell_coords[level], subdomain_radii[level], k_c.grid_data(),
                            r_min, r_max, k_max ) );
                    Kokkos::fence();
                }
                else if ( coeff_type == CoeffType::UnalignedLateralStep )
                {
                    Kokkos::parallel_for(
                        "coefficient interpolation (unaligned lateral step)",
                        local_domain_md_range_policy_nodes( domains[level] ),
                        KInterpolatorUnalignedLateralStep(
                            subdomain_shell_coords[level], subdomain_radii[level], k_c.grid_data(), k_max ) );
                    Kokkos::fence();
                }
                else
                {
                    Kokkos::parallel_for(
                        "coefficient interpolation (tanh)",
                        local_domain_md_range_policy_nodes( domains[level] ),
                        KInterpolator(
                            subdomain_shell_coords[level], subdomain_radii[level], k_c.grid_data(),
                            r_min, r_max, alpha, k_max ) );
                    Kokkos::fence();
                }
            }

            A_c.push_back( make_operator_fe( level, k_c.grid_data(), true, false ) );

            A_c.back().set_stored_matrix_mode(
                linalg::OperatorStoredMatrixMode::Full, level - min_level, GCAElements.grid_data() );

            P_additive.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // Assemble GCA coarse operators with the chosen interpolation mode.
    for ( int level = max_level - 1; level >= min_level; level-- )
    {
        if ( level == max_level - 1 )
        {
            TwoGridGCA< ScalarType, DivKGrad >(
                A_neumann, A_c[level - min_level], level - min_level, GCAElements.grid_data(), true, interp_mode );
        }
        else
        {
            TwoGridGCA< ScalarType, DivKGrad >(
                A_c[level + 1 - min_level], A_c[level - min_level], level - min_level, GCAElements.grid_data(),
                true, interp_mode );
        }
    }

    // Setup smoothers with power-iteration-based damping.
    for ( int level = min_level; level <= max_level; level++ )
    {
        VectorQ1Scalar< ScalarType > tmp_smoother(
            "tmp_smoothers_level_" + std::to_string( level ), domains[level], mask_data[level] );
        VectorQ1Scalar< ScalarType > inverse_diagonal(
            "inv_diag_level_" + std::to_string( level ), domains[level], mask_data[level] );
        VectorQ1Scalar< ScalarType > tmp_pi_0(
            "tmp_pi_0_level_" + std::to_string( level ), domains[level], mask_data[level] );
        VectorQ1Scalar< ScalarType > tmp_pi_1(
            "tmp_pi_1_level_" + std::to_string( level ), domains[level], mask_data[level] );

        assign( tmp_smoother, 1.0 );
        if ( level < max_level )
        {
            A_c[level - min_level].set_diagonal( true );
            apply( A_c[level - min_level], tmp_smoother, inverse_diagonal );
            A_c[level - min_level].set_diagonal( false );
        }
        else
        {
            A.set_diagonal( true );
            apply( A, tmp_smoother, inverse_diagonal );
            A.set_diagonal( false );
        }
        linalg::invert_entries( inverse_diagonal );

        smoothers.emplace_back( inverse_diagonal, prepost_smooth, tmp_smoother, 2.0 / 3.0 );
    }

    // Setup RHS and solution.
    VectorQ1Scalar< ScalarType > u( "u", domains.back(), mask_data.back() );
    VectorQ1Scalar< ScalarType > f( "f", domains.back(), mask_data.back() );
    VectorQ1Scalar< ScalarType > solution( "solution", domains.back(), mask_data.back() );
    VectorQ1Scalar< ScalarType > error( "error", domains.back(), mask_data.back() );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data.back(), grid::NodeOwnershipFlag::OWNED );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;
    Mass M( domains.back(), subdomain_shell_coords.back(), subdomain_radii.back(), false );

    const bool has_analytical_solution = ( coeff_type == CoeffType::Constant || coeff_type == CoeffType::Tanh );

    if ( has_analytical_solution )
    {
        Kokkos::parallel_for(
            "solution interpolation",
            local_domain_md_range_policy_nodes( domains.back() ),
            SolutionInterpolator(
                subdomain_shell_coords.back(), subdomain_radii.back(), solution.grid_data(), false ) );
        Kokkos::fence();

        if ( coeff_type == CoeffType::Constant )
        {
            Kokkos::parallel_for(
                "rhs interpolation (constant k)",
                local_domain_md_range_policy_nodes( domains.back() ),
                RHSInterpolatorConstantK(
                    subdomain_shell_coords.back(), subdomain_radii.back(), error.grid_data() ) );
        }
        else
        {
            Kokkos::parallel_for(
                "rhs interpolation (variable k)",
                local_domain_md_range_policy_nodes( domains.back() ),
                RHSInterpolator(
                    subdomain_shell_coords.back(), subdomain_radii.back(), error.grid_data(), r_min, r_max, alpha,
                    k_max ) );
        }
        Kokkos::fence();

        linalg::apply( M, error, f );
        assign( error, 0.0 );
        assign( u, 0.0 );

        Kokkos::parallel_for(
            "boundary interpolation",
            local_domain_md_range_policy_nodes( domains.back() ),
            SolutionInterpolator(
                subdomain_shell_coords.back(), subdomain_radii.back(), u.grid_data(), true ) );
        Kokkos::fence();

        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            A_neumann, A_neumann_diag, u, error, f, boundary_mask_data.back(),
            grid::shell::ShellBoundaryFlag::BOUNDARY );
        assign( u, 0.0 );
        Kokkos::fence();
    }
    else
    {
        // No analytical solution: use f = 1 as RHS, zero Dirichlet BCs.
        assign( error, 1.0 );
        linalg::apply( M, error, f );
        assign( error, 0.0 );
        assign( u, 0.0 );

        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            A_neumann, A_neumann_diag, u, error, f, boundary_mask_data.back(),
            grid::shell::ShellBoundaryFlag::BOUNDARY );
        assign( u, 0.0 );
        Kokkos::fence();
    }

    // Solve.
    linalg::solvers::IterativeSolverParameters solver_params{ 1000, 1e-8, 1e-8 };
    CoarseGridSolver coarse_grid_solver( solver_params, table, coarse_grid_tmps );

    linalg::solvers::Multigrid< DivKGrad, Prolongation, Restriction, Smoother, CoarseGridSolver > multigrid_solver(
        P_additive, R, A_c, tmp_r_c, tmp_e_c, tmp, smoothers, smoothers, coarse_grid_solver, 100, 1e-6 );
    multigrid_solver.collect_statistics( table );

    Kokkos::fence();
    linalg::solvers::solve( multigrid_solver, A, u, f );
    Kokkos::fence();

    const int num_cycles =
        static_cast< int >( table->query_rows_equals( "tag", "multigrid" ).rows().size() );

    if ( has_analytical_solution )
    {
        assign( error, 0.0 );
        linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
        const auto l2_error = linalg::norm_2_scaled( error, 1.0 / static_cast< T >( num_dofs ) );
        return { num_cycles, l2_error };
    }
    else
    {
        return { num_cycles, T( -1 ) };
    }
}

/// @brief Dispatches run_multigrid_solve with the correct CoeffEval type for the given scenario.
template < std::floating_point T >
std::pair< int, T > dispatch_multigrid_solve(
    int                                           min_level,
    int                                           max_level,
    const std::shared_ptr< util::Table >&         table,
    int                                           prepost_smooth,
    double                                        alpha,
    double                                        k_max,
    InterpolationMode                             interp_mode,
    CoeffType                                     coeff_type,
    bool                                          use_exact_eval,
    Kokkos::View< const double* >                 profile_radii  = {},
    Kokkos::View< const double* >                 profile_values = {} )
{
    const double r_min = 0.5;
    const double r_max = 1.0;

    if ( use_exact_eval )
    {
        switch ( coeff_type )
        {
        case CoeffType::Constant:
            return run_multigrid_solve< T >(
                min_level, max_level, table, prepost_smooth, alpha, k_max, interp_mode, coeff_type,
                CoeffEvalConstant{} );
        case CoeffType::Tanh:
            return run_multigrid_solve< T >(
                min_level, max_level, table, prepost_smooth, alpha, k_max, interp_mode, coeff_type,
                CoeffEvalTanh{ alpha, r_min, r_max, k_max } );
        case CoeffType::RadialStep:
            return run_multigrid_solve< T >(
                min_level, max_level, table, prepost_smooth, alpha, k_max, interp_mode, coeff_type,
                CoeffEvalRadialStep{ 0.5 * ( r_min + r_max ), k_max } );
        case CoeffType::LateralStep:
            return run_multigrid_solve< T >(
                min_level, max_level, table, prepost_smooth, alpha, k_max, interp_mode, coeff_type,
                CoeffEvalLateralStep{ k_max } );
        case CoeffType::UnalignedRadialStep:
            return run_multigrid_solve< T >(
                min_level, max_level, table, prepost_smooth, alpha, k_max, interp_mode, coeff_type,
                CoeffEvalUnalignedRadialStep{ r_min + ( r_max - r_min ) / M_PI, k_max } );
        case CoeffType::UnalignedLateralStep:
            return run_multigrid_solve< T >(
                min_level, max_level, table, prepost_smooth, alpha, k_max, interp_mode, coeff_type,
                CoeffEvalUnalignedLateralStep{ k_max } );
        case CoeffType::RadialProfile:
            return run_multigrid_solve< T >(
                min_level, max_level, table, prepost_smooth, alpha, k_max, interp_mode, coeff_type,
                CoeffEvalRadialProfile{ profile_radii, profile_values } );
        }
    }
    else
    {
        return run_multigrid_solve< T >(
            min_level, max_level, table, prepost_smooth, alpha, k_max, interp_mode, coeff_type,
            NoCoeffEval{} );
    }

    // Unreachable.
    return { 0, T( -1 ) };
}

std::string interp_mode_name( InterpolationMode mode )
{
    switch ( mode )
    {
    case InterpolationMode::Constant:
        return "Constant";
    case InterpolationMode::Linear:
        return "Linear";
    case InterpolationMode::OperatorDependent:
        return "OperatorDependent";
    case InterpolationMode::UnknownBasedAMG:
        return "UnknownBasedAMG";
    case InterpolationMode::UnknownBasedAMGLateral:
        return "UB_AMG_Lateral";
    default:
        return "Unknown";
    }
}

struct CoeffScenario
{
    std::string name;
    double      alpha;
    double      k_max;
    CoeffType   coeff_type;
    int         min_level;
    int         max_level;

    // Only used for RadialProfile scenarios.
    Kokkos::View< const double* > profile_radii;
    Kokkos::View< const double* > profile_values;
};

struct Result
{
    int    level;
    int    cycles;
    double l2_error;
    double order;
};

template < std::floating_point T >
void print_summary(
    const std::string&                                              scenario_name,
    const std::vector< InterpolationMode >&                         modes,
    const std::map< InterpolationMode, std::vector< Result > >&     all_results )
{
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << "  " << scenario_name << std::endl;
    std::cout << "------------------------------------------------------------\n" << std::endl;

    std::cout << std::setw( 20 ) << "Mode"
              << std::setw( 8 ) << "Level"
              << std::setw( 10 ) << "Cycles"
              << std::setw( 16 ) << "L2 Error"
              << std::setw( 12 ) << "Order"
              << std::endl;
    std::cout << std::string( 66, '-' ) << std::endl;

    for ( auto mode : modes )
    {
        const auto it = all_results.find( mode );
        if ( it == all_results.end() )
            continue;
        for ( const auto& r : it->second )
        {
            std::cout << std::setw( 20 ) << interp_mode_name( mode )
                      << std::setw( 8 ) << r.level
                      << std::setw( 10 ) << r.cycles;
            if ( r.l2_error >= 0 )
            {
                std::cout << std::setw( 16 ) << std::scientific << std::setprecision( 4 ) << r.l2_error
                          << std::setw( 12 ) << std::fixed << std::setprecision( 2 ) << r.order;
            }
            else
            {
                std::cout << std::setw( 16 ) << "n/a"
                          << std::setw( 12 ) << "n/a";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

template < std::floating_point T >
int run_test()
{
    const int  prepost_smooth = 3;
    const bool use_exact_eval = true;

    std::vector< InterpolationMode > modes = {
        InterpolationMode::Constant,
        InterpolationMode::Linear,
        InterpolationMode::OperatorDependent,
        InterpolationMode::UnknownBasedAMG,
        InterpolationMode::UnknownBasedAMGLateral,
    };
    std::vector< double > beta_values = { 0.0 };

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

        // Sort ascending by radius.
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
        { "Constant coefficient (k=1)", 0.0, 1.0, CoeffType::Constant, 0, 3, {}, {} },
        { "Variable coefficient (k_max=10, alpha=1)", 1.0, 10.0, CoeffType::Tanh, 0, 3, {}, {} },
        { "Unaligned lateral step (k_max=10)", 0.0, 10.0, CoeffType::UnalignedLateralStep, 0, 3, {}, {} },
        { "Unaligned lateral step (k_max=100)", 0.0, 100.0, CoeffType::UnalignedLateralStep, 0, 3, {}, {} },
        { "Unaligned lateral step (k_max=1000)", 0.0, 1000.0, CoeffType::UnalignedLateralStep, 0, 3, {}, {} },
        { "Unaligned lateral step (k_max=1e4)", 0.0, 1e4, CoeffType::UnalignedLateralStep, 0, 3, {}, {} },
        { "Unaligned lateral step (k_max=1e5)", 0.0, 1e5, CoeffType::UnalignedLateralStep, 0, 3, {}, {} },
        { "Unaligned lateral step (k_max=1e6)", 0.0, 1e6, CoeffType::UnalignedLateralStep, 0, 3, {}, {} },
        { "Lin et al. 2022 viscosity profile", 0.0, 0.0, CoeffType::RadialProfile, 0, 3,
          lin_radii, lin_values },
        { "Stotz et al. 2017 viscosity profile", 0.0, 0.0, CoeffType::RadialProfile, 0, 3,
          stotz_radii, stotz_values },
    };

    // Collect results per scenario.
    std::vector< std::map< InterpolationMode, std::vector< Result > > > scenario_results( scenarios.size() );

    for ( size_t si = 0; si < scenarios.size(); si++ )
    {
        const auto& scenario = scenarios[si];

        if ( mpi::rank() == 0 )
        {
            std::cout << "\n========================================================" << std::endl;
            std::cout << "  SCENARIO: " << scenario.name << std::endl;
            std::cout << "========================================================" << std::endl;
        }

        for ( auto mode : modes )
        {
            {
                std::string label = interp_mode_name( mode );

                if ( mpi::rank() == 0 )
                    std::cout << "\n  --- " << label << " ---\n" << std::endl;

                T prev_l2_error = 1.0;

                for ( int level = scenario.min_level + 1; level <= scenario.max_level; level++ )
                {
                    auto table = std::make_shared< util::Table >();

                    if ( mpi::rank() == 0 )
                        std::cout << "    level " << level << ": " << std::flush;
                    auto [cycles, l2_error] = dispatch_multigrid_solve< T >(
                        scenario.min_level, level, table, prepost_smooth,
                        scenario.alpha, scenario.k_max, mode, scenario.coeff_type, use_exact_eval,
                        scenario.profile_radii, scenario.profile_values );

                    double order = ( l2_error > 0 && prev_l2_error > 0 ) ? prev_l2_error / l2_error : 0.0;
                    if ( l2_error > 0 )
                        prev_l2_error = l2_error;

                    if ( mpi::rank() == 0 )
                    {
                        std::cout << "cycles = " << cycles;
                        if ( l2_error >= 0 )
                        {
                            std::cout << ", l2_error = " << std::scientific << std::setprecision( 4 ) << l2_error
                                      << ", order = " << std::fixed << std::setprecision( 2 ) << order;
                        }
                        std::cout << std::endl;
                    }

                    scenario_results[si][mode].push_back( { level, cycles, l2_error, order } );
                }
            }
        }
    }

    // Print summary tables.
    if ( mpi::rank() == 0 )
    {
        std::cout << "\n\n============================================================" << std::endl;
        std::cout << "  SUMMARY: GCA Interpolation Mode Comparison" << std::endl;
        std::cout << "============================================================" << std::endl;

        for ( size_t si = 0; si < scenarios.size(); si++ )
        {
            print_summary< T >( scenarios[si].name, modes, scenario_results[si] );
        }

        std::cout << "PASS: All scenarios completed." << std::endl;
    }
    return EXIT_SUCCESS;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    return run_test< double >();
}
