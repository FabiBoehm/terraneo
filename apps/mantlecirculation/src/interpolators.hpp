#pragma once

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "parameters.hpp"
#include "util/bit_masking.hpp"

namespace terra::mantlecirculation {

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;

struct InitialConditionInterpolator
{
    ScalarType                                         r_min_;
    ScalarType                                         r_max_;
    ScalarType                                         T_min_;
    ScalarType                                         T_max_;
    Grid3DDataVec< ScalarType, 3 >                     grid_;
    Grid2DDataScalar< ScalarType >                     radii_;
    Grid4DDataScalar< ScalarType >                     data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_data_;
    bool                                               only_boundary_;

    InitialConditionInterpolator(
        const ScalarType                                          r_min,
        const ScalarType                                          r_max,
        const ScalarType                                          T_min,
        const ScalarType                                          T_max,
        const Grid3DDataVec< ScalarType, 3 >&                     grid,
        const Grid2DDataScalar< ScalarType >&                     radii,
        const Grid4DDataScalar< ScalarType >&                     data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask_data,
        bool                                                      only_boundary )
    : r_min_( r_min )
    , r_max_( r_max )
    , T_min_( T_min )
    , T_max_( T_max )
    , grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_data_( mask_data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto mask_value  = mask_data_( local_subdomain_id, x, y, r );
        const auto is_boundary = util::has_flag( mask_value, grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || is_boundary )
        {
            const dense::Vec< ScalarType, 3 > coords =
                grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
            const auto frac                      = ( r_max_ - coords.norm() ) / ( r_max_ - r_min_ );
            data_( local_subdomain_id, x, y, r ) = T_min_ + ( T_max_ - T_min_ ) * Kokkos::pow( frac, 5 );
        }
    }
};

// Interpolate from custom radial profile to Q1 field
struct RadialProfileToQ1
{
    Grid2DDataScalar< ScalarType > profile_;
    Grid4DDataScalar< ScalarType > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        data_( id, x, y, r ) = profile_( id, r );
    }
};

// Subtracts laterally constant profile data from Grid4DDataScalar.
// Computes src_(id, x, y, r) - profile_(id, r) = dst_(id, x, y, r).
struct SubtractRadialProfile
{
    Grid2DDataScalar< ScalarType > profile_;
    Grid4DDataScalar< ScalarType > src_;
    Grid4DDataScalar< ScalarType > dst_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        dst_( id, x, y, r ) = src_( id, x, y, r ) - profile_( id, r );
    }
};

/// Initial condition for Q1 temperature (conductive profile + spherical harmonic perturbation):
/// T = T_cond(r) + eps * Y_l^m(theta, phi)
/// where T_cond is the steady-state spherical conduction solution:
///   T_cond(r) = r_min * r_max / r  -  r_min
struct ConductiveProfileInterpolator
{
    ScalarType                     r_min_, r_max_, eps_;
    ScalarType                     T_min_;
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > data_;
    Grid3DDataScalar< ScalarType > sph_coeffs_;
    bool                           has_sph_;
    // Canonical TALA (match HyTeG): start on the adiabat T̄(r)=T_ad,s·exp(Di·(r_max−r))
    // rather than the conductive profile, so the background is consistent with the
    // buoyancy reference (Tdev(t=0) = perturbation). Off by default.
    bool                           adiabatic_ = false;
    ScalarType                     Di_        = ScalarType( 0 );
    ScalarType                     T_ad_s_    = ScalarType( 0 );

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const ScalarType                  radius = coords.norm();

        // Guard against zero radius (non-owned ghost nodes may have zero coordinates).
        if ( radius < ScalarType( 1e-15 ) )
        {
            data_( sd, x, y, r ) = ScalarType( 0 );
            return;
        }

        const ScalarType T_bg = adiabatic_ ? ( T_ad_s_ * Kokkos::exp( Di_ * ( r_max_ - radius ) ) )
                                           : ( ( r_min_ * r_max_ / radius - r_min_ ) / ( r_max_ - r_min_ ) + T_min_ );

        ScalarType T_val = T_bg;
        if ( has_sph_ )
        {
            T_val += eps_ * sph_coeffs_( sd, x, y );
        }

        data_( sd, x, y, r ) = T_val;
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_u_;
    Grid4DDataScalar< ScalarType > data_T_;
    // Reference density ρ̄(r). Used only when weight_by_rho_ is true (canonical
    // King-2010 TALA buoyancy f = Ra·ρ̄·T′·n̂). ρ̄≡1 for incompressible runs, so
    // the multiply is harmless there regardless of the flag.
    Grid4DDataScalar< ScalarType > rho_;
    ScalarType                     rayleigh_number_;
    bool                           weight_by_rho_ = false;

    RHSVelocityInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data_u,
        const Grid4DDataScalar< ScalarType >& data_T,
        const Grid4DDataScalar< ScalarType >& rho,
        ScalarType                            rayleigh_number,
        bool                                  weight_by_rho )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , data_T_( data_T )
    , rho_( rho )
    , rayleigh_number_( rayleigh_number )
    , weight_by_rho_( weight_by_rho )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const auto n = coords.normalized();

        // Canonical TALA buoyancy carries the reference density ρ̄(r); the legacy
        // ρ̄-free form uses 1. (ρ̄≡1 for incompressible, so this is a no-op there.)
        const ScalarType rho =
            weight_by_rho_ ? rho_( local_subdomain_id, x, y, r ) : ScalarType( 1 );

        for ( int d = 0; d < 3; d++ )
        {
            data_u_( local_subdomain_id, x, y, r, d ) =
                rayleigh_number_ * rho * n( d ) * data_T_( local_subdomain_id, x, y, r );
        }
    }
};

struct NoiseAdder
{
    ScalarType                                  T_min_;
    ScalarType                                  T_max_;
    Grid3DDataVec< ScalarType, 3 >              grid_;
    Grid2DDataScalar< ScalarType >              radii_;
    Grid4DDataScalar< ScalarType >              data_T_;
    Grid4DDataScalar< grid::NodeOwnershipFlag > mask_;
    Kokkos::Random_XorShift64_Pool<>            rand_pool_;

    NoiseAdder(
        const ScalarType                                   T_min,
        const ScalarType                                   T_max,
        const Grid3DDataVec< ScalarType, 3 >&              grid,
        const Grid2DDataScalar< ScalarType >&              radii,
        const Grid4DDataScalar< ScalarType >&              data_T,
        const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask )
    : T_min_( T_min )
    , T_max_( T_max )
    , grid_( grid )
    , radii_( radii )
    , data_T_( data_T )
    , mask_( mask )
    , rand_pool_( 12345 )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        auto generator = rand_pool_.get_state();

        const ScalarType eps          = 1e-1;
        const auto       perturbation = eps * ( 2.0 * generator.drand() - 1.0 );

        const auto process_ownes_point =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::NodeOwnershipFlag::OWNED );

        if ( process_ownes_point )
        {
            data_T_( local_subdomain_id, x, y, r ) =
                Kokkos::clamp( data_T_( local_subdomain_id, x, y, r ) + perturbation, T_min_, T_max_ );
        }
        else
        {
            data_T_( local_subdomain_id, x, y, r ) = T_min_;
        }

        rand_pool_.free_state( generator );
    }
};

/// Computes viscosity from temperature according to the selected viscosity law.
struct ViscosityFromTemperature
{
    ViscosityLaw                   law_;
    ScalarType                     rmu_;
    Grid4DDataScalar< ScalarType > eta_;
    Grid4DDataScalar< ScalarType > T_;
    // Radial reference profile eta_ref(r), multiplied into the T-dependent factor.
    // All-ones when no radial profile is configured, so the product is a no-op then.
    Grid2DDataScalar< ScalarType > eta_ref_;
    // Radial coordinate + shell bounds for the Arrhenius depth term (unused otherwise).
    Grid2DDataScalar< ScalarType > radii_;
    ScalarType                     r_min_             = ScalarType( 0 );
    ScalarType                     r_max_             = ScalarType( 0 );
    ScalarType                     activation_energy_ = ScalarType( 0 );
    ScalarType                     depth_factor_      = ScalarType( 0 );
    ScalarType                     visc_min_          = ScalarType( 0 );
    ScalarType                     visc_max_          = ScalarType( 0 );

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType T_val = T_( id, x, y, r );

        switch ( law_ )
        {
        case ViscosityLaw::FRANK_KAMENETSKII:
            // Zhong et al. (2008) form: mu = rmu^(0.5 - T), times the radial
            // reference profile eta_ref(r).  Total cold/hot contrast = rmu * (eta_ref range).
            eta_( id, x, y, r ) = Kokkos::pow( rmu_, ScalarType( 0.5 ) - T_val ) * eta_ref_( id, r );
            break;
        case ViscosityLaw::ARRHENIUS:
        {
            // HyTeG type-3 law (src/terraneo/helpers/Viscosity.hpp):
            //   eta = exp( E_a·(1/(T+0.25) − 1.45) + D·(r_max−r)/(r_max−r_min) )
            // clamped to [visc_min, visc_max]. Depth term carries the radial dependence,
            // so no separate eta_ref profile is needed (eta_ref_ ≡ 1 for a pure Arrhenius run).
            const ScalarType radius     = radii_( id, r );
            const ScalarType depth_norm = ( r_max_ - radius ) / ( r_max_ - r_min_ );
            ScalarType       eta        = Kokkos::exp(
                activation_energy_ * ( ScalarType( 1 ) / ( T_val + ScalarType( 0.25 ) ) - ScalarType( 1.45 ) ) +
                depth_factor_ * depth_norm );
            eta                 = Kokkos::clamp( eta, visc_min_, visc_max_ );
            eta_( id, x, y, r ) = eta * eta_ref_( id, r );
            break;
        }
        case ViscosityLaw::CONSTANT:
        default:
            // eta is already set, nothing to do.
            break;
        }
    }
};

struct DensityInit
{
    Grid4DDataScalar< ScalarType > rho_;
    Grid2DDataScalar< ScalarType > radii_;
    ScalarType                     r_max_;
    ScalarType                     surface_density_;
    ScalarType                     dissipation_number_;
    ScalarType                     grueneisen_parameter_;
    bool                           compressible_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        if ( compressible_ )
        {
            // Adiabatic compression
            const ScalarType radius = radii_( id, r );

            rho_( id, x, y, r ) =
                surface_density_ * Kokkos::exp( dissipation_number_ * ( r_max_ - radius ) / grueneisen_parameter_ );
        }
        else
        {
            rho_( id, x, y, r ) = 1.0;
        }
    }
};

/// @brief Nodal adiabatic (compression) heating source for the TALA energy
/// equation.
///
/// Fills a Q1 scalar field with
///
///     S_adiab = prefactor · Di · (u·n) · T
///
/// where n = coords.normalized() is the outward radial (anti-gravity) unit
/// vector, u·n the radial velocity, T the temperature and Di the dissipation
/// number. The nodal field is meant to be L²-projected onto the RHS via the
/// mass matrix (like the constant internal-heating source) and interpolated
/// into the entropy-viscosity residual.
///
/// The nondimensional coefficient is Di alone. Notably there is NO reference
/// density ρ̄ and NO thermal expansivity α: this matches the buoyancy force
/// used by the Stokes solve, Ra·δT·n, which is likewise ρ̄- and α-free (α is
/// folded into Di and Ra). Weighting the adiabatic term by ρ̄ while the
/// buoyancy is unweighted would break the dissipation balance ⟨Φ⟩=⟨W⟩; keeping
/// both ρ̄-free is the self-consistent choice for this branch's formulation.
///
/// The physically-correct prefactor is −1: rising material (u·n > 0) does work
/// against gravity and cools, so it must contribute negatively to DT/Dt. The
/// prefactor is left configurable to calibrate against this branch's exact
/// nondimensionalisation / temperature-offset convention.
struct AdiabaticHeatingSource
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > u_;
    Grid4DDataScalar< ScalarType > T_;
    Grid4DDataScalar< ScalarType > dst_;
    ScalarType                     dissipation_number_;
    ScalarType                     prefactor_ = ScalarType( -1 );

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( id, x, y, r, grid_, radii_ );
        const auto                        n      = coords.normalized();

        ScalarType u_r = ScalarType( 0 );
        for ( int d = 0; d < 3; ++d )
            u_r += u_( id, x, y, r, d ) * n( d );

        dst_( id, x, y, r ) = prefactor_ * dissipation_number_ * u_r * T_( id, x, y, r );
    }
};

} // namespace terra::mantlecirculation
