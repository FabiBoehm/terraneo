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

// Interpolate from radial profile to Q1 field
struct RadialProfileToQ1
{
    Grid4DDataScalar< ScalarType > data_;
    Grid2DDataScalar< ScalarType > radial_profile_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        data_( id, x, y, r ) = radial_profile_( id, r );
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

        const ScalarType T_cond = ( r_min_ * r_max_ / radius - r_min_ ) / ( r_max_ - r_min_ ) + T_min_;

        ScalarType T_val = T_cond;
        if ( has_sph_ )
        {
            T_val += eps_ * sph_coeffs_( sd, x, y );
        }

        data_( sd, x, y, r ) = T_val;
    }
};

struct BuoyancyForceAssembly
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_f_;
    Grid4DDataScalar< ScalarType > data_T_;
    Grid4DDataScalar< ScalarType > data_rho_;
    Grid2DDataScalar< ScalarType > alpha_;
    ScalarType                     rayleigh_number_;
    ScalarType                     prefactor_;

    BuoyancyForceAssembly(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data_f,
        const Grid4DDataScalar< ScalarType >& data_T,
        const Grid4DDataScalar< ScalarType >& data_rho,
        const Grid2DDataScalar< ScalarType >& alpha,
        const ScalarType                      rayleigh_number,
        const ScalarType                      prefactor = ScalarType( 1 ) )
    : grid_( grid )
    , radii_( radii )
    , data_f_( data_f )
    , data_T_( data_T )
    , data_rho_( data_rho )
    , alpha_( alpha )
    , rayleigh_number_( rayleigh_number )
    , prefactor_( prefactor )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( id, x, y, r, grid_, radii_ );

        const auto n = coords.normalized();

        for ( int d = 0; d < 3; d++ )
        {
            data_f_( id, x, y, r, d ) = prefactor_ * rayleigh_number_ * n( d ) * alpha_( id, r ) *
                                        data_rho_( id, x, y, r ) * data_T_( id, x, y, r );
        }
    }
};

struct NoiseAdder
{
    ScalarType                                  eps_;
    ScalarType                                  T_min_;
    ScalarType                                  T_max_;
    ScalarType                                  r_min_;
    ScalarType                                  r_max_;
    bool                                        taper_near_boundaries_;
    Grid3DDataVec< ScalarType, 3 >              grid_;
    Grid2DDataScalar< ScalarType >              radii_;
    Grid4DDataScalar< ScalarType >              data_;
    Grid4DDataScalar< grid::NodeOwnershipFlag > mask_;
    Kokkos::Random_XorShift64_Pool<>            rand_pool_;

    NoiseAdder(
        const ScalarType                                   eps,
        const ScalarType                                   T_min,
        const ScalarType                                   T_max,
        const ScalarType                                   r_min,
        const ScalarType                                   r_max,
        const bool                                         taper_near_boundaries,
        const Grid3DDataVec< ScalarType, 3 >&              grid,
        const Grid2DDataScalar< ScalarType >&              radii,
        const Grid4DDataScalar< ScalarType >&              data,
        const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask )
    : eps_( eps )
    , T_min_( T_min )
    , T_max_( T_max )
    , r_min_( r_min )
    , r_max_( r_max )
    , taper_near_boundaries_( taper_near_boundaries )
    , grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_( mask )
    , rand_pool_( 12345 )
    {}

    static constexpr ScalarType taper_width_ = ScalarType( 0.05 );

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        auto generator = rand_pool_.get_state();

        const ScalarType perturbation = eps_ * ( 2.0 * generator.drand() - 1.0 );
        ScalarType       taper        = ScalarType( 1 );

        // apply tapering near top and bottom boundary
        if ( taper_near_boundaries_ )
        {
            const ScalarType radius = radii_( id, r );

            const ScalarType dist_to_boundary = Kokkos::min( radius - r_min_, r_max_ - radius );
            const ScalarType t = Kokkos::clamp( dist_to_boundary / taper_width_, ScalarType( 0 ), ScalarType( 1 ) );

            taper = t * t * ( ScalarType( 3 ) - ScalarType( 2 ) * t );
        }

        // Only write to owned nodes
        const auto process_owns_point = util::has_flag( mask_( id, x, y, r ), grid::NodeOwnershipFlag::OWNED );

        if ( process_owns_point )
        {
            data_( id, x, y, r ) = Kokkos::clamp( data_( id, x, y, r ) + taper * perturbation, T_min_, T_max_ );
        }
        else
        {
            data_( id, x, y, r ) = ScalarType( 0 );
        }

        rand_pool_.free_state( generator );
    }
};

struct SphericalHarmonicPerturbationAdder
{
    ScalarType                                  eps_;
    ScalarType                                  T_min_, T_max_;
    ScalarType                                  r_min_, r_max_;
    bool                                        taper_near_boundaries_;
    Grid2DDataScalar< ScalarType >              radii_; // Only used if taper_near_boundaries == 'true'
    Grid3DDataScalar< ScalarType >              sph_coeffs_;
    Grid4DDataScalar< ScalarType >              data_;
    Grid4DDataScalar< grid::NodeOwnershipFlag > mask_;

    static constexpr ScalarType taper_width_ = ScalarType( 0.05 );

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        ScalarType taper = ScalarType( 1 );

        // apply tapering near top and bottom boundary
        if ( taper_near_boundaries_ )
        {
            const ScalarType radius           = radii_( id, r );
            const ScalarType dist_to_boundary = Kokkos::min( radius - r_min_, r_max_ - radius );
            const ScalarType t = Kokkos::clamp( dist_to_boundary / taper_width_, ScalarType( 0 ), ScalarType( 1 ) );

            taper = t * t * ( ScalarType( 3 ) - ScalarType( 2 ) * t );
        }

        // Only write to owned nodes
        const auto process_owns_point = util::has_flag( mask_( id, x, y, r ), grid::NodeOwnershipFlag::OWNED );

        if ( process_owns_point )
        {
            data_( id, x, y, r ) =
                Kokkos::clamp( data_( id, x, y, r ) + taper * eps_ * sph_coeffs_( id, x, y ), T_min_, T_max_ );
        }
        else
        {
            data_( id, x, y, r ) = ScalarType( 0 );
        }
    }
};

/// Initial condition for FV cell-centred temperature: same radial profile as the Q1 version,
/// evaluated at the precomputed cell centres.
struct FVInitialConditionInterpolator
{
    ScalarType                     r_min_, r_max_;
    ScalarType                     T_min_, T_max_;
    Grid4DDataVec< ScalarType, 3 > cell_centers_;
    Grid4DDataScalar< ScalarType > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType cx     = cell_centers_( id, x, y, r, 0 );
        const ScalarType cy     = cell_centers_( id, x, y, r, 1 );
        const ScalarType cz     = cell_centers_( id, x, y, r, 2 );
        const ScalarType radius = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
        const ScalarType frac   = ( r_max_ - radius ) / ( r_max_ - r_min_ );
        data_( id, x, y, r )    = T_min_ + ( T_max_ - T_min_ ) * Kokkos::pow( frac, ScalarType( 5 ) );
    }
};

/// Noise adder for FV cells.  All non-ghost cells are owned by the local subdomain,
/// so no ownership mask is needed.
struct FVNoiseAdder
{
    ScalarType                       T_min_, T_max_;
    Grid4DDataScalar< ScalarType >   data_T_;
    Kokkos::Random_XorShift64_Pool<> rand_pool_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        auto             gen          = rand_pool_.get_state();
        const ScalarType eps          = 1e-1;
        const ScalarType perturbation = eps * ( 2.0 * gen.drand() - 1.0 );
        data_T_( id, x, y, r )        = Kokkos::clamp( data_T_( id, x, y, r ) + perturbation, T_min_, T_max_ );
        rand_pool_.free_state( gen );
    }
};

/// Computes viscosity from temperature according to the selected viscosity law.
struct ViscosityFromTemperature
{
    ViscosityLaw                   law_;
    ScalarType                     rmu_;
    Grid4DDataScalar< ScalarType > eta_;
    Grid4DDataScalar< ScalarType > T_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType T_val = T_( id, x, y, r );

        switch ( law_ )
        {
        case ViscosityLaw::FRANK_KAMENETSKII:
            // Zhong et al. (2008) form: mu = rmu^(0.5 - T).
            // Total viscosity contrast (cold/hot) = rmu.
            eta_( id, x, y, r ) = Kokkos::pow( rmu_, ScalarType( 0.5 ) - T_val );
            break;
        case ViscosityLaw::CONSTANT:
        default:
            // eta is already set, nothing to do.
            break;
        }
    }
};

} // namespace terra::mantlecirculation
