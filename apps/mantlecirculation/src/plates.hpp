#pragma once

#include <string>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"
#include "parameters.hpp"
#include "terra/plates/PlateVelocityProvider.hpp"
#include "terra/plates/types.hpp"
#include "util/logging.hpp"

namespace terra::mantlecirculation {

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataVec;

struct PlateVelocityWriter
{
    Grid3DDataVec< ScalarType, 3 > surface_velocity_;
    Grid4DDataVec< ScalarType, 3 > velocity_field_;
    int                            r_idx_surface_;

    PlateVelocityWriter(
        const Grid3DDataVec< ScalarType, 3 >& surface_velocity,
        const Grid4DDataVec< ScalarType, 3 >& velocity_field,
        const int                             r_idx_surface )
    : surface_velocity_( surface_velocity )
    , velocity_field_( velocity_field )
    , r_idx_surface_( r_idx_surface )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y ) const
    {
        // Write plate velocity to surface layer of velocity field
        for ( int d = 0; d < 3; ++d )
            velocity_field_( id, x, y, r_idx_surface_, d ) = surface_velocity_( id, x, y, d );
    }
};

template < typename GridType, typename RadiiType, typename DataType, typename VelocityFn >
struct ComputePlateVelocities
{
    GridType   grid_;
    RadiiType  radii_;
    DataType   plate_data_;
    VelocityFn computeVelocity;

    ComputePlateVelocities(
        const GridType&  grid,
        const RadiiType& radii,
        const DataType&  plate_data,
        VelocityFn       velocityFn )
    : grid_( grid )
    , radii_( radii )
    , plate_data_( plate_data )
    , computeVelocity( std::move( velocityFn ) )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y ) const
    {
        const dense::Vec< ScalarType, 3 > coords =
            grid::shell::coords( id, x, y, radii_.extent( 1 ) - 1, grid_, radii_ );

        const vec3D v = computeVelocity( coords );
        for ( int d = 0; d < 3; ++d )
            plate_data_( id, x, y, d ) = v( d );
    }
};

void extract_plate_velocities(
    ScalarType                            plate_age,
    grid::Grid3DDataVec< ScalarType, 3 >& plate_velocities,
    plates::PlateVelocityProvider&        oracle,
    const Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const Grid2DDataScalar< ScalarType >& coords_radii,
    const bool                            interpolate_in_time,
    const ScalarType                      scale_factor )
{
    using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

    plates::StatisticsPlateNotFoundHandler    errorHandler;
    plates::UniformCirclesPointWeightProvider pointWeightProvider( { { 1.0 / 100.0, 6 } }, 1e-1 );

    if ( !interpolate_in_time )
        plate_age = std::ceil( plate_age );

    util::logroot << "Updating plates..... Plate age: " << plate_age << " Ma." << std::endl;

    // Mirror the needed Kokkos::Views to the host
    auto coords_host = Kokkos::create_mirror_view( coords_shell );
    Kokkos::deep_copy( coords_host, coords_shell );
    auto radii_host = Kokkos::create_mirror_view( coords_radii );
    Kokkos::deep_copy( radii_host, coords_radii );
    auto plate_velocities_host = Kokkos::create_mirror_view( plate_velocities );

    // Callback function for computing velocity components
    auto getPointVelocity =
        [&oracle, plate_age, &pointWeightProvider, &errorHandler, interpolate_in_time, scale_factor](
            const vec3D& point ) {
            vec3D velocity;
            if ( interpolate_in_time )
            {
                velocity = oracle.getLocallyAveragedPointVelocityInterpolatedInTime(
                    point, plate_age, pointWeightProvider, errorHandler );
            }

            else
            {
                velocity =
                    oracle.getLocallyAveragedPointVelocity( point, plate_age, pointWeightProvider, errorHandler );
            }

            // Nondimensionalise and scale before returning
            return velocity * scale_factor;
        };

    // Extract plate velocities from data
    // Explicitly on host since underlying plates functionality is not device-callable.
    Kokkos::parallel_for(
        "ComputePlateVelocities",
        Kokkos::MDRangePolicy< HostExecSpace, Kokkos::Rank< 3 > >(
            { 0, 0, 0 }, { coords_host.extent( 0 ), coords_host.extent( 1 ), coords_host.extent( 2 ) } ),
        ComputePlateVelocities( coords_host, radii_host, plate_velocities_host, getPointVelocity ) );
    Kokkos::fence();

    // Copy to device
    Kokkos::deep_copy( plate_velocities, plate_velocities_host );

    util::logroot << "Plate data extracted." << std::endl;
}

void apply_plate_velocities(
    const Grid3DDataVec< ScalarType, 3 >& plate_velocities,
    Grid4DDataVec< ScalarType, 3 >&       velocity_field,
    const int                             r_idx_surface )
{
    // Write given plate velocities to velocity field
    Kokkos::parallel_for(
        "PlateVelocityWriter",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
            { 0, 0, 0 }, { plate_velocities.extent( 0 ), plate_velocities.extent( 1 ), plate_velocities.extent( 2 ) } ),
        PlateVelocityWriter( plate_velocities, velocity_field, r_idx_surface ) );
    Kokkos::fence();

    util::logroot << "Surface velocity updated." << std::endl;
}

inline std::shared_ptr< plates::PlateVelocityProvider >
    initialise_plates( const std::string& fnameTopologies, const std::string& fnameReconstructions )
{
    std::shared_ptr< plates::PlateVelocityProvider > oracle;

    util::logroot << "Setting up Oracle for plates." << std::endl;

    oracle = std::make_shared< plates::PlateVelocityProvider >( fnameTopologies, fnameReconstructions );

    return oracle;
}

} // namespace terra::mantlecirculation