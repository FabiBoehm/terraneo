#pragma once

#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/plates/plate_storage.hpp"
#include "terra/types.hpp"

#include <vector>

/// @file
///
/// Device-callable plate lookup: which tectonic plate does a surface point belong to?
///
/// The host-side plate machinery in `src/terra/plates` walks polygon containers and is not callable from a
/// device kernel. This flattens a \ref terra::plates::PlateStorage stage into two flat Kokkos views -- one
/// with every plate's boundary vertices back to back, one with per-plate offsets -- so the point-in-polygon
/// test can run on the GPU alongside everything else.
///
/// Containment uses a winding-number sum over the spherical polygon rather than a planar point-in-polygon
/// test, which is orientation-agnostic and therefore needs no equivalent of Boost.Geometry's `correct()`.

namespace terra {

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;

namespace plates {

struct PlateCoords
{
    double lon{};
    double lat{};
};

struct PlateData
{
    uint_t id;
    int    vertex_offset;
    int    num_vertices;
};

typedef enum
{
    PLATE_IDS,
    VELOCITIES,
    VELOCITIES_AND_IDS
} job_t;

// ---------------------------------------------------------------------
// Small vector helpers (all in double precision, all device-callable)
// ---------------------------------------------------------------------

/// Convert lon/lat in degrees to a unit Cartesian vector on the sphere.
/// (Same convention as sph2cart() with radius = 1.)
KOKKOS_INLINE_FUNCTION
void lonLatDegToUnitXYZ( double lonDeg, double latDeg, double& x, double& y, double& z )
{
    const double lon    = lonDeg * ( Kokkos::numbers::pi_v< double > / 180.0 );
    const double lat    = latDeg * ( Kokkos::numbers::pi_v< double > / 180.0 );
    const double coslat = Kokkos::cos( lat );

    x = coslat * Kokkos::cos( lon );
    y = coslat * Kokkos::sin( lon );
    z = Kokkos::sin( lat );
}

KOKKOS_INLINE_FUNCTION
double dot3( double ax, double ay, double az, double bx, double by, double bz )
{
    return ax * bx + ay * by + az * bz;
}

KOKKOS_INLINE_FUNCTION
void cross3( double ax, double ay, double az, double bx, double by, double bz, double& cx, double& cy, double& cz )
{
    cx = ay * bz - az * by;
    cy = az * bx - ax * bz;
    cz = ax * by - ay * bx;
}

KOKKOS_INLINE_FUNCTION
double clampToUnitRange( double v )
{
    return v < -1.0 ? -1.0 : ( v > 1.0 ? 1.0 : v );
}

/// Angular distance (radians) between two unit vectors.
KOKKOS_INLINE_FUNCTION
double angularDistanceUnit( double ax, double ay, double az, double bx, double by, double bz )
{
    return Kokkos::acos( clampToUnitRange( dot3( ax, ay, az, bx, by, bz ) ) );
}

using PlateCoordsContainerDeviceView = Kokkos::View< PlateCoords* >;
using PlateContainerDeviceView       = Kokkos::View< PlateData* >;

KOKKOS_INLINE_FUNCTION
bool is_point_in_plate(
    dense::Vec< double, 3 >        point,
    PlateData                      plate,
    PlateCoordsContainerDeviceView plate_coords_container )
{
    double windingSum = 0.0;
    double minDist    = Kokkos::Experimental::finite_max_v< double >;

    const int nVerts = plate.num_vertices;

    const double px = point(0);
    const double py = point(1);
    const double pz = point(2);

    for ( int i = 0; i < nVerts; ++i )
    {
        const int j = ( i + 1 == nVerts ) ? 0 : i + 1;

        double ax, ay, az;
        double bx, by, bz;
        lonLatDegToUnitXYZ(
            plate_coords_container( i + plate.vertex_offset ).lon,
            plate_coords_container( i + plate.vertex_offset ).lat,
            ax,
            ay,
            az );
        lonLatDegToUnitXYZ(
            plate_coords_container( j + plate.vertex_offset ).lon,
            plate_coords_container( j + plate.vertex_offset ).lat,
            bx,
            by,
            bz );

        // --- winding-number contribution: signed angle at P between A and B ---
        // (replaces boost::geometry::within(); orientation-agnostic, so no
        //  equivalent of boost::geometry::correct() is needed)
        double n1x, n1y, n1z;
        double n2x, n2y, n2z;
        cross3( px, py, pz, ax, ay, az, n1x, n1y, n1z );
        cross3( px, py, pz, bx, by, bz, n2x, n2y, n2z );

        double cx, cy, cz;
        cross3( n1x, n1y, n1z, n2x, n2y, n2z, cx, cy, cz );

        const double sinAngle = dot3( cx, cy, cz, px, py, pz );
        const double cosAngle = dot3( n1x, n1y, n1z, n2x, n2y, n2z );

        windingSum += Kokkos::atan2( sinAngle, cosAngle );

        // --- point-to-great-circle-arc distance ---
        // (replaces boost::geometry::for_each_segment + boost::geometry::distance)
        double nx, ny, nz;
        cross3( ax, ay, az, bx, by, bz, nx, ny, nz );
        const double nLen = Kokkos::sqrt( dot3( nx, ny, nz, nx, ny, nz ) );

        double edgeDist;
        if ( nLen < 1e-14 )
        {
            // degenerate edge (A == B numerically): fall back to point distance
            edgeDist = angularDistanceUnit( px, py, pz, ax, ay, az );
        }
        else
        {
            nx /= nLen;
            ny /= nLen;
            nz /= nLen;

            // foot of perpendicular from P onto the great circle through A,B
            const double pDotN = dot3( px, py, pz, nx, ny, nz );
            double       fx    = px - pDotN * nx;
            double       fy    = py - pDotN * ny;
            double       fz    = pz - pDotN * nz;
            const double fLen  = Kokkos::sqrt( dot3( fx, fy, fz, fx, fy, fz ) );
            fx /= fLen;
            fy /= fLen;
            fz /= fLen;

            const double abDot = clampToUnitRange( dot3( ax, ay, az, bx, by, bz ) );
            const double afDot = clampToUnitRange( dot3( ax, ay, az, fx, fy, fz ) );
            const double bfDot = clampToUnitRange( dot3( bx, by, bz, fx, fy, fz ) );

            const bool footOnSegment = ( afDot >= abDot ) && ( bfDot >= abDot );

            if ( footOnSegment )
            {
                edgeDist = Kokkos::asin( clampToUnitRange( Kokkos::fabs( pDotN ) ) );
            }
            else
            {
                const double dA = angularDistanceUnit( px, py, pz, ax, ay, az );
                const double dB = angularDistanceUnit( px, py, pz, bx, by, bz );
                edgeDist        = ( dA < dB ) ? dA : dB;
            }
        }

        minDist = ( edgeDist < minDist ) ? edgeDist : minDist;
    }

    const bool inside = Kokkos::fabs( windingSum ) > 1.98 * Kokkos::numbers::pi_v< double >;
    // if ( inside )
    // {
    //     distanceRad = minDist;
    // }
    return inside;
}

struct PlateIDInterpolator
{
    double r_max_;

    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    uint_t n_plates_;

    PlateCoordsContainerDeviceView plate_coords_container_;
    PlateContainerDeviceView       plate_data_container_;

    PlateIDInterpolator(
        double                         r_max,
        Grid3DDataVec< double, 3 >     grid,
        Grid2DDataScalar< double >     radii,
        Grid4DDataScalar< double >     data,
        uint_t                         n_plates,
        PlateCoordsContainerDeviceView plate_coords_container,
        PlateContainerDeviceView       plate_data_container )
    : r_max_( r_max )
    , grid_( grid )
    , radii_( radii )
    , data_( data )
    , n_plates_( n_plates )
    , plate_coords_container_( plate_coords_container )
    , plate_data_container_( plate_data_container )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double r_val = coords.norm();

        if ( Kokkos::abs( r_val - r_max_ ) < 1e-5 )
        {
            for ( uint_t i_plate = 0u; i_plate < n_plates_; i_plate++ )
            {
                bool found = is_point_in_plate( coords, plate_data_container_( i_plate ), plate_coords_container_ );

                if( found )
                {
                    data_( local_subdomain_id, x, y, r ) = plate_data_container_( i_plate ).id;
                    break;
                }
            }

            // data_( local_subdomain_id, x, y, r ) = r_val;
        }
        else
        {
            data_( local_subdomain_id, x, y, r ) = 0.0;
        }
    }
};

/// @brief Flattens the plates of one reconstruction stage into device views for \ref PlateIDInterpolator.
///
/// Returns the coordinate container, the per-plate descriptor container, and the number of plates. Both views
/// are filled on the host and copied once; the lookup itself never touches host memory.
template < typename PlatesForStageType >
inline std::tuple< PlateCoordsContainerDeviceView, PlateContainerDeviceView, uint_t >
    build_device_plate_containers( const PlatesForStageType& plates )
{
    std::vector< int > offsets;
    offsets.reserve( plates.size() + 1 );
    offsets.push_back( 0 );

    for ( size_t i = 0; i < plates.size(); ++i )
        offsets.push_back( offsets[i] + static_cast< int >( plates[i].boundary.size() ) );

    PlateCoordsContainerDeviceView coords_device( "plate_coords_container", offsets[plates.size()] );
    PlateContainerDeviceView       data_device( "plate_data_container", plates.size() );

    auto coords_host = Kokkos::create_mirror_view( coords_device );
    auto data_host   = Kokkos::create_mirror_view( data_device );

    int vertex = 0;
    for ( size_t i = 0; i < plates.size(); ++i )
    {
        data_host( i ).id            = plates[i].id;
        data_host( i ).vertex_offset = offsets[i];
        data_host( i ).num_vertices  = offsets[i + 1] - offsets[i];

        for ( size_t b = 0; b < plates[i].boundary.size(); ++b )
        {
            coords_host( vertex ).lon = plates[i].boundary[b]( 0 );
            coords_host( vertex ).lat = plates[i].boundary[b]( 1 );
            ++vertex;
        }
    }

    Kokkos::deep_copy( coords_device, coords_host );
    Kokkos::deep_copy( data_device, data_host );

    return { coords_device, data_device, static_cast< uint_t >( plates.size() ) };
}

} // namespace plates
} // namespace terra
