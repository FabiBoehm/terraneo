/*
 * Copyright (c) 2017-2022 Dominik Thoennes, Nils Kohl, Marcus Mohr, Fatemeh Rezaei.
 *
 * This file is part of HyTeG
 * (see https://i10git.cs.fau.de/hyteg/hyteg).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <cmath>
#include <csignal>
#include <sstream>
#include <string>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "mpi/mpi.hpp"
#include "src/parameters.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/plates/plate_velocity_provider.hpp"
#include "terra/plates/types.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/timer.hpp"

namespace terra {
using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
// using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;

using grid::shell::DistributedDomain;

using util::logall;
using util::logroot;

// using linalg::VectorQ1IsoQ2Q1;

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

} // namespace plates
} // namespace terra

using namespace terra;
using namespace plates;

struct Parameters
{
    uint_t      max_level = 6;
    uint_t      min_level = 1;
    double      r_min     = 0.55;
    double      r_max     = 1.0;
    std::string jobType   = "both";
    int         beginAge  = 10;
    int         endAge    = 0;
    std::string outdir    = "./output";
    std::string dataDir   = "../../../TERRA-NG/data/plates/Chen2025-tomopac/";
};

// ========
//  Driver
// ========
int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    // ------------
    //  Parameters
    // ------------

    logroot << "*** STEP 1: Obtaining Steering Parameters" << std::endl;

    // TO-DO : implement reading a parameter file
    // Fill with default parameters.

    Parameters parameters;

    CLI::App app{ "Plate Velocity Demo" };

    terra::util::add_option_with_default( app, "--max-level", parameters.max_level );
    terra::util::add_option_with_default( app, "--min-level", parameters.min_level );
    terra::util::add_option_with_default( app, "--r-min", parameters.r_min );
    terra::util::add_option_with_default( app, "--r-max", parameters.r_max );
    terra::util::add_option_with_default( app, "--jobType", parameters.jobType );
    terra::util::add_option_with_default( app, "--beginAge", parameters.beginAge );
    terra::util::add_option_with_default( app, "--endAge", parameters.endAge );
    terra::util::add_option_with_default( app, "--data-dir", parameters.dataDir );

    CLI11_PARSE( app, argc, argv );

    terra::util::prepare_empty_directory_or_abort( parameters.outdir );

    const auto xdmf_dir            = parameters.outdir + "/xdmf";
    const auto radial_profiles_dir = parameters.outdir + "/radial_profiles";
    const auto timer_trees_dir     = parameters.outdir + "/timer_trees";

    terra::util::prepare_empty_directory_or_abort( xdmf_dir );
    terra::util::prepare_empty_directory_or_abort( radial_profiles_dir );
    terra::util::prepare_empty_directory_or_abort( timer_trees_dir );

    logroot << "Running with the following steering parameters:" << std::endl;
    logroot << "--max-level " << parameters.max_level << std::endl;
    logroot << "--min-level " << parameters.min_level << std::endl;
    logroot << "--r-min " << parameters.r_min << std::endl;
    logroot << "--r-max " << parameters.r_max << std::endl;
    logroot << "--jobType " << parameters.jobType << std::endl;
    logroot << "--beginAge " << parameters.beginAge << std::endl;
    logroot << "--endAge " << parameters.endAge << std::endl;
    logroot << "--data-dir " << parameters.dataDir << std::endl;

    // make sure beginAge > endAge -- we simulate forward in time
    if ( parameters.endAge > parameters.beginAge )
    {
        logroot << "## Specified endAge is larger than beginAge. Aborting.." << std::endl;
        std::abort();
    }

    // determine job type
    terra::plates::job_t jobType;
    jobType = terra::plates::VELOCITIES_AND_IDS;

    // ---------
    //  Meshing
    // ---------

    logroot << "*** STEP 2: Generating Mesh" << std::endl;

    std::vector< terra::grid::shell::DistributedDomain >   domains;
    std::vector< terra::grid::Grid3DDataVec< double, 3 > > coords_shell;
    std::vector< terra::grid::Grid2DDataScalar< double > > coords_radii;

    for ( int level = parameters.min_level; level <= parameters.max_level; level++ ) // not needed
    {
        const int idx = level - parameters.min_level;

        domains.push_back( terra::grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
            level, level, parameters.r_min, parameters.r_max ) );
        coords_shell.push_back(
            terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domains[idx] ) );
        coords_radii.push_back( terra::grid::shell::subdomain_shell_radii< double >( domains[idx] ) );
    }

    const auto   num_levels     = domains.size();
    const uint_t velocity_level = num_levels - 1;
    logroot << "velocity level : " << velocity_level << std::endl;

    // Mirror mesh details to host
    auto coords_shell_host = Kokkos::create_mirror_view( coords_shell[velocity_level] );
    Kokkos::deep_copy( coords_shell_host, coords_shell[velocity_level] );
    auto coords_radii_host = Kokkos::create_mirror_view( coords_radii[velocity_level] );
    Kokkos::deep_copy( coords_radii_host, coords_radii[velocity_level] );

    terra::grid::Grid4DDataScalar< double > plate_id_fe(
        "plate_id_fe",
        domains[velocity_level].subdomains().size(),
        domains[velocity_level].domain_info().subdomain_num_nodes_per_side_laterally(),
        domains[velocity_level].domain_info().subdomain_num_nodes_per_side_laterally(),
        domains[velocity_level].domain_info().subdomain_num_nodes_radially() );

    io::XDMFOutput< double > xdmf_output(
        "./output-xdmf/", domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level] );
    xdmf_output.add( plate_id_fe );

    // --------
    //  Oracle
    // --------

    logroot << "*** STEP 3: Generating an Oracle" << std::endl;

    //std::string dataDir{ "/import/freenas-m-04-students/frezaei/TerraNeoX/TERRA-NG/apps/plate_velocities/data/plates/" };
    const std::string& dataDir = parameters.dataDir;
    std::string fnameTopologies      = dataDir + "topologies_0-410Ma.geojson";
    std::string fnameReconstructions = dataDir + "TomoPAC2.rot";
    terra::plates::PlateVelocityProvider oracle( fnameTopologies, fnameReconstructions );

    logroot << "*** STEP 4: Checking plate stages to work with" << std::endl;

    auto stages = oracle.getListOfPlateStages();

    if ( parameters.beginAge > oracle.getMaxAge() || parameters.endAge < oracle.getMinAge() )
    {
        logroot << "Specified age range extends beyond available plate data. " << std::endl;
        std::abort();
    }

    logroot << " - Running from " << parameters.beginAge << " Ma to " << parameters.endAge << " Ma" << std::endl;

    auto& plates = oracle.plateTopologies().getPlatesForStage( parameters.endAge );

    std::vector< int > plate_data_offset;

    plate_data_offset.reserve( plates.size() + 1 );
    plate_data_offset.push_back( 0 );

    for ( int i_plate = 0; i_plate < plates.size(); i_plate++ )
    {
        PlateStorage::PlateInfo plate = plates[i_plate];

        plate_data_offset.push_back( plate_data_offset[i_plate] + plate.boundary.size() );
    }

    PlateCoordsContainerDeviceView plate_coords_device( "plate_coords_container", plate_data_offset[plates.size()] );
    auto                           plate_coords_host = Kokkos::create_mirror_view( plate_coords_device );

    PlateContainerDeviceView plate_data_device( "plate_data_container", plates.size() );
    auto                     plate_data_host = Kokkos::create_mirror_view( plate_data_device );

    int i_plate_coords_counter = 0;
    for ( int i_plate = 0; i_plate < plates.size(); i_plate++ )
    {
        plate_data_host( i_plate ).id = plates[i_plate].id;

        plate_data_host( i_plate ).vertex_offset = plate_data_offset[i_plate];

        plate_data_host( i_plate ).num_vertices = plate_data_offset[i_plate + 1] - plate_data_offset[i_plate];

        for ( int i_boundary = 0; i_boundary < plates[i_plate].boundary.size(); i_boundary++ )
        {
            plate_coords_host( i_plate_coords_counter ).lon = plates[i_plate].boundary[i_boundary]( 0 );
            plate_coords_host( i_plate_coords_counter ).lat = plates[i_plate].boundary[i_boundary]( 1 );

            i_plate_coords_counter++;
        }
    }

    Kokkos::deep_copy( plate_data_device, plate_data_host );
    Kokkos::deep_copy( plate_coords_device, plate_coords_host );

    uint_t n_plates = plates.size();

    Kokkos::parallel_for(
        "plate_id_interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        PlateIDInterpolator(
            parameters.r_max,
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            plate_id_fe,
            n_plates,
            plate_coords_device,
            plate_data_device ) );

    xdmf_output.write( 0 );

    return EXIT_SUCCESS;
}
