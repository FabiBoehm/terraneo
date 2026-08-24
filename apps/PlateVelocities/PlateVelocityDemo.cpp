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
#include "terra/plates/PlateVelocityProvider.hpp"
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

template < typename T >
std::string demangledTypeName( const T& obj )
{
    int         status    = 0;
    char*       demangled = abi::__cxa_demangle( typeid( obj ).name(), nullptr, nullptr, &status );
    std::string result    = ( status == 0 ) ? demangled : typeid( obj ).name();
    free( demangled );
    return result;
}

typedef enum
{
    PLATE_IDS,
    VELOCITIES,
    VELOCITIES_AND_IDS
} job_t;

using uint_t = unsigned int;

std::string genOutputFileName( double age )
{
    std::stringstream sstr;
    sstr << "PlateVelocitiesDemo_age=";
    sstr << std::setfill( '0' );
    sstr << std::setw( 5 );
    sstr << std::fixed;
    sstr << std::setprecision( 1 );
    sstr << age;
    return sstr.str();
}

std::tuple< uint_t, uint_t > decodeRangeString( std::string& rangeStr )
{
    size_t fPos = rangeStr.find( '-' );
    if ( fPos == std::string::npos )
    {
        logall << "Cannot decode range string '" << rangeStr << "'\n";
        std::abort();
    }

    uint_t stageInit = stoul( rangeStr.substr( 0, fPos ) );
    uint_t stageStop = stoul( rangeStr.substr( fPos + 1 ) );

    return std::make_tuple( stageInit, stageStop );
}

template < typename GridType, typename RadiiType, typename DataType >
struct PlateIDInterpolator 
{
    GridType  grid_;
    RadiiType radii_;
    DataType  data_;

    std::function< double( const vec3D& ) > findPlateID;

    PlateIDInterpolator( 
        const GridType&  grid, 
        const RadiiType& radii,
        const DataType&  data,      
        std::function< double( const vec3D& ) > plateIDFn )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , findPlateID( std::move( plateIDFn ) )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y ) const
    {
        const dense::Vec< double, 3 > coords =
            grid::shell::coords( local_subdomain_id, x, y, radii_.extent( 1 ) - 1, grid_, radii_ );

        data_( local_subdomain_id, x, y, radii_.extent( 1 ) - 1 ) = findPlateID( coords );
    }
};

template < typename GridType, typename RadiiType, typename DataType >
struct PlateVelocityInterpolator
{
    GridType  grid_;
    RadiiType radii_;
    DataType  data_;

    std::function< double( const vec3D& ) > computeVelocityComponent;

    PlateVelocityInterpolator(
        const GridType&  grid,
        const RadiiType& radii,
        const DataType&  data, 
        std::function< double( const vec3D& ) > velocityFn )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , computeVelocityComponent( std::move( velocityFn ) )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y ) const
    {
        const dense::Vec< double, 3 > coords =
            grid::shell::coords( local_subdomain_id, x, y, radii_.extent( 1 ) - 1, grid_, radii_ );

        data_( local_subdomain_id, x, y, radii_.extent( 1 ) - 1 ) = computeVelocityComponent( coords );
    }
};

// =================================================================================
//  Function to test computation of a velocity field from plate reconstruction data
//  for all DoFs on the surface of a sphere
// =================================================================================
// template < typename terra::grid::Grid4DDataScalar< double > >
void performComputations(
    uint_t                                    level,
    std::vector< DistributedDomain >          domains,
    std::vector< Grid3DDataVec< double, 3 > > coords_shell,
    std::vector< Grid2DDataScalar< double > > coords_radii,
    double                                    age,
    job_t                                     jobType,
    plates::PlateVelocityProvider&            oracle,
    std::string                               indent,
    std::string                               xdmf_dir,
    io::XDMFOutput< double >                  xdmf_output,
    uint_t                                    currentStage )
{
    // need that here, to capture it below ;-)
    uint_t                                 coordIdx = 0;
    plates::StatisticsPlateNotFoundHandler handlerWithStatistics;

    // plates::UniformCirclesPointWeightProvider avgPointProvider( { { 1.0 / 100.0, 6 }, { 1.0 / 50.0, 12 } }, 1e-1 );
    plates::UniformCirclesPointWeightProvider avgPointProvider( { { 1.0 / 100.0, 6 } }, 1e-1 );

    // callback function for computing the velocity components
    std::function< double( const vec3D& ) > computeVelocityComponent =
        [&oracle, age, &coordIdx, &handlerWithStatistics, &avgPointProvider]( const vec3D& point ) {
            vec3D coords{ point( 0 ), point( 1 ), point( 2 ) };

            vec3D velocity =
                oracle.getLocallyAveragedPointVelocity( coords, age, avgPointProvider, handlerWithStatistics );
            // vec3D velocity =
            //     oracle.getPointVelocity( coords, age, terraneo::plates::LinearDistanceSmoother{ 0.015 }, handlerWithStatistics );
            return velocity( static_cast< int >( coordIdx ) );
        };

    // callback function for determining plate IDs
    std::function< double( const vec3D& ) > findPlateID = [&oracle, age, &handlerWithStatistics]( const vec3D& point ) {
        vec3D  coords{ point( 0 ), point( 1 ), point( 2 ) };
        uint_t id = oracle.findPlateID( coords, age );
        if ( id == oracle.idWhenNoPlateFound )
        {
            handlerWithStatistics( coords, age );
        }
        return id;
    };

    // use pointers so that we only request memory for function in the desired jobType
    std::shared_ptr< terra::grid::Grid4DDataScalar< double > > surfaceVelocity{ nullptr };
    std::shared_ptr< terra::grid::Grid4DDataScalar< double > > plateID{ nullptr };

    surfaceVelocity = std::make_shared< terra::grid::Grid4DDataScalar< double > >(
        "plateVelocities",
        coords_shell[level].extent( 0 ),
        coords_shell[level].extent( 1 ),
        coords_shell[level].extent( 2 ),
        coords_radii[level].extent( 1 ) );

    plateID = std::make_shared< terra::grid::Grid4DDataScalar< double > >(
        "plateID",
        coords_shell[level].extent( 0 ),
        coords_shell[level].extent( 1 ),
        coords_shell[level].extent( 2 ),
        coords_radii[level].extent( 1 ) );

    std::cout << " Kokkos::DefaultExecutionSpace::name() " << Kokkos::DefaultExecutionSpace::name() << std::endl;

    using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

    // Mirror Kokkos::Views for host-access
    auto coords_shell_host = Kokkos::create_mirror_view( coords_shell[level] );
    Kokkos::deep_copy( coords_shell_host, coords_shell[level] );

    auto coords_radii_host = Kokkos::create_mirror_view( coords_radii[level] );
    Kokkos::deep_copy( coords_radii_host, coords_radii[level] );

    auto plateID_host = Kokkos::create_mirror_view( *plateID );
    auto surfaceVelocity_host = Kokkos::create_mirror_view( *surfaceVelocity );

    Kokkos::parallel_for(
        "Plate ID interpolation",
        Kokkos::MDRangePolicy< HostExecSpace, Kokkos::Rank< 3 > >(
            { 0, 0, 0 },
            { coords_shell_host.extent( 0 ), coords_shell_host.extent( 1 ), coords_shell_host.extent( 2 ) } ),
        PlateIDInterpolator( coords_shell_host, coords_radii_host, ( plateID_host ), findPlateID ) );

    Kokkos::fence();

    Kokkos::parallel_for(
        "Plate Velocity interpolation",
        Kokkos::MDRangePolicy< HostExecSpace, Kokkos::Rank< 3 > >(
            { 0, 0, 0 },
            { coords_shell_host.extent( 0 ), coords_shell_host.extent( 1 ), coords_shell_host.extent( 2 ) } ),
        PlateVelocityInterpolator(
            coords_shell_host, coords_radii_host, ( surfaceVelocity_host ), computeVelocityComponent ) );

    Kokkos::fence();

    // Copy results back to original Views
    Kokkos::deep_copy( *plateID, plateID_host );
    Kokkos::deep_copy( *surfaceVelocity, surfaceVelocity_host );

    // export results
    std::string fName = genOutputFileName( age );

    logroot << indent << "Exporting simulation data to file with basename '" << fName << "'" << std::endl;

    xdmf_output.add( *plateID );
    xdmf_output.add( *surfaceVelocity );

    xdmf_output.set_write_counter( currentStage );
    xdmf_output.write();
}
} // namespace plates
} // namespace terra

struct Parameters
{
    uint_t      max_level = 5;
    uint_t      min_level = 1;
    double      r_min     = 1.2;
    double      r_max     = 2.2;
    std::string jobType   = "both";
    int         beginAge  = 27;
    int         endAge    = 30;
    std::string outdir    = "./output";
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

    terra::util::prepare_empty_directory_or_abort( parameters.outdir );

    const auto xdmf_dir            = parameters.outdir + "/xdmf";
    const auto radial_profiles_dir = parameters.outdir + "/radial_profiles";
    const auto timer_trees_dir     = parameters.outdir + "/timer_trees";

    terra::util::prepare_empty_directory_or_abort( xdmf_dir );
    terra::util::prepare_empty_directory_or_abort( radial_profiles_dir );
    terra::util::prepare_empty_directory_or_abort( timer_trees_dir );

    logroot << "Running with the following steering parameters:" << std::endl;
    logroot << "--max-level" << parameters.max_level << std::endl;
    logroot << "--min-level" << parameters.min_level << std::endl;
    logroot << "--r-min" << parameters.r_min << std::endl;
    logroot << "--r-max" << parameters.r_max << std::endl;
    logroot << "--jobType" << parameters.jobType << std::endl;
    logroot << "--beginAge" << parameters.beginAge << std::endl;
    logroot << "--endAge" << parameters.endAge << std::endl;

    // determine job type
    terra::plates::job_t jobType;
    jobType = terra::plates::VELOCITIES_AND_IDS;

    // determine age indices
    std::string rangeStr        = " 10 - 11 ";
    auto [stageInit, stageStop] = terra::plates::decodeRangeString( rangeStr );

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

        domains.push_back(
            terra::grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
                level, level, parameters.r_min, parameters.r_max ) );
        coords_shell.push_back(
            terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domains[idx] ) );
        coords_radii.push_back( terra::grid::shell::subdomain_shell_radii< double >( domains[idx] ) );
    }

    const auto   num_levels     = domains.size();
    const uint_t velocity_level = num_levels - 1;
    std::cout << "velocity level : " << velocity_level << std::endl;

    // --------
    //  Oracle
    // --------

    logroot << "*** STEP 3: Generating an Oracle" << std::endl;

    std::string dataDir{ "/import/freenas-m-04-students/frezaei/TerraNeoX/TERRA-NG/apps/PlateVelocities/data/plates/" };
    std::string fnameTopologies      = dataDir + "topologies0-100Ma.geojson";
    std::string fnameReconstructions = dataDir + "Global_EarthByte_230-0Ma_GK07_AREPS.rot";
    terra::plates::PlateVelocityProvider oracle( fnameTopologies, fnameReconstructions );

    logroot << "*** STEP 4: Checking plate stages to work with" << std::endl;

    auto stages = oracle.getListOfPlateStages();
    if ( stageStop >= stages.size() )
    {
        logall << "ageRange parameter does not work for available plate stages!" << std::endl;
        std::abort();
    }

    logroot << " - Running from stage[" << stageInit << "] = " << stages[stageInit] << " Ma back to stage[" << stageStop
            << "] = " << stages[stageStop] << " Ma" << std::endl;

    // ------------
    //  Delegation
    // ------------

    logroot << "*** STEP 5: Running the actual computations" << std::endl;

    terra::io::XDMFOutput xdmf_output(
        xdmf_dir, ( domains[velocity_level] ), coords_shell[velocity_level], coords_radii[velocity_level] );

    for ( uint_t currentStage = stageInit; currentStage <= stageStop; ++currentStage )
    {
        logroot << " - stage[" << std::setw( 3 ) << currentStage << "] = " << std::fixed << std::setprecision( 1 )
                << stages[currentStage] << " Ma" << std::endl;

        performComputations(
            velocity_level,
            domains,
            coords_shell,
            coords_radii,
            stages[currentStage],
            jobType,
            oracle,
            "   ",
            xdmf_dir,
            xdmf_output,
            currentStage );
    }

    return EXIT_SUCCESS;
}
