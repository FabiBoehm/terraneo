/*
 * Copyright (c) 2022-2025 Berta Vilacis, Marcus Mohr, Nils Kohl.
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

#pragma once

// #include "terraneo/dataimport/FileIO.hpp"
#include "terra/plates/PlateNotFoundHandlers.hpp"
#include "terra/plates/PlateRotationProvider.hpp"
#include "terra/plates/PlateStorage.hpp"
#include "terra/plates/SmoothingStrategies.hpp"
#include "terra/plates/conversions.hpp"

// preserve ordering of includes
#include "LocalAveragingPointWeightProvider.hpp"
#include "terra/plates/functionsForPlates.hpp"

// namespace terraneo {
namespace plates {

/// API class for computation of velocities from plate reconstructions
class PlateVelocityProvider
{
  public:
    PlateVelocityProvider( std::string nameOfTopologiesFile, std::string nameOfRotationsFile )
    : plateTopologies_(
          nameOfTopologiesFile,
          static_cast< double >( 1 ),
          []( const std::string& filename ) { return io::readJsonFile( filename ); } )
    , plateRotations_( nameOfRotationsFile, []( const std::string& filename ) {
        return io::readRotationsFile( filename );
    } )
    {}

    /// Returns the plate ID for a point at a given age stage
    ///
    /// This is basically an auxilliary function for testing plate detection and allows to
    /// generate data to visualise plate movement.
    uint_t findPlateID( const vec3D& point, const double age ) const
    {
        uint_t plateID{ idWhenNoPlateFound };
        bool   plateFound{ false };
        double distance{ static_cast< double >( -1 ) };

        // Transform the point to Lon, Lat, Radius - to preform all caculations
        // We use the Lon, Lat coordinates
        const vec3D pointLonLat = terraneo::conversions::cart2sph( point );

        std::tie( plateFound, plateID, distance ) =
            findPlateAndDistance( age, plateTopologies_, pointLonLat, idWhenNoPlateFound );
        return plateID;
    }

    // This function checks if a point is on/very close to the boundary at a certain age
    // Primarily written to be used in the viscosity function of a Circulation Model
    bool findPlateBoundaries( const vec3D& point, const double age )
    {
        double eps = static_cast< double >( 1e-2 ); // 3e-2
        uint_t plateID{ 0 };
        bool   plateFound{ false };
        double distance{ static_cast< double >( -1 ) };

        // Transform the point to Lon, Lat, Radius - to perform all calculations
        // We use the Lon, Lat coordinates
        const vec3D pointLonLat = conversions::cart2sph( point );

        std::tie( plateFound, plateID, distance ) =
            findPlateAndDistance( age, plateTopologies_, pointLonLat, idWhenNoPlateFound );
        distance /= plates::constants::earthRadiusInKm;

        if ( distance < eps )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Returns velocity vector for a point determined from the velocity of the associated plate at given age stage
    ///
    /// This is the convenience (non-expert) version of the method which uses a
    /// - LinearDistanceSmoother{ 0.015 }
    /// - DefaultPlateNotFoundHandler{}
    vec3D getPointVelocity( const vec3D& point, const double age )
    { return getPointVelocity( point, age, LinearDistanceSmoother{ 0.015 }, DefaultPlateNotFoundHandler{} ); }

    /// Returns velocity vector for a point determined from the velocity of the associated plate at given age stage
    ///
    /// This is the expert version of the method which allows to explicitly set a SmoothingStrategy and
    /// a PlateNotFoundStrategy.
    template < typename SmoothingStrategy, typename PlateNotFoundStrategy >
    vec3D getPointVelocity(
        const vec3D&            point,
        const double            age,
        SmoothingStrategy       computeSmoothing,
        PlateNotFoundStrategy&& errorHandler )
    {
        uint_t plateID{ 0 };
        bool   plateFound{ false };
        double distance{ static_cast< double >( -1 ) };

        // Transform the point to Lon, Lat, Radius - to preform all calculations
        // We use the Lon, Lat coordinates
        vec3D pointLonLat = conversions::cart2sph( point );

        std::tie( plateFound, plateID, distance ) =
            findPlateAndDistance( age, plateTopologies_, pointLonLat, idWhenNoPlateFound );

        if ( !plateFound )
        {
            return errorHandler( point, age );
        }
        // else
        // {
        //    WALBERLA_LOG_DETAIL_ON_ROOT( "Point found on plate with ID = " << plateID << ", distance to boundary = " << distance );
        // }

        const double smoothingFactor = computeSmoothing( distance );
        // if ( mpi::rank == 0 )
        // {
            logroot << "Smoothing Factor: " << smoothingFactor << "\n"
                      << "Plate ID: " << plateID << std::endl;
        // }

        return computeCartesianVelocityVector( plateRotations_, plateID, age, pointLonLat, smoothingFactor );
    }

    /// Computes a weighted average of the velocity around the given point using the provided point and weights.
    ///
    /// Averaging is only applied if at least one of the provided points is located on a different plate.
    template < typename PlateNotFoundStrategy >
    vec3D getLocallyAveragedPointVelocity(
        const vec3D&                             point,
        const double                             age,
        const LocalAveragingPointWeightProvider& pointWeightProvider,
        PlateNotFoundStrategy&&                  errorHandler )
    {
        uint_t plateID{ 0 };
        bool   plateFound{ false };
        double distance{ static_cast< double >( -1 ) };

        // Transform the point to Lon, Lat, Radius - to perform all calculations
        // We use the Lon, Lat coordinates
        const vec3D pointLonLat = conversions::cart2sph( point );

        std::tie( plateFound, plateID, distance ) =
            findPlateAndDistance( age, plateTopologies_, pointLonLat, idWhenNoPlateFound );

        if ( !plateFound )
        {
            return errorHandler( point, age );
        }

        vec3D  avgVelCart( 0, 0, 0 );
        double weightSum = 0;

        if ( pointWeightProvider.maxDistance( pointLonLat ) < distance )
        {
            // We do not apply averaging since all points that would be used for averaging are on the same plate.
            // if ( mpi::rank() == 0 )
            // {
                logroot << "No averaging.\n" << "Plate ID: " << plateID << std::endl;
            // }

            return computeCartesianVelocityVector( plateRotations_, plateID, age, pointLonLat, 1.0 );
        }

        const auto pointsAndWeights = pointWeightProvider.samplePointsAndWeightsLonLat( pointLonLat );

        // We average since at least some of the samples are possibly on at least one other plate.
        for ( const auto& [samplePointSphLonLat, weight] : pointsAndWeights )
        {
            uint_t avgPointPlateID{ 0 };
            bool   avgPointPlateFound{ false };
            double avgPointDistance{ static_cast< double >( -1 ) };

            std::tie( avgPointPlateFound, avgPointPlateID, avgPointDistance ) =
                findPlateAndDistance( age, plateTopologies_, samplePointSphLonLat, idWhenNoPlateFound );

            if ( avgPointPlateFound )
            {
                // This is possibly slightly inaccurate since we are averaging over the cartesian velocity vectors and then projecting
                // out the normal component. It would be better to average in the "lonlat-space" and then convert and return the
                // cartesian vector. On the other hand, averaging the plate velocities is already a somewhat arbitrary and physically
                // meaningless approximation in the first place, so this might just work.
                avgVelCart += weight * computeCartesianVelocityVector(
                                           plateRotations_, avgPointPlateID, age, samplePointSphLonLat, 1.0 );
                weightSum += weight;
            }
        }

        avgVelCart /= weightSum;

        const auto n                = point.normalized();
        const auto dot              = n.dot( avgVelCart );
        const auto normalComponent  = dot * n;
        const auto tangentComponent = avgVelCart - normalComponent;

        return tangentComponent;
    }

    /// Query function to obtain a vector of plate stages available in the datafiles
    const std::vector< double >& getListOfPlateStages() const { return plateTopologies_.getListOfPlateStages(); }

    /// Plate ID to be used when no associated plate was found for a point
    const uint_t idWhenNoPlateFound{ 0 };

    double getMinAge() const { return plateTopologies_.getMinAge(); }
    double getMaxAge() const { return plateTopologies_.getMaxAge(); }

  private:
    PlateStorage          plateTopologies_;
    PlateRotationProvider plateRotations_;
};

} // namespace plates
// } // namespace terraneo
