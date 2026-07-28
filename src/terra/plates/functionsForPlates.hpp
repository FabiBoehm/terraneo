/*
 * Copyright (c) 2022 Berta Vilacis, Marcus Mohr.
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

#include <limits>
#include <boost/geometry.hpp>
#include <vector>
#include <array>

#include "terra/dense/vec.hpp"
#include "terra/plates/conversions.hpp"
#include "terra/plates/functionsForGeometry.hpp"
#include "terra/plates/functionsForRotations.hpp"
#include "terra/plates/types.hpp"

// namespace terraneo {
namespace plates {

inline vec3D sph2cart( const std::vector< real_t >& lonlat, const real_t radius = static_cast< double >( 1 ) )
{
   vec3D xyz;
   xyz[0] = radius * cos( conversions::degToRad( lonlat[1] ) ) * cos( conversions::degToRad( lonlat[0] ) );
   xyz[1] = radius * cos( conversions::degToRad( lonlat[1] ) ) * sin( conversions::degToRad( lonlat[0] ) );
   xyz[2] = radius * sin( conversions::degToRad( lonlat[1] ) );
   return xyz;
}

typedef boost::geometry::model::point<double, 2, boost::geometry::cs::spherical_equatorial<boost::geometry::degree>> spherical_point;
typedef boost::geometry::model::polygon<spherical_point> polygon_on_sphere;

/// Determine to which plate a point belongs
///
/// The function returns a bool to indicate whether any plate matched, the plate's ID and
/// the distance from this plate's boundary
inline std::tuple< bool, uint_t, double >
    findPlateAndDistance( const double age, const PlateStorage& plateStore, const vec3D& point, uint_t idWhenNoPlateFound )
{
   // query all plates for given age stage
   auto& plates = plateStore.getPlatesForStage( std::ceil( age ) );

   // be pessimistic
   bool   plateFound{ false };
   uint_t plateID{ idWhenNoPlateFound };
   double distance{ std::numeric_limits< double >::max() };

   // Create the point in the surface of a sphere from the library boost::geometry 
   spherical_point pntSph(point[0], point[1]);

   //loop over the plates available
   for ( auto& currentPlate : plates )
   {
      // create the polygon on the surface of a sphere and populate with the plate boundary coordinates
      const Polygon& bdrPolygon = currentPlate.boundary;
      polygon_on_sphere polygonOnSphere;
      for ( int index = 0; index < bdrPolygon.size(); ++index )
      {
         boost::geometry::append(polygonOnSphere.outer(), spherical_point(bdrPolygon[index]( 0 ), bdrPolygon[index]( 1 )));
      }
      // Correct the geometry
      boost::geometry::correct(polygonOnSphere);

      // check if the point belongs to the polygon
      if (boost::geometry::within(pntSph, polygonOnSphere))
      {
         // calculate the distace from the polygon to the point
         boost::geometry::for_each_segment(polygonOnSphere, [&distance, &pntSph](const auto& segment){
            distance = std::min<double>(distance, boost::geometry::distance(segment, pntSph));
         });
         // distance at the surface of the Earth
         distance = distance * plates::constants::earthRadiusInKm; 
         // plate is found 
         plateFound = true;
      }
      else{
         // ;
      }

      // plate found then leave loop
      if ( plateFound )
      {
         plateID = currentPlate.id;
         break;
      }
   }
   return std::make_tuple( plateFound, plateID, distance );
}

/// From the Euler vector compute the surface velocity in xyz
inline vec3D eulerVectorToVelocity( const vec3D& point, vec3D& wXYZ, const double smoothing )
{
   double earthRadius = plates::constants::earthRadiusInKm * static_cast< double >( 1e3 );
   double toms        = static_cast< double >( 3600 * 24 * 365 ); // conversions factor cm/yr -> m/s

   vec3D eVector;
   vec3D pxyz;

   eVector = conversions::degToRad( wXYZ ) * static_cast< double >( 1e-6 );

   // Transform to the point to the xyz in a sphere of earthRadius;
   pxyz    = conversions::sph2cart( { point[0], point[1] }, earthRadius );
   vec3D v = eVector.cross( pxyz );

   v *= smoothing / toms;

   return v;
}

/// Get the velocity in given the plate id, create the reconstruction path, get
/// the rotations and calculate the velocity
inline vec3D computeCartesianVelocityVector( const PlateRotationProvider& rotData,
                                             const int                    plateID,
                                             const double                 age,
                                             const vec3D&                 point,
                                             const double                 smoothing )
{
   // age of the euler pole is defined by ((age1 + age2)/2)
   // This is valid when the velocities are calculated every 1 Myrs. 
   // Otherwise this needs to be changed to the desired time step
   // taking into conserdation the time resolution of the plate boundaries available.
   std::array< double, 2 >     time{ age, age + 1 };
   std::vector< RotationInfo > recTree;

   std::vector< FiniteRotation >      FinRot;
   const std::vector< RotationInfo >& rotations = rotData.getRotations();

   int pID = plateID;

   using rotIter_t = std::vector< RotationInfo >::const_iterator;

   while ( pID != 0 )
   {
      rotIter_t rangeBegin;
      rotIter_t rangeEnd;

      for ( rotIter_t it = rotations.begin(); it != rotations.end(); ++it )
      {
         if ( it->plateID == pID )
         {
            rangeBegin = it;
            rangeEnd   = rangeBegin + 1;
            while ( rangeEnd->plateID == pID )
            {
               ++rangeEnd;
            }
            // append to list of finite rotations
            pID = plates::determineSeriesOfFiniteRotations( rangeBegin, rangeEnd, time, FinRot );
            break;
         }
      }
      // WALBERLA_LOG_DETAIL_ON_ROOT( "Looping ... (pID = " << pID << ")" );
   }

   std::array< FiniteRotation, 2 > finNahs = plates::combineSeriesOfFiniteRotations( FinRot );

   // compute Euler Vector
   vec3D lonlatang = plates::stagePoleF( finNahs[0].lonLatAng, finNahs[1].lonLatAng );
   lonlatang[2]    = lonlatang[2] / ( finNahs[1].time - finNahs[0].time );
   vec3D wXYZ      = conversions::sph2cart( { lonlatang[0], lonlatang[1] }, lonlatang[2] );

   return eulerVectorToVelocity( point, wXYZ, smoothing );
}

} // namespace plates
// } // namespace terraneo
