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

// #include "core/math/Constants.h"

// #include "terraneo/helpers/typeAliases.hpp"
#include "terra/dense/vec.hpp"

namespace plates::conversions {

inline constexpr long double pi = 3.14159265358979323846L;

using vec3D = dense::Vec< double, 3 >; 

/// Transforms angle from degrees to radians
inline double degToRad( double degree )
{
   return ( degree * ( pi / real_c( 180 ) ) );
}

/// Transforms vector of angles componentwise from degrees to radians
inline vec3D degToRad( vec3D degree )
{
   return ( degree * ( pi / real_c( 180 ) ) );
}

/// Transforms angle from radians to degrees
inline double radToDeg( double radian )
{
   return ( radian * ( real_c( 180 ) / pi ) );
}

/// Transforms vector of angles componentwise from radians to degrees
inline vec3D radToDeg( vec3D radian )
{
   return ( radian * ( real_c( 180 ) / pi ) );
}

/// Transform 3D vector from spherical to cartesian coordintates
///
/// Transform 3D vector from spherical coordinates (lon, lat, rad) to cartesian
/// ones (x,y,z), is radius is not given assume unit point on sphere.
inline vec3D sph2cart( const std::vector< double >& lonlat, const double radius = real_c( 1 ) )
{
   vec3D xyz;
   xyz[0] = radius * cos( degToRad( lonlat[1] ) ) * cos( degToRad( lonlat[0] ) );
   xyz[1] = radius * cos( degToRad( lonlat[1] ) ) * sin( degToRad( lonlat[0] ) );
   xyz[2] = radius * sin( degToRad( lonlat[1] ) );
   return xyz;
}

/// Transform 3D vector from cartesian (x, y, z) to spherical coordinates (lon, lat, rad)
vec3D cart2sph( const vec3D& xyz )
{
   vec3D lonlatrad;
   lonlatrad[0] = radToDeg( atan2( xyz[1], xyz[0] ) );
   lonlatrad[1] = radToDeg( atan2( xyz[2], sqrt( xyz[0] * xyz[0] + xyz[1] * xyz[1] ) ) );
   lonlatrad[2] = xyz.norm();

   return lonlatrad;
}

} // namespace terraneo::conversions
