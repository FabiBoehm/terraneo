#pragma once

// Depth-aware geometry for the AMR forest: where a subdivision-k leaf's nodes sit in space.
//
// Dependency-free host helpers (no Kokkos). The Kokkos coordinate/radii builders in
// spherical_shell.hpp call these to place adaptive leaves; keeping the math here lets it be unit-tested
// on the host and keeps the two ideas separate.
//
// LATERAL: a subdivision-k leaf is exactly the existing per-subdomain coordinate routine evaluated at
// the effective diamond refinement (LDR + k) with (S_lat << k) subdomains per side, at the leaf's
// own-level block index. Block size = 2^(LDR+k) / (S_lat<<k) = 2^LDR / S_lat = the base node count, so
// every leaf keeps the base-block resolution. A subdivision-0 leaf reproduces the uniform mesh exactly.
//
// RADIAL: a subdivided leaf's shells fall between the base shell radii, obtained by linear (norm-based)
// interpolation of the base radii -- the same rule the linear prolongation uses. A subdivision-0 leaf
// lands exactly on the base radii.

#include <cmath>
#include <vector>

#include "subdomain_info.hpp"

namespace terra::grid::shell::amr {

// Own-level index along one axis: a leaf's finest-frame anchor coordinate divided back down to its own
// subdivision grid. (anchor = own << (M - subdivision), so own = anchor >> (M - subdivision).)
inline int own_level_index( int finest_anchor_coord, int max_subdivision, int subdivision )
{
    return finest_anchor_coord >> ( max_subdivision - subdivision );
}

// Parameters to feed the existing unit_sphere_single_shell_subdomain_coords(...) for a leaf.
struct LateralCoordParams
{
    int global_refinements;      // effective diamond lateral refinement = LDR + subdivision
    int num_subdomains_per_side; // effective subdomains per diamond side = S_lat << subdivision
    int subdomain_i;             // own-level lateral x index of the leaf
    int subdomain_j;             // own-level lateral y index of the leaf
};

inline LateralCoordParams adaptive_lateral_coord_params( int                  lateral_diamond_refinement_level,
                                                         int                  lateral_subdomains_base,
                                                         int                  max_subdivision,
                                                         int                  subdivision,
                                                         const SubdomainInfo& finest_anchor )
{
    return LateralCoordParams{
        lateral_diamond_refinement_level + subdivision,
        lateral_subdomains_base << subdivision,
        own_level_index( finest_anchor.subdomain_x(), max_subdivision, subdivision ),
        own_level_index( finest_anchor.subdomain_y(), max_subdivision, subdivision ) };
}

// Radius of node j (0..layers_per_subdomain) of a subdivision-k leaf, by linear interpolation of the
// base shell radii. layers_per_subdomain = (base_radii.size()-1) / radial_subdomains_base.
inline double adaptive_shell_radius( const std::vector< double >& base_radii,
                                     int                          radial_subdomains_base,
                                     int                          max_subdivision,
                                     int                          subdivision,
                                     const SubdomainInfo&         finest_anchor,
                                     int                          node_j )
{
    const int num_layers          = static_cast< int >( base_radii.size() ) - 1;
    const int layers_per_subdomain = num_layers / radial_subdomains_base;
    const int own_r                = own_level_index( finest_anchor.subdomain_r(), max_subdivision, subdivision );

    // Continuous base-layer coordinate of node j: (own_r * layers_per_subdomain + j) / 2^subdivision.
    const double c  = static_cast< double >( own_r * layers_per_subdomain + node_j ) / ( 1 << subdivision );
    const int    i0 = static_cast< int >( std::floor( c ) );
    if ( i0 >= num_layers )
        return base_radii[num_layers]; // outermost shell (c == num_layers exactly)
    const double frac = c - i0;
    return ( 1.0 - frac ) * base_radii[i0] + frac * base_radii[i0 + 1];
}

} // namespace terra::grid::shell::amr
