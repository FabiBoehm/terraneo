#pragma once

// Adaptive (2:1) face-neighbor tables for the AMR forest.
//
// The uniform SubdomainNeighborhood allows exactly one neighbor per face. An adaptive mesh needs up to
// four (a coarse face abuts 2x2 finer faces), each tagged with its relative refinement. This is a
// SEPARATE, dependency-free structure (no Kokkos/MPI) so the uniform neighbor/halo path is untouched;
// the halo and hanging-node steps consume it.
//
// Each neighbor is identified by its finest-frame anchor (the unique key also used by the adaptive
// DistributedDomain), so a consumer can look up the neighbor's local index / subdivision / owner rank.

#include <map>
#include <vector>

#include "adaptive_forest.hpp"
#include "subdomain_info.hpp"

namespace terra::grid::shell::amr {

// One face-neighbor of a leaf.
struct AdaptiveFaceNeighbor
{
    SubdomainInfo anchor;        // finest-frame anchor of the neighbor leaf (unique global key)
    Face          my_face;       // which of my faces this neighbor sits across
    Face          neighbor_face; // the neighbor's face touching me (opposite of my_face)
    int           rel_level;     // neighbor.subdivision - my.subdivision: 0 same, -1 coarser, +1 finer
    int           sub_octant;    // which quarter (0..3) of my face a finer neighbor covers; 0 otherwise
    int           rank;          // MPI rank owning the neighbor
};

// All face-neighbors of one leaf (across all 6 faces; boundary/diamond-seam faces contribute nothing).
struct AdaptiveNeighborhood
{
    std::vector< AdaptiveFaceNeighbor > faces;
};

// Build the face-neighbor list of a single leaf. `rank_of` maps a finest-frame anchor to its owner
// rank (templated to keep this header free of mpi types).
template < typename RankOf >
inline AdaptiveNeighborhood face_neighborhood_of( const AdaptiveForest& forest, const ForestLeaf& leaf,
                                                  RankOf rank_of )
{
    static constexpr Face kFaces[6] = { Face::XLOW, Face::XHIGH, Face::YLOW,
                                        Face::YHIGH, Face::RLOW, Face::RHIGH };
    AdaptiveNeighborhood nbh;
    for ( Face f : kFaces )
    {
        const auto fn = forest.face_neighbors( leaf, f );
        if ( fn.kind != NeighborKind::Interior )
            continue;
        for ( const auto& nb : fn.neighbors )
        {
            const SubdomainInfo nanchor = forest.finest_anchor( nb.leaf );
            nbh.faces.push_back(
                AdaptiveFaceNeighbor{ nanchor, f, nb.neighbor_face, nb.rel_level, nb.sub_octant,
                                      rank_of( nanchor ) } );
        }
    }
    return nbh;
}

} // namespace terra::grid::shell::amr
