#pragma once

// Adaptive mesh refinement -- forest-of-octree-leaves bookkeeper for the icosahedral shell.
//
// This header is intentionally DEPENDENCY-FREE (no Kokkos, no MPI, no other terra headers): the forest
// is pure integer arithmetic in a "finest-level" index space, so it compiles and unit-tests on the host
// with a bare C++ compiler. Integration with grid::shell::SubdomainInfo happens at the boundary (a thin
// converter, added in a later step) -- the core stays trivially testable.
//
// Index conventions (the whole design rests on these):
//   * D                     = finest octree depth. depth 0 = a base subdomain (today's uniform block);
//                             depth d = split d times.
//   * finest-index space    = coordinates measured in units where a depth-D leaf spans 1. A depth-d leaf
//                             has span s = 2^(D-d) per axis. Its anchor (min corner) is s-aligned.
//   * per diamond           = lateral index in [0, S_lat * 2^D), radial in [0, S_rad * 2^D), where
//                             S_lat = base subdomains per diamond side, S_rad = base radial subdomains.
//   * 10 diamonds           = the icosahedral macro-structure (hardcoded, as in spherical_shell.hpp).
//
// Refining a leaf replaces it with its 8 children (2x2x2), each at depth+1, each covering 1/8 the volume
// but the SAME node count -- which is why it fits terra's single rectangular Kokkos::View storage.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace terra::grid::shell::amr {

// The 6 faces of a block, named by axis (x/y lateral, r radial) and side (low/high).
enum class Face
{
    XLOW,
    XHIGH,
    YLOW,
    YHIGH,
    RLOW,
    RHIGH
};

inline Face opposite( Face f )
{
    switch ( f )
    {
    case Face::XLOW:
        return Face::XHIGH;
    case Face::XHIGH:
        return Face::XLOW;
    case Face::YLOW:
        return Face::YHIGH;
    case Face::YHIGH:
        return Face::YLOW;
    case Face::RLOW:
        return Face::RHIGH;
    case Face::RHIGH:
        return Face::RLOW;
    }
    return f;
}

// A block identifier in finest-level index units (maps to SubdomainInfo's (diamond,x,y,r) at finest res).
struct BrickId
{
    int diamond = -1;
    int x       = -1; // lateral, finest-index units
    int y       = -1; // lateral, finest-index units
    int r       = -1; // radial,  finest-index units

    bool operator==( const BrickId& o ) const
    {
        return diamond == o.diamond && x == o.x && y == o.y && r == o.r;
    }
};

// A leaf of the forest: a block anchored at its min-corner finest index, at octree depth `depth`.
struct ForestLeaf
{
    BrickId anchor;
    int     depth = -1;

    bool operator==( const ForestLeaf& o ) const { return anchor == o.anchor && depth == o.depth; }
};

// One neighbor across a face. rel_level = neighbor.depth - my.depth (0 same, -1 coarser, +1 finer).
// sub_octant (0..3) identifies which quarter of my face a finer neighbor covers; 0 for same/coarser.
struct FaceNeighbor
{
    ForestLeaf leaf;
    int        rel_level;
    int        sub_octant;
    Face       neighbor_face; // the face of `leaf` that touches me (opposite of the queried face)
};

// What lies across a face: interior neighbor(s), the CMB/surface radial boundary, or a diamond seam.
enum class NeighborKind
{
    Interior,
    DomainBoundary,  // radial: CMB or surface -- no neighbor
    DiamondCrossing  // lateral diamond seam -- resolved later via the existing pole logic (Milestone A
                     // keeps such blocks at depth 0, so this is only ever hit by base leaves)
};

struct FaceNeighbors
{
    NeighborKind              kind = NeighborKind::Interior;
    std::vector< FaceNeighbor > neighbors; // empty if boundary; 1 if same/coarser; up to 4 if finer
};

class AdaptiveForest
{
  public:
    static constexpr int kNumDiamonds = 10;

    // Start from a uniform mesh: every base block present at depth 0.
    // finest_level D: deepest refinement permitted. lateral/radial base subdomain counts per diamond.
    AdaptiveForest( int finest_level, int lateral_subdomains_per_diamond, int radial_subdomains )
    : D_( finest_level )
    , S_lat_( lateral_subdomains_per_diamond )
    , S_rad_( radial_subdomains )
    {
        const int base_span = 1 << D_;
        for ( int d = 0; d < kNumDiamonds; ++d )
            for ( int bx = 0; bx < S_lat_; ++bx )
                for ( int by = 0; by < S_lat_; ++by )
                    for ( int br = 0; br < S_rad_; ++br )
                        leaves_.push_back(
                            ForestLeaf{ BrickId{ d, bx * base_span, by * base_span, br * base_span }, 0 } );
        normalize();
    }

    // --- geometry / arithmetic ---------------------------------------------------------------------

    [[nodiscard]] int finest_level() const { return D_; }
    [[nodiscard]] int lateral_extent() const { return S_lat_ << D_; } // finest units, per diamond side
    [[nodiscard]] int radial_extent() const { return S_rad_ << D_; }  // finest units

    // Side length of a depth-`depth` leaf, in finest-index units.
    [[nodiscard]] int span( int depth ) const { return 1 << ( D_ - depth ); }

    [[nodiscard]] bool aligned( const ForestLeaf& l ) const
    {
        const int s = span( l.depth );
        return l.anchor.x % s == 0 && l.anchor.y % s == 0 && l.anchor.r % s == 0;
    }

    [[nodiscard]] bool in_range( const ForestLeaf& l ) const
    {
        const int s = span( l.depth );
        return l.anchor.diamond >= 0 && l.anchor.diamond < kNumDiamonds && l.anchor.x >= 0 &&
               l.anchor.x + s <= lateral_extent() && l.anchor.y >= 0 && l.anchor.y + s <= lateral_extent() &&
               l.anchor.r >= 0 && l.anchor.r + s <= radial_extent();
    }

    // Parent: depth d -> d-1, anchor rounded down to the coarser alignment.
    [[nodiscard]] ForestLeaf parent( const ForestLeaf& l ) const
    {
        const int ps = span( l.depth - 1 ); // = 2 * span(l.depth)
        return ForestLeaf{ BrickId{ l.anchor.diamond, ( l.anchor.x / ps ) * ps, ( l.anchor.y / ps ) * ps,
                                    ( l.anchor.r / ps ) * ps },
                           l.depth - 1 };
    }

    // The 8 children (2x2x2) at depth+1. octant bit 0 = x, bit 1 = y, bit 2 = r.
    [[nodiscard]] std::array< ForestLeaf, 8 > children( const ForestLeaf& l ) const
    {
        const int cs = span( l.depth + 1 ); // half the parent span
        std::array< ForestLeaf, 8 > out{};
        for ( int oct = 0; oct < 8; ++oct )
            out[oct] = ForestLeaf{ BrickId{ l.anchor.diamond, l.anchor.x + ( ( oct >> 0 ) & 1 ) * cs,
                                            l.anchor.y + ( ( oct >> 1 ) & 1 ) * cs,
                                            l.anchor.r + ( ( oct >> 2 ) & 1 ) * cs },
                                   l.depth + 1 };
        return out;
    }

    // --- mutation ----------------------------------------------------------------------------------

    // Replace each given leaf with its 8 children. (2:1 balancing is a separate step.)
    void refine( const std::vector< ForestLeaf >& to_split )
    {
        for ( const auto& l : to_split )
            refine_one( l );
        normalize();
    }

    // Merge each 8-sibling group (identified by any one representative) back into its parent.
    void coarsen( const std::vector< ForestLeaf >& sibling_reps )
    {
        for ( const auto& rep : sibling_reps )
            coarsen_one( rep );
        normalize();
    }

    // Enforce the 2:1 rule: no leaf may have a face-neighbor more than one level finer. Repeatedly
    // split any offending (coarse) leaf until a fixed point. Splitting only ever makes the mesh finer
    // and depth is capped at D, so this terminates. Idempotent on an already-balanced mesh.
    void balance_2to1()
    {
        static constexpr Face kFaces[6] = { Face::XLOW, Face::XHIGH, Face::YLOW,
                                            Face::YHIGH, Face::RLOW, Face::RHIGH };
        bool changed = true;
        while ( changed )
        {
            changed = false;
            std::vector< ForestLeaf > to_split;
            for ( const auto& L : leaves_ )
            {
                if ( L.depth >= D_ )
                    continue;
                bool flagged = false;
                for ( int fi = 0; fi < 6 && !flagged; ++fi )
                {
                    const auto fn = face_neighbors( L, kFaces[fi] );
                    for ( const auto& nb : fn.neighbors )
                        if ( nb.rel_level >= 2 ) // a neighbor is 2+ levels finer -> must split L
                        {
                            to_split.push_back( L );
                            flagged = true;
                            break;
                        }
                }
            }
            if ( !to_split.empty() )
            {
                refine( to_split ); // re-normalizes; next pass re-checks the freshly created children
                changed = true;
            }
        }
    }

    // --- queries -----------------------------------------------------------------------------------

    [[nodiscard]] const std::vector< ForestLeaf >& leaves() const { return leaves_; }
    [[nodiscard]] std::size_t                      size() const { return leaves_.size(); }

    [[nodiscard]] bool contains( const ForestLeaf& l ) const { return lookup_.count( key( l ) ) > 0; }

    // Return the index (into leaves()) of the leaf covering the given finest-index point, if any.
    // O(D): tries candidate depths finest->coarsest, aligning the point to each.
    [[nodiscard]] std::optional< std::size_t > leaf_at( int diamond, int fx, int fy, int fr ) const
    {
        for ( int cd = D_; cd >= 0; --cd )
        {
            const int  shift = D_ - cd;
            ForestLeaf cand{ BrickId{ diamond, ( fx >> shift ) << shift, ( fy >> shift ) << shift,
                                      ( fr >> shift ) << shift },
                             cd };
            auto       it = lookup_.find( key( cand ) );
            if ( it != lookup_.end() )
                return it->second;
        }
        return std::nullopt;
    }

    // All neighbors across one face of `L`. Interior faces return 1 neighbor (same/coarser) or up to 4
    // (finer). Radial-boundary faces return DomainBoundary; lateral diamond-seam faces DiamondCrossing.
    // Detection is by sampling the 2x2 finer sub-face centers just outside the face and looking up which
    // leaf covers each -- one distinct hit => same/coarser, four distinct hits => finer.
    [[nodiscard]] FaceNeighbors face_neighbors( const ForestLeaf& L, Face f ) const
    {
        FaceNeighbors res;
        const int     s      = span( L.depth );
        const int     base[3] = { L.anchor.x, L.anchor.y, L.anchor.r };

        int axis, side; // axis: 0=x,1=y,2=r ; side: +1 high, -1 low
        switch ( f )
        {
        case Face::XLOW:  axis = 0; side = -1; break;
        case Face::XHIGH: axis = 0; side = +1; break;
        case Face::YLOW:  axis = 1; side = -1; break;
        case Face::YHIGH: axis = 1; side = +1; break;
        case Face::RLOW:  axis = 2; side = -1; break;
        default:          axis = 2; side = +1; break; // RHIGH
        }

        const int out_coord = side > 0 ? base[axis] + s : base[axis] - 1;
        const int extent     = ( axis == 2 ) ? radial_extent() : lateral_extent();
        if ( out_coord < 0 || out_coord >= extent )
        {
            res.kind = ( axis == 2 ) ? NeighborKind::DomainBoundary : NeighborKind::DiamondCrossing;
            return res;
        }

        const int a1 = ( axis == 0 ) ? 1 : 0;         // first in-face axis
        const int a2 = ( axis == 2 ) ? 1 : 2;         // second in-face axis
        const int nsamp = ( s > 1 ) ? 2 : 1;
        auto      off   = [&]( int q ) { return s > 1 ? ( q == 0 ? s / 4 : 3 * s / 4 ) : 0; };

        std::map< std::size_t, int > found; // leaf index -> first quadrant it was sampled from
        for ( int qi = 0; qi < nsamp; ++qi )
            for ( int qj = 0; qj < nsamp; ++qj )
            {
                int pt[3];
                pt[axis] = out_coord;
                pt[a1]   = base[a1] + off( qi );
                pt[a2]   = base[a2] + off( qj );
                auto hit = leaf_at( L.anchor.diamond, pt[0], pt[1], pt[2] );
                if ( hit )
                    found.emplace( *hit, qi + 2 * qj );
            }

        const bool single = ( found.size() == 1 );
        for ( const auto& [idx, quad] : found )
        {
            const ForestLeaf& N = leaves_[idx];
            res.neighbors.push_back(
                FaceNeighbor{ N, N.depth - L.depth, single ? 0 : quad, opposite( f ) } );
        }
        return res;
    }

    // True if `L` sits at a diamond corner (both lateral block indices at an extreme). Milestone A
    // forbids refining these so the hardcoded pole stitching stays valid at base level.
    [[nodiscard]] bool touches_diamond_corner( const ForestLeaf& L ) const
    {
        const int sp   = span( L.depth );
        const int bx   = L.anchor.x / sp;
        const int by   = L.anchor.y / sp;
        const int nblk = S_lat_ << L.depth;
        const bool xext = ( bx == 0 || bx == nblk - 1 );
        const bool yext = ( by == 0 || by == nblk - 1 );
        return xext && yext;
    }

    // Consistency check for tests: alignment, in-range, no duplicates, and a volume partition check.
    [[nodiscard]] bool validate() const
    {
        std::unordered_map< uint64_t, int > seen;
        std::uint64_t                       vol = 0;
        for ( const auto& l : leaves_ )
        {
            if ( !aligned( l ) || !in_range( l ) )
                return false;
            if ( ++seen[key( l )] > 1 )
                return false;
            const std::uint64_t s = static_cast< std::uint64_t >( span( l.depth ) );
            vol += s * s * s;
        }
        const std::uint64_t total = static_cast< std::uint64_t >( kNumDiamonds ) * lateral_extent() *
                                    lateral_extent() * radial_extent();
        return vol == total;
    }

  private:
    int D_, S_lat_, S_rad_;

    std::vector< ForestLeaf >                    leaves_; // kept sorted + unique
    std::unordered_map< std::uint64_t, std::size_t > lookup_; // key(leaf) -> index into leaves_

    // Pack (diamond, depth, x, y, r) into 64 bits. x,y,r < 2^18, depth < 2^6, diamond < 2^4.
    static std::uint64_t key( const ForestLeaf& l )
    {
        return ( static_cast< std::uint64_t >( l.anchor.diamond ) << 60 ) |
               ( static_cast< std::uint64_t >( l.depth ) << 54 ) |
               ( static_cast< std::uint64_t >( l.anchor.x ) << 36 ) |
               ( static_cast< std::uint64_t >( l.anchor.y ) << 18 ) |
               ( static_cast< std::uint64_t >( l.anchor.r ) );
    }

    void refine_one( const ForestLeaf& l )
    {
        auto it = lookup_.find( key( l ) );
        if ( it == lookup_.end() )
            return; // not present; ignore
        // Mark for removal by clearing depth; children appended. normalize() rebuilds.
        leaves_[it->second].depth = -1;
        for ( const auto& c : children( l ) )
            leaves_.push_back( c );
    }

    void coarsen_one( const ForestLeaf& rep )
    {
        const ForestLeaf par  = parent( rep );
        const auto       kids = children( par );
        for ( const auto& c : kids )
            if ( !contains( c ) )
                return; // not a complete sibling group; ignore
        for ( const auto& c : kids )
            leaves_[lookup_.at( key( c ) )].depth = -1; // mark removed
        leaves_.push_back( par );
    }

    // Drop removed leaves (depth < 0), sort, dedupe, rebuild lookup.
    void normalize()
    {
        leaves_.erase( std::remove_if( leaves_.begin(), leaves_.end(),
                                       []( const ForestLeaf& l ) { return l.depth < 0; } ),
                       leaves_.end() );
        std::sort( leaves_.begin(), leaves_.end(),
                   []( const ForestLeaf& a, const ForestLeaf& b ) { return key( a ) < key( b ); } );
        leaves_.erase( std::unique( leaves_.begin(), leaves_.end() ), leaves_.end() );
        lookup_.clear();
        for ( std::size_t i = 0; i < leaves_.size(); ++i )
            lookup_[key( leaves_[i] )] = i;
    }
};

} // namespace terra::grid::shell::amr
