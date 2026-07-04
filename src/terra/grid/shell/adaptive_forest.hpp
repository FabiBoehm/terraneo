#pragma once

// Adaptive mesh refinement -- forest-of-octree-leaves bookkeeper for the icosahedral shell.
//
// A leaf is one macro-block: terra's own grid::shell::SubdomainInfo (diamond, x, y, r) plus a
// `subdivision` count = how many times that base subdomain has been split (0 = a base block, exactly
// as terra creates today). Splitting a subdomain replaces it with its 8 children (2x2x2), each at
// subdivision+1, each covering 1/8 the volume but the SAME node count -- which is why it fits terra's
// single rectangular Kokkos::View storage.
//
// Indices are at the block's OWN subdivision grid: a subdivision-k block lives on a lateral grid of
// S_lat * 2^k per diamond side (radial S_rad * 2^k). So a uniform forest with every leaf at
// subdivision 0 IS today's uniform mesh, using the identical SubdomainInfo indices -- no translation.
//
// Note: `subdivision` is a NEW axis (splitting subdomains). It is orthogonal to terra's multigrid
// "refinement level" (node density within a fixed subdomain); the two must not be conflated.
//
// Internally the neighbor/balance arithmetic maps a leaf to a "finest-frame" interval on the fly
// (index << (max_subdivision - subdivision)); the finest frame is a common grid where a leaf at the
// deepest subdivision spans 1. Nothing is stored twice -- the finest frame is derived, not kept.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "subdomain_info.hpp"

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

// A leaf of the forest: a terra subdomain at a given subdivision.
struct ForestLeaf
{
    SubdomainInfo id;             // (diamond, x, y, r), indices at this leaf's own subdivision grid
    int           subdivision = -1;

    bool operator==( const ForestLeaf& o ) const { return id == o.id && subdivision == o.subdivision; }
};

// Which diamond lies across a lateral face of diamond `d`, which of ITS faces touches us, and whether
// the in-face lateral coordinate reverses (the radial coordinate never does). Transcribed from the
// uniform pole wiring in spherical_shell.hpp (setup_neighborhood_faces): same-pole neighbors iterate
// forward, cross-pole neighbors backward. Cross-checked against SubdomainNeighborhood by the tests.
struct DiamondSeam
{
    int  nb_diamond;
    Face nb_face;
    bool reversed;
};

inline DiamondSeam diamond_seam( int d, Face f )
{
    const bool north = d < 5;
    const int  j     = d % 5;
    switch ( f )
    {
    case Face::XLOW:  return { north ? ( j + 1 ) % 5 : ( j + 1 ) % 5 + 5, Face::YLOW, false };
    case Face::YLOW:  return { north ? ( j + 4 ) % 5 : ( j + 4 ) % 5 + 5, Face::XLOW, false };
    case Face::YHIGH: return { north ? j + 5 : ( j + 1 ) % 5, Face::XHIGH, true };
    case Face::XHIGH: return { north ? ( j + 4 ) % 5 + 5 : j, Face::YHIGH, true };
    default:          return { -1, f, false }; // radial faces never cross a seam
    }
}

// One neighbor across a face. rel_level = neighbor.subdivision - my.subdivision (0 same, -1 coarser,
// +1 finer). sub_octant (0..3) identifies which quarter of my face a finer neighbor covers (always
// labeled in MY frame; 0 otherwise). For cross-diamond neighbors, seam_reversed says the neighbor's
// in-face lateral index runs backwards relative to mine.
struct FaceNeighbor
{
    ForestLeaf leaf;
    int        rel_level;
    int        sub_octant;
    Face       neighbor_face;         // the face of `leaf` that touches me
    bool       seam_reversed = false; // cross-diamond only; lateral in-face reversal
};

// What lies across a face. face_neighbors only ever sets Interior or DomainBoundary: a lateral diamond
// seam is resolved INLINE (probe points are transformed into the neighbor diamond's frame) and reported
// as Interior with the neighbor(s) filled in -- including finer neighbors, so seam-edge blocks refine
// freely; node classes across the seam are merged later by union-find in build_2to1_tables. Only
// pole/CORNER blocks (both lateral indices extreme) are restricted from non-conforming refinement
// (see corner_guard in adaptive_exchange.hpp).
enum class NeighborKind
{
    Interior,
    DomainBoundary,  // radial: CMB or surface -- no neighbor
    DiamondCrossing  // reserved/unused: seam crossings are reported as Interior (see note above)
};

struct FaceNeighbors
{
    NeighborKind                kind = NeighborKind::Interior;
    std::vector< FaceNeighbor > neighbors; // empty if boundary; 1 if same/coarser; up to 4 if finer
};

class AdaptiveForest
{
  public:
    static constexpr int kNumDiamonds = 10;

    // Start from a uniform mesh: every base block present at subdivision 0.
    // max_subdivision: deepest split permitted. lateral/radial base subdomain counts per diamond.
    AdaptiveForest( int max_subdivision, int lateral_subdomains_per_diamond, int radial_subdomains )
    : M_( max_subdivision )
    , S_lat_( lateral_subdomains_per_diamond )
    , S_rad_( radial_subdomains )
    {
        for ( int d = 0; d < kNumDiamonds; ++d )
            for ( int bx = 0; bx < S_lat_; ++bx )
                for ( int by = 0; by < S_lat_; ++by )
                    for ( int br = 0; br < S_rad_; ++br )
                        leaves_.push_back( ForestLeaf{ SubdomainInfo{ d, bx, by, br }, 0 } );
        normalize();
    }

    // --- geometry / arithmetic ---------------------------------------------------------------------

    [[nodiscard]] int max_subdivision() const { return M_; }
    [[nodiscard]] int lateral_subdomains_per_diamond() const { return S_lat_; } // base count per diamond side
    [[nodiscard]] int radial_subdomains() const { return S_rad_; }             // base radial count
    [[nodiscard]] int lateral_extent() const { return S_lat_ << M_; } // finest-frame units, per diamond
    [[nodiscard]] int radial_extent() const { return S_rad_ << M_; }  // finest-frame units

    // Side length of a subdivision-k leaf in finest-frame units.
    [[nodiscard]] int finest_span( int subdivision ) const { return 1 << ( M_ - subdivision ); }

    // A leaf's min-corner in the finest frame, as a SubdomainInfo. Unique per leaf across the whole
    // mesh (a partition has no two cells sharing a min corner), so it is a safe global key even though
    // the leaf's own-level id is not (a coarse and a fine leaf can share own-level indices).
    [[nodiscard]] SubdomainInfo finest_anchor( const ForestLeaf& l ) const
    {
        const int sh = M_ - l.subdivision;
        return SubdomainInfo{ l.id.diamond_id(), l.id.subdomain_x() << sh, l.id.subdomain_y() << sh,
                              l.id.subdomain_r() << sh };
    }

    [[nodiscard]] bool in_range( const ForestLeaf& l ) const
    {
        const int nx = S_lat_ << l.subdivision;
        const int nr = S_rad_ << l.subdivision;
        return l.id.diamond_id() >= 0 && l.id.diamond_id() < kNumDiamonds && l.id.subdomain_x() >= 0 &&
               l.id.subdomain_x() < nx && l.id.subdomain_y() >= 0 && l.id.subdomain_y() < nx &&
               l.id.subdomain_r() >= 0 && l.id.subdomain_r() < nr && l.subdivision >= 0 &&
               l.subdivision <= M_;
    }

    // Parent: subdivision k -> k-1, indices halved.
    [[nodiscard]] ForestLeaf parent( const ForestLeaf& l ) const
    {
        return ForestLeaf{ SubdomainInfo{ l.id.diamond_id(), l.id.subdomain_x() / 2, l.id.subdomain_y() / 2,
                                          l.id.subdomain_r() / 2 },
                           l.subdivision - 1 };
    }

    // The 8 children (2x2x2) at subdivision+1. octant bit 0 = x, bit 1 = y, bit 2 = r.
    [[nodiscard]] std::array< ForestLeaf, 8 > children( const ForestLeaf& l ) const
    {
        const int x = l.id.subdomain_x() * 2, y = l.id.subdomain_y() * 2, r = l.id.subdomain_r() * 2;
        std::array< ForestLeaf, 8 > out{};
        for ( int oct = 0; oct < 8; ++oct )
            out[oct] = ForestLeaf{ SubdomainInfo{ l.id.diamond_id(), x + ( ( oct >> 0 ) & 1 ),
                                                  y + ( ( oct >> 1 ) & 1 ), r + ( ( oct >> 2 ) & 1 ) },
                                   l.subdivision + 1 };
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
    // and subdivision is capped at M, so this terminates. Idempotent on an already-balanced mesh.
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
                if ( L.subdivision >= M_ )
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

    // Return the index (into leaves()) of the leaf covering a given finest-frame point, if any.
    // O(M): tries candidate subdivisions finest->coarsest, mapping the point down to each.
    [[nodiscard]] std::optional< std::size_t > leaf_at( int diamond, int fx, int fy, int fr ) const
    {
        for ( int cd = M_; cd >= 0; --cd )
        {
            const int shift = M_ - cd;
            const auto it = lookup_.find( key( diamond, fx >> shift, fy >> shift, fr >> shift, cd ) );
            if ( it != lookup_.end() )
                return it->second;
        }
        return std::nullopt;
    }

    // All neighbors across one face of `L`. Interior faces return 1 neighbor (same/coarser) or up to 4
    // (finer). Radial-boundary faces return DomainBoundary. A lateral diamond seam is resolved inline
    // (probe points transformed into the neighbor diamond's frame) and reported as Interior neighbors.
    // Detection: sample the 2x2 finer sub-face centers just outside the face (in the finest frame) and
    // look up which leaf covers each -- one distinct hit => same/coarser, four => finer.
    [[nodiscard]] FaceNeighbors face_neighbors( const ForestLeaf& L, Face f ) const
    {
        FaceNeighbors res;
        const int     s = finest_span( L.subdivision );
        // leaf's min corner in the finest frame:
        const int base[3] = { L.id.subdomain_x() << ( M_ - L.subdivision ),
                              L.id.subdomain_y() << ( M_ - L.subdivision ),
                              L.id.subdomain_r() << ( M_ - L.subdivision ) };

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

        // radial out-of-range: CMB/surface, no neighbor. Lateral out-of-range: a diamond seam --
        // resolved below by transforming the probe points into the neighbor diamond's frame.
        const bool crosses_seam = ( axis != 2 ) && ( out_coord < 0 || out_coord >= extent );
        if ( axis == 2 && ( out_coord < 0 || out_coord >= extent ) )
        {
            res.kind = NeighborKind::DomainBoundary;
            return res;
        }
        const DiamondSeam seam =
            crosses_seam ? diamond_seam( L.id.diamond_id(), f ) : DiamondSeam{ L.id.diamond_id(), opposite( f ), false };

        const int a1    = ( axis == 0 ) ? 1 : 0; // first in-face axis (the other lateral)
        const int a2    = ( axis == 2 ) ? 1 : 2; // second in-face axis
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

                std::optional< std::size_t > hit;
                if ( !crosses_seam )
                {
                    hit = leaf_at( L.id.diamond_id(), pt[0], pt[1], pt[2] );
                }
                else
                {
                    // my in-face lateral coordinate, possibly reversed on the other side
                    int u = pt[a1];
                    if ( seam.reversed )
                        u = lateral_extent() - 1 - u;
                    // the neighbor's fixed lateral axis and its boundary layer
                    const int nb_axis  = ( seam.nb_face == Face::XLOW || seam.nb_face == Face::XHIGH ) ? 0 : 1;
                    const int nb_fixed = ( seam.nb_face == Face::XLOW || seam.nb_face == Face::YLOW )
                                             ? 0
                                             : lateral_extent() - 1;
                    int tpt[3];
                    tpt[nb_axis]     = nb_fixed;
                    tpt[1 - nb_axis] = u;      // the neighbor's in-face lateral axis
                    tpt[2]           = pt[2];  // radial coordinates align across diamonds
                    hit              = leaf_at( seam.nb_diamond, tpt[0], tpt[1], tpt[2] );
                }
                if ( hit )
                    found.emplace( *hit, qi + 2 * qj );
            }

        const bool single = ( found.size() == 1 );
        for ( const auto& [idx, quad] : found )
        {
            const ForestLeaf& N = leaves_[idx];
            res.neighbors.push_back( FaceNeighbor{ N, N.subdivision - L.subdivision, single ? 0 : quad,
                                                   seam.nb_face, crosses_seam && seam.reversed } );
        }
        return res;
    }

    // True if `L` sits at a diamond corner (both lateral block indices at an extreme). Milestone A
    // forbids refining these so the hardcoded pole stitching stays valid at base level.
    [[nodiscard]] bool touches_diamond_corner( const ForestLeaf& L ) const
    {
        const int nblk = S_lat_ << L.subdivision;
        const bool xext = ( L.id.subdomain_x() == 0 || L.id.subdomain_x() == nblk - 1 );
        const bool yext = ( L.id.subdomain_y() == 0 || L.id.subdomain_y() == nblk - 1 );
        return xext && yext;
    }

    // Consistency check for tests: in-range, no duplicates, and a volume partition check.
    [[nodiscard]] bool validate() const
    {
        std::map< std::uint64_t, int > seen;
        std::uint64_t                  vol = 0;
        for ( const auto& l : leaves_ )
        {
            if ( !in_range( l ) )
                return false;
            if ( ++seen[key( l )] > 1 )
                return false;
            const std::uint64_t s = static_cast< std::uint64_t >( finest_span( l.subdivision ) );
            vol += s * s * s;
        }
        const std::uint64_t total = static_cast< std::uint64_t >( kNumDiamonds ) * lateral_extent() *
                                    lateral_extent() * radial_extent();
        return vol == total;
    }

  private:
    int M_, S_lat_, S_rad_;

    std::vector< ForestLeaf >                        leaves_; // kept sorted + unique
    std::map< std::uint64_t, std::size_t >           lookup_; // key -> index into leaves_

    // Pack (subdivision, diamond, x, y, r) into 64 bits. x,y,r < 2^18, diamond < 2^4, subdivision < 2^6.
    static std::uint64_t key( int diamond, int x, int y, int r, int subdivision )
    {
        return ( static_cast< std::uint64_t >( subdivision ) << 58 ) |
               ( static_cast< std::uint64_t >( diamond ) << 54 ) | ( static_cast< std::uint64_t >( x ) << 36 ) |
               ( static_cast< std::uint64_t >( y ) << 18 ) | ( static_cast< std::uint64_t >( r ) );
    }
    static std::uint64_t key( const ForestLeaf& l )
    {
        return key( l.id.diamond_id(), l.id.subdomain_x(), l.id.subdomain_y(), l.id.subdomain_r(),
                    l.subdivision );
    }

    void refine_one( const ForestLeaf& l )
    {
        auto it = lookup_.find( key( l ) );
        if ( it == lookup_.end() )
            return; // not present; ignore
        leaves_[it->second].subdivision = -1; // mark removed; children appended, normalize() rebuilds
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
            leaves_[lookup_.at( key( c ) )].subdivision = -1; // mark removed
        leaves_.push_back( par );
    }

    // Drop removed leaves (subdivision < 0), sort, dedupe, rebuild lookup.
    void normalize()
    {
        leaves_.erase( std::remove_if( leaves_.begin(), leaves_.end(),
                                       []( const ForestLeaf& l ) { return l.subdivision < 0; } ),
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
