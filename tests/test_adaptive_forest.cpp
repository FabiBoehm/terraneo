// Unit test for terra::grid::shell::amr::AdaptiveForest.
//
// A leaf is a terra SubdomainInfo (diamond,x,y,r) at a `subdivision`. A uniform forest at subdivision 0
// is exactly today's uniform mesh, using the identical subdomain indices.
//
//   g++ -std=c++17 -I src tests/test_adaptive_forest.cpp -o test_adaptive_forest && ./test_adaptive_forest
// (also wired into CMake via add_terra_test.)

#include "terra/grid/shell/adaptive_forest.hpp"

#include <cstdio>
#include <cstdlib>

using namespace terra::grid::shell;      // SubdomainInfo
using namespace terra::grid::shell::amr; // AdaptiveForest, ForestLeaf, Face, ...

static int g_failures = 0;
static int g_checks   = 0;
#define CHECK( cond )                                                             \
    do                                                                            \
    {                                                                             \
        ++g_checks;                                                               \
        if ( !( cond ) )                                                          \
        {                                                                         \
            std::printf( "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond );       \
            ++g_failures;                                                         \
        }                                                                         \
    } while ( 0 )

static ForestLeaf leaf( int d, int x, int y, int r, int subdiv )
{
    return ForestLeaf{ SubdomainInfo{ d, x, y, r }, subdiv };
}

int main()
{
    // M=3 -> finest span 8; S_lat=2 -> finest lateral extent 16; S_rad=1 -> radial extent 8.
    const int M = 3, S_lat = 2, S_rad = 1;

    // --- construction: uniform base mesh == today's uniform subdomain set --------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        CHECK( f.size() == 10u * S_lat * S_lat * S_rad ); // 40 base leaves
        for ( const auto& l : f.leaves() )
            CHECK( l.subdivision == 0 );
        CHECK( f.validate() );
        CHECK( f.lateral_extent() == 16 );
        CHECK( f.radial_extent() == 8 );
        CHECK( f.finest_span( 0 ) == 8 && f.finest_span( 1 ) == 4 && f.finest_span( 3 ) == 1 );
        CHECK( f.contains( leaf( 0, 0, 0, 0, 0 ) ) );   // terra-style index (0,0,0,0)
        CHECK( f.contains( leaf( 0, 1, 0, 0, 0 ) ) );   // and its neighbor (1,0,0)
    }

    // --- arithmetic: parent(child) round-trip ------------------------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        ForestLeaf     L = leaf( 0, 0, 0, 0, 0 );
        auto           kids = f.children( L );
        for ( int oct = 0; oct < 8; ++oct )
        {
            CHECK( kids[oct].subdivision == 1 );
            CHECK( f.in_range( kids[oct] ) );
            CHECK( f.parent( kids[oct] ) == L );
        }
        CHECK( kids[0].id == SubdomainInfo( 0, 0, 0, 0 ) ); // octant 0
        CHECK( kids[7].id == SubdomainInfo( 0, 1, 1, 1 ) ); // octant 7 -> (2*0+1) in each axis
    }

    // --- refine one leaf: size +7, parent gone, children present, validate ------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        ForestLeaf     L = leaf( 0, 0, 0, 0, 0 );
        CHECK( f.contains( L ) );
        f.refine( { L } );
        CHECK( f.size() == 47u ); // 40 - 1 + 8
        CHECK( !f.contains( L ) );
        for ( const auto& c : f.children( L ) )
            CHECK( f.contains( c ) );
        CHECK( f.validate() );
    }

    // --- leaf_at: finest-frame point in a refined child vs an unrefined base block -----------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        f.refine( { leaf( 0, 0, 0, 0, 0 ) } );
        // finest point (5,1,1) -> subdivision-1 own index (1,0,0)
        auto hit = f.leaf_at( 0, 5, 1, 1 );
        CHECK( hit.has_value() );
        if ( hit )
        {
            const auto& l = f.leaves()[*hit];
            CHECK( l.subdivision == 1 );
            CHECK( l.id == SubdomainInfo( 0, 1, 0, 0 ) );
        }
        // untouched base block in diamond 1
        auto hit2 = f.leaf_at( 1, 2, 2, 2 );
        CHECK( hit2.has_value() );
        if ( hit2 )
            CHECK( f.leaves()[*hit2].subdivision == 0 );
    }

    // --- coarsen inverts refine -------------------------------------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        ForestLeaf     L = leaf( 0, 0, 0, 0, 0 );
        f.refine( { L } );
        CHECK( f.size() == 47u );
        f.coarsen( { f.children( L )[0] } );
        CHECK( f.size() == 40u );
        CHECK( f.contains( L ) );
        CHECK( f.validate() );
    }

    // --- nested refine: subdivision-2 leaf ---------------------------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        f.refine( { leaf( 0, 0, 0, 0, 0 ) } );
        f.refine( { leaf( 0, 1, 0, 0, 1 ) } ); // a child of L
        CHECK( f.size() == 54u );              // 47 - 1 + 8
        CHECK( f.validate() );
        auto hit = f.leaf_at( 0, 7, 1, 1 ); // finest -> subdivision-2 own index (3,0,0)
        CHECK( hit.has_value() );
        if ( hit )
        {
            const auto& l = f.leaves()[*hit];
            CHECK( l.subdivision == 2 );
            CHECK( l.id == SubdomainInfo( 0, 3, 0, 0 ) );
        }
    }

    // === face_neighbors + touches_diamond_corner =================================================

    // --- same-level neighbor -----------------------------------------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        auto fn = f.face_neighbors( leaf( 0, 0, 0, 0, 0 ), Face::XHIGH );
        CHECK( fn.kind == NeighborKind::Interior );
        CHECK( fn.neighbors.size() == 1 );
        if ( fn.neighbors.size() == 1 )
        {
            CHECK( fn.neighbors[0].rel_level == 0 );
            CHECK( fn.neighbors[0].leaf.id == SubdomainInfo( 0, 1, 0, 0 ) );
            CHECK( fn.neighbors[0].neighbor_face == Face::XLOW );
        }
    }

    // --- finer neighbor (4) then coarser neighbor from a child's side ------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        ForestLeaf     L = leaf( 0, 0, 0, 0, 0 );
        f.refine( { leaf( 0, 1, 0, 0, 0 ) } ); // refine the block across XHIGH
        auto fn = f.face_neighbors( L, Face::XHIGH );
        CHECK( fn.kind == NeighborKind::Interior );
        CHECK( fn.neighbors.size() == 4 );
        int octmask = 0;
        for ( const auto& nb : fn.neighbors )
        {
            CHECK( nb.rel_level == +1 );
            CHECK( nb.leaf.subdivision == 1 );
            octmask |= ( 1 << nb.sub_octant );
        }
        CHECK( octmask == 0b1111 );

        ForestLeaf child = leaf( 0, 2, 0, 0, 1 ); // finest x in [8,12), adjacent to L
        auto       fnc   = f.face_neighbors( child, Face::XLOW );
        CHECK( fnc.kind == NeighborKind::Interior );
        CHECK( fnc.neighbors.size() == 1 );
        if ( fnc.neighbors.size() == 1 )
        {
            CHECK( fnc.neighbors[0].rel_level == -1 );
            CHECK( fnc.neighbors[0].leaf == L );
        }
    }

    // --- radial domain boundary + lateral diamond seam --------------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        ForestLeaf     L = leaf( 0, 0, 0, 0, 0 );
        CHECK( f.face_neighbors( L, Face::RLOW ).kind == NeighborKind::DomainBoundary );
        CHECK( f.face_neighbors( L, Face::RHIGH ).kind == NeighborKind::DomainBoundary );
        CHECK( f.face_neighbors( L, Face::XLOW ).kind == NeighborKind::DiamondCrossing );
        CHECK( f.face_neighbors( L, Face::YLOW ).kind == NeighborKind::DiamondCrossing );
    }

    // --- touches_diamond_corner (S_lat=4: interior base blocks are not corners) --------------------
    {
        AdaptiveForest f4( M, 4, S_rad );
        CHECK( f4.touches_diamond_corner( leaf( 0, 0, 0, 0, 0 ) ) );   // corner
        CHECK( f4.touches_diamond_corner( leaf( 0, 3, 3, 0, 0 ) ) );   // corner (index 3 == max)
        CHECK( !f4.touches_diamond_corner( leaf( 0, 1, 1, 0, 0 ) ) );  // interior
        CHECK( !f4.touches_diamond_corner( leaf( 0, 1, 0, 0, 0 ) ) );  // edge, not corner
    }

    // === balance_2to1 ============================================================================

    auto is_balanced = []( const AdaptiveForest& f ) {
        const Face faces[6] = { Face::XLOW, Face::XHIGH, Face::YLOW, Face::YHIGH, Face::RLOW, Face::RHIGH };
        for ( const auto& L : f.leaves() )
            for ( Face fc : faces )
            {
                auto fn = f.face_neighbors( L, fc );
                if ( fn.kind != NeighborKind::Interior )
                    continue;
                for ( const auto& nb : fn.neighbors )
                    if ( nb.rel_level > 1 || nb.rel_level < -1 )
                        return false;
            }
        return true;
    };

    // --- already-balanced mesh is untouched --------------------------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        CHECK( is_balanced( f ) );
        const std::size_t before = f.size();
        f.balance_2to1();
        CHECK( f.size() == before );
        CHECK( is_balanced( f ) );
    }

    // --- a 2-level jump forces the coarse neighbor to split ---------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        ForestLeaf     L = leaf( 0, 0, 0, 0, 0 );
        f.refine( { leaf( 0, 1, 0, 0, 0 ) } ); // subdivision 1 across L's XHIGH face
        f.refine( { leaf( 0, 2, 0, 0, 1 ) } ); // subdivision 2 in the adjacent quadrant

        CHECK( !is_balanced( f ) );
        bool saw_rel2 = false;
        for ( const auto& nb : f.face_neighbors( L, Face::XHIGH ).neighbors )
            if ( nb.rel_level == 2 )
                saw_rel2 = true;
        CHECK( saw_rel2 );

        f.balance_2to1();
        CHECK( is_balanced( f ) );
        CHECK( f.validate() );
        CHECK( !f.contains( L ) );
        for ( const auto& c : f.children( L ) )
            CHECK( f.contains( c ) );

        const std::size_t after = f.size();
        f.balance_2to1();
        CHECK( f.size() == after ); // idempotent
    }

    // --- deeper cascade: a subdivision-3 pocket forces a multi-level ripple ------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        f.refine( { leaf( 0, 1, 0, 0, 0 ) } );
        f.refine( { leaf( 0, 2, 0, 0, 1 ) } );
        f.refine( { leaf( 0, 4, 0, 0, 2 ) } ); // subdivision-3 pocket
        CHECK( !is_balanced( f ) );
        f.balance_2to1();
        CHECK( is_balanced( f ) );
        CHECK( f.validate() );
    }

    if ( g_failures == 0 )
        std::printf( "test_adaptive_forest: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_forest: %d/%d FAILURE(S)\n", g_failures, g_checks );
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
