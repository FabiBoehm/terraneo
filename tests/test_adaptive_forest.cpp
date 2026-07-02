// Unit test for terra::grid::shell::amr::AdaptiveForest (Part 1, Step 1: core data model).
//
// Dependency-free: compile standalone with
//   g++ -std=c++17 -I src tests/test_adaptive_forest.cpp -o test_adaptive_forest && ./test_adaptive_forest
// (also wired into CMake via add_terra_test, but does not require Kokkos/MPI to run.)

#include "terra/grid/shell/adaptive_forest.hpp"

#include <cstdio>
#include <cstdlib>

using namespace terra::grid::shell::amr;

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

int main()
{
    // D=3 -> base_span 8; S_lat=2 -> lateral extent 16; S_rad=1 -> radial extent 8.
    const int D = 3, S_lat = 2, S_rad = 1;

    // --- construction: uniform base mesh -----------------------------------------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        CHECK( f.size() == 10u * S_lat * S_lat * S_rad ); // 40 base leaves
        for ( const auto& l : f.leaves() )
            CHECK( l.depth == 0 );
        CHECK( f.validate() );
        CHECK( f.lateral_extent() == 16 );
        CHECK( f.radial_extent() == 8 );
        CHECK( f.span( 0 ) == 8 && f.span( 1 ) == 4 && f.span( 3 ) == 1 );
    }

    // --- arithmetic: parent(child) round-trip, children aligned/in-range ---------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        auto           kids = f.children( L );
        for ( int oct = 0; oct < 8; ++oct )
        {
            CHECK( kids[oct].depth == 1 );
            CHECK( f.aligned( kids[oct] ) );
            CHECK( f.in_range( kids[oct] ) );
            CHECK( f.parent( kids[oct] ) == L );
        }
        // children occupy the 8 distinct octant corners
        CHECK( kids[0].anchor == ( BrickId{ 0, 0, 0, 0 } ) );
        CHECK( kids[7].anchor == ( BrickId{ 0, 4, 4, 4 } ) );
    }

    // --- refine one leaf: size +7, parent gone, children present, validate ------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        CHECK( f.contains( L ) );
        f.refine( { L } );
        CHECK( f.size() == 47u ); // 40 - 1 + 8
        CHECK( !f.contains( L ) );
        for ( const auto& c : f.children( L ) )
            CHECK( f.contains( c ) );
        CHECK( f.validate() );
    }

    // --- leaf_at: point inside a refined child vs an unrefined base block --------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        f.refine( { L } );
        // point (5,1,1) sits in child anchored (4,0,0), depth 1
        auto hit = f.leaf_at( 0, 5, 1, 1 );
        CHECK( hit.has_value() );
        if ( hit )
        {
            const auto& l = f.leaves()[*hit];
            CHECK( l.depth == 1 );
            CHECK( l.anchor == ( BrickId{ 0, 4, 0, 0 } ) );
        }
        // point in an untouched base block (diamond 1) resolves to a depth-0 leaf
        auto hit2 = f.leaf_at( 1, 2, 2, 2 );
        CHECK( hit2.has_value() );
        if ( hit2 )
            CHECK( f.leaves()[*hit2].depth == 0 );
    }

    // --- coarsen inverts refine -------------------------------------------------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        f.refine( { L } );
        CHECK( f.size() == 47u );
        f.coarsen( { f.children( L )[0] } ); // any sibling as representative
        CHECK( f.size() == 40u );
        CHECK( f.contains( L ) );
        CHECK( f.validate() );
    }

    // --- nested refine: depth 2 leaf, validate + leaf_at ------------------------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        f.refine( { L } );
        ForestLeaf child{ BrickId{ 0, 4, 0, 0 }, 1 };
        f.refine( { child } );
        CHECK( f.size() == 54u ); // 47 - 1 + 8
        CHECK( f.validate() );
        auto hit = f.leaf_at( 0, 7, 1, 1 ); // in grandchild anchored (6,0,0), depth 2
        CHECK( hit.has_value() );
        if ( hit )
        {
            const auto& l = f.leaves()[*hit];
            CHECK( l.depth == 2 );
            CHECK( l.anchor == ( BrickId{ 0, 6, 0, 0 } ) );
        }
    }

    // === Step 2: face_neighbors + touches_diamond_corner =========================================
    // Use S_lat=2 so interior lateral faces exist (base anchors at 0 and 8, extent 16).

    // --- same-level neighbor -----------------------------------------------------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        auto           fn = f.face_neighbors( L, Face::XHIGH );
        CHECK( fn.kind == NeighborKind::Interior );
        CHECK( fn.neighbors.size() == 1 );
        if ( fn.neighbors.size() == 1 )
        {
            CHECK( fn.neighbors[0].rel_level == 0 );
            CHECK( fn.neighbors[0].leaf.anchor == ( BrickId{ 0, 8, 0, 0 } ) );
            CHECK( fn.neighbors[0].neighbor_face == Face::XLOW );
        }
    }

    // --- finer neighbor: refine the block across XHIGH, expect 4 finer neighbors -------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        ForestLeaf     R{ BrickId{ 0, 8, 0, 0 }, 0 };
        f.refine( { R } );
        auto fn = f.face_neighbors( L, Face::XHIGH );
        CHECK( fn.kind == NeighborKind::Interior );
        CHECK( fn.neighbors.size() == 4 );
        int octmask = 0;
        for ( const auto& nb : fn.neighbors )
        {
            CHECK( nb.rel_level == +1 );
            CHECK( nb.leaf.depth == 1 );
            octmask |= ( 1 << nb.sub_octant );
        }
        CHECK( octmask == 0b1111 ); // all four quadrants distinct

        // --- coarser neighbor: from a child's XLOW, the neighbor is L (one level coarser) ----------
        ForestLeaf child{ BrickId{ 0, 8, 0, 0 }, 1 };
        auto       fnc = f.face_neighbors( child, Face::XLOW );
        CHECK( fnc.kind == NeighborKind::Interior );
        CHECK( fnc.neighbors.size() == 1 );
        if ( fnc.neighbors.size() == 1 )
        {
            CHECK( fnc.neighbors[0].rel_level == -1 );
            CHECK( fnc.neighbors[0].leaf == L );
        }
    }

    // --- radial domain boundary (S_rad=1: both radial faces are CMB/surface) -----------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        CHECK( f.face_neighbors( L, Face::RLOW ).kind == NeighborKind::DomainBoundary );
        CHECK( f.face_neighbors( L, Face::RHIGH ).kind == NeighborKind::DomainBoundary );
    }

    // --- lateral diamond seam ----------------------------------------------------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };
        CHECK( f.face_neighbors( L, Face::XLOW ).kind == NeighborKind::DiamondCrossing );
        CHECK( f.face_neighbors( L, Face::YLOW ).kind == NeighborKind::DiamondCrossing );
    }

    // --- touches_diamond_corner (S_lat=4: interior base blocks are not corners) --------------------
    {
        AdaptiveForest f4( D, 4, S_rad ); // base anchors at 0,8,16,24; extent 32
        CHECK( f4.touches_diamond_corner( ForestLeaf{ BrickId{ 0, 0, 0, 0 }, 0 } ) );    // corner
        CHECK( f4.touches_diamond_corner( ForestLeaf{ BrickId{ 0, 24, 24, 0 }, 0 } ) );  // corner
        CHECK( !f4.touches_diamond_corner( ForestLeaf{ BrickId{ 0, 8, 8, 0 }, 0 } ) );   // interior
        CHECK( !f4.touches_diamond_corner( ForestLeaf{ BrickId{ 0, 8, 0, 0 }, 0 } ) );   // edge, not corner
    }

    // === Step 3: balance_2to1 ====================================================================

    // Global 2:1 check: every interior face-neighbor is within one level.
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
        AdaptiveForest f( D, S_lat, S_rad );
        CHECK( is_balanced( f ) );
        const std::size_t before = f.size();
        f.balance_2to1();
        CHECK( f.size() == before ); // no spurious splits
        CHECK( is_balanced( f ) );
    }

    // --- a 2-level jump forces the coarse neighbor to split ---------------------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        ForestLeaf     L{ BrickId{ 0, 0, 0, 0 }, 0 };     // stays coarse initially
        ForestLeaf     R{ BrickId{ 0, 8, 0, 0 }, 0 };
        f.refine( { R } );                                // depth 1 across L's XHIGH face
        f.refine( { ForestLeaf{ BrickId{ 0, 8, 0, 0 }, 1 } } ); // depth 2 in the adjacent quadrant

        // Now L (depth 0) has a depth-2 neighbor across XHIGH: a 2-level jump.
        CHECK( !is_balanced( f ) );
        auto pre = f.face_neighbors( L, Face::XHIGH );
        bool saw_rel2 = false;
        for ( const auto& nb : pre.neighbors )
            if ( nb.rel_level == 2 )
                saw_rel2 = true;
        CHECK( saw_rel2 );

        f.balance_2to1();

        CHECK( is_balanced( f ) );   // ripple resolved it
        CHECK( f.validate() );       // still a clean partition
        CHECK( !f.contains( L ) );   // L was split to depth 1
        for ( const auto& c : f.children( L ) )
            CHECK( f.contains( c ) );

        // idempotent: balancing again changes nothing
        const std::size_t after = f.size();
        f.balance_2to1();
        CHECK( f.size() == after );
    }

    // --- deeper cascade: a depth-3 pocket forces a multi-level ripple ------------------------------
    {
        AdaptiveForest f( D, S_lat, S_rad );
        // Drive one corner region down to the finest level, step by step.
        f.refine( { ForestLeaf{ BrickId{ 0, 8, 0, 0 }, 0 } } );
        f.refine( { ForestLeaf{ BrickId{ 0, 8, 0, 0 }, 1 } } );
        f.refine( { ForestLeaf{ BrickId{ 0, 8, 0, 0 }, 2 } } ); // depth 3 pocket
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
