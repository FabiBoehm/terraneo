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
#define CHECK( cond )                                                             \
    do                                                                            \
    {                                                                             \
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

    if ( g_failures == 0 )
        std::printf( "test_adaptive_forest: ALL PASS\n" );
    else
        std::printf( "test_adaptive_forest: %d FAILURE(S)\n", g_failures );
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
