// Unit test for the adaptive (2:1) face-neighbor tables (adaptive_neighborhood.hpp).
//
// Pure host (no Kokkos/MPI):
//   g++ -std=c++17 -I src tests/test_adaptive_neighborhood.cpp -o t && ./t

#include "terra/grid/shell/adaptive_neighborhood.hpp"

#include <cstdio>
#include <cstdlib>

using namespace terra::grid::shell;
using namespace terra::grid::shell::amr;

static int g_failures = 0;
static int g_checks   = 0;
#define CHECK( cond )                                                       \
    do                                                                      \
    {                                                                       \
        ++g_checks;                                                         \
        if ( !( cond ) )                                                    \
        {                                                                   \
            std::printf( "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond ); \
            ++g_failures;                                                   \
        }                                                                   \
    } while ( 0 )

static ForestLeaf leaf( int d, int x, int y, int r, int s ) { return ForestLeaf{ SubdomainInfo{ d, x, y, r }, s }; }
static auto       rank0 = []( const SubdomainInfo& ) { return 0; };

// count neighbors of a given relative level
static int count_rel( const AdaptiveNeighborhood& n, int rel )
{
    int c = 0;
    for ( const auto& f : n.faces )
        if ( f.rel_level == rel )
            ++c;
    return c;
}

int main()
{
    const int M = 3;

    // --- corner base leaf (S_lat=2): 2 in-diamond + 2 seam face-neighbors, all same level ----------
    {
        AdaptiveForest f( M, 2, 1 );
        auto           n = face_neighborhood_of( f, leaf( 0, 0, 0, 0, 0 ), rank0 );
        CHECK( n.faces.size() == 4 );
        CHECK( count_rel( n, 0 ) == 4 );
        // in-diamond: finest anchors of base blocks (1,0,0) and (0,1,0): x/y = 1<<M = 8.
        // seams: d0 XLOW -> d1 block (0,0,0); d0 YLOW -> d4 block (0,0,0) (both forward).
        bool saw_x = false, saw_y = false, saw_s1 = false, saw_s4 = false;
        for ( const auto& fn : n.faces )
        {
            CHECK( fn.rank == 0 );
            if ( fn.anchor == SubdomainInfo( 0, 8, 0, 0 ) ) { saw_x = true; CHECK( fn.my_face == Face::XHIGH ); }
            if ( fn.anchor == SubdomainInfo( 0, 0, 8, 0 ) ) { saw_y = true; CHECK( fn.my_face == Face::YHIGH ); }
            if ( fn.anchor == SubdomainInfo( 1, 0, 0, 0 ) )
            {
                saw_s1 = true;
                CHECK( fn.my_face == Face::XLOW );
                CHECK( fn.neighbor_face == Face::YLOW );
                CHECK( !fn.seam_reversed );
            }
            if ( fn.anchor == SubdomainInfo( 4, 0, 0, 0 ) )
            {
                saw_s4 = true;
                CHECK( fn.my_face == Face::YLOW );
                CHECK( fn.neighbor_face == Face::XLOW );
                CHECK( !fn.seam_reversed );
            }
        }
        CHECK( saw_x && saw_y && saw_s1 && saw_s4 );
    }

    // --- interior base leaf (S_lat=4): 4 same-level face-neighbors -------------------------------
    {
        AdaptiveForest f( M, 4, 1 );
        auto           n = face_neighborhood_of( f, leaf( 0, 1, 1, 0, 0 ), rank0 );
        CHECK( n.faces.size() == 4 );      // x-low/high, y-low/high all interior
        CHECK( count_rel( n, 0 ) == 4 );   // radial faces are domain boundary (S_rad=1)
    }

    // --- refined neighbor: coarse leaf sees 4 finer across x-high, same-level elsewhere -----------
    {
        AdaptiveForest f( M, 2, 1 );
        f.refine( { leaf( 0, 1, 0, 0, 0 ) } );
        auto n = face_neighborhood_of( f, leaf( 0, 0, 0, 0, 0 ), rank0 );
        CHECK( n.faces.size() == 7 );      // 4 finer (x-high) + 1 same (y-high) + 2 same (seams)
        CHECK( count_rel( n, +1 ) == 4 );
        CHECK( count_rel( n, 0 ) == 3 );
        int octmask = 0;
        for ( const auto& fn : n.faces )
            if ( fn.rel_level == +1 )
            {
                CHECK( fn.my_face == Face::XHIGH );
                octmask |= ( 1 << fn.sub_octant );
            }
        CHECK( octmask == 0b1111 );

        // a fine child sees the coarse leaf across x-low (rel_level -1)
        auto nc = face_neighborhood_of( f, leaf( 0, 2, 0, 0, 1 ), rank0 );
        CHECK( count_rel( nc, -1 ) >= 1 );
        bool saw_indiamond_coarse = false;
        for ( const auto& fn : nc.faces )
            if ( fn.rel_level == -1 && fn.anchor == SubdomainInfo( 0, 0, 0, 0 ) )
                saw_indiamond_coarse = true; // (a seam face may add another coarse neighbor)
        CHECK( saw_indiamond_coarse );
    }

    // --- rank is taken from the NEIGHBOR's anchor (not mine) -------------------------------------
    {
        AdaptiveForest f( M, 4, 1 );
        auto           rank_by_x = []( const SubdomainInfo& a ) { return a.subdomain_x(); };
        auto           n         = face_neighborhood_of( f, leaf( 0, 1, 1, 0, 0 ), rank_by_x );
        for ( const auto& fn : n.faces )
            CHECK( fn.rank == fn.anchor.subdomain_x() );
    }

    if ( g_failures == 0 )
        std::printf( "test_adaptive_neighborhood: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_neighborhood: %d/%d FAILURE(S)\n", g_failures, g_checks );
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
