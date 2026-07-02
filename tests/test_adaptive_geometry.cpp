// Unit test for the depth-aware geometry helpers (adaptive_geometry.hpp).
//
// Pure host math (no Kokkos/MPI):
//   g++ -std=c++17 -I src tests/test_adaptive_geometry.cpp -o t && ./t
// Verifies that subdivision-0 leaves reproduce the uniform mesh exactly, and that subdivided leaves
// place lateral blocks and radial shells at the correct finer positions.

#include "terra/grid/shell/adaptive_geometry.hpp"

#include <cmath>
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
static bool close( double a, double b ) { return std::fabs( a - b ) < 1e-12; }

int main()
{
    // ===================== lateral =====================
    // LDR=3 (8 cells/diamond side), S_lat=2 base, M=2.
    {
        const int LDR = 3, S_lat = 2, M = 2;

        // subdivision 0, base block own (1,0): finest anchor x = 1<<M = 4.
        auto p0 = adaptive_lateral_coord_params( LDR, S_lat, M, 0, SubdomainInfo{ 0, 4, 0, 0 } );
        CHECK( p0.global_refinements == 3 );      // == LDR (uniform)
        CHECK( p0.num_subdomains_per_side == 2 ); // == S_lat (uniform)
        CHECK( p0.subdomain_i == 1 && p0.subdomain_j == 0 );

        // subdivision 1, own (1,0): finest anchor x = 1<<(M-1) = 2.
        auto p1 = adaptive_lateral_coord_params( LDR, S_lat, M, 1, SubdomainInfo{ 0, 2, 0, 0 } );
        CHECK( p1.global_refinements == 4 );      // LDR + 1
        CHECK( p1.num_subdomains_per_side == 4 ); // S_lat << 1
        CHECK( p1.subdomain_i == 1 && p1.subdomain_j == 0 );

        // block size (cells/side) is invariant across subdivision -> fixed node count
        CHECK( ( 1 << p0.global_refinements ) / p0.num_subdomains_per_side == 4 );
        CHECK( ( 1 << p1.global_refinements ) / p1.num_subdomains_per_side == 4 );
    }

    // ===================== radial =====================
    // base radii: 4 layers, S_rad=1 base, M=2 -> layers_per_subdomain = 4.
    {
        const std::vector< double > R = { 1.0, 1.5, 2.0, 2.5, 3.0 };
        const int                   S_rad = 1, M = 2;

        // subdivision 0 (single radial subdomain, own_r=0): nodes land exactly on base radii.
        for ( int j = 0; j <= 4; ++j )
            CHECK( close( adaptive_shell_radius( R, S_rad, M, 0, SubdomainInfo{ 0, 0, 0, 0 }, j ), R[j] ) );

        // subdivision 1, inner leaf (own_r=0, finest anchor r=0): {1.0,1.25,1.5,1.75,2.0}
        const double inner[5] = { 1.0, 1.25, 1.5, 1.75, 2.0 };
        for ( int j = 0; j <= 4; ++j )
            CHECK( close( adaptive_shell_radius( R, S_rad, M, 1, SubdomainInfo{ 0, 0, 0, 0 }, j ), inner[j] ) );

        // subdivision 1, outer leaf (own_r=1, finest anchor r=1<<(M-1)=2): {2.0,2.25,2.5,2.75,3.0}
        const double outer[5] = { 2.0, 2.25, 2.5, 2.75, 3.0 };
        for ( int j = 0; j <= 4; ++j )
            CHECK( close( adaptive_shell_radius( R, S_rad, M, 1, SubdomainInfo{ 0, 0, 0, 2 }, j ), outer[j] ) );

        // the two subdivision-1 leaves share the interface radius (2.0) and are each finer than base
        CHECK( close( inner[4], outer[0] ) );
        CHECK( close( inner[1] - inner[0], 0.25 ) ); // half the base spacing 0.5

        // endpoints hit the true domain boundaries
        CHECK( close( adaptive_shell_radius( R, S_rad, M, 2, SubdomainInfo{ 0, 0, 0, 0 }, 0 ), 1.0 ) );
        CHECK( close( adaptive_shell_radius( R, S_rad, M, 0, SubdomainInfo{ 0, 0, 0, 0 }, 4 ), 3.0 ) );

        // monotonically increasing within a leaf
        double prev = -1;
        for ( int j = 0; j <= 4; ++j )
        {
            double r = adaptive_shell_radius( R, S_rad, M, 1, SubdomainInfo{ 0, 0, 0, 0 }, j );
            CHECK( r > prev );
            prev = r;
        }
    }

    if ( g_failures == 0 )
        std::printf( "test_adaptive_geometry: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_geometry: %d/%d FAILURE(S)\n", g_failures, g_checks );
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
