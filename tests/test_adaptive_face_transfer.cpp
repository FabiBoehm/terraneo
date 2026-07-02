// Unit test for the 2:1 face-node correspondence (adaptive_face_transfer.hpp).
//
// Pure host:  g++ -std=c++17 -I src tests/test_adaptive_face_transfer.cpp -o t && ./t

#include "terra/grid/shell/adaptive_face_transfer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

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

int main()
{
    const int n1 = 5, n2 = 5; // 4 cells/side; H1 = H2 = 2

    for ( int oct = 0; oct < 4; ++oct )
    {
        auto fc = face_correspondence( n1, n2, oct );
        CHECK( fc.size() == 25u );

        int coincident = 0;
        for ( const auto& nd : fc )
        {
            // weights sum to 1
            double s = 0;
            for ( int p = 0; p < nd.n_parents; ++p )
                s += nd.w[p];
            CHECK( std::fabs( s - 1.0 ) < 1e-12 );

            // parents in range and n_parents consistent with coincidence
            for ( int p = 0; p < nd.n_parents; ++p )
            {
                CHECK( nd.pa1[p] >= 0 && nd.pa1[p] < n1 );
                CHECK( nd.pa2[p] >= 0 && nd.pa2[p] < n2 );
            }
            CHECK( nd.coincident == ( nd.n_parents == 1 ) );
            if ( nd.coincident )
                ++coincident;
        }
        CHECK( coincident == 9 ); // 3x3 even-even fine nodes coincide with coarse nodes
    }

    // --- specific stencils in quadrant 0 (q1=q2=0) ------------------------------------------------
    {
        auto fc = face_correspondence( n1, n2, 0 );
        auto at = [&]( int jf, int kf ) -> const FineFaceNode& { return fc[jf * n2 + kf]; };

        CHECK( at( 0, 0 ).coincident && at( 0, 0 ).pa1[0] == 0 && at( 0, 0 ).pa2[0] == 0 );
        CHECK( at( 2, 2 ).coincident && at( 2, 2 ).pa1[0] == 1 && at( 2, 2 ).pa2[0] == 1 );
        CHECK( at( 4, 4 ).coincident && at( 4, 4 ).pa1[0] == 2 && at( 4, 4 ).pa2[0] == 2 );

        // edge-midpoint (1,0): parents (0,0) & (1,0)
        CHECK( at( 1, 0 ).n_parents == 2 && at( 1, 0 ).pa1[0] == 0 && at( 1, 0 ).pa1[1] == 1 );
        // face-centre (1,1): 4 parents around coarse cell (0,0)
        CHECK( at( 1, 1 ).n_parents == 4 );
    }

    // --- quadrant 3 (q1=q2=1) is offset to the far corner ----------------------------------------
    {
        auto fc = face_correspondence( n1, n2, 3 );
        auto at = [&]( int jf, int kf ) -> const FineFaceNode& { return fc[jf * n2 + kf]; };
        CHECK( at( 0, 0 ).coincident && at( 0, 0 ).pa1[0] == 2 && at( 0, 0 ).pa2[0] == 2 );
        CHECK( at( 4, 4 ).coincident && at( 4, 4 ).pa1[0] == 4 && at( 4, 4 ).pa2[0] == 4 );
    }

    // --- linear reproduction: interpolating a linear coarse field is exact ------------------------
    // Strong correctness check: a fine node at coarse-coordinate (q1*H1 + jf/2, q2*H2 + kf/2) must
    // reproduce f = a*jc + b*kc + c under the stencil, for every quadrant and every fine node.
    {
        const int H1 = ( n1 - 1 ) / 2, H2 = ( n2 - 1 ) / 2;
        const double trials[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 2.0, -3.0, 0.5 } };
        for ( int oct = 0; oct < 4; ++oct )
        {
            const int q1 = oct & 1, q2 = ( oct >> 1 ) & 1;
            auto      fc = face_correspondence( n1, n2, oct );
            for ( const auto& t : trials )
            {
                const double a = t[0], b = t[1], c = t[2];
                for ( const auto& nd : fc )
                {
                    double interp = 0;
                    for ( int p = 0; p < nd.n_parents; ++p )
                        interp += nd.w[p] * ( a * nd.pa1[p] + b * nd.pa2[p] + c );
                    const double expected =
                        a * ( q1 * H1 + nd.a1 / 2.0 ) + b * ( q2 * H2 + nd.a2 / 2.0 ) + c;
                    CHECK( std::fabs( interp - expected ) < 1e-12 );
                }
            }
        }
    }

    if ( g_failures == 0 )
        std::printf( "test_adaptive_face_transfer: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_face_transfer: %d/%d FAILURE(S)\n", g_failures, g_checks );
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
