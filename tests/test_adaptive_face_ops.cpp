// Unit test for the 2:1 face transfer kernels (adaptive_face_ops.hpp).
//
// Pure host:  g++ -std=c++17 -I src tests/test_adaptive_face_ops.cpp -o t && ./t

#include "terra/grid/shell/adaptive_face_ops.hpp"

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
static bool close( double a, double b ) { return std::fabs( a - b ) < 1e-12; }

int main()
{
    const int n1 = 5, n2 = 5, N = n1 * n2;

    // --- prolongate reproduces a linear coarse field exactly (P consistency) ----------------------
    {
        const int H1 = ( n1 - 1 ) / 2, H2 = ( n2 - 1 ) / 2;
        for ( int oct = 0; oct < 4; ++oct )
        {
            const int             q1 = oct & 1, q2 = ( oct >> 1 ) & 1;
            std::vector< double > coarse( N );
            for ( int j = 0; j < n1; ++j )
                for ( int k = 0; k < n2; ++k )
                    coarse[j * n2 + k] = 2.0 * j - 3.0 * k + 0.5; // linear
            auto fine = prolongate_face( coarse, n1, n2, oct );
            for ( int jf = 0; jf < n1; ++jf )
                for ( int kf = 0; kf < n2; ++kf )
                {
                    const double expected = 2.0 * ( q1 * H1 + jf / 2.0 ) - 3.0 * ( q2 * H2 + kf / 2.0 ) + 0.5;
                    CHECK( close( fine[jf * n2 + kf], expected ) );
                }
        }
    }

    // --- P and P^T are exact transposes: <P c, f> == <c, P^T f> for every quadrant -----------------
    {
        for ( int oct = 0; oct < 4; ++oct )
        {
            std::vector< double > c( N ), f( N );
            for ( int i = 0; i < N; ++i )
            {
                c[i] = 1.0 + ( ( i * 7 ) % 11 );  // deterministic pseudo-values
                f[i] = 2.0 - ( ( i * 5 ) % 13 ) * 0.3;
            }
            auto Pc = prolongate_face( c, n1, n2, oct );

            std::vector< double > PTf( N, 0.0 );
            restrict_face_add( f, PTf, n1, n2, oct );

            double lhs = 0, rhs = 0;
            for ( int i = 0; i < N; ++i )
            {
                lhs += Pc[i] * f[i];   // <P c, f>_fine
                rhs += c[i] * PTf[i];  // <c, P^T f>_coarse
            }
            CHECK( close( lhs, rhs ) );
        }
    }

    // --- restrict impulse response == stencil weights (transpose of a single fine node) -----------
    {
        std::vector< double > fine( N, 0.0 ), coarse( N, 0.0 );
        fine[1 * n2 + 0] = 1.0; // edge-midpoint node (1,0) in quadrant 0
        restrict_face_add( fine, coarse, n1, n2, 0 );
        CHECK( close( coarse[0 * n2 + 0], 0.5 ) ); // parents (0,0) & (1,0), weight 1/2 each
        CHECK( close( coarse[1 * n2 + 0], 0.5 ) );
        double total = 0;
        for ( double v : coarse )
            total += v;
        CHECK( close( total, 1.0 ) ); // mass conserved by P^T of a unit impulse
    }

    // --- assemble_face_2to1: additive assembly + broadcast to coincident fine nodes ---------------
    {
        // coarse partial = 10 everywhere; one fine octant with a unit at coincident node (2,2).
        std::vector< double >                  coarse( N, 10.0 );
        std::map< int, std::vector< double > > fine;
        fine[0] = std::vector< double >( N, 0.0 );
        fine[0][2 * n2 + 2] = 1.0; // coincident with coarse node (1,1)

        assemble_face_2to1( coarse, fine, n1, n2 );

        // coarse node (1,1) got 10 + 1 = 11; the fine coincident node is broadcast the assembled value
        CHECK( close( coarse[1 * n2 + 1], 11.0 ) );
        CHECK( close( fine[0][2 * n2 + 2], 11.0 ) );
        // a coarse node with no fine contribution is unchanged; its coincident fine node mirrors it
        CHECK( close( coarse[0 * n2 + 0], 10.0 ) );
        CHECK( close( fine[0][0 * n2 + 0], 10.0 ) );
    }

    // --- constant field: P^T row-sum at an interior coarse node = fine node count factor ----------
    // Restricting a constant-1 fine slab lands the local "measure" on coarse nodes; total is conserved.
    {
        std::vector< double > fine( N, 1.0 ), coarse( N, 0.0 );
        restrict_face_add( fine, coarse, n1, n2, 0 );
        double total = 0;
        for ( double v : coarse )
            total += v;
        CHECK( close( total, static_cast< double >( N ) ) ); // sum of weights over all fine nodes = N
    }

    if ( g_failures == 0 )
        std::printf( "test_adaptive_face_ops: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_face_ops: %d/%d FAILURE(S)\n", g_failures, g_checks );
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
