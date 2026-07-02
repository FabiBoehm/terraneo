// Integration test for the domain-level 2:1 face exchange (adaptive_exchange.hpp).
//
// Builds a real adaptive DistributedDomain with an interior 2:1 interface, runs the additive exchange on
// a Grid4D-indexed std::vector field, and checks the assembled coarse face against an independent P^T
// prediction plus the broadcast to the fine coincident nodes. Uses MPI but no Kokkos -> runs on a login
// node under mpiexec -np 1.

#include "terra/grid/shell/adaptive_exchange.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
static bool close( double a, double b ) { return std::fabs( a - b ) < 1e-10; }

// Grid4D-indexed (subdomain, x, y, r) field, row-major (== Kokkos LayoutRight).
struct Field
{
    std::vector< double > v;
    int                   nx, ny, nr;
    Field( int nsub, int nx_, int ny_, int nr_ ) : v( (std::size_t) nsub * nx_ * ny_ * nr_, 0.0 ), nx( nx_ ), ny( ny_ ), nr( nr_ ) {}
    double&       operator()( int s, int x, int y, int r ) { return v[( ( (std::size_t) s * nx + x ) * ny + y ) * nr + r]; }
    double        operator()( int s, int x, int y, int r ) const { return v[( ( (std::size_t) s * nx + x ) * ny + y ) * nr + r]; }
};

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    {
        const int                   LDR = 4, S_lat = 4, S_rad = 1, M = 3; // nodes_lat=5, nodes_rad=5
        const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 };

        AdaptiveForest f( M, S_lat, S_rad );
        f.refine( { ForestLeaf{ SubdomainInfo{ 0, 2, 1, 0 }, 0 } } ); // refine interior block -> A sees 4 fine
        CHECK( f.validate() );

        auto dom = DistributedDomain::create_adaptive_on_comm(
            MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );

        const int nx = dom.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nr = dom.domain_info().subdomain_num_nodes_radially();
        const int ny = nx;
        CHECK( nx == 5 && nr == 5 );

        // A = coarse interior block (1,1,0), finest anchor (8,8,0); its x-high face sees B's 4 children.
        const SubdomainInfo A_anchor{ 0, 1 << M, 1 << M, 0 };
        CHECK( dom.subdomains().count( A_anchor ) == 1 );
        const int A_sub = std::get< 0 >( dom.subdomains().at( A_anchor ) );

        // collect A's x-high finer neighbors (should be 4, octants 0..3)
        std::vector< const AdaptiveFaceNeighbor* > kids;
        for ( const auto& nb : dom.adaptive_neighborhood( A_anchor ).faces )
            if ( nb.my_face == Face::XHIGH && nb.rel_level == +1 )
                kids.push_back( &nb );
        CHECK( kids.size() == 4 );

        Field field( (int) dom.subdomains().size(), nx, ny, nr );

        // give each child's shared (x-low) face a distinct known pattern; A's face starts at 0.
        auto pattern = []( int oct, int a1, int a2 ) { return ( oct + 1 ) * 100.0 + a1 * 10.0 + a2; };
        std::vector< double > expected_coarse( (std::size_t) ny * nr, 0.0 ); // independent P^T prediction
        for ( const auto* nb : kids )
        {
            const int      ksub = std::get< 0 >( dom.subdomains().at( nb->anchor ) );
            const FaceAxes kfa  = face_axes( nb->neighbor_face, nx, ny, nr ); // child's x-low face
            std::vector< double > slab( (std::size_t) ny * nr );
            for ( int a1 = 0; a1 < ny; ++a1 )
                for ( int a2 = 0; a2 < nr; ++a2 )
                    slab[a1 * nr + a2] = pattern( nb->sub_octant, a1, a2 );
            insert_slab( field, ksub, kfa, slab );
            restrict_face_add( slab, expected_coarse, ny, nr, nb->sub_octant ); // predict A's assembled face
        }

        // run the exchange
        exchange_faces_2to1( dom, field, nx, ny, nr );

        // Only face-INTERIOR nodes are exclusively owned by this 2:1 interface; edge/corner nodes are
        // shared with other faces and also receive same-level (rel 0) contributions handled elsewhere.
        auto interior = []( int a1, int a2, int m1, int m2 ) {
            return a1 > 0 && a1 < m1 - 1 && a2 > 0 && a2 < m2 - 1;
        };

        // (1) A's assembled x-high face matches the independent P^T sum (interior nodes)
        const FaceAxes A_fa = face_axes( Face::XHIGH, nx, ny, nr );
        auto           A_face = extract_slab( field, A_sub, A_fa );
        bool           coarse_ok = true;
        int            interior_checked = 0;
        for ( int a1 = 0; a1 < ny; ++a1 )
            for ( int a2 = 0; a2 < nr; ++a2 )
                if ( interior( a1, a2, ny, nr ) )
                {
                    ++interior_checked;
                    if ( !close( A_face[a1 * nr + a2], expected_coarse[a1 * nr + a2] ) )
                        coarse_ok = false;
                }
        CHECK( coarse_ok );
        CHECK( interior_checked == 9 ); // 3x3 interior nodes actually exercised

        // (2) each child's interior coincident nodes broadcast the assembled value; interior hanging
        //     nodes unchanged
        bool bcast_ok = true, hanging_kept = true;
        for ( const auto* nb : kids )
        {
            const int      ksub = std::get< 0 >( dom.subdomains().at( nb->anchor ) );
            const FaceAxes kfa  = face_axes( nb->neighbor_face, nx, ny, nr );
            auto           kslab = extract_slab( field, ksub, kfa );
            for ( const auto& nd : face_correspondence( ny, nr, nb->sub_octant ) )
            {
                if ( !interior( nd.a1, nd.a2, ny, nr ) )
                    continue;
                const double val = kslab[nd.a1 * nr + nd.a2];
                if ( nd.coincident )
                {
                    if ( !close( val, expected_coarse[nd.pa1[0] * nr + nd.pa2[0]] ) )
                        bcast_ok = false;
                }
                else if ( !close( val, pattern( nb->sub_octant, nd.a1, nd.a2 ) ) )
                    hanging_kept = false;
            }
        }
        CHECK( bcast_ok );
        CHECK( hanging_kept );
    }

    const int rc = g_failures;
    if ( g_failures == 0 )
        std::printf( "test_adaptive_exchange: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_exchange: %d/%d FAILURE(S)\n", g_failures, g_checks );
    MPI_Finalize();
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
