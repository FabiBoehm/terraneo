// Integration test for the hanging-node constraint (apply_constraint_tables / constrain_hanging_faces).
//
// Builds a real adaptive DistributedDomain with an interior 2:1 interface, puts a linear field on the
// coarse face and garbage on the fine faces, applies the constraint, and checks every interior hanging
// node equals the interpolation of its coarse parents (coincident nodes untouched -- those belong to the
// exchange broadcast). Also checks idempotence and the full exchange+constraint pipeline (conforming
// interface). MPI, no Kokkos -> runs on a login node under mpiexec -np 1.

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

struct Field
{
    std::vector< double > v;
    int                   nx, ny, nr;
    Field( int nsub, int nx_, int ny_, int nr_ )
    : v( (std::size_t) nsub * nx_ * ny_ * nr_, 0.0 ), nx( nx_ ), ny( ny_ ), nr( nr_ )
    {}
    double& operator()( int s, int x, int y, int r )
    {
        return v[( ( (std::size_t) s * nx + x ) * ny + y ) * nr + r];
    }
    double operator()( int s, int x, int y, int r ) const
    {
        return v[( ( (std::size_t) s * nx + x ) * ny + y ) * nr + r];
    }
};

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    {
        const int                   LDR = 4, S_lat = 4, S_rad = 1, M = 3;
        const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 };

        AdaptiveForest f( M, S_lat, S_rad );
        f.refine( { ForestLeaf{ SubdomainInfo{ 0, 2, 1, 0 }, 0 } } );
        auto dom = DistributedDomain::create_adaptive_on_comm(
            MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );

        const int nx = dom.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nr = dom.domain_info().subdomain_num_nodes_radially();
        const int ny = nx;

        const SubdomainInfo A_anchor{ 0, 1 << M, 1 << M, 0 };
        const int           A_sub = local_index( dom, A_anchor );
        std::vector< const AdaptiveFaceNeighbor* > kids;
        for ( const auto& nb : dom.adaptive_neighborhood( A_anchor ).faces )
            if ( nb.my_face == Face::XHIGH && nb.rel_level == +1 )
                kids.push_back( &nb );
        CHECK( kids.size() == 4 );

        const FaceAxes A_fa = face_axes( Face::XHIGH, nx, ny, nr );
        auto interior = [&]( int a1, int a2 ) { return a1 > 0 && a1 < ny - 1 && a2 > 0 && a2 < nr - 1; };

        // ---- (1) constraint alone: hanging = interpolation of the coarse face -------------------
        {
            Field field( (int) dom.subdomains().size(), nx, ny, nr );

            // linear pattern on A's x-high face; garbage 999 on the children's shared faces
            std::vector< double > A_slab( (std::size_t) ny * nr );
            for ( int a1 = 0; a1 < ny; ++a1 )
                for ( int a2 = 0; a2 < nr; ++a2 )
                    A_slab[a1 * nr + a2] = 2.0 * a1 - 3.0 * a2 + 0.5;
            insert_slab( field, A_sub, A_fa, A_slab );
            for ( const auto* nb : kids )
                insert_slab( field, local_index( dom, nb->anchor ),
                             face_axes( nb->neighbor_face, nx, ny, nr ),
                             std::vector< double >( (std::size_t) ny * nr, 999.0 ) );

            constrain_hanging_faces( dom, field, nx, ny, nr );

            // A's own face untouched by the constraint
            auto A_after = extract_slab( field, A_sub, A_fa );
            bool A_kept  = true;
            for ( std::size_t i = 0; i < A_after.size(); ++i )
                if ( !close( A_after[i], A_slab[i] ) )
                    A_kept = false;
            CHECK( A_kept );

            // interior hanging nodes = P(A_slab); interior coincident nodes still 999
            int  hanging_checked = 0;
            bool hang_ok = true, coin_kept = true;
            for ( const auto* nb : kids )
            {
                auto kslab = extract_slab( field, local_index( dom, nb->anchor ),
                                           face_axes( nb->neighbor_face, nx, ny, nr ) );
                auto pf    = prolongate_face( A_slab, ny, nr, nb->sub_octant );
                for ( const auto& nd : face_correspondence( ny, nr, nb->sub_octant ) )
                {
                    if ( !interior( nd.a1, nd.a2 ) )
                        continue;
                    const double val = kslab[nd.a1 * nr + nd.a2];
                    if ( nd.coincident )
                    {
                        if ( !close( val, 999.0 ) )
                            coin_kept = false;
                    }
                    else
                    {
                        ++hanging_checked;
                        if ( !close( val, pf[nd.a1 * nr + nd.a2] ) )
                            hang_ok = false;
                    }
                }
            }
            CHECK( hang_ok );
            CHECK( coin_kept );
            CHECK( hanging_checked == 4 * 8 ); // 8 interior hanging nodes per 5x5 quadrant face

            // idempotence: a second application changes nothing anywhere
            const std::vector< double > snapshot = field.v;
            constrain_hanging_faces( dom, field, nx, ny, nr );
            bool idem = true;
            for ( std::size_t i = 0; i < snapshot.size(); ++i )
                if ( !close( snapshot[i], field.v[i] ) )
                    idem = false;
            CHECK( idem );
        }

        // ---- (2) pipeline: exchange then constraint -> conforming interface ---------------------
        {
            Field field( (int) dom.subdomains().size(), nx, ny, nr );
            for ( const auto* nb : kids ) // distinct pattern per child face
            {
                std::vector< double > slab( (std::size_t) ny * nr );
                for ( int a1 = 0; a1 < ny; ++a1 )
                    for ( int a2 = 0; a2 < nr; ++a2 )
                        slab[a1 * nr + a2] = ( nb->sub_octant + 1 ) * 100.0 + a1 * 10.0 + a2;
                insert_slab( field, local_index( dom, nb->anchor ),
                             face_axes( nb->neighbor_face, nx, ny, nr ), slab );
            }

            exchange_faces_2to1( dom, field, nx, ny, nr );
            constrain_hanging_faces( dom, field, nx, ny, nr );

            // conforming: each child's interior face == P(assembled coarse face), coincident AND hanging
            auto assembled = extract_slab( field, A_sub, A_fa );
            bool conforming = true;
            for ( const auto* nb : kids )
            {
                auto kslab = extract_slab( field, local_index( dom, nb->anchor ),
                                           face_axes( nb->neighbor_face, nx, ny, nr ) );
                auto pf    = prolongate_face( assembled, ny, nr, nb->sub_octant );
                for ( const auto& nd : face_correspondence( ny, nr, nb->sub_octant ) )
                    if ( interior( nd.a1, nd.a2 ) && !close( kslab[nd.a1 * nr + nd.a2], pf[nd.a1 * nr + nd.a2] ) )
                        conforming = false;
            }
            CHECK( conforming );
        }
    }

    const int rc = g_failures;
    if ( g_failures == 0 )
        std::printf( "test_adaptive_constraint: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_constraint: %d/%d FAILURE(S)\n", g_failures, g_checks );
    MPI_Finalize();
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
