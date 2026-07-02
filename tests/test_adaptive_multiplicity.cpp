// Integration test for edge/vertex assembly via global node classes (adaptive_exchange.hpp).
//
// (1) Multiplicity counting on a UNIFORM adaptive domain: with field == 1 everywhere, the assembled
//     value at every node equals the number of subdomains sharing it -- 1 (interior), 2 (face),
//     4 (edge), 8 (vertex). This is exactly the property the face-only exchange could not deliver.
// (2) Mass conservation on a REFINED domain (with a lateral AND a radial 2:1 interface): each copy
//     contributes exactly one unit (genuine copies via class sums, hanging copies via P^T weights that
//     sum to 1), so the total over genuine DoFs equals the total number of copies.
//
// MPI, no Kokkos -> runs on a login node under mpiexec -np 1.

#include "terra/grid/shell/adaptive_exchange.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
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
    : v( (std::size_t) nsub * nx_ * ny_ * nr_, 1.0 ), nx( nx_ ), ny( ny_ ), nr( nr_ )
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
        // S_rad = 2 so true 8-way block vertices exist. nodes: nx = ny = 5, nr = 4/2 + 1 = 3.
        const int                   LDR = 4, S_lat = 4, S_rad = 2, M = 3;
        const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 };

        // ---- (1) uniform domain: assembled constant-1 field counts sharing subdomains -------------
        {
            AdaptiveForest f( M, S_lat, S_rad );
            auto           dom = DistributedDomain::create_adaptive_on_comm(
                MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );
            const int nx = dom.domain_info().subdomain_num_nodes_per_side_laterally();
            const int nr = dom.domain_info().subdomain_num_nodes_radially();
            const int ny = nx;
            CHECK( nx == 5 && nr == 3 );

            Field field( (int) dom.subdomains().size(), nx, ny, nr ); // all ones
            exchange_faces_2to1( dom, field, nx, ny, nr );

            // block (diamond 1, own (1,1,0)) -> finest anchor (8,8,0); interior in-diamond laterally,
            // r-low face is the CMB (domain boundary), r-high face shared with block (1,1,1).
            const int B = local_index( dom, SubdomainInfo{ 1, 8, 8, 0 } );

            CHECK( close( field( B, 2, 2, 1 ), 1.0 ) ); // block-interior node
            CHECK( close( field( B, 0, 2, 1 ), 2.0 ) ); // x-low face interior
            CHECK( close( field( B, 2, 2, 2 ), 2.0 ) ); // r-high face interior
            CHECK( close( field( B, 0, 0, 1 ), 4.0 ) ); // x-low/y-low lateral edge
            CHECK( close( field( B, 0, 2, 2 ), 4.0 ) ); // x-low/r-high edge
            CHECK( close( field( B, 0, 0, 2 ), 8.0 ) ); // x-low/y-low/r-high vertex (8 blocks)
            CHECK( close( field( B, 0, 0, 0 ), 4.0 ) ); // same edge at the CMB: only 4 blocks

            // consistency: all copies of the vertex class hold the same assembled value
            const int B2 = local_index( dom, SubdomainInfo{ 1, 0, 0, 0 } ); // block (0,0,0)
            CHECK( close( field( B2, nx - 1, ny - 1, nr - 1 ), 8.0 ) );     // same vertex from (0,0,0)
        }

        // ---- (2) refined domain: exact mass conservation into genuine DoFs ------------------------
        {
            AdaptiveForest f( M, S_lat, S_rad );
            f.refine( { ForestLeaf{ SubdomainInfo{ 0, 2, 1, 0 }, 0 } } ); // lateral AND radial 2:1
            f.balance_2to1();
            CHECK( f.validate() );
            auto dom = DistributedDomain::create_adaptive_on_comm(
                MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );
            const int nsub = (int) dom.subdomains().size();
            const int nx   = dom.domain_info().subdomain_num_nodes_per_side_laterally();
            const int nr   = dom.domain_info().subdomain_num_nodes_radially();
            const int ny   = nx;

            const auto t = build_2to1_tables( dom, nx, ny, nr );
            CHECK( !t.con_np.empty() );
            CHECK( !t.cls_members.empty() );

            Field field( nsub, nx, ny, nr ); // all ones
            apply_exchange_tables( t, field );

            // classify copies: hanging (skip), non-canonical class member (counted via canonical),
            // else count the copy's value.
            std::set< std::array< int, 4 > > hang, noncanon;
            for ( const auto& d : t.con_dst )
                hang.insert( { d.s, d.x, d.y, d.r } );
            for ( std::size_t c = 0; c + 1 < t.cls_offsets.size(); ++c )
                for ( int m = t.cls_offsets[c] + 1; m < t.cls_offsets[c + 1]; ++m )
                {
                    const Idx4& i = t.cls_members[m];
                    noncanon.insert( { i.s, i.x, i.y, i.r } );
                }

            double      total   = 0.0;
            long        ncopies = 0;
            for ( int s = 0; s < nsub; ++s )
                for ( int x = 0; x < nx; ++x )
                    for ( int y = 0; y < ny; ++y )
                        for ( int r = 0; r < nr; ++r )
                        {
                            ++ncopies;
                            const std::array< int, 4 > k{ s, x, y, r };
                            if ( hang.count( k ) || noncanon.count( k ) )
                                continue;
                            total += field( s, x, y, r );
                        }
            CHECK( std::fabs( total - (double) ncopies ) < 1e-8 ); // every copy delivered exactly once
        }
    }

    const int rc = g_failures;
    if ( g_failures == 0 )
        std::printf( "test_adaptive_multiplicity: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_multiplicity: %d/%d FAILURE(S)\n", g_failures, g_checks );
    MPI_Finalize();
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
