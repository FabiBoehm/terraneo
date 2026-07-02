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

            // ---- diamond seams --------------------------------------------------------------------
            // block (diamond 0, own (0,0,0)): its x-low and y-low faces are diamond seams; its (0,0)
            // corner is the north pole (5 diamonds meet).
            const int C = local_index( dom, SubdomainInfo{ 0, 0, 0, 0 } );
            CHECK( close( field( C, 0, 2, 1 ), 2.0 ) );  // seam face interior: mine + 1 cross-diamond
            CHECK( close( field( C, 0, 4, 1 ), 4.0 ) );  // seam + within-diamond block edge: 2 + 2
            CHECK( close( field( C, 0, 0, 1 ), 5.0 ) );  // north pole, r block-interior: 5 diamonds
            CHECK( close( field( C, 0, 0, 2 ), 10.0 ) ); // north pole at radial interface: 5 x 2
            // south pole via a southern diamond
            const int S = local_index( dom, SubdomainInfo{ 5, 0, 0, 0 } );
            CHECK( close( field( S, 0, 0, 1 ), 5.0 ) );
        }

        // ---- guard: refining a pole/corner-touching block must throw ------------------------------
        {
            AdaptiveForest f( M, S_lat, S_rad );
            f.refine( { ForestLeaf{ SubdomainInfo{ 0, 0, 0, 0 }, 0 } } ); // pole-corner block
            auto dom = DistributedDomain::create_adaptive_on_comm(
                MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );
            const int nx = dom.domain_info().subdomain_num_nodes_per_side_laterally();
            const int nr = dom.domain_info().subdomain_num_nodes_radially();
            bool threw = false;
            try
            {
                (void) build_2to1_tables( dom, nx, nx, nr );
            }
            catch ( const std::runtime_error& )
            {
                threw = true;
            }
            CHECK( threw );
        }

        // ---- 2:1 seam: refining NON-corner diamond-boundary blocks is now supported ----------------
        // d0 block (1,0,0) sits on d0's y-low seam (d4's x-low, FORWARD); d0 block (1,3,0) sits on
        // d0's y-high seam (d5's x-high, REVERSED). Refine both, balance (ripples into d4 and d5),
        // assemble a constant-1 field: exact mass conservation must hold across both seam kinds,
        // covering the flipped and unflipped node identification and the seam-hanging P^T scatters.
        {
            AdaptiveForest f( M, S_lat, S_rad );
            f.refine( { ForestLeaf{ SubdomainInfo{ 0, 1, 0, 0 }, 0 },
                        ForestLeaf{ SubdomainInfo{ 0, 1, 3, 0 }, 0 } } );
            f.balance_2to1();
            CHECK( f.validate() );
            auto dom = DistributedDomain::create_adaptive_on_comm(
                MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );
            const int nsub = (int) dom.subdomains().size();
            const int nx   = dom.domain_info().subdomain_num_nodes_per_side_laterally();
            const int nr   = dom.domain_info().subdomain_num_nodes_radially();
            const int ny   = nx;

            const auto t = build_2to1_tables( dom, nx, ny, nr ); // must NOT throw
            CHECK( !t.con_np.empty() );                          // seam-hanging nodes exist

            // Constraint rows are now local to each fine block (parents = own even nodes), so the
            // cross-diamond coupling lives in the ASSEMBLY: some node class must span diamonds -- that
            // is what makes the seam-hanging nodes' even parents seam-consistent after the exchange.
            auto diamond_of_sub = [&]( int s ) {
                return dom.subdomain_info_from_local_idx( s ).diamond_id();
            };
            for ( std::size_t i = 0; i < t.con_np.size(); ++i )   // constraint rows stay in-diamond
                for ( int p = 0; p < t.con_np[i]; ++p )
                    CHECK( diamond_of_sub( t.con_dst[i].s ) == diamond_of_sub( t.con_src[i][p].s ) );
            bool cross_diamond_class = false;                     // but a class crosses the seam
            for ( std::size_t c = 0; c + 1 < t.cls_offsets.size(); ++c )
            {
                const int d0 = diamond_of_sub( t.cls_members[t.cls_offsets[c]].s );
                for ( int m = t.cls_offsets[c] + 1; m < t.cls_offsets[c + 1]; ++m )
                    if ( diamond_of_sub( t.cls_members[m].s ) != d0 )
                        cross_diamond_class = true;
            }
            CHECK( cross_diamond_class );

            Field field( nsub, nx, ny, nr ); // all ones
            apply_exchange_tables( t, field );

            std::set< std::array< int, 4 > > hang, noncanon;
            for ( const auto& d : t.con_dst )
                hang.insert( { d.s, d.x, d.y, d.r } );
            for ( std::size_t c = 0; c + 1 < t.cls_offsets.size(); ++c )
                for ( int m = t.cls_offsets[c] + 1; m < t.cls_offsets[c + 1]; ++m )
                {
                    const Idx4& i = t.cls_members[m];
                    noncanon.insert( { i.s, i.x, i.y, i.r } );
                }
            double total   = 0.0;
            long   ncopies = 0;
            for ( int s = 0; s < nsub; ++s )
                for ( int x = 0; x < nx; ++x )
                    for ( int y = 0; y < ny; ++y )
                        for ( int r = 0; r < nr; ++r )
                        {
                            ++ncopies;
                            const std::array< int, 4 > k4{ s, x, y, r };
                            if ( hang.count( k4 ) || noncanon.count( k4 ) )
                                continue;
                            total += field( s, x, y, r );
                        }
            CHECK( std::fabs( total - (double) ncopies ) < 1e-8 );
        }

        // ---- uniformly subdivided mesh: EVERY block refined once, pole/corner blocks included -----
        // Conforming seams everywhere (same subdivision on both sides), so the corner guard must NOT
        // throw; the k=1 pole corners still form 5-way classes.
        {
            AdaptiveForest f( M, S_lat, S_rad );
            const auto     base = f.leaves(); // copy: refine() mutates the leaf set
            f.refine( base );
            CHECK( f.size() == 8u * base.size() );
            CHECK( f.validate() );
            auto dom = DistributedDomain::create_adaptive_on_comm(
                MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );
            const int nx = dom.domain_info().subdomain_num_nodes_per_side_laterally();
            const int nr = dom.domain_info().subdomain_num_nodes_radially();

            const auto t = build_2to1_tables( dom, nx, nx, nr ); // must NOT throw
            CHECK( t.con_np.empty() );                           // fully conforming: no hanging nodes

            Field field( (int) dom.subdomains().size(), nx, nx, nr ); // all ones
            apply_exchange_tables( t, field );

            // north pole at subdivision 1: still exactly 5 diamonds
            const int C1 = local_index( dom, SubdomainInfo{ 0, 0, 0, 0 } ); // corner child, k=1
            CHECK( close( field( C1, 0, 0, 1 ), 5.0 ) );
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

        // ---- uniformly subdivided mesh: corner blocks refined CONFORMINGLY are allowed -------------
        // (the corner guard only rejects 2:1 at pentagon corners; a fully subdivided sphere is fine)
        {
            AdaptiveForest f( M, S_lat, S_rad );
            const auto     base = f.leaves(); // copy before mutating
            f.refine( base );
            CHECK( f.validate() );
            CHECK( f.size() == 8u * 10 * S_lat * S_lat * S_rad );

            auto dom = DistributedDomain::create_adaptive_on_comm(
                MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );
            const int nsub = (int) dom.subdomains().size();
            const int nx   = dom.domain_info().subdomain_num_nodes_per_side_laterally();
            const int nr   = dom.domain_info().subdomain_num_nodes_radially();
            const int ny   = nx;

            const auto t = build_2to1_tables( dom, nx, ny, nr ); // must NOT throw
            CHECK( t.con_np.empty() );                           // fully conforming: no hanging nodes

            Field field( nsub, nx, ny, nr ); // all ones
            apply_exchange_tables( t, field );

            // the north pole is still a 5-way class at subdivision 1
            const int C = local_index( dom, SubdomainInfo{ 0, 0, 0, 0 } ); // d0 pole-corner child
            CHECK( close( field( C, 0, 0, 1 ), 5.0 ) );

            std::set< std::array< int, 4 > > noncanon2;
            for ( std::size_t c = 0; c + 1 < t.cls_offsets.size(); ++c )
                for ( int m = t.cls_offsets[c] + 1; m < t.cls_offsets[c + 1]; ++m )
                {
                    const Idx4& i = t.cls_members[m];
                    noncanon2.insert( { i.s, i.x, i.y, i.r } );
                }
            double total   = 0.0;
            long   ncopies = 0;
            for ( int s = 0; s < nsub; ++s )
                for ( int x = 0; x < nx; ++x )
                    for ( int y = 0; y < ny; ++y )
                        for ( int r = 0; r < nr; ++r )
                        {
                            ++ncopies;
                            if ( noncanon2.count( { s, x, y, r } ) )
                                continue;
                            total += field( s, x, y, r );
                        }
            CHECK( std::fabs( total - (double) ncopies ) < 1e-8 );
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
