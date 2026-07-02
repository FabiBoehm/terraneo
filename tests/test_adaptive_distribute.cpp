// Serial test of the distributed comm-plan construction (adaptive_distribute.hpp). Because the plan is
// a pure function of the replicated forest + partition, every rank's plan can be built on ONE process
// and cross-checked: the partition covers the forest exactly once, and for each (owner, member) pair
// the owner's reduce/broadcast buffers line up in size with the member's. No MPI ranks, no Kokkos.

#include "terra/grid/shell/adaptive_distribute.hpp"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <map>
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

// find the RankBuffer for a partner (or nullptr)
static const RankBuffer* find_buf( const std::vector< RankBuffer >& v, int partner )
{
    for ( const auto& b : v )
        if ( b.partner == partner )
            return &b;
    return nullptr;
}
static std::size_t buf_size( const std::vector< RankBuffer >& v, int partner )
{
    auto* b = find_buf( v, partner );
    return b ? b->nodes.size() : 0;
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    {
        const int                   LDR = 3, S_lat = 2, S_rad = 1, M = 3;
        const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 };

        AdaptiveForest f( M, S_lat, S_rad ); // base uniform mesh: diamond seams give cross-rank classes
        CHECK( f.validate() );

        const int nx = ( 1 << LDR ) / S_lat + 1, nr = 5, ny = nx;

        // global class structure (all leaves on one rank)
        const auto dom_all = DistributedDomain::create_adaptive_for_rank(
            LDR, radii, f, subdomain_to_rank_all_root, 0 );
        const auto t_all = build_2to1_tables( dom_all, nx, ny, nr );

        for ( int nprocs : { 2, 4, 10 } )
        {
            const AnchorRankFn rank_of = [nprocs]( const SubdomainInfo& a ) {
                return static_cast< int >( static_cast< long >( a.diamond_id() ) * nprocs / 10 );
            };

            // partition covers every leaf exactly once
            std::vector< int > owned( nprocs, 0 );
            for ( const auto& leaf : f.leaves() )
                owned[rank_of( f.finest_anchor( leaf ) )]++;
            long total = 0;
            for ( int r = 0; r < nprocs; ++r )
                total += owned[r];
            CHECK( total == (long) f.size() );

            // build every rank's plan
            std::vector< DistributedPlan > plans;
            for ( int r = 0; r < nprocs; ++r )
                plans.push_back( build_distributed_plan( f, dom_all, t_all, rank_of, r, nprocs ) );

            // cross-rank buffers line up: owner's reduce_recv[m] == member m's reduce_send[owner],
            // and owner's bcast_send[m] == member m's bcast_recv[owner]
            bool any_crossrank = false;
            for ( int owner = 0; owner < nprocs; ++owner )
                for ( int m = 0; m < nprocs; ++m )
                {
                    if ( m == owner )
                        continue;
                    const std::size_t rr = buf_size( plans[owner].reduce_recv, m );
                    const std::size_t rs = buf_size( plans[m].reduce_send, owner );
                    CHECK( rr == rs );
                    const std::size_t bs = buf_size( plans[owner].bcast_send, m );
                    const std::size_t br = buf_size( plans[m].bcast_recv, owner );
                    CHECK( bs == br );
                    CHECK( rr == bs ); // reduce and broadcast touch the same class reps
                    if ( rr > 0 )
                        any_crossrank = true;
                }
            CHECK( any_crossrank ); // diamond seams guarantee cross-rank classes

            // an owner never sends its own partial to itself; a non-owner never owns
            for ( int r = 0; r < nprocs; ++r )
            {
                for ( const auto& b : plans[r].reduce_send )
                    CHECK( b.partner < r );  // send only to a lower-ranked owner
                for ( const auto& b : plans[r].reduce_recv )
                    CHECK( b.partner > r );  // receive only from higher-ranked members
            }
        }
    }

    if ( g_failures == 0 )
        std::printf( "test_adaptive_distribute: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_distribute: %d/%d FAILURE(S)\n", g_failures, g_checks );
    MPI_Finalize();
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
