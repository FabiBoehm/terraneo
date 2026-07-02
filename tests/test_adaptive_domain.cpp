// Unit test for DistributedDomain::create_adaptive_on_comm (AMR factory).
//
// Builds a DistributedDomain from an AdaptiveForest and checks the resulting subdomain map: one
// subdomain per leaf, keyed by unique finest-frame anchors, with the right per-leaf subdivision and a
// dense local-index bijection. Only std::maps are touched (no grid allocation), so this runs on a
// login node with plain MPI -- no Kokkos initialization required.

#include "terra/grid/shell/spherical_shell.hpp"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <set>

using namespace terra::grid::shell;
using amr::AdaptiveForest;
using amr::ForestLeaf;

static int g_failures = 0;
static int g_checks   = 0;
#define CHECK( cond )                                                             \
    do                                                                            \
    {                                                                             \
        ++g_checks;                                                               \
        if ( !( cond ) )                                                          \
        {                                                                         \
            std::printf( "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond );       \
            ++g_failures;                                                         \
        }                                                                         \
    } while ( 0 )

// Every leaf's finest anchor is a subdomain key, with matching subdivision; sizes agree; local indices
// form a dense 0..N-1 bijection.
static void check_domain_matches_forest( const DistributedDomain& dom, const AdaptiveForest& f )
{
    CHECK( dom.adaptive() );
    CHECK( dom.subdomains().size() == f.size() ); // 1:1, and anchors are unique (no collisions)

    for ( const auto& leaf : f.leaves() )
    {
        const SubdomainInfo anchor = f.finest_anchor( leaf );
        CHECK( dom.subdomains().count( anchor ) == 1 );
        CHECK( dom.subdivision_of( anchor ) == leaf.subdivision );
    }

    std::set< int > idxs;
    for ( const auto& [sub, tup] : dom.subdomains() )
    {
        const int idx = std::get< 0 >( tup );
        idxs.insert( idx );
        CHECK( dom.subdomain_info_from_local_idx( idx ) == sub ); // round-trip
    }
    CHECK( idxs.size() == dom.subdomains().size() );
    if ( !idxs.empty() )
    {
        CHECK( *idxs.begin() == 0 );
        CHECK( *idxs.rbegin() == static_cast< int >( dom.subdomains().size() ) - 1 ); // dense 0..N-1
    }
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );

    const int                   LDR   = 3;                              // base block = 8/2 = 4 cells/side
    const int                   S_lat = 2, S_rad = 1, M = 3;
    const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 }; // 4 layers, divisible by S_rad

    // --- uniform base forest == today's uniform subdomain set -------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        auto           dom = DistributedDomain::create_adaptive_on_comm(
            MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );
        CHECK( dom.subdomains().size() == 40u );
        CHECK( dom.domain_info().num_global_subdomains() == 40 ); // base DomainInfo agrees
        for ( const auto& [sub, tup] : dom.subdomains() )
            CHECK( dom.subdivision_of( sub ) == 0 );
        check_domain_matches_forest( dom, f );
    }

    // --- refined + balanced forest ----------------------------------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        f.refine( { ForestLeaf{ SubdomainInfo{ 0, 1, 0, 0 }, 0 } } );
        f.refine( { ForestLeaf{ SubdomainInfo{ 0, 2, 0, 0 }, 1 } } );
        f.balance_2to1();
        CHECK( f.validate() );

        auto dom = DistributedDomain::create_adaptive_on_comm(
            MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );
        check_domain_matches_forest( dom, f );

        // at least one subdivision-1 leaf exists, and its anchor maps to subdivision 1 in the domain
        bool checked_a_subdiv1 = false;
        for ( const auto& leaf : f.leaves() )
            if ( leaf.subdivision == 1 )
            {
                const SubdomainInfo a = f.finest_anchor( leaf );
                CHECK( dom.subdomains().count( a ) == 1 );
                CHECK( dom.subdivision_of( a ) == 1 );
                checked_a_subdiv1 = true;
                break;
            }
        CHECK( checked_a_subdiv1 );
    }

    // --- adaptive neighborhoods populated and self-consistent -------------------------------------
    {
        AdaptiveForest f( M, S_lat, S_rad );
        f.refine( { ForestLeaf{ SubdomainInfo{ 0, 1, 0, 0 }, 0 } } ); // single-level jump (already 2:1)
        CHECK( f.validate() );

        auto dom = DistributedDomain::create_adaptive_on_comm(
            MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );

        // coarse leaf (0,0,0,0) sees 4 finer across x-high + 1 same across y-high
        const SubdomainInfo Lanchor{ 0, 0, 0, 0 };
        CHECK( dom.subdomains().count( Lanchor ) == 1 );
        const auto& nbh = dom.adaptive_neighborhood( Lanchor );
        CHECK( nbh.faces.size() == 5u );
        int finer = 0, same = 0;
        for ( const auto& fn : nbh.faces )
        {
            if ( fn.rel_level == +1 ) ++finer;
            if ( fn.rel_level == 0 ) ++same;
        }
        CHECK( finer == 4 && same == 1 );

        // closure: on a single rank, every referenced neighbor anchor is itself a local subdomain
        for ( const auto& [sub, tup] : dom.subdomains() )
            for ( const auto& fn : dom.adaptive_neighborhood( sub ).faces )
                CHECK( dom.subdomains().count( fn.anchor ) == 1 );
    }

    // --- uniform (non-adaptive) domain is unaffected: adaptive() false, no neighborhoods -----------
    {
        auto uni = DistributedDomain::create_uniform( LDR, radii, 1, 0, subdomain_to_rank_all_root );
        CHECK( !uni.adaptive() );
        for ( const auto& [sub, tup] : uni.subdomains() )
        {
            CHECK( uni.subdivision_of( sub ) == 0 );
            CHECK( uni.adaptive_neighborhood( sub ).faces.empty() );
        }
    }

    int rc = g_failures;
    if ( g_failures == 0 )
        std::printf( "test_adaptive_domain: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_domain: %d/%d FAILURE(S)\n", g_failures, g_checks );

    MPI_Finalize();
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
