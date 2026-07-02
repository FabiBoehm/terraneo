// GPU integration test for depth-aware geometry: allocate + fill the coord/radii grids of an adaptive
// DistributedDomain on the device and validate them.
//
// (1) A subdivision-0 adaptive mesh must reproduce the trusted uniform mesh's geometry exactly.
// (2) A refined mesh must keep all lateral nodes on the unit sphere and produce in-bounds, monotonic,
//     genuinely-finer radii.
//
// Needs a GPU (Kokkos::initialize with CUDA) -- run on an H100 node.

#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace terra;
using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::SubdomainInfo;
using grid::shell::subdomain_to_rank_all_root;
using grid::shell::amr::AdaptiveForest;
using grid::shell::amr::ForestLeaf;

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

// Gather all lateral node positions of a domain into a sorted flat vector (for set comparison).
static std::vector< std::array< double, 3 > > gather_points( const DistributedDomain& dom )
{
    auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom );
    auto host   = Kokkos::create_mirror_view( coords );
    Kokkos::deep_copy( host, coords );
    const int nlat = dom.domain_info().subdomain_num_nodes_per_side_laterally();
    std::vector< std::array< double, 3 > > pts;
    for ( std::size_t s = 0; s < dom.subdomains().size(); ++s )
        for ( int i = 0; i < nlat; ++i )
            for ( int j = 0; j < nlat; ++j )
                pts.push_back( { host( s, i, j, 0 ), host( s, i, j, 1 ), host( s, i, j, 2 ) } );
    std::sort( pts.begin(), pts.end() );
    return pts;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        const int                   LDR = 3, S_lat = 2, S_rad = 1, M = 3;
        const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 }; // 4 layers, spacing 0.25
        const double                r_min = 1.0, r_max = 2.0, base_dr = 0.25;

        // ---- (1) subdivision-0 adaptive geometry == uniform geometry ----------------------------
        {
            AdaptiveForest base( M, S_lat, S_rad ); // all leaves at subdivision 0
            auto dom_a = DistributedDomain::create_adaptive_on_comm(
                MPI_COMM_WORLD, LDR, radii, base, subdomain_to_rank_all_root );
            auto dom_u = DistributedDomain::create_uniform( LDR, radii, 1, 0, subdomain_to_rank_all_root );

            CHECK( dom_a.subdomains().size() == dom_u.subdomains().size() );

            auto pa = gather_points( dom_a );
            auto pu = gather_points( dom_u );
            CHECK( pa.size() == pu.size() );
            bool same = ( pa.size() == pu.size() );
            for ( std::size_t k = 0; same && k < pa.size(); ++k )
                for ( int c = 0; c < 3; ++c )
                    if ( std::fabs( pa[k][c] - pu[k][c] ) > 1e-13 )
                        same = false;
            CHECK( same ); // subdivision-0 mesh is geometrically identical to the uniform mesh
        }

        // ---- (2) refined adaptive geometry sanity ----------------------------------------------
        {
            AdaptiveForest f( M, S_lat, S_rad );
            f.refine( { ForestLeaf{ SubdomainInfo{ 0, 1, 0, 0 }, 0 } } );
            f.balance_2to1();
            auto dom = DistributedDomain::create_adaptive_on_comm(
                MPI_COMM_WORLD, LDR, radii, f, subdomain_to_rank_all_root );

            // lateral: every node on the unit sphere
            auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom );
            auto ch     = Kokkos::create_mirror_view( coords );
            Kokkos::deep_copy( ch, coords );
            const int nlat = dom.domain_info().subdomain_num_nodes_per_side_laterally();
            bool      on_sphere = true;
            for ( std::size_t s = 0; s < dom.subdomains().size(); ++s )
                for ( int i = 0; i < nlat; ++i )
                    for ( int j = 0; j < nlat; ++j )
                    {
                        const double n = std::sqrt( ch( s, i, j, 0 ) * ch( s, i, j, 0 ) +
                                                    ch( s, i, j, 1 ) * ch( s, i, j, 1 ) +
                                                    ch( s, i, j, 2 ) * ch( s, i, j, 2 ) );
                        if ( std::fabs( n - 1.0 ) > 1e-12 )
                            on_sphere = false;
                    }
            CHECK( on_sphere );

            // radial: in bounds, monotonic per subdomain; some refined leaf has finer spacing
            auto rad = grid::shell::subdomain_shell_radii< double >( dom );
            auto rh  = Kokkos::create_mirror_view( rad );
            Kokkos::deep_copy( rh, rad );
            const int nrad = dom.domain_info().subdomain_num_nodes_radially();

            bool in_bounds = true, monotonic = true, saw_finer = false;
            for ( const auto& [sub, tup] : dom.subdomains() )
            {
                const int s = std::get< 0 >( tup );
                for ( int j = 0; j < nrad; ++j )
                    if ( rh( s, j ) < r_min - 1e-12 || rh( s, j ) > r_max + 1e-12 )
                        in_bounds = false;
                for ( int j = 1; j < nrad; ++j )
                    if ( !( rh( s, j ) > rh( s, j - 1 ) ) )
                        monotonic = false;
                const int subdiv = dom.subdivision_of( sub );
                if ( subdiv >= 1 )
                {
                    const double dr = rh( s, 1 ) - rh( s, 0 );
                    if ( std::fabs( dr - base_dr / ( 1 << subdiv ) ) < 1e-12 && dr < base_dr - 1e-12 )
                        saw_finer = true;
                }
            }
            CHECK( in_bounds );
            CHECK( monotonic );
            CHECK( saw_finer ); // at least one subdivided leaf has the expected finer radial spacing
        }
    }

    const int rc = g_failures;
    if ( g_failures == 0 )
        std::printf( "test_adaptive_geometry_gpu: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_geometry_gpu: %d/%d FAILURE(S)\n", g_failures, g_checks );

    // MPI/Kokkos are finalized by the terra singleton destructors at program exit (the Kokkos Views
    // above are already out of scope by then).
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
