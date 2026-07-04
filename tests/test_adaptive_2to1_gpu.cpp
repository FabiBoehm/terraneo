// GPU integration test: 2:1 exchange + hanging-node constraint applied ON DEVICE.
//
// Builds an adaptive DistributedDomain, fills a device Grid4DDataScalar with a deterministic pattern,
// applies the device kernels (apply_exchange_device + apply_constraint_device), and compares EVERY node
// against a host golden computed with the (independently verified) host appliers from the same tables.
// Also checks device-side idempotence of the constraint. Needs a GPU -- run on an H100 node.

#include "terra/grid/shell/adaptive_2to1_kokkos.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

using namespace terra;
using grid::shell::DistributedDomain;
using grid::shell::SubdomainInfo;
using grid::shell::subdomain_to_rank_all_root;
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

struct HostField
{
    std::vector< double > v;
    int                   nx, ny, nr;
    HostField( int nsub, int nx_, int ny_, int nr_ )
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
    util::terra_initialize( &argc, &argv );
    {
        const int                   LDR = 4, S_lat = 4, S_rad = 1, M = 3;
        const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 };

        AdaptiveForest fst( M, S_lat, S_rad );
        fst.refine( { ForestLeaf{ SubdomainInfo{ 0, 2, 1, 0 }, 0 } } );
        fst.balance_2to1();
        CHECK( fst.validate() );

        auto dom = DistributedDomain::create_adaptive_on_comm(
            MPI_COMM_WORLD, LDR, radii, fst, subdomain_to_rank_all_root );

        const int nsub = (int) dom.subdomains().size();
        const int nx   = dom.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nr   = dom.domain_info().subdomain_num_nodes_radially();
        const int ny   = nx;
        CHECK( nsub == 167 && nx == 5 && nr == 5 );

        // build tables once; they drive host golden AND device kernels
        const auto t = build_2to1_tables( dom, nx, ny, nr );
        CHECK( !t.asm_w.empty() );
        CHECK( !t.cls_members.empty() );
        CHECK( !t.con_dst.empty() );
        CHECK( t.cls_offsets.size() >= 2 );

        // deterministic per-node pattern
        auto pattern = []( int s, int x, int y, int r ) {
            return 8.0 * s + 0.5 * x + 0.25 * y + 0.125 * r;
        };

        // device field
        auto field = grid::shell::allocate_scalar_grid< double >( "amr2to1_field", dom );
        auto hmirr = Kokkos::create_mirror_view( field );
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        hmirr( s, x, y, r ) = pattern( s, x, y, r );
        Kokkos::deep_copy( field, hmirr );

        // host golden from the same pattern + same tables
        HostField golden( nsub, nx, ny, nr );
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        golden( s, x, y, r ) = pattern( s, x, y, r );
        apply_exchange_tables( t, golden );
        apply_constraint_tables( t, golden );

        // device: exchange + constraint
        const auto dt = upload_2to1_tables( t );
        apply_exchange_device( dt, field );
        apply_constraint_device( dt, field );

        // compare every node (atomics only reorder additions -> tight tolerance)
        Kokkos::deep_copy( hmirr, field );
        long mismatches = 0;
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        if ( std::fabs( hmirr( s, x, y, r ) - golden( s, x, y, r ) ) > 1e-9 )
                        {
                            if ( mismatches < 5 )
                                std::printf( "  mismatch (%d,%d,%d,%d): dev %.15g vs host %.15g\n", s,
                                             x, y, r, hmirr( s, x, y, r ), golden( s, x, y, r ) );
                            ++mismatches;
                        }
        CHECK( mismatches == 0 ); // device pipeline == host-verified golden, all 167*125 nodes

        // device idempotence of the constraint: reapplying changes nothing
        apply_constraint_device( dt, field );
        auto hmirr2 = Kokkos::create_mirror_view( field );
        Kokkos::deep_copy( hmirr2, field );
        long idem_mismatches = 0;
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        if ( std::fabs( hmirr2( s, x, y, r ) - golden( s, x, y, r ) ) > 1e-9 )
                            ++idem_mismatches;
        CHECK( idem_mismatches == 0 );
    }

    // ==== geometric oracle: assembled multiplicity == number of physically coincident nodes ========
    // Uniform domain over the whole sphere (incl. diamond seams and poles). Group every node by its
    // physical position (device-computed unit coords x radius, quantized); with field == 1 the
    // assembled value at every copy must equal its group's size. Validates the complete class topology
    // -- within-diamond faces/edges/vertices, seams, 5-way poles, 3-way tiling vertices -- against
    // geometry, with no hand-derived expectations.
    {
        const int                   LDR = 3, S_lat = 2, S_rad = 1, M = 3;
        const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 };

        AdaptiveForest fst( M, S_lat, S_rad ); // uniform
        auto           dom = DistributedDomain::create_adaptive_on_comm(
            MPI_COMM_WORLD, LDR, radii, fst, subdomain_to_rank_all_root );
        const int nsub = (int) dom.subdomains().size();
        const int nx   = dom.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nr   = dom.domain_info().subdomain_num_nodes_radially();
        const int ny   = nx;

        auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom );
        auto rads   = grid::shell::subdomain_shell_radii< double >( dom );
        auto ch     = Kokkos::create_mirror_view( coords );
        auto rh     = Kokkos::create_mirror_view( rads );
        Kokkos::deep_copy( ch, coords );
        Kokkos::deep_copy( rh, rads );

        // physical-position groups (quantized to 1e-7; node spacing is O(0.1))
        std::map< std::array< long long, 3 >, int > group_size;
        auto pos_key = [&]( int s, int x, int y, int r ) {
            const double rad = rh( s, r );
            return std::array< long long, 3 >{ llround( ch( s, x, y, 0 ) * rad * 1e7 ),
                                               llround( ch( s, x, y, 1 ) * rad * 1e7 ),
                                               llround( ch( s, x, y, 2 ) * rad * 1e7 ) };
        };
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        group_size[pos_key( s, x, y, r )]++;

        // assemble a constant-1 field on device
        auto field = grid::shell::allocate_scalar_grid< double >( "amr_oracle_field", dom );
        auto hf    = Kokkos::create_mirror_view( field );
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        hf( s, x, y, r ) = 1.0;
        Kokkos::deep_copy( field, hf );
        const auto t  = build_2to1_tables( dom, nx, ny, nr );
        const auto dt = upload_2to1_tables( t );
        apply_exchange_device( dt, field );
        Kokkos::deep_copy( hf, field );

        long oracle_mismatches = 0;
        int  max_mult          = 0;
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                    {
                        const int mult = group_size.at( pos_key( s, x, y, r ) );
                        max_mult       = std::max( max_mult, mult );
                        if ( std::fabs( hf( s, x, y, r ) - (double) mult ) > 1e-9 )
                        {
                            if ( oracle_mismatches < 5 )
                                std::printf( "  oracle (%d,%d,%d,%d): assembled %.15g vs coincident %d\n",
                                             s, x, y, r, hf( s, x, y, r ), mult );
                            ++oracle_mismatches;
                        }
                    }
        CHECK( oracle_mismatches == 0 );
        CHECK( max_mult == 5 ); // deepest sharing on this config (S_rad=1): the 5-way pole corners
    }

    // ==== geometric oracle on a SEAM-REFINED mesh + parent bracketing ==============================
    // Refine one block on a FORWARD seam (d0 y-low -> d4) and one on a REVERSED seam (d0 y-high ->
    // d5), balance across the seams, and check on device:
    //  (a) assembled constant-1 multiplicity == geometric coincidence-group size at every genuine
    //      node (hanging positions excluded) -- a wrong seam flip would pair non-coincident nodes;
    //  (b) every hanging node's position ~= the weighted average of its parents' positions (chord
    //      tolerance) -- directly validates the flipped parent selection across the seam.
    {
        const int                   LDR = 4, S_lat = 4, S_rad = 1, M = 3;
        const std::vector< double > radii = { 1.0, 1.25, 1.5, 1.75, 2.0 };

        AdaptiveForest fst( M, S_lat, S_rad );
        fst.refine( { ForestLeaf{ SubdomainInfo{ 0, 1, 0, 0 }, 0 },     // forward seam block
                      ForestLeaf{ SubdomainInfo{ 0, 1, 3, 0 }, 0 } } ); // reversed seam block
        fst.balance_2to1();
        CHECK( fst.validate() );
        auto dom = DistributedDomain::create_adaptive_on_comm(
            MPI_COMM_WORLD, LDR, radii, fst, subdomain_to_rank_all_root );
        const int nsub = (int) dom.subdomains().size();
        const int nx   = dom.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nr   = dom.domain_info().subdomain_num_nodes_radially();
        const int ny   = nx;

        auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom );
        auto rads   = grid::shell::subdomain_shell_radii< double >( dom );
        auto ch     = Kokkos::create_mirror_view( coords );
        auto rh     = Kokkos::create_mirror_view( rads );
        Kokkos::deep_copy( ch, coords );
        Kokkos::deep_copy( rh, rads );

        auto pos = [&]( int s, int x, int y, int r ) {
            const double rad = rh( s, r );
            return std::array< double, 3 >{ ch( s, x, y, 0 ) * rad, ch( s, x, y, 1 ) * rad,
                                            ch( s, x, y, 2 ) * rad };
        };
        auto pos_key = [&]( int s, int x, int y, int r ) {
            const auto p = pos( s, x, y, r );
            return std::array< long long, 3 >{ llround( p[0] * 1e7 ), llround( p[1] * 1e7 ),
                                               llround( p[2] * 1e7 ) };
        };

        const auto t = build_2to1_tables( dom, nx, ny, nr );
        CHECK( !t.con_dst.empty() );

        // (b) parent bracketing for EVERY constraint row (seam rows included)
        double worst_bracket = 0.0;
        for ( std::size_t i = 0; i < t.con_dst.size(); ++i )
        {
            const auto            d = t.con_dst[i];
            const auto            pd = pos( d.s, d.x, d.y, d.r );
            std::array< double, 3 > pi{ 0, 0, 0 };
            for ( int p = t.con_off[i]; p < t.con_off[i + 1]; ++p )
            {
                const auto s  = t.con_par[p];
                const auto ps = pos( s.s, s.x, s.y, s.r );
                for ( int c = 0; c < 3; ++c )
                    pi[c] += t.con_wt[p] * ps[c];
            }
            const double dist = std::sqrt( ( pd[0] - pi[0] ) * ( pd[0] - pi[0] ) +
                                           ( pd[1] - pi[1] ) * ( pd[1] - pi[1] ) +
                                           ( pd[2] - pi[2] ) * ( pd[2] - pi[2] ) );
            worst_bracket = std::max( worst_bracket, dist );
        }
        CHECK( worst_bracket < 5e-3 ); // chord-vs-arc only; wrong parents would be ~0.04+

        // every hanging copy's mass is scattered exactly once (weights of its row sum to 1)
        double worst_mass = 0.0;
        for ( std::size_t i = 0; i < t.con_dst.size(); ++i )
        {
            double wsum = 0.0;
            for ( int p = t.con_off[i]; p < t.con_off[i + 1]; ++p )
                wsum += t.con_wt[p];
            worst_mass = std::max( worst_mass, std::fabs( wsum - 1.0 ) );
        }
        CHECK( worst_mass < 1e-12 );

        // (a) multiplicity oracle, hanging positions excluded. With field == 1, a genuine node's
        // assembled value = its coincidence-group size PLUS the hanging weights P^T-scattered into it
        // (the asm table's dst positions are themselves geometry-validated by the bracketing check).
        std::map< std::array< long long, 3 >, int > group_size;
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        group_size[pos_key( s, x, y, r )]++;
        std::set< std::array< long long, 3 > > hang_pos;
        for ( const auto& d : t.con_dst )
            hang_pos.insert( pos_key( d.s, d.x, d.y, d.r ) );
        std::map< std::array< long long, 3 >, double > hang_inflow;
        for ( std::size_t i = 0; i < t.asm_w.size(); ++i )
        {
            const auto& d = t.asm_dst[i];
            hang_inflow[pos_key( d.s, d.x, d.y, d.r )] += t.asm_w[i];
        }

        auto field = grid::shell::allocate_scalar_grid< double >( "amr_seam_oracle", dom );
        auto hf    = Kokkos::create_mirror_view( field );
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        hf( s, x, y, r ) = 1.0;
        Kokkos::deep_copy( field, hf );
        const auto dt = upload_2to1_tables( t );
        apply_exchange_device( dt, field );
        Kokkos::deep_copy( hf, field );

        long seam_mismatches = 0;
        for ( int s = 0; s < nsub; ++s )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                    {
                        const auto k3 = pos_key( s, x, y, r );
                        if ( hang_pos.count( k3 ) )
                            continue;
                        const auto   fit      = hang_inflow.find( k3 );
                        const double expected = group_size.at( k3 ) +
                                                ( fit == hang_inflow.end() ? 0.0 : fit->second );
                        if ( std::fabs( hf( s, x, y, r ) - expected ) > 1e-9 )
                        {
                            if ( seam_mismatches < 5 )
                                std::printf(
                                    "  seam-oracle (%d,%d,%d,%d): assembled %.15g vs expected %.15g\n",
                                    s, x, y, r, hf( s, x, y, r ), expected );
                            ++seam_mismatches;
                        }
                    }
        CHECK( seam_mismatches == 0 );
    }

    const int rc = g_failures;
    if ( g_failures == 0 )
        std::printf( "test_adaptive_2to1_gpu: ALL PASS (%d checks)\n", g_checks );
    else
        std::printf( "test_adaptive_2to1_gpu: %d/%d FAILURE(S)\n", g_failures, g_checks );
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
