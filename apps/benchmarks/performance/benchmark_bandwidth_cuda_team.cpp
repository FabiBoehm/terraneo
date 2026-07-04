/**
 * Mirror of the EpsilonDivDivKerngen (fast DN) launch pattern applied to a
 * STREAM-triad — intended as the "realistic roof" for the operator's L2/HBM
 * bandwidth attainment.
 *
 * Replicates exactly (from epsilon_divdiv_kerngen.hpp):
 *   - TeamPolicy<LaunchBounds<128, 5>>
 *   - lat_tile = 4, r_tile = 8, r_passes = 2 -> team_size = 128, r_tile_block = 16
 *   - tiling over (subdomains, hex_lat, hex_lat, hex_rad) with the same team/block layout
 *   - thread mapping  tr = tid%r_tile,  tx = (tid/r_tile)%lat_tile,  ty = tid/(r_tile*lat_tile)
 *   - sequential r_passes loop inside the kernel
 *   - 4D Kokkos::View<double****> (sub, x, y, r) for each of A, B, C
 *
 * Kernel body (per element): A(...) = B(...) + scalar * C(...)
 *
 * Usage:
 *   benchmark_bandwidth_cuda_team [subdomains] [hex_lat] [hex_rad]
 *   (defaults to the L8 shape observed in benchmark_operators: 10 x 256 x 256)
 */
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

using Team = Kokkos::TeamPolicy<>::member_type;
using View4D = Kokkos::View< double**** >;

struct Result
{
    double bw_gbs;
    double time_ms;
    double bytes_moved;
};

// Kernel functor mirroring the operator's team launch + r_passes loop
struct TriadTeam
{
    View4D A, B, C;
    double scalar;
    int    lat_tiles, r_tiles;      // # tiles per subdomain
    int    lat_tile, r_tile, r_passes;
    int    hex_lat, hex_rad;

    KOKKOS_INLINE_FUNCTION
    void operator()( const Team& team ) const
    {
        const int tid = team.team_rank();
        const int lid = team.league_rank();

        // Decode block id: (subdomain, lat_x, lat_y, r) tile coordinates
        int tmp           = lid;
        const int r_tile_id = tmp % r_tiles;   tmp /= r_tiles;
        const int lat_y_id  = tmp % lat_tiles; tmp /= lat_tiles;
        const int lat_x_id  = tmp % lat_tiles;
        const int sub_id    = tmp / lat_tiles;

        const int x0 = lat_x_id * lat_tile;
        const int y0 = lat_y_id * lat_tile;
        const int r0 = r_tile_id * ( r_tile * r_passes );

        // Thread mapping identical to operator (r fastest-varying for coalescing)
        const int tr = tid % r_tile;
        const int tx = ( tid / r_tile ) % lat_tile;
        const int ty = tid / ( r_tile * lat_tile );

        // Inactive threads (shouldn't happen with team_size = lat_tile*lat_tile*r_tile)
        if ( tr >= r_tile )
            return;

        const int x = x0 + tx;
        const int y = y0 + ty;

        // Bounds-check once per thread
        if ( sub_id >= (int)A.extent( 0 ) || x >= (int)A.extent( 1 ) || y >= (int)A.extent( 2 ) )
            return;

        for ( int pass = 0; pass < r_passes; ++pass )
        {
            const int r = r0 + pass * r_tile + tr;
            if ( r >= (int)A.extent( 3 ) )
                continue;
            A( sub_id, x, y, r ) = B( sub_id, x, y, r ) + scalar * C( sub_id, x, y, r );
        }
    }
};

Result run_triad_team( int subdomains, int hex_lat, int hex_rad, int warmup, int iters, double scalar )
{
    // Operator launch constants (from kerngen)
    constexpr int lat_tile  = 4;
    constexpr int r_tile    = 8;
    constexpr int r_passes  = 2;
    constexpr int team_size = lat_tile * lat_tile * r_tile; // 128

    const int lat_tiles = ( hex_lat + lat_tile - 1 ) / lat_tile;
    const int r_tiles   = ( hex_rad + ( r_tile * r_passes ) - 1 ) / ( r_tile * r_passes );
    const int blocks    = subdomains * lat_tiles * lat_tiles * r_tiles;

    // Allocate 4D views sized like VectorQ1Vec: (sub, hex_lat+1, hex_lat+1, hex_rad+1)
    // (+1 to include the shared boundary nodes just like the operator's grid)
    const int nx = hex_lat + 1;
    const int ny = hex_lat + 1;
    const int nr = hex_rad + 1;
    View4D A( Kokkos::view_alloc( Kokkos::WithoutInitializing, "A" ), subdomains, nx, ny, nr );
    View4D B( Kokkos::view_alloc( Kokkos::WithoutInitializing, "B" ), subdomains, nx, ny, nr );
    View4D C( Kokkos::view_alloc( Kokkos::WithoutInitializing, "C" ), subdomains, nx, ny, nr );

    // Init on device
    Kokkos::parallel_for(
        "init",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { subdomains, nx, ny, nr } ),
        KOKKOS_LAMBDA( int s, int i, int j, int k ) {
            A( s, i, j, k ) = 0.0;
            B( s, i, j, k ) = 1.0;
            C( s, i, j, k ) = 2.0;
        } );
    Kokkos::fence();

    TriadTeam kernel{ A, B, C, scalar, lat_tiles, r_tiles, lat_tile, r_tile, r_passes, hex_lat, hex_rad };

    using Policy = Kokkos::TeamPolicy< Kokkos::LaunchBounds< 128, 5 > >;

    // warm-up
    for ( int w = 0; w < warmup; ++w )
        Kokkos::parallel_for( "triad_team_warmup", Policy( blocks, team_size ), kernel );
    Kokkos::fence();

    Kokkos::Timer timer;
    for ( int it = 0; it < iters; ++it )
        Kokkos::parallel_for( "triad_team", Policy( blocks, team_size ), kernel );
    Kokkos::fence();

    const double elapsed_per = timer.seconds() / iters;

    // Bytes moved per iter = active_threads * r_passes * 3 * sizeof(double)
    // where active_threads = blocks * team_size (all threads active since lat_tile*lat_tile*r_tile == team_size)
    const size_t active_threads = (size_t)blocks * team_size;
    const double bytes_moved    = (double)active_threads * r_passes * 3.0 * sizeof( double );

    return { bytes_moved / elapsed_per / 1e9, elapsed_per * 1e3, bytes_moved };
}

int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope( argc, argv );

    int subdomains = ( argc > 1 ) ? std::atoi( argv[1] ) : 10;
    int hex_lat    = ( argc > 2 ) ? std::atoi( argv[2] ) : 256;
    int hex_rad    = ( argc > 3 ) ? std::atoi( argv[3] ) : 256;

    const int    warmup = 5;
    const int    iters  = 50;
    const double scalar = 3.14159;

    std::cout << "TriadTeam benchmark (EpsDivDiv fast-DN launch mirror)\n";
    std::cout << "  subdomains=" << subdomains << ", hex_lat=" << hex_lat << ", hex_rad=" << hex_rad << "\n";

    // Sweep a few sizes to show L2 vs HBM crossover.
    // Rows: (subdomains, hex_lat, hex_rad)
    struct Case
    {
        int subs, lat, rad;
        const char* label;
    };
    Case cases[] = {
        { 1, 16, 16,  "L2-tiny   " },
        { 1, 32, 32,  "L2-small  " },
        { 1, 64, 64,  "L2-mid    " },
        { 1, 128, 128, "mid       " },
        { 1, 256, 256, "L1-large  " },
        { 10, 256, 256, "L8 op match" },
        { subdomains, hex_lat, hex_rad, "user      " },
    };

    std::cout << "\n" << std::setw( 12 ) << "label"
              << std::setw( 10 ) << "subs"
              << std::setw( 8 ) << "hex_lat"
              << std::setw( 8 ) << "hex_rad"
              << std::setw( 12 ) << "data (MB)"
              << std::setw( 12 ) << "time (ms)"
              << std::setw( 14 ) << "BW (GB/s)"
              << "\n";
    std::cout << std::string( 76, '-' ) << "\n";

    std::cerr << "label,subs,hex_lat,hex_rad,total_mb,time_ms,bw_gbs\n";

    for ( const auto& c : cases )
    {
        auto r         = run_triad_team( c.subs, c.lat, c.rad, warmup, iters, scalar );
        double arr_mb  = (double)c.subs * ( c.lat + 1 ) * ( c.lat + 1 ) * ( c.rad + 1 ) * sizeof( double ) / 1.0e6;
        double total_mb = 3.0 * arr_mb;
        std::cout << std::setw( 12 ) << c.label
                  << std::setw( 10 ) << c.subs
                  << std::setw( 8 ) << c.lat
                  << std::setw( 8 ) << c.rad
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << total_mb
                  << std::setw( 12 ) << std::fixed << std::setprecision( 2 ) << r.time_ms
                  << std::setw( 14 ) << std::fixed << std::setprecision( 1 ) << r.bw_gbs
                  << "\n";
        std::cerr << c.label << "," << c.subs << "," << c.lat << "," << c.rad << ","
                  << total_mb << "," << r.time_ms << "," << r.bw_gbs << "\n";
    }

    return 0;
}
