// EpsDivDivKerngen team-tile sweep at every precision.

#include "benchmark_epsdivdiv_common.hpp"

/// Team-tile tuning for the production kernel, at every precision.
///
/// The tile shape is (lat, lat, r) with r_passes radial passes, so the team holds
/// lat^2 * r threads and stages ( r * r_passes + 1 ) * ( lat + 1 )^2 nodes. The fast
/// Dirichlet/Neumann path is launched with LaunchBounds< 128, 5 >, which caps the team
/// at 128 threads; the candidates below respect that, and set_tiling() rejects anything
/// that would not launch or would overflow 64 KiB of scratch.
void run_tile_sweep( const int min_level,
                     const int max_level,
                     const int executions,
                     const int refinement_level_subdomains )
{
    using half  = Kokkos::Experimental::half_t;
    using bhalf = Kokkos::Experimental::bhalf_t;

    struct Tile
    {
        int lat, r, passes;
    };
    // lat^2 * r <= 128, spanning lateral-heavy to radial-heavy shapes.
    const std::vector< Tile > tiles = {
        { 1, 32, 1 }, { 1, 64, 1 }, { 1, 128, 1 }, { 1, 32, 2 }, { 1, 64, 2 },
        { 2, 8, 1 },  { 2, 16, 1 }, { 2, 32, 1 },  { 2, 8, 2 },  { 2, 16, 2 },
        { 2, 32, 2 }, { 4, 2, 1 },  { 4, 4, 1 },   { 4, 8, 1 },  { 4, 4, 2 },
        { 4, 8, 2 },  { 4, 8, 4 },  { 8, 2, 1 },   { 8, 2, 2 },  { 8, 2, 4 },  { 8, 2, 8 },
        // r = 1 column walks for the warp-owner scatter (--tile-scatter 2); its radial
        // run is streamed, so any length launches.
        { 8, 1, 8 },  { 8, 1, 16 }, { 8, 1, 32 },  { 8, 1, 64 }, { 8, 1, 128 }, { 8, 1, 256 },
        // Teams above 128 threads: set_tiling rejects these unless the operator was
        // built with a raised TERRA_EPSDIVDIV_MAX_TEAM, and a rejected shape is skipped.
        { 2, 64, 1 }, { 4, 16, 1 }, { 4, 32, 1 },  { 8, 4, 1 },  { 8, 8, 1 },
    };

    logroot << "Team-tile sweep for EpsDivDivKerngen, all precisions." << std::endl;
    logroot << "min_level " << min_level << ", max_level " << max_level << ", executions "
            << executions << std::endl
            << std::endl;

    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    util::Table table;

    for ( int level = min_level; level <= max_level; ++level )
    {
        for ( const auto& t : tiles )
        {
            struct Arm
            {
                const char* name;
                BenchmarkData ( *fn )( int, int, int, int, int, int );
            };
            const Arm arms[] = {
                { "double", &run_precision< double, double > },
                { "float", &run_precision< float, float > },
                { "bf16", &run_precision< bhalf, bhalf > },
                { "fp16", &run_precision< half, half > },
            };

            for ( const auto& arm : arms )
            {
                const auto d = arm.fn( level, executions, refinement_level_subdomains, t.lat, t.r, t.passes );
                const bool ok = d.duration > 0.0;
                table.add_row( { { "precision", std::string( arm.name ) },
                                 { "level", level },
                                 { "lat", t.lat },
                                 { "r", t.r },
                                 { "passes", t.passes },
                                 { "team", t.lat * t.lat * t.r },
                                 { "dofs", d.dofs },
                                 { "duration (s)", ok ? d.duration : 0.0 },
                                 { "GDoF/s", ok ? d.dofs / d.duration / 1e9 : 0.0 },
                                 { "launched", ok ? 1 : 0 } } );
            }
        }
        logroot << "level " << level << " done" << std::endl;
    }

    table.print_pretty();
    if ( mpi::rank() == 0 )
    {
        std::ofstream out( "./csv/tiles_np" + std::to_string( world_size ) + "_ml" +
                           std::to_string( max_level ) + ".csv" );
        table.print_csv( out );
    }
    table.print_csv( logroot );
}

int main( int argc, char** argv )
{
    return sweep_main( argc, argv, "EpsDivDivKerngen team-tile sweep at every precision.", []( const SweepParameters& p ) {
        run_tile_sweep( p.min_level, p.max_level, p.executions, p.refinement_level_subdomains );
    } );
}
