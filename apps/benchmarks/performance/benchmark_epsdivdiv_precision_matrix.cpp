// EpsDivDivKerngen storage/arithmetic breakdown for each narrow precision.

#include "benchmark_epsdivdiv_common.hpp"

/// Storage / arithmetic breakdown. Each 16-bit precision appears three times:
/// matched (both narrow), storage-only (narrow fields, double arithmetic) and
/// arithmetic-only (double fields, narrow arithmetic). The differences separate
/// the cost of the narrow load/store path from the cost of the narrow arithmetic.

void run_precision_matrix( const int min_level,
                           const int max_level,
                           const int executions,
                           const int refinement_level_subdomains )
{
    using half  = Kokkos::Experimental::half_t;
    using bhalf = Kokkos::Experimental::bhalf_t;

    logroot << "Storage / arithmetic breakdown for the EpsDivDivKerngen fast path." << std::endl;
    logroot << "min_level: " << min_level << ", max_level: " << max_level
            << ", executions per level: " << executions << std::endl
            << std::endl;

    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    const int mn = min_level, mx = max_level, ex = executions, r = refinement_level_subdomains;

    sweep_one_precision< double, double >( "double", mn, mx, ex, r, world_size );

    sweep_one_precision< float, float >( "float_matched", mn, mx, ex, r, world_size );
    sweep_one_precision< float, double >( "float_store_only", mn, mx, ex, r, world_size );
    sweep_one_precision< double, float >( "float_arith_only", mn, mx, ex, r, world_size );

    sweep_one_precision< bhalf, bhalf >( "bf16_matched", mn, mx, ex, r, world_size );
    sweep_one_precision< bhalf, double >( "bf16_store_only", mn, mx, ex, r, world_size );
    sweep_one_precision< double, bhalf >( "bf16_arith_only", mn, mx, ex, r, world_size );

    sweep_one_precision< half, half >( "fp16_matched", mn, mx, ex, r, world_size );
    sweep_one_precision< half, double >( "fp16_store_only", mn, mx, ex, r, world_size );
    sweep_one_precision< double, half >( "fp16_arith_only", mn, mx, ex, r, world_size );
}

int main( int argc, char** argv )
{
    return sweep_main( argc, argv, "EpsDivDivKerngen storage/arithmetic breakdown for each narrow precision.", []( const SweepParameters& p ) {
        run_precision_matrix( p.min_level, p.max_level, p.executions, p.refinement_level_subdomains );
    } );
}
