// EpsDivDivKerngen throughput over double/float/bf16/fp16 per level.

#include "benchmark_epsdivdiv_common.hpp"

void run_precision_sweep( const int min_level,
                          const int max_level,
                          const int executions,
                          const int refinement_level_subdomains )
{
    logroot << "Running the EpsDivDivKerngen precision sweep." << std::endl;
    logroot << "min_level:            " << min_level << std::endl;
    logroot << "max_level:            " << max_level << std::endl;
    logroot << "executions per level: " << executions << std::endl;
    logroot << std::endl;

    // Kokkos falls back to float when a backend has no native 16-bit type, which
    // would make a 4-byte run read as 2-byte. Say so rather than let it pass.
#if defined( KOKKOS_HALF_T_IS_FLOAT ) && KOKKOS_HALF_T_IS_FLOAT
    logroot << "WARNING: half_t is float on this backend; the fp16 arm is emulated." << std::endl;
#endif
#if defined( KOKKOS_BHALF_T_IS_FLOAT ) && KOKKOS_BHALF_T_IS_FLOAT
    logroot << "WARNING: bhalf_t is float on this backend; the bf16 arm is emulated." << std::endl;
#endif

    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    sweep_one_precision< double >( "double", min_level, max_level, executions, refinement_level_subdomains, world_size );
    sweep_one_precision< float >( "float", min_level, max_level, executions, refinement_level_subdomains, world_size );
    sweep_one_precision< Kokkos::Experimental::bhalf_t >(
        "bf16", min_level, max_level, executions, refinement_level_subdomains, world_size );
    sweep_one_precision< Kokkos::Experimental::half_t >(
        "fp16", min_level, max_level, executions, refinement_level_subdomains, world_size );
}

int main( int argc, char** argv )
{
    return sweep_main( argc, argv, "EpsDivDivKerngen throughput over double/float/bf16/fp16 per level.", []( const SweepParameters& p ) {
        run_precision_sweep( p.min_level, p.max_level, p.executions, p.refinement_level_subdomains );
    } );
}
