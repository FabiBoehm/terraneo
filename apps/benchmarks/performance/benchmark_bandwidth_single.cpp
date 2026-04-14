/**
 * Single-size bandwidth benchmark for profiling with rocprof.
 * Usage: ./benchmark_bandwidth_single --size-kb <KB>
 * Runs STREAM triad with raw HIP at the given size.
 */
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK( call )                                                         \
    do                                                                            \
    {                                                                             \
        hipError_t err = call;                                                    \
        if ( err != hipSuccess )                                                  \
        {                                                                         \
            std::cerr << "HIP error: " << hipGetErrorString( err ) << " at "      \
                      << __FILE__ << ":" << __LINE__ << "\n";                     \
            std::exit( 1 );                                                       \
        }                                                                         \
    } while ( 0 )

__global__ void triad( double* __restrict__ A,
                       const double* __restrict__ B,
                       const double* __restrict__ C,
                       const double scalar,
                       const size_t N )
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N )
        A[i] = B[i] + scalar * C[i];
}

int main( int argc, char* argv[] )
{
    size_t size_kb   = 1024;  // default 1 MB
    int    warmup    = 20;
    int    iters     = 100;

    for ( int a = 1; a < argc; ++a )
    {
        if ( std::strcmp( argv[a], "--size-kb" ) == 0 && a + 1 < argc )
            size_kb = std::atol( argv[++a] );
        else if ( std::strcmp( argv[a], "--iters" ) == 0 && a + 1 < argc )
            iters = std::atoi( argv[++a] );
    }

    const size_t total_bytes = size_kb * 1024;
    const size_t N           = total_bytes / sizeof( double );
    const int block_size     = 256;
    const int grid           = ( N + block_size - 1 ) / block_size;
    const double scalar      = 3.14159;

    std::cout << "Size: " << size_kb << " KB (" << total_bytes / ( 1024.0 * 1024.0 )
              << " MB), N=" << N << ", grid=" << grid << ", iters=" << iters << "\n";

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    HIP_CHECK( hipMalloc( &d_A, N * sizeof( double ) ) );
    HIP_CHECK( hipMalloc( &d_B, N * sizeof( double ) ) );
    HIP_CHECK( hipMalloc( &d_C, N * sizeof( double ) ) );
    HIP_CHECK( hipMemset( d_A, 0, N * sizeof( double ) ) );
    HIP_CHECK( hipMemset( d_B, 1, N * sizeof( double ) ) );
    HIP_CHECK( hipMemset( d_C, 2, N * sizeof( double ) ) );
    HIP_CHECK( hipDeviceSynchronize() );

    // Warmup
    for ( int w = 0; w < warmup; ++w )
        hipLaunchKernelGGL( triad, dim3( grid ), dim3( block_size ), 0, 0, d_A, d_B, d_C, scalar, N );
    HIP_CHECK( hipDeviceSynchronize() );

    // Timed (also profiled by rocprof)
    hipEvent_t start, stop;
    HIP_CHECK( hipEventCreate( &start ) );
    HIP_CHECK( hipEventCreate( &stop ) );
    HIP_CHECK( hipEventRecord( start ) );
    for ( int it = 0; it < iters; ++it )
        hipLaunchKernelGGL( triad, dim3( grid ), dim3( block_size ), 0, 0, d_A, d_B, d_C, scalar, N );
    HIP_CHECK( hipEventRecord( stop ) );
    HIP_CHECK( hipEventSynchronize( stop ) );

    float ms = 0;
    HIP_CHECK( hipEventElapsedTime( &ms, start, stop ) );
    double elapsed     = ( ms / 1000.0 ) / iters;
    double bytes_moved = 3.0 * N * sizeof( double );
    double bw_gbs      = bytes_moved / elapsed / 1e9;

    std::cout << "Time/iter: " << elapsed * 1e6 << " us, BW (naive): " << bw_gbs << " GB/s\n";

    HIP_CHECK( hipEventDestroy( start ) );
    HIP_CHECK( hipEventDestroy( stop ) );
    HIP_CHECK( hipFree( d_A ) );
    HIP_CHECK( hipFree( d_B ) );
    HIP_CHECK( hipFree( d_C ) );
    return 0;
}
