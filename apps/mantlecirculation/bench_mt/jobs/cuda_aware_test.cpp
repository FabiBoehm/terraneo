// Minimal CUDA-aware MPI check: rank 0 sends a *device* buffer to rank 1.
// If MPI is GPU-aware at runtime the recv'd data is correct; otherwise it
// crashes or yields garbage. Run on 2 nodes (1 rank each) with UCX_PROTO_INFO=y
// to see whether the transfer uses cuda/gdr transports or stages through host.
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    int rank = 0, size = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    int ndev = 0;
    cudaGetDeviceCount( &ndev );
    const char* lr = getenv( "OMPI_COMM_WORLD_LOCAL_RANK" );
    int local = lr ? atoi( lr ) : 0;
    cudaSetDevice( ndev > 0 ? local % ndev : 0 );

    char host[256];
    gethostname( host, sizeof( host ) );

    const int n = 1 << 20;            // 1M doubles = 8 MB (rendezvous range)
    double*   d = nullptr;
    if ( cudaMalloc( &d, n * sizeof( double ) ) != cudaSuccess ) {
        printf( "[rank %d %s] cudaMalloc FAILED\n", rank, host ); MPI_Abort( MPI_COMM_WORLD, 1 );
    }

    if ( rank == 0 ) {
        double* h = (double*) malloc( n * sizeof( double ) );
        for ( int i = 0; i < n; ++i ) h[i] = 3.14159;
        cudaMemcpy( d, h, n * sizeof( double ), cudaMemcpyHostToDevice );
        free( h );
        printf( "[rank 0 %s] sending DEVICE pointer %p (%d devices)\n", host, (void*) d, ndev );
        fflush( stdout );
        MPI_Send( d, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD );
        printf( "[rank 0 %s] send returned OK\n", host );
    } else if ( rank == 1 ) {
        MPI_Recv( d, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        double* h = (double*) malloc( n * sizeof( double ) );
        cudaMemcpy( h, d, n * sizeof( double ), cudaMemcpyDeviceToHost );
        int ok = ( h[0] == 3.14159 && h[n - 1] == 3.14159 );
        printf( "[rank 1 %s] recv into DEVICE pointer %p -> h[0]=%f h[last]=%f  ==> %s\n",
                host, (void*) d, h[0], h[n - 1],
                ok ? "GPU-AWARE MPI WORKS (device buffer transferred correctly)"
                   : "WRONG DATA (not GPU-aware / corrupted)" );
        free( h );
    }

    cudaFree( d );
    MPI_Finalize();
    return 0;
}
