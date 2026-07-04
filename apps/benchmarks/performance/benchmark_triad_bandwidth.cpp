// Triad bandwidth benchmark: a[i] = b[i] + s * c[i]
//
// Measures achievable GPU bandwidth for two data layouts:
//   (1) raw device pointers (Kokkos::kokkos_malloc) accessed via T*
//   (2) Kokkos::View<T*>
//
// Two working-set regimes:
//   - "main memory": arrays sized to exceed the L2 cache -> streams from HBM
//   - "L2":          arrays sized to fit in L2           -> streams from L2
//                    (first iter warms L2; steady-state iters are L2-resident)
//
// Output is averaged GB/s, where bytes/element = 3*sizeof(T) (2 reads + 1 write).

#include <cstdio>
#include <string>
#include <vector>

#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/info.hpp"

namespace {

using Scalar = double;

constexpr size_t BYTES_PER_ELEM = 3 * sizeof( Scalar ); // 2 loads + 1 store

struct Result
{
    std::string label;
    size_t      bytes_per_array;
    int         iters;
    double      seconds_total;
    double      gb_per_s;
};

// ----- Kernels --------------------------------------------------------------

template < typename T >
struct RawTriad
{
    T*          a;
    const T*    b;
    const T*    c;
    T           s;
    KOKKOS_INLINE_FUNCTION void operator()( const size_t i ) const { a[i] = b[i] + s * c[i]; }
};

template < typename ViewT >
struct ViewTriad
{
    ViewT                  a;
    typename ViewT::const_type b;
    typename ViewT::const_type c;
    typename ViewT::value_type s;
    KOKKOS_INLINE_FUNCTION void operator()( const size_t i ) const { a( i ) = b( i ) + s * c( i ); }
};

// ----- Drivers --------------------------------------------------------------

template < typename T >
double run_raw( size_t N, int iters )
{
    T* a = static_cast< T* >( Kokkos::kokkos_malloc< Kokkos::DefaultExecutionSpace >( "a", N * sizeof( T ) ) );
    T* b = static_cast< T* >( Kokkos::kokkos_malloc< Kokkos::DefaultExecutionSpace >( "b", N * sizeof( T ) ) );
    T* c = static_cast< T* >( Kokkos::kokkos_malloc< Kokkos::DefaultExecutionSpace >( "c", N * sizeof( T ) ) );

    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy< size_t >( 0, N ), KOKKOS_LAMBDA( const size_t i ) {
            a[i] = T( 0 );
            b[i] = T( 1 );
            c[i] = T( 2 );
        } );

    RawTriad< T > kernel{ a, b, c, T( 3.14 ) };

    // warm-up
    Kokkos::parallel_for( "triad_raw_warm", Kokkos::RangePolicy< size_t >( 0, N ), kernel );
    Kokkos::fence();

    Kokkos::Timer timer;
    for ( int it = 0; it < iters; ++it )
    {
        Kokkos::parallel_for( "triad_raw", Kokkos::RangePolicy< size_t >( 0, N ), kernel );
    }
    Kokkos::fence();
    const double sec = timer.seconds();

    Kokkos::kokkos_free< Kokkos::DefaultExecutionSpace >( a );
    Kokkos::kokkos_free< Kokkos::DefaultExecutionSpace >( b );
    Kokkos::kokkos_free< Kokkos::DefaultExecutionSpace >( c );

    return sec;
}

template < typename T >
double run_view( size_t N, int iters )
{
    Kokkos::View< T* > a( Kokkos::view_alloc( Kokkos::WithoutInitializing, "a" ), N );
    Kokkos::View< T* > b( Kokkos::view_alloc( Kokkos::WithoutInitializing, "b" ), N );
    Kokkos::View< T* > c( Kokkos::view_alloc( Kokkos::WithoutInitializing, "c" ), N );

    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy< size_t >( 0, N ), KOKKOS_LAMBDA( const size_t i ) {
            a( i ) = T( 0 );
            b( i ) = T( 1 );
            c( i ) = T( 2 );
        } );

    ViewTriad< Kokkos::View< T* > > kernel{ a, b, c, T( 3.14 ) };

    // warm-up
    Kokkos::parallel_for( "triad_view_warm", Kokkos::RangePolicy< size_t >( 0, N ), kernel );
    Kokkos::fence();

    Kokkos::Timer timer;
    for ( int it = 0; it < iters; ++it )
    {
        Kokkos::parallel_for( "triad_view", Kokkos::RangePolicy< size_t >( 0, N ), kernel );
    }
    Kokkos::fence();
    return timer.seconds();
}

Result make_result( const std::string& label, size_t N, int iters, double sec )
{
    Result r;
    r.label           = label;
    r.bytes_per_array = N * sizeof( Scalar );
    r.iters           = iters;
    r.seconds_total   = sec;
    const double bytes_total = static_cast< double >( iters ) * static_cast< double >( N ) * BYTES_PER_ELEM;
    r.gb_per_s               = ( bytes_total / sec ) / 1.0e9;
    return r;
}

void print_row( const Result& r )
{
    std::printf( "  %-40s  size/array=%7.1f MB  iters=%4d  time=%7.3f ms  BW=%7.1f GB/s\n",
                 r.label.c_str(),
                 r.bytes_per_array / 1.0e6,
                 r.iters,
                 r.seconds_total * 1.0e3,
                 r.gb_per_s );
}

} // namespace

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope( argc, argv );
    terra::util::print_general_info( argc, argv );

    // --- sizing ---
    // H100 has a 50 MB L2. Pick:
    //   L2-resident  : 3 arrays x 14 MB = 42 MB total (fits in L2, saturates SMs)
    //   HBM-streaming: 3 arrays x 512 MB = 1.5 GB    (vastly exceeds L2)
    const size_t N_l2  = ( 14ull * 1024 * 1024 ) / sizeof( Scalar );  //  14 MB/array
    const size_t N_hbm = ( 512ull * 1024 * 1024 ) / sizeof( Scalar ); // 512 MB/array

    const int iters_l2  = 2000; // amortize launch overhead; data stays in L2
    const int iters_hbm = 20;

    std::printf( "\nTriad bandwidth benchmark (a = b + s*c, Scalar=%s)\n", "double" );
    std::printf( "Execution space: %s\n\n", Kokkos::DefaultExecutionSpace::name() );

    std::vector< Result > results;

    // L2-resident regime
    {
        const double sec_raw  = run_raw< Scalar >( N_l2, iters_l2 );
        const double sec_view = run_view< Scalar >( N_l2, iters_l2 );
        results.push_back( make_result( "L2  | raw pointer", N_l2, iters_l2, sec_raw ) );
        results.push_back( make_result( "L2  | Kokkos::View", N_l2, iters_l2, sec_view ) );
    }

    // HBM-streaming regime
    {
        const double sec_raw  = run_raw< Scalar >( N_hbm, iters_hbm );
        const double sec_view = run_view< Scalar >( N_hbm, iters_hbm );
        results.push_back( make_result( "HBM | raw pointer", N_hbm, iters_hbm, sec_raw ) );
        results.push_back( make_result( "HBM | Kokkos::View", N_hbm, iters_hbm, sec_view ) );
    }

    std::printf( "\nResults:\n" );
    for ( const auto& r : results )
    {
        print_row( r );
    }
    std::printf( "\n" );

    MPI_Finalize();
    return 0;
}
