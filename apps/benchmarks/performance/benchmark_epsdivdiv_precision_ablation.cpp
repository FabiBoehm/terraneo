// EpsDivDivKerngen all-low precision, ablated to double one component at a time (S, A, G).

#include "benchmark_epsdivdiv_common.hpp"

/// @brief Everything in low precision, then ablated back to double one component at
///        a time.
///
/// The three independent precisions are the stored fields (S), the fast-path
/// contraction (A) and the geometry (G: coordinates, radii and everything the
/// Jacobian is built from). The all-low arm is the only one that answers "what does
/// this cost if nothing is held back"; each single-component flip to double then says
/// what that one component was worth. Reported for both 16-bit formats, since their
/// conversion costs differ sharply on hardware without a native bf16 convert.
template < typename H >
void run_ablation_for( const char* tag,
                       const int   mn,
                       const int   mx,
                       const int   ex,
                       const int   r,
                       const int   world_size )
{
    const auto nm = [tag]( const char* suffix ) { return std::string( tag ) + suffix; };

    //                        S       A       G
    sweep_one_precision< H, H, H >( nm( "_all_low" ), mn, mx, ex, r, world_size );
    sweep_one_precision< double, H, H >( nm( "_store_dbl" ), mn, mx, ex, r, world_size );
    sweep_one_precision< H, double, H >( nm( "_arith_dbl" ), mn, mx, ex, r, world_size );
    sweep_one_precision< H, H, double >( nm( "_geom_dbl" ), mn, mx, ex, r, world_size );
    sweep_one_precision< H, H, float >( nm( "_geom_flt" ), mn, mx, ex, r, world_size );
    sweep_one_precision< H, double, double >( nm( "_only_store_low" ), mn, mx, ex, r, world_size );
    sweep_one_precision< double, H, double >( nm( "_only_arith_low" ), mn, mx, ex, r, world_size );
    sweep_one_precision< double, double, H >( nm( "_only_geom_low" ), mn, mx, ex, r, world_size );
}

void run_precision_ablation( const int min_level,
                             const int max_level,
                             const int executions,
                             const int refinement_level_subdomains )
{
    using half  = Kokkos::Experimental::half_t;
    using bhalf = Kokkos::Experimental::bhalf_t;

    logroot << "All-low-precision EpsDivDivKerngen, ablated to double component by "
               "component (S = storage, A = arithmetic, G = geometry)."
            << std::endl;
    logroot << "min_level: " << min_level << ", max_level: " << max_level
            << ", executions per level: " << executions << std::endl
            << std::endl;

    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    const int mn = min_level, mx = max_level, ex = executions, r = refinement_level_subdomains;

    sweep_one_precision< double, double, double >( "all_double", mn, mx, ex, r, world_size );
    sweep_one_precision< float, float, float >( "float_all_low", mn, mx, ex, r, world_size );
    sweep_one_precision< float, float, double >( "float_geom_dbl", mn, mx, ex, r, world_size );
    run_ablation_for< half >( "fp16", mn, mx, ex, r, world_size );
    run_ablation_for< bhalf >( "bf16", mn, mx, ex, r, world_size );
}

int main( int argc, char** argv )
{
    return sweep_main( argc, argv, "EpsDivDivKerngen all-low precision, ablated to double one component at a time (S, A, G).", []( const SweepParameters& p ) {
        run_precision_ablation( p.min_level, p.max_level, p.executions, p.refinement_level_subdomains );
    } );
}
