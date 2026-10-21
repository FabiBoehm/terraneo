// EpsDivDivKerngen: stored local matrices (double/float/bf16/fp16) against the matrix-free apply.

#include "benchmark_epsdivdiv_common.hpp"

/// One stored-local-matrix arm: fields in S, matrices in MatT. The precompute is
/// timed separately from the apply; a level whose matrices do not fit the device
/// is reported as -1 like an unlaunchable tile.
struct StoredBenchmarkData
{
    BenchmarkData data;
    double        precompute_s = -1.0;
    double        matrix_bytes = 0.0;
};

template < typename S, typename MatT >
StoredBenchmarkData run_stored( const int level, const int executions, const int refinement_level_subdomains )
{
    const auto domain = grid::shell::DistributedDomain::create_uniform(
        level, level, 0.5, 1.0, refinement_level_subdomains, refinement_level_subdomains );

    using G = DefaultGeom< S >;
    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< G >( domain );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto dofs_scalar = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    const auto dofs_vec    = 3 * dofs_scalar;

    VectorQ1Vec< S >    src( "src_stored", domain, mask_data );
    VectorQ1Vec< S >    dst( "dst_stored", domain, mask_data );
    VectorQ1Scalar< S > coeff( "coeff_stored", domain, mask_data );

    linalg::assign( coeff, util::narrow< S >( 1.0 ) );
    linalg::randomize( src );

    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };

    // Storage S, arithmetic S, geometry G, matrices MatT.
    EpsilonDivDivKerngen< S, 3, G, S, MatT > op(
        domain, coords_shell, coords_radii, boundary_mask_data, coeff.grid_data(), bcs, false );
    op.set_stored_local_matrices( true );

    StoredBenchmarkData out;
    out.data         = BenchmarkData{ level, dofs_vec, -1.0 };
    out.matrix_bytes = static_cast< double >( op.stored_local_matrices_bytes() );
    try
    {
        Kokkos::Timer timer;
        op.precompute_stored_local_matrices();
        out.precompute_s  = timer.seconds();
        out.data.duration = measure_run_time( executions, op, src, dst );
    }
    catch ( const std::exception& e )
    {
        logroot << "[run_stored] level " << level << ": " << e.what() << std::endl;
    }
    return out;
}

template < typename S, typename MatT >
void sweep_one_stored( const std::string& name,
                       const int          min_level,
                       const int          max_level,
                       const int          executions,
                       const int          refinement_level_subdomains,
                       const int          world_size )
{
    logroot << "EpsDivDivKerngen stored local matrices: " << name << " (fields " << sizeof( S ) << " B, matrices "
            << sizeof( MatT ) << " B)" << std::endl;

    util::Table table;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const auto   r     = run_stored< S, MatT >( level, executions, refinement_level_subdomains );
        const bool   ok    = r.data.duration > 0.0;
        const double gdofs = ok ? r.data.dofs / r.data.duration / 1e9 : -1.0;
        // Bandwidth on the matrices alone: what the apply moves that the matrix-free path does not.
        const double mgbs = ok ? r.matrix_bytes / r.data.duration / 1e9 : -1.0;

        table.add_row( { { "precision", name },
                         { "bytes", static_cast< long >( sizeof( S ) ) },
                         { "matrix_bytes", static_cast< long >( sizeof( MatT ) ) },
                         { "level", level },
                         { "dofs", r.data.dofs },
                         { "duration (s)", r.data.duration },
                         { "GDoF/s", gdofs },
                         { "matrix GB", r.matrix_bytes / 1e9 },
                         { "matrix GB/s", mgbs },
                         { "precompute (s)", r.precompute_s } } );
    }

    table.print_pretty();

    if ( mpi::rank() == 0 )
    {
        std::ofstream out( "./csv/stored_" + name + "_np" + std::to_string( world_size ) + "_ml" +
                           std::to_string( max_level ) + ".csv" );
        table.print_csv( out );
    }
    table.print_csv( logroot );
    logroot << std::endl << std::endl;
}

/// The stored-local-matrix path next to the matrix-free one, across matrix precisions.
void run_stored_sweep( const int min_level,
                       const int max_level,
                       const int executions,
                       const int refinement_level_subdomains )
{
    using half  = Kokkos::Experimental::half_t;
    using bhalf = Kokkos::Experimental::bhalf_t;

    logroot << "EpsDivDivKerngen: stored local matrices vs matrix-free." << std::endl;
    logroot << "min_level: " << min_level << ", max_level: " << max_level
            << ", executions per level: " << executions << std::endl
            << std::endl;

    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    const int mn = min_level, mx = max_level, ex = executions, r = refinement_level_subdomains;

    sweep_one_precision< double >( "mf_double", mn, mx, ex, r, world_size );
    sweep_one_precision< float >( "mf_float", mn, mx, ex, r, world_size );
    sweep_one_stored< double, double >( "d_m8", mn, mx, ex, r, world_size );
    sweep_one_stored< double, float >( "d_m4", mn, mx, ex, r, world_size );
    sweep_one_stored< double, bhalf >( "d_m2bf", mn, mx, ex, r, world_size );
    sweep_one_stored< double, half >( "d_m2h", mn, mx, ex, r, world_size );
    sweep_one_stored< float, float >( "f_m4", mn, mx, ex, r, world_size );
    sweep_one_stored< float, half >( "f_m2h", mn, mx, ex, r, world_size );
}

int main( int argc, char** argv )
{
    return sweep_main( argc, argv, "EpsDivDivKerngen: stored local matrices (double/float/bf16/fp16) against the matrix-free apply.", []( const SweepParameters& p ) {
        run_stored_sweep( p.min_level, p.max_level, p.executions, p.refinement_level_subdomains );
    } );
}
