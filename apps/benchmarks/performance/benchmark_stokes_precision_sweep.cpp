// Throughput of the Stokes operator and its blocks -- EpsDivDiv, Gradient, Divergence
// and the full EpsDivDivStokes apply -- over double/float/bf16/fp16 per level. Same
// harness and columns as the EpsDivDiv precision sweep; the DoF rate of every row is
// in fine velocity DoFs per second, so the rows compare directly.

#include "benchmark_epsdivdiv_common.hpp"

#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"

using fe::wedge::operators::shell::EpsDivDivStokes;

template < typename S, typename G = DefaultGeom< S > >
void sweep_stokes_precision( const std::string& name, const int min_level, const int max_level, const int executions,
                             const int refinement_level_subdomains, const int world_size )
{
    logroot << "Stokes precision sweep: " << name << " (storage " << sizeof( S ) << " B, geometry " << sizeof( G )
            << " B)" << std::endl;

    util::Table table;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const auto dom_f = grid::shell::DistributedDomain::create_uniform(
            level, level, 0.5, 1.0, refinement_level_subdomains, refinement_level_subdomains );
        const auto dom_c = grid::shell::DistributedDomain::create_uniform(
            level - 1, level - 1, 0.5, 1.0, refinement_level_subdomains, refinement_level_subdomains );

        const auto coords_f = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( dom_f );
        const auto radii_f  = grid::shell::subdomain_shell_radii< G >( dom_f );
        auto       mask_f   = grid::setup_node_ownership_mask_data( dom_f );
        auto       mask_c   = grid::setup_node_ownership_mask_data( dom_c );
        auto       bmask_f  = grid::shell::setup_boundary_mask_data( dom_f );

        const auto dofs_scalar = kernels::common::count_masked< long >( mask_f, grid::NodeOwnershipFlag::OWNED );
        const auto dofs_vec    = 3 * dofs_scalar;

        VectorQ1Scalar< S > coeff( "coeff", dom_f, mask_f );
        linalg::assign( coeff, util::narrow< S >( 1.0 ) );

        BoundaryConditions bcs = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } };

        using Stokes = EpsDivDivStokes< S, 3, G, S >;
        Stokes K( dom_f, dom_c, coords_f, radii_f, bmask_f, coeff.grid_data(), bcs, false );

        linalg::VectorQ1IsoQ2Q1< S > x( "x", dom_f, dom_c, mask_f, mask_c );
        linalg::VectorQ1IsoQ2Q1< S > y( "y", dom_f, dom_c, mask_f, mask_c );
        linalg::randomize( x );

        auto row = [&]( const char* op, double duration ) {
            const bool   ok    = duration > 0.0;
            const double gdofs = ok ? dofs_vec / duration / 1e9 : -1.0;
            table.add_row( { { "precision", name },
                             { "operator", std::string( op ) },
                             { "bytes", static_cast< long >( sizeof( S ) ) },
                             { "geom_bytes", static_cast< long >( sizeof( G ) ) },
                             { "level", level },
                             { "dofs", dofs_vec },
                             { "duration (s)", duration },
                             { "GDoF/s", gdofs } } );
            logroot << "  level " << level << "  " << op << "  " << duration << " s  " << gdofs << " GDoF/s"
                    << std::endl;
        };

        // Warm-up (first launches, comm plan).
        apply( K, x, y );
        Kokkos::fence();

        row( "epsdivdiv", measure_run_time( executions, K.block_11(), x.block_1(), y.block_1() ) );
        row( "gradient", measure_run_time( executions, K.block_12(), x.block_2(), y.block_1() ) );
        row( "divergence", measure_run_time( executions, K.block_21(), x.block_1(), y.block_2() ) );
        row( "stokes", measure_run_time( executions, K, x, y ) );

        // The energy operator (mass + advection + diffusion + SUPG) on the same fine
        // grid: a scalar field advected by the velocity block of x. Its DoF rate is
        // reported per velocity DoF like the others; divide by 3 for scalar DoFs.
        using Energy = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< S, 3, G, S >;
        Energy energy( dom_f, coords_f, radii_f, bmask_f, x.block_1(), util::narrow< S >( 1e-2 ),
                       util::narrow< S >( 1e-3 ), true, false, util::narrow< S >( 1.0 ), false );
        VectorQ1Scalar< S > T( "T", dom_f, mask_f ), yT( "yT", dom_f, mask_f );
        linalg::randomize( T );
        apply( energy, T, yT );
        Kokkos::fence();
        row( "energy_supg", measure_run_time( executions, energy, T, yT ) );
    }

    table.print_pretty();
    if ( mpi::rank() == 0 )
    {
        std::ofstream out( "./csv/stokes_" + name + "_np" + std::to_string( world_size ) + "_ml" +
                           std::to_string( max_level ) + ".csv" );
        table.print_csv( out );
    }
    table.print_csv( logroot );
    logroot << std::endl << std::endl;
}

int main( int argc, char** argv )
{
    return sweep_main( argc, argv, "Stokes operator throughput over double/float/bf16/fp16 per level.", []( const SweepParameters& p ) {
        int world_size = 0;
        MPI_Comm_size( MPI_COMM_WORLD, &world_size );
        const int min_level = std::max( p.min_level, 2 );
        sweep_stokes_precision< double >( "double", min_level, p.max_level, p.executions, p.refinement_level_subdomains, world_size );
        sweep_stokes_precision< float >( "float", min_level, p.max_level, p.executions, p.refinement_level_subdomains, world_size );
        sweep_stokes_precision< Kokkos::Experimental::bhalf_t >( "bf16", min_level, p.max_level, p.executions, p.refinement_level_subdomains, world_size );
        sweep_stokes_precision< Kokkos::Experimental::half_t >( "fp16", min_level, p.max_level, p.executions, p.refinement_level_subdomains, world_size );
    } );
}
