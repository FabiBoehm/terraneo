#pragma once

// Shared pieces of the EpsDivDivKerngen throughput benchmarks: the timing loop,
// one precision arm, and the main() every sweep executable runs through. Each sweep
// is its own executable so that it instantiates only its own operator types -- the
// single benchmark binary that held them all took 17 minutes to compile, one
// translation unit nvc++ cannot parallelise.

#include <kernels/common/grid_operations.hpp>

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/cli11_helper.hpp"
#include "util/floating_point.hpp"
#include "util/info.hpp"
#include "util/table.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

using namespace terra;

using fe::wedge::operators::shell::EpsilonDivDivKerngen;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::apply;
using linalg::DstOf;
using linalg::OperatorLike;
using linalg::SrcOf;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using terra::grid::shell::BoundaryConditions;
using util::logroot;

struct BenchmarkData
{
    int    level;
    long   dofs;
    double duration;
};

template < OperatorLike OperatorT >
double measure_run_time( int executions, OperatorT& A, const SrcOf< OperatorT >& src, DstOf< OperatorT >& dst )
{
    Kokkos::Timer timer;

    Kokkos::fence();
    MPI_Barrier( MPI_COMM_WORLD );
    timer.reset();

    for ( int i = 0; i < executions; ++i )
    {
        apply( A, src, dst );
    }

    Kokkos::fence();

    // Ensure stuff is not optimized out?!
    // const auto mm = kernels::common::max_abs_entry( dst.grid_data() );
    // std::cout << "Printing some derived value to ensure nothing is optimized out: " << mm << std::endl;
    MPI_Barrier( MPI_COMM_WORLD );
    double duration     = timer.seconds() / executions;
    double duration_max = 0.0;
    MPI_Allreduce( &duration, &duration_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return duration_max;
}

// =============================================================================
// Precision sweep over the EpsDivDivKerngen fast path.
// =============================================================================
//
// The enum-driven benchmarks above pin each arm to one scalar type, with the
// field vectors built up front in run(). A 16-bit arm cannot reuse that: the
// storage type decides the vectors, the coefficient and the operator's own
// arithmetic, so each precision gets its own domain and its own fields here.
//
// The configuration deliberately mirrors the mixed-precision A-block study
// (--ablock-prec M --ablock-arith low): geometry stays in double while the
// stored fields AND the fast-path contraction follow M. That is
// EpsilonDivDivKerngen< M, 3, double, M >.
//
// What this measures is throughput of one matvec, so it answers the question the
// end-to-end solve timings could not: whether the operator is bandwidth-bound at
// a given level. If it is, halving the bytes per node should show up as nearly
// double the DoF rate; if the kernel is launch- or latency-bound, the precisions
// collapse onto one another.

/// Traffic model for one application: read src (3 components), write dst (3),
/// read the coefficient (1), all in M. Geometry is double and largely cached
/// across the tile, so it is excluded rather than guessed at.
template < typename M >
double modelled_bytes( long dofs_scalar )
{
    return static_cast< double >( dofs_scalar ) * ( 3.0 + 3.0 + 1.0 ) * sizeof( M );
}

/// @tparam S storage precision of the fields (src, dst, coefficient)
/// @tparam A arithmetic precision of the fast-path contraction
/// @tparam G geometry precision: the stored coordinates and radii, i.e. CoordScalarT
///
/// Keeping these separate is what attributes the cost: S alone changes the bytes
/// moved and the load/store path, A alone changes the arithmetic. Kokkos' 16-bit
/// scalar operators convert to float per operation, so an A = half arm pays a
/// conversion on every multiply-add whether or not the storage is narrow.
/// Default geometry precision: the storage precision, widened to float. A 16-bit
/// geometry would put the coordinate differencing at three decimal digits.
template < typename S >
using DefaultGeom = std::conditional_t< ( sizeof( S ) >= 4 ), S, float >;

/// Tile override for the sweeps that do not sweep tiles themselves; zero keeps the
/// operator's own default. Set once from the command line.
struct TileOverride
{
    int lat    = 0;
    int r      = 0;
    int passes = 0;
};
inline TileOverride tile_override{};

/// -1 keeps each precision's default (on for 16-bit storage), 0/1 force the DN tile
/// scatter off/on. Set once from the command line.
inline int tile_scatter_override = -1;

template < typename S, typename A = S, typename G = DefaultGeom< S > >
BenchmarkData run_precision( const int level,
                             const int executions,
                             const int refinement_level_subdomains,
                             const int tile_lat    = 0,
                             const int tile_r      = 0,
                             const int tile_passes = 0 )
{
    const auto domain = grid::shell::DistributedDomain::create_uniform(
        level, level, 0.5, 1.0, refinement_level_subdomains, refinement_level_subdomains );

    // Geometry in G. The solver runs this in double; G = S is what "everything in
    // low precision" means, and flipping G back to double on its own is what
    // separates the geometry's cost from the field's.
    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< G >( domain );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto dofs_scalar = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    const auto dofs_vec    = 3 * dofs_scalar;

    VectorQ1Vec< S >    src( "src_prec", domain, mask_data );
    VectorQ1Vec< S >    dst( "dst_prec", domain, mask_data );
    VectorQ1Scalar< S > coeff( "coeff_prec", domain, mask_data );

    linalg::assign( coeff, util::narrow< S >( 1.0 ) );
    linalg::randomize( src );

    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };

    // Storage S, fast-path arithmetic A, geometry G.
    EpsilonDivDivKerngen< S, 3, G, A > op(
        domain, coords_shell, coords_radii, boundary_mask_data, coeff.grid_data(), bcs, false );

    if ( tile_lat > 0 )
    {
        // A rejected tiling leaves the operator at its default and reports -1, so the
        // sweep can tell "not launchable here" apart from "slow".
        if ( !op.set_tiling( tile_lat, tile_r, tile_passes, false ) )
        {
            return BenchmarkData{ level, dofs_vec, -1.0 };
        }
    }
    if ( tile_scatter_override >= 0 )
    {
        // The warp-owner scatter only has lat-8, r-1 shapes; any other tile in a
        // mode-2 sweep is "not launchable" rather than silently retiled.
        if ( tile_scatter_override == 2 && !( tile_lat == 8 && tile_r == 1 ) )
            return BenchmarkData{ level, dofs_vec, -1.0 };
        op.set_tile_scatter( tile_scatter_override );
    }
    logroot << "[run_precision] tile scatter mode " << op.tile_scatter() << std::endl;

    // set_tiling() checks the team size against the launch bounds but not the
    // scratch against the device: a wide-arithmetic arm on a tall tile can stage
    // more than a block may hold, which Kokkos reports as a zero maximum team size
    // at launch. That is "not launchable here" too, so it gets the same -1.
    double duration = -1.0;
    try
    {
        duration = measure_run_time( executions, op, src, dst );
    }
    catch ( const std::runtime_error& e )
    {
        logroot << "[run_precision] tile " << tile_lat << "/" << tile_r << "/" << tile_passes
                << " not launchable for this arm: " << e.what() << std::endl;
    }

    return BenchmarkData{ level, dofs_vec, duration };
}

template < typename S, typename A = S, typename G = DefaultGeom< S > >
void sweep_one_precision( const std::string& name,
                          const int   min_level,
                          const int   max_level,
                          const int   executions,
                          const int   refinement_level_subdomains,
                          const int   world_size )
{
    logroot << "EpsDivDivKerngen precision sweep: " << name << " (storage " << sizeof( S )
            << " B, arithmetic " << sizeof( A ) << " B, geometry " << sizeof( G ) << " B)" << std::endl;

    util::Table table;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const auto data = run_precision< S, A, G >(
            level, executions, refinement_level_subdomains, tile_override.lat, tile_override.r, tile_override.passes );
        const bool   ok    = data.duration > 0.0;
        const double gdofs = ok ? data.dofs / data.duration / 1e9 : -1.0;
        const double gbs   = ok ? modelled_bytes< S >( data.dofs / 3 ) / data.duration / 1e9 : -1.0;

        table.add_row( { { "precision", name },
                         { "bytes", static_cast< long >( sizeof( S ) ) },
                         { "arith_bytes", static_cast< long >( sizeof( A ) ) },
                         { "geom_bytes", static_cast< long >( sizeof( G ) ) },
                         { "level", level },
                         { "dofs", data.dofs },
                         { "duration (s)", data.duration },
                         { "GDoF/s", gdofs },
                         { "modelled GB/s", gbs } } );
    }

    table.print_pretty();

    if ( mpi::rank() == 0 )
    {
        std::ofstream out( "./csv/prec_" + name + "_np" + std::to_string( world_size ) + "_ml" +
                           std::to_string( max_level ) + ".csv" );
        table.print_csv( out );
    }
    table.print_csv( logroot );
    logroot << std::endl << std::endl;
}

/// What every sweep executable takes from the command line.
struct SweepParameters
{
    int min_level                   = 1;
    int max_level                   = 6;
    int executions                  = 5;
    int refinement_level_subdomains = 0;
    int tile_lat                    = 0;
    int tile_r                      = 0;
    int tile_passes                 = 0;
    int tile_scatter                = -1;
};

/// MPI and Kokkos setup, the common options, the tile override, then `run( p )`.
template < typename Run >
int sweep_main( int argc, char** argv, const char* description, Run&& run )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    util::print_general_info( argc, argv );

    CLI::App        app{ description };
    SweepParameters p{};

    util::add_option_with_default( app, "--min-level", p.min_level, "Min refinement level." );
    util::add_option_with_default( app, "--max-level", p.max_level, "Max refinement level." );
    util::add_option_with_default(
        app, "--refinement-level-subdomains", p.refinement_level_subdomains,
        "Refinement level applied to form the subdomains." );
    util::add_option_with_default(
        app, "--executions", p.executions, "Number of matrix-vector multiplications to be executed." );
    util::add_option_with_default( app, "--tile-lat", p.tile_lat, "Lateral tile size (0: operator default)." );
    util::add_option_with_default( app, "--tile-r", p.tile_r, "Radial tile size (0: operator default)." );
    util::add_option_with_default( app, "--tile-passes", p.tile_passes, "Radial passes per team (0: operator default)." );
    util::add_option_with_default(
        app, "--tile-scatter", p.tile_scatter, "DN scatter: -1 per-precision default, 0 straight to global, 1 tile buffer, 2 warp-owner (lat 8, r 1)." );

    CLI11_PARSE( app, argc, argv );

    tile_override = { p.tile_lat, p.tile_r, p.tile_passes };
    tile_scatter_override = p.tile_scatter;

    if ( p.min_level < 1 )
    {
        logroot << "Error: min-level must be >= 1." << std::endl;
        return 1;
    }

    logroot << "\n" << description << "\n\n";
    util::print_cli_summary( app, logroot );
    logroot << "\n\n";

    run( p );

    MPI_Finalize();
    return 0;
}
