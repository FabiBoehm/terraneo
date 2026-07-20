
// Test: pure diffusion with the explicit FCT scheme.
//
// No advection (u = 0), diffusivity = 1, homogeneous zero initial condition,
// Dirichlet BCs: T = 1 at the CMB (inner boundary), T = 0 at the surface (outer).
//
// Analytical steady state in a spherical shell (r_min, r_max):
//
//   T_ss(r) = (1/r - 1/r_max) / (1/r_min - 1/r_max)
//
// After integrating long enough the numerical solution should match this to within
// a few percent at the grid resolution used here (level 3, n_steps ~ 1000).

#include <cmath>
#include <iostream>
#include <mpi.h>

#include "communication/shell/communication.hpp"
#include "fv/hex/conversion.hpp"
#include "fv/hex/helpers.hpp"
#include "fv/hex/operators/fct_advection_diffusion.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using linalg::VectorFVScalar;
using linalg::VectorFVVec;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using ScalarType = double;

// ============================================================================
// Analytical steady state: T_ss(r) = (1/r - 1/r_max) / (1/r_min - 1/r_max)
// ============================================================================

ScalarType T_steady_state( const ScalarType r, const ScalarType r_min, const ScalarType r_max )
{
    return ( 1.0 / r - 1.0 / r_max ) / ( 1.0 / r_min - 1.0 / r_max );
}

// ============================================================================
// Relative L2 error vs. analytical steady state, cell-averaged.
// ============================================================================

ScalarType compute_error(
    const DistributedDomain&                       domain,
    const Grid4DDataScalar< ScalarType >&           T,
    const Grid4DDataVec< ScalarType, 3 >&           cell_centers,
    const ScalarType                                r_min,
    const ScalarType                                r_max )
{
    ScalarType sum_sq_diff = 0;
    ScalarType sum_sq_ref  = 0;

    Kokkos::parallel_reduce(
        "error_num",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r, ScalarType& acc ) {
            const ScalarType cx     = cell_centers( id, x, y, r, 0 );
            const ScalarType cy     = cell_centers( id, x, y, r, 1 );
            const ScalarType cz     = cell_centers( id, x, y, r, 2 );
            const ScalarType radius = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
            const ScalarType T_ref  = ( 1.0 / radius - 1.0 / r_max ) / ( 1.0 / r_min - 1.0 / r_max );
            const ScalarType diff   = T( id, x, y, r ) - T_ref;
            acc += diff * diff;
        },
        sum_sq_diff );

    Kokkos::parallel_reduce(
        "error_den",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r, ScalarType& acc ) {
            const ScalarType cx     = cell_centers( id, x, y, r, 0 );
            const ScalarType cy     = cell_centers( id, x, y, r, 1 );
            const ScalarType cz     = cell_centers( id, x, y, r, 2 );
            const ScalarType radius = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
            const ScalarType T_ref  = ( 1.0 / radius - 1.0 / r_max ) / ( 1.0 / r_min - 1.0 / r_max );
            acc += T_ref * T_ref;
        },
        sum_sq_ref );

    Kokkos::fence();

    ScalarType global_num = 0, global_den = 0;
    MPI_Allreduce( &sum_sq_diff, &global_num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( &sum_sq_ref, &global_den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    return global_den > 1e-30 ? std::sqrt( global_num / global_den ) : 0.0;
}

// ============================================================================
// Main test
// ============================================================================

void test( const int level )
{
    const ScalarType r_min = 0.5;
    const ScalarType r_max = 1.0;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    // Temperature field (FV, cell-centred).  Initialised to 0.
    VectorFVScalar< ScalarType > T( "T", domain );

    // Cell centres (precomputed once).
    VectorFVVec< ScalarType, 3 > cell_centers( "cell_centers", domain );
    fv::hex::initialize_cell_centers( cell_centers, domain, coords_shell, coords_radii );

    // Zero velocity (pure diffusion).
    VectorQ1Vec< ScalarType > u( "u", domain, mask_data );
    assign( u, ScalarType( 0 ) );

    // FCT scratch buffers.
    fv::hex::operators::FVFCTBuffers< ScalarType > bufs( domain );

    // Dirichlet BCs: T = 1 at CMB (inner), T = 0 at surface (outer).
    const fv::hex::DirichletBCs< ScalarType > bcs{
        .T_cmb         = ScalarType( 1 ),
        .T_surface     = ScalarType( 0 ),
        .apply_cmb     = true,
        .apply_surface = true,
    };

    // Apply BCs to initial condition so boundaries start at the correct values.
    // This also sets the ghost cells at the physical radial boundaries (CMB r=0,
    // surface r=N), which update_fv_ghost_layers never fills (no subdomain neighbour).
    fv::hex::apply_dirichlet_bcs( T, boundary_mask_data, bcs, domain );
    // Populate lateral ghost layers so all ghost cells are correct before step 1.
    communication::shell::update_fv_ghost_layers( domain, T.grid_data() );

    // Diffusion CFL for explicit scheme on a 3D non-orthogonal mesh:
    //   dt <= h^2 / (6*kappa)  — orthogonal 3D stability limit.
    // Diamond-grid cells near pentagon corners are non-orthogonal, further tightening
    // the constraint.  Use a conservative factor of 0.05 (about 3x below the orthogonal
    // 3D limit) to remain stable across all cells.
    const ScalarType kappa = 1.0;
    const ScalarType h_min = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt    = 0.05 * h_min * h_min / kappa;

    // Run long enough to reach steady state: t_end >> (r_max - r_min)^2 / kappa.
    const ScalarType t_end    = ScalarType( 5 ) * ( r_max - r_min ) * ( r_max - r_min ) / kappa;
    const int        n_steps  = static_cast< int >( std::ceil( t_end / dt ) );

    util::logroot << "test_fct_diffusion:  level=" << level
                  << "  kappa=" << kappa
                  << "  h_min=" << h_min
                  << "  dt="    << dt
                  << "  t_end=" << t_end
                  << "  n_steps=" << n_steps << "\n";

    // FE field + helpers for XDMF output.
    VectorQ1Scalar< ScalarType >                T_fe( "T_fe", domain, mask_data );
    std::vector< VectorQ1Scalar< ScalarType > > tmps_fe;
    for ( int i = 0; i < 5; ++i )
        tmps_fe.emplace_back( "tmpfe", domain, mask_data );

    io::XDMFOutput xdmf( "test_fct_diffusion_out", domain, coords_shell, coords_radii );
    xdmf.add( T_fe.grid_data() );

    constexpr int output_interval = 100; // print error and write XDMF every N steps

    auto write_output = [&]( const int ts ) {
        const ScalarType t   = ts * dt;
        const ScalarType err = compute_error( domain, T.grid_data(), cell_centers.grid_data(), r_min, r_max );
        util::logroot << "  ts=" << ts << "  t=" << t << "  rel-L2 err=" << err << "\n";
        fv::hex::l2_project_fv_to_fe( T_fe, T, domain, coords_shell, coords_radii, tmps_fe );
        xdmf.write( ts );
    };

    write_output( 0 );

    for ( int ts = 1; ts <= n_steps; ++ts )
    {
        fv::hex::operators::fct_explicit_step(
            domain, T, u, cell_centers.grid_data(), coords_shell, coords_radii, dt, bufs, kappa );

        fv::hex::apply_dirichlet_bcs( T, boundary_mask_data, bcs, domain );

        if ( ts % output_interval == 0 || ts == n_steps )
            write_output( ts );
    }

    const ScalarType err_final = compute_error( domain, T.grid_data(), cell_centers.grid_data(), r_min, r_max );

    constexpr ScalarType tol = 0.05; // 5 % at this resolution
    if ( err_final > tol )
    {
        util::logroot << "FAILED: error " << err_final << " > tolerance " << tol << "\n";
        Kokkos::abort( "test_fct_diffusion: steady-state error too large" );
    }

    util::logroot << "PASSED\n";
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    test( 5 );
    return 0;
}
