
#include <kernels/common/grid_operations.hpp>

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen_hip.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/table.hpp"

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

enum class BenchmarkType : int
{
    EpsDivDivKerngenDouble,
    EpsDivDivKerngenHipDouble
};

constexpr auto all_benchmark_types = {
    BenchmarkType::EpsDivDivKerngenDouble,
    BenchmarkType::EpsDivDivKerngenHipDouble,
};

const std::map< BenchmarkType, std::string > benchmark_description = {
    { BenchmarkType::EpsDivDivKerngenDouble, "EpsDivDivKerngen Kokkos (double)" },
    { BenchmarkType::EpsDivDivKerngenHipDouble, "EpsDivDivKerngen HIP native (double)" } };

struct BenchmarkData
{
    int    level;
    long   dofs;
    double duration;
};

struct Parameters
{
    int min_level                   = 1;
    int max_level                   = 6;
    int executions                  = 5;
    int refinement_level_subdomains = 0;
    int lat_tile                    = 0; // 0 = use default
    int r_tile                      = 0;
    int r_passes                    = 0;
    bool check_hip                  = false;
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

BenchmarkData
    run( const BenchmarkType benchmark, const int level, const int executions, const int refinement_level_subdomains,
         const int lat_tile = 0, const int r_tile = 0, const int r_passes = 0 )
{
    if ( level < 1 )
    {
        Kokkos::abort( "level must be >= 1" );
    }

    const auto domain = grid::shell::DistributedDomain::create_uniform(
        level, level, 0.5, 1.0, refinement_level_subdomains, refinement_level_subdomains );
    const auto subdomain_distr = grid::shell::subdomain_distribution( domain );
    logroot << "Subdomain distribution: \n";
    logroot << " - total: " << subdomain_distr.total << "\n";
    logroot << " - min:   " << subdomain_distr.min << "\n";
    logroot << " - avg:   " << subdomain_distr.avg << "\n";
    logroot << " - max:   " << subdomain_distr.max << "\n\n";

    const auto domain_coarse = grid::shell::DistributedDomain::create_uniform(
        level - 1, level - 1, 0.5, 1.0, refinement_level_subdomains, refinement_level_subdomains );

    const auto coords_shell_double = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto coords_radii_double = grid::shell::subdomain_shell_radii< double >( domain );

    const auto coords_shell_float = grid::shell::subdomain_unit_sphere_single_shell_coords< float >( domain );
    const auto coords_radii_float = grid::shell::subdomain_shell_radii< float >( domain );

    const auto coords_shell_coarse_double =
        grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain_coarse );
    const auto coords_radii_coarse_double = grid::shell::subdomain_shell_radii< double >( domain_coarse );

    const auto coords_shell_coarse_float =
        grid::shell::subdomain_unit_sphere_single_shell_coords< float >( domain_coarse );
    const auto coords_radii_coarse_float = grid::shell::subdomain_shell_radii< float >( domain_coarse );

    auto mask_data        = grid::setup_node_ownership_mask_data( domain );
    auto mask_data_coarse = grid::setup_node_ownership_mask_data( domain_coarse );

    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto dofs_scalar = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    const auto dofs_vec    = 3 * dofs_scalar;

    VectorQ1Vec< double > src_vec_double( "src_vec_double", domain, mask_data );
    VectorQ1Vec< double > dst_vec_double( "dst_vec_double", domain, mask_data );

    VectorQ1Scalar< double > coeff_double( "coeff_double", domain, mask_data );
    linalg::assign( coeff_double, 1.0 );
    linalg::randomize( src_vec_double );
    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    double duration = 0.0;
    long   dofs     = 0;
    if ( benchmark == BenchmarkType::EpsDivDivKerngenDouble )
    {
        EpsilonDivDivKerngen A(
            domain,
            coords_shell_double,
            coords_radii_double,
            boundary_mask_data,
            coeff_double.grid_data(),
            bcs,
            false,
            linalg::OperatorApplyMode::Replace,
            linalg::OperatorCommunicationMode::CommunicateAdditively,
            linalg::OperatorStoredMatrixMode::Off,
            lat_tile, r_tile, r_passes );
        util::Timer t( "EpsDivDivKerngen - double" );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenHipDouble )
    {
        using namespace terra::fe::wedge::operators::shell::hip_kernels;

        const int lt  = lat_tile > 0 ? lat_tile : 4;
        const int rt  = r_tile   > 0 ? r_tile   : 8;
        const int rp  = r_passes > 0 ? r_passes : 2;
        const int rtb = rt * rp;

        const int hex_lat_val = static_cast< int >( coords_shell_double.extent( 1 ) ) - 1;
        const int hex_rad_val = static_cast< int >( coords_radii_double.extent( 1 ) ) - 1;
        const int lat_tiles_val = ( hex_lat_val + lt - 1 ) / lt;
        const int r_tiles_val   = ( hex_rad_val + rtb - 1 ) / rtb;
        const int num_blocks    = static_cast< int >( coords_shell_double.extent( 0 ) )
                                * lat_tiles_val * lat_tiles_val * r_tiles_val;
        const int team_size_val = lt * lt * rt * 2; // 2 threads per hex cell (one per wedge)

        const int nlev = rtb + 1;
        const int nxy  = ( lt + 1 ) * ( lt + 1 );
        const size_t shmem_bytes = sizeof( double ) * ( nxy * 3 + nxy * 3 * nlev + nxy * nlev + nlev );

        // Build params — just pointers + extents, no strides
        DNKernelParams params{};
        params.grid       = coords_shell_double.data();
        params.nx_grid    = static_cast< int >( coords_shell_double.extent( 1 ) );
        params.ny_grid    = static_cast< int >( coords_shell_double.extent( 2 ) );
        params.radii      = coords_radii_double.data();
        params.nr_radii   = static_cast< int >( coords_radii_double.extent( 1 ) );
        params.k          = coeff_double.grid_data().data();
        params.src_0      = src_vec_double.grid_data().comp_[0].data();
        params.src_1      = src_vec_double.grid_data().comp_[1].data();
        params.src_2      = src_vec_double.grid_data().comp_[2].data();
        params.dst_0      = dst_vec_double.grid_data().comp_[0].data();
        params.dst_1      = dst_vec_double.grid_data().comp_[1].data();
        params.dst_2      = dst_vec_double.grid_data().comp_[2].data();
        params.nx         = static_cast< int >( src_vec_double.grid_data().comp_[0].extent( 1 ) );
        params.ny         = static_cast< int >( src_vec_double.grid_data().comp_[0].extent( 2 ) );
        params.nr         = static_cast< int >( src_vec_double.grid_data().comp_[0].extent( 3 ) );
        params.mask       = reinterpret_cast< const uint8_t* >( boundary_mask_data.data() );
        params.bc_cmb     = static_cast< uint8_t >( bcs[0].bcf );
        params.bc_surface = static_cast< uint8_t >( bcs[1].bcf );
        params.lat_tile   = lt;
        params.r_tile     = rt;
        params.r_passes   = rp;
        params.r_tile_block = rtb;
        params.hex_lat    = hex_lat_val;
        params.hex_rad    = hex_rad_val;
        params.lat_tiles  = lat_tiles_val;
        params.r_tiles    = r_tiles_val;

        // Warmup
        for ( int w = 0; w < 3; ++w )
        {
            linalg::assign( dst_vec_double, 0.0 );
            launch_epsdivdiv_dn_matvec( params, num_blocks, team_size_val, shmem_bytes );
        }
        hipDeviceSynchronize();

        // Timed runs (zero + kernel + sync per iteration)
        MPI_Barrier( MPI_COMM_WORLD );
        Kokkos::Timer timer;
        for ( int i = 0; i < executions; ++i )
        {
            linalg::assign( dst_vec_double, 0.0 );
            launch_epsdivdiv_dn_matvec( params, num_blocks, team_size_val, shmem_bytes );
            hipDeviceSynchronize();
        }
        MPI_Barrier( MPI_COMM_WORLD );
        duration = timer.seconds() / executions;
        double duration_max = 0.0;
        MPI_Allreduce( &duration, &duration_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        duration = duration_max;
        dofs     = dofs_vec;
    }
    else
    {
        Kokkos::abort( "Unknown benchmark type" );
    }

    return BenchmarkData{ level, dofs, duration };
}

void run_all( const int min_level, const int max_level, const int executions, const int refinement_level_subdomains,
              const int lat_tile = 0, const int r_tile = 0, const int r_passes = 0 )
{
    logroot << "Running operator (matvec) benchmarks." << std::endl;
    logroot << "min_level:            " << min_level << std::endl;
    logroot << "max_level:            " << max_level << std::endl;
    logroot << "executions per level: " << executions << std::endl;
    logroot << "refinement for subdomains " << refinement_level_subdomains << std::endl;
    logroot << std::endl;
    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size ); // total number of MPI processes

    for ( auto benchmark : all_benchmark_types )
    {
        logroot << benchmark_description.at( benchmark ) << std::endl;

        util::Table table;

        for ( int i = min_level; i <= max_level; ++i )
        {
            const auto data = run( benchmark, i, executions, refinement_level_subdomains, lat_tile, r_tile, r_passes );
            table.add_row(
                { { "level", i },
                  { "dofs", data.dofs },
                  { "duration (s)", data.duration },
                  { "updated dofs/sec", data.dofs / data.duration } } );
        }

        table.print_pretty();

        // output a csv table of results
        if ( mpi::rank() == 0 )
        {
            std::ofstream out(
                "./csv/bo_np" + std::to_string( world_size ) + "_sdr" + std::to_string( refinement_level_subdomains ) +
                "_ml" + std::to_string( max_level ) + ".csv" );
            table.print_csv( out );
        }
        table.print_csv( logroot );

        logroot << std::endl;
        logroot << std::endl;
    }

    util::TimerTree::instance().aggregate_mpi();
    if ( mpi::rank() == 0 )
    {
        std::ofstream out(
            "./tts/bo_np" + std::to_string( world_size ) + "_sdr" + std::to_string( refinement_level_subdomains ) +
            "_ml" + std::to_string( max_level ) + ".json" );
        out << util::TimerTree::instance().json_aggregate();
        out.close();
    }
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    util::print_general_info( argc, argv );

    const auto description =
        "Operator benchmark. Runs a couple of matrix-vector multiplications for various operators to get an idea of the throughput.";
    CLI::App app{ description };

    Parameters parameters{};

    util::add_option_with_default( app, "--min-level", parameters.min_level, "Min refinement level." );
    util::add_option_with_default( app, "--max-level", parameters.max_level, "Max refinement level." );
    util::add_option_with_default(
        app,
        "--refinement-level-subdomains",
        parameters.refinement_level_subdomains,
        "Refinement level applied to form the subdomains." );
    util::add_option_with_default(
        app, "--executions", parameters.executions, "Number of matrix-vector multiplications to be executed." );
    util::add_option_with_default(
        app, "--lat-tile", parameters.lat_tile, "Lateral tile size (0 = default)." );
    util::add_option_with_default(
        app, "--r-tile", parameters.r_tile, "Radial tile size (0 = default)." );
    util::add_option_with_default(
        app, "--r-passes", parameters.r_passes, "Number of radial passes (0 = default)." );
    app.add_flag( "--check-hip", parameters.check_hip, "Run HIP correctness check against Kokkos." );

    CLI11_PARSE( app, argc, argv );

    if ( parameters.min_level < 1 )
    {
        logroot << "Error: min-level must be >= 1." << std::endl;
        return 1;
    }

    logroot << "\n" << description << "\n\n";

    util::print_cli_summary( app, logroot );
    logroot << "\n\n";

    if ( parameters.check_hip )
    {
        using namespace terra::fe::wedge::operators::shell;
        using namespace terra::fe::wedge::operators::shell::hip_kernels;

        const int level = parameters.max_level;
        logroot << "=== HIP correctness check at level " << level << " ===\n";

        const auto domain = grid::shell::DistributedDomain::create_uniform(
            level, level, 0.5, 1.0, parameters.refinement_level_subdomains, parameters.refinement_level_subdomains );
        auto mask_data          = grid::setup_node_ownership_mask_data( domain );
        auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );
        const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
        const auto coords_radii = grid::shell::subdomain_shell_radii< double >( domain );

        VectorQ1Scalar< double > coeff( "coeff", domain, mask_data );
        linalg::assign( coeff, 1.0 );
        VectorQ1Vec< double > src( "src", domain, mask_data );
        linalg::randomize( src );

        BoundaryConditions bcs = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } };

        // --- Kokkos reference (no communication, pure kernel comparison) ---
        VectorQ1Vec< double > dst_kokkos( "dst_kokkos", domain, mask_data );
        {
            EpsilonDivDivKerngen< double > A(
                domain, coords_shell, coords_radii, boundary_mask_data, coeff.grid_data(), bcs, false,
                linalg::OperatorApplyMode::Replace,
                linalg::OperatorCommunicationMode::SkipCommunication );
            apply( A, src, dst_kokkos );
        }
        Kokkos::fence();

        // --- HIP kernel ---
        VectorQ1Vec< double > dst_hip( "dst_hip", domain, mask_data );
        {
            // Zero dst (same as Kokkos Replace mode)
            linalg::assign( dst_hip, 0.0 );
            Kokkos::fence();
            const int lt  = parameters.lat_tile > 0 ? parameters.lat_tile : 4;
            const int rt  = parameters.r_tile   > 0 ? parameters.r_tile   : 8;
            const int rp  = parameters.r_passes > 0 ? parameters.r_passes : 2;
            const int rtb = rt * rp;
            const int hex_lat_val   = static_cast< int >( coords_shell.extent( 1 ) ) - 1;
            const int hex_rad_val   = static_cast< int >( coords_radii.extent( 1 ) ) - 1;
            const int lat_tiles_val = ( hex_lat_val + lt - 1 ) / lt;
            const int r_tiles_val   = ( hex_rad_val + rtb - 1 ) / rtb;
            const int num_blocks    = static_cast< int >( coords_shell.extent( 0 ) )
                                    * lat_tiles_val * lat_tiles_val * r_tiles_val;
            const int team_size_val = lt * lt * rt * 2; // 2 threads per hex cell (one per wedge)
            const int nlev = rtb + 1;
            const int nxy  = ( lt + 1 ) * ( lt + 1 );
            const size_t shmem_bytes = sizeof( double ) * ( nxy * 3 + nxy * 3 * nlev + nxy * nlev + nlev );

            DNKernelParams params{};
            params.grid       = coords_shell.data();
            params.nx_grid    = static_cast< int >( coords_shell.extent( 1 ) );
            params.ny_grid    = static_cast< int >( coords_shell.extent( 2 ) );
            params.radii      = coords_radii.data();
            params.nr_radii   = static_cast< int >( coords_radii.extent( 1 ) );
            params.k          = coeff.grid_data().data();
            params.src_0      = src.grid_data().comp_[0].data();
            params.src_1      = src.grid_data().comp_[1].data();
            params.src_2      = src.grid_data().comp_[2].data();
            params.dst_0      = dst_hip.grid_data().comp_[0].data();
            params.dst_1      = dst_hip.grid_data().comp_[1].data();
            params.dst_2      = dst_hip.grid_data().comp_[2].data();
            params.nx         = static_cast< int >( src.grid_data().comp_[0].extent( 1 ) );
            params.ny         = static_cast< int >( src.grid_data().comp_[0].extent( 2 ) );
            params.nr         = static_cast< int >( src.grid_data().comp_[0].extent( 3 ) );
            params.mask       = reinterpret_cast< const uint8_t* >( boundary_mask_data.data() );
            params.bc_cmb     = static_cast< uint8_t >( bcs[0].bcf );
            params.bc_surface = static_cast< uint8_t >( bcs[1].bcf );
            params.lat_tile   = lt;
            params.r_tile     = rt;
            params.r_passes   = rp;
            params.r_tile_block = rtb;
            params.hex_lat    = hex_lat_val;
            params.hex_rad    = hex_rad_val;
            params.lat_tiles  = lat_tiles_val;
            params.r_tiles    = r_tiles_val;

            launch_epsdivdiv_dn_matvec( params, num_blocks, team_size_val, shmem_bytes );
            hipDeviceSynchronize();
        }

        // --- Compare ---
        double max_abs_diff = 0.0;
        double max_abs_ref  = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            auto h_kokkos = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, dst_kokkos.grid_data().comp_[d] );
            auto h_hip    = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, dst_hip.grid_data().comp_[d] );

            for ( size_t i = 0; i < h_kokkos.span(); ++i )
            {
                const double diff = std::abs( h_kokkos.data()[i] - h_hip.data()[i] );
                const double ref  = std::abs( h_kokkos.data()[i] );
                if ( diff > max_abs_diff ) max_abs_diff = diff;
                if ( ref  > max_abs_ref )  max_abs_ref  = ref;
            }
        }

        const double rel_err = max_abs_ref > 0.0 ? max_abs_diff / max_abs_ref : 0.0;
        logroot << "Max absolute diff:  " << max_abs_diff << "\n";
        logroot << "Max absolute ref:   " << max_abs_ref << "\n";
        logroot << "Relative error:     " << rel_err << "\n";
        logroot << ( rel_err < 1e-10 ? "PASS" : "FAIL" ) << "\n\n";

        MPI_Finalize();
        return rel_err < 1e-10 ? 0 : 1;
    }

    run_all(
        parameters.min_level, parameters.max_level, parameters.executions, parameters.refinement_level_subdomains,
        parameters.lat_tile, parameters.r_tile, parameters.r_passes );

    MPI_Finalize();
}
