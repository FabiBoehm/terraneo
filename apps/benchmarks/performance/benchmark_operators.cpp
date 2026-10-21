
#include <kernels/common/grid_operations.hpp>

#include "fe/wedge/operators/shell/epsilon_divdiv.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_simple.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v01_initial.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v02_split_dimij.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v03_teams_precomp.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v04_shmem_coords.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v05_shmem_src_k.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v06_xy_tiling.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v07_split_paths.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v08_scalar_coalesced.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v09_separate_scatter.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v10_seq_rpasses.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/floating_point.hpp"
#include "util/table.hpp"

using namespace terra;

using fe::wedge::operators::shell::EpsDivDivStokes;
using fe::wedge::operators::shell::EpsilonDivDiv;
using fe::wedge::operators::shell::EpsilonDivDivSimple;
using fe::wedge::operators::shell::EpsilonDivDivKerngen;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV01Initial;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV02SplitDimij;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV03TeamsPrecomp;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV04ShmemCoords;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV05ShmemSrcK;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV06XyTiling;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV07SplitPaths;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV08ScalarCoalesced;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV09SeparateScatter;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV10SeqRpasses;
using fe::wedge::operators::shell::Laplace;
using fe::wedge::operators::shell::LaplaceSimple;
using fe::wedge::operators::shell::Stokes;
using fe::wedge::operators::shell::VectorLaplaceSimple;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::apply;
using linalg::DstOf;
using linalg::OperatorLike;
using linalg::SrcOf;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using terra::grid::shell::BoundaryConditions;
using util::logroot;

enum class BenchmarkType : int
{
    LaplaceFloat,
    LaplaceDouble,
    LaplaceSimpleDouble,
    VectorLaplaceFloat,
    VectorLaplaceDouble,
    VectorLaplaceNeumannDouble,
    EpsDivDivSimpleDouble,
    EpsDivDivFloat,
    EpsDivDivDouble,
    EpsDivDivKerngenDouble,
    EpsDivDivKerngenV01Initial,
    EpsDivDivKerngenV02SplitDimij,
    EpsDivDivKerngenV03TeamsPrecomp,
    EpsDivDivKerngenV04ShmemCoords,
    EpsDivDivKerngenV05ShmemSrcK,
    EpsDivDivKerngenV06XyTiling,
    EpsDivDivKerngenV07SplitPaths,
    EpsDivDivKerngenV08ScalarCoalesced,
    EpsDivDivKerngenV09SeparateScatter,
    EpsDivDivKerngenV10SeqRpasses,
    StokesDouble,
    EpsDivDivStokesDouble
};

constexpr auto all_benchmark_types = {
    BenchmarkType::EpsDivDivKerngenV07SplitPaths,
    BenchmarkType::EpsDivDivKerngenV08ScalarCoalesced,
    BenchmarkType::EpsDivDivKerngenV09SeparateScatter,
    BenchmarkType::EpsDivDivKerngenV10SeqRpasses,
    BenchmarkType::EpsDivDivKerngenDouble,
};

const std::map< BenchmarkType, std::string > benchmark_description = {
    { BenchmarkType::LaplaceFloat, "Laplace (float)" },
    { BenchmarkType::LaplaceSimpleDouble, "LaplaceSimple (double)" },
    { BenchmarkType::LaplaceDouble, "Laplace (double)" },
    { BenchmarkType::VectorLaplaceFloat, "VectorLaplace (float)" },
    { BenchmarkType::VectorLaplaceDouble, "VectorLaplace (double)" },
    { BenchmarkType::VectorLaplaceNeumannDouble, "VectorLaplaceNeumann (double)" },
    { BenchmarkType::EpsDivDivSimpleDouble, "EpsDivDivSimple (double, naive baseline)" },
    { BenchmarkType::EpsDivDivFloat, "EpsDivDiv (float)" },
    { BenchmarkType::EpsDivDivDouble, "EpsDivDiv (double, fused matvec)" },
    { BenchmarkType::EpsDivDivKerngenDouble, "EpsDivDivKerngen (double)" },
    { BenchmarkType::EpsDivDivKerngenV01Initial, "v01 initial (1t/cell, 6qp, 3x3 dimij)" },
    { BenchmarkType::EpsDivDivKerngenV02SplitDimij, "v02 split dimij (2x3 complexity)" },
    { BenchmarkType::EpsDivDivKerngenV03TeamsPrecomp, "v03 teams + precomputation" },
    { BenchmarkType::EpsDivDivKerngenV04ShmemCoords, "v04 shmem coords, collapsed qp" },
    { BenchmarkType::EpsDivDivKerngenV05ShmemSrcK, "v05 shmem src + k dofs" },
    { BenchmarkType::EpsDivDivKerngenV06XyTiling, "v06 xy tiling" },
    { BenchmarkType::EpsDivDivKerngenV07SplitPaths, "v07 split fast/slow paths" },
    { BenchmarkType::EpsDivDivKerngenV08ScalarCoalesced, "v08 scalar coalesced access" },
    { BenchmarkType::EpsDivDivKerngenV09SeparateScatter, "v09 separate scatter (7.6 Gdofs)" },
    { BenchmarkType::EpsDivDivKerngenV10SeqRpasses, "v10 seq r_passes (7.8 Gdofs)" },
    { BenchmarkType::StokesDouble, "Stokes (double)" },
    { BenchmarkType::EpsDivDivStokesDouble, "EpsDivDivStokes (double)" } };

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
    int  executions                  = 5;
    int  refinement_level_subdomains = 0;
    bool precision_sweep             = false;
    bool precision_matrix            = false;
    bool precision_ablation          = false;
    bool tile_sweep                  = false;
    int  tile_lat                    = 0;
    int  tile_r                      = 0;
    int  tile_passes                 = 0;
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
    run( const BenchmarkType benchmark, const int level, const int executions, const int refinement_level_subdomains )
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
    const auto dofs_scalar_coarse =
        kernels::common::count_masked< long >( mask_data_coarse, grid::NodeOwnershipFlag::OWNED );
    const auto dofs_stokes = dofs_vec + dofs_scalar_coarse;

    VectorQ1Scalar< double > src_scalar_double( "src_scalar_double", domain, mask_data );
    VectorQ1Scalar< double > dst_scalar_double( "dst_scalar_double", domain, mask_data );

    VectorQ1Scalar< float > src_scalar_float( "src_scalar_float", domain, mask_data );
    VectorQ1Scalar< float > dst_scalar_float( "dst_scalar_float", domain, mask_data );

    VectorQ1Vec< double > src_vec_double( "src_vec_double", domain, mask_data );
    VectorQ1Vec< double > dst_vec_double( "dst_vec_double", domain, mask_data );

    VectorQ1Vec< float > src_vec_float( "src_vec_float", domain, mask_data );
    VectorQ1Vec< float > dst_vec_float( "dst_vec_float", domain, mask_data );

    VectorQ1IsoQ2Q1< double > src_stokes_double(
        "src_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );
    VectorQ1IsoQ2Q1< double > dst_stokes_double(
        "dst_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );

    VectorQ1IsoQ2Q1< float > src_stokes_float(
        "src_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );
    VectorQ1IsoQ2Q1< float > dst_stokes_float(
        "dst_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );

    VectorQ1Scalar< double > coeff_double( "coeff_double", domain, mask_data );
    VectorQ1Scalar< float >  coeff_float( "coeff_float", domain, mask_data );

    linalg::assign( coeff_double, 1.0 );
    linalg::assign( coeff_float, 1.0 );

    linalg::randomize( src_scalar_double );
    linalg::randomize( src_scalar_float );
    linalg::randomize( src_vec_double );
    linalg::randomize( src_vec_float );
    linalg::randomize( src_stokes_double );
    linalg::randomize( src_stokes_float );
    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    double duration = 0.0;
    long   dofs     = 0;
    if ( benchmark == BenchmarkType::LaplaceFloat )
    {
        LaplaceSimple< float > A( domain, coords_shell_float, coords_radii_float, true, false );
        duration = measure_run_time( executions, A, src_scalar_float, dst_scalar_float );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::LaplaceSimpleDouble )
    {
        LaplaceSimple< double > A( domain, coords_shell_double, coords_radii_double, true, false );
        util::Timer             t( "Laplace - double" );
        duration = measure_run_time( executions, A, src_scalar_double, dst_scalar_double );
        dofs     = dofs_scalar;
    }
    else if ( benchmark == BenchmarkType::LaplaceDouble )
    {
        Laplace< double > A( domain, coords_shell_double, coords_radii_double, boundary_mask_data, true, false );
        util::Timer       t( "Laplace - double" );
        duration = measure_run_time( executions, A, src_scalar_double, dst_scalar_double );
        dofs     = dofs_scalar;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceFloat )
    {
        VectorLaplaceSimple< float > A( domain, coords_shell_float, coords_radii_float, true, false );
        duration = measure_run_time( executions, A, src_vec_float, dst_vec_float );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceDouble )
    {
        VectorLaplaceSimple< double > A( domain, coords_shell_double, coords_radii_double, true, false );
        util::Timer                   t( "VectorLaplace - double" );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceNeumannDouble )
    {
        VectorLaplaceSimple< double > A( domain, coords_shell_double, coords_radii_double, false, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivSimpleDouble )
    {
        EpsilonDivDivSimple< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), true, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivFloat )
    {
        EpsilonDivDiv A(
            domain, coords_shell_float, coords_radii_float, boundary_mask_data, coeff_float.grid_data(), true, false );
        util::Timer t( "EpsDivDiv - float" );
        duration = measure_run_time( executions, A, src_vec_float, dst_vec_float );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivDouble )
    {
        EpsilonDivDiv A(
            domain,
            coords_shell_double,
            coords_radii_double,
            boundary_mask_data,
            coeff_double.grid_data(),
            true,
            false );
        util::Timer t( "EpsDivDiv - double" );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenDouble )
    {
        EpsilonDivDivKerngen A(
            domain,
            coords_shell_double,
            coords_radii_double,
            boundary_mask_data,
            coeff_double.grid_data(),
            bcs,
            false );
        util::Timer t( "EpsDivDivKerngen - double" );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV01Initial )
    {
        EpsilonDivDivKerngenV01Initial< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV02SplitDimij )
    {
        EpsilonDivDivKerngenV02SplitDimij< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV03TeamsPrecomp )
    {
        EpsilonDivDivKerngenV03TeamsPrecomp< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV04ShmemCoords )
    {
        EpsilonDivDivKerngenV04ShmemCoords< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV05ShmemSrcK )
    {
        EpsilonDivDivKerngenV05ShmemSrcK< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV06XyTiling )
    {
        EpsilonDivDivKerngenV06XyTiling< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV07SplitPaths )
    {
        EpsilonDivDivKerngenV07SplitPaths< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV08ScalarCoalesced )
    {
        EpsilonDivDivKerngenV08ScalarCoalesced< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV09SeparateScatter )
    {
        EpsilonDivDivKerngenV09SeparateScatter< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV10SeqRpasses )
    {
        EpsilonDivDivKerngenV10SeqRpasses< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::StokesDouble )
    {
        Stokes< double > A(
            domain, domain_coarse, coords_shell_double, coords_radii_double, boundary_mask_data, bcs, false );
        duration = measure_run_time( executions, A, src_stokes_double, dst_stokes_double );
        dofs     = dofs_stokes;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivStokesDouble )
    {
        EpsDivDivStokes< double > A(
            domain,
            domain_coarse,
            coords_shell_double,
            coords_radii_double,
            boundary_mask_data,
            coeff_double.grid_data(),
            bcs,
            false );
        duration = measure_run_time( executions, A, src_stokes_double, dst_stokes_double );
        dofs     = dofs_stokes;
    }
    else
    {
        Kokkos::abort( "Unknown benchmark type" );
    }

    return BenchmarkData{ level, dofs, duration };
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
static TileOverride tile_override{};

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

    const double duration = measure_run_time( executions, op, src, dst );

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
        const double gdofs = data.dofs / data.duration / 1e9;
        const double gbs   = modelled_bytes< S >( data.dofs / 3 ) / data.duration / 1e9;

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


/// Storage / arithmetic breakdown. Each 16-bit precision appears three times:
/// matched (both narrow), storage-only (narrow fields, double arithmetic) and
/// arithmetic-only (double fields, narrow arithmetic). The differences separate
/// the cost of the narrow load/store path from the cost of the narrow arithmetic.

/// Team-tile tuning for the production kernel, at every precision.
///
/// The tile shape is (lat, lat, r) with r_passes radial passes, so the team holds
/// lat^2 * r threads and stages ( r * r_passes + 1 ) * ( lat + 1 )^2 nodes. The fast
/// Dirichlet/Neumann path is launched with LaunchBounds< 128, 5 >, which caps the team
/// at 128 threads; the candidates below respect that, and set_tiling() rejects anything
/// that would not launch or would overflow 64 KiB of scratch.
void run_tile_sweep( const int min_level,
                     const int max_level,
                     const int executions,
                     const int refinement_level_subdomains )
{
    using half  = Kokkos::Experimental::half_t;
    using bhalf = Kokkos::Experimental::bhalf_t;

    struct Tile
    {
        int lat, r, passes;
    };
    // lat^2 * r <= 128, spanning lateral-heavy to radial-heavy shapes.
    const std::vector< Tile > tiles = {
        { 1, 32, 1 }, { 1, 64, 1 }, { 1, 128, 1 }, { 1, 32, 2 }, { 1, 64, 2 },
        { 2, 8, 1 },  { 2, 16, 1 }, { 2, 32, 1 },  { 2, 8, 2 },  { 2, 16, 2 },
        { 2, 32, 2 }, { 4, 2, 1 },  { 4, 4, 1 },   { 4, 8, 1 },  { 4, 4, 2 },
        { 4, 8, 2 },  { 4, 8, 4 },  { 8, 2, 1 },   { 8, 2, 2 },  { 8, 2, 4 },
        // Teams above 128 threads: set_tiling rejects these unless the operator was
        // built with a raised TERRA_EPSDIVDIV_MAX_TEAM, and a rejected shape is skipped.
        { 2, 64, 1 }, { 4, 16, 1 }, { 4, 32, 1 },  { 8, 4, 1 },  { 8, 8, 1 },
    };

    logroot << "Team-tile sweep for EpsDivDivKerngen, all precisions." << std::endl;
    logroot << "min_level " << min_level << ", max_level " << max_level << ", executions "
            << executions << std::endl
            << std::endl;

    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    util::Table table;

    for ( int level = min_level; level <= max_level; ++level )
    {
        for ( const auto& t : tiles )
        {
            struct Arm
            {
                const char* name;
                BenchmarkData ( *fn )( int, int, int, int, int, int );
            };
            const Arm arms[] = {
                { "double", &run_precision< double, double > },
                { "float", &run_precision< float, float > },
                { "bf16", &run_precision< bhalf, bhalf > },
                { "fp16", &run_precision< half, half > },
            };

            for ( const auto& arm : arms )
            {
                const auto d = arm.fn( level, executions, refinement_level_subdomains, t.lat, t.r, t.passes );
                const bool ok = d.duration > 0.0;
                table.add_row( { { "precision", std::string( arm.name ) },
                                 { "level", level },
                                 { "lat", t.lat },
                                 { "r", t.r },
                                 { "passes", t.passes },
                                 { "team", t.lat * t.lat * t.r },
                                 { "dofs", d.dofs },
                                 { "duration (s)", ok ? d.duration : 0.0 },
                                 { "GDoF/s", ok ? d.dofs / d.duration / 1e9 : 0.0 },
                                 { "launched", ok ? 1 : 0 } } );
            }
        }
        logroot << "level " << level << " done" << std::endl;
    }

    table.print_pretty();
    if ( mpi::rank() == 0 )
    {
        std::ofstream out( "./csv/tiles_np" + std::to_string( world_size ) + "_ml" +
                           std::to_string( max_level ) + ".csv" );
        table.print_csv( out );
    }
    table.print_csv( logroot );
}

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

void run_precision_matrix( const int min_level,
                           const int max_level,
                           const int executions,
                           const int refinement_level_subdomains )
{
    using half  = Kokkos::Experimental::half_t;
    using bhalf = Kokkos::Experimental::bhalf_t;

    logroot << "Storage / arithmetic breakdown for the EpsDivDivKerngen fast path." << std::endl;
    logroot << "min_level: " << min_level << ", max_level: " << max_level
            << ", executions per level: " << executions << std::endl
            << std::endl;

    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    const int mn = min_level, mx = max_level, ex = executions, r = refinement_level_subdomains;

    sweep_one_precision< double, double >( "double", mn, mx, ex, r, world_size );

    sweep_one_precision< float, float >( "float_matched", mn, mx, ex, r, world_size );
    sweep_one_precision< float, double >( "float_store_only", mn, mx, ex, r, world_size );
    sweep_one_precision< double, float >( "float_arith_only", mn, mx, ex, r, world_size );

    sweep_one_precision< bhalf, bhalf >( "bf16_matched", mn, mx, ex, r, world_size );
    sweep_one_precision< bhalf, double >( "bf16_store_only", mn, mx, ex, r, world_size );
    sweep_one_precision< double, bhalf >( "bf16_arith_only", mn, mx, ex, r, world_size );

    sweep_one_precision< half, half >( "fp16_matched", mn, mx, ex, r, world_size );
    sweep_one_precision< half, double >( "fp16_store_only", mn, mx, ex, r, world_size );
    sweep_one_precision< double, half >( "fp16_arith_only", mn, mx, ex, r, world_size );
}

void run_all( const int min_level, const int max_level, const int executions, const int refinement_level_subdomains )
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
            const auto data = run( benchmark, i, executions, refinement_level_subdomains );
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
        app, "--tile-lat", parameters.tile_lat, "Lateral tile size for the precision sweeps (0: operator default)." );
    util::add_option_with_default(
        app, "--tile-r", parameters.tile_r, "Radial tile size for the precision sweeps (0: operator default)." );
    util::add_option_with_default(
        app, "--tile-passes", parameters.tile_passes, "Radial passes per team for the precision sweeps (0: operator default)." );
    app.add_flag( "--tile-sweep",
                  parameters.tile_sweep,
                  "Sweep team-tile shapes for the production kernel at every precision." );
    app.add_flag( "--precision-matrix",
                  parameters.precision_matrix,
                  "Separate storage precision from arithmetic precision for each narrow type." );
    app.add_flag( "--precision-ablation",
                  parameters.precision_ablation,
                  "Run everything in low precision, then return storage, arithmetic and "
                  "geometry to double one at a time." );
    app.add_flag( "--precision-sweep",
                  parameters.precision_sweep,
                  "Instead of the operator list, sweep EpsDivDivKerngen over double/float/bf16/fp16 per level." );

    CLI11_PARSE( app, argc, argv );

    tile_override = { parameters.tile_lat, parameters.tile_r, parameters.tile_passes };

    if ( parameters.min_level < 1 )
    {
        logroot << "Error: min-level must be >= 1." << std::endl;
        return 1;
    }

    logroot << "\n" << description << "\n\n";

    util::print_cli_summary( app, logroot );
    logroot << "\n\n";

    if ( parameters.tile_sweep )
    {
        run_tile_sweep(
            parameters.min_level, parameters.max_level, parameters.executions, parameters.refinement_level_subdomains );
    }
    else if ( parameters.precision_ablation )
    {
        run_precision_ablation(
            parameters.min_level, parameters.max_level, parameters.executions, parameters.refinement_level_subdomains );
    }
    else if ( parameters.precision_matrix )
    {
        run_precision_matrix(
            parameters.min_level, parameters.max_level, parameters.executions, parameters.refinement_level_subdomains );
    }
    else if ( parameters.precision_sweep )
    {
        run_precision_sweep(
            parameters.min_level, parameters.max_level, parameters.executions, parameters.refinement_level_subdomains );
    }
    else
    {
        run_all(
            parameters.min_level, parameters.max_level, parameters.executions, parameters.refinement_level_subdomains );
    }

    MPI_Finalize();
}
