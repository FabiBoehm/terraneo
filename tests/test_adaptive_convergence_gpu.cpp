// Convergence study on the ADAPTIVE mesh code path.
//
// A manufactured solution for the production EpsilonDivDivKerngen viscous operator (u, k = 2 + sin z and
// the analytic RHS are verbatim from test_epsilon_divdiv_cg). We solve on a sequence of UNIFORMLY refined
// meshes -- refine EVERY leaf each level, so h halves and the mesh stays conforming (no hanging nodes) --
// and measure the discrete L2 error  ||u_h - u||_{L2} = sqrt( e^T M e )  with the (assembled) mass matrix.
//
// Q1 elements => the discretization is 2nd order: the error must drop by ~4x per uniform refinement
// (rate = log2(err_coarse / err_fine) -> 2). This confirms the AMR assembly (coincident-class summation,
// the constrained C^T A C operator, the distributed halo) reproduces a CONSISTENT, 2nd-order operator --
// not just "a" answer. Hanging-node consistency under local refinement is covered separately by
// test_adaptive_epsilon_divdiv_gpu (T1 all-refined == uniform, T2/T3 local/nested error <= base).
//
// Needs a GPU -- run on an H100 node. Degenerates cleanly at np=1; identical error at np=1/2/4.

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/pcg.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/adaptive_distribute.hpp"
#include "terra/grid/shell/adaptive_solve.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using grid::shell::BoundaryConditions;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using namespace terra::grid::shell::amr;

static int g_failures = 0;
static int g_checks   = 0;
#define CHECK( cond )                                                       \
    do                                                                      \
    {                                                                       \
        ++g_checks;                                                         \
        if ( !( cond ) )                                                    \
        {                                                                   \
            std::printf( "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond ); \
            ++g_failures;                                                   \
        }                                                                   \
    } while ( 0 )

// --- manufactured solution / coefficient / RHS, verbatim from test_epsilon_divdiv_cg.cpp -----------
struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const bool on_boundary = util::has_flag( mask_( s, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );
        if ( !only_boundary_ || on_boundary )
        {
            data_( s, x, y, r, 0 ) = Kokkos::sin( 2 * c( 0 ) ) * Kokkos::sin( 2 * c( 2 ) ) * Kokkos::sinh( c( 1 ) );
            data_( s, x, y, r, 1 ) = 2 * Kokkos::sin( 2 * c( 1 ) ) * Kokkos::sin( 2 * c( 2 ) ) * Kokkos::sinh( c( 0 ) );
            data_( s, x, y, r, 2 ) = 4 * Kokkos::sin( 2 * c( 0 ) ) * Kokkos::sin( 2 * c( 1 ) ) * Kokkos::sinh( c( 2 ) );
        }
    }
};

struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r )             = 2 + Kokkos::sin( c( 2 ) );
    }
};

struct RHSInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( s, x, y, r, grid_, radii_ );
        // x component
        {
            const real_t x0  = Kokkos::sinh( coords( 1 ) );
            const real_t x1  = Kokkos::sin( coords( 2 ) ) + 2;
            const real_t x2  = 2 * coords( 0 );
            const real_t x3  = Kokkos::sin( x2 );
            const real_t x4  = 2 * coords( 2 );
            const real_t x5  = Kokkos::sin( x4 );
            const real_t x6  = x0 * x3;
            const real_t x7  = Kokkos::cos( x2 );
            const real_t x8  = 2 * coords( 1 );
            const real_t x9  = Kokkos::sin( x8 );
            const real_t x10 = x5 * x6;
            const real_t x11 = Kokkos::cosh( coords( 2 ) );
            const real_t x12 = 2 * x1;
            const real_t x13 = x5 * Kokkos::cos( x8 ) * Kokkos::cosh( coords( 0 ) );
            data_( s, x, y, r, 0 ) =
                8.0 * x0 * x1 * x3 * x5 + 0.66666666666666663 * x1 * ( -4 * x10 + 8 * x11 * x7 * x9 + 4 * x13 ) -
                x12 * ( -2.0 * x10 + 4.0 * x11 * x7 * x9 ) - x12 * ( 0.5 * x10 + 2.0 * x13 ) -
                2 * ( 1.0 * x6 * Kokkos::cos( x4 ) + 4.0 * x7 * x9 * Kokkos::sinh( coords( 2 ) ) ) *
                    Kokkos::cos( coords( 2 ) );
        }
        // y component
        {
            const real_t x0  = Kokkos::sinh( coords( 0 ) );
            const real_t x1  = Kokkos::sin( coords( 2 ) ) + 2;
            const real_t x2  = 2 * coords( 1 );
            const real_t x3  = Kokkos::sin( x2 );
            const real_t x4  = 2 * coords( 2 );
            const real_t x5  = Kokkos::sin( x4 );
            const real_t x6  = Kokkos::cos( x2 );
            const real_t x7  = 2 * coords( 0 );
            const real_t x8  = Kokkos::sin( x7 );
            const real_t x9  = 4.0 * x6 * x8;
            const real_t x10 = x0 * x3;
            const real_t x11 = Kokkos::cosh( coords( 2 ) );
            const real_t x12 = x10 * x5;
            const real_t x13 = 2 * x1;
            const real_t x14 = 1.0 * x5;
            const real_t x15 = std::cos( x7 ) * Kokkos::cosh( coords( 1 ) );
            data_( s, x, y, r, 1 ) =
                16.0 * x0 * x1 * x3 * x5 + 0.66666666666666663 * x1 * ( 8 * x11 * x6 * x8 - 8 * x12 + 2 * x15 * x5 ) -
                x13 * ( x10 * x14 + x14 * x15 ) - x13 * ( x11 * x9 - 4.0 * x12 ) -
                2 * ( 2.0 * x10 * Kokkos::cos( x4 ) + x9 * Kokkos::sinh( coords( 2 ) ) ) * Kokkos::cos( coords( 2 ) );
        }
        // z component
        {
            const real_t x0  = Kokkos::cos( coords( 2 ) );
            const real_t x1  = 2 * coords( 0 );
            const real_t x2  = 2 * coords( 1 );
            const real_t x3  = Kokkos::sin( x1 ) * Kokkos::sin( x2 );
            const real_t x4  = x3 * Kokkos::cosh( coords( 2 ) );
            const real_t x5  = Kokkos::sin( coords( 2 ) ) + 2;
            const real_t x6  = x3 * Kokkos::sinh( coords( 2 ) );
            const real_t x7  = 8.0 * x6;
            const real_t x8  = Kokkos::sinh( coords( 1 ) );
            const real_t x9  = Kokkos::cos( x1 );
            const real_t x10 = 2 * coords( 2 );
            const real_t x11 = Kokkos::cos( x10 );
            const real_t x12 = 2 * x5;
            const real_t x13 = Kokkos::sinh( coords( 0 ) );
            const real_t x14 = Kokkos::cos( x2 );
            const real_t x15 = Kokkos::sin( x10 );
            const real_t x16 = x8 * x9;
            const real_t x17 = x13 * x14;
            data_( s, x, y, r, 2 ) =
                -8.0 * x0 * x4 + 0.66666666666666663 * x0 * ( 2 * x15 * x16 + 4 * x15 * x17 + 4 * x4 ) -
                x12 * ( 4.0 * x11 * x13 * x14 - x7 ) - x12 * ( 2.0 * x11 * x8 * x9 - x7 ) - x5 * x7 +
                0.66666666666666663 * x5 * ( 4 * x11 * x16 + 8 * x11 * x17 + 4 * x6 );
        }
    }
};

struct LevelResult
{
    long   dofs;
    double l2;     // ||u_h - u||_{L2} = sqrt(e^T M e)
    double relres; // solver residual (must be << discretization error)
};

// Solve the manufactured problem on `mesh` and return the mass-weighted L2 error. Operator setup is
// identical to test_adaptive_epsilon_divdiv_gpu's solve_eps_distributed.
static LevelResult solve_level( const DistributedAdaptiveMesh& mesh, const std::shared_ptr< util::Table >& table,
                                const std::string& tag )
{
    using ScalarType  = double;
    using Epsilon     = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarType, 3 >;
    using Mass        = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;
    using Wrapper     = AdaptiveDistributedConstrainedOperator< Epsilon >;
    using MassWrapper = AdaptiveDistributedConstrainedOperator< Mass >;

    const auto& dom   = mesh.domain;
    const auto  mask  = adaptive_ownership_mask( mesh );
    const auto  bmask = adaptive_boundary_mask( dom );

    VectorQ1Vec< ScalarType >    u( "u", dom, mask ), g( "g", dom, mask ), tmp( "tmp", dom, mask );
    VectorQ1Vec< ScalarType >    Adiagg( "Adiagg", dom, mask ), solution( "solution", dom, mask );
    VectorQ1Vec< ScalarType >    error( "error", dom, mask ), Merror( "Merror", dom, mask );
    VectorQ1Vec< ScalarType >    b( "b", dom, mask ), rr( "rr", dom, mask );
    VectorQ1Vec< ScalarType >    scratch_a( "sa", dom, mask ), scratch_n( "sn", dom, mask ), scratch_m( "sm", dom, mask );
    VectorQ1Scalar< ScalarType > k( "k", dom, mask );

    const auto ndofs   = kernels::common::count_masked< long >( mask, grid::NodeOwnershipFlag::OWNED, mesh.comm );
    const auto coords  = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom );
    const auto radii_g = grid::shell::subdomain_shell_radii< ScalarType >( dom );

    Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          KInterpolator{ coords, radii_g, k.grid_data() } );
    Kokkos::parallel_for( "sol", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator{ coords, radii_g, solution.grid_data(), bmask, false } );
    Kokkos::parallel_for( "g", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator{ coords, radii_g, g.grid_data(), bmask, true } );
    Kokkos::parallel_for( "rhs", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          RHSInterpolator{ coords, radii_g, tmp.grid_data() } );
    Kokkos::fence();

    BoundaryConditions bcs_nn = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };
    Epsilon A_loc( dom, coords, radii_g, bmask, k.grid_data(), bcs_nn, false,
                   linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::SkipCommunication );
    Mass    M_loc( dom, coords, radii_g, false, linalg::OperatorApplyMode::Replace,
                linalg::OperatorCommunicationMode::SkipCommunication );

    Wrapper     A_sys( A_loc, mesh, scratch_a, bmask, grid::shell::ShellBoundaryFlag::BOUNDARY );
    Wrapper     A_neu( A_loc, mesh, scratch_n );
    MassWrapper M_c( M_loc, mesh, scratch_m );

    // rhs: b = C^T M C f, lifted by the boundary data g, identity rows on the Dirichlet DoFs
    linalg::apply( M_c, tmp, b );
    linalg::apply( A_neu, g, tmp );
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );
    kernels::common::assign_masked_else_keep_old( b.grid_data(), g.grid_data(), bmask,
                                                  grid::shell::ShellBoundaryFlag::BOUNDARY );
    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 8000, 1e-12, 1e-12 };
    linalg::solvers::PCG< Wrapper >            pcg( solver_params, table, { tmp, Adiagg, error, rr } );
    pcg.set_tag( tag );
    linalg::solvers::solve( pcg, A_sys, u, b );
    Kokkos::fence();
    apply_constraint_device( mesh.t_local_d, u.grid_data() );

    // mass-weighted discrete L2 error: e = u - u_exact,  ||e||_L2 = sqrt(e^T M e)
    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    linalg::apply( M_c, error, Merror );
    const double l2 = std::sqrt( std::fabs( dot( Merror, error ) ) );

    linalg::apply( A_sys, u, rr );
    linalg::lincomb( rr, { 1.0, -1.0 }, { b, rr } );
    const double relres = std::sqrt( dot( rr, rr ) / dot( b, b ) );

    if ( mpi::rank( mesh.comm ) == 0 )
        std::printf( "  [%s] dofs %9ld  L2 %.6e  relres %.2e  hanging %zu\n", tag.c_str(), ndofs, l2, relres,
                     mesh.t_local.con_dst.size() );
    return LevelResult{ ndofs, l2, relres };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        auto table = std::make_shared< util::Table >();

        const std::vector< double > radii = { 0.5, 0.75, 1.0 }; // 2 radial layers at base
        const int                   LDR = 2, S_lat = 2, S_rad = 1;
        const int                   NLEV = 3; // meshes at levels 0..NLEV (NLEV uniform refinements)

        AdaptiveForest f( NLEV, S_lat, S_rad ); // budget M = NLEV so the finest level can still exist

        std::vector< LevelResult > R;
        for ( int lvl = 0; lvl <= NLEV; ++lvl )
        {
            auto mesh = build_distributed_adaptive_mesh( MPI_COMM_WORLD, LDR, radii, f,
                                                         grid::shell::subdomain_to_rank_by_diamond );
            CHECK( mesh.t_local.con_dst.empty() ); // uniform => conforming, no hanging nodes
            R.push_back( solve_level( mesh, table, "level" + std::to_string( lvl ) ) );

            if ( lvl < NLEV )
            {
                auto all = f.leaves(); // copy: refine() mutates the leaf set
                f.refine( all );       // refine EVERY leaf -> uniform one level finer (h halves)
                f.balance_2to1();      // no-op on a uniform mesh
                CHECK( f.validate() );
            }
        }

        // --- convergence table. Q1 => O(h^2): each uniform refinement HALVES h and so must drop the L2
        //     error by ~2^2 = 4x. (Equivalently order p = log2(drop) -> 2.) --------------------------
        if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        {
            std::printf( "\n  level      dofs        h~        L2 error    err drop   order\n" );
            std::printf( "  ---------------------------------------------------------------\n" );
            for ( int l = 0; l <= NLEV; ++l )
            {
                const double h = 1.0 / static_cast< double >( 1 << l );
                if ( l == 0 )
                    std::printf( "  %3d  %10ld  %8.4f  %.6e      --       --\n", l, R[l].dofs, h, R[l].l2 );
                else
                {
                    const double drop = R[l - 1].l2 / R[l].l2; // ~4 for 2nd order
                    std::printf( "  %3d  %10ld  %8.4f  %.6e   %6.2fx   %5.2f\n", l, R[l].dofs, h, R[l].l2,
                                 drop, std::log2( drop ) );
                }
            }
            std::printf( "\n" );
        }

        // solver must not pollute the discretization error, and each refinement must reduce the error
        for ( int l = 0; l <= NLEV; ++l )
            CHECK( R[l].relres < 1e-8 );
        for ( int l = 1; l <= NLEV; ++l )
            CHECK( R[l].l2 < R[l - 1].l2 );

        // asymptotic 2nd order: the finest uniform refinement must drop the error by ~4x (order ~2)
        const double last_drop = R[NLEV - 1].l2 / R[NLEV].l2;
        CHECK( last_drop > 3.4 ); // 2^1.77; allow pre-asymptotic slack below the ideal 4x
        if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
            std::printf( "  finest error drop = %.2fx  (Q1 2nd order => ~4x; order = %.2f)\n", last_drop,
                         std::log2( last_drop ) );
    }

    const bool ok = ( g_failures == 0 );
    if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        std::printf( "test_adaptive_convergence_gpu: %s (%d checks)\n", ok ? "ALL PASS" : "FAILURE(S)", g_checks );
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
