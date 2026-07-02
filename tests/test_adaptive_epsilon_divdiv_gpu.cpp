// GPU integration test: the FIRST SOLVE ON AN ADAPTIVE MESH -- CG on the production
// EpsilonDivDivKerngen operator wrapped into the constrained operator C^T A C (adaptive_solve.hpp),
// with Dirichlet imposed on the ASSEMBLED-CONSTRAINED system (the wrapper's eliminated variant).
//
// NOTE on boundary conditions: kerngen's internal, mask-driven Dirichlet treatment is adaptive-safe
// and reproduces the uniform reference EXACTLY on hanging-free meshes -- but with hanging nodes whose
// constraint parents lie on the Dirichlet boundary, the folding C^T A C regenerates couplings to
// boundary DoFs that element-level elimination cannot see, which double-counts against the RHS lifting
// (measured: 2.2x larger l2 on a boundary-touching refined block, job 534969). Hence the post-folding
// projection here; see adaptive_solve.hpp for the full derivation. Manufactured solution, RHS and
// k = 2 + sin z are those of test_epsilon_divdiv_cg.
//
// T1 (equivalence): refining EVERY block once yields the same physical mesh as the uniform domain one
//     level up, and the discretizations are element-identical -- so the l2 errors of the adaptive solve
//     (AMR assembly, no hanging nodes) and the trusted uniform solve must agree to solver accuracy.
//     Both sides use the same post-folding Dirichlet projection, so the comparison isolates exactly
//     the adaptive machinery: geometry, ownership, class assembly.
// T2 (local refinement): one non-corner block refined (2:1, hanging nodes active in the solve through
//     the constraint). CG must converge and the l2 error must not exceed the uniform base mesh's.
//
// Needs a GPU -- run on an H100 node.

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
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using grid::shell::subdomain_to_rank_all_root;
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
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                       only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        bool                              only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_( mask )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

          const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

          if ( !only_boundary_ || on_boundary )
            {
            data_( local_subdomain_id, x, y, r, 0 ) =
                Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sin( 2 * coords( 2 ) ) * Kokkos::sinh( coords( 1 ) );
            data_( local_subdomain_id, x, y, r, 1 ) =
                2 * Kokkos::sin( 2 * coords( 1 ) ) * Kokkos::sin( 2 * coords( 2 ) ) * Kokkos::sinh( coords( 0 ) );
            data_( local_subdomain_id, x, y, r, 2 ) =
                4 * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sin( 2 * coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
        }
    }
};

struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  value  = 2 + Kokkos::sin( coords( 2 ) );
        data_( local_subdomain_id, x, y, r ) = value;
    }
};

struct RHSInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    RHSInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        // x component of rhs
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
            data_( local_subdomain_id, x, y, r, 0 ) =
                8.0 * x0 * x1 * x3 * x5 + 0.66666666666666663 * x1 * ( -4 * x10 + 8 * x11 * x7 * x9 + 4 * x13 ) -
                x12 * ( -2.0 * x10 + 4.0 * x11 * x7 * x9 ) - x12 * ( 0.5 * x10 + 2.0 * x13 ) -
                2 * ( 1.0 * x6 * Kokkos::cos( x4 ) + 4.0 * x7 * x9 * Kokkos::sinh( coords( 2 ) ) ) *
                    Kokkos::cos( coords( 2 ) );
        }

        // y component of rhs
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
            data_( local_subdomain_id, x, y, r, 1 ) =
                16.0 * x0 * x1 * x3 * x5 + 0.66666666666666663 * x1 * ( 8 * x11 * x6 * x8 - 8 * x12 + 2 * x15 * x5 ) -
                x13 * ( x10 * x14 + x14 * x15 ) - x13 * ( x11 * x9 - 4.0 * x12 ) -
                2 * ( 2.0 * x10 * Kokkos::cos( x4 ) + x9 * Kokkos::sinh( coords( 2 ) ) ) * Kokkos::cos( coords( 2 ) );
        }

        // z component of rhs
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
            data_( local_subdomain_id, x, y, r, 2 ) =
                -8.0 * x0 * x4 + 0.66666666666666663 * x0 * ( 2 * x15 * x16 + 4 * x15 * x17 + 4 * x4 ) -
                x12 * ( 4.0 * x11 * x13 * x14 - x7 ) - x12 * ( 2.0 * x11 * x8 * x9 - x7 ) - x5 * x7 +
                0.66666666666666663 * x5 * ( 4 * x11 * x16 + 8 * x11 * x17 + 4 * x6 );
        }
    }
};

struct SolveResult
{
    double      l2;
    long        ndofs;
    double      relres;
    std::size_t hanging;
};

// One solve, shared by the adaptive and the uniform-reference paths. For `adaptive`, masks come from
// the AMR tables and the local operators skip their own communication (the tables do the assembly);
// for the uniform reference, terra's standard masks and additive communication are used and the AMR
// tables are empty (constraint/exchange are no-ops) -- the wrapper then reduces to the pure post-
// assembly Dirichlet projection, so BOTH sides solve literally the same kind of system.
static SolveResult solve_eps( const DistributedDomain& dom, const TwoToOneTables& t, bool adaptive,
                              const std::shared_ptr< util::Table >& table, const std::string& tag )
{
    using ScalarType  = double;
    using Epsilon     = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarType, 3 >;
    using Mass        = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;
    using Wrapper     = AdaptiveConstrainedOperator< Epsilon >;
    using MassWrapper = AdaptiveConstrainedOperator< Mass >;

    const auto dt = upload_2to1_tables( t );

    const auto mask  = adaptive ? adaptive_node_ownership_mask( dom, t )
                                : grid::setup_node_ownership_mask_data( dom );
    const auto bmask = adaptive ? adaptive_boundary_mask( dom )
                                : grid::shell::setup_boundary_mask_data( dom );
    const auto comm  = adaptive ? linalg::OperatorCommunicationMode::SkipCommunication
                                : linalg::OperatorCommunicationMode::CommunicateAdditively;

    VectorQ1Vec< ScalarType >    u( "u", dom, mask );
    VectorQ1Vec< ScalarType >    g( "g", dom, mask );
    VectorQ1Vec< ScalarType >    tmp( "tmp", dom, mask );
    VectorQ1Vec< ScalarType >    Adiagg( "Adiagg", dom, mask );
    VectorQ1Vec< ScalarType >    solution( "solution", dom, mask );
    VectorQ1Vec< ScalarType >    error( "error", dom, mask );
    VectorQ1Vec< ScalarType >    b( "b", dom, mask );
    VectorQ1Vec< ScalarType >    rr( "rr", dom, mask );
    VectorQ1Vec< ScalarType >    scratch_a( "scratch_a", dom, mask );
    VectorQ1Vec< ScalarType >    scratch_n( "scratch_n", dom, mask );
    VectorQ1Vec< ScalarType >    scratch_m( "scratch_m", dom, mask );
    VectorQ1Scalar< ScalarType > k( "k", dom, mask );

    const auto ndofs = kernels::common::count_masked< long >( mask, grid::NodeOwnershipFlag::OWNED );

    const auto coords  = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom );
    const auto radii_g = grid::shell::subdomain_shell_radii< ScalarType >( dom );

    Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          KInterpolator( coords, radii_g, k.grid_data() ) );
    Kokkos::parallel_for( "sol", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator( coords, radii_g, solution.grid_data(), bmask, false ) );
    Kokkos::parallel_for( "g", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator( coords, radii_g, g.grid_data(), bmask, true ) );
    Kokkos::parallel_for( "rhs", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          RHSInterpolator( coords, radii_g, tmp.grid_data() ) );
    Kokkos::fence();

    // Neumann-assembled operator; Dirichlet is imposed on the ASSEMBLED-CONSTRAINED system by the
    // wrapper's projection (see the file header note: element-level elimination is inconsistent with
    // hanging nodes whose parents lie on the boundary).
    BoundaryConditions bcs_nn = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };

    Epsilon A_loc( dom, coords, radii_g, bmask, k.grid_data(), bcs_nn, false,
                   linalg::OperatorApplyMode::Replace, comm );
    Mass    M_loc( dom, coords, radii_g, false, linalg::OperatorApplyMode::Replace, comm );

    Wrapper     A_sys( A_loc, dt, scratch_a, bmask, grid::shell::ShellBoundaryFlag::BOUNDARY );
    Wrapper     A_neu( A_loc, dt, scratch_n );
    MassWrapper M_c( M_loc, dt, scratch_m );

    // rhs: b = C^T M C f, lifted by the boundary data g, identity rows on the Dirichlet DoFs
    linalg::apply( M_c, tmp, b );
    linalg::apply( A_neu, g, tmp );
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );
    kernels::common::assign_masked_else_keep_old( b.grid_data(), g.grid_data(), bmask,
                                                  grid::shell::ShellBoundaryFlag::BOUNDARY );
    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 4000, 1e-12, 1e-12 };
    linalg::solvers::PCG< Wrapper >            pcg( solver_params, table, { tmp, Adiagg, error, rr } );
    pcg.set_tag( tag );
    linalg::solvers::solve( pcg, A_sys, u, b );
    Kokkos::fence();

    apply_constraint_device( dt, u.grid_data() ); // final conformity (no-op without hanging nodes)

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const double l2 = std::sqrt( dot( error, error ) / static_cast< double >( ndofs ) );

    linalg::apply( A_sys, u, rr );
    linalg::lincomb( rr, { 1.0, -1.0 }, { b, rr } );
    const double relres = std::sqrt( dot( rr, rr ) / dot( b, b ) );

    std::printf( "  [%s] dofs %ld  l2 %.12e  relres %.3e  hanging %zu\n", tag.c_str(), ndofs, l2,
                 relres, t.con_np.size() );
    return SolveResult{ l2, ndofs, relres, t.con_np.size() };
}

// Distributed (multi-rank) adaptive solve. Same physics/BCs as solve_eps; the operator is the
// distributed constrained wrapper (local constraint + element apply + cross-rank additive halo), and
// masks/ownership are the distributed ones. Degenerates to the single-rank adaptive solve at np=1.
static SolveResult solve_eps_distributed( const DistributedAdaptiveMesh&        mesh,
                                          const std::shared_ptr< util::Table >& table,
                                          const std::string&                    tag )
{
    using ScalarType  = double;
    using Epsilon     = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarType, 3 >;
    using Mass        = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;
    using Wrapper     = AdaptiveDistributedConstrainedOperator< Epsilon >;
    using MassWrapper = AdaptiveDistributedConstrainedOperator< Mass >;

    const auto& dom   = mesh.domain;
    const auto  mask  = distributed_ownership_mask( mesh );
    const auto  bmask = adaptive_boundary_mask( dom );

    VectorQ1Vec< ScalarType >    u( "u", dom, mask ), g( "g", dom, mask ), tmp( "tmp", dom, mask );
    VectorQ1Vec< ScalarType >    Adiagg( "Adiagg", dom, mask ), solution( "solution", dom, mask );
    VectorQ1Vec< ScalarType >    error( "error", dom, mask ), b( "b", dom, mask ), rr( "rr", dom, mask );
    VectorQ1Vec< ScalarType >    scratch_a( "sa", dom, mask ), scratch_n( "sn", dom, mask ), scratch_m( "sm", dom, mask );
    VectorQ1Scalar< ScalarType > k( "k", dom, mask );

    const auto ndofs = kernels::common::count_masked< long >( mask, grid::NodeOwnershipFlag::OWNED, mesh.comm );

    const auto coords  = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom );
    const auto radii_g = grid::shell::subdomain_shell_radii< ScalarType >( dom );

    Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          KInterpolator( coords, radii_g, k.grid_data() ) );
    Kokkos::parallel_for( "sol", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator( coords, radii_g, solution.grid_data(), bmask, false ) );
    Kokkos::parallel_for( "g", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator( coords, radii_g, g.grid_data(), bmask, true ) );
    Kokkos::parallel_for( "rhs", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          RHSInterpolator( coords, radii_g, tmp.grid_data() ) );
    Kokkos::fence();

    BoundaryConditions bcs_nn = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };
    Epsilon A_loc( dom, coords, radii_g, bmask, k.grid_data(), bcs_nn, false,
                   linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::SkipCommunication );
    Mass    M_loc( dom, coords, radii_g, false, linalg::OperatorApplyMode::Replace,
                linalg::OperatorCommunicationMode::SkipCommunication );

    Wrapper     A_sys( A_loc, mesh, scratch_a, bmask, grid::shell::ShellBoundaryFlag::BOUNDARY );
    Wrapper     A_neu( A_loc, mesh, scratch_n );
    MassWrapper M_c( M_loc, mesh, scratch_m );

    linalg::apply( M_c, tmp, b );
    linalg::apply( A_neu, g, tmp );
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );
    kernels::common::assign_masked_else_keep_old( b.grid_data(), g.grid_data(), bmask,
                                                  grid::shell::ShellBoundaryFlag::BOUNDARY );
    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 4000, 1e-12, 1e-12 };
    linalg::solvers::PCG< Wrapper >            pcg( solver_params, table, { tmp, Adiagg, error, rr } );
    pcg.set_tag( tag );
    linalg::solvers::solve( pcg, A_sys, u, b );
    Kokkos::fence();

    apply_constraint_device( mesh.t_local_d, u.grid_data() ); // final local conformity

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const double l2 = std::sqrt( dot( error, error ) / static_cast< double >( ndofs ) );
    linalg::apply( A_sys, u, rr );
    linalg::lincomb( rr, { 1.0, -1.0 }, { b, rr } );
    const double relres = std::sqrt( dot( rr, rr ) / dot( b, b ) );

    if ( mpi::rank( mesh.comm ) == 0 )
        std::printf( "  [%s] dofs %ld  l2 %.12e  relres %.3e  hanging(local r0) %zu\n", tag.c_str(),
                     ndofs, l2, relres, mesh.t_local.con_np.size() );
    return SolveResult{ l2, ndofs, relres, mesh.t_local.con_np.size() };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        auto table = std::make_shared< util::Table >();

        const std::vector< double > radii5 = { 0.5, 0.625, 0.75, 0.875, 1.0 };
        std::vector< double >       radii9;
        for ( std::size_t i = 0; i + 1 < radii5.size(); ++i )
        {
            radii9.push_back( radii5[i] );
            radii9.push_back( 0.5 * ( radii5[i] + radii5[i + 1] ) );
        }
        radii9.push_back( radii5.back() );

        const int nprocs = mpi::num_processes( MPI_COMM_WORLD );

        // ---- T1: all-refined adaptive (DISTRIBUTED) == uniform reference ---------------------------
        {
            const int      LDR = 3, S_lat = 2, S_rad = 1, M = 1;
            AdaptiveForest f( M, S_lat, S_rad );
            const auto     base = f.leaves(); // copy: refine() mutates the leaf set
            f.refine( base );
            CHECK( f.validate() );

            auto mesh_a = build_distributed_mesh( MPI_COMM_WORLD, LDR, radii5, f,
                                                  grid::shell::subdomain_to_rank_by_diamond );
            CHECK( mesh_a.t_local.con_np.empty() ); // uniformly subdivided: conforming everywhere
            const auto res_a = solve_eps_distributed( mesh_a, table, "T1_adaptive_all_refined" );

            const auto     dom_u = DistributedDomain::create_uniform( LDR + 1, radii9, 2, 1 );
            TwoToOneTables t_empty;
            t_empty.cls_offsets.push_back( 0 );
            const auto res_u = solve_eps( dom_u, t_empty, false, table, "T1_uniform_reference" );

            CHECK( res_a.ndofs == res_u.ndofs ); // same physical node set
            CHECK( res_a.relres < 1e-6 );
            CHECK( res_u.relres < 1e-6 );
            CHECK( res_a.l2 < 0.1 && res_u.l2 < 0.1 );
            CHECK( std::fabs( res_a.l2 - res_u.l2 ) < 1e-5 * res_u.l2 ); // identical discretization
        }

        // ---- T2: locally refined adaptive (DISTRIBUTED), hanging nodes active in the solve ---------
        {
            const int      LDR = 4, S_lat = 4, S_rad = 1, M = 1;
            AdaptiveForest f( M, S_lat, S_rad );
            f.refine( { ForestLeaf{ SubdomainInfo{ 0, 1, 1, 0 }, 0 } } ); // non-corner block
            f.balance_2to1();
            CHECK( f.validate() );

            auto       mesh_a = build_distributed_mesh( MPI_COMM_WORLD, LDR, radii5, f,
                                                        grid::shell::subdomain_to_rank_by_diamond );
            long       hanging_local = (long) mesh_a.t_local.con_np.size(), hanging_total = 0;
            MPI_Allreduce( &hanging_local, &hanging_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD );
            CHECK( hanging_total > 0 ); // 2:1 interfaces present somewhere
            const auto res_a = solve_eps_distributed( mesh_a, table, "T2_adaptive_refined_block" );

            const auto     dom_u = DistributedDomain::create_uniform( LDR, radii5, 2, 0 );
            TwoToOneTables t_empty;
            t_empty.cls_offsets.push_back( 0 );
            const auto res_u = solve_eps( dom_u, t_empty, false, table, "T2_uniform_base" );

            CHECK( res_a.relres < 1e-6 );
            CHECK( res_u.relres < 1e-6 );
            CHECK( res_a.l2 < 0.1 && res_u.l2 < 0.1 );
            CHECK( res_a.l2 <= res_u.l2 * 1.0001 ); // locally finer mesh must not be worse
        }

        if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
            std::printf( "  (ran on %d rank%s)\n", nprocs, nprocs == 1 ? "" : "s" );
    }

    const bool ok = ( g_failures == 0 );
    if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
    {
        if ( ok )
            std::printf( "test_adaptive_epsilon_divdiv_gpu: ALL PASS (%d checks)\n", g_checks );
        else
            std::printf( "test_adaptive_epsilon_divdiv_gpu: %d/%d FAILURE(S)\n", g_failures, g_checks );
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
