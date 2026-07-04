// Viscosity tanh-kink study: uniform vs adaptive refinement as the kink STEEPNESS increases.
//
// Manufactured problem for the EpsilonDivDivKerngen operator with a RADIAL tanh kink in the viscosity at
// RK=0.75 (contrast 10, fixed) and a co-located radial velocity pulse, both sharpening with the steepness
// S (a runtime parameter; RHS f = -div(2k(eps(u)-1/3 div u I)) is CAS-generated with S kept symbolic in
// mms_visckink_rhs.inc). For each S we run:
//   UNIFORM   -- refine EVERY leaf each level (h halves everywhere),
//   ADAPTIVE  -- refine only blocks flagged by the VISCOSITY-gradient indicator (max_K k - min_K k),
// and record BOTH the solver convergence and the mass-weighted discrete L2 error ||u_h - u||.
//
// The solver is the ADAPTIVE LDR-multigrid preconditioner (see test_adaptive_mg_gpu) inside FGMRES: each
// mesh gets its own LDR hierarchy (min_ldr..LDR on the SAME forest, coarser intra-block grids), Chebyshev
// smoothers, a coarse PCG, and the C_coarse/C_fine-constrained transfer operators. So "solver convergence
// vs steepness" here measures whether the MG preconditioner stays robust as the viscosity kink sharpens
// (a real question -- sharp coefficients stress geometric MG), not just unpreconditioned CG tracking dofs.
//
// S_lat=2 so per-block nx = 2^LDR/S_lat+1 >= 3 at every MG level (LDR 2 and 3): a 2:1 hanging interface
// needs an interior midpoint, so nx>=3 (see the guard in adaptive_exchange.hpp).
//
// Needs a GPU -- run on an H100 node.

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/multigrid.hpp"
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
#include <map>
#include <memory>
#include <string>
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

// radial tanh kink -- match mms_visckink_gen.py
static constexpr double DX = 0.8451, DY = 0.5071, DZ = 0.1690; // fixed flow direction (unit)
static constexpr double R0 = 0.5, R1 = 1.0, RK = 0.75, AMP = 1.0;
static constexpr double KLO = 1.0, KHI = 10.0; // viscosity below / above the kink (contrast 10)

static int g_failures = 0, g_checks = 0;
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

using Epsilon = fe::wedge::operators::shell::EpsilonDivDivKerngen< double, 3 >;
using Wrapper = AdaptiveDistributedConstrainedOperator< Epsilon >;

// exact solution: fixed-direction field with a CO-LOCATED velocity kink at RK -- a steep tanh transition at
// the viscosity jump (curvature => error peaks at RK +/- 0.66/S, inside the k-gradient band we refine).
struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;
    double                                             S_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const bool on_b = util::has_flag( mask_( s, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );
        if ( !only_boundary_ || on_b )
        {
            const double rn  = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
            const double env = Kokkos::tanh( S_ * ( rn - RK ) ); // co-located velocity kink at the viscosity jump
            const double amp = AMP * env;
            data_( s, x, y, r, 0 ) = amp * DX;
            data_( s, x, y, r, 1 ) = amp * DY;
            data_( s, x, y, r, 2 ) = amp * DZ;
        }
    }
};

// viscosity: smooth radial tanh kink at RK, KLO below -> KHI above, steepness S.
struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    double                     S_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c  = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const double                  rn = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
        data_( s, x, y, r )              = KLO + ( KHI - KLO ) * 0.5 * ( 1.0 + Kokkos::tanh( S_ * ( rn - RK ) ) );
    }
};

struct RHSInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    double                     S_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const double                  cx = coords( 0 ), cy = coords( 1 ), cz = coords( 2 );
        const double                  S = S_;
#include "mms_visckink_rhs.inc"
    }
};

// ---- adaptive LDR transfer operators (exact transpose pair; constrain BOTH coarse and fine ends) -------
class AdaptiveProlongation
{
  public:
    using SrcVectorType = VectorQ1Vec< double, 3 >;
    using DstVectorType = VectorQ1Vec< double, 3 >;
    using ScalarType    = double;

    AdaptiveProlongation( const DistributedAdaptiveMesh& fine, const DistributedAdaptiveMesh& coarse,
                          linalg::OperatorApplyMode mode = linalg::OperatorApplyMode::Add )
    : geom_( mode )
    , fine_( &fine )
    , coarse_( &coarse )
    , coarse_mask_( adaptive_ownership_mask( coarse ) )
    , coarse_tmp_( "prol_cc_tmp", coarse.domain, coarse_mask_ )
    {}

    void apply_impl( const SrcVectorType& coarse, DstVectorType& fine )
    {
        amr_deep_copy( coarse_tmp_.grid_data(), coarse.grid_data() );
        apply_constraint_device( coarse_->t_local_d, coarse_tmp_.grid_data() ); // C_coarse
        geom_.apply_impl( coarse_tmp_, fine );                                  // scatter
        apply_constraint_device( fine_->t_local_d, fine.grid_data() );          // C_fine
    }

  private:
    fe::wedge::operators::shell::ProlongationVecConstant< double, 3 > geom_;
    const DistributedAdaptiveMesh*                                    fine_;
    const DistributedAdaptiveMesh*                                    coarse_;
    Grid4DDataScalar< grid::NodeOwnershipFlag >                       coarse_mask_;
    VectorQ1Vec< double, 3 >                                          coarse_tmp_;
};

class AdaptiveRestriction
{
  public:
    using SrcVectorType = VectorQ1Vec< double, 3 >;
    using DstVectorType = VectorQ1Vec< double, 3 >;
    using ScalarType    = double;

    AdaptiveRestriction( const DistributedAdaptiveMesh& coarse, const DistributedAdaptiveMesh& fine )
    : geom_( coarse.domain, linalg::OperatorApplyMode::Replace )
    , coarse_( &coarse )
    , fine_( &fine )
    , fine_mask_( adaptive_ownership_mask( fine ) )
    , fine_tmp_( "rest_ct_tmp", fine.domain, fine_mask_ )
    {}

    void apply_impl( const SrcVectorType& fine, DstVectorType& coarse )
    {
        amr_deep_copy( fine_tmp_.grid_data(), fine.grid_data() );
        apply_constraint_transpose_device( fine_->t_local_d, fine_tmp_.grid_data() ); // C_fine^T
        geom_.apply_impl( fine_tmp_, coarse );                                         // gather
        assemble_distributed( *coarse_, coarse.grid_data() );                          // coarse class-sum
    }

  private:
    fe::wedge::operators::shell::RestrictionVecConstant< double, 3 > geom_;
    const DistributedAdaptiveMesh*                                  coarse_;
    const DistributedAdaptiveMesh*                                  fine_;
    Grid4DDataScalar< grid::NodeOwnershipFlag >                     fine_mask_;
    VectorQ1Vec< double, 3 >                                        fine_tmp_;
};

// Inverse-diagonal (Jacobi) preconditioner for the MG coarse solver (LDR=2 level). Without it the coarse
// PCG is unpreconditioned and dominates the runtime on the big meshes.
template < linalg::OperatorLike OperatorT >
struct InverseDiagonalPreconditioner
{
    using OperatorType       = OperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;
    SolutionVectorType inv_diag_;
    explicit InverseDiagonalPreconditioner( const SolutionVectorType& d ) : inv_diag_( d ) {}
    void solve_impl( OperatorType& /*A*/, SolutionVectorType& x, const RHSVectorType& b )
    {
        linalg::assign( x, b );
        linalg::scale_in_place( x, inv_diag_ );
    }
};

struct Res
{
    long                  dofs = 0;
    int                   iters = 0;
    double                l2 = 0, relres = 0;
    std::vector< double > indicator; // per block: max_K k - min_K k (viscosity gradient)
};

// Build an LDR-multigrid on `forest` (levels min_ldr..ldr_fine, same forest at coarser intra-block grids),
// use it as the FGMRES preconditioner for the manufactured tanh-kink problem at steepness S, and report
// the FGMRES iteration count + discrete L2 error + per-block viscosity indicator (drives refinement).
static Res solve_level( MPI_Comm comm, const AdaptiveForest& forest, int ldr_fine, int S_rad, double S,
                        const std::shared_ptr< util::Table >& table, const std::string& tag )
{
    using Mass  = fe::wedge::operators::shell::VectorMass< double, 3 >;
    using MassW = AdaptiveDistributedConstrainedOperator< Mass >;

    const int min_ldr = 2;
    const int nlev    = ldr_fine - min_ldr + 1;
    const int F       = nlev - 1;
    auto      radii_at = [&]( int L ) {
        return grid::shell::uniform_shell_radii< double >( 0.5, 1.0, S_rad * ( 1 << ( min_ldr + L ) ) + 1 );
    };

    // ---- per-level hierarchy (stable storage) ----------------------------------------------------------
    std::vector< DistributedAdaptiveMesh >                           mesh;
    std::vector< Grid3DDataVec< double, 3 > >                        coords;
    std::vector< Grid2DDataScalar< double > >                        radii_g;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >       mask;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > bmask;
    std::vector< VectorQ1Scalar< double > >                          k;
    mesh.reserve( nlev );
    coords.reserve( nlev ); radii_g.reserve( nlev ); mask.reserve( nlev ); bmask.reserve( nlev ); k.reserve( nlev );
    for ( int L = 0; L < nlev; ++L )
    {
        mesh.push_back( build_distributed_adaptive_mesh( comm, min_ldr + L, radii_at( L ), forest,
                                                         grid::shell::subdomain_to_rank_by_diamond ) );
        coords.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< double >( mesh[L].domain ) );
        radii_g.push_back( grid::shell::subdomain_shell_radii< double >( mesh[L].domain ) );
        mask.push_back( adaptive_ownership_mask( mesh[L] ) );
        bmask.push_back( adaptive_boundary_mask( mesh[L].domain ) );
        k.emplace_back( "k_" + std::to_string( L ), mesh[L].domain, mask[L] );
        Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( mesh[L].domain ),
                              KInterpolator{ coords[L], radii_g[L], k[L].grid_data(), S } );
    }
    Kokkos::fence();

    BoundaryConditions bcs_nn = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };
    std::vector< std::unique_ptr< Epsilon > > A_loc;
    std::vector< VectorQ1Vec< double > >      scratch;
    std::vector< std::unique_ptr< Wrapper > > A;
    A_loc.reserve( nlev ); scratch.reserve( nlev ); A.reserve( nlev );
    for ( int L = 0; L < nlev; ++L )
    {
        A_loc.push_back( std::make_unique< Epsilon >(
            mesh[L].domain, coords[L], radii_g[L], bmask[L], k[L].grid_data(), bcs_nn, false,
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::SkipCommunication ) );
        scratch.emplace_back( "sa_" + std::to_string( L ), mesh[L].domain, mask[L] );
        A.push_back( std::make_unique< Wrapper >( *A_loc[L], mesh[L], scratch[L], bmask[L],
                                                  grid::shell::ShellBoundaryFlag::BOUNDARY ) );
    }

    // Inverse diagonal for the Chebyshev smoother. Take the block-local element diagonal and CLASS-SUM only
    // (apply_class_sum_device) -- do NOT fold the hanging DoFs' block diagonal into their parents. The old
    // path (assemble_distributed) folded them with the linear constraint weight, which inflates a parent's
    // diagonal by ~Σ w·A_hh; A_hh scales with the local viscosity, so in the k=10 kink band the parent
    // diagonal was ~10x too large, mis-scaling Chebyshev right at the 2:1 interface (the coefficient-amplified
    // adaptive penalty). The true diag(C^T A C) parent self-term is w^2 and is largely cancelled by the
    // negative parent-hanging cross term, so the parent's own assembled A_pp is far closer. Uniform meshes are
    // unchanged (no hanging => class-sum-only == assemble_distributed).
    std::vector< VectorQ1Vec< double > > inv_diag;
    inv_diag.reserve( nlev );
    for ( int L = 0; L < nlev; ++L )
    {
        inv_diag.emplace_back( "invd_" + std::to_string( L ), mesh[L].domain, mask[L] );
        VectorQ1Vec< double > ones( "ones_" + std::to_string( L ), mesh[L].domain, mask[L] );
        linalg::assign( ones, 1.0 );
        A_loc[L]->set_diagonal( true );
        linalg::apply( *A_loc[L], ones, inv_diag[L] );
        A_loc[L]->set_diagonal( false );
        assemble_distributed( mesh[L], inv_diag[L].grid_data() );
        kernels::common::assign_masked_else_keep_old( inv_diag[L].grid_data(), 1.0, bmask[L],
                                                      grid::shell::ShellBoundaryFlag::BOUNDARY );
        linalg::invert_entries( inv_diag[L] );
    }

    using Smoother = linalg::solvers::Chebyshev< Wrapper >;
    std::vector< Smoother > smoothers;
    smoothers.reserve( nlev );
    std::vector< std::vector< VectorQ1Vec< double > > > cheby_tmps( nlev );
    for ( int L = 0; L < nlev; ++L )
    {
        cheby_tmps[L].emplace_back( "ct0_" + std::to_string( L ), mesh[L].domain, mask[L] );
        cheby_tmps[L].emplace_back( "ct1_" + std::to_string( L ), mesh[L].domain, mask[L] );
        smoothers.emplace_back( /*order=*/8, inv_diag[L], cheby_tmps[L], /*prepost=*/6 ); // DISCRIMINATOR: strong smoother
    }

    std::vector< AdaptiveProlongation > P;
    std::vector< AdaptiveRestriction >  R;
    P.reserve( nlev - 1 ); R.reserve( nlev - 1 );
    for ( int L = 0; L < nlev - 1; ++L )
    {
        P.emplace_back( mesh[L + 1], mesh[L] );
        R.emplace_back( mesh[L], mesh[L + 1] );
    }

    std::vector< VectorQ1Vec< double > > tmp_r, tmp_e, tmp_all;
    tmp_r.reserve( nlev - 1 ); tmp_e.reserve( nlev - 1 ); tmp_all.reserve( nlev );
    for ( int L = 0; L < nlev; ++L )
        tmp_all.emplace_back( "tmp_" + std::to_string( L ), mesh[L].domain, mask[L] );
    for ( int L = 0; L < nlev - 1; ++L )
    {
        tmp_r.emplace_back( "tmpr_" + std::to_string( L ), mesh[L].domain, mask[L] );
        tmp_e.emplace_back( "tmpe_" + std::to_string( L ), mesh[L].domain, mask[L] );
    }

    std::vector< Wrapper > A_c;
    A_c.reserve( nlev - 1 );
    for ( int L = 0; L < nlev - 1; ++L )
        A_c.push_back( *A[L] );

    std::vector< VectorQ1Vec< double > > coarse_tmps;
    for ( int i = 0; i < 4; ++i )
        coarse_tmps.emplace_back( "cst_" + std::to_string( i ), mesh[0].domain, mask[0] );
    auto coarse_table = std::make_shared< util::Table >();
    linalg::solvers::IterativeSolverParameters coarse_params{ 300, 1e-8, 1e-14 };
    using CoarsePrec = InverseDiagonalPreconditioner< Wrapper >;
    linalg::solvers::PCG< Wrapper, CoarsePrec > coarse_solver( coarse_params, coarse_table, coarse_tmps,
                                                               CoarsePrec( inv_diag[0] ) );
    coarse_solver.set_tag( "mg_coarse" );

    using MG = linalg::solvers::Multigrid< Wrapper, AdaptiveProlongation, AdaptiveRestriction, Smoother,
                                           linalg::solvers::PCG< Wrapper, CoarsePrec > >;
    MG mg( P, R, A_c, tmp_r, tmp_e, tmp_all, smoothers, smoothers, coarse_solver, 1, 1e-16 );

    // ---- manufactured RHS on the finest level (Dirichlet lifting) --------------------------------------
    const auto& dom = mesh[F].domain;
    using Mass_  = Mass;
    VectorQ1Vec< double > u( "u", dom, mask[F] ), g( "g", dom, mask[F] ), tmp( "tmp", dom, mask[F] );
    VectorQ1Vec< double > b( "b", dom, mask[F] ), soln( "soln", dom, mask[F] ), err( "err", dom, mask[F] );
    VectorQ1Vec< double > Merr( "Merr", dom, mask[F] ), rr( "rr", dom, mask[F] );
    VectorQ1Vec< double > scr_m( "scrm", dom, mask[F] ), scr_n( "scrn", dom, mask[F] );

    Kokkos::parallel_for( "sol", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator{ coords[F], radii_g[F], soln.grid_data(), bmask[F], false, S } );
    Kokkos::parallel_for( "g", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator{ coords[F], radii_g[F], g.grid_data(), bmask[F], true, S } );
    Kokkos::parallel_for( "rhs", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          RHSInterpolator{ coords[F], radii_g[F], tmp.grid_data(), S } );
    Kokkos::fence();

    Mass_   M_loc( dom, coords[F], radii_g[F], false, linalg::OperatorApplyMode::Replace,
                 linalg::OperatorCommunicationMode::SkipCommunication );
    MassW   M_c( M_loc, mesh[F], scr_m );
    Wrapper A_neu( *A_loc[F], mesh[F], scr_n );
    linalg::apply( M_c, tmp, b );
    linalg::apply( A_neu, g, tmp );
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );
    kernels::common::assign_masked_else_keep_old( b.grid_data(), g.grid_data(), bmask[F],
                                                  grid::shell::ShellBoundaryFlag::BOUNDARY );
    Kokkos::fence();

    // ---- MG-preconditioned FGMRES (flexible: the MG coarse solver is an inner Krylov -> non-linear) ----
    linalg::assign( u, 0.0 );
    constexpr int restart = 50; // 3-level MG converges fast; smaller basis => memory headroom on LDR4 meshes
    linalg::solvers::FGMRESOptions< double > opts;
    opts.restart                     = restart;
    opts.max_iterations              = 150;
    opts.relative_residual_tolerance = 1e-9;
    opts.absolute_residual_tolerance = 1e-14;
    std::vector< VectorQ1Vec< double > > fg;
    for ( int i = 0; i < 2 * restart + 4; ++i )
        fg.emplace_back( "fg" + std::to_string( i ), dom, mask[F] );
    linalg::solvers::FGMRES< Wrapper, MG > fgmres( fg, opts, table, mg );
    fgmres.set_tag( tag );
    linalg::solvers::solve( fgmres, *A[F], u, b );
    Kokkos::fence();
    apply_constraint_device( mesh[F].t_local_d, u.grid_data() );

    // ---- error, residual, viscosity indicator ---------------------------------------------------------
    linalg::lincomb( err, { 1.0, -1.0 }, { u, soln } );
    linalg::apply( M_c, err, Merr );
    const double l2 = std::sqrt( std::fabs( dot( Merr, err ) ) );
    linalg::apply( *A[F], u, rr );
    linalg::lincomb( rr, { 1.0, -1.0 }, { b, rr } );
    const double relres = std::sqrt( std::fabs( dot( rr, rr ) ) / std::fabs( dot( b, b ) ) );

    Res       res;
    res.dofs   = kernels::common::count_masked< long >( mask[F], grid::NodeOwnershipFlag::OWNED, comm );
    res.l2     = l2;
    res.relres = relres;
    res.iters  = static_cast< int >( table->query_rows_equals( "tag", tag ).rows().size() ) - 1;

    const int nsub = static_cast< int >( dom.subdomains().size() );
    auto      kh   = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, k[F].grid_data() );
    const int nx = kh.extent( 1 ), ny = kh.extent( 2 ), nr = kh.extent( 3 );
    res.indicator.assign( nsub, 0.0 );
    for ( int s = 0; s < nsub; ++s )
    {
        double mx = -1e300, mn = 1e300;
        for ( int i = 0; i < nx; ++i )
            for ( int j = 0; j < ny; ++j )
                for ( int m = 0; m < nr; ++m )
                {
                    const double v = kh( s, i, j, m );
                    mx = std::max( mx, v );
                    mn = std::min( mn, v );
                }
        res.indicator[s] = mx - mn;
    }
    if ( mpi::rank( comm ) == 0 )
        std::printf( "    [%-16s] dofs %9ld  FGMRES %4d  L2 %.6e  relres %.2e  hanging %zu\n", tag.c_str(),
                     res.dofs, res.iters, l2, relres, mesh[F].t_local.con_dst.size() );
    return res;
}

// NAKED-CG DIAGNOSTIC: same manufactured tanh-kink problem, but solve the finest constrained operator with
// UNPRECONDITIONED CG (no multigrid). Compares the operator's intrinsic conditioning on a UNIFORM vs an
// ADAPTIVE (2:1 hanging) mesh at similar size. If naked CG on the adaptive mesh needs ~the same iters as
// uniform at similar dofs, the ~2x penalty lives in the MG (transfer/coarse correction), not the operator.
static Res solve_naked( MPI_Comm comm, const AdaptiveForest& forest, int ldr, int S_rad, double S,
                        const std::shared_ptr< util::Table >& table, const std::string& tag )
{
    using Mass  = fe::wedge::operators::shell::VectorMass< double, 3 >;
    using MassW = AdaptiveDistributedConstrainedOperator< Mass >;

    auto radii = grid::shell::uniform_shell_radii< double >( 0.5, 1.0, S_rad * ( 1 << ldr ) + 1 );
    auto mesh  = build_distributed_adaptive_mesh( comm, ldr, radii, forest,
                                                  grid::shell::subdomain_to_rank_by_diamond );
    const auto& dom     = mesh.domain;
    auto        coords  = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom );
    auto        radii_g = grid::shell::subdomain_shell_radii< double >( dom );
    auto        mask    = adaptive_ownership_mask( mesh );
    auto        bmask   = adaptive_boundary_mask( dom );

    VectorQ1Scalar< double > k( "k", dom, mask );
    Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          KInterpolator{ coords, radii_g, k.grid_data(), S } );
    Kokkos::fence();

    BoundaryConditions bcs_nn = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };
    Epsilon A_loc( dom, coords, radii_g, bmask, k.grid_data(), bcs_nn, false,
                   linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::SkipCommunication );

    VectorQ1Vec< double > u( "u", dom, mask ), g( "g", dom, mask ), tmp( "tmp", dom, mask );
    VectorQ1Vec< double > b( "b", dom, mask ), soln( "soln", dom, mask ), err( "err", dom, mask );
    VectorQ1Vec< double > Merr( "Merr", dom, mask ), rr( "rr", dom, mask ), adg( "adg", dom, mask );
    VectorQ1Vec< double > sa( "sa", dom, mask ), sn( "sn", dom, mask ), sm( "sm", dom, mask );

    Kokkos::parallel_for( "sol", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator{ coords, radii_g, soln.grid_data(), bmask, false, S } );
    Kokkos::parallel_for( "g", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          SolutionInterpolator{ coords, radii_g, g.grid_data(), bmask, true, S } );
    Kokkos::parallel_for( "rhs", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          RHSInterpolator{ coords, radii_g, tmp.grid_data(), S } );
    Kokkos::fence();

    Mass    M_loc( dom, coords, radii_g, false, linalg::OperatorApplyMode::Replace,
                   linalg::OperatorCommunicationMode::SkipCommunication );
    MassW   M_c( M_loc, mesh, sm );
    Wrapper A_neu( A_loc, mesh, sn );
    Wrapper A_sys( A_loc, mesh, sa, bmask, grid::shell::ShellBoundaryFlag::BOUNDARY );
    linalg::apply( M_c, tmp, b );
    linalg::apply( A_neu, g, tmp );
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );
    kernels::common::assign_masked_else_keep_old( b.grid_data(), g.grid_data(), bmask,
                                                  grid::shell::ShellBoundaryFlag::BOUNDARY );
    Kokkos::fence();

    // ---- naked (unpreconditioned) CG ----
    linalg::assign( u, 0.0 );
    linalg::solvers::IterativeSolverParameters params{ 20000, 1e-8, 1e-14 };
    linalg::solvers::PCG< Wrapper >            pcg( params, table, { tmp, adg, err, rr } );
    pcg.set_tag( tag );
    linalg::solvers::solve( pcg, A_sys, u, b );
    Kokkos::fence();
    apply_constraint_device( mesh.t_local_d, u.grid_data() );

    linalg::lincomb( err, { 1.0, -1.0 }, { u, soln } );
    linalg::apply( M_c, err, Merr );
    const double l2 = std::sqrt( std::fabs( dot( Merr, err ) ) );
    linalg::apply( A_sys, u, rr );
    linalg::lincomb( rr, { 1.0, -1.0 }, { b, rr } );
    const double relres = std::sqrt( std::fabs( dot( rr, rr ) ) / std::fabs( dot( b, b ) ) );

    Res res;
    res.dofs   = kernels::common::count_masked< long >( mask, grid::NodeOwnershipFlag::OWNED, comm );
    res.l2     = l2;
    res.relres = relres;
    res.iters  = static_cast< int >( table->query_rows_equals( "tag", tag ).rows().size() ) - 1;

    const int nsub = static_cast< int >( dom.subdomains().size() );
    auto      kh   = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, k.grid_data() );
    const int nx = kh.extent( 1 ), ny = kh.extent( 2 ), nr = kh.extent( 3 );
    res.indicator.assign( nsub, 0.0 );
    for ( int s = 0; s < nsub; ++s )
    {
        double mx = -1e300, mn = 1e300;
        for ( int i = 0; i < nx; ++i )
            for ( int j = 0; j < ny; ++j )
                for ( int m = 0; m < nr; ++m )
                {
                    const double v = kh( s, i, j, m );
                    mx = std::max( mx, v );
                    mn = std::min( mn, v );
                }
        res.indicator[s] = mx - mn;
    }
    if ( mpi::rank( comm ) == 0 )
        std::printf( "    [%-16s] dofs %9ld  CG %6d  L2 %.6e  relres %.2e  hanging %zu\n", tag.c_str(),
                     res.dofs, res.iters, l2, relres, mesh.t_local.con_dst.size() );
    return res;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        auto table = std::make_shared< util::Table >();

        const int    LDR = 3, S_lat = 2, S_rad = 2, M = 8; // smaller grid for the naked-CG diagnostic (single level)
        const int    NU = 2;   // uniform levels 0..NU (L2=subdiv2: the fair same-resolution baseline for R2)
        const int    NA = 2;   // adaptive rounds 0..NA (R2 subdiv2 is localized, stays tractable)
        const double FRAC = 0.30;
        const std::vector< double > steepness = { 8.0 }; // NAKED-CG DIAGNOSTIC: is the ~2x penalty in the operator or the MG?

        struct Row { double S; long u_dofs, a_dofs; int u_it, a_it; double u_l2, a_l2; };
        std::vector< Row > summary;

        for ( double S : steepness )
        {
            if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
                std::printf( "\n===== kink steepness S = %.0f =====\n", S );
            Res u_last{}, a_last{};

            // ---- UNIFORM sweep ----
            {
                AdaptiveForest f( NU, S_lat, S_rad );
                for ( int lvl = 0; lvl <= NU; ++lvl )
                {
                    u_last = solve_naked( MPI_COMM_WORLD, f, LDR, S_rad, S, table,
                                          "S" + std::to_string( (int) S ) + "_uni_L" + std::to_string( lvl ) );
                    CHECK( u_last.relres < 1e-6 );
                    if ( lvl < NU )
                    {
                        auto all = f.leaves(); // COPY: refine() mutates leaves_, so passing the ref invalidates it
                        f.refine( all );
                        f.balance_2to1();
                    }
                }
            }

            // ---- ADAPTIVE sweep (viscosity-gradient indicator, refines at the kink) ----
            {
                AdaptiveForest f( M, S_lat, S_rad );
                for ( int round = 0; round <= NA; ++round )
                {
                    auto leaves = f.leaves();
                    a_last = solve_naked( MPI_COMM_WORLD, f, LDR, S_rad, S, table,
                                          "S" + std::to_string( (int) S ) + "_ada_R" + std::to_string( round ) );
                    CHECK( a_last.relres < 1e-6 );
                    if ( round == NA )
                        break;

                    // GLOBAL refinement decision (replicated forest): scatter local indicators to global leaf
                    // order keyed by finest anchor, all-reduce, global emax.
                    std::map< SubdomainInfo, double > ind_of_anchor;
                    auto                              fine_mesh = build_distributed_adaptive_mesh(
                        MPI_COMM_WORLD, LDR, grid::shell::uniform_shell_radii< double >( 0.5, 1.0, S_rad * ( 1 << LDR ) + 1 ),
                        f, grid::shell::subdomain_to_rank_by_diamond );
                    for ( std::size_t s = 0; s < a_last.indicator.size(); ++s )
                        ind_of_anchor[fine_mesh.domain.subdomain_info_from_local_idx( static_cast< int >( s ) )] =
                            a_last.indicator[s];
                    std::vector< double > gind( leaves.size(), 0.0 );
                    for ( std::size_t i = 0; i < leaves.size(); ++i )
                    {
                        auto it = ind_of_anchor.find( f.finest_anchor( leaves[i] ) );
                        if ( it != ind_of_anchor.end() )
                            gind[i] = it->second;
                    }
                    MPI_Allreduce( MPI_IN_PLACE, gind.data(), static_cast< int >( gind.size() ), MPI_DOUBLE,
                                   MPI_SUM, MPI_COMM_WORLD );
                    double emax = 0.0;
                    for ( double e : gind )
                        emax = std::max( emax, e );
                    std::vector< ForestLeaf > to_refine;
                    for ( std::size_t i = 0; i < leaves.size(); ++i )
                        if ( leaves[i].subdivision < M && gind[i] >= FRAC * emax )
                            to_refine.push_back( leaves[i] );
                    if ( to_refine.empty() )
                        break;
                    f.refine( to_refine );
                    f.balance_2to1();
                    CHECK( f.validate() );
                }
            }

            summary.push_back( { S, u_last.dofs, a_last.dofs, u_last.iters, a_last.iters, u_last.l2, a_last.l2 } );
        }

        if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        {
            std::printf( "\n===== SUMMARY: naked CG, viscosity tanh-kink, uniform vs adaptive (finest of each) =====\n" );
            std::printf( "   S  | uniform: dofs   CG   L2err   | adaptive: dofs   CG   L2err   | L2 gain  dof ratio\n" );
            for ( const auto& r : summary )
                std::printf( "  %3.0f | %9ld %4d %.3e | %9ld %4d %.3e | %6.2fx  %5.2fx\n", r.S, r.u_dofs, r.u_it,
                             r.u_l2, r.a_dofs, r.a_it, r.a_l2, r.u_l2 / r.a_l2, (double) r.u_dofs / (double) r.a_dofs );
        }
        for ( const auto& r : summary )
        {
            CHECK( r.a_l2 <= r.u_l2 * 1.5 );            // adaptive not worse than uniform
            CHECK( r.u_it > 0 && r.u_it < 20000 );      // naked CG converged
            CHECK( r.a_it > 0 && r.a_it < 20000 );
        }
    }

    const bool ok = ( g_failures == 0 );
    if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        std::printf( "test_adaptive_visckink_gpu: %s (%d checks)\n", ok ? "ALL PASS" : "FAILURE(S)", g_checks );
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
