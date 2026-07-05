// Adaptive geometric multigrid on the REFINEMENT-LEVEL (LDR) axis, used as the A-block preconditioner.
//
// LDR-only MG (per the agreed scope): the forest/subdivisions are FIXED; each MG level is the SAME
// adaptive forest built at a coarser lateral-diamond-refinement level (coarser intra-block node grid).
// Reuses terra's mesh-agnostic V-cycle driver (linalg::solvers::Multigrid) + Chebyshev smoother +
// coarse PCG. The two adaptive-specific pieces are the transfer operators:
//   - AdaptiveProlongation = geometric per-block scatter (ProlongationVecConstant) THEN the fine-level
//     hanging constraint C (apply_constraint_device) so the prolonged field is conforming.
//   - AdaptiveRestriction  = geometric per-block gather (RestrictionVecConstant; its uniform subdomain
//     halo is inert on an adaptive domain) THEN the adaptive class/halo assembly (assemble_distributed).
// The level operators are AdaptiveDistributedConstrainedOperator<EpsilonDivDiv> (C^T A C + Dirichlet),
// one per LDR level with its own 2:1 tables. Coarse (DCA) operators are re-discretized per level.
//
// Goal: show the MG preconditioner collapses the CG iteration count vs unpreconditioned CG on a refined
// adaptive mesh (the unpreconditioned-CG wall). Manufactured two-bubble solution (mms_plume_rhs.inc).
//
// STATUS: written without GPU validation (cluster GPU congestion) -- expected to need a debug pass.
// Needs a GPU. S_lat=2 => big blocks so the LDR axis has ~5 coarsening levels.

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
#include <memory>
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

// ---- manufactured two-bubble solution (matches mms_plume_gen.py / the AMR test) --------------------
static constexpr double DX = 0.8451, DY = 0.5071, DZ = 0.1690;
static constexpr double R0 = 0.5, R1 = 1.0, WL = 0.05, AMP = 1.0;

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
        const bool on_b = util::has_flag( mask_( s, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );
        if ( !only_boundary_ || on_b )
        {
            const double rn  = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
            const double axc = c( 0 ) * DX + c( 1 ) * DY + c( 2 ) * DZ;
            const double p2  = rn * rn - axc * axc;
            const double env = Kokkos::sin( M_PI * ( rn - R0 ) / ( R1 - R0 ) );
            const double amp = AMP * env * Kokkos::exp( -p2 / ( 2.0 * WL * WL ) );
            data_( s, x, y, r, 0 ) = amp * c( 0 ) / rn;
            data_( s, x, y, r, 1 ) = amp * c( 1 ) / rn;
            data_( s, x, y, r, 2 ) = amp * c( 2 ) / rn;
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
        // 4-LEVEL DEPTH TEST: contrast-10 radial tanh kink at r=0.75 (like the visckink S=8), so the
        // coefficient jump sits on the hanging interface -- does the 4-level MG show the ~2x penalty?
        const double rn = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
        data_( s, x, y, r ) = 1.0 + 9.0 * 0.5 * ( 1.0 + Kokkos::tanh( 8.0 * ( rn - 0.75 ) ) );
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
        const double                  cx = coords( 0 ), cy = coords( 1 ), cz = coords( 2 );
#include "mms_plume_rhs.inc"
    }
};

using Epsilon = fe::wedge::operators::shell::EpsilonDivDivKerngen< double, 3 >;
using Wrapper = AdaptiveDistributedConstrainedOperator< Epsilon >;

// ---- adaptive transfer operators (OperatorLike) ----------------------------------------------------
// Prolongation: geometric per-block scatter (no halo) then the FINE hanging constraint.
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
        // P = C_fine o S o C_coarse. The coarse correction from the coarse solve has UNCONSTRAINED coarse
        // hanging DoFs (null space of the masked coarse op); constrain them FIRST so the geometric scatter
        // does not read that garbage into the fine interface. C_coarse also makes P the exact transpose of
        // AdaptiveRestriction (which applies C_coarse^T via its trailing assemble_distributed).
        amr_deep_copy( coarse_tmp_.grid_data(), coarse.grid_data() );
        apply_constraint_device( coarse_->t_local_d, coarse_tmp_.grid_data() ); // C_coarse (conforming corr.)
        geom_.apply_impl( coarse_tmp_, fine );                                  // per-block trilinear scatter
        apply_constraint_device( fine_->t_local_d, fine.grid_data() );          // C_fine (conforming at hanging)
    }

  private:
    fe::wedge::operators::shell::ProlongationVecConstant< double, 3 > geom_;
    const DistributedAdaptiveMesh*                                    fine_;
    const DistributedAdaptiveMesh*                                    coarse_;
    Grid4DDataScalar< grid::NodeOwnershipFlag >                       coarse_mask_;
    VectorQ1Vec< double, 3 >                                          coarse_tmp_;
};

// Restriction = EXACT transpose of AdaptiveProlongation.  P = C_fine o S (scatter then fine hanging
// constraint), so P^T = S^T o C_fine^T: first fold the FINE hanging DoFs back into their parents
// (apply_constraint_transpose_device, on a copy so the caller's residual is untouched), THEN the
// geometric per-block gather S^T = RestrictionVecConstant (which carries the coarse-grid communication,
// matching ProlongationVecConstant's).  No coarse-side constraint -- P applies none on the coarse input.
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
        apply_constraint_transpose_device( fine_->t_local_d, fine_tmp_.grid_data() ); // C_fine^T (fold hanging)
        geom_.apply_impl( fine_tmp_, coarse );                                         // S^T (per-block gather)
        assemble_distributed( *coarse_, coarse.grid_data() );                          // coarse class-sum + halo
    }

  private:
    fe::wedge::operators::shell::RestrictionVecConstant< double, 3 > geom_;
    const DistributedAdaptiveMesh*                                  coarse_;
    const DistributedAdaptiveMesh*                                  fine_;
    Grid4DDataScalar< grid::NodeOwnershipFlag >                     fine_mask_;
    VectorQ1Vec< double, 3 >                                        fine_tmp_;
};

// Result of one naked-MG case (uniform or adaptive), for the side-by-side comparison.
struct CaseResult
{
    double vc_rate      = 1.0;  // MG-as-solver asymptotic per-cycle rate (last of 25 cycles)
    double vc_final_rel = 1.0;  // MG-as-solver relative residual after 25 cycles
    int    fgmres_iters = -1;   // MG-preconditioned FGMRES iteration count
    double rel_err      = 1.0;  // discretization error ||u - u_exact|| / ||u_exact||
    long   hanging      = 0;    // finest-level hanging-node count (2:1 interface size)
    long   dofs         = 0;    // finest-level owned dofs
};

// Build the LDR-MG on the given forest and (a) run it NAKED as a solver (per-cycle rate history) and
// (b) use it as the FGMRES preconditioner. A flat forest => uniform shell (no hanging nodes); a refined
// forest => 2:1 hanging interface. Same LDR hierarchy either way, so the two rates are directly comparable.
static CaseResult run_case( const AdaptiveForest& forest, const char* label )
{
    CaseResult                  out;
    {
        const int                   min_ldr = 2, max_ldr = 5; // >=2 cells/block for the 2:1 tables; 4 MG levels
        const int                   nlev = max_ldr - min_ldr + 1;
        const int                   S_rad = 2; // CONTROL: 2 radial subdomains => RADIAL-face 2:1 hanging (mild k)

        // LDR refines isotropically: at each level the RADIAL refinement level == LDR, so the shell has
        // S_rad*2^LDR radial intervals -- lateral and radial coarsen 2:1 together as the MG steps down LDR
        // (matches the reference uniform MG's create_uniform(level, level, ...)).
        auto radii_at = [&]( int L ) {
            return grid::shell::uniform_shell_radii< double >( 0.5, 1.0, S_rad * ( 1 << ( min_ldr + L ) ) + 1 );
        };

        // ---- per-level state (STABLE storage: reserve so wrappers' refs/ptrs stay valid) ------------
        std::vector< DistributedAdaptiveMesh >        mesh;
        std::vector< Grid3DDataVec< double, 3 > >     coords;
        std::vector< Grid2DDataScalar< double > >     radii_g;
        std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask;
        std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > bmask;
        std::vector< VectorQ1Scalar< double > >       k;
        mesh.reserve( nlev );
        coords.reserve( nlev ); radii_g.reserve( nlev ); mask.reserve( nlev ); bmask.reserve( nlev ); k.reserve( nlev );

        for ( int L = 0; L < nlev; ++L )
        {
            mesh.push_back( build_distributed_adaptive_mesh( MPI_COMM_WORLD, min_ldr + L, radii_at( L ), forest,
                                                             grid::shell::subdomain_to_rank_by_diamond ) );
            coords.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< double >( mesh[L].domain ) );
            radii_g.push_back( grid::shell::subdomain_shell_radii< double >( mesh[L].domain ) );
            mask.push_back( adaptive_ownership_mask( mesh[L] ) );
            bmask.push_back( adaptive_boundary_mask( mesh[L].domain ) );
            k.emplace_back( "k_" + std::to_string( L ), mesh[L].domain, mask[L] );
            Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( mesh[L].domain ),
                                  KInterpolator{ coords[L], radii_g[L], k[L].grid_data() } );
        }
        Kokkos::fence();

        BoundaryConditions bcs_nn = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };

        // Local (block) EpsilonDivDiv per level; the wrappers reference these -> keep stable.
        std::vector< std::unique_ptr< Epsilon > > A_loc;
        std::vector< VectorQ1Vec< double > >      scratch;
        std::vector< std::unique_ptr< Wrapper > > A;   // constrained operator per level (Dirichlet-projected)
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

        // ---- approximate inverse diagonal per level (assembled A_loc diagonal, boundary set to 1) ----
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
            assemble_distributed( mesh[L], inv_diag[L].grid_data() );             // class-sum the diagonal
            kernels::common::assign_masked_else_keep_old( inv_diag[L].grid_data(), 1.0, bmask[L],
                                                          grid::shell::ShellBoundaryFlag::BOUNDARY ); // Dirichlet rows = 1
            linalg::invert_entries( inv_diag[L] );
        }

        // ---- Chebyshev smoothers per level -----------------------------------------------------------
        using Smoother = linalg::solvers::Chebyshev< Wrapper >;
        std::vector< Smoother > smoothers;
        smoothers.reserve( nlev );
        std::vector< std::vector< VectorQ1Vec< double > > > cheby_tmps( nlev );
        for ( int L = 0; L < nlev; ++L )
        {
            cheby_tmps[L].emplace_back( "ct0_" + std::to_string( L ), mesh[L].domain, mask[L] );
            cheby_tmps[L].emplace_back( "ct1_" + std::to_string( L ), mesh[L].domain, mask[L] );
            smoothers.emplace_back( /*order=*/4, inv_diag[L], cheby_tmps[L], /*prepost iters=*/3 );
        }

        // ---- transfer operators (coarse levels 0..nlev-2) --------------------------------------------
        std::vector< AdaptiveProlongation > P;
        std::vector< AdaptiveRestriction >  R;
        P.reserve( nlev - 1 ); R.reserve( nlev - 1 );
        for ( int L = 0; L < nlev - 1; ++L )
        {
            P.emplace_back( mesh[L + 1], mesh[L] );  // coarse level L -> fine level L+1 (C_coarse, scatter, C_fine)
            R.emplace_back( mesh[L], mesh[L + 1] );   // fine level L+1 -> coarse level L (C_fine^T, gather, C_coarse^T)
        }

        // ---- MG scratch: tmp_r/tmp_e for coarse levels, tmp for all levels ---------------------------
        std::vector< VectorQ1Vec< double > > tmp_r, tmp_e, tmp_all;
        tmp_r.reserve( nlev - 1 ); tmp_e.reserve( nlev - 1 ); tmp_all.reserve( nlev );
        for ( int L = 0; L < nlev; ++L )
            tmp_all.emplace_back( "tmp_" + std::to_string( L ), mesh[L].domain, mask[L] );
        for ( int L = 0; L < nlev - 1; ++L )
        {
            tmp_r.emplace_back( "tmpr_" + std::to_string( L ), mesh[L].domain, mask[L] );
            tmp_e.emplace_back( "tmpe_" + std::to_string( L ), mesh[L].domain, mask[L] );
        }

        // A_c vector for the MG = the COARSE-level operators (levels 0..nlev-2); finest passed to solve.
        std::vector< Wrapper > A_c; // NOTE: Wrapper is copyable (refs/ptr + handles); copies of A[L]
        A_c.reserve( nlev - 1 );
        for ( int L = 0; L < nlev - 1; ++L )
            A_c.push_back( *A[L] );

        // ---- coarse-grid solver: PCG on level 0 ------------------------------------------------------
        auto coarse_table = std::make_shared< util::Table >();
        std::vector< VectorQ1Vec< double > > coarse_tmps;
        for ( int i = 0; i < 4; ++i )
            coarse_tmps.emplace_back( "cst_" + std::to_string( i ), mesh[0].domain, mask[0] );
        linalg::solvers::IterativeSolverParameters coarse_params{ 200, 1e-8, 1e-14 };
        linalg::solvers::PCG< Wrapper > coarse_solver( coarse_params, coarse_table, coarse_tmps );
        coarse_solver.set_tag( "mg_coarse" );

        // ---- assemble the multigrid ------------------------------------------------------------------
        using MG = linalg::solvers::Multigrid< Wrapper, AdaptiveProlongation, AdaptiveRestriction, Smoother,
                                               linalg::solvers::PCG< Wrapper > >;
        auto mg_table = std::make_shared< util::Table >();
        MG   mg( P, R, A_c, tmp_r, tmp_e, tmp_all, smoothers, smoothers, coarse_solver,
                 /*num_cycles=*/1, /*rel_thresh=*/1e-16 );

        // ---- RHS on the finest level (manufactured, Dirichlet lifting) -------------------------------
        const int F = nlev - 1;
        out.hanging = static_cast< long >( mesh[F].t_local.con_dst.size() );
        out.dofs    = kernels::common::count_masked< long >( mask[F], grid::NodeOwnershipFlag::OWNED,
                                                             MPI_COMM_WORLD );
        using Mass  = fe::wedge::operators::shell::VectorMass< double, 3 >;
        using MassW = AdaptiveDistributedConstrainedOperator< Mass >;

        VectorQ1Vec< double > u( "u", mesh[F].domain, mask[F] ), g( "g", mesh[F].domain, mask[F] );
        VectorQ1Vec< double > b( "b", mesh[F].domain, mask[F] ), tmp( "tmpF", mesh[F].domain, mask[F] );
        VectorQ1Vec< double > soln( "solnF", mesh[F].domain, mask[F] ), scr_m( "scrm", mesh[F].domain, mask[F] );
        VectorQ1Vec< double > scr_n( "scrn", mesh[F].domain, mask[F] );

        Kokkos::parallel_for( "sol", grid::shell::local_domain_md_range_policy_nodes( mesh[F].domain ),
                              SolutionInterpolator{ coords[F], radii_g[F], soln.grid_data(), bmask[F], false } );
        Kokkos::parallel_for( "g", grid::shell::local_domain_md_range_policy_nodes( mesh[F].domain ),
                              SolutionInterpolator{ coords[F], radii_g[F], g.grid_data(), bmask[F], true } );
        Kokkos::parallel_for( "rhs", grid::shell::local_domain_md_range_policy_nodes( mesh[F].domain ),
                              RHSInterpolator{ coords[F], radii_g[F], tmp.grid_data() } );
        Kokkos::fence();

        Mass   M_loc( mesh[F].domain, coords[F], radii_g[F], false, linalg::OperatorApplyMode::Replace,
                    linalg::OperatorCommunicationMode::SkipCommunication );
        MassW  M_c( M_loc, mesh[F], scr_m );
        Wrapper A_neu( *A_loc[F], mesh[F], scr_n ); // Neumann (no Dirichlet elimination) for the lifting
        linalg::apply( M_c, tmp, b );
        linalg::apply( A_neu, g, tmp );
        linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );
        kernels::common::assign_masked_else_keep_old( b.grid_data(), g.grid_data(), bmask[F],
                                                      grid::shell::ShellBoundaryFlag::BOUNDARY );
        Kokkos::fence();

        // ---- DIAGNOSTIC: run the MG AS A SOLVER, read the per-V-cycle convergence rate ----------------
        // A good V-cycle gives rate ~0.1-0.3 (=> the PCG failure is R!=P^T symmetry); a rate ~0.9 means
        // the transfer/smoother/coarse is broken. Mirrors test_epsilon_divdiv_ablock_mg_gca.cpp.
        {
            auto diag_table = std::make_shared< util::Table >();
            MG   mg_diag( P, R, A_c, tmp_r, tmp_e, tmp_all, smoothers, smoothers, coarse_solver,
                          /*num_cycles=*/25, /*rel_thresh=*/1e-8 );
            mg_diag.collect_statistics( diag_table );
            VectorQ1Vec< double > u_d( "u_d", mesh[F].domain, mask[F] );
            linalg::assign( u_d, 0.0 );
            linalg::solvers::solve( mg_diag, *A[F], u_d, b );
            std::printf( "  --- [%s] NAKED MG-as-solver V-cycle history (rate<1 good) ---\n", label );
            diag_table->query_rows_equals( "tag", "multigrid" )
                .select_columns( { "absolute_residual", "relative_residual", "residual_convergence_rate" } )
                .print_pretty();
            const auto rows = diag_table->query_rows_equals( "tag", "multigrid" ).rows();
            if ( !rows.empty() )
                out.vc_final_rel = std::get< double >( rows.back().at( "relative_residual" ) );
            if ( rows.size() >= 2 )
                out.vc_rate = std::get< double >( rows.back().at( "residual_convergence_rate" ) );
        }

        // ---- solve: MG-preconditioned PCG vs unpreconditioned PCG ------------------------------------
        auto count_iters = []( const std::shared_ptr< util::Table >& t, const std::string& tag ) {
            const auto rows = t->query_rows_equals( "tag", tag ).rows();
            return rows.empty() ? -1 : static_cast< int >( rows.size() ) - 1;
        };

        // (a) unpreconditioned (disabled to speed up MG-tuning cycles; flip to true to restore the baseline)
        if ( false )
        {
            auto tbl = std::make_shared< util::Table >();
            VectorQ1Vec< double > u0( "u0", mesh[F].domain, mask[F] );
            std::vector< VectorQ1Vec< double > > s;
            for ( int i = 0; i < 4; ++i ) s.emplace_back( "s" + std::to_string( i ), mesh[F].domain, mask[F] );
            linalg::solvers::IterativeSolverParameters pp{ 4000, 1e-10, 1e-14 };
            linalg::solvers::PCG< Wrapper > pcg( pp, tbl, s );
            pcg.set_tag( "unprec" );
            linalg::solvers::solve( pcg, *A[F], u0, b );
            std::printf( "  unpreconditioned PCG iterations: %d\n", count_iters( tbl, "unprec" ) );
        }
        // (b) MG-preconditioned FGMRES. FGMRES (flexible), NOT PCG: the MG preconditioner is NON-LINEAR
        // (its coarse solver is an inner PCG), which breaks outer PCG's conjugacy. Flexible GMRES tolerates
        // a varying preconditioner -- same reason the reference uses FGMRES as the Stokes outer solver.
        {
            auto tbl = std::make_shared< util::Table >();
            linalg::assign( u, 0.0 );
            constexpr int restart = 60;
            linalg::solvers::FGMRESOptions< double > opts;
            opts.restart                     = restart;
            opts.max_iterations              = 120;
            opts.relative_residual_tolerance = 1e-8;
            opts.absolute_residual_tolerance = 1e-14;
            std::vector< VectorQ1Vec< double > > s;
            for ( int i = 0; i < 2 * restart + 4; ++i )
                s.emplace_back( "sm" + std::to_string( i ), mesh[F].domain, mask[F] );
            linalg::solvers::FGMRES< Wrapper, MG > fgmres( s, opts, tbl, mg );
            fgmres.set_tag( "mgprec" );
            linalg::solvers::solve( fgmres, *A[F], u, b );
            const int it = count_iters( tbl, "mgprec" );
            out.fgmres_iters = it;
            std::printf( "  [%s] MG-preconditioned FGMRES iterations: %d\n", label, it );
            CHECK( it > 0 && it < 60 ); // MG should converge in a small, mesh-independent iteration count
        }

        // accuracy check on the MG-preconditioned solution: RELATIVE L2 error vs the manufactured field
        // (FGMRES converged to 1e-8, so u is the true FE solution; this is the discretization error).
        const double exact_norm = std::sqrt( std::fabs( dot( soln, soln ) ) );  // ||u_exact||
        apply_constraint_device( mesh[F].t_local_d, u.grid_data() );
        linalg::lincomb( soln, { 1.0, -1.0 }, { u, soln } );                    // soln <- u - u_exact
        const double err = std::sqrt( std::fabs( dot( soln, soln ) ) );
        const double rel = ( exact_norm > 0 ) ? err / exact_norm : err;
        out.rel_err      = rel;
        std::printf( "  [%s] relative L2 error ||u_mg - u_exact|| / ||u_exact|| = %.3e  (abs %.3e)\n", label, rel, err );
        CHECK( rel < 5e-2 ); // mesh-appropriate discretization error, not a solve failure
    }
    return out;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        const int S_lat = 2, S_rad = 2, M = 2; // CONTROL: S_rad=2 -> radial-face hanging, still mild k (2+sin z)

        // Build a forest: refine the first n_sd1 base leaves to subdivision 1, then the first n_sd2 of the
        // resulting subdivision-1 leaves to subdivision 2 (a graded 0->1->2 interface). Copies before
        // refine() (refine mutates leaves_). This sweeps the SIZE and DEPTH of the hanging interface while
        // the MG (4 levels LDR 2->5, coarse tol 1e-8) is held fixed -- isolating "does much/deep refinement
        // slow the tuned MG?" from the study's weakened 2-level/1e-6 MG.
        auto build = [&]( int n_sd1, int n_sd2 ) {
            AdaptiveForest f( M, S_lat, S_rad );
            if ( n_sd1 > 0 )
            {
                auto                      lv = f.leaves();
                std::vector< ForestLeaf > b( lv.begin(), lv.begin() + std::min< std::size_t >( n_sd1, lv.size() ) );
                f.refine( b );
                f.balance_2to1();
            }
            if ( n_sd2 > 0 )
            {
                auto                      lv = f.leaves();
                std::vector< ForestLeaf > b;
                for ( const auto& l : lv )
                    if ( l.subdivision == 1 && static_cast< int >( b.size() ) < n_sd2 )
                        b.push_back( l );
                f.refine( b );
                f.balance_2to1();
            }
            return f;
        };

        struct Case { const char* label; int n_sd1, n_sd2; };
        const std::vector< Case > cases = {
            { "uniform (flat)", 0, 0 },
            { "1 block  sd1", 1, 0 },
            { "8 blocks sd1", 8, 0 },
            { "1 block  ->sd2 (graded)", 1, 1 },
            { "8 blocks ->4 sd2", 8, 4 },
        };

        std::vector< std::pair< std::string, CaseResult > > results;
        for ( const auto& c : cases )
        {
            auto f = build( c.n_sd1, c.n_sd2 );
            CHECK( f.validate() );
            std::printf( "\n===== case: %s =====\n", c.label );
            results.emplace_back( c.label, run_case( f, c.label ) );
        }

        std::printf( "\n===== REFINEMENT SWEEP: FGMRES iters vs hanging-interface size/depth (fixed 4-level MG) =====\n" );
        std::printf( "  %-24s | dofs      | hanging | FGMRES | disc.err\n", "case" );
        for ( const auto& [lbl, r] : results )
            std::printf( "  %-24s | %9ld | %7ld | %6d | %.2e\n", lbl.c_str(), r.dofs, r.hanging, r.fgmres_iters,
                         r.rel_err );

        for ( const auto& [lbl, r] : results )
            CHECK( r.fgmres_iters > 0 );
    }

    const bool ok = ( g_failures == 0 );
    if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        std::printf( "test_adaptive_mg_gpu: %s (%d checks)\n", ok ? "ALL PASS" : "FAILURE(S)", g_checks );
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
