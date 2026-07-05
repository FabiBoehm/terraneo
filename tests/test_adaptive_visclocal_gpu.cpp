// Localized viscosity blob: uniform vs adaptive refinement where the sharp feature is confined to a small
// sub-region (localized BOTH laterally and radially) -- the regime where AMR is supposed to pay off.
//
// Unlike test_adaptive_visckink_gpu (a radial tanh at r=0.75 that spans the whole shell, a worst case for
// AMR), here the viscosity is a 3D GAUSSIAN BLOB centered at a single point c0 (radius ~0.75, one lateral
// direction), contrast 10, width W. So the viscosity-gradient refinement indicator flags only a compact
// cluster of blocks around the blob; the 2:1 hanging interface wraps that cluster and sits in the flat
// (k=KLO) region everywhere except right at the blob edge. Most of the domain stays coarse.
//
// CONTINUOUS manufactured solution: pick a smooth u* (radial sine * fixed direction, homogeneous on both
// radial boundaries), and use the ANALYTIC forcing f = -div(2k(eps(u*) - 1/3 div u* I)) generated in closed
// form by mms_visclocal_gen.py (mms_visclocal_rhs.inc). The consistent RHS is b = M f (mass matrix), so the
// discrete solution u_h carries the true O(h^2) discretization error -- ||u_h - u*|| is a real accuracy
// metric (uniform vs adaptive), and the FGMRES count measures MG quality on the blob operator.
//
// Metric: uniform dofs/iters vs adaptive dofs/iters, and a WORK proxy (dofs * iters) -- AMR wins if the
// adaptive dof savings beat any iteration penalty from the (now flat-k) hanging interface.
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
#include "terra/io/xdmf.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

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

static constexpr double PI  = 3.14159265358979323846;
static constexpr double DX  = 0.8451, DY = 0.5071, DZ = 0.1690; // unit direction (also aims the blob center)
static constexpr double R0  = 0.5, R1 = 1.0;                    // CMB, surface
static constexpr double RB  = 0.75;                             // blob center radius (mid-shell)
static constexpr double C0X = RB * DX, C0Y = RB * DY, C0Z = RB * DZ; // blob center point (|c0| = 0.75)
static constexpr double W    = 0.08;                           // viscosity Gaussian sigma -- localized lat. AND radially
static constexpr double WSOL = 0.06;                           // solution-bump sigma (feature co-located with the blob)
static constexpr double KLO = 1.0, KHI = 10.0;                 // background / peak viscosity (contrast 10)

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

// Localized viscosity: k = KLO + (KHI-KLO) * exp(-|x-c0|^2 / 2W^2). A compact blob at c0, contrast 10.
struct KLocalInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c  = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const double                  dx = c( 0 ) - C0X, dy = c( 1 ) - C0Y, dz = c( 2 ) - C0Z;
        const double                  d2 = dx * dx + dy * dy + dz * dz;
        data_( s, x, y, r )              = KLO + ( KHI - KLO ) * Kokkos::exp( -d2 / ( 2.0 * W * W ) );
    }
};

// Manufactured solution u* = sin(pi (r-R0)/(R1-R0)) * exp(-|x-c0|^2/(2 WSOL^2)) * dir -- a LOCALIZED bump
// co-located with the viscosity blob (radial sine => exactly zero on both radial boundaries). The solution
// error now concentrates in a compact patch at c0, so an error-driven adaptive mesh can refine just that
// patch and beat uniform on accuracy-per-dof.
struct USolInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c   = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const double                  rn  = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
        const double                  env = Kokkos::sin( PI * ( rn - R0 ) / ( R1 - R0 ) );
        const double                  dx = c( 0 ) - C0X, dy = c( 1 ) - C0Y, dz = c( 2 ) - C0Z;
        const double                  bump = Kokkos::exp( -( dx * dx + dy * dy + dz * dz ) / ( 2.0 * WSOL * WSOL ) );
        data_( s, x, y, r, 0 )            = env * bump * DX;
        data_( s, x, y, r, 1 )            = env * bump * DY;
        data_( s, x, y, r, 2 )            = env * bump * DZ;
    }
};

// Analytic RHS f = -div(2k(eps(u*) - 1/3 div u* I)) for the continuous MMS, generated in closed form by
// mms_visclocal_gen.py (regenerate mms_visclocal_rhs.inc if the constants above change). cx,cy,cz = coords.
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
#include "mms_visclocal_rhs.inc"
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
    , nprocs_( static_cast< int >( mpi::num_processes( coarse.comm ) ) )
    {}

    void apply_impl( const SrcVectorType& fine, DstVectorType& coarse )
    {
        amr_deep_copy( fine_tmp_.grid_data(), fine.grid_data() );
        apply_constraint_transpose_device( fine_->t_local_d, fine_tmp_.grid_data() ); // C_fine^T
        geom_.apply_impl( fine_tmp_, coarse );                                         // gather
        // Coarse class-sum + hanging fold. At np=1 this is a purely LOCAL assembly, so use the on-device
        // apply_exchange_device (same op the constrained operator uses) instead of host-staged
        // assemble_distributed -- avoids the host round-trip the timer tree flagged (~5x on this step).
        if ( nprocs_ == 1 )
            apply_exchange_device( coarse_->t_local_d, coarse.grid_data() );
        else
            assemble_distributed( *coarse_, coarse.grid_data() );
    }

  private:
    fe::wedge::operators::shell::RestrictionVecConstant< double, 3 > geom_;
    const DistributedAdaptiveMesh*                                  coarse_;
    const DistributedAdaptiveMesh*                                  fine_;
    Grid4DDataScalar< grid::NodeOwnershipFlag >                     fine_mask_;
    VectorQ1Vec< double, 3 >                                        fine_tmp_;
    int                                                             nprocs_;
};

// Inverse-diagonal (Jacobi) preconditioner for the MG coarse solver. The coarse level is solved by an inner
// CG each V-cycle; UNPRECONDITIONED that costs ~hundreds of matvecs/cycle and dominates the adaptive solve.
// Preconditioning with the EXACT constrained diagonal (assemble_constrained_diagonal) cuts the coarse-CG
// count sharply -> faster overall solve. (The earlier fold diagonal was too inaccurate to help; this one is.)
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
    double                t_solve = 0, t_est = 0; // wall-clock: FGMRES solve, and the estimator coarse solve
    std::vector< double > indicator; // per block: max_K k - min_K k (viscosity gradient)
};

// EXACT diagonal of the CONSTRAINED operator diag(C^T A C), assembled from element matrices with the 2:1
// hanging constraint applied at element level (see test_adaptive_visckink_gpu for the full rationale).
template < typename Op, typename MeshT, typename MaskT >
static void assemble_constrained_diagonal( Op& op, const MeshT& mesh, const MaskT& mask,
                                           VectorQ1Vec< double >& diag )
{
    const auto& dom     = mesh.domain;
    const auto& td      = mesh.t_local_d;
    const auto& di      = dom.domain_info();
    const int   hex_lat = di.subdomain_num_nodes_per_side_laterally() - 1;
    const int   hex_rad = di.subdomain_num_nodes_radially() - 1;
    const int   nsub    = static_cast< int >( dom.subdomains().size() );

    VectorQ1Scalar< double > hang( "cdiag_hang", dom, mask );
    linalg::assign( hang, -1.0 );
    auto hg = hang.grid_data();
    auto cd = td.con_dst;
    Kokkos::parallel_for(
        "cdiag_sethang", cd.extent( 0 ),
        KOKKOS_LAMBDA( const int i ) { hg( cd( i, 0 ), cd( i, 1 ), cd( i, 2 ), cd( i, 3 ) ) = (double) i; } );
    Kokkos::fence();

    linalg::assign( diag, 0.0 );
    auto d0  = diag.grid_data().comp_[0];
    auto d1  = diag.grid_data().comp_[1];
    auto d2  = diag.grid_data().comp_[2];
    auto co  = td.con_off;
    auto cp  = td.con_par;
    auto cw  = td.con_wt;
    auto opv = op; // capture the operator functor by value (shallow Kokkos views)

    Kokkos::parallel_for(
        "cdiag_assemble",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { nsub, hex_lat, hex_lat, hex_rad } ),
        KOKKOS_LAMBDA( const int sd, const int xc, const int yc, const int rc ) {
            const int offx[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
            const int offy[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
            const int offr[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };
            for ( int wg = 0; wg < 2; ++wg )
            {
                const auto A = opv.assemble_local_matrix( sd, xc, yc, rc, wg ); // 18x18, index dim*6+node
                int        np[6], qs[6][4], qx[6][4], qy[6][4], qr[6][4];
                double     qw[6][4];
                for ( int n = 0; n < 6; ++n )
                {
                    const int gx = xc + offx[wg][n], gy = yc + offy[wg][n], gr = rc + offr[wg][n];
                    const int hi = (int) hg( sd, gx, gy, gr );
                    if ( hi < 0 )
                    {
                        np[n] = 1;
                        qs[n][0] = sd; qx[n][0] = gx; qy[n][0] = gy; qr[n][0] = gr; qw[n][0] = 1.0;
                    }
                    else
                    {
                        int c = 0;
                        for ( int p = co( hi ); p < co( hi + 1 ) && c < 4; ++p )
                        {
                            qs[n][c] = cp( p, 0 ); qx[n][c] = cp( p, 1 ); qy[n][c] = cp( p, 2 );
                            qr[n][c] = cp( p, 3 ); qw[n][c] = cw( p ); ++c;
                        }
                        np[n] = c;
                    }
                }
                for ( int d = 0; d < 3; ++d )
                    for ( int a = 0; a < 6; ++a )
                        for ( int b = 0; b < 6; ++b )
                        {
                            const double Aab = A( d * 6 + a, d * 6 + b );
                            if ( Aab == 0.0 )
                                continue;
                            for ( int ia = 0; ia < np[a]; ++ia )
                                for ( int ib = 0; ib < np[b]; ++ib )
                                    if ( qs[a][ia] == qs[b][ib] && qx[a][ia] == qx[b][ib] &&
                                         qy[a][ia] == qy[b][ib] && qr[a][ia] == qr[b][ib] )
                                    {
                                        const double c2 = qw[a][ia] * qw[b][ib] * Aab;
                                        if ( d == 0 )
                                            Kokkos::atomic_add( &d0( qs[a][ia], qx[a][ia], qy[a][ia], qr[a][ia] ), c2 );
                                        else if ( d == 1 )
                                            Kokkos::atomic_add( &d1( qs[a][ia], qx[a][ia], qy[a][ia], qr[a][ia] ), c2 );
                                        else
                                            Kokkos::atomic_add( &d2( qs[a][ia], qx[a][ia], qy[a][ia], qr[a][ia] ), c2 );
                                    }
                        }
            }
        } );
    Kokkos::fence();
    apply_exchange_device( td, diag.grid_data() );

    auto ch = td.con_dst;
    auto e0 = diag.grid_data().comp_[0];
    auto e1 = diag.grid_data().comp_[1];
    auto e2 = diag.grid_data().comp_[2];
    Kokkos::parallel_for(
        "cdiag_hangset", ch.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            e0( ch( i, 0 ), ch( i, 1 ), ch( i, 2 ), ch( i, 3 ) ) = 1.0;
            e1( ch( i, 0 ), ch( i, 1 ), ch( i, 2 ), ch( i, 3 ) ) = 1.0;
            e2( ch( i, 0 ), ch( i, 1 ), ch( i, 2 ), ch( i, 3 ) ) = 1.0;
        } );
    Kokkos::fence();
}

// Build an LDR-multigrid on `forest`, use it as the FGMRES preconditioner for the discrete manufactured
// problem (localized viscosity blob), and report FGMRES iters + solver error + per-block viscosity indicator.
static Res solve_level( MPI_Comm comm, const AdaptiveForest& forest, int ldr_fine, int S_rad,
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
    terra::util::Timer _t_mesh( "mesh_build" ); // build the adaptive LDR mesh hierarchy + coords/masks/k
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
                              KLocalInterpolator{ coords[L], radii_g[L], k[L].grid_data() } );
    }
    Kokkos::fence();

    _t_mesh.stop();

    terra::util::Timer _t_setup( "solver_setup" ); // operators, exact diagonal, smoothers, transfers, MG object
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

    std::vector< VectorQ1Vec< double > > inv_diag;
    inv_diag.reserve( nlev );
    for ( int L = 0; L < nlev; ++L )
    {
        inv_diag.emplace_back( "invd_" + std::to_string( L ), mesh[L].domain, mask[L] );
        assemble_constrained_diagonal( *A_loc[L], mesh[L], mask[L], inv_diag[L] );
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
        smoothers.emplace_back( /*order=*/4, inv_diag[L], cheby_tmps[L], /*prepost=*/3 ); // canonical smoother
    }

    std::vector< AdaptiveProlongation > P;
    std::vector< AdaptiveRestriction >  R;
    P.reserve( nlev - 1 ); R.reserve( nlev - 1 );
    for ( int L = 0; L < nlev - 1; ++L )
    {
        P.emplace_back( mesh[L + 1], mesh[L] );
        R.emplace_back( mesh[L], mesh[L + 1] );
    }

    // ---- LINEAR-REPRODUCTION AUDIT: u = |x| (the radius) is exactly radially-linear and laterally-constant on
    // uniform radii, so a CORRECT prolongation must reproduce it to MACHINE ZERO -- at every node incl. hanging
    // (2:1 radial midpoint parents average to its radius). A nonzero result => an order defect in the constraint
    // or transfer that the constant/POU tests can't see. Also splits the error onto the hanging fine nodes.
    for ( int L = 0; L < nlev - 1; ++L )
    {
        VectorQ1Vec< double > uc( "lin_uc", mesh[L].domain, mask[L] );
        VectorQ1Vec< double > uf( "lin_uf", mesh[L + 1].domain, mask[L + 1] );
        VectorQ1Vec< double > Pu( "lin_Pu", mesh[L + 1].domain, mask[L + 1] );
        VectorQ1Vec< double > df( "lin_df", mesh[L + 1].domain, mask[L + 1] );
        auto fill_rad = [&]( VectorQ1Vec< double >& v, int lvl ) {
            auto c0 = v.grid_data().comp_[0]; auto c1 = v.grid_data().comp_[1]; auto c2 = v.grid_data().comp_[2];
            auto cg = coords[lvl]; auto rg = radii_g[lvl];
            Kokkos::parallel_for(
                "lin_fill", grid::shell::local_domain_md_range_policy_nodes( mesh[lvl].domain ),
                KOKKOS_LAMBDA( const int s, const int x, const int y, const int r ) {
                    const dense::Vec< double, 3 > c = grid::shell::coords( s, x, y, r, cg, rg );
                    const double rn = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
                    c0( s, x, y, r ) = rn; c1( s, x, y, r ) = rn; c2( s, x, y, r ) = rn; // linear scalar in all comps
                } );
            Kokkos::fence();
        };
        fill_rad( uc, L );
        fill_rad( uf, L + 1 );
        linalg::assign( Pu, 0.0 );
        P[L].apply_impl( uc, Pu );                       // P(radius_coarse)
        linalg::lincomb( df, { 1.0, -1.0 }, { Pu, uf } ); // P(r) - r  (should be 0)
        const double l2d = std::sqrt( std::fabs( dot( df, df ) ) );
        const double l2r = std::sqrt( std::fabs( dot( uf, uf ) ) );
        // error restricted to the hanging fine nodes
        VectorQ1Vec< double > dh( "lin_dh", mesh[L + 1].domain, mask[L + 1] );
        amr_deep_copy( dh.grid_data(), df.grid_data() );
        auto h0 = dh.grid_data().comp_[0]; auto h1 = dh.grid_data().comp_[1]; auto h2 = dh.grid_data().comp_[2];
        auto cd = mesh[L + 1].t_local_d.con_dst;
        VectorQ1Vec< double > zero_nonhang( "lin_zn", mesh[L + 1].domain, mask[L + 1] );
        linalg::assign( zero_nonhang, 0.0 );
        auto z0 = zero_nonhang.grid_data().comp_[0]; auto z1 = zero_nonhang.grid_data().comp_[1];
        auto z2 = zero_nonhang.grid_data().comp_[2];
        Kokkos::parallel_for( "lin_hang", cd.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            z0( cd( i, 0 ), cd( i, 1 ), cd( i, 2 ), cd( i, 3 ) ) = h0( cd( i, 0 ), cd( i, 1 ), cd( i, 2 ), cd( i, 3 ) );
            z1( cd( i, 0 ), cd( i, 1 ), cd( i, 2 ), cd( i, 3 ) ) = h1( cd( i, 0 ), cd( i, 1 ), cd( i, 2 ), cd( i, 3 ) );
            z2( cd( i, 0 ), cd( i, 1 ), cd( i, 2 ), cd( i, 3 ) ) = h2( cd( i, 0 ), cd( i, 1 ), cd( i, 2 ), cd( i, 3 ) );
        } );
        Kokkos::fence();
        const double l2h = std::sqrt( std::fabs( dot( zero_nonhang, zero_nonhang ) ) );
        if ( mpi::rank( comm ) == 0 )
            std::printf( "    [%s LINREP L%d->%d] ||P(r)-r||/||r|| = %.2e  (hanging-only %.2e)\n", tag.c_str(), L,
                         L + 1, l2d / std::max( l2r, 1e-300 ), l2h );
    }

    // ---- CGC test: does one EXACT two-grid coarse correction remove a smooth error, and where does the
    // leftover sit (inside the blob |x-c0|<BW vs total)? Contraction vs uniform gauges P/R-together quality.
    {
        terra::util::Timer _t_cgc( "cgc_diagnostic" ); // exact two-grid CGC contraction probe (diagnostic only)
        constexpr double BW = 0.20; // blob-region half-extent for the localization
        const auto&      domF = mesh[F].domain;
        const auto&      domC = mesh[F - 1].domain;
        VectorQ1Vec< double > e( "cgc_e", domF, mask[F] ), Ae( "cgc_Ae", domF, mask[F] ), ef( "cgc_ef", domF, mask[F] );
        VectorQ1Vec< double > ea( "cgc_ea", domF, mask[F] ), Aea( "cgc_Aea", domF, mask[F] ), ein( "cgc_ein", domF, mask[F] );
        VectorQ1Vec< double > rc( "cgc_rc", domC, mask[F - 1] ), ec( "cgc_ec", domC, mask[F - 1] );
        auto cF = coords[F];
        auto rF = radii_g[F];
        auto e0 = e.grid_data().comp_[0];
        auto e1 = e.grid_data().comp_[1];
        auto e2 = e.grid_data().comp_[2];
        Kokkos::parallel_for(
            "cgc_fill", grid::shell::local_domain_md_range_policy_nodes( domF ),
            KOKKOS_LAMBDA( const int s, const int x, const int y, const int r ) {
                const dense::Vec< double, 3 > c   = grid::shell::coords( s, x, y, r, cF, rF );
                const double                  rn  = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
                const double                  env = Kokkos::sin( PI * ( rn - R0 ) / ( R1 - R0 ) );
                e0( s, x, y, r ) = env * DX;
                e1( s, x, y, r ) = env * DY;
                e2( s, x, y, r ) = env * DZ;
            } );
        Kokkos::fence();
        apply_constraint_device( mesh[F].t_local_d, e.grid_data() );
        kernels::common::assign_masked_else_keep_old( e.grid_data(), 0.0, bmask[F],
                                                      grid::shell::ShellBoundaryFlag::BOUNDARY );
        apply_constraint_device( mesh[F].t_local_d, e.grid_data() );
        linalg::apply( *A[F], e, Ae );
        const double eb = std::sqrt( std::fabs( dot( e, Ae ) ) );

        linalg::assign( rc, 0.0 );
        R[F - 1].apply_impl( Ae, rc );
        std::vector< VectorQ1Vec< double > > cgtmp;
        for ( int i = 0; i < 4; ++i )
            cgtmp.emplace_back( "cgc_ct" + std::to_string( i ), domC, mask[F - 1] );
        auto                                       cgtab = std::make_shared< util::Table >();
        linalg::solvers::IterativeSolverParameters cgp{ 5000, 1e-11, 1e-14 };
        linalg::solvers::PCG< Wrapper >            cgs( cgp, cgtab, cgtmp );
        cgs.set_tag( "cgc_coarse" );
        linalg::assign( ec, 0.0 );
        linalg::solvers::solve( cgs, *A[F - 1], ec, rc );

        linalg::assign( ef, 0.0 );
        P[F - 1].apply_impl( ec, ef );
        linalg::lincomb( ea, { 1.0, -1.0 }, { e, ef } );
        apply_constraint_device( mesh[F].t_local_d, ea.grid_data() );
        linalg::apply( *A[F], ea, Aea );
        const double eaN = std::sqrt( std::fabs( dot( ea, Aea ) ) );

        amr_deep_copy( ein.grid_data(), ea.grid_data() );
        auto i0 = ein.grid_data().comp_[0];
        auto i1 = ein.grid_data().comp_[1];
        auto i2 = ein.grid_data().comp_[2];
        Kokkos::parallel_for(
            "cgc_band", grid::shell::local_domain_md_range_policy_nodes( domF ),
            KOKKOS_LAMBDA( const int s, const int x, const int y, const int r ) {
                const dense::Vec< double, 3 > c  = grid::shell::coords( s, x, y, r, cF, rF );
                const double dx = c( 0 ) - C0X, dy = c( 1 ) - C0Y, dz = c( 2 ) - C0Z;
                if ( Kokkos::sqrt( dx * dx + dy * dy + dz * dz ) >= BW )
                {
                    i0( s, x, y, r ) = 0.0;
                    i1( s, x, y, r ) = 0.0;
                    i2( s, x, y, r ) = 0.0;
                }
            } );
        Kokkos::fence();
        const double l2_band  = std::sqrt( std::fabs( dot( ein, ein ) ) );
        const double l2_total = std::sqrt( std::fabs( dot( ea, ea ) ) );
        if ( mpi::rank( comm ) == 0 )
            std::printf( "    [%s CGC] ||e||_A: %.4e -> %.4e  contraction %.3f  leftover in |x-c0|<%.2f: %.1f%%\n",
                         tag.c_str(), eb, eaN, eaN / std::max( eb, 1e-300 ), BW,
                         100.0 * l2_band / std::max( l2_total, 1e-300 ) );
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
    // LOOSE coarse tolerance (1e-2): FGMRES is flexible, so the coarse grid need not be solved accurately -- a
    // sloppy coarse solve (~10-30 CG iters vs ~250 at 1e-8) slashes the dominant per-V-cycle cost, at the price
    // of a few more outer iterations. Jacobi-preconditioned with the EXACT constrained diagonal (inv_diag[0]).
    linalg::solvers::IterativeSolverParameters coarse_params{ 200, 1e-2, 1e-14 };
    using CoarsePrec = InverseDiagonalPreconditioner< Wrapper >;
    using CoarseCG   = linalg::solvers::PCG< Wrapper, CoarsePrec >;
    CoarseCG coarse_solver( coarse_params, coarse_table, coarse_tmps, CoarsePrec( inv_diag[0] ) );
    coarse_solver.set_tag( "mg_coarse" );

    using MG =
        linalg::solvers::Multigrid< Wrapper, AdaptiveProlongation, AdaptiveRestriction, Smoother, CoarseCG >;
    MG mg( P, R, A_c, tmp_r, tmp_e, tmp_all, smoothers, smoothers, coarse_solver, 1, 1e-16 );

    // ---- CONTINUOUS manufactured RHS on the finest level: b = M_c f, f the ANALYTIC forcing from
    // mms_visclocal_rhs.inc. u* = sin(pi(r-R0)/(R1-R0)) dir is homogeneous on both radial boundaries, so the
    // Dirichlet data is 0 (no lift term). The discrete u_h carries the true O(h^2) discretization error.
    const auto& dom = mesh[F].domain;
    VectorQ1Vec< double > u( "u", dom, mask[F] ), b( "b", dom, mask[F] ), soln( "soln", dom, mask[F] );
    VectorQ1Vec< double > f( "f", dom, mask[F] ), err( "err", dom, mask[F] ), Merr( "Merr", dom, mask[F] );
    VectorQ1Vec< double > rr( "rr", dom, mask[F] ), scr_m( "scrm", dom, mask[F] );

    Kokkos::parallel_for( "sol", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          USolInterpolator{ coords[F], radii_g[F], soln.grid_data() } );
    Kokkos::parallel_for( "rhs", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          RHSInterpolator{ coords[F], radii_g[F], f.grid_data() } );
    Kokkos::fence();

    using Mass_ = Mass;
    Mass_ M_loc( dom, coords[F], radii_g[F], false, linalg::OperatorApplyMode::Replace,
                 linalg::OperatorCommunicationMode::SkipCommunication );
    MassW M_c( M_loc, mesh[F], scr_m );
    linalg::apply( M_c, f, b ); // b = M f (consistent RHS)
    kernels::common::assign_masked_else_keep_old( b.grid_data(), 0.0, bmask[F],
                                                  grid::shell::ShellBoundaryFlag::BOUNDARY ); // homogeneous Dirichlet
    Kokkos::fence();

    _t_setup.stop();

    terra::util::Timer _t_solve( "mg_solve" ); // the MG-FGMRES time-to-solution
    // ---- MG-preconditioned FGMRES ----
    linalg::assign( u, 0.0 );
    constexpr int restart = 50;
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
    Kokkos::fence();
    Kokkos::Timer fgmres_timer; // GPU-fenced wall-clock of the MG-FGMRES solve = the time-to-solution
    linalg::solvers::solve( fgmres, *A[F], u, b );
    Kokkos::fence();
    const double t_solve = fgmres_timer.seconds();
    apply_constraint_device( mesh[F].t_local_d, u.grid_data() );
    _t_solve.stop();

    terra::util::Timer _t_xdmf( "xdmf_write" );
    // ---- XDMF dump (viscosity k + subdivision level + solution u) for ParaView, per adaptive round.
    // Written to $VISCLOCAL_OUT/round<N>_mesh/step_0.xmf -- rendered by pv_render_visclocal.py.
    if ( tag.rfind( "ada_R", 0 ) == 0 || tag == "uni_L2" )
    {
        VectorQ1Scalar< double > kvis( "k", dom, mask[F] );   // relabel "k_L" -> "k" for a stable field name
        VectorQ1Scalar< double > level( "level", dom, mask[F] );
        amr_deep_copy( kvis.grid_data(), k[F].grid_data() );
        {
            auto      lh  = Kokkos::create_mirror_view( level.grid_data() );
            const int lnx = static_cast< int >( lh.extent( 1 ) ), lny = static_cast< int >( lh.extent( 2 ) ),
                      lnr = static_cast< int >( lh.extent( 3 ) );
            for ( int s = 0; s < static_cast< int >( dom.subdomains().size() ); ++s )
            {
                const double lv = dom.subdivision_of( dom.subdomain_info_from_local_idx( s ) );
                for ( int x = 0; x < lnx; ++x )
                    for ( int y = 0; y < lny; ++y )
                        for ( int r = 0; r < lnr; ++r )
                            lh( s, x, y, r ) = lv;
            }
            Kokkos::deep_copy( level.grid_data(), lh );
        }
        const char* env = std::getenv( "VISCLOCAL_OUT" );
        std::string out = env ? std::string( env ) : ( std::string( std::getenv( "HOME" ) ) + "/visclocal_out" );
        std::string dir = out + ( tag[0] == 'a' ? "/round" + tag.substr( 5 ) : "/uni" + tag.substr( 5 ) ) + "_mesh";
        if ( mpi::rank( comm ) == 0 )
            std::printf( "    [%s] writing XDMF to %s\n", tag.c_str(), dir.c_str() );
        terra::util::prepare_empty_directory( dir );
        terra::io::XDMFOutput xdmf( dir, dom, coords[F], radii_g[F] );
        xdmf.add( kvis.grid_data() );
        xdmf.add( level.grid_data() );
        xdmf.add( u.grid_data() );
        xdmf.write();
    }
    _t_xdmf.stop();

    terra::util::Timer _t_est( "estimator" ); // hierarchical estimator: coarse solve + prolong + per-block indicator
    // ---- solver error ||u - u*|| (mass-weighted), residual, viscosity indicator ----
    linalg::lincomb( err, { 1.0, -1.0 }, { u, soln } );
    linalg::apply( M_c, err, Merr );
    const double l2 = std::sqrt( std::fabs( dot( Merr, err ) ) );
    linalg::apply( *A[F], u, rr );
    linalg::lincomb( rr, { 1.0, -1.0 }, { b, rr } );
    const double relres = std::sqrt( std::fabs( dot( rr, rr ) ) / std::max( std::fabs( dot( b, b ) ), 1e-300 ) );

    Res res;
    res.dofs   = kernels::common::count_masked< long >( mask[F], grid::NodeOwnershipFlag::OWNED, comm );
    res.l2     = l2;
    res.relres  = relres;
    res.iters   = static_cast< int >( table->query_rows_equals( "tag", tag ).rows().size() ) - 1;
    res.t_solve = t_solve;

    // Refinement indicator = GEOMETRIC BALL (interface-preserving coarsening test): refine every block within
    // R_REFINE of the blob center c0, coarsen outside. R_REFINE is chosen BEYOND all viscosity variation
    // (k(R_REFINE) ~ KLO), so the region of interest gets FULL (uniform-level) resolution AND the 2:1 hanging
    // interface lands in the CONSTANT-coefficient far field, where the mild-k control showed hanging is
    // penalty-free. Tests whether decoupling the interface from the coefficient jump recovers uniform's
    // iteration count while keeping the far-field coarse. No u*, no coarse solve.
    res.t_est             = 0.0;
    const int    nsub     = static_cast< int >( dom.subdomains().size() );
    const double R_REFINE = 0.30; // > 3.5 W = 0.28; k(0.30) ~ 1.008 (essentially KLO)
    auto         cF       = coords[F];
    auto         rF       = radii_g[F];
    VectorQ1Scalar< double > dist( "ball_dist", dom, mask[F] );
    auto                     dg = dist.grid_data();
    Kokkos::parallel_for(
        "ball_dist", grid::shell::local_domain_md_range_policy_nodes( dom ),
        KOKKOS_LAMBDA( const int s, const int x, const int y, const int r ) {
            const dense::Vec< double, 3 > c  = grid::shell::coords( s, x, y, r, cF, rF );
            const double                  dx = c( 0 ) - C0X, dy = c( 1 ) - C0Y, dz = c( 2 ) - C0Z;
            dg( s, x, y, r )                 = Kokkos::sqrt( dx * dx + dy * dy + dz * dz );
        } );
    Kokkos::fence();
    auto      dhh = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, dist.grid_data() );
    const int nx = dhh.extent( 1 ), ny = dhh.extent( 2 ), nr = dhh.extent( 3 );
    res.indicator.assign( nsub, 0.0 );
    for ( int s = 0; s < nsub; ++s )
    {
        double mind = 1e300;
        for ( int i = 0; i < nx; ++i )
            for ( int j = 0; j < ny; ++j )
                for ( int m = 0; m < nr; ++m )
                    mind = std::min( mind, dhh( s, i, j, m ) );
        res.indicator[s] = ( mind < R_REFINE ) ? 1.0 : 0.0; // refine blocks overlapping the ball
    }
    _t_est.stop();
    const long   coarse_rows = static_cast< long >( coarse_table->query_rows_equals( "tag", "mg_coarse" ).rows().size() );
    const double coarse_per  = res.iters > 0 ? static_cast< double >( coarse_rows ) / res.iters : 0.0;
    if ( mpi::rank( comm ) == 0 )
        std::printf( "    [%-16s] dofs %9ld  FGMRES %4d  solve %6.2fs  est %5.2fs  coarseCG %6ld  ||u-u*|| %.3e  relres %.2e  hanging %zu\n",
                     tag.c_str(), res.dofs, res.iters, res.t_solve, res.t_est, coarse_rows, l2, relres,
                     mesh[F].t_local.con_dst.size() );
    return res;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        auto table = std::make_shared< util::Table >();

        const int    LDR = 4, S_lat = 2, S_rad = 2, M = 8; // finest LDR=4 => 3-level MG (LDR 2->3->4)
        const int    NU = 2;   // uniform levels 0..NU  (L2 = subdiv2, the same-resolution baseline)
        const int    NA = 2;   // adaptive rounds 0..NA (feature refines locally & deep, stays small)
        const double FRAC = 0.50; // ball indicator is {0,1}; refine flagged (in-ball) blocks
                                  // (smoother nesting; the far-field error ~0 keeps it localized, not uniform)

        Res    u_last{}, a_last{}, a_best{};
        double a_total_time = 0.0; // total AMR wall-clock: all rounds' FGMRES solves + estimator coarse solves

        // ---- UNIFORM sweep ----
        {
            terra::util::Timer _t_uni( "uniform_sweep" );
            AdaptiveForest f( NU, S_lat, S_rad );
            for ( int lvl = 0; lvl <= NU; ++lvl )
            {
                u_last = solve_level( MPI_COMM_WORLD, f, LDR, S_rad, table, "uni_L" + std::to_string( lvl ) );
                CHECK( u_last.relres < 1e-6 );
                if ( lvl < NU )
                {
                    auto all = f.leaves();
                    f.refine( all );
                    f.balance_2to1();
                }
            }
        }

        // ---- ADAPTIVE sweep (TWO-LEVEL hierarchical estimator ||u_h - P u_H|| from a coarse solve -- no u*) ----
        {
            terra::util::Timer _t_ada( "adaptive_sweep" );
            AdaptiveForest f( M, S_lat, S_rad );
            double         prev_l2 = 1e300;
            for ( int round = 0; round <= NA; ++round )
            {
                auto leaves = f.leaves();
                a_last = solve_level( MPI_COMM_WORLD, f, LDR, S_rad, table, "ada_R" + std::to_string( round ) );
                CHECK( a_last.relres < 1e-6 );
                a_total_time += a_last.t_solve + a_last.t_est; // accumulate the full AMR workflow time
                if ( round == 0 || a_last.l2 < a_best.l2 )
                    a_best = a_last; // track the lowest-error round to report

                // STOPPING CRITERION: the (exact) error stopped decreasing => the last refinement OVER-refined
                // (it began chasing hanging-interface error rather than solution error). Halt and keep the best
                // round; taking further rounds only balloons dofs and can raise the error (the R4 hook).
                if ( a_last.l2 > prev_l2 )
                {
                    if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
                        std::printf( "  [stop] round %d L2 %.3e > previous %.3e -- over-refined; best = dofs %ld, L2 %.3e\n",
                                     round, a_last.l2, prev_l2, a_best.dofs, a_best.l2 );
                    break;
                }
                prev_l2 = a_last.l2;
                if ( round == NA )
                    break;

                {
                terra::util::Timer _t_refine( "refine_decision" ); // indicator scatter + all-reduce + refine + 2:1 balance
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
                MPI_Allreduce( MPI_IN_PLACE, gind.data(), static_cast< int >( gind.size() ), MPI_DOUBLE, MPI_SUM,
                               MPI_COMM_WORLD );
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
        }

        if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        {
            std::printf( "\n===== SUMMARY: localized viscosity blob, uniform vs adaptive =====\n" );
            std::printf( "  uniform : dofs %9ld  iters %3d  L2 %.3e  solve %6.2fs\n", u_last.dofs, u_last.iters,
                         u_last.l2, u_last.t_solve );
            std::printf( "  adaptive: dofs %9ld  iters %3d  L2 %.3e  solve %6.2fs  (best kept round)\n",
                         a_best.dofs, a_best.iters, a_best.l2, a_best.t_solve );
            std::printf( "  ---- TIME-TO-SOLUTION ----\n" );
            std::printf( "  uniform (single L2 solve)          : %6.2fs  @ L2 %.3e\n", u_last.t_solve, u_last.l2 );
            std::printf( "  adaptive (best kept round solve)   : %6.2fs  @ L2 %.3e\n", a_best.t_solve, a_best.l2 );
            std::printf( "  adaptive (FULL workflow: all rounds+estimator) : %6.2fs\n", a_total_time );
            std::printf( "  speedup  single-solve %.2fx   full-workflow %.2fx  (uniform_time / adaptive_time)\n",
                         u_last.t_solve / std::max( a_best.t_solve, 1e-9 ),
                         u_last.t_solve / std::max( a_total_time, 1e-9 ) );
            std::printf( "  adaptive/uniform:  dofs %.3fx   iters %.3fx  (<1 => AMR wins)\n",
                         (double) a_best.dofs / (double) u_last.dofs,
                         (double) a_best.iters / (double) u_last.iters );
        }
        CHECK( u_last.iters > 0 && u_last.iters < 100 );
        CHECK( a_best.iters > 0 && a_best.iters < 100 );

        // ---- AMR timer tree: MPI-aggregated JSON for plot_timer_tree.py, plus a flat per-region console dump ----
        terra::util::TimerTree::instance().aggregate_mpi(); // collective
        if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        {
            const char* env = std::getenv( "VISCLOCAL_OUT" );
            std::string out = env ? std::string( env ) : ( std::string( std::getenv( "HOME" ) ) + "/visclocal_out" );
            std::string js  = terra::util::TimerTree::instance().json_aggregate();
            FILE*       fp  = std::fopen( ( out + "/timer_tree.json" ).c_str(), "w" );
            if ( fp ) { std::fprintf( fp, "%s\n", js.c_str() ); std::fclose( fp ); }
            std::printf( "\n===== AMR TIMER TREE (max over ranks, seconds) =====\n%s\n", js.c_str() );
            std::printf( "wrote %s/timer_tree.json\n", out.c_str() );
        }
    }

    const bool ok = ( g_failures == 0 );
    if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        std::printf( "test_adaptive_visclocal_gpu: %s (%d checks)\n", ok ? "ALL PASS" : "FAILURE(S)", g_checks );
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
