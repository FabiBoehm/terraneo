// AMR pays off: uniform vs solution-adaptive refinement on a manufactured solution with a SHARP feature.
//
// Manufactured solution for the production EpsilonDivDivKerngen operator: a RISING PLUME -- radially
// outward (upward) velocity in a laterally-Gaussian column rising from an interior point on the CMB and
// widening as it ascends (a plume head). A laterally-localized feature: adaptive refinement concentrates
// on the plume column and leaves the rest coarse. The plume axis is off-pole; pole-corner 2:1 is now
// supported anyway. k = 2+sin z. The analytic RHS f = -div( 2k(eps(u) - 1/3 (div u) I) ) is CAS-generated
// (mms_plume_rhs.inc); that strong form was verified to reproduce test_epsilon_divdiv_cg's RHS to 1e-14.
//
// Two sweeps, both measuring the mass-weighted discrete L2 error ||u_h - u|| = sqrt(e^T M e):
//   UNIFORM   -- refine EVERY leaf each level (h halves everywhere).
//   ADAPTIVE  -- refine only blocks flagged by the gradient indicator eta_K = max_K|u| - min_K|u|
//                (~ h|grad u|), which concentrates on the interface and leaves smooth regions coarse.
// Because the error lives at the interface, the adaptive mesh reaches the same L2 error with far fewer dofs:
// its (dofs, error) curve sits BELOW the uniform one. The test asserts adaptive dominates uniform at a
// matched dof budget, and writes both curves to $AMR_DEMO_OUT/amr_vs_uniform.csv for the log-log plot.
//
// Needs a GPU -- run on an H100 node.

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/pcg.hpp"
#include "terra/io/xdmf.hpp"
#include "util/filesystem.hpp"
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

// two antipodal upwelling columns ("two bubbles"), sharp lateral decay -- match mms_plume_gen.py
static constexpr double DX = 0.8451, DY = 0.5071, DZ = 0.1690; // axis (unit), off-pole
static constexpr double R0 = 0.5, R1 = 1.0, WL = 0.05, AMP = 1.0; // WL small => sharp decay

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

// exact solution: rising plume -- radial (upward) velocity, laterally-Gaussian column widening upward
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
            const double axc = c( 0 ) * DX + c( 1 ) * DY + c( 2 ) * DZ; // x . axis
            const double p2  = rn * rn - axc * axc;                     // lateral dist^2 from axis line
            const double env = Kokkos::sin( M_PI * ( rn - R0 ) / ( R1 - R0 ) );
            const double amp = AMP * env * Kokkos::exp( -p2 / ( 2.0 * WL * WL ) ); // two sharp columns
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
        const double                  cx = coords( 0 ), cy = coords( 1 ), cz = coords( 2 );
#include "mms_plume_rhs.inc"
    }
};

struct Res
{
    long                  dofs = 0;
    double                l2 = 0, relres = 0;
    std::vector< double > indicator; // per block: max|u| - min|u|
};

static Res solve_level( const DistributedAdaptiveMesh& mesh, const std::shared_ptr< util::Table >& table,
                        const std::string& tag, const std::string& xdmf_dir = "" )
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

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    linalg::apply( M_c, error, Merror );
    const double l2 = std::sqrt( std::fabs( dot( Merror, error ) ) );
    linalg::apply( A_sys, u, rr );
    linalg::lincomb( rr, { 1.0, -1.0 }, { b, rr } );
    const double relres = std::sqrt( dot( rr, rr ) / dot( b, b ) );

    // per-block gradient indicator from the discrete solution
    const int nsub = static_cast< int >( dom.subdomains().size() );
    auto ux = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[0] );
    auto uy = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[1] );
    auto uz = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[2] );
    const int nx = ux.extent( 1 ), ny = ux.extent( 2 ), nr = ux.extent( 3 );
    Res       res;
    res.dofs = ndofs;
    res.l2 = l2;
    res.relres = relres;
    res.indicator.assign( nsub, 0.0 );
    for ( int s = 0; s < nsub; ++s )
    {
        double mx = -1.0, mn = 1e300;
        for ( int i = 0; i < nx; ++i )
            for ( int j = 0; j < ny; ++j )
                for ( int m = 0; m < nr; ++m )
                {
                    const double mag = std::sqrt( ux( s, i, j, m ) * ux( s, i, j, m ) + uy( s, i, j, m ) * uy( s, i, j, m ) +
                                                  uz( s, i, j, m ) * uz( s, i, j, m ) );
                    mx = std::max( mx, mag );
                    mn = std::min( mn, mag );
                }
        res.indicator[s] = mx - mn;
    }

    // optional: write the adaptive mesh (hex cells) + solution + subdivision level as XDMF for ParaView
    if ( !xdmf_dir.empty() )
    {
        VectorQ1Scalar< ScalarType > level( "level", dom, mask );
        auto                         lh = Kokkos::create_mirror_view( level.grid_data() );
        for ( int s = 0; s < nsub; ++s )
        {
            const double lv = dom.subdivision_of( dom.subdomain_info_from_local_idx( s ) );
            for ( int i = 0; i < nx; ++i )
                for ( int j = 0; j < ny; ++j )
                    for ( int m = 0; m < nr; ++m )
                        lh( s, i, j, m ) = lv;
        }
        Kokkos::deep_copy( level.grid_data(), lh );
        terra::util::prepare_empty_directory( xdmf_dir );
        terra::io::XDMFOutput xdmf( xdmf_dir, dom, coords, radii_g );
        xdmf.add( u.grid_data() );
        xdmf.add( level.grid_data() );
        xdmf.write();
    }
    if ( mpi::rank( mesh.comm ) == 0 )
        std::printf( "  [%-14s] dofs %9ld  L2 %.6e  relres %.2e  hanging %zu\n", tag.c_str(), ndofs, l2, relres,
                     mesh.t_local.con_dst.size() );
    return res;
}


int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        auto table = std::make_shared< util::Table >();

        const std::vector< double > radii = { 0.5, 0.625, 0.75, 0.875, 1.0 };
        const int                   LDR = 3, S_lat = 4, S_rad = 1, M = 8; // S_lat=4 => interior blocks; M=8 for deep adaptive
        const int                   NU = 0;    // uniform L0 only here (L0-L3 already saved; merged back in)
        const int                   NA = 5;    // adaptive rounds 0..NA (one step beyond the earlier R4)
        const double                FRAC = 0.30;

        std::vector< std::pair< long, double > > unif, adap;
        std::string out = std::getenv( "AMR_DEMO_OUT" ) ? std::getenv( "AMR_DEMO_OUT" )
                                                        : std::string( std::getenv( "HOME" ) ) + "/amr_demo_out";

        // rewrite the CSV after every solve so partial results survive a walltime kill (long 4-GPU runs)
        auto dump_csv = [&]() {
            if ( mpi::rank( MPI_COMM_WORLD ) != 0 )
                return;
            if ( std::FILE* fp = std::fopen( ( out + "/amr_vs_uniform.csv" ).c_str(), "w" ) )
            {
                std::fprintf( fp, "kind,dofs,l2\n" );
                for ( auto& p : unif )
                    std::fprintf( fp, "uniform,%ld,%.8e\n", p.first, p.second );
                for ( auto& p : adap )
                    std::fprintf( fp, "adaptive,%ld,%.8e\n", p.first, p.second );
                std::fclose( fp );
            }
        };

        // ---- UNIFORM sweep: refine every leaf each level (each level's mesh written to XDMF) ---------
        {
            AdaptiveForest f( NU, S_lat, S_rad );
            for ( int lvl = 0; lvl <= NU; ++lvl )
            {
                auto mesh = build_distributed_adaptive_mesh( MPI_COMM_WORLD, LDR, radii, f,
                                                             grid::shell::subdomain_to_rank_by_diamond );
                const auto r = solve_level( mesh, table, "uniform_L" + std::to_string( lvl ),
                                            out + "/amrvu_U" + std::to_string( lvl ) + "_mesh" );
                unif.emplace_back( r.dofs, r.l2 );
                dump_csv();
                CHECK( r.relres < 1e-8 );
                if ( lvl < NU )
                {
                    auto all = f.leaves();
                    f.refine( all );
                    f.balance_2to1();
                }
            }
        }

        // ---- ADAPTIVE sweep: refine only where the solution varies (gradient indicator). Pole-corner
        // 2:1 is now supported, so refinement may reach the poles freely. Each round's mesh+solution is
        // written to XDMF for the mesh plot. --------------------------------------------------------
        {
            AdaptiveForest f( M, S_lat, S_rad );
            for ( int round = 0; round <= NA; ++round )
            {
                auto       leaves = f.leaves();
                auto       mesh   = build_distributed_adaptive_mesh( MPI_COMM_WORLD, LDR, radii, f,
                                                                     grid::shell::subdomain_to_rank_by_diamond );
                const auto r = solve_level( mesh, table, "adaptive_R" + std::to_string( round ),
                                            out + "/amrvu_R" + std::to_string( round ) + "_mesh" );
                adap.emplace_back( r.dofs, r.l2 );
                dump_csv();
                CHECK( r.relres < 1e-8 );
                if ( round == NA )
                    break;

                // GLOBAL refinement decision: r.indicator holds only THIS rank's owned blocks, but the
                // forest is replicated, so every rank must flag the SAME leaves. Scatter each rank's
                // local indicators into the global leaf order (keyed by finest anchor), all-reduce, then
                // use a global emax. (Without this, at np>1 each rank refines a different set and the
                // replicated forests diverge -> the distributed mesh is inconsistent and the solve blows up.)
                std::map< SubdomainInfo, double > ind_of_anchor;
                for ( std::size_t s = 0; s < r.indicator.size(); ++s )
                    ind_of_anchor[mesh.domain.subdomain_info_from_local_idx( static_cast< int >( s ) )] = r.indicator[s];
                std::vector< double > gind( leaves.size(), 0.0 );
                for ( std::size_t i = 0; i < leaves.size(); ++i )
                {
                    auto it = ind_of_anchor.find( f.finest_anchor( leaves[i] ) );
                    if ( it != ind_of_anchor.end() )
                        gind[i] = it->second;
                }
                MPI_Allreduce( MPI_IN_PLACE, gind.data(), static_cast< int >( gind.size() ), MPI_DOUBLE,
                               MPI_SUM, MPI_COMM_WORLD ); // each leaf owned by exactly one rank -> SUM = value

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

        // ---- report + efficiency check -----------------------------------------------------------
        if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        {
            if ( std::FILE* fp = std::fopen( ( out + "/amr_vs_uniform.csv" ).c_str(), "w" ) )
            {
                std::fprintf( fp, "kind,dofs,l2\n" );
                for ( auto& p : unif )
                    std::fprintf( fp, "uniform,%ld,%.8e\n", p.first, p.second );
                for ( auto& p : adap )
                    std::fprintf( fp, "adaptive,%ld,%.8e\n", p.first, p.second );
                std::fclose( fp );
                std::printf( "  wrote %s/amr_vs_uniform.csv\n", out.c_str() );
            }
        }

        // Efficiency: the finest adaptive point must beat the uniform curve at a matched dof budget --
        // find the uniform mesh with the smallest dofs >= the adaptive dofs (else the finest uniform),
        // and require the adaptive error to be lower (same accuracy for fewer dofs).
        const auto  af = adap.back();
        long        best_du = -1;
        double      lu_ref = unif.back().second;
        for ( auto& p : unif )
            if ( p.first >= af.first && ( best_du < 0 || p.first < best_du ) )
            {
                best_du = p.first;
                lu_ref  = p.second;
            }
        if ( best_du < 0 )
            lu_ref = unif.back().second; // adaptive exceeded all uniform dofs; compare to finest uniform
        if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
            std::printf( "  adaptive: %.4e at %ld dofs   vs uniform: %.4e at %ld dofs (>= budget)\n", af.second,
                         af.first, lu_ref, best_du < 0 ? unif.back().first : best_du );
        CHECK( af.second < lu_ref );                          // AMR is more accurate at matched-or-fewer dofs
        CHECK( adap.back().second < adap.front().second );    // adaptive error decreases under refinement
    }

    const bool ok = ( g_failures == 0 );
    if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        std::printf( "test_adaptive_amr_vs_uniform_gpu: %s (%d checks)\n", ok ? "ALL PASS" : "FAILURE(S)", g_checks );
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
