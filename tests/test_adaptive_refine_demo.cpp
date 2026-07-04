// AMR demonstration: solution-adaptive refinement driven by a velocity-gradient indicator.
//
// Physics: the production EpsilonDivDivKerngen viscous operator (same as test_adaptive_epsilon_divdiv_gpu),
// forced by a LOCALIZED radial buoyancy blob  f(x) = amp * exp(-|x-x0|^2 / 2sigma^2) * (x/|x|)  --  a
// mantle-plume-like body force centred at an interior (non-pole) point x0. Homogeneous Dirichlet
// (no-slip) on both shell boundaries. The response u is a localized flow concentrated near x0.
//
// Adaptivity: a-posteriori. We SOLVE on the current mesh, form a per-block indicator from the DISCRETE
// solution  eta_K = max_K |u|  (the feature itself), flag the blocks carrying the strongest flow, refine
// them (2:1 balanced), and re-solve. So the refinement hugs the plume core rather than its flanks (a
// gradient indicator max|u|-min|u| would refine the flanks, sitting next to the peak instead of on it).
//
// Output: per refinement round, a CSV of every node (x,y,z, |u|, subdivision level) written to
// $AMR_DEMO_OUT (default ~/amr_demo_out). plot_amr_demo.py renders the refined mesh + solution.
//
// Single rank (np=1): the distributed mesh degenerates to one owner, and forest.leaves()[i] is exactly
// local subdomain i (create_adaptive_for_rank iterates leaves in order), so the per-block indicator maps
// straight back to the leaf to refine. Needs a GPU -- run on an H100 node.

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

// --- viscosity k = 2 + sin z (mild lateral variation), verbatim from the epsilon test ---------------
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

// --- localized radial buoyancy body force (the "plume") ---------------------------------------------
struct BlobForce
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    double                     x0_, y0_, z0_, inv2s2_, amp_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c  = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const double                  dx = c( 0 ) - x0_, dy = c( 1 ) - y0_, dz = c( 2 ) - z0_;
        const double                  d2 = dx * dx + dy * dy + dz * dz;
        const double                  g  = amp_ * Kokkos::exp( -d2 * inv2s2_ );
        const double rn = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) ) + 1e-30;
        data_( s, x, y, r, 0 )           = g * c( 0 ) / rn;
        data_( s, x, y, r, 1 )           = g * c( 1 ) / rn;
        data_( s, x, y, r, 2 )           = g * c( 2 ) / rn;
    }
};

// --- write Cartesian node positions into a vector field (so we can mirror them to host) --------------
struct PositionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r, 0 )          = c( 0 );
        data_( s, x, y, r, 1 )          = c( 1 );
        data_( s, x, y, r, 2 )          = c( 2 );
    }
};

// Host-side results of one solve: per-node geometry+solution, and the per-block indicator.
struct DemoData
{
    std::vector< double > px, py, pz, umag; // per node
    std::vector< int >    level;            // per node (its block's subdivision)
    std::vector< double > indicator;        // per block: max|u| - min|u| over the block
    long                  ndofs = 0;
};

// Solve A u = M f on the given adaptive mesh and read the solution back to the host. Mirrors
// solve_eps_distributed's operator setup exactly, but with the blob force and homogeneous Dirichlet.
static DemoData demo_solve( const DistributedAdaptiveMesh& mesh, double x0, double y0, double z0,
                            double sigma, const std::shared_ptr< util::Table >& table,
                            const std::string& tag, const std::string& xdmf_dir )
{
    using ScalarType  = double;
    using Epsilon     = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarType, 3 >;
    using Mass        = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;
    using Wrapper     = AdaptiveDistributedConstrainedOperator< Epsilon >;
    using MassWrapper = AdaptiveDistributedConstrainedOperator< Mass >;

    const auto& dom   = mesh.domain;
    const auto  mask  = adaptive_ownership_mask( mesh );
    const auto  bmask = adaptive_boundary_mask( dom );

    VectorQ1Vec< ScalarType >    u( "u", dom, mask ), f( "f", dom, mask ), b( "b", dom, mask );
    VectorQ1Vec< ScalarType >    pos( "pos", dom, mask ), rr( "rr", dom, mask );
    VectorQ1Vec< ScalarType >    Adiagg( "Adiagg", dom, mask ), error( "error", dom, mask ), tmp( "tmp", dom, mask );
    VectorQ1Vec< ScalarType >    scratch_a( "sa", dom, mask ), scratch_m( "sm", dom, mask );
    VectorQ1Scalar< ScalarType > k( "k", dom, mask );

    const auto ndofs   = kernels::common::count_masked< long >( mask, grid::NodeOwnershipFlag::OWNED, mesh.comm );
    const auto coords  = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom );
    const auto radii_g = grid::shell::subdomain_shell_radii< ScalarType >( dom );

    Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          KInterpolator{ coords, radii_g, k.grid_data() } );
    Kokkos::parallel_for( "f", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          BlobForce{ coords, radii_g, f.grid_data(), x0, y0, z0,
                                     1.0 / ( 2.0 * sigma * sigma ), 1.0 } );
    Kokkos::parallel_for( "pos", grid::shell::local_domain_md_range_policy_nodes( dom ),
                          PositionInterpolator{ coords, radii_g, pos.grid_data() } );
    Kokkos::fence();

    BoundaryConditions bcs_nn = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };
    Epsilon A_loc( dom, coords, radii_g, bmask, k.grid_data(), bcs_nn, false,
                   linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::SkipCommunication );
    Mass    M_loc( dom, coords, radii_g, false, linalg::OperatorApplyMode::Replace,
                linalg::OperatorCommunicationMode::SkipCommunication );

    Wrapper     A_sys( A_loc, mesh, scratch_a, bmask, grid::shell::ShellBoundaryFlag::BOUNDARY );
    MassWrapper M_c( M_loc, mesh, scratch_m );

    // Homogeneous Dirichlet: b = C^T M C f, boundary rows zeroed (no lifting term, g == 0).
    linalg::apply( M_c, f, b );
    kernels::common::assign_masked_else_keep_old( b.grid_data(), 0.0, bmask,
                                                  grid::shell::ShellBoundaryFlag::BOUNDARY );
    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 4000, 1e-10, 1e-10 };
    linalg::solvers::PCG< Wrapper >            pcg( solver_params, table, { tmp, Adiagg, error, rr } );
    pcg.set_tag( tag );
    linalg::solvers::solve( pcg, A_sys, u, b );
    Kokkos::fence();
    apply_constraint_device( mesh.t_local_d, u.grid_data() ); // final conformity at hanging nodes

    const int nsub = static_cast< int >( dom.subdomains().size() );

    // --- per-node refinement-level field, so ParaView can colour the mesh by subdivision ------------
    VectorQ1Scalar< ScalarType > level( "level", dom, mask );
    {
        auto      lh  = Kokkos::create_mirror_view( level.grid_data() );
        const int lnx = static_cast< int >( lh.extent( 1 ) ), lny = static_cast< int >( lh.extent( 2 ) ),
                  lnr = static_cast< int >( lh.extent( 3 ) );
        for ( int s = 0; s < nsub; ++s )
        {
            const double lv = dom.subdivision_of( dom.subdomain_info_from_local_idx( s ) );
            for ( int x = 0; x < lnx; ++x )
                for ( int y = 0; y < lny; ++y )
                    for ( int r = 0; r < lnr; ++r )
                        lh( s, x, y, r ) = lv;
        }
        Kokkos::deep_copy( level.grid_data(), lh );
    }

    // --- write the adaptive mesh (hex cells) + solution + level as XDMF/HDF for ParaView ------------
    {
        terra::util::prepare_empty_directory( xdmf_dir );
        terra::io::XDMFOutput xdmf( xdmf_dir, dom, coords, radii_g );
        xdmf.add( u.grid_data() );
        xdmf.add( level.grid_data() );
        xdmf.write();
    }

    // --- read solution + positions to the host, build per-node and per-block arrays -----------------
    auto      ux_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[0] );
    auto      uy_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[1] );
    auto      uz_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[2] );
    auto      px_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, pos.grid_data().comp_[0] );
    auto      py_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, pos.grid_data().comp_[1] );
    auto      pz_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, pos.grid_data().comp_[2] );

    const int nx = static_cast< int >( ux_h.extent( 1 ) );
    const int ny = static_cast< int >( ux_h.extent( 2 ) );
    const int nr = static_cast< int >( ux_h.extent( 3 ) );

    DemoData d;
    d.ndofs = ndofs;
    d.indicator.assign( nsub, 0.0 );
    for ( int s = 0; s < nsub; ++s )
    {
        const int level = dom.subdivision_of( dom.subdomain_info_from_local_idx( s ) );
        double    umax = -1.0;
        for ( int x = 0; x < nx; ++x )
            for ( int y = 0; y < ny; ++y )
                for ( int r = 0; r < nr; ++r )
                {
                    const double mag = std::sqrt( ux_h( s, x, y, r ) * ux_h( s, x, y, r ) +
                                                  uy_h( s, x, y, r ) * uy_h( s, x, y, r ) +
                                                  uz_h( s, x, y, r ) * uz_h( s, x, y, r ) );
                    d.px.push_back( px_h( s, x, y, r ) );
                    d.py.push_back( py_h( s, x, y, r ) );
                    d.pz.push_back( pz_h( s, x, y, r ) );
                    d.umag.push_back( mag );
                    d.level.push_back( level );
                    umax = std::max( umax, mag );
                }
        d.indicator[s] = umax; // track the feature itself: refine where |u| is largest (the plume core)
    }
    std::printf( "  [%s] blocks %d  dofs %ld\n", tag.c_str(), nsub, ndofs );
    return d;
}

static void write_csv( const std::string& path, const DemoData& d )
{
    std::FILE* fp = std::fopen( path.c_str(), "w" );
    if ( !fp )
    {
        std::printf( "  WARN: cannot open %s\n", path.c_str() );
        return;
    }
    std::fprintf( fp, "x,y,z,umag,level\n" );
    for ( std::size_t i = 0; i < d.px.size(); ++i )
        std::fprintf( fp, "%.6f,%.6f,%.6f,%.6e,%d\n", d.px[i], d.py[i], d.pz[i], d.umag[i], d.level[i] );
    std::fclose( fp );
    std::printf( "  wrote %s  (%zu nodes)\n", path.c_str(), d.px.size() );
}

// A block touches a pole (diamond corner) -> refining it would trip the global 2:1 corner guard.
static bool is_diamond_corner( const SubdomainInfo& si, int S_lat, int subdivision )
{
    const int L    = S_lat * ( 1 << subdivision );
    const bool xend = ( si.subdomain_x() == 0 || si.subdomain_x() == L - 1 );
    const bool yend = ( si.subdomain_y() == 0 || si.subdomain_y() == L - 1 );
    return xend && yend;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        auto table = std::make_shared< util::Table >();

        const std::vector< double > radii = { 0.5, 0.625, 0.75, 0.875, 1.0 };
        const int                   LDR = 4, S_lat = 4, S_rad = 1, M = 3; // 3 refinement levels budget
        const int                   ROUNDS = 3;
        const double                REFINE_FRAC = 0.30; // flag blocks with eta >= 0.30 * max eta

        // interior (equatorial, off-pole) plume centre at mid-radius
        const double dnorm = std::sqrt( 1.0 + 0.6 * 0.6 + 0.2 * 0.2 );
        const double r0 = 0.78, x0 = r0 * 1.0 / dnorm, y0 = r0 * 0.6 / dnorm, z0 = r0 * 0.2 / dnorm;
        const double sigma = 0.11;

        std::string outdir = std::getenv( "AMR_DEMO_OUT" ) ? std::getenv( "AMR_DEMO_OUT" )
                                                           : std::string( std::getenv( "HOME" ) ) + "/amr_demo_out";

        AdaptiveForest f( M, S_lat, S_rad ); // starts uniform (no leaves refined)

        for ( int round = 0; round <= ROUNDS; ++round )
        {
            auto       leaves = f.leaves(); // snapshot: local subdomain i <-> leaves[i]
            const auto mesh    = build_distributed_adaptive_mesh( MPI_COMM_WORLD, LDR, radii, f,
                                                         grid::shell::subdomain_to_rank_by_diamond );
            const auto tag     = "round" + std::to_string( round );
            const auto d       = demo_solve( mesh, x0, y0, z0, sigma, table, tag, outdir + "/" + tag + "_mesh" );
            write_csv( outdir + "/amr_demo_" + tag + ".csv", d );

            if ( round == ROUNDS )
                break; // final mesh solved + exported; no further refinement

            // --- indicator-driven refinement: flag the fastest-varying non-corner blocks ------------
            double emax = 0.0;
            for ( double e : d.indicator )
                emax = std::max( emax, e );

            std::vector< ForestLeaf > to_refine;
            for ( std::size_t s = 0; s < leaves.size(); ++s )
            {
                const auto& leaf = leaves[s];
                if ( leaf.subdivision >= M )
                    continue;
                if ( is_diamond_corner( leaf.id, S_lat, leaf.subdivision ) )
                    continue;
                if ( d.indicator[s] >= REFINE_FRAC * emax )
                    to_refine.push_back( leaf );
            }
            std::printf( "  round %d: eta_max %.3e  refining %zu / %zu blocks\n", round, emax,
                         to_refine.size(), leaves.size() );
            if ( to_refine.empty() )
                break;
            f.refine( to_refine );
            f.balance_2to1();
            if ( !f.validate() )
            {
                std::printf( "  ERROR: forest invalid after refinement\n" );
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}
