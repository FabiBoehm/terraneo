// The warp-owner path of UnsteadyAdvectionDiffusionSUPG against the per-cell path.
//
// Same operator -- mass, advection, diffusion and the SUPG term -- on the same
// temperature and velocity fields, for the boundary treatment on and off, the lumped
// and the consistent mass, the full operator and its diagonal, in Replace and Add
// mode. In double the two paths must agree to roundoff; the float arm reports how
// far single-precision fields and geometry move the result. On a Serial build both
// paths are the per-cell path and the test passes trivially.

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

template < typename S >
struct VectorFieldInterpolator
{
    Grid3DDataVec< S, 3 > grid_;
    Grid2DDataScalar< S > radii_;
    Grid4DDataVec< S, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        // A rotation plus a radial swirl: O(1) velocities, so the Peclet number is large
        // somewhere and small elsewhere (both branches of tau get exercised).
        data_( local_subdomain_id, x, y, r, 0 ) = -c( 1 ) + 0.3 * Kokkos::sin( 2.0 * c( 2 ) );
        data_( local_subdomain_id, x, y, r, 1 ) = c( 0 ) + 0.3 * Kokkos::cos( 3.0 * c( 0 ) );
        data_( local_subdomain_id, x, y, r, 2 ) = 0.5 * Kokkos::sin( c( 0 ) * c( 1 ) );
    }
};

template < typename S >
struct ScalarFieldInterpolator
{
    Grid3DDataVec< S, 3 > grid_;
    Grid2DDataScalar< S > radii_;
    Grid4DDataScalar< S > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        data_( local_subdomain_id, x, y, r ) =
            1.0 + 0.3 * c( 0 ) * c( 1 ) + Kokkos::cos( 2.5 * c( 2 ) ) * Kokkos::sin( 1.5 * c( 0 ) );
    }
};

struct Config
{
    const char* name;
    bool        treat_boundary, diagonal, lumped;
    double      kappa, dt;
};

struct Arm
{
    double inf, l2, add_inf;
};

template < typename S, typename G >
Arm run_arm( const DistributedDomain& dom, const Config& c, const VectorQ1Scalar< double >& ref,
             const VectorQ1Scalar< double >& add_ref, const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask,
             const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& bmask )
{
    using Op = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< S, 3, G, S >;

    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( dom );
    const auto radii  = grid::shell::subdomain_shell_radii< G >( dom );

    VectorQ1Scalar< S > T( "T", dom, mask );
    VectorQ1Vec< S >    u( "u", dom, mask );
    {
        VectorQ1Scalar< G > Tg( "Tg", dom, mask );
        VectorQ1Vec< G >    ug( "ug", dom, mask );
        Kokkos::parallel_for( "T", grid::shell::local_domain_md_range_policy_nodes( dom ),
                              ScalarFieldInterpolator< G >{ coords, radii, Tg.grid_data() } );
        Kokkos::parallel_for( "u", grid::shell::local_domain_md_range_policy_nodes( dom ),
                              VectorFieldInterpolator< G >{ coords, radii, ug.grid_data() } );
        Kokkos::fence();
        linalg::convert( Tg, T );
        linalg::convert( ug, u );
    }

    // Both paths in S; the double reference handed in is used for the double arm
    // only (for a reduced precision the per-cell path in S is the reference, so
    // the comparison isolates the kernel, not the precision).
    auto run = [&]( bool fast, VectorQ1Scalar< S >& y, VectorQ1Scalar< S >& y_add ) {
        Op op( dom, coords, radii, bmask, u, util::narrow< S >( c.kappa ), util::narrow< S >( c.dt ), c.treat_boundary,
               c.diagonal, util::narrow< S >( 1.5 ), c.lumped );
        op.set_fast_path( fast );
        linalg::apply( op, T, y );
        linalg::assign( y_add, T );
        op.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Add,
                                                       linalg::OperatorCommunicationMode::CommunicateAdditively );
        linalg::apply( op, T, y_add );
    };
    VectorQ1Scalar< S > y( "y", dom, mask ), y_add( "y_add", dom, mask );
    run( true, y, y_add );

    VectorQ1Scalar< double > ref_s( "ref_s", dom, mask ), add_ref_s( "add_ref_s", dom, mask );
    if constexpr ( std::is_same_v< S, double > )
    {
        linalg::assign( ref_s, ref );
        linalg::assign( add_ref_s, add_ref );
    }
    else
    {
        VectorQ1Scalar< S > ys( "ys", dom, mask ), ys_add( "ys_add", dom, mask );
        run( false, ys, ys_add );
        linalg::convert( ys, ref_s );
        linalg::convert( ys_add, add_ref_s );
    }

    VectorQ1Scalar< double > y_d( "y_d", dom, mask ), ya_d( "ya_d", dom, mask );
    linalg::convert( y, y_d );
    linalg::convert( y_add, ya_d );
    const double ref_inf = linalg::norm_inf( ref_s );
    const double ref_l2  = std::sqrt( static_cast< double >( linalg::dot( ref_s, ref_s ) ) );
    linalg::lincomb( y_d, { 1.0, -1.0 }, { y_d, ref_s } );
    linalg::lincomb( ya_d, { 1.0, -1.0 }, { ya_d, add_ref_s } );
    Arm a;
    a.inf     = linalg::norm_inf( y_d ) / ref_inf;
    a.l2      = std::sqrt( static_cast< double >( linalg::dot( y_d, y_d ) ) ) / ref_l2;
    a.add_inf = linalg::norm_inf( ya_d ) / linalg::norm_inf( add_ref_s );
    return a;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int failures = 0;
    {
        const int min_level = ( argc > 1 ) ? std::atoi( argv[1] ) : 3;
        const int max_level = ( argc > 2 ) ? std::atoi( argv[2] ) : 5;

        using Op    = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< double >;
        using Half  = Kokkos::Experimental::half_t;
        using BHalf = Kokkos::Experimental::bhalf_t;

        const Config configs[] = {
            { "boundary, consistent mass", true, false, false, 1e-2, 1e-3 },
            { "boundary, lumped mass", true, false, true, 1e-2, 1e-3 },
            { "no boundary treatment", false, false, false, 1.0, 1e-2 },
            { "diagonal", true, true, false, 1e-2, 1e-3 },
            { "diagonal, lumped", true, true, true, 1e-4, 1e-3 },
            { "advection dominated", true, false, false, 1e-6, 1e-3 },
        };

        for ( int level = min_level; level <= max_level; ++level )
        {
            const auto dom   = DistributedDomain::create_uniform( level, level, 1.22, 2.22, 0, 0 );
            auto       mask  = grid::setup_node_ownership_mask_data( dom );
            auto       bmask = grid::shell::setup_boundary_mask_data( dom );

            const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom );
            const auto radii  = grid::shell::subdomain_shell_radii< double >( dom );

            VectorQ1Scalar< double > T( "T_ref", dom, mask );
            VectorQ1Vec< double >    u( "u_ref", dom, mask );
            Kokkos::parallel_for( "T", grid::shell::local_domain_md_range_policy_nodes( dom ),
                                  ScalarFieldInterpolator< double >{ coords, radii, T.grid_data() } );
            Kokkos::parallel_for( "u", grid::shell::local_domain_md_range_policy_nodes( dom ),
                                  VectorFieldInterpolator< double >{ coords, radii, u.grid_data() } );
            Kokkos::fence();

            for ( const auto& c : configs )
            {
                Op ref_op( dom, coords, radii, bmask, u, c.kappa, c.dt, c.treat_boundary, c.diagonal, 1.5, c.lumped );
                ref_op.set_fast_path( false );
                VectorQ1Scalar< double > ref( "ref", dom, mask ), add_ref( "add_ref", dom, mask );
                linalg::apply( ref_op, T, ref );
                linalg::assign( add_ref, T );
                ref_op.set_operator_apply_and_communication_modes(
                    linalg::OperatorApplyMode::Add, linalg::OperatorCommunicationMode::CommunicateAdditively );
                linalg::apply( ref_op, T, add_ref );

                Op probe( dom, coords, radii, bmask, u, c.kappa, c.dt, c.treat_boundary, c.diagonal, 1.5, c.lumped );
                std::cout << "level " << level << "  " << std::setw( 26 ) << c.name << "  warp path "
                          << ( probe.fast_path_runs() ? "on" : "OFF (per-cell both)" ) << std::endl;

                auto report = [&]( const char* name, const Arm& a, double tol ) {
                    const bool ok = a.inf <= tol && a.add_inf <= tol && std::isfinite( a.l2 );
                    std::cout << "    " << std::setw( 7 ) << name << "  inf " << std::scientific << std::setprecision( 2 )
                              << a.inf << "  l2 " << a.l2 << "  add inf " << a.add_inf << ( ok ? "" : "   <-- FAIL" )
                              << std::endl;
                    if ( !ok )
                        ++failures;
                };
                report( "double", run_arm< double, double >( dom, c, ref, add_ref, mask, bmask ), 1e-10 );
                report( "float", run_arm< float, float >( dom, c, ref, add_ref, mask, bmask ), 2e-5 );
                report( "flt/dg", run_arm< float, double >( dom, c, ref, add_ref, mask, bmask ), 2e-5 );
                // 16-bit storage and arithmetic on float geometry (informational
                // tolerances: the per-cell path rounds every atomic add into the 16-bit
                // destination, the warp path reduces in 16-bit registers).
                report( "bf16", run_arm< BHalf, float >( dom, c, ref, add_ref, mask, bmask ), 2e-1 );
                report( "fp16", run_arm< Half, float >( dom, c, ref, add_ref, mask, bmask ), 5e-2 );
            }
        }
    }

    std::cout << ( failures == 0 ? "PASSED" : "FAILED" ) << " (" << failures << " arm(s) above tolerance)" << std::endl;
    return failures == 0 ? 0 : 1;
}
