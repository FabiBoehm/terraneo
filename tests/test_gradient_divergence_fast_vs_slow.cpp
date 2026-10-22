// The warp-owner paths of Gradient and Divergence against the per-cell paths they
// stand in for.
//
// Same bilinear forms, same fields, same boundary conditions; the only difference is
// the kernel. In double the two must agree to roundoff for every boundary set in
// Replace and in Add mode; the float arm compares the float warp path against the
// float per-cell path the same way. Independently of the reference, the
// divergence must be the transpose of the gradient: < G p, u > = < p, D u >.
//
// On a Serial build both paths are the per-cell path and the test is a no-op that
// passes; on a device it compares the warp path against the per-cell path.

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/divergence.hpp"
#include "fe/wedge/operators/shell/gradient.hpp"
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
using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
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
        data_( local_subdomain_id, x, y, r, 0 ) = Kokkos::sin( 2.0 * c( 0 ) ) * Kokkos::cosh( c( 1 ) );
        data_( local_subdomain_id, x, y, r, 1 ) = Kokkos::sin( 3.0 * c( 1 ) ) * Kokkos::cosh( c( 2 ) );
        data_( local_subdomain_id, x, y, r, 2 ) = Kokkos::cos( 4.0 * c( 2 ) ) * Kokkos::cosh( c( 0 ) );
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

struct Arm
{
    double grad_inf, grad_l2, div_inf, div_l2, grad_add_inf, transpose_rel;
};

/// Fields and operators in S (geometry in G), the per-cell path as the reference.
template < typename S, typename G >
Arm run_arm( const DistributedDomain& dom_f, const DistributedDomain& dom_c, BoundaryConditions bcs,
             const VectorQ1Vec< double >& grad_ref, const VectorQ1Scalar< double >& div_ref,
             const VectorQ1Vec< double >& grad_add_ref,
             const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_f,
             const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_c,
             const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& bmask_f,
             bool double_reference )
{
    using Grad = fe::wedge::operators::shell::Gradient< S, G, S >;
    using Div  = fe::wedge::operators::shell::Divergence< S, G, S >;

    const auto coords_f = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( dom_f );
    const auto radii_f  = grid::shell::subdomain_shell_radii< G >( dom_f );
    const auto coords_c = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( dom_c );
    const auto radii_c  = grid::shell::subdomain_shell_radii< G >( dom_c );

    VectorQ1Scalar< S > p( "p", dom_c, mask_c );
    VectorQ1Vec< S >    u( "u", dom_f, mask_f );
    {
        // The fields are generated in G and narrowed, so every arm sees the same
        // function up to its own storage precision.
        VectorQ1Scalar< G > pg( "pg", dom_c, mask_c );
        VectorQ1Vec< G >    ug( "ug", dom_f, mask_f );
        Kokkos::parallel_for( "p", grid::shell::local_domain_md_range_policy_nodes( dom_c ),
                              ScalarFieldInterpolator< G >{ coords_c, radii_c, pg.grid_data() } );
        Kokkos::parallel_for( "u", grid::shell::local_domain_md_range_policy_nodes( dom_f ),
                              VectorFieldInterpolator< G >{ coords_f, radii_f, ug.grid_data() } );
        Kokkos::fence();
        linalg::convert( pg, p );
        linalg::convert( ug, u );
    }

    Grad grad( dom_f, dom_c, coords_f, radii_f, bmask_f, bcs );
    Div  div( dom_f, dom_c, coords_f, radii_f, bmask_f, bcs );

    VectorQ1Vec< S >    gp( "gp", dom_f, mask_f );
    VectorQ1Scalar< S > du( "du", dom_c, mask_c );
    linalg::apply( grad, p, gp );
    linalg::apply( div, u, du );

    // The reference of a reduced-precision arm is the per-cell path in the same
    // precision: what the comparison isolates is the kernel, not the precision (a
    // float free-slip projection differs from double by 1e-4 in the max norm at level
    // 6, in either path).
    VectorQ1Vec< double >    grad_ref_s( "grad_ref_s", dom_f, mask_f );
    VectorQ1Scalar< double > div_ref_s( "div_ref_s", dom_c, mask_c );
    VectorQ1Vec< double >    grad_add_ref_s( "grad_add_ref_s", dom_f, mask_f );
    if ( !double_reference )
    {
        Grad grad_s( dom_f, dom_c, coords_f, radii_f, bmask_f, bcs );
        Div  div_s( dom_f, dom_c, coords_f, radii_f, bmask_f, bcs );
        grad_s.set_fast_path( false );
        div_s.set_fast_path( false );
        VectorQ1Vec< S >    gp_s( "gp_s", dom_f, mask_f );
        VectorQ1Scalar< S > du_s( "du_s", dom_c, mask_c );
        linalg::apply( grad_s, p, gp_s );
        linalg::apply( div_s, u, du_s );
        linalg::convert( gp_s, grad_ref_s );
        linalg::convert( du_s, div_ref_s );
        VectorQ1Vec< S > gpa_s( "gpa_s", dom_f, mask_f );
        linalg::assign( gpa_s, u );
        grad_s.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Add,
                                                           linalg::OperatorCommunicationMode::CommunicateAdditively );
        linalg::apply( grad_s, p, gpa_s );
        linalg::convert( gpa_s, grad_add_ref_s );
    }
    else
    {
        linalg::assign( grad_ref_s, grad_ref );
        linalg::assign( div_ref_s, div_ref );
        linalg::assign( grad_add_ref_s, grad_add_ref );
    }

    // Transpose check in the arm's own precision.
    const double lhs = static_cast< double >( linalg::dot( gp, u ) );
    const double rhs = static_cast< double >( linalg::dot( p, du ) );
    const double transpose_rel = std::abs( lhs - rhs ) / std::max( std::abs( lhs ), 1e-300 );

    // Add mode: dst holds u beforehand (the Stokes operator adds B^T p onto A u with
    // the communication still pending, so the seams of the prefilled vector are
    // summed too; the reference was produced the same way).
    VectorQ1Vec< S > gp_add( "gp_add", dom_f, mask_f );
    linalg::assign( gp_add, u );
    grad.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Add,
                                                     linalg::OperatorCommunicationMode::CommunicateAdditively );
    linalg::apply( grad, p, gp_add );

    // Compare against the double reference.
    VectorQ1Vec< double >    gp_d( "gp_d", dom_f, mask_f );
    VectorQ1Scalar< double > du_d( "du_d", dom_c, mask_c );
    VectorQ1Vec< double >    gpa_d( "gpa_d", dom_f, mask_f );
    linalg::convert( gp, gp_d );
    linalg::convert( du, du_d );
    linalg::convert( gp_add, gpa_d );

    const double gref_inf = linalg::norm_inf( grad_ref_s );
    const double gref_l2  = std::sqrt( linalg::dot( grad_ref_s, grad_ref_s ) );
    const double dref_inf = linalg::norm_inf( div_ref_s );
    const double dref_l2  = std::sqrt( linalg::dot( div_ref_s, div_ref_s ) );

    linalg::lincomb( gpa_d, { 1.0, -1.0 }, { gpa_d, grad_add_ref_s } );
    linalg::lincomb( gp_d, { 1.0, -1.0 }, { gp_d, grad_ref_s } );
    linalg::lincomb( du_d, { 1.0, -1.0 }, { du_d, div_ref_s } );

    Arm a;
    a.grad_inf      = linalg::norm_inf( gp_d ) / gref_inf;
    a.grad_l2       = std::sqrt( linalg::dot( gp_d, gp_d ) ) / gref_l2;
    a.div_inf       = linalg::norm_inf( du_d ) / dref_inf;
    a.div_l2        = std::sqrt( linalg::dot( du_d, du_d ) ) / dref_l2;
    a.grad_add_inf  = linalg::norm_inf( gpa_d ) / linalg::norm_inf( grad_add_ref_s );
    a.transpose_rel = transpose_rel;
    return a;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int failures = 0;
    {
        const int min_level = ( argc > 1 ) ? std::atoi( argv[1] ) : 3;
        const int max_level = ( argc > 2 ) ? std::atoi( argv[2] ) : 5;

        using Grad = fe::wedge::operators::shell::Gradient< double >;
        using Div  = fe::wedge::operators::shell::Divergence< double >;

        struct BCSet
        {
            const char*        name;
            BoundaryConditions bcs;
        };
        BCSet bc_sets[] = {
            { "dirichlet/dirichlet", { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } } },
            { "freeslip/freeslip", { { CMB, FREESLIP }, { SURFACE, FREESLIP } } },
            { "neumann/neumann", { { CMB, NEUMANN }, { SURFACE, NEUMANN } } },
            { "dirichlet/freeslip", { { CMB, DIRICHLET }, { SURFACE, FREESLIP } } },
            { "freeslip/neumann", { { CMB, FREESLIP }, { SURFACE, NEUMANN } } },
        };

        for ( int level = min_level; level <= max_level; ++level )
        {
            const auto dom_f = DistributedDomain::create_uniform( level, level, 1.22, 2.22, 0, 0 );
            const auto dom_c = DistributedDomain::create_uniform( level - 1, level - 1, 1.22, 2.22, 0, 0 );

            auto mask_f  = grid::setup_node_ownership_mask_data( dom_f );
            auto mask_c  = grid::setup_node_ownership_mask_data( dom_c );
            auto bmask_f = grid::shell::setup_boundary_mask_data( dom_f );

            const auto coords_f = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom_f );
            const auto radii_f  = grid::shell::subdomain_shell_radii< double >( dom_f );
            const auto coords_c = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom_c );
            const auto radii_c  = grid::shell::subdomain_shell_radii< double >( dom_c );

            VectorQ1Scalar< double > p( "p_ref", dom_c, mask_c );
            VectorQ1Vec< double >    u( "u_ref", dom_f, mask_f );
            Kokkos::parallel_for( "p", grid::shell::local_domain_md_range_policy_nodes( dom_c ),
                                  ScalarFieldInterpolator< double >{ coords_c, radii_c, p.grid_data() } );
            Kokkos::parallel_for( "u", grid::shell::local_domain_md_range_policy_nodes( dom_f ),
                                  VectorFieldInterpolator< double >{ coords_f, radii_f, u.grid_data() } );
            Kokkos::fence();

            for ( auto& bc : bc_sets )
            {
                // Reference: the per-cell path in double.
                Grad grad_ref( dom_f, dom_c, coords_f, radii_f, bmask_f, bc.bcs );
                Div  div_ref( dom_f, dom_c, coords_f, radii_f, bmask_f, bc.bcs );
                grad_ref.set_fast_path( false );
                div_ref.set_fast_path( false );
                VectorQ1Vec< double >    gp( "gp_ref", dom_f, mask_f );
                VectorQ1Scalar< double > du( "du_ref", dom_c, mask_c );
                linalg::apply( grad_ref, p, gp );
                linalg::apply( div_ref, u, du );
                VectorQ1Vec< double > gp_add( "gp_add_ref", dom_f, mask_f );
                linalg::assign( gp_add, u );
                grad_ref.set_operator_apply_and_communication_modes(
                    linalg::OperatorApplyMode::Add, linalg::OperatorCommunicationMode::CommunicateAdditively );
                linalg::apply( grad_ref, p, gp_add );

                const double lhs = static_cast< double >( linalg::dot( gp, u ) ), rhs = static_cast< double >( linalg::dot( p, du ) );
                const double ref_transpose = std::abs( lhs - rhs ) / std::abs( lhs );

                Grad probe( dom_f, dom_c, coords_f, radii_f, bmask_f, bc.bcs );
                std::cout << "level " << level << "  " << std::setw( 20 ) << bc.name
                          << "  warp path " << ( probe.fast_path_runs() ? "on" : "OFF (per-cell both)" )
                          << "  reference transpose rel " << std::scientific << std::setprecision( 2 )
                          << ref_transpose << std::endl;

                // The transpose identity is checked with the arm's own dot products, whose
                // accumulation is in the arm's precision: float gets ~1e-4 over 1e5 terms.
                auto report = [&]( const char* name, const Arm& a, double tol, double transpose_tol ) {
                    const bool ok = a.grad_inf <= tol && a.div_inf <= tol && a.grad_add_inf <= tol &&
                                    a.transpose_rel <= transpose_tol && std::isfinite( a.grad_l2 ) &&
                                    std::isfinite( a.div_l2 );
                    std::cout << "    " << std::setw( 7 ) << name << "  grad inf " << a.grad_inf << " l2 " << a.grad_l2
                              << "  div inf " << a.div_inf << " l2 " << a.div_l2 << "  grad(add) inf "
                              << a.grad_add_inf << "  transpose " << a.transpose_rel << ( ok ? "" : "   <-- FAIL" )
                              << std::endl;
                    if ( !ok )
                        ++failures;
                };

                report( "double", run_arm< double, double >( dom_f, dom_c, bc.bcs, gp, du, gp_add, mask_f, mask_c, bmask_f, true ), 1e-11, 1e-10 );
                report( "float", run_arm< float, float >( dom_f, dom_c, bc.bcs, gp, du, gp_add, mask_f, mask_c, bmask_f, false ), 2e-5, 2e-3 );
            }
        }
    }

    std::cout << ( failures == 0 ? "PASSED" : "FAILED" ) << " (" << failures << " arm(s) above tolerance)" << std::endl;
    return failures == 0 ? 0 : 1;
}
