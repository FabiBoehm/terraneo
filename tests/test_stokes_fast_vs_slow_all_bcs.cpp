// Every kernel path of the Stokes operator against the per-cell path it stands in
// for, on all nine CMB/SURFACE combinations of Dirichlet, Neumann and free-slip.
//
// EpsDivDiv: the fast Dirichlet/Neumann (warp-owner) and fast free-slip kernels
// against the slow per-cell path -- matvec and diagonal, hourglass off and on, in
// Replace and Add mode. Gradient and Divergence are covered block by block in
// test_gradient_divergence_fast_vs_slow; here the composed EpsDivDivStokes apply
// (A with its communication deferred, B^T added on top, then B) is checked against
// the same composition out of the three slow paths. Storage double, float, bf16 and
// fp16 with the geometry in double, float, float, float. The double arms must agree
// with the slow path to roundoff; float to single-precision roundoff; the 16-bit
// arms are compared against the double reference (their slow path assembles the
// local matrix in 16 bits and is no reference) and must stay within the storage
// format's resolution.
//
// On a Serial build every path is the slow path and the test passes trivially.

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "linalg/vector_q1.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

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
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using Half  = Kokkos::Experimental::half_t;
using BHalf = Kokkos::Experimental::bhalf_t;

/// Geometry precision of a storage type: itself for 4 and 8 bytes, float below.
template < typename S >
using Geom = std::conditional_t< ( sizeof( S ) >= 4 ), S, float >;

using OwnerMask = Grid4DDataScalar< grid::NodeOwnershipFlag >;
using BMask     = Grid4DDataScalar< grid::shell::ShellBoundaryFlag >;

struct VectorFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        data_( local_subdomain_id, x, y, r, 0 ) = Kokkos::sin( 2.0 * c( 0 ) ) * Kokkos::cosh( c( 1 ) );
        data_( local_subdomain_id, x, y, r, 1 ) = Kokkos::sin( 3.0 * c( 1 ) ) * Kokkos::cosh( c( 2 ) );
        data_( local_subdomain_id, x, y, r, 2 ) = Kokkos::cos( 4.0 * c( 2 ) ) * Kokkos::cosh( c( 0 ) );
    }
};

struct ScalarFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        data_( local_subdomain_id, x, y, r ) =
            1.0 + 0.3 * c( 0 ) * c( 1 ) + Kokkos::cos( 2.5 * c( 2 ) ) * Kokkos::sin( 1.5 * c( 0 ) );
    }
};

struct CoeffInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto c  = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const auto rr = radii_( local_subdomain_id, r );
        data_( local_subdomain_id, x, y, r ) =
            1.0 + 0.1 * ( rr - 1.7 ) + 0.05 * Kokkos::cos( c( 0 ) ) * Kokkos::cosh( 0.25 * c( 1 ) );
    }
};

/// The fields of one level, generated in double.
struct Level
{
    const DistributedDomain&         dom_f;
    const DistributedDomain&         dom_c;
    const OwnerMask&                 mask_f;
    const OwnerMask&                 mask_c;
    const BMask&                     bmask_f;
    const VectorQ1Vec< double >&     u;
    const VectorQ1Scalar< double >&  p;
    const VectorQ1Scalar< double >&  k;
};

/// Relative errors of two double vectors: max norm over the max norm of the reference.
template < typename V >
double rel_inf( V& diff, const V& ref )
{
    linalg::lincomb( diff, { 1.0, -1.0 }, { diff, ref } );
    return linalg::norm_inf( diff ) / std::max( linalg::norm_inf( ref ), 1e-300 );
}

struct EpsErrors
{
    double vs_slow, vs_slow_add, vs_double, vs_double_add;
    bool   warp;
};

/// EpsDivDiv in storage S: fast path against the slow path in S (Replace and Add), and
/// against the double slow-path results handed in.
template < typename S >
EpsErrors eps_arm( const Level& lv, BoundaryConditions bcs, bool diagonal, double hourglass,
                   const VectorQ1Vec< double >& ref, const VectorQ1Vec< double >& ref_add )
{
    using G  = Geom< S >;
    using Op = fe::wedge::operators::shell::EpsilonDivDivKerngen< S, 3, G, S >;

    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( lv.dom_f );
    const auto radii  = grid::shell::subdomain_shell_radii< G >( lv.dom_f );

    VectorQ1Vec< S >    u( "u", lv.dom_f, lv.mask_f );
    VectorQ1Scalar< S > k( "k", lv.dom_f, lv.mask_f );
    VectorQ1Scalar< S > no_gca( "no_gca", lv.dom_f, lv.mask_f );
    linalg::convert( lv.u, u );
    linalg::convert( lv.k, k );
    linalg::assign( no_gca, util::narrow< S >( 0.0 ) );

    Op fast( lv.dom_f, coords, radii, lv.bmask_f, k.grid_data(), bcs, diagonal );
    Op slow( lv.dom_f, coords, radii, lv.bmask_f, k.grid_data(), bcs, diagonal );
    fast.set_hourglass_coefficient( hourglass );
    slow.set_hourglass_coefficient( hourglass );
    // Any stored-matrix mode routes the apply through the slow per-cell path; with no
    // element marked nothing is stored, so this IS the slow path.
    slow.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Selective, 0, no_gca.grid_data() );

    auto run = [&]( Op& op, VectorQ1Vec< S >& y, VectorQ1Vec< S >& y_add ) {
        op.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace,
                                                       linalg::OperatorCommunicationMode::CommunicateAdditively );
        linalg::apply( op, u, y );
        // Add mode onto a prefilled vector whose seams are still unsummed (the Stokes
        // operator's B^T p onto A u); the reference was produced the same way.
        linalg::assign( y_add, u );
        op.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Add,
                                                       linalg::OperatorCommunicationMode::CommunicateAdditively );
        linalg::apply( op, u, y_add );
    };
    VectorQ1Vec< S > yf( "yf", lv.dom_f, lv.mask_f ), yfa( "yfa", lv.dom_f, lv.mask_f );
    VectorQ1Vec< S > ys( "ys", lv.dom_f, lv.mask_f ), ysa( "ysa", lv.dom_f, lv.mask_f );
    run( fast, yf, yfa );
    run( slow, ys, ysa );

    VectorQ1Vec< double > yf_d( "yf_d", lv.dom_f, lv.mask_f ), yfa_d( "yfa_d", lv.dom_f, lv.mask_f );
    VectorQ1Vec< double > ys_d( "ys_d", lv.dom_f, lv.mask_f ), ysa_d( "ysa_d", lv.dom_f, lv.mask_f );
    VectorQ1Vec< double > d( "d", lv.dom_f, lv.mask_f );
    linalg::convert( yf, yf_d );
    linalg::convert( yfa, yfa_d );
    linalg::convert( ys, ys_d );
    linalg::convert( ysa, ysa_d );

    EpsErrors e;
    linalg::assign( d, yf_d );
    e.vs_slow = rel_inf( d, ys_d );
    linalg::assign( d, yfa_d );
    e.vs_slow_add = rel_inf( d, ysa_d );
    linalg::assign( d, yf_d );
    e.vs_double = rel_inf( d, ref );
    linalg::assign( d, yfa_d );
    e.vs_double_add = rel_inf( d, ref_add );
    e.warp = fast.warp_path_runs();
    return e;
}

struct StokesErrors
{
    double u_vs_slow, p_vs_slow, u_vs_double, p_vs_double;
};

/// The composed Stokes apply (all fast paths) against the composition of the slow
/// EpsDivDiv, the per-cell Gradient and the per-cell Divergence in the same storage.
template < typename S >
StokesErrors stokes_arm( const Level& lv, BoundaryConditions bcs, const VectorQ1IsoQ2Q1< double >& ref )
{
    using G      = Geom< S >;
    using Stokes = fe::wedge::operators::shell::EpsDivDivStokes< S, 3, G, S >;
    using A      = typename Stokes::Block11Type;
    using Grad   = typename Stokes::Block12Type;
    using Div    = typename Stokes::Block21Type;

    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( lv.dom_f );
    const auto radii  = grid::shell::subdomain_shell_radii< G >( lv.dom_f );

    VectorQ1IsoQ2Q1< S > x( "x", lv.dom_f, lv.dom_c, lv.mask_f, lv.mask_c );
    VectorQ1Scalar< S >  k( "k", lv.dom_f, lv.mask_f ), no_gca( "no_gca", lv.dom_f, lv.mask_f );
    linalg::convert( lv.u, x.block_1() );
    linalg::convert( lv.p, x.block_2() );
    linalg::convert( lv.k, k );
    linalg::assign( no_gca, util::narrow< S >( 0.0 ) );

    Stokes               K( lv.dom_f, lv.dom_c, coords, radii, lv.bmask_f, k.grid_data(), bcs, false );
    VectorQ1IsoQ2Q1< S > yf( "yf", lv.dom_f, lv.dom_c, lv.mask_f, lv.mask_c );
    linalg::apply( K, x, yf );

    // The same composition out of the slow paths (EpsDivDivStokes::apply_impl).
    A a( lv.dom_f, coords, radii, lv.bmask_f, k.grid_data(), bcs, false );
    a.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Selective, 0, no_gca.grid_data() );
    Grad g( lv.dom_f, lv.dom_c, coords, radii, lv.bmask_f, bcs );
    Div  dv( lv.dom_f, lv.dom_c, coords, radii, lv.bmask_f, bcs );
    g.set_fast_path( false );
    dv.set_fast_path( false );
    VectorQ1IsoQ2Q1< S > ys( "ys", lv.dom_f, lv.dom_c, lv.mask_f, lv.mask_c );
    a.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace,
                                                  linalg::OperatorCommunicationMode::SkipCommunication );
    g.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Add,
                                                  linalg::OperatorCommunicationMode::CommunicateAdditively );
    dv.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace,
                                                   linalg::OperatorCommunicationMode::CommunicateAdditively );
    linalg::apply( a, x.block_1(), ys.block_1() );
    linalg::apply( g, x.block_2(), ys.block_1() );
    linalg::apply( dv, x.block_1(), ys.block_2() );

    VectorQ1IsoQ2Q1< double > yf_d( "yf_d", lv.dom_f, lv.dom_c, lv.mask_f, lv.mask_c );
    VectorQ1IsoQ2Q1< double > ys_d( "ys_d", lv.dom_f, lv.dom_c, lv.mask_f, lv.mask_c );
    VectorQ1IsoQ2Q1< double > d( "d", lv.dom_f, lv.dom_c, lv.mask_f, lv.mask_c );
    linalg::convert( yf.block_1(), yf_d.block_1() );
    linalg::convert( yf.block_2(), yf_d.block_2() );
    linalg::convert( ys.block_1(), ys_d.block_1() );
    linalg::convert( ys.block_2(), ys_d.block_2() );

    StokesErrors e{};
    linalg::assign( d.block_1(), yf_d.block_1() );
    e.u_vs_slow = rel_inf( d.block_1(), ys_d.block_1() );
    linalg::assign( d.block_2(), yf_d.block_2() );
    e.p_vs_slow = rel_inf( d.block_2(), ys_d.block_2() );
    linalg::assign( d.block_1(), yf_d.block_1() );
    e.u_vs_double = rel_inf( d.block_1(), ref.block_1() );
    linalg::assign( d.block_2(), yf_d.block_2() );
    e.p_vs_double = rel_inf( d.block_2(), ref.block_2() );
    return e;
}

/// The double slow composition for the Stokes reference.
VectorQ1IsoQ2Q1< double > stokes_slow_double( const Level& lv, BoundaryConditions bcs )
{
    using Stokes = fe::wedge::operators::shell::EpsDivDivStokes< double >;
    using A      = Stokes::Block11Type;
    using Grad   = Stokes::Block12Type;
    using Div    = Stokes::Block21Type;
    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( lv.dom_f );
    const auto radii  = grid::shell::subdomain_shell_radii< double >( lv.dom_f );
    VectorQ1IsoQ2Q1< double > x( "x", lv.dom_f, lv.dom_c, lv.mask_f, lv.mask_c );
    linalg::assign( x.block_1(), lv.u );
    linalg::assign( x.block_2(), lv.p );
    VectorQ1Scalar< double > no_gca( "no_gca", lv.dom_f, lv.mask_f );
    linalg::assign( no_gca, 0.0 );
    A a( lv.dom_f, coords, radii, lv.bmask_f, lv.k.grid_data(), bcs, false );
    a.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Selective, 0, no_gca.grid_data() );
    Grad g( lv.dom_f, lv.dom_c, coords, radii, lv.bmask_f, bcs );
    Div  dv( lv.dom_f, lv.dom_c, coords, radii, lv.bmask_f, bcs );
    g.set_fast_path( false );
    dv.set_fast_path( false );
    VectorQ1IsoQ2Q1< double > y( "y_slow_double", lv.dom_f, lv.dom_c, lv.mask_f, lv.mask_c );
    a.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace,
                                                  linalg::OperatorCommunicationMode::SkipCommunication );
    g.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Add,
                                                  linalg::OperatorCommunicationMode::CommunicateAdditively );
    dv.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace,
                                                   linalg::OperatorCommunicationMode::CommunicateAdditively );
    linalg::apply( a, x.block_1(), y.block_1() );
    linalg::apply( g, x.block_2(), y.block_1() );
    linalg::apply( dv, x.block_1(), y.block_2() );
    return y;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int failures = 0;
    {
        const int min_level = ( argc > 1 ) ? std::atoi( argv[1] ) : 3;
        const int max_level = ( argc > 2 ) ? std::atoi( argv[2] ) : 5;

        struct BcSet
        {
            const char*                        name;
            grid::shell::BoundaryConditionFlag cmb, surface;
        };
        const std::vector< BcSet > bc_sets = {
            { "D/D", DIRICHLET, DIRICHLET }, { "D/N", DIRICHLET, NEUMANN }, { "D/FS", DIRICHLET, FREESLIP },
            { "N/D", NEUMANN, DIRICHLET },   { "N/N", NEUMANN, NEUMANN },   { "N/FS", NEUMANN, FREESLIP },
            { "FS/D", FREESLIP, DIRICHLET }, { "FS/N", FREESLIP, NEUMANN }, { "FS/FS", FREESLIP, FREESLIP },
        };

        // Pass criteria per storage: relative max-norm error of the fast path against
        // the slow path in the same storage (double, float) or against the double
        // reference (16-bit: bf16 resolves 3 digits, fp16 4; the Add arms carry the
        // prefilled field through one more rounding).
        struct Tol
        {
            double double_, float_, bf16, fp16;
        } const tol{ 1e-11, 5e-5, 1e-1, 2e-2 };

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

            VectorQ1Vec< double >    u( "u_ref", dom_f, mask_f );
            VectorQ1Scalar< double > p( "p_ref", dom_c, mask_c );
            VectorQ1Scalar< double > k( "k_ref", dom_f, mask_f );
            Kokkos::parallel_for( "u", grid::shell::local_domain_md_range_policy_nodes( dom_f ),
                                  VectorFieldInterpolator{ coords_f, radii_f, u.grid_data() } );
            Kokkos::parallel_for( "p", grid::shell::local_domain_md_range_policy_nodes( dom_c ),
                                  ScalarFieldInterpolator{ coords_c, radii_c, p.grid_data() } );
            Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( dom_f ),
                                  CoeffInterpolator{ coords_f, radii_f, k.grid_data() } );
            Kokkos::fence();

            const Level lv{ dom_f, dom_c, mask_f, mask_c, bmask_f, u, p, k };

            for ( const auto& bc : bc_sets )
            {
                BoundaryConditions bcs = { { CMB, bc.cmb }, { SURFACE, bc.surface } };

                // ---- EpsDivDiv --------------------------------------------------------
                for ( const auto& [diagonal, hourglass] :
                      { std::pair{ false, 0.0 }, std::pair{ false, 0.3 }, std::pair{ true, 0.3 } } )
                {
                    // Double slow path = the reference of every arm.
                    using RefOp = fe::wedge::operators::shell::EpsilonDivDivKerngen< double >;
                    VectorQ1Scalar< double > no_gca( "no_gca_ref", dom_f, mask_f );
                    linalg::assign( no_gca, 0.0 );
                    RefOp ref_op( dom_f, coords_f, radii_f, bmask_f, k.grid_data(), bcs, diagonal );
                    ref_op.set_hourglass_coefficient( hourglass );
                    ref_op.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Selective, 0, no_gca.grid_data() );
                    VectorQ1Vec< double > ref( "ref", dom_f, mask_f ), ref_add( "ref_add", dom_f, mask_f );
                    linalg::apply( ref_op, u, ref );
                    linalg::assign( ref_add, u );
                    ref_op.set_operator_apply_and_communication_modes(
                        linalg::OperatorApplyMode::Add, linalg::OperatorCommunicationMode::CommunicateAdditively );
                    linalg::apply( ref_op, u, ref_add );

                    std::cout << "level " << level << "  " << std::setw( 5 ) << bc.name << "  epsdivdiv "
                              << ( diagonal ? "diagonal" : "matvec  " ) << " hourglass " << hourglass << std::endl;

                    auto report = [&]( const char* name, const EpsErrors& e, double t, bool against_double ) {
                        const double crit = against_double ? std::max( e.vs_double, e.vs_double_add )
                                                           : std::max( e.vs_slow, e.vs_slow_add );
                        const bool ok = std::isfinite( crit ) && crit <= t;
                        std::cout << "    " << std::setw( 6 ) << name << ( e.warp ? "  warp " : "  tile " )
                                  << std::scientific << std::setprecision( 2 ) << " vs slow " << e.vs_slow
                                  << " add " << e.vs_slow_add << "   vs double " << e.vs_double << " add "
                                  << e.vs_double_add << ( ok ? "" : "   <-- FAIL" ) << std::endl;
                        if ( !ok )
                            ++failures;
                    };
                    report( "double", eps_arm< double >( lv, bcs, diagonal, hourglass, ref, ref_add ), tol.double_, false );
                    report( "float", eps_arm< float >( lv, bcs, diagonal, hourglass, ref, ref_add ), tol.float_, false );
                    report( "bf16", eps_arm< BHalf >( lv, bcs, diagonal, hourglass, ref, ref_add ), tol.bf16, true );
                    report( "fp16", eps_arm< Half >( lv, bcs, diagonal, hourglass, ref, ref_add ), tol.fp16, true );
                }

                // ---- composed Stokes apply ----------------------------------------------
                {
                    const auto ref = stokes_slow_double( lv, bcs );
                    std::cout << "level " << level << "  " << std::setw( 5 ) << bc.name << "  stokes apply" << std::endl;
                    auto report = [&]( const char* name, const StokesErrors& e, double t, bool against_double ) {
                        const double crit = against_double ? std::max( e.u_vs_double, e.p_vs_double )
                                                           : std::max( e.u_vs_slow, e.p_vs_slow );
                        const bool ok = std::isfinite( crit ) && crit <= t;
                        std::cout << "    " << std::setw( 6 ) << name << std::scientific << std::setprecision( 2 )
                                  << "  u vs slow " << e.u_vs_slow << " p " << e.p_vs_slow << "   vs double u "
                                  << e.u_vs_double << " p " << e.p_vs_double << ( ok ? "" : "   <-- FAIL" ) << std::endl;
                        if ( !ok )
                            ++failures;
                    };
                    report( "double", stokes_arm< double >( lv, bcs, ref ), tol.double_, false );
                    report( "float", stokes_arm< float >( lv, bcs, ref ), tol.float_, false );
                    report( "bf16", stokes_arm< BHalf >( lv, bcs, ref ), tol.bf16, true );
                    report( "fp16", stokes_arm< Half >( lv, bcs, ref ), tol.fp16, true );
                }
            }
        }
    }

    std::cout << ( failures == 0 ? "PASSED" : "FAILED" ) << " (" << failures << " arm(s) above tolerance)" << std::endl;
    return failures == 0 ? 0 : 1;
}
