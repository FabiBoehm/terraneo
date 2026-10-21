// Accuracy of the reduced-precision EpsilonDivDivKerngen instantiations.
//
// Every arm applies the operator to the same smooth field and is compared, in
// double, against the all-double operator. The three template precisions are
// varied separately -- storage S, fast-path arithmetic A and geometry G -- so a
// regression in one of them shows up as a jump in exactly one column. The float
// arms are the ones that must pass: with the Jacobian now formed in GeoT, a float
// geometry has to stay at float roundoff, not drift to something a wrong inverse
// would produce. The 16-bit arms are reported for reference.

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
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
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

struct VectorFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    VectorFieldInterpolator(
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
        const auto coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        data_( local_subdomain_id, x, y, r, 0 ) =
            Kokkos::sin( 2.0 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        data_( local_subdomain_id, x, y, r, 1 ) =
            Kokkos::sin( 3.0 * coords( 1 ) ) * Kokkos::sinh( coords( 1 ) );
        data_( local_subdomain_id, x, y, r, 2 ) =
            Kokkos::sin( 4.0 * coords( 2 ) ) * Kokkos::sinh( coords( 1 ) );
    }
};

struct ScalarCoeffInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    ScalarCoeffInterpolator(
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
        const auto coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const auto rr     = radii_( local_subdomain_id, r );

        data_( local_subdomain_id, x, y, r ) =
            1.0 + 0.1 * ( rr - 0.75 ) +
            0.05 * Kokkos::cos( coords( 0 ) ) * Kokkos::cosh( 0.25 * coords( 1 ) );
    }
};

/// Everything one arm needs that does not depend on its precisions.
struct Reference
{
    const DistributedDomain&                                                 domain;
    const Grid4DDataScalar< grid::NodeOwnershipFlag >&        mask_data;
    const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_data;
    const VectorQ1Vec< double >&                                             src;
    const VectorQ1Scalar< double >&                                          k;
    const VectorQ1Vec< double >&                                             dst;
    double                                                                   inf_norm;
    double                                                                   l2_norm;
    long                                                                     num_dofs;
    bool                                                                     diagonal;
    double                                                                   hourglass;
};

/// Applies < S, 3, G, A > and returns the error against the double result, relative
/// to the reference norm: ( inf, l2 ).
template < typename S, typename A, typename G >
std::pair< double, double > run_arm( const Reference& ref )
{
    using Op = fe::wedge::operators::shell::EpsilonDivDivKerngen< S, 3, G, A >;

    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< G >( ref.domain );
    const auto radii  = grid::shell::subdomain_shell_radii< G >( ref.domain );

    VectorQ1Vec< S >    src( "src_arm", ref.domain, ref.mask_data );
    VectorQ1Scalar< S > k( "k_arm", ref.domain, ref.mask_data );
    VectorQ1Vec< S >    dst( "dst_arm", ref.domain, ref.mask_data );
    linalg::convert( ref.src, src );
    linalg::convert( ref.k, k );

    BoundaryConditions bcs = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } };

    Op op( ref.domain, coords, radii, ref.boundary_mask_data, k.grid_data(), bcs, ref.diagonal,
           linalg::OperatorApplyMode::Replace,
           linalg::OperatorCommunicationMode::CommunicateAdditively,
           linalg::OperatorStoredMatrixMode::Off );
    op.set_hourglass_coefficient( ref.hourglass );

    linalg::apply( op, src, dst );

    VectorQ1Vec< double > dst_d( "dst_arm_d", ref.domain, ref.mask_data );
    VectorQ1Vec< double > err( "err_arm", ref.domain, ref.mask_data );
    linalg::convert( dst, dst_d );
    linalg::lincomb( err, { 1.0, -1.0 }, { dst_d, ref.dst } );

    const double inf = linalg::norm_inf( err ) / ref.inf_norm;
    const double l2  = std::sqrt( linalg::dot( err, err ) / ref.num_dofs ) / ref.l2_norm;
    return { inf, l2 };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    using Half  = Kokkos::Experimental::half_t;
    using BHalf = Kokkos::Experimental::bhalf_t;

    // float carries 24 mantissa bits; the operator sums a few dozen products of
    // coordinate differences, so a correct float arm sits within a small multiple
    // of 1e-6. Anything near 1e-3 is a float Jacobian gone wrong, not roundoff.
    constexpr double float_tol = 1e-4;

    const int min_level = ( argc > 1 ) ? std::atoi( argv[1] ) : 3;
    const int max_level = ( argc > 2 ) ? std::atoi( argv[2] ) : 5;

    int failures = 0;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

        auto mask_data          = grid::setup_node_ownership_mask_data( domain );
        auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

        const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
        const auto coords_radii = grid::shell::subdomain_shell_radii< double >( domain );

        VectorQ1Vec< double >    src( "src", domain, mask_data );
        VectorQ1Scalar< double > k( "k", domain, mask_data );
        VectorQ1Vec< double >    dst( "dst", domain, mask_data );

        Kokkos::parallel_for( "interpolate src", local_domain_md_range_policy_nodes( domain ),
                              VectorFieldInterpolator( coords_shell, coords_radii, src.grid_data() ) );
        Kokkos::parallel_for( "interpolate k", local_domain_md_range_policy_nodes( domain ),
                              ScalarCoeffInterpolator( coords_shell, coords_radii, k.grid_data() ) );
        Kokkos::fence();

        const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );

        for ( const auto [diagonal, hourglass] : { std::pair{ false, 0.0 }, std::pair{ false, 0.3 }, std::pair{ true, 0.3 } } )
        {
            using RefOp = fe::wedge::operators::shell::EpsilonDivDivKerngen< double >;
            BoundaryConditions bcs = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } };
            RefOp ref_op( domain, coords_shell, coords_radii, boundary_mask_data, k.grid_data(), bcs, diagonal,
                          linalg::OperatorApplyMode::Replace,
                          linalg::OperatorCommunicationMode::CommunicateAdditively,
                          linalg::OperatorStoredMatrixMode::Off );
            ref_op.set_hourglass_coefficient( hourglass );
            linalg::apply( ref_op, src, dst );

            const Reference ref{ domain,
                                 mask_data,
                                 boundary_mask_data,
                                 src,
                                 k,
                                 dst,
                                 static_cast< double >( linalg::norm_inf( dst ) ),
                                 std::sqrt( linalg::dot( dst, dst ) / num_dofs ),
                                 num_dofs,
                                 diagonal,
                                 hourglass };

            std::cout << "level " << level << ", " << ( diagonal ? "diagonal" : "matvec" ) << ", hourglass "
                      << hourglass << ", dofs " << 3 * num_dofs << std::endl;
            std::cout << "  " << std::left << std::setw( 26 ) << "arm (S / A / G)" << std::right
                      << std::setw( 14 ) << "rel inf" << std::setw( 14 ) << "rel l2" << std::endl;

            auto report = [&]( const char* name, std::pair< double, double > e, bool must_pass ) {
                const bool ok = !must_pass || ( e.first < float_tol && std::isfinite( e.first ) );
                failures += ok ? 0 : 1;
                std::cout << "  " << std::left << std::setw( 26 ) << name << std::right << std::scientific
                          << std::setprecision( 3 ) << std::setw( 14 ) << e.first << std::setw( 14 ) << e.second
                          << ( must_pass ? ( ok ? "   ok" : "   FAIL" ) : "" ) << std::endl;
            };

            report( "float / float / float", run_arm< float, float, float >( ref ), true );
            report( "float / float / double", run_arm< float, float, double >( ref ), true );
            report( "float / double / double", run_arm< float, double, double >( ref ), true );
            report( "fp16 / fp16 / float", run_arm< Half, Half, float >( ref ), false );
            report( "fp16 / fp16 / double", run_arm< Half, Half, double >( ref ), false );
            report( "bf16 / bf16 / float", run_arm< BHalf, BHalf, float >( ref ), false );
        }
    }

    std::cout << ( failures == 0 ? "PASSED" : "FAILED" ) << " (" << failures << " float arm(s) above "
              << float_tol << ")" << std::endl;
    return failures == 0 ? 0 : 1;
}
