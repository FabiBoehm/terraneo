// The stored-local-matrix path against the matrix-free path it stands in for.
//
// Same operator, same field, same boundary conditions; the only difference is
// whether the wedge matrices are rebuilt from the geometry on every apply or
// precomputed once and kept in MatT. With MatT = double the two must agree to
// roundoff for every boundary set, with and without the hourglass term, and the
// diagonal -- which the stored path hands back to the matrix-free path -- must be
// bit-identical. The narrower MatT arms report how much a 4- or 2-byte matrix
// costs, which is the number the stored path exists to put next to its speed.
//
// On a Serial build the matrix-free reference is the slow path; on a device it is
// the fast Dirichlet/Neumann or free-slip path. Both are valid references.

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
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
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

struct Setup
{
    const DistributedDomain&                                          domain;
    const Grid4DDataScalar< grid::NodeOwnershipFlag >&                mask_data;
    const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&         boundary_mask_data;
    const Grid3DDataVec< double, 3 >&                                 coords_shell;
    const Grid2DDataScalar< double >&                                 coords_radii;
    const VectorQ1Vec< double >&                                      src;
    const VectorQ1Scalar< double >&                                   k;
    long                                                              num_dofs;
};

/// Applies the stored path with matrices in MatT and returns the error against the
/// matrix-free apply `ref`, relative to its norm: ( inf, l2 ).
template < typename MatT >
std::pair< double, double > stored_arm( const Setup& s, BoundaryConditions bcs, bool diagonal, double hourglass,
                                        const VectorQ1Vec< double >& ref, double ref_inf, double ref_l2 )
{
    using Op = fe::wedge::operators::shell::EpsilonDivDivKerngen< double, 3, double, double, MatT >;
    Op op( s.domain, s.coords_shell, s.coords_radii, s.boundary_mask_data, s.k.grid_data(), bcs, diagonal,
           linalg::OperatorApplyMode::Replace,
           linalg::OperatorCommunicationMode::CommunicateAdditively,
           linalg::OperatorStoredMatrixMode::Off );
    op.set_hourglass_coefficient( hourglass );
    op.set_stored_local_matrices( true );

    VectorQ1Vec< double > dst( "dst_stored", s.domain, s.mask_data );
    VectorQ1Vec< double > err( "err_stored", s.domain, s.mask_data );
    linalg::apply( op, s.src, dst );
    linalg::lincomb( err, { 1.0, -1.0 }, { dst, ref } );

    const double inf = linalg::norm_inf( err ) / ref_inf;
    const double l2  = std::sqrt( linalg::dot( err, err ) / s.num_dofs ) / ref_l2;
    return { inf, l2 };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    using Half  = Kokkos::Experimental::half_t;
    using BHalf = Kokkos::Experimental::bhalf_t;

    // Same matrix, assembled at one quadrature point in both paths; the stored one
    // only reorders the floating-point operations.
    constexpr double double_tol = 1e-11;

    const int min_level = ( argc > 1 ) ? std::atoi( argv[1] ) : 2;
    const int max_level = ( argc > 2 ) ? std::atoi( argv[2] ) : 4;

    // BoundaryConditions is a plain array, so the sets are carried as their two flags.
    struct BcSet
    {
        const char*                        name;
        grid::shell::BoundaryConditionFlag cmb;
        grid::shell::BoundaryConditionFlag surface;
    };
    // All nine CMB/SURFACE combinations of Dirichlet, Neumann and free-slip.
    const std::vector< BcSet > bc_sets = {
        { "D/D", DIRICHLET, DIRICHLET },   { "D/N", DIRICHLET, NEUMANN },   { "D/FS", DIRICHLET, FREESLIP },
        { "N/D", NEUMANN, DIRICHLET },     { "N/N", NEUMANN, NEUMANN },     { "N/FS", NEUMANN, FREESLIP },
        { "FS/D", FREESLIP, DIRICHLET },   { "FS/N", FREESLIP, NEUMANN },   { "FS/FS", FREESLIP, FREESLIP },
    };

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

        Kokkos::parallel_for( "interpolate src", local_domain_md_range_policy_nodes( domain ),
                              VectorFieldInterpolator( coords_shell, coords_radii, src.grid_data() ) );
        Kokkos::parallel_for( "interpolate k", local_domain_md_range_policy_nodes( domain ),
                              ScalarCoeffInterpolator( coords_shell, coords_radii, k.grid_data() ) );
        Kokkos::fence();

        const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
        const Setup s{ domain, mask_data, boundary_mask_data, coords_shell, coords_radii, src, k, num_dofs };

        for ( const auto& bc : bc_sets )
        {
            const char*        bc_name = bc.name;
            BoundaryConditions bcs     = { { CMB, bc.cmb }, { SURFACE, bc.surface } };
            for ( const auto [diagonal, hourglass] :
                  { std::pair{ false, 0.0 }, std::pair{ false, 0.3 }, std::pair{ true, 0.3 } } )
            {
                using RefOp = fe::wedge::operators::shell::EpsilonDivDivKerngen< double >;
                RefOp ref_op( domain, coords_shell, coords_radii, boundary_mask_data, k.grid_data(), bcs, diagonal,
                              linalg::OperatorApplyMode::Replace,
                              linalg::OperatorCommunicationMode::CommunicateAdditively,
                              linalg::OperatorStoredMatrixMode::Off );
                ref_op.set_hourglass_coefficient( hourglass );
                VectorQ1Vec< double > ref( "ref", domain, mask_data );
                linalg::apply( ref_op, src, ref );
                const double ref_inf = linalg::norm_inf( ref );
                const double ref_l2  = std::sqrt( linalg::dot( ref, ref ) / num_dofs );

                std::cout << "level " << level << ", " << bc_name << ", " << ( diagonal ? "diagonal" : "matvec" )
                          << ", hourglass " << hourglass << ", dofs " << 3 * num_dofs << std::endl;
                std::cout << "  " << std::left << std::setw( 18 ) << "matrices" << std::right << std::setw( 14 )
                          << "rel inf" << std::setw( 14 ) << "rel l2" << std::endl;

                auto report = [&]( const char* name, std::pair< double, double > e, bool must_pass ) {
                    const bool ok = !must_pass || ( e.first < double_tol && std::isfinite( e.first ) );
                    failures += ok ? 0 : 1;
                    std::cout << "  " << std::left << std::setw( 18 ) << name << std::right << std::scientific
                              << std::setprecision( 3 ) << std::setw( 14 ) << e.first << std::setw( 14 ) << e.second
                              << ( must_pass ? ( ok ? "   ok" : "   FAIL" ) : "" ) << std::endl;
                };

                report( "double", stored_arm< double >( s, bcs, diagonal, hourglass, ref, ref_inf, ref_l2 ), true );
                if ( !diagonal )
                {
                    report( "float", stored_arm< float >( s, bcs, diagonal, hourglass, ref, ref_inf, ref_l2 ), false );
                    report( "fp16", stored_arm< Half >( s, bcs, diagonal, hourglass, ref, ref_inf, ref_l2 ), false );
                    report( "bf16", stored_arm< BHalf >( s, bcs, diagonal, hourglass, ref, ref_inf, ref_l2 ), false );
                }
            }
        }
    }

    std::cout << ( failures == 0 ? "PASSED" : "FAILED" ) << " (" << failures << " double-matrix arm(s) above "
              << double_tol << ")" << std::endl;
    return failures == 0 ? 0 : 1;
}
