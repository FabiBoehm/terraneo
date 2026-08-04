

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fv/hex/conversion.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/vector_fv.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorFVScalar;
using linalg::VectorQ1Scalar;

using ScalarType = double;

struct SomeFunctionInterpolator
{
    KOKKOS_INLINE_FUNCTION
    ScalarType operator()( const dense::Vec< ScalarType, 3 >& x ) const { return x.norm() + Kokkos::sin( 4 * x( 0 ) ); }
};

void test( int level )
{
    const auto domain = DistributedDomain::create_uniform( level, level, 0.5, 1.0, 0, 0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorFVScalar< ScalarType > u_fv( "u_fv", domain );
    VectorQ1Scalar< ScalarType > u_fe( "u_fe", domain, mask_data );

    std::vector< VectorQ1Scalar< ScalarType > > tmps;

    for ( int i = 0; i < 5; i++ )
    {
        tmps.emplace_back( "tmp", domain, mask_data );
    }

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    fv::hex::l2_project_analytical_to_fv( u_fv, SomeFunctionInterpolator(), coords_shell, coords_radii );

    fv::hex::l2_project_fv_to_fe( u_fe, u_fv, domain, coords_shell, coords_radii, tmps );

    io::XDMFOutput xdmf( "test_finite_volume_basics_out", domain, coords_shell, coords_radii );
    xdmf.add( u_fe.grid_data() );
    xdmf.write( 0 );
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int level = 4;
    test( level );
}
