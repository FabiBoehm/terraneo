
#pragma once

#include <type_traits>

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/deterministic.hpp"
#include "fe/wedge/integrands.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

// CoordScalarT is the precision of the STORED node coordinates and radii. It
// defaults to ScalarT, so existing instantiations are unchanged. Pass double to
// decouple it: the Jacobian is formed by differencing coordinates, which loses the
// geometry entirely in 16-bit and is already marginal in single. ScalarT governs
// the stored fields src/dst and the coefficient -- where the memory traffic is.
template < typename ScalarT, int VecDim = 3, typename CoordScalarT = ScalarT >
class VectorMass
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< CoordScalarT, 3 > grid_;
    grid::Grid2DDataScalar< CoordScalarT > radii_;

    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, 3 > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, 3 > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

  public:
    VectorMass(
        const grid::shell::DistributedDomain&   domain,
        const grid::Grid3DDataVec< CoordScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< CoordScalarT >& radii,
        const bool                              diagonal,
        linalg::OperatorApplyMode               operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode       operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        det::for_each_colour( det::requested(), [&]( const int c ) {
            colour_ = c;
            Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        } );

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    /// Deterministic colouring (fe/wedge/deterministic.hpp): the launch only scatters
    /// the cells of this parity colour, -1 all of them.
    int colour_ = -1;

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        if ( det::skip( colour_, x_cell, y_cell, r_cell ) )
            return;
        // First all the r-independent stuff.
        // Gather surface points for each wedge.

        // Geometry and the whole local-matrix assembly run in GeomT (= CoordScalarT):
        // the Jacobian is formed by differencing coordinates, so it must not inherit a
        // reduced ScalarT. Only the contraction with the stored fields below uses
        // ScalarT, which is where the memory traffic -- and the saving -- actually is.
        using GeomT = CoordScalarT;

        dense::Vec< GeomT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Compute lateral part of Jacobian.

        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< GeomT, 3 > quad_points[num_quad_points];
        GeomT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        GeomT det_jac_lat[num_wedges_per_hex_cell][num_quad_points] = {};

        jacobian_lat_determinant( det_jac_lat, wedge_phy_surf, quad_points );

        // Only now we introduce radially dependent terms.
        const GeomT r_1 = radii_( local_subdomain_id, r_cell );
        const GeomT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // For now, compute the local element matrix. We'll improve that later.
        dense::Mat< GeomT, 6, 6 > A[num_wedges_per_hex_cell] = {};

        const GeomT grad_r = grad_forward_map_rad( r_1, r_2 );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                const GeomT r = forward_map_rad( r_1, r_2, quad_points[q]( 2 ) );

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const GeomT shape_i = shape( i, quad_points[q] );
                        const GeomT shape_j = shape( j, quad_points[q] );

                        A[wedge]( i, j ) +=
                            quad_weights[q] * ( shape_i * shape_j * r * r * grad_r * det_jac_lat[wedge][q] );
                    }
                }
            }
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        // Narrow the assembled 6x6 blocks to the field precision once per cell (36
        // values), so the contraction below and the extract/accumulate helpers all
        // operate in ScalarT without any mixed-type matrix-vector product.
        dense::Mat< ScalarT, 6, 6 > A_s[num_wedges_per_hex_cell] = {};
        if constexpr ( std::is_same_v< ScalarT, GeomT > )
        {
            for ( int w = 0; w < num_wedges_per_hex_cell; w++ )
                A_s[w] = A[w];
        }
        else
        {
            for ( int w = 0; w < num_wedges_per_hex_cell; w++ )
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        A_s[w]( i, j ) = static_cast< ScalarT >( A[w]( i, j ) );
        }

        for ( int d = 0; d < VecDim; d++ )
        {
            dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
            extract_local_wedge_vector_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, d, src_ );

            dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

            dst[0] = A_s[0] * src[0];
            dst[1] = A_s[1] * src[1];

            atomically_add_local_wedge_vector_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, d, dst );
        }
    }
};

static_assert( linalg::OperatorLike< VectorMass< double, 3 > > );
// Reduced field precision over double-precision geometry.
static_assert( linalg::OperatorLike< VectorMass< float, 3, double > > );

} // namespace terra::fe::wedge::operators::shell