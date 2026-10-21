#pragma once

// Slow path of EpsilonDivDivKerngen: per-cell local matrices, assembled on the fly or
//
// taken from the GCA storage, applied with a dense 18x18 product. Also the on-demand
// assembly itself, which the stored-local-matrix path reuses for its precompute.

#include "../epsilon_divdiv_kerngen.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
KOKKOS_INLINE_FUNCTION void
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::assemble_trial_test_vecs(
    const int                            wedge,
    const dense::Vec< GeomT, VecDim >&   quad_point,
    const GeomT                          quad_weight,
    const GeomT                          r_1,
    const GeomT                          r_2,
    dense::Vec< GeomT, 3 > ( *wedge_phy_surf )[3],
    const dense::Vec< ScalarT, 6 >*   k_local_hex,
    const int                         dimi,
    const int                         dimj,
    dense::Mat< GeomT, VecDim, VecDim >* sym_grad_i,
    dense::Mat< GeomT, VecDim, VecDim >* sym_grad_j,
    GeomT&                               jdet_keval_quadweight ) const
{
    dense::Mat< GeomT, VecDim, VecDim >       J       = jac( wedge_phy_surf[wedge], r_1, r_2, quad_point );
    const auto                                det     = J.det();
    const auto                                abs_det = Kokkos::abs( det );
    const dense::Mat< GeomT, VecDim, VecDim > J_inv_transposed = J.inv_transposed( det );

    // The coefficient is a stored field, so it is read in ScalarT and promoted.
    GeomT k_eval = 0.0;
    for ( int k = 0; k < num_nodes_per_wedge; k++ )
    {
        k_eval += shape( k, quad_point ) * k_local_hex[wedge]( k );
    }

    for ( int k = 0; k < num_nodes_per_wedge; k++ )
    {
        sym_grad_i[k] = symmetric_grad( J_inv_transposed, quad_point, k, dimi );
        sym_grad_j[k] = symmetric_grad( J_inv_transposed, quad_point, k, dimj );
    }

    jdet_keval_quadweight = quad_weight * k_eval * abs_det;
}

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
template < typename OutT >
KOKKOS_INLINE_FUNCTION auto
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::assemble_local_matrix(
    const int local_subdomain_id,
    const int x_cell,
    const int y_cell,
    const int r_cell,
    const int wedge ) const -> dense::Mat< OutT, LocalMatrixDim, LocalMatrixDim >
{
    dense::Vec< GeomT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
    wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

    const GeomT r_1 = radii_( local_subdomain_id, r_cell );
    const GeomT r_2 = radii_( local_subdomain_id, r_cell + 1 );

    dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
    extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

    dense::Mat< OutT, LocalMatrixDim, LocalMatrixDim > A = {};

    for ( int dimi = 0; dimi < 3; ++dimi )
    {
        for ( int dimj = 0; dimj < 3; ++dimj )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                dense::Mat< GeomT, VecDim, VecDim > sym_grad_i[num_nodes_per_wedge];
                dense::Mat< GeomT, VecDim, VecDim > sym_grad_j[num_nodes_per_wedge];
                // Assembly runs in GeomT; the accumulation into A narrows to OutT.
                GeomT                               jdet_keval_quadweight = 0;

                assemble_trial_test_vecs(
                    wedge,
                    quad_points[q],
                    quad_weights[q],
                    r_1,
                    r_2,
                    wedge_phy_surf,
                    k_local_hex,
                    dimi,
                    dimj,
                    sym_grad_i,
                    sym_grad_j,
                    jdet_keval_quadweight );

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        A( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) +=
                            jdet_keval_quadweight *
                            ( 2 * sym_grad_j[j].double_contract( sym_grad_i[i] ) -
                              2.0 / 3.0 * sym_grad_j[j]( dimj, dimj ) * sym_grad_i[i]( dimi, dimi ) );
                    }
                }
            }
        }
    }

    return A;
}

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
KOKKOS_INLINE_FUNCTION void
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::operator_slow_path(
    const Team& team,
    const int   local_subdomain_id,
    const int   x0,
    const int   y0,
    const int   r0,
    const int   tx,
    const int   ty,
    const int   tr,
    const int   x_cell,
    const int   y_cell,
    const int   r_cell,
    const bool  at_cmb,
    const bool  at_surface ) const
{
    (void) team;
    (void) x0;
    (void) y0;
    (void) r0;
    (void) tx;
    (void) ty;
    (void) tr;

    if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ || r_cell >= hex_rad_ )
        return;

    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A[num_wedges_per_hex_cell] = { 0 };

    if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Full )
    {
        A[0] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
        A[1] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
    }
    else if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Selective )
    {
        if ( local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 ) &&
             local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 ) )
        {
            A[0] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            A[1] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
        }
        else
        {
            A[0] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            A[1] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
        }
    }
    else
    {
        A[0] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
        A[1] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
    }

    dense::Vec< ScalarT, 18 > src[num_wedges_per_hex_cell];
    for ( int dimj = 0; dimj < 3; dimj++ )
    {
        dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
        extract_local_wedge_vector_coefficients( src_d, local_subdomain_id, x_cell, y_cell, r_cell, dimj, src_ );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int i = 0; i < num_nodes_per_wedge; i++ )
            {
                src[wedge]( dimj * num_nodes_per_wedge + i ) = src_d[wedge]( i );
            }
        }
    }

    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > boundary_mask;
    boundary_mask.fill( 1.0 );

    bool                                                  freeslip_reorder = false;
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > R[num_wedges_per_hex_cell];

    if ( at_cmb || at_surface )
    {
        ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
        BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );

        if ( bcf == DIRICHLET )
        {
            dirichlet_boundary_mask( at_cmb, at_surface, boundary_mask );
        }
        else if ( bcf == FREESLIP )
        {
            freeslip_reorder                                                                     = true;
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A_tmp[num_wedges_per_hex_cell] = { 0 };

            for ( int wedge = 0; wedge < 2; ++wedge )
            {
                for ( int dimi = 0; dimi < 3; ++dimi )
                {
                    for ( int node_idxi = 0; node_idxi < num_nodes_per_wedge; node_idxi++ )
                    {
                        for ( int dimj = 0; dimj < 3; ++dimj )
                        {
                            for ( int node_idxj = 0; node_idxj < num_nodes_per_wedge; node_idxj++ )
                            {
                                A_tmp[wedge]( node_idxi * 3 + dimi, node_idxj * 3 + dimj ) = A[wedge](
                                    node_idxi + dimi * num_nodes_per_wedge,
                                    node_idxj + dimj * num_nodes_per_wedge );
                            }
                        }
                    }
                }
                reorder_local_dofs( DoFOrdering::DIMENSIONWISE, DoFOrdering::NODEWISE, src[wedge] );
            }

            for ( int wedge = 0; wedge < 2; ++wedge )
            {
                // coords() comes back in GeomT. The rotation itself stays in double: a
                // narrower coordinate *store* should not quietly narrow the boundary
                // algebra built on top of it.
                freeslip_rotation( wedge, local_subdomain_id, x_cell, y_cell, r_cell, at_cmb, R[wedge] );

                A[wedge] = R[wedge] * A_tmp[wedge] * R[wedge].transposed();

                auto src_tmp = R[wedge] * src[wedge];
                for ( int i = 0; i < 18; ++i )
                {
                    src[wedge]( i ) = src_tmp( i );
                }

                const int node_start = at_surface ? 3 : 0;
                const int node_end   = at_surface ? 6 : 3;

                for ( int node_idx = node_start; node_idx < node_end; node_idx++ )
                {
                    const int idx = node_idx * 3;
                    for ( int k = 0; k < 18; ++k )
                    {
                        if ( k != idx )
                        {
                            boundary_mask( idx, k ) = 0.0;
                            boundary_mask( k, idx ) = 0.0;
                        }
                    }
                }
            }
        }
    }

    for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
    {
        A[wedge].hadamard_product( boundary_mask );
    }

    if ( diagonal_ )
    {
        A[0] = A[0].diagonal();
        A[1] = A[1].diagonal();
    }

    dense::Vec< ScalarT, LocalMatrixDim > dst[num_wedges_per_hex_cell];
    dst[0] = A[0] * src[0];
    dst[1] = A[1] * src[1];

    if ( freeslip_reorder )
    {
        dense::Vec< ScalarT, LocalMatrixDim > dst_tmp[num_wedges_per_hex_cell];
        dst_tmp[0] = R[0].transposed() * dst[0];
        dst_tmp[1] = R[1].transposed() * dst[1];

        for ( int i = 0; i < 18; ++i )
        {
            dst[0]( i ) = dst_tmp[0]( i );
            dst[1]( i ) = dst_tmp[1]( i );
        }

        reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst[0] );
        reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst[1] );
    }

    for ( int dimi = 0; dimi < 3; dimi++ )
    {
        dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
        dst_d[0] = dst[0].template slice< 6 >( dimi * num_nodes_per_wedge );
        dst_d[1] = dst[1].template slice< 6 >( dimi * num_nodes_per_wedge );

        atomically_add_local_wedge_vector_coefficients(
            dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst_d );
    }
}

} // namespace terra::fe::wedge::operators::shell
