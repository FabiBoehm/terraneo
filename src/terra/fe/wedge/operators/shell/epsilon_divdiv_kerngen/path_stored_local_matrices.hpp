#pragma once

// Stored-local-matrix path of EpsilonDivDivKerngen.
//
// The matrix-free paths rebuild every wedge's 18x18 matrix from the geometry on
// each apply. This path builds it once, folds the boundary treatment in, and keeps
// it in MatT -- so an apply is a gather, a packed symmetric 18x18 product and a
// scatter, with no geometry touched at all. It exists to measure the other side of
// the matrix-free trade: the apply is bound by the bytes of the matrices, which is
// the one regime where a narrower storage type halves the time rather than the
// instruction count. It is also what the stored matrices cost: 2 x 171 entries per
// hex cell, 5.5 KB in double, which is why the matrix-free paths exist.
//
// Layout. Each wedge matrix is symmetric, so only the upper triangle is kept, 171
// entries, row-major from the diagonal. The view is entry-major, stored_( entry,
// cell ): a warp of consecutive cells then reads consecutive addresses for each
// entry, which is what keeps the kernel at memory bandwidth. It is one flat view
// indexed by hand in size_t: at level 7 the matrices run past 2^32 entries.
//
// Semantics. The matrix is the one the matrix-free path would apply: the on-demand
// assembly in GeoT, the hourglass term when the Dirichlet/Neumann path would add it,
// Dirichlet rows and columns cut to their diagonal, and at free-slip boundaries the
// normal/tangential rotation folded in, R^T mask( R A R^T ) R, so the apply needs no
// rotation of its own. Only the matvec goes through here; the diagonal falls back to
// the matrix-free path, which defines it.

#include "../epsilon_divdiv_kerngen.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
void EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::set_stored_local_matrices(
    const bool on )
{
    use_stored_matrices_ = on;
    update_kernel_path_flag_host_only();
    util::logroot << "[EpsilonDivDiv] stored local matrices " << ( on ? "on" : "off" )
                  << ", kernel path = " << kernel_path_name( kernel_path_ ) << std::endl;
}

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
size_t EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::stored_local_matrices_bytes()
    const
{
    return size_t( num_wedges_per_hex_cell ) * PackedLocalMatrixSize * sizeof( MatT ) * size_t( num_cells() );
}

/// Dirichlet boundary nodes: their rows and columns are cut to the diagonal, so the
/// row returns diag * u and the interior rows see no coupling to them. Same rule as
/// the slow path's boundary mask; `mask` is multiplied in entrywise.
template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
template < typename T >
KOKKOS_INLINE_FUNCTION void
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::dirichlet_boundary_mask(
    const bool                                       at_cmb,
    const bool                                       at_surface,
    dense::Mat< T, LocalMatrixDim, LocalMatrixDim >& mask )
{
    mask.fill( 1.0 );
    for ( int dimi = 0; dimi < 3; ++dimi )
        for ( int dimj = 0; dimj < 3; ++dimj )
            for ( int i = 0; i < num_nodes_per_wedge; i++ )
                for ( int j = 0; j < num_nodes_per_wedge; j++ )
                {
                    if ( ( at_cmb && ( ( dimi == dimj && i != j && ( i < 3 || j < 3 ) ) ||
                                       ( dimi != dimj && ( i < 3 || j < 3 ) ) ) ) ||
                         ( at_surface && ( ( dimi == dimj && i != j && ( i >= 3 || j >= 3 ) ) ||
                                           ( dimi != dimj && ( i >= 3 || j >= 3 ) ) ) ) )
                    {
                        mask( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) = 0.0;
                    }
                }
}

/// Block-diagonal rotation into normal/tangential components at the three boundary
/// nodes of one wedge (identity elsewhere), in NODEWISE ordering. The normal is the
/// unit-sphere coordinate of the node; the rotation is formed in double whatever the
/// stored coordinate precision, as in the slow path.
template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
template < typename T >
KOKKOS_INLINE_FUNCTION void
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::freeslip_rotation(
    const int                                        wedge,
    const int                                        local_subdomain_id,
    const int                                        x_cell,
    const int                                        y_cell,
    const int                                        r_cell,
    const bool                                       at_cmb,
    dense::Mat< T, LocalMatrixDim, LocalMatrixDim >& R ) const
{
    constexpr int layer_hex_offset_x[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
    constexpr int layer_hex_offset_y[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };

    R.fill( 0.0 );
    for ( int i = 0; i < LocalMatrixDim; ++i )
        R( i, i ) = 1.0;

    for ( int boundary_node_idx = 0; boundary_node_idx < 3; boundary_node_idx++ )
    {
        const auto stored_normal = grid::shell::coords(
            local_subdomain_id,
            x_cell + layer_hex_offset_x[wedge][boundary_node_idx],
            y_cell + layer_hex_offset_y[wedge][boundary_node_idx],
            r_cell + ( at_cmb ? 0 : 1 ),
            grid_,
            radii_ );

        dense::Vec< double, 3 > normal;
        normal( 0 ) = static_cast< double >( stored_normal( 0 ) );
        normal( 1 ) = static_cast< double >( stored_normal( 1 ) );
        normal( 2 ) = static_cast< double >( stored_normal( 2 ) );

        const auto R_i = trafo_mat_cartesian_to_normal_tangential( normal );

        const int offset_in_R = at_cmb ? 0 : 9;
        for ( int dimi = 0; dimi < 3; ++dimi )
            for ( int dimj = 0; dimj < 3; ++dimj )
                R( offset_in_R + boundary_node_idx * 3 + dimi, offset_in_R + boundary_node_idx * 3 + dimj ) =
                    static_cast< T >( R_i( dimi, dimj ) );
    }
}

/// The matrix one apply of the matrix-free path would use for this wedge, in GeoT:
/// assembly, hourglass, boundary treatment, all folded in. DIMENSIONWISE ordering
/// (node + 6 * component), symmetric.
template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
KOKKOS_INLINE_FUNCTION auto
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::effective_local_matrix(
    const int  local_subdomain_id,
    const int  x_cell,
    const int  y_cell,
    const int  r_cell,
    const int  wedge,
    const bool at_cmb,
    const bool at_surface ) const -> dense::Mat< GeoT, LocalMatrixDim, LocalMatrixDim >
{
    using LocalMat = dense::Mat< GeoT, LocalMatrixDim, LocalMatrixDim >;

    LocalMat A = this->template assemble_local_matrix< GeoT >( local_subdomain_id, x_cell, y_cell, r_cell, wedge );

    const bool at_boundary = at_cmb || at_surface;
    BoundaryConditionFlag bcf = DIRICHLET;
    if ( at_boundary )
        bcf = get_boundary_condition_flag( bcs_, at_cmb ? CMB : SURFACE );
    const bool dirichlet = at_boundary && ( bcf == DIRICHLET );
    const bool freeslip  = at_boundary && ( bcf == FREESLIP );

    // Hourglass, as the Dirichlet/Neumann path adds it: eps ( G1 G1^T + G2 G2^T ) on
    // each component block, over the non-Dirichlet nodes only. Its eps is
    // stab_c * mean( diag ) / |g|^2 with mean( diag ) = trace / 18 and |g|^2 = 4/3,
    // i.e. stab_c * trace / 24 -- the trace of the *unmasked* matrix, which is
    // kwJ * ( 10 / 3 ) * sum_n |grad N_n|^2, the same sum the matrix-free path forms.
    if ( stored_matrices_hourglass_ )
    {
        constexpr double THIRD = 1.0 / 3.0;
        constexpr double G1[6] = { THIRD, -2.0 * THIRD, THIRD, -THIRD, 2.0 * THIRD, -THIRD };
        constexpr double G2[6] = { THIRD, THIRD, -2.0 * THIRD, -THIRD, -THIRD, 2.0 * THIRD };

        GeoT trace = 0;
        for ( int i = 0; i < LocalMatrixDim; ++i )
            trace += A( i, i );
        const GeoT eps = static_cast< GeoT >( stab_c_ ) * trace / static_cast< GeoT >( 24.0 );

        const int n_begin = ( dirichlet && at_cmb ) ? 3 : 0;
        const int n_end   = ( dirichlet && at_surface ) ? 3 : 6;
        for ( int d = 0; d < 3; ++d )
            for ( int n = n_begin; n < n_end; ++n )
                for ( int m = n_begin; m < n_end; ++m )
                    A( n + d * num_nodes_per_wedge, m + d * num_nodes_per_wedge ) +=
                        eps * static_cast< GeoT >( G1[n] * G1[m] + G2[n] * G2[m] );
    }

    if ( dirichlet )
    {
        LocalMat mask;
        dirichlet_boundary_mask( at_cmb, at_surface, mask );
        A.hadamard_product( mask );
    }
    else if ( freeslip )
    {
        // Into NODEWISE ordering, rotate, cut the normal component's row and column
        // to its diagonal, rotate back, and return to DIMENSIONWISE: the stored
        // matrix then acts on Cartesian components directly.
        LocalMat A_nw;
        for ( int ni = 0; ni < num_nodes_per_wedge; ++ni )
            for ( int di = 0; di < 3; ++di )
                for ( int nj = 0; nj < num_nodes_per_wedge; ++nj )
                    for ( int dj = 0; dj < 3; ++dj )
                        A_nw( ni * 3 + di, nj * 3 + dj ) = A( ni + di * num_nodes_per_wedge, nj + dj * num_nodes_per_wedge );

        LocalMat R;
        freeslip_rotation( wedge, local_subdomain_id, x_cell, y_cell, r_cell, at_cmb, R );
        LocalMat B = R * A_nw * R.transposed();

        const int node_start = at_surface ? 3 : 0;
        const int node_end   = at_surface ? 6 : 3;
        for ( int node_idx = node_start; node_idx < node_end; node_idx++ )
        {
            const int idx = node_idx * 3;
            for ( int k = 0; k < LocalMatrixDim; ++k )
                if ( k != idx )
                    B( idx, k ) = B( k, idx ) = 0.0;
        }
        B = R.transposed() * B * R;

        for ( int ni = 0; ni < num_nodes_per_wedge; ++ni )
            for ( int di = 0; di < 3; ++di )
                for ( int nj = 0; nj < num_nodes_per_wedge; ++nj )
                    for ( int dj = 0; dj < 3; ++dj )
                        A( ni + di * num_nodes_per_wedge, nj + dj * num_nodes_per_wedge ) = B( ni * 3 + di, nj * 3 + dj );
    }

    return A;
}

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
void EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::precompute_stored_local_matrices()
{
    util::Timer timer( "epsilon_divdiv_stored_matrices_precompute" );

    const int ncells = num_cells();
    if ( num_cells_ != ncells )
    {
        num_cells_       = ncells;
        stored_matrices_ = Kokkos::View< MatT* >(
            "epsilon_divdiv_stored_local_matrices",
            size_t( num_wedges_per_hex_cell ) * PackedLocalMatrixSize * size_t( ncells ) );
    }
    util::logroot << "[EpsilonDivDiv] precomputing stored local matrices: " << ncells << " cells, "
                  << stored_local_matrices_bytes() / 1.0e9 << " GB" << std::endl;

    // The matrices stand in for the matrix-free path this build would otherwise run,
    // and only the Dirichlet/Neumann kernel carries the hourglass term: the free-slip
    // kernel does not, and neither does the slow path a Serial build falls back to.
    stored_matrices_hourglass_ =
        ( stab_c_ > 0.0 ) && ( matrix_free_kernel_path() == KernelPath::FastDirichletNeumann );

    Kokkos::parallel_for(
        "epsilon_divdiv_precompute_local_matrices",
        Kokkos::RangePolicy<>( 0, ncells ),
        KOKKOS_CLASS_LAMBDA( const int cell ) {
            int local_subdomain_id, x_cell, y_cell, r_cell;
            this->decode_cell_index( cell, local_subdomain_id, x_cell, y_cell, r_cell );

            const bool at_cmb     = this->has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
            const bool at_surface = this->has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                const auto A = this->effective_local_matrix(
                    local_subdomain_id, x_cell, y_cell, r_cell, wedge, at_cmb, at_surface );
                for ( int i = 0; i < LocalMatrixDim; ++i )
                    for ( int j = i; j < LocalMatrixDim; ++j )
                        this->stored_entry( wedge * PackedLocalMatrixSize + packed_index( i, j ), cell ) =
                            static_cast< MatT >( A( i, j ) );
            }
        } );
    Kokkos::fence();

    stored_matrices_valid_ = true;

    util::logroot << "[EpsilonDivDiv] stored local matrices: " << ncells << " cells x 2 x " << PackedLocalMatrixSize
                  << " entries in " << sizeof( MatT ) << " B = " << stored_local_matrices_bytes() / 1.0e9 << " GB"
                  << ( stored_matrices_hourglass_ ? ", hourglass folded in" : "" ) << std::endl;
}

/// One hex cell: gather the two wedges' values, the packed symmetric products, scatter.
template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
KOKKOS_INLINE_FUNCTION void
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::run_stored_local_matrices(
    const int local_subdomain_id,
    const int x_cell,
    const int y_cell,
    const int r_cell ) const
{
    // The stored matrices' cell numbering (decode_cell_index): r fastest.
    const int cell = ( ( local_subdomain_id * hex_lat_ + x_cell ) * hex_lat_ + y_cell ) * hex_rad_ + r_cell;

    // DIMENSIONWISE local vectors, node + 6 * component, as the assembly orders them.
    ArithT x[num_wedges_per_hex_cell][LocalMatrixDim];
    ArithT y[num_wedges_per_hex_cell][LocalMatrixDim];
    for ( int d = 0; d < 3; ++d )
    {
        dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
        extract_local_wedge_vector_coefficients( src_d, local_subdomain_id, x_cell, y_cell, r_cell, d, src_ );
        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            for ( int i = 0; i < num_nodes_per_wedge; ++i )
            {
                x[wedge][i + d * num_nodes_per_wedge] = static_cast< ArithT >( src_d[wedge]( i ) );
                y[wedge][i + d * num_nodes_per_wedge] = 0;
            }
    }

    for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
    {
        const int base = wedge * PackedLocalMatrixSize;
#pragma unroll
        for ( int i = 0; i < LocalMatrixDim; ++i )
        {
            const ArithT a_ii = static_cast< ArithT >( stored_entry( base + packed_index( i, i ), cell ) );
            y[wedge][i] += a_ii * x[wedge][i];
#pragma unroll
            for ( int j = i + 1; j < LocalMatrixDim; ++j )
            {
                const ArithT a = static_cast< ArithT >( stored_entry( base + packed_index( i, j ), cell ) );
                y[wedge][i] += a * x[wedge][j];
                y[wedge][j] += a * x[wedge][i];
            }
        }
    }

    for ( int d = 0; d < 3; ++d )
    {
        dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            for ( int i = 0; i < num_nodes_per_wedge; ++i )
                dst_d[wedge]( i ) = static_cast< ScalarT >( y[wedge][i + d * num_nodes_per_wedge] );
        atomically_add_local_wedge_vector_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, d, dst_d );
    }
}

} // namespace terra::fe::wedge::operators::shell
