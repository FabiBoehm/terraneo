#pragma once

#include "solver.hpp"

#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"

namespace terra::linalg::solvers {

/// @brief Block-Jacobi iterative solver/smoother for linear systems with coupled DoFs per node.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Generalizes point-wise Jacobi by replacing the scalar diagonal entries with small dense blocks.
/// The update rule is:
/// \f[ x^{(k+1)} = x^{(k)} + \omega D_{\text{block}}^{-1} (b - Ax^{(k)}) \f]
/// where \f$ D_{\text{block}} \f$ is the block-diagonal of \f$ A \f$ (one \f$ m \times m \f$ block per node)
/// and \f$ \omega \f$ is the relaxation parameter.
///
/// This smoother is particularly effective for coupled systems (e.g., Stokes, elasticity) where
/// multiple DoFs per node are tightly coupled. Unlike point-wise Jacobi which ignores inter-component
/// coupling at each node, block Jacobi captures it via the small dense block inverse.
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam BlockSize Size of the diagonal blocks (number of DoFs per node).
template < OperatorLike OperatorT, int BlockSize >
class BlockJacobi
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Scalar type for computations.
    using ScalarType = SolutionVectorType::ScalarType;

    /// @brief Dense block matrix type (BlockSize x BlockSize).
    using BlockMatrixType = dense::Mat< ScalarType, BlockSize, BlockSize >;

    /// @brief Kokkos view storing one inverse diagonal block per grid node.
    /// Layout: (local_subdomain, x, y, r).
    using InverseBlockDiagonalType = Kokkos::View< BlockMatrixType****, grid::Layout >;

    static_assert( BlockSize > 0, "BlockSize must be positive." );

    /// @brief Construct a BlockJacobi solver.
    /// @param inverse_block_diagonal Pre-computed inverse diagonal blocks, one per grid node.
    /// @param iterations Number of BlockJacobi iterations to perform.
    /// @param tmp Temporary vector for workspace.
    /// @param omega Relaxation parameter (default 2/3).
    BlockJacobi(
        const InverseBlockDiagonalType& inverse_block_diagonal,
        const int                       iterations,
        const SolutionVectorType&       tmp,
        const ScalarType                omega = static_cast< ScalarType >( 2.0 / 3.0 ) )
    : inverse_block_diagonal_( inverse_block_diagonal )
    , iterations_( iterations )
    , tmp_( tmp )
    , omega_( omega )
    {}

    /// @brief Solve the linear system using block Jacobi iteration.
    /// Applies the update rule for the specified number of iterations.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iteration = 0; iteration < iterations_; ++iteration )
        {
            // tmp = A * x
            apply( A, x, tmp_ );

            // tmp = b - A * x  (residual)
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );

            // tmp = D_block^{-1} * tmp  (apply inverse block diagonal)
            apply_inverse_block_diagonal( tmp_ );

            // x = x + omega * tmp
            lincomb( x, { 1.0, omega_ }, { x, tmp_ } );
        }
    }

    /// @brief Access the inverse block diagonal data.
    InverseBlockDiagonalType& get_inverse_block_diagonal() { return inverse_block_diagonal_; }

  private:
    /// @brief Apply the inverse block diagonal to a vector in-place.
    ///
    /// For each grid node, performs a dense matrix-vector multiply:
    /// \f$ v_i \gets D_i^{-1} v_i \f$
    /// where \f$ v_i \f$ is the BlockSize-component vector at node i.
    ///
    /// @param v Vector to which the inverse block diagonal is applied (modified in-place).
    void apply_inverse_block_diagonal( SolutionVectorType& v )
    {
        auto grid_data = v.grid_data();
        auto inv_blocks = inverse_block_diagonal_;

        Kokkos::parallel_for(
            "BlockJacobi::apply_inverse_block_diagonal",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { static_cast< int >( grid_data.extent( 0 ) ),
                  static_cast< int >( grid_data.extent( 1 ) ),
                  static_cast< int >( grid_data.extent( 2 ) ),
                  static_cast< int >( grid_data.extent( 3 ) ) } ),
            KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
                // Gather the BlockSize components at this node into a small dense vector.
                dense::Vec< ScalarType, BlockSize > node_vec;
                for ( int d = 0; d < BlockSize; ++d )
                {
                    node_vec( d ) = grid_data( local_subdomain, i, j, k, d );
                }

                // Multiply by the inverse block diagonal.
                const auto result = inv_blocks( local_subdomain, i, j, k ) * node_vec;

                // Scatter back.
                for ( int d = 0; d < BlockSize; ++d )
                {
                    grid_data( local_subdomain, i, j, k, d ) = result( d );
                }
            } );

        Kokkos::fence();
    }

    InverseBlockDiagonalType inverse_block_diagonal_; ///< Inverse diagonal blocks.
    int                      iterations_;              ///< Number of iterations.
    SolutionVectorType       tmp_;                     ///< Temporary workspace vector.
    ScalarType               omega_;                   ///< Relaxation parameter.
};

/// @brief Compute the inverse block diagonal by extracting self-coupling blocks from local element matrices.
///
/// Iterates over all hex cells and their two wedges. For each wedge, the local element matrix
/// (LocalMatrixDim x LocalMatrixDim, e.g. 18x18 for 3D vector operators with 6 nodes per wedge)
/// is retrieved. The BlockSize x BlockSize self-coupling block for each local node is extracted
/// and atomically accumulated at the corresponding global node. Finally, all blocks are inverted.
///
/// The local matrix layout is:
/// \f[
///   A_{\text{local}}(i + d_i \cdot N, j + d_j \cdot N)
/// \f]
/// where \f$ i, j \in [0, N) \f$ are local node indices (N = num_nodes_per_wedge = 6),
/// and \f$ d_i, d_j \in [0, \text{BlockSize}) \f$ are component indices.
/// The self-coupling block for local node \f$ n \f$ is:
/// \f[
///   D_n(d_i, d_j) = A_{\text{local}}(n + d_i \cdot N, n + d_j \cdot N)
/// \f]
///
/// @tparam OperatorT Operator type (must provide get_local_matrix() and LocalMatrixDim).
/// @tparam BlockSize Number of DoFs per node (must match VecDim of the operator).
/// @param A Operator (uses stored local matrices if available, otherwise assembles on-the-fly).
/// @param domain Distributed domain for cell iteration.
/// @return Kokkos view of inverse diagonal blocks, one per grid node.
template < typename OperatorT, int BlockSize >
Kokkos::View< dense::Mat< typename OperatorT::ScalarType, BlockSize, BlockSize >****, grid::Layout >
compute_inverse_block_diagonal(
    const OperatorT&                          A,
    const grid::shell::DistributedDomain&     domain )
{
    using ScalarType      = typename OperatorT::ScalarType;
    using BlockMatrixType = dense::Mat< ScalarType, BlockSize, BlockSize >;

    constexpr int num_nodes_per_wedge   = 6;
    constexpr int local_matrix_dim      = OperatorT::LocalMatrixDim;

    static_assert(
        local_matrix_dim == num_nodes_per_wedge * BlockSize,
        "LocalMatrixDim must equal num_nodes_per_wedge * BlockSize." );

    const auto num_subdomains = domain.subdomains().size();
    const auto num_nodes_x    = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const auto num_nodes_y    = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const auto num_nodes_r    = domain.domain_info().subdomain_num_nodes_radially();

    // Allocate and zero-initialize storage for the diagonal blocks (one per grid node).
    Kokkos::View< BlockMatrixType****, grid::Layout > blocks(
        "inverse_block_diagonal", num_subdomains, num_nodes_x, num_nodes_y, num_nodes_r );

    // Wedge local node to hex node offset mapping.
    //
    // Hex cell node numbering:
    //   r = r_cell + 1 (outer)        r = r_cell (inner)
    //   6--7                          2--3
    //   |\ |                          |\ |
    //   | \|                          | \|
    //   4--5                          0--1
    //
    // Wedge 0: local nodes (0,1,2,3,4,5) -> hex nodes (0,1,2,4,5,6)
    // Wedge 1: local nodes (0,1,2,3,4,5) -> hex nodes (3,2,1,7,6,5)
    //
    // Hex node -> (dx, dy, dr) offset from (x_cell, y_cell, r_cell):
    //   0:(0,0,0)  1:(1,0,0)  2:(0,1,0)  3:(1,1,0)
    //   4:(0,0,1)  5:(1,0,1)  6:(0,1,1)  7:(1,1,1)

    // Wedge 0: local node n -> (dx, dy, dr)
    constexpr int w0_dx[6] = { 0, 1, 0, 0, 1, 0 };
    constexpr int w0_dy[6] = { 0, 0, 1, 0, 0, 1 };
    constexpr int w0_dr[6] = { 0, 0, 0, 1, 1, 1 };

    // Wedge 1: local node n -> (dx, dy, dr)
    constexpr int w1_dx[6] = { 1, 0, 1, 1, 0, 1 };
    constexpr int w1_dy[6] = { 1, 1, 0, 1, 1, 0 };
    constexpr int w1_dr[6] = { 0, 0, 0, 1, 1, 1 };

    // Iterate over all hex cells.
    Kokkos::parallel_for(
        "BlockJacobi::extract_block_diagonals",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int local_subdomain, int x_cell, int y_cell, int r_cell ) {
            // Process both wedges per hex cell.
            for ( int wedge = 0; wedge < 2; ++wedge )
            {
                const auto local_mat = A.get_local_matrix( local_subdomain, x_cell, y_cell, r_cell, wedge );

                for ( int n = 0; n < num_nodes_per_wedge; ++n )
                {
                    // Map local node n to global grid coordinates.
                    const int gx = x_cell + ( wedge == 0 ? w0_dx[n] : w1_dx[n] );
                    const int gy = y_cell + ( wedge == 0 ? w0_dy[n] : w1_dy[n] );
                    const int gr = r_cell + ( wedge == 0 ? w0_dr[n] : w1_dr[n] );

                    // Extract the BlockSize x BlockSize self-coupling block for node n.
                    // local_mat layout: row = n + dimi * 6, col = n + dimj * 6
                    for ( int di = 0; di < BlockSize; ++di )
                    {
                        for ( int dj = 0; dj < BlockSize; ++dj )
                        {
                            Kokkos::atomic_add(
                                &blocks( local_subdomain, gx, gy, gr )( di, dj ),
                                local_mat( n + di * num_nodes_per_wedge, n + dj * num_nodes_per_wedge ) );
                        }
                    }
                }
            }
        } );
    Kokkos::fence();

    // Invert all blocks.
    Kokkos::parallel_for(
        "BlockJacobi::invert_blocks",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { static_cast< int >( blocks.extent( 0 ) ),
              static_cast< int >( blocks.extent( 1 ) ),
              static_cast< int >( blocks.extent( 2 ) ),
              static_cast< int >( blocks.extent( 3 ) ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            blocks( local_subdomain, i, j, k ) = blocks( local_subdomain, i, j, k ).inv();
        } );
    Kokkos::fence();

    return blocks;
}

} // namespace terra::linalg::solvers
