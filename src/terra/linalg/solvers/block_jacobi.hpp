#pragma once

#include "solver.hpp"

#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/grid_types.hpp"

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

/// @brief Compute the inverse block diagonal from an operator by probing.
///
/// For each node and each of the BlockSize DoFs, applies the operator to a unit vector
/// that is 1 at that DoF and 0 elsewhere. The resulting column of the diagonal block
/// is read from the output. The assembled blocks are then inverted.
///
/// This requires the operator to support a "diagonal only" mode where off-diagonal
/// coupling is suppressed (e.g., via the diagonal assembly flag in the operator constructor).
///
/// @tparam OperatorT Operator type.
/// @tparam VectorT Vector type (must provide grid_data()).
/// @tparam BlockSize Number of DoFs per node.
/// @param A_diag Operator assembled in diagonal mode.
/// @param tmp_src Temporary source vector.
/// @param tmp_dst Temporary destination vector.
/// @return Kokkos view of inverse diagonal blocks.
template < OperatorLike OperatorT, VectorLike VectorT, int BlockSize >
Kokkos::View< dense::Mat< typename VectorT::ScalarType, BlockSize, BlockSize >****, grid::Layout >
compute_inverse_block_diagonal( OperatorT& A_diag, VectorT& tmp_src, VectorT& tmp_dst )
{
    using ScalarType      = typename VectorT::ScalarType;
    using BlockMatrixType = dense::Mat< ScalarType, BlockSize, BlockSize >;

    auto src_data = tmp_src.grid_data();
    auto dst_data = tmp_dst.grid_data();

    // Allocate storage for the diagonal blocks.
    Kokkos::View< BlockMatrixType****, grid::Layout > blocks(
        "inverse_block_diagonal",
        src_data.extent( 0 ),
        src_data.extent( 1 ),
        src_data.extent( 2 ),
        src_data.extent( 3 ) );

    // Probe each DoF direction to extract the diagonal block columns.
    for ( int d = 0; d < BlockSize; ++d )
    {
        // Set tmp_src to unit vector in direction d.
        assign( tmp_src, static_cast< ScalarType >( 0 ) );

        // Set the d-th component to 1 everywhere.
        auto grid_data_src = tmp_src.grid_data();
        Kokkos::parallel_for(
            "BlockJacobi::set_unit_vec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { static_cast< int >( grid_data_src.extent( 0 ) ),
                  static_cast< int >( grid_data_src.extent( 1 ) ),
                  static_cast< int >( grid_data_src.extent( 2 ) ),
                  static_cast< int >( grid_data_src.extent( 3 ) ) } ),
            KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
                grid_data_src( local_subdomain, i, j, k, d ) = static_cast< ScalarType >( 1 );
            } );
        Kokkos::fence();

        // Apply the diagonal-mode operator.
        apply( A_diag, tmp_src, tmp_dst );

        // Read off column d of the diagonal block at each node.
        auto grid_data_dst = tmp_dst.grid_data();
        Kokkos::parallel_for(
            "BlockJacobi::extract_column",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { static_cast< int >( grid_data_dst.extent( 0 ) ),
                  static_cast< int >( grid_data_dst.extent( 1 ) ),
                  static_cast< int >( grid_data_dst.extent( 2 ) ),
                  static_cast< int >( grid_data_dst.extent( 3 ) ) } ),
            KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
                for ( int row = 0; row < BlockSize; ++row )
                {
                    blocks( local_subdomain, i, j, k )( row, d ) =
                        grid_data_dst( local_subdomain, i, j, k, row );
                }
            } );
        Kokkos::fence();
    }

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
