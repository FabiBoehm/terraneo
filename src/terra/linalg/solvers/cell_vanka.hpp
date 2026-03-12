#pragma once

#include "solver.hpp"

#include "communication/shell/communication.hpp"
#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"

namespace terra::linalg::solvers {

/// @brief Cell-based Vanka iterative solver/smoother for linear systems with coupled DoFs per node.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Generalizes block Jacobi by inverting the full local system per hex cell (8 nodes)
/// rather than just the diagonal block per node.
/// The update rule is (additive Vanka):
/// \f[ x^{(k+1)} = x^{(k)} + \omega \sum_C P_C^T V_C^{-1} P_C (b - Ax^{(k)}) \f]
/// where the sum runs over all hex cells C, \f$ V_C \f$ is the assembled local system
/// of cell C (CellDim x CellDim), \f$ P_C \f$ gathers/scatters between global and
/// cell-local vectors, and \f$ \omega \f$ is the relaxation parameter.
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam BlockSize Size of the per-node blocks (number of DoFs per node).
template < OperatorLike OperatorT, int BlockSize >
class CellVanka
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

    /// @brief Number of nodes per hex cell.
    static constexpr int NumNodesPerCell = 8;

    /// @brief Dimension of the cell-local system (8 nodes * BlockSize DoFs/node).
    static constexpr int CellDim = NumNodesPerCell * BlockSize;

    /// @brief Dense cell matrix type (CellDim x CellDim).
    using CellMatrixType = dense::Mat< ScalarType, CellDim, CellDim >;

    /// @brief Kokkos view storing one inverse cell matrix per hex cell.
    /// Layout: (local_subdomain, x_cell, y_cell, r_cell).
    using InverseCellMatricesType = Kokkos::View< CellMatrixType****, grid::Layout >;

    static_assert( BlockSize > 0, "BlockSize must be positive." );

    /// @brief Construct a CellVanka solver.
    /// @param inverse_cell_matrices Pre-computed inverse cell Vanka matrices.
    /// @param iterations Number of Vanka smoothing iterations to perform.
    /// @param tmp Temporary vector for workspace (residual).
    /// @param correction Temporary vector for workspace (accumulated correction).
    /// @param omega Relaxation parameter (default 1/8 for additive Vanka with ~8 overlapping cells).
    /// @param domain Optional domain pointer for inter-subdomain communication of corrections.
    ///               When provided, corrections are additively communicated across subdomain boundaries
    ///               after each Vanka sweep, ensuring consistency at shared nodes.
    CellVanka(
        const InverseCellMatricesType& inverse_cell_matrices,
        const int                      iterations,
        const SolutionVectorType&      tmp,
        const SolutionVectorType&      correction,
        const ScalarType               omega  = static_cast< ScalarType >( 1.0 / 8.0 ),
        const grid::shell::DistributedDomain* domain = nullptr )
    : inverse_cell_matrices_( inverse_cell_matrices )
    , iterations_( iterations )
    , tmp_( tmp )
    , correction_( correction )
    , omega_( omega )
    , domain_( domain )
    {}

    /// @brief Solve the linear system using additive cell Vanka iteration.
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

            // Zero correction vector.
            linalg::assign( correction_, 0.0 );

            // Apply cell Vanka: for each cell, gather residual, multiply by inverse, scatter correction.
            apply_cell_vanka( tmp_ );

            // Communicate corrections across subdomain boundaries (additive reduction).
            if ( domain_ )
            {
                communicate_correction();
            }

            // x = x + omega * correction
            lincomb( x, { 1.0, omega_ }, { x, correction_ } );
        }
    }

    /// @brief Access the inverse cell matrices data.
    InverseCellMatricesType& get_inverse_cell_matrices() { return inverse_cell_matrices_; }

  private:
    /// @brief Apply additive cell Vanka to accumulate corrections from all cells.
    ///
    /// For each hex cell, gathers the residual at its 8 nodes, multiplies by the
    /// pre-computed inverse cell matrix, and atomically scatters the result into
    /// the correction vector.
    ///
    /// @param residual The residual vector (input, not modified).
    void apply_cell_vanka( const SolutionVectorType& residual )
    {
        auto res_data        = residual.grid_data();
        auto correction_data = correction_.grid_data();
        auto inv_cells       = inverse_cell_matrices_;

        // Hex node (i) -> offset from cell origin (x_cell, y_cell, r_cell).
        // Node i = (i%2, (i/2)%2, i/4).

        const auto num_subdomains = static_cast< int >( inv_cells.extent( 0 ) );
        const auto num_cells_x    = static_cast< int >( inv_cells.extent( 1 ) );
        const auto num_cells_y    = static_cast< int >( inv_cells.extent( 2 ) );
        const auto num_cells_r    = static_cast< int >( inv_cells.extent( 3 ) );

        Kokkos::parallel_for(
            "CellVanka::apply",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { num_subdomains, num_cells_x, num_cells_y, num_cells_r } ),
            KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
                // Gather residual at the 8 cell nodes into a local CellDim-vector.
                dense::Vec< ScalarType, CellDim > local_res;
                for ( int node = 0; node < NumNodesPerCell; ++node )
                {
                    const int gx = xc + ( node % 2 );
                    const int gy = yc + ( ( node / 2 ) % 2 );
                    const int gr = rc + ( node / 4 );

                    for ( int d = 0; d < BlockSize; ++d )
                    {
                        local_res( node * BlockSize + d ) = res_data( local_subdomain, gx, gy, gr, d );
                    }
                }

                // Multiply by inverse cell matrix.
                const auto local_corr = inv_cells( local_subdomain, xc, yc, rc ) * local_res;

                // Scatter correction back (atomic add due to overlapping cells).
                for ( int node = 0; node < NumNodesPerCell; ++node )
                {
                    const int gx = xc + ( node % 2 );
                    const int gy = yc + ( ( node / 2 ) % 2 );
                    const int gr = rc + ( node / 4 );

                    for ( int d = 0; d < BlockSize; ++d )
                    {
                        Kokkos::atomic_add(
                            &correction_data( local_subdomain, gx, gy, gr, d ),
                            local_corr( node * BlockSize + d ) );
                    }
                }
            } );

        Kokkos::fence();
    }

    /// @brief Additively communicate correction vector across subdomain boundaries.
    void communicate_correction()
    {
        auto corr_data = correction_.grid_data();
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, BlockSize > send_buf( *domain_ );
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, BlockSize > recv_buf( *domain_ );
        communication::shell::pack_send_and_recv_local_subdomain_boundaries( *domain_, corr_data, send_buf, recv_buf );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( *domain_, corr_data, recv_buf );
    }

    InverseCellMatricesType inverse_cell_matrices_;  ///< Inverse cell Vanka matrices.
    int                     iterations_;              ///< Number of iterations.
    SolutionVectorType      tmp_;                     ///< Temporary workspace (residual).
    SolutionVectorType      correction_;              ///< Temporary workspace (correction).
    ScalarType              omega_;                   ///< Relaxation parameter.
    const grid::shell::DistributedDomain* domain_;   ///< Optional domain for correction communication.
};

/// @brief Compute the inverse cell Vanka matrices by assembling the full local coupling matrix
///        for each hex cell from element matrices of its neighborhood.
///
/// For each hex cell C at (xc, yc, rc), the CellDim x CellDim Vanka matrix is assembled by
/// iterating over all hex cells in the 3x3x3 neighborhood. For each neighbor cell's wedges,
/// the element matrix is retrieved and the coupling entries between nodes that belong to C
/// are accumulated.
///
/// The resulting matrices are then inverted using LU decomposition.
///
/// @note In the current implementation, only element matrices from the same local subdomain
///       are used. Cells near subdomain boundaries may have slightly incomplete Vanka matrices
///       (missing contributions from elements on adjacent subdomains). This is an acceptable
///       approximation for a smoother.
///
/// @tparam OperatorT Operator type (must provide get_local_matrix() and LocalMatrixDim).
/// @tparam BlockSize Number of DoFs per node.
/// @param A Operator.
/// @param domain Distributed domain for cell iteration.
/// @return Kokkos view of inverse cell Vanka matrices, one per hex cell.
template < typename OperatorT, int BlockSize >
Kokkos::View< dense::Mat< typename OperatorT::ScalarType, 8 * BlockSize, 8 * BlockSize >****, grid::Layout >
compute_cell_vanka_matrices(
    const OperatorT&                      A,
    const grid::shell::DistributedDomain& domain )
{
    using ScalarType     = typename OperatorT::ScalarType;
    using CellMatrixType = dense::Mat< ScalarType, 8 * BlockSize, 8 * BlockSize >;

    constexpr int num_nodes_per_wedge = 6;
    constexpr int local_matrix_dim    = OperatorT::LocalMatrixDim;
    constexpr int CellDim             = 8 * BlockSize;

    static_assert(
        local_matrix_dim == num_nodes_per_wedge * BlockSize,
        "LocalMatrixDim must equal num_nodes_per_wedge * BlockSize." );

    const auto num_subdomains = domain.subdomains().size();
    const auto num_cells_x    = domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto num_cells_y    = domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto num_cells_r    = domain.domain_info().subdomain_num_nodes_radially() - 1;

    // Allocate and zero-initialize storage for the cell Vanka matrices.
    Kokkos::View< CellMatrixType****, grid::Layout > cell_matrices(
        "cell_vanka_matrices", num_subdomains, num_cells_x, num_cells_y, num_cells_r );

    // Wedge local node to hex node offset mapping.
    //
    // Hex cell node numbering:
    //   r = r_cell + 1 (outer)        r = r_cell (inner)
    //   6--7                          2--3
    //   |\ |                          |\ |
    //   | \|                          | \|
    //   4--5                          0--1
    //
    // Hex node -> (dx, dy, dr) offset from (x_cell, y_cell, r_cell):
    //   0:(0,0,0)  1:(1,0,0)  2:(0,1,0)  3:(1,1,0)
    //   4:(0,0,1)  5:(1,0,1)  6:(0,1,1)  7:(1,1,1)
    //
    // Cell-local node index: i = dx + 2*dy + 4*dr  (0..7)

    // Wedge 0: local node n -> (dx, dy, dr)
    constexpr int w0_dx[6] = { 0, 1, 0, 0, 1, 0 };
    constexpr int w0_dy[6] = { 0, 0, 1, 0, 0, 1 };
    constexpr int w0_dr[6] = { 0, 0, 0, 1, 1, 1 };

    // Wedge 1: local node n -> (dx, dy, dr)
    constexpr int w1_dx[6] = { 1, 0, 1, 1, 0, 1 };
    constexpr int w1_dy[6] = { 1, 1, 0, 1, 1, 0 };
    constexpr int w1_dr[6] = { 0, 0, 0, 1, 1, 1 };

    const int nc_x = static_cast< int >( num_cells_x );
    const int nc_y = static_cast< int >( num_cells_y );
    const int nc_r = static_cast< int >( num_cells_r );

    // Iterate over all hex cells (target cells).
    Kokkos::parallel_for(
        "CellVanka::assemble",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
            CellMatrixType V;
            V.fill( ScalarType( 0 ) );

            // Loop over the 3x3x3 neighborhood of hex cells.
            for ( int dx = -1; dx <= 1; ++dx )
            {
                const int xn = xc + dx;
                if ( xn < 0 || xn >= nc_x )
                    continue;

                for ( int dy = -1; dy <= 1; ++dy )
                {
                    const int yn = yc + dy;
                    if ( yn < 0 || yn >= nc_y )
                        continue;

                    for ( int dr = -1; dr <= 1; ++dr )
                    {
                        const int rn = rc + dr;
                        if ( rn < 0 || rn >= nc_r )
                            continue;

                        // Process both wedges of the neighbor cell.
                        for ( int wedge = 0; wedge < 2; ++wedge )
                        {
                            const auto local_mat =
                                A.get_local_matrix( local_subdomain, xn, yn, rn, wedge );

                            // For each pair of local nodes in this wedge, check if both
                            // belong to the target cell C at (xc, yc, rc).
                            for ( int a = 0; a < num_nodes_per_wedge; ++a )
                            {
                                // Map wedge local node a to global grid coordinates.
                                const int gxa = xn + ( wedge == 0 ? w0_dx[a] : w1_dx[a] );
                                const int gya = yn + ( wedge == 0 ? w0_dy[a] : w1_dy[a] );
                                const int gra = rn + ( wedge == 0 ? w0_dr[a] : w1_dr[a] );

                                // Check if node a is in cell C.
                                const int dxa = gxa - xc;
                                const int dya = gya - yc;
                                const int dra = gra - rc;
                                if ( dxa < 0 || dxa > 1 || dya < 0 || dya > 1 || dra < 0 || dra > 1 )
                                    continue;

                                // Cell-local index of node a.
                                const int ia = dxa + 2 * dya + 4 * dra;

                                for ( int b = 0; b < num_nodes_per_wedge; ++b )
                                {
                                    // Map wedge local node b to global grid coordinates.
                                    const int gxb = xn + ( wedge == 0 ? w0_dx[b] : w1_dx[b] );
                                    const int gyb = yn + ( wedge == 0 ? w0_dy[b] : w1_dy[b] );
                                    const int grb = rn + ( wedge == 0 ? w0_dr[b] : w1_dr[b] );

                                    // Check if node b is in cell C.
                                    const int dxb = gxb - xc;
                                    const int dyb = gyb - yc;
                                    const int drb = grb - rc;
                                    if ( dxb < 0 || dxb > 1 || dyb < 0 || dyb > 1 || drb < 0 || drb > 1 )
                                        continue;

                                    // Cell-local index of node b.
                                    const int ib = dxb + 2 * dyb + 4 * drb;

                                    // Accumulate the BlockSize x BlockSize coupling block.
                                    for ( int di = 0; di < BlockSize; ++di )
                                    {
                                        for ( int dj = 0; dj < BlockSize; ++dj )
                                        {
                                            V( ia * BlockSize + di, ib * BlockSize + dj ) +=
                                                local_mat(
                                                    a + di * num_nodes_per_wedge,
                                                    b + dj * num_nodes_per_wedge );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            cell_matrices( local_subdomain, xc, yc, rc ) = V;
        } );
    Kokkos::fence();

    // Invert all cell matrices.
    Kokkos::parallel_for(
        "CellVanka::invert",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
            cell_matrices( local_subdomain, xc, yc, rc ) =
                cell_matrices( local_subdomain, xc, yc, rc ).inv();
        } );
    Kokkos::fence();

    return cell_matrices;
}

} // namespace terra::linalg::solvers
