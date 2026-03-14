#pragma once

#include "solver.hpp"

#include "communication/shell/communication.hpp"
#include "terra/dense/mat.hpp"
#include "terra/dense/packed_sym_mat.hpp"
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

    /// @brief Storage scalar type for packed inverse matrices.
    /// Using float halves memory bandwidth for the dominant matrix load
    /// (1200 vs 2400 bytes per cell for N=24). The matvec is computed in
    /// double via PackedSymMat's mixed-precision operator*.
    using StorageScalarType = float;

    /// @brief Packed symmetric matrix type for the inverse cell matrices.
    using PackedCellMatrixType = dense::PackedSymMat< StorageScalarType, CellDim >;

    /// @brief Kokkos view storing one packed inverse cell matrix per hex cell.
    /// Layout: (local_subdomain, x_cell, y_cell, r_cell).
    using InverseCellMatricesType = Kokkos::View< PackedCellMatrixType****, grid::Layout >;

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

    /// @brief Solve the linear system using cell Vanka iteration.
    ///
    /// Uses a single kernel over all cells with fused residual computation and atomic
    /// scatter, replacing the previous 8-color approach (8 separate kernel launches).
    /// The residual b - Ax is computed inline during the gather, eliminating a separate
    /// full-grid residual kernel.
    ///
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        if ( domain_ )
        {
            // Communication path: accumulate corrections, communicate, then update x.
            for ( int iteration = 0; iteration < iterations_; ++iteration )
            {
                apply( A, x, tmp_ );
                zero_correction();
                apply_cell_vanka_fused_scaled( b );
                communicate_correction();
                update_x_no_scale( x );
            }
        }
        else
        {
            // No-communication path: update x directly with atomic scatter.
            for ( int iteration = 0; iteration < iterations_; ++iteration )
            {
                apply( A, x, tmp_ );
                apply_cell_vanka_fused_direct( x, b );
            }
        }
    }

    /// @brief Access the inverse cell matrices data.
    InverseCellMatricesType& get_inverse_cell_matrices() { return inverse_cell_matrices_; }

  private:
    /// @brief Zero the correction vector.
    void zero_correction()
    {
        auto corr_data = correction_.grid_data();

        Kokkos::parallel_for(
            "CellVanka::zero_correction",
            Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >(
                { 0, 0, 0, 0, 0 },
                { static_cast< long long >( corr_data.extent( 0 ) ),
                  static_cast< long long >( corr_data.extent( 1 ) ),
                  static_cast< long long >( corr_data.extent( 2 ) ),
                  static_cast< long long >( corr_data.extent( 3 ) ),
                  static_cast< long long >( corr_data.extent( 4 ) ) } ),
            KOKKOS_LAMBDA( int s, int i, int j, int k, int d ) {
                corr_data( s, i, j, k, d ) = 0;
            } );
        Kokkos::fence();
    }

    /// @brief Fused residual + Vanka apply with direct x update (no-communication path).
    ///
    /// Single kernel over all cells. Computes residual (b - Ax) inline during gather,
    /// eliminating a separate full-grid residual kernel. Uses atomic scatter to x,
    /// replacing the 8-color approach (8 kernel launches) with a single launch.
    void apply_cell_vanka_fused_direct( SolutionVectorType& x, const RHSVectorType& b )
    {
        auto ax_data   = tmp_.grid_data();
        auto b_data    = b.grid_data();
        auto x_data    = x.grid_data();
        auto inv_cells = inverse_cell_matrices_;
        auto omega     = omega_;

        const auto num_subdomains = static_cast< int >( inv_cells.extent( 0 ) );
        const auto num_cells_x   = static_cast< int >( inv_cells.extent( 1 ) );
        const auto num_cells_y   = static_cast< int >( inv_cells.extent( 2 ) );
        const auto num_cells_r   = static_cast< int >( inv_cells.extent( 3 ) );

        Kokkos::parallel_for(
            "CellVanka::apply_fused_direct",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { num_subdomains, num_cells_x, num_cells_y, num_cells_r } ),
            KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
                // Gather residual (b - Ax) at the 8 cell nodes — fused, no separate residual pass.
                dense::Vec< ScalarType, CellDim > local_res;
                for ( int node = 0; node < NumNodesPerCell; ++node )
                {
                    const int gx = xc + ( node % 2 );
                    const int gy = yc + ( ( node / 2 ) % 2 );
                    const int gr = rc + ( node / 4 );

                    for ( int d = 0; d < BlockSize; ++d )
                    {
                        local_res( node * BlockSize + d ) =
                            b_data( local_subdomain, gx, gy, gr, d ) -
                            ax_data( local_subdomain, gx, gy, gr, d );
                    }
                }

                // Multiply by inverse cell matrix.
                const auto local_corr = inv_cells( local_subdomain, xc, yc, rc ) * local_res;

                // Scatter with atomic add (replaces 8-color approach).
                for ( int node = 0; node < NumNodesPerCell; ++node )
                {
                    const int gx = xc + ( node % 2 );
                    const int gy = yc + ( ( node / 2 ) % 2 );
                    const int gr = rc + ( node / 4 );

                    for ( int d = 0; d < BlockSize; ++d )
                    {
                        Kokkos::atomic_add(
                            &x_data( local_subdomain, gx, gy, gr, d ),
                            omega * local_corr( node * BlockSize + d ) );
                    }
                }
            } );

        Kokkos::fence();
    }

    /// @brief Fused residual + Vanka apply into correction vector (communication path).
    ///
    /// Single kernel over all cells. Computes residual (b - Ax) inline during gather.
    /// Uses atomic scatter to correction vector. Correction must be zeroed before calling.
    void apply_cell_vanka_fused_scaled( const RHSVectorType& b )
    {
        auto ax_data         = tmp_.grid_data();
        auto b_data          = b.grid_data();
        auto correction_data = correction_.grid_data();
        auto inv_cells       = inverse_cell_matrices_;
        auto omega           = omega_;

        const auto num_subdomains = static_cast< int >( inv_cells.extent( 0 ) );
        const auto num_cells_x   = static_cast< int >( inv_cells.extent( 1 ) );
        const auto num_cells_y   = static_cast< int >( inv_cells.extent( 2 ) );
        const auto num_cells_r   = static_cast< int >( inv_cells.extent( 3 ) );

        Kokkos::parallel_for(
            "CellVanka::apply_fused_scaled",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { num_subdomains, num_cells_x, num_cells_y, num_cells_r } ),
            KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
                // Gather residual (b - Ax) at the 8 cell nodes — fused, no separate residual pass.
                dense::Vec< ScalarType, CellDim > local_res;
                for ( int node = 0; node < NumNodesPerCell; ++node )
                {
                    const int gx = xc + ( node % 2 );
                    const int gy = yc + ( ( node / 2 ) % 2 );
                    const int gr = rc + ( node / 4 );

                    for ( int d = 0; d < BlockSize; ++d )
                    {
                        local_res( node * BlockSize + d ) =
                            b_data( local_subdomain, gx, gy, gr, d ) -
                            ax_data( local_subdomain, gx, gy, gr, d );
                    }
                }

                // Multiply by inverse cell matrix.
                const auto local_corr = inv_cells( local_subdomain, xc, yc, rc ) * local_res;

                // Scatter with atomic add (replaces 8-color approach).
                for ( int node = 0; node < NumNodesPerCell; ++node )
                {
                    const int gx = xc + ( node % 2 );
                    const int gy = yc + ( ( node / 2 ) % 2 );
                    const int gr = rc + ( node / 4 );

                    for ( int d = 0; d < BlockSize; ++d )
                    {
                        Kokkos::atomic_add(
                            &correction_data( local_subdomain, gx, gy, gr, d ),
                            omega * local_corr( node * BlockSize + d ) );
                    }
                }
            } );

        Kokkos::fence();
    }

    /// @brief Update x += correction (omega already baked into correction).
    void update_x_no_scale( SolutionVectorType& x )
    {
        auto x_data    = x.grid_data();
        auto corr_data = correction_.grid_data();

        Kokkos::parallel_for(
            "CellVanka::update_x",
            Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >(
                { 0, 0, 0, 0, 0 },
                { static_cast< long long >( x_data.extent( 0 ) ),
                  static_cast< long long >( x_data.extent( 1 ) ),
                  static_cast< long long >( x_data.extent( 2 ) ),
                  static_cast< long long >( x_data.extent( 3 ) ),
                  static_cast< long long >( x_data.extent( 4 ) ) } ),
            KOKKOS_LAMBDA( int s, int i, int j, int k, int d ) {
                x_data( s, i, j, k, d ) += corr_data( s, i, j, k, d );
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

// ---------------------------------------------------------------------------
// Ghost element helpers for cross-subdomain Vanka matrix assembly.
// ---------------------------------------------------------------------------

namespace detail {

/// @brief Describes the ghost element mapping for one face boundary.
///
/// Captures all parameters needed by a Kokkos kernel to process ghost elements
/// from a neighboring subdomain across a shared face boundary.
struct GhostFaceInfo
{
    int sender_local_subdomain_id;    ///< Neighbor's local subdomain index on this rank.
    int receiver_local_subdomain_id;  ///< Local subdomain index.

    // Sender's boundary cell layer geometry.
    // The sender's face has one fixed axis (the boundary normal) and two varying axes.
    // Axes are encoded as: 0 = x, 1 = y, 2 = r.
    int sender_fixed_axis;   ///< Which axis is normal to the sender's face.
    int sender_var0_axis;    ///< Sender's first varying axis.
    int sender_var1_axis;    ///< Sender's second varying axis.
    int sender_fixed_cell;   ///< Cell index along fixed axis for boundary cells.
    int sender_var0_cells;   ///< Number of cells along sender's varying axis 0.
    int sender_var1_cells;   ///< Number of cells along sender's varying axis 1.

    // Node mapping: sender boundary node value and receiver boundary node value.
    int sender_boundary_node;   ///< Node index at sender's boundary (0 or num_nodes-1).
    int receiver_boundary_node; ///< Node index at receiver's boundary (0 or num_nodes-1).

    // Receiver's face axes.
    int receiver_fixed_axis;  ///< Which axis is normal to the receiver's face.
    int receiver_var0_axis;   ///< Receiver's first varying axis.
    int receiver_var1_axis;   ///< Receiver's second varying axis.

    // Whether sender's boundary is at P1 (end). Needed to determine which wedge
    // node offsets land on the boundary: offset==1 for P1, offset==0 for P0.
    bool sender_at_end;

    // Direction flags: whether the varying axes need reversal during mapping.
    bool reverse_var0;
    bool reverse_var1;

    // Grid dimensions.
    int num_nodes_lat;
    int num_nodes_r;
};

/// @brief Returns (fixed_axis, var0_axis, var1_axis) for a boundary face type.
///
/// The face type encodes which axis is fixed (boundary normal) and which two vary:
///   F_0YR, F_1YR: fixed = x (0), varying = (y, r) = (1, 2)
///   F_X0R, F_X1R: fixed = y (1), varying = (x, r) = (0, 2)
///   F_XY0, F_XY1: fixed = r (2), varying = (x, y) = (0, 1)
inline void face_axes( grid::BoundaryFace face, int& fixed, int& var0, int& var1 )
{
    using BF = grid::BoundaryFace;
    switch ( face )
    {
    case BF::F_0YR:
    case BF::F_1YR:
        fixed = 0;
        var0  = 1;
        var1  = 2;
        break;
    case BF::F_X0R:
    case BF::F_X1R:
        fixed = 1;
        var0  = 0;
        var1  = 2;
        break;
    case BF::F_XY0:
    case BF::F_XY1:
        fixed = 2;
        var0  = 0;
        var1  = 1;
        break;
    }
}

/// @brief Returns true if the face boundary is at the P1 (end) side.
inline bool face_is_at_end( grid::BoundaryFace face )
{
    using BF = grid::BoundaryFace;
    return face == BF::F_1YR || face == BF::F_X1R || face == BF::F_XY1;
}

/// @brief Build a GhostFaceInfo for a face boundary between a local subdomain and its neighbor.
///
/// @param receiver_local_id  Local subdomain index of the receiver.
/// @param sender_local_id    Local subdomain index of the sender (neighbor) — must be on same rank.
/// @param receiver_face      Boundary face on the receiver.
/// @param sender_face        Boundary face on the sender (neighbor).
/// @param dir_0              Unpacking direction for first varying axis.
/// @param dir_1              Unpacking direction for second varying axis.
/// @param num_nodes_lat      Number of lateral grid nodes per subdomain side.
/// @param num_nodes_r        Number of radial grid nodes per subdomain.
inline GhostFaceInfo build_ghost_face_info(
    int                    receiver_local_id,
    int                    sender_local_id,
    grid::BoundaryFace     receiver_face,
    grid::BoundaryFace     sender_face,
    grid::BoundaryDirection dir_0,
    grid::BoundaryDirection dir_1,
    int                    num_nodes_lat,
    int                    num_nodes_r )
{
    GhostFaceInfo info{};

    info.sender_local_subdomain_id   = sender_local_id;
    info.receiver_local_subdomain_id = receiver_local_id;

    // Sender axes.
    face_axes( sender_face, info.sender_fixed_axis, info.sender_var0_axis, info.sender_var1_axis );

    int num_cells[3] = { num_nodes_lat - 1, num_nodes_lat - 1, num_nodes_r - 1 };
    int num_nodes[3] = { num_nodes_lat, num_nodes_lat, num_nodes_r };

    info.sender_at_end       = face_is_at_end( sender_face );
    info.sender_fixed_cell   = info.sender_at_end ? ( num_cells[info.sender_fixed_axis] - 1 ) : 0;
    info.sender_var0_cells   = num_cells[info.sender_var0_axis];
    info.sender_var1_cells   = num_cells[info.sender_var1_axis];
    info.sender_boundary_node = info.sender_at_end ? ( num_nodes[info.sender_fixed_axis] - 1 ) : 0;

    // Receiver axes.
    face_axes( receiver_face, info.receiver_fixed_axis, info.receiver_var0_axis, info.receiver_var1_axis );
    info.receiver_boundary_node = face_is_at_end( receiver_face ) ?
                                      ( num_nodes[info.receiver_fixed_axis] - 1 ) : 0;

    // Direction flags (from receiver's unpacking perspective).
    info.reverse_var0 = ( dir_0 == grid::BoundaryDirection::BACKWARD );
    info.reverse_var1 = ( dir_1 == grid::BoundaryDirection::BACKWARD );

    info.num_nodes_lat = num_nodes_lat;
    info.num_nodes_r   = num_nodes_r;

    return info;
}

} // namespace detail

/// @brief Accumulate ghost element contributions from face-adjacent neighbor subdomains
///        into the cell Vanka matrices.
///
/// For each face boundary where the neighbor is on the same MPI rank, this function
/// reads the neighbor's element matrices (via the operator) and scatters the coupling
/// entries between shared boundary nodes into the local Vanka cell matrices.
///
/// The element matrix values are coordinate-system independent (assembled from physical
/// coordinates), but the row/column indices encode a wedge-local node ordering that depends
/// on the subdomain's (x,y,r) axes. When axes differ between sender and receiver (cross-
/// diamond boundaries), the node positions are permuted using the BoundaryDirection flags
/// from the subdomain neighborhood.
///
/// @tparam OperatorT Operator type.
/// @tparam BlockSize Number of DoFs per node.
/// @param A                The operator (for reading neighbor element matrices).
/// @param domain           Distributed domain.
/// @param cell_matrices    [in/out] Cell Vanka matrices to accumulate into.
template < typename OperatorT, int BlockSize >
void accumulate_ghost_element_contributions(
    const OperatorT&                      A,
    const grid::shell::DistributedDomain& domain,
    Kokkos::View< dense::Mat< typename OperatorT::ScalarType, 8 * BlockSize, 8 * BlockSize >****, grid::Layout >&
        cell_matrices )
{
    using ScalarType = typename OperatorT::ScalarType;

    constexpr int num_nodes_per_wedge = 6;
    constexpr int CellDim             = 8 * BlockSize;

    const int num_nodes_lat = static_cast< int >( domain.domain_info().subdomain_num_nodes_per_side_laterally() );
    const int num_nodes_r   = static_cast< int >( domain.domain_info().subdomain_num_nodes_radially() );
    const int nc_x          = num_nodes_lat - 1;
    const int nc_y          = num_nodes_lat - 1;
    const int nc_r          = num_nodes_r - 1;

    // Wedge node offset tables (same as in the main assembly).
    // Wedge 0: local node n -> (dx, dy, dr) offset from cell origin.
    constexpr int w_dx[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
    constexpr int w_dy[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
    constexpr int w_dr[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

    // Process each local subdomain's face boundaries.
    for ( const auto& [subdomain_info, idx_and_neighborhood] : domain.subdomains() )
    {
        const auto& [receiver_local_id, neighborhood] = idx_and_neighborhood;

        for ( const auto& [receiver_face, neighbor_tuple] : neighborhood.neighborhood_face() )
        {
            const auto& [neighbor_info, sender_face, directions, neighbor_rank] = neighbor_tuple;
            const auto [dir_0, dir_1]                                           = directions;

            // Only handle local neighbors (same MPI rank) for now.
            // TODO: Add MPI communication for remote neighbors.
            if ( neighbor_rank != mpi::rank() )
                continue;

            if ( !domain.subdomains().contains( neighbor_info ) )
                continue;

            const int sender_local_id = std::get< 0 >( domain.subdomains().at( neighbor_info ) );

            const auto info = detail::build_ghost_face_info(
                receiver_local_id, sender_local_id,
                receiver_face, sender_face,
                dir_0, dir_1,
                num_nodes_lat, num_nodes_r );

            // Launch a Kokkos kernel over the sender's boundary cell layer.
            // Each work item processes one ghost cell (both wedges) and scatters
            // coupling entries between boundary node pairs into receiver's cell matrices.
            auto cm = cell_matrices; // capture by value for lambda
            Kokkos::parallel_for(
                "CellVanka::ghost_face",
                Kokkos::MDRangePolicy< Kokkos::Rank< 2 > >(
                    { 0, 0 }, { info.sender_var0_cells, info.sender_var1_cells } ),
                KOKKOS_LAMBDA( int ci, int cj ) {
                    // Reconstruct sender's cell coordinates.
                    int sender_cell[3];
                    sender_cell[info.sender_fixed_axis] = info.sender_fixed_cell;
                    sender_cell[info.sender_var0_axis]  = ci;
                    sender_cell[info.sender_var1_axis]  = cj;

                    const int num_nodes[3] = { info.num_nodes_lat, info.num_nodes_lat, info.num_nodes_r };

                    for ( int wedge = 0; wedge < 2; ++wedge )
                    {
                        const auto elem_mat = A.get_local_matrix(
                            info.sender_local_subdomain_id,
                            sender_cell[0], sender_cell[1], sender_cell[2], wedge );

                        // For each wedge node, compute sender grid position and check if
                        // it is on the shared boundary. If so, map to receiver coordinates.
                        int  recv_pos[6][3];
                        bool on_boundary[6];

                        for ( int n = 0; n < num_nodes_per_wedge; ++n )
                        {
                            const int dx = w_dx[wedge][n];
                            const int dy = w_dy[wedge][n];
                            const int dr = w_dr[wedge][n];

                            int sender_pos[3];
                            sender_pos[0] = sender_cell[0] + dx;
                            sender_pos[1] = sender_cell[1] + dy;
                            sender_pos[2] = sender_cell[2] + dr;

                            // A node is on the sender's boundary if its position along the
                            // fixed axis equals the boundary node value.
                            // For P1 (end): boundary node = num_nodes-1, cell offset must be 1.
                            // For P0 (start): boundary node = 0, cell offset must be 0.
                            const int offset_along_fixed[3] = { dx, dy, dr };
                            const bool at_boundary =
                                info.sender_at_end ?
                                    ( offset_along_fixed[info.sender_fixed_axis] == 1 ) :
                                    ( offset_along_fixed[info.sender_fixed_axis] == 0 );

                            on_boundary[n] = at_boundary;

                            if ( at_boundary )
                            {
                                // Map to receiver coordinates.
                                // Fixed axis: place at receiver's boundary.
                                recv_pos[n][info.receiver_fixed_axis] = info.receiver_boundary_node;

                                // Varying axes: map sender's varying values through direction flags.
                                const int sv0 = sender_pos[info.sender_var0_axis];
                                const int sv1 = sender_pos[info.sender_var1_axis];

                                recv_pos[n][info.receiver_var0_axis] =
                                    info.reverse_var0 ?
                                        ( num_nodes[info.receiver_var0_axis] - 1 - sv0 ) : sv0;
                                recv_pos[n][info.receiver_var1_axis] =
                                    info.reverse_var1 ?
                                        ( num_nodes[info.receiver_var1_axis] - 1 - sv1 ) : sv1;
                            }
                        }

                        // For each pair of boundary nodes in this ghost element, scatter
                        // the coupling entries into all receiver Vanka cells that contain
                        // both nodes.
                        for ( int a = 0; a < num_nodes_per_wedge; ++a )
                        {
                            if ( !on_boundary[a] )
                                continue;

                            for ( int b = 0; b < num_nodes_per_wedge; ++b )
                            {
                                if ( !on_boundary[b] )
                                    continue;

                                const int ga[3] = { recv_pos[a][0], recv_pos[a][1], recv_pos[a][2] };
                                const int gb[3] = { recv_pos[b][0], recv_pos[b][1], recv_pos[b][2] };

                                // Find all Vanka cells C on the receiver containing both nodes.
                                // Cell C at (xc, yc, rc) contains node g if xc <= gx <= xc+1, etc.
                                // So xc in [gx-1, gx], clamped to [0, nc-1].
                                const int xc_lo = ( ga[0] - 1 > gb[0] - 1 ? ga[0] - 1 : gb[0] - 1 ) > 0 ?
                                                      ( ga[0] - 1 > gb[0] - 1 ? ga[0] - 1 : gb[0] - 1 ) : 0;
                                const int xc_hi = ( ga[0] < gb[0] ? ga[0] : gb[0] ) < nc_x - 1 ?
                                                      ( ga[0] < gb[0] ? ga[0] : gb[0] ) : nc_x - 1;

                                const int yc_lo = ( ga[1] - 1 > gb[1] - 1 ? ga[1] - 1 : gb[1] - 1 ) > 0 ?
                                                      ( ga[1] - 1 > gb[1] - 1 ? ga[1] - 1 : gb[1] - 1 ) : 0;
                                const int yc_hi = ( ga[1] < gb[1] ? ga[1] : gb[1] ) < nc_y - 1 ?
                                                      ( ga[1] < gb[1] ? ga[1] : gb[1] ) : nc_y - 1;

                                const int rc_lo = ( ga[2] - 1 > gb[2] - 1 ? ga[2] - 1 : gb[2] - 1 ) > 0 ?
                                                      ( ga[2] - 1 > gb[2] - 1 ? ga[2] - 1 : gb[2] - 1 ) : 0;
                                const int rc_hi = ( ga[2] < gb[2] ? ga[2] : gb[2] ) < nc_r - 1 ?
                                                      ( ga[2] < gb[2] ? ga[2] : gb[2] ) : nc_r - 1;

                                for ( int xc = xc_lo; xc <= xc_hi; ++xc )
                                {
                                    for ( int yc = yc_lo; yc <= yc_hi; ++yc )
                                    {
                                        for ( int rc = rc_lo; rc <= rc_hi; ++rc )
                                        {
                                            // Cell-local node indices.
                                            const int ia = ( ga[0] - xc ) + 2 * ( ga[1] - yc ) + 4 * ( ga[2] - rc );
                                            const int ib = ( gb[0] - xc ) + 2 * ( gb[1] - yc ) + 4 * ( gb[2] - rc );

                                            // Accumulate the BlockSize x BlockSize coupling block.
                                            for ( int di = 0; di < BlockSize; ++di )
                                            {
                                                for ( int dj = 0; dj < BlockSize; ++dj )
                                                {
                                                    Kokkos::atomic_add(
                                                        &cm( info.receiver_local_subdomain_id, xc, yc, rc )(
                                                            ia * BlockSize + di, ib * BlockSize + dj ),
                                                        elem_mat(
                                                            a + di * num_nodes_per_wedge,
                                                            b + dj * num_nodes_per_wedge ) );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } );

            Kokkos::fence();
        }
    }
}

/// @brief Compute the inverse cell Vanka matrices by assembling the full local coupling matrix
///        for each hex cell from element matrices of its neighborhood, including ghost element
///        contributions from neighboring subdomains.
///
/// For each hex cell C at (xc, yc, rc), the CellDim x CellDim Vanka matrix is assembled by
/// iterating over all hex cells in the 3x3x3 neighborhood. For each neighbor cell's wedges,
/// the element matrix is retrieved and the coupling entries between nodes that belong to C
/// are accumulated.
///
/// After local assembly, ghost element contributions from face-adjacent subdomains are
/// accumulated. For boundary cells, neighboring subdomains have elements that couple shared
/// boundary nodes. These ghost elements' coupling entries are mapped to the receiver's
/// coordinate system (permuting row/column indices as needed for cross-diamond orientation
/// differences) and added to the cell matrices.
///
/// The resulting matrices are then inverted using LU decomposition.
///
/// @tparam OperatorT Operator type (must provide get_local_matrix() and LocalMatrixDim).
/// @tparam BlockSize Number of DoFs per node.
/// @param A Operator.
/// @param domain Distributed domain for cell iteration.
/// @return Kokkos view of inverse cell Vanka matrices, one per hex cell.
template < typename OperatorT, int BlockSize >
Kokkos::View< dense::PackedSymMat< float, 8 * BlockSize >****, grid::Layout >
compute_cell_vanka_matrices(
    const OperatorT&                      A,
    const grid::shell::DistributedDomain& domain )
{
    using ScalarType     = typename OperatorT::ScalarType;
    using CellMatrixType = dense::Mat< ScalarType, 8 * BlockSize, 8 * BlockSize >;
    using PackedType     = dense::PackedSymMat< float, 8 * BlockSize >;

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

    // -----------------------------------------------------------------------
    // Phase 1: Local assembly — same as before.
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Phase 2: Ghost element contributions from neighboring subdomains.
    //
    // For boundary cells, some elements that couple shared nodes live on
    // neighboring subdomains. We read those elements' matrices and scatter
    // their coupling entries (between pairs of shared boundary nodes) into
    // the local Vanka cell matrices.
    //
    // The element matrix values are coordinate-system independent, but the
    // row/column indices are tied to the sender's wedge-local node ordering.
    // When axes differ between sender and receiver (cross-diamond boundaries),
    // the node positions are permuted using the BoundaryDirection flags.
    //
    // Currently handles face boundaries with local neighbors (same MPI rank).
    // Edge and vertex boundary neighbors, and remote (MPI) neighbors, are not
    // yet handled — these affect a small minority of boundary cells.
    // -----------------------------------------------------------------------

    accumulate_ghost_element_contributions< OperatorT, BlockSize >( A, domain, cell_matrices );

    // -----------------------------------------------------------------------
    // Phase 2b: Fix diagonal blocks via additive cross-subdomain communication.
    //
    // Ghost element contributions (Phase 2) only handle face boundaries.
    // Edge/vertex boundary cells still have incomplete diagonal blocks.
    // Compute the complete block diagonal using the same additive communication
    // pattern as BlockJacobi, then replace the diagonal blocks in each cell matrix.
    // This ensures the dominant self-coupling entries are correct everywhere.
    // -----------------------------------------------------------------------

    {
        using BlockMatrixType = dense::Mat< ScalarType, BlockSize, BlockSize >;

        const auto num_nodes_x = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        const auto num_nodes_y = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        const auto num_nodes_r = domain.domain_info().subdomain_num_nodes_radially();

        // 1. Assemble block diagonal from local elements.
        Kokkos::View< BlockMatrixType****, grid::Layout > diag_blocks(
            "vanka_diag_blocks", num_subdomains, num_nodes_x, num_nodes_y, num_nodes_r );

        Kokkos::parallel_for(
            "CellVanka::extract_diag",
            grid::shell::local_domain_md_range_policy_cells( domain ),
            KOKKOS_LAMBDA( int local_subdomain, int x_cell, int y_cell, int r_cell ) {
                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    const auto local_mat = A.get_local_matrix( local_subdomain, x_cell, y_cell, r_cell, wedge );

                    for ( int n = 0; n < num_nodes_per_wedge; ++n )
                    {
                        const int gx = x_cell + ( wedge == 0 ? w0_dx[n] : w1_dx[n] );
                        const int gy = y_cell + ( wedge == 0 ? w0_dy[n] : w1_dy[n] );
                        const int gr = r_cell + ( wedge == 0 ? w0_dr[n] : w1_dr[n] );

                        for ( int di = 0; di < BlockSize; ++di )
                        {
                            for ( int dj = 0; dj < BlockSize; ++dj )
                            {
                                Kokkos::atomic_add(
                                    &diag_blocks( local_subdomain, gx, gy, gr )( di, dj ),
                                    local_mat( n + di * num_nodes_per_wedge, n + dj * num_nodes_per_wedge ) );
                            }
                        }
                    }
                }
            } );
        Kokkos::fence();

        // 2. Communicate block diagonal additively across all subdomain boundaries.
        for ( int row = 0; row < BlockSize; ++row )
        {
            grid::Grid4DDataVec< ScalarType, BlockSize > row_data(
                "vanka_diag_row", num_subdomains, num_nodes_x, num_nodes_y, num_nodes_r );

            Kokkos::parallel_for(
                "CellVanka::pack_diag_row",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 },
                    { static_cast< long long >( num_subdomains ),
                      static_cast< long long >( num_nodes_x ),
                      static_cast< long long >( num_nodes_y ),
                      static_cast< long long >( num_nodes_r ) } ),
                KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                    for ( int col = 0; col < BlockSize; ++col )
                    {
                        row_data( s, i, j, k, col ) = diag_blocks( s, i, j, k )( row, col );
                    }
                } );
            Kokkos::fence();

            communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, BlockSize > send_buf( domain );
            communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, BlockSize > recv_buf( domain );
            communication::shell::pack_send_and_recv_local_subdomain_boundaries( domain, row_data, send_buf, recv_buf );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain, row_data, recv_buf );

            Kokkos::parallel_for(
                "CellVanka::unpack_diag_row",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 },
                    { static_cast< long long >( num_subdomains ),
                      static_cast< long long >( num_nodes_x ),
                      static_cast< long long >( num_nodes_y ),
                      static_cast< long long >( num_nodes_r ) } ),
                KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                    for ( int col = 0; col < BlockSize; ++col )
                    {
                        diag_blocks( s, i, j, k )( row, col ) = row_data( s, i, j, k, col );
                    }
                } );
            Kokkos::fence();
        }

        // 3. Replace diagonal blocks in cell matrices with the fully communicated values.
        Kokkos::parallel_for(
            "CellVanka::fix_diag",
            grid::shell::local_domain_md_range_policy_cells( domain ),
            KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
                for ( int node = 0; node < 8; ++node )
                {
                    const int gx = xc + ( node % 2 );
                    const int gy = yc + ( ( node / 2 ) % 2 );
                    const int gr = rc + ( node / 4 );

                    for ( int di = 0; di < BlockSize; ++di )
                    {
                        for ( int dj = 0; dj < BlockSize; ++dj )
                        {
                            cell_matrices( local_subdomain, xc, yc, rc )( node * BlockSize + di, node * BlockSize + dj ) =
                                diag_blocks( local_subdomain, gx, gy, gr )( di, dj );
                        }
                    }
                }
            } );
        Kokkos::fence();
    }

    // -----------------------------------------------------------------------
    // Phase 3: Invert cell matrices and pack into mixed-precision symmetric storage.
    //
    // V^{-1} is symmetric (since V is SPD from the FEM assembly).
    // Store only the lower triangle in packed format using float:
    //   - Packed: N*(N+1)/2 = 300 entries (vs 576 for full N*N)
    //   - Float:  300 * 4B = 1200 bytes (vs 300 * 8B = 2400 bytes with double)
    // This quarters memory bandwidth vs full double storage in the smoother.
    // The matvec is computed in double via PackedSymMat's cross-type operator*.
    // -----------------------------------------------------------------------

    Kokkos::View< PackedType****, grid::Layout > packed_matrices(
        "cell_vanka_packed", num_subdomains, num_cells_x, num_cells_y, num_cells_r );

    Kokkos::parallel_for(
        "CellVanka::invert_and_pack",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
            const auto inv = cell_matrices( local_subdomain, xc, yc, rc ).inv();
            packed_matrices( local_subdomain, xc, yc, rc ) = PackedType::from_symmetric( inv );
        } );
    Kokkos::fence();

    return packed_matrices;
}

} // namespace terra::linalg::solvers
