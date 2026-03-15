#pragma once

#include "solver.hpp"

#include "communication/shell/communication.hpp"
#include "terra/dense/packed_sym_mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/bit_masks.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"

namespace terra::linalg::solvers {

/// @brief Multiplicative cell-based Vanka smoother with 8-coloring.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Unlike additive CellVanka which applies all cell corrections simultaneously,
/// this variant processes cells color-by-color, recomputing the residual between
/// colors so that each color sees the corrections from previous colors.
///
/// 8-coloring: color = (xc%2) + 2*(yc%2) + 4*(rc%2).
/// Cells of the same color share no DOFs within a subdomain, so corrections
/// within a color can be applied without race conditions.
///
/// After each color pass, x is synchronized across subdomains: halo DOFs are
/// zeroed and additively communicated, effectively replacing them with the
/// owning subdomain's values. This is necessary because the FE operator
/// communicates Ax (not x), so halo values of x become stale after local updates.
///
/// Each smoothing step requires 8 operator applies (one per color), but the
/// multiplicative nature gives better convergence per step.
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam BlockSize Size of the per-node blocks (number of DoFs per node).
template < OperatorLike OperatorT, int BlockSize >
class CellVankaMultiplicative
{
  public:
    using OperatorType      = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType     = DstOf< OperatorType >;
    using ScalarType        = typename SolutionVectorType::ScalarType;

    static constexpr int NumNodesPerCell = 8;
    static constexpr int CellDim         = NumNodesPerCell * BlockSize;

    using CellMatrixType       = dense::Mat< ScalarType, CellDim, CellDim >;
    using StorageScalarType    = float;
    using PackedCellMatrixType = dense::PackedSymMat< StorageScalarType, CellDim >;
    using InverseCellMatricesType = Kokkos::View< PackedCellMatrixType****, grid::Layout >;

    static_assert( BlockSize > 0, "BlockSize must be positive." );

    /// @brief Construct a multiplicative CellVanka solver.
    /// @param inverse_cell_matrices Pre-computed inverse cell Vanka matrices.
    /// @param iterations Number of full multiplicative sweeps (each sweep = 8 color passes).
    /// @param tmp Temporary vector for workspace (residual Ax).
    /// @param correction Unused (kept for API compatibility with CellVanka).
    /// @param omega Relaxation parameter (default 1.0 for multiplicative).
    /// @param domain Domain pointer for inter-subdomain x synchronization.
    CellVankaMultiplicative(
        const InverseCellMatricesType& inverse_cell_matrices,
        const int                      iterations,
        const SolutionVectorType&      tmp,
        const SolutionVectorType&      /* correction */,
        const ScalarType               omega  = ScalarType( 1 ),
        const grid::shell::DistributedDomain* domain = nullptr,
        const ScalarType               /* chebyshev_lambda_max */ = ScalarType( 0 ) )
    : inverse_cell_matrices_( inverse_cell_matrices )
    , iterations_( iterations )
    , tmp_( tmp )
    , omega_( omega )
    , domain_( domain )
    {}

    /// @brief Multiplicative Vanka sweep: 8 color passes per iteration.
    ///
    /// For each color:
    ///   1. Compute Ax (operator communicates Ax additively)
    ///   2. Apply Vanka for cells of this color → update x directly
    ///   3. Synchronize x across subdomains (zero halo + additive communicate)
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iteration = 0; iteration < iterations_; ++iteration )
        {
            for ( int color = 0; color < 8; ++color )
            {
                apply( A, x, tmp_ );
                apply_cell_vanka_color_direct( x, b, color );
                if ( domain_ )
                    synchronize_x( x );
            }
        }
    }

    InverseCellMatricesType& get_inverse_cell_matrices() { return inverse_cell_matrices_; }

  private:
    /// @brief Synchronize x across subdomains after a color pass.
    ///
    /// Zeroes halo DOFs (non-owned) then additively communicates x.
    /// Since halo DOFs are 0 and owned DOFs have the correct value,
    /// the additive communication effectively replaces halo values
    /// with the owning subdomain's values.
    void synchronize_x( SolutionVectorType& x )
    {
        auto x_data = x.grid_data();
        auto mask   = x.mask_data();

        // Zero non-owned (halo) DOFs.
        Kokkos::parallel_for(
            "CellVankaMult::zero_halo",
            Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >(
                { 0, 0, 0, 0, 0 },
                { static_cast< long long >( x_data.extent( 0 ) ),
                  static_cast< long long >( x_data.extent( 1 ) ),
                  static_cast< long long >( x_data.extent( 2 ) ),
                  static_cast< long long >( x_data.extent( 3 ) ),
                  static_cast< long long >( x_data.extent( 4 ) ) } ),
            KOKKOS_LAMBDA( int s, int i, int j, int k, int d ) {
                if ( mask( s, i, j, k ) != grid::NodeOwnershipFlag::OWNED )
                    x_data( s, i, j, k, d ) = ScalarType( 0 );
            } );
        Kokkos::fence();

        // Additive communication: halo DOFs receive owner's values.
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, BlockSize > send_buf( *domain_ );
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, BlockSize > recv_buf( *domain_ );
        communication::shell::pack_send_and_recv_local_subdomain_boundaries( *domain_, x_data, send_buf, recv_buf );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( *domain_, x_data, recv_buf );
    }

    static constexpr bool is_gpu =
#if defined( KOKKOS_ENABLE_CUDA ) || defined( KOKKOS_ENABLE_HIP ) || defined( KOKKOS_ENABLE_SYCL )
        true;
#else
        false;
#endif

    using TeamPolicy = Kokkos::TeamPolicy<>;
    using TeamMember = typename TeamPolicy::member_type;
    using ScratchSpace = typename Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchFloatView =
        Kokkos::View< StorageScalarType*, ScratchSpace, Kokkos::MemoryTraits< Kokkos::Unmanaged > >;

    static constexpr int PackedSize = PackedCellMatrixType::size;

    /// @brief Apply Vanka correction for all cells of a given color directly to x.
    ///
    /// Cells of the same color do not share any DOFs within a subdomain,
    /// so corrections are applied directly to x without atomics.
    void apply_cell_vanka_color_direct( SolutionVectorType& x, const RHSVectorType& b, const int color )
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

        const int color_x = color % 2;
        const int color_y = ( color / 2 ) % 2;
        const int color_r = color / 4;

        const int num_color_cells_x = ( num_cells_x - color_x + 1 ) / 2;
        const int num_color_cells_y = ( num_cells_y - color_y + 1 ) / 2;
        const int num_color_cells_r = ( num_cells_r - color_r + 1 ) / 2;

        if ( num_color_cells_x <= 0 || num_color_cells_y <= 0 || num_color_cells_r <= 0 )
            return;

        if constexpr ( is_gpu )
        {
            const int total_cells = num_subdomains * num_color_cells_x * num_color_cells_y * num_color_cells_r;
            if ( total_cells == 0 )
                return;

            const size_t scratch_bytes =
                ScratchFloatView::shmem_size( PackedSize ) +
                ScratchFloatView::shmem_size( CellDim );

            TeamPolicy policy( total_cells, Kokkos::AUTO );
            policy = policy.set_scratch_size( 0, Kokkos::PerTeam( scratch_bytes ) );

            Kokkos::parallel_for(
                "CellVankaMult::apply_color",
                policy,
                KOKKOS_LAMBDA( const TeamMember& team ) {
                    int tmp_idx = team.league_rank();
                    const int cr_idx = tmp_idx % num_color_cells_r;
                    tmp_idx /= num_color_cells_r;
                    const int cy_idx = tmp_idx % num_color_cells_y;
                    tmp_idx /= num_color_cells_y;
                    const int cx_idx = tmp_idx % num_color_cells_x;
                    const int local_subdomain = tmp_idx / num_color_cells_x;

                    const int xc = color_x + 2 * cx_idx;
                    const int yc = color_y + 2 * cy_idx;
                    const int rc = color_r + 2 * cr_idx;

                    ScratchFloatView mat_scratch( team.team_shmem(), PackedSize );
                    ScratchFloatView res_scratch( team.team_shmem(), CellDim );

                    const auto& packed_mat = inv_cells( local_subdomain, xc, yc, rc );
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange( team, PackedSize ),
                        [&]( int k ) { mat_scratch( k ) = packed_mat.data[k]; } );

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange( team, CellDim ),
                        [&]( int k ) {
                            const int node = k / BlockSize;
                            const int d    = k % BlockSize;
                            const int gx   = xc + ( node % 2 );
                            const int gy   = yc + ( ( node / 2 ) % 2 );
                            const int gr   = rc + ( node / 4 );
                            res_scratch( k ) = static_cast< StorageScalarType >(
                                b_data( local_subdomain, gx, gy, gr, d ) -
                                ax_data( local_subdomain, gx, gy, gr, d ) );
                        } );

                    team.team_barrier();

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange( team, CellDim ),
                        [&]( int i ) {
                            StorageScalarType sum = StorageScalarType( 0 );
                            const int         base_i = i * ( i + 1 ) / 2;
                            for ( int j = 0; j <= i; ++j )
                                sum += mat_scratch( base_i + j ) * res_scratch( j );
                            for ( int j = i + 1; j < CellDim; ++j )
                                sum += mat_scratch( j * ( j + 1 ) / 2 + i ) * res_scratch( j );

                            const int node = i / BlockSize;
                            const int d    = i % BlockSize;
                            const int gx   = xc + ( node % 2 );
                            const int gy   = yc + ( ( node / 2 ) % 2 );
                            const int gr   = rc + ( node / 4 );
                            x_data( local_subdomain, gx, gy, gr, d ) +=
                                omega * static_cast< ScalarType >( sum );
                        } );
                } );
        }
        else
        {
            Kokkos::parallel_for(
                "CellVankaMult::apply_color",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 },
                    { num_subdomains, num_color_cells_x, num_color_cells_y, num_color_cells_r } ),
                KOKKOS_LAMBDA( int local_subdomain, int cx_idx, int cy_idx, int cr_idx ) {
                    const int xc = color_x + 2 * cx_idx;
                    const int yc = color_y + 2 * cy_idx;
                    const int rc = color_r + 2 * cr_idx;

                    dense::Vec< StorageScalarType, CellDim > local_res;
                    for ( int node = 0; node < NumNodesPerCell; ++node )
                    {
                        const int gx = xc + ( node % 2 );
                        const int gy = yc + ( ( node / 2 ) % 2 );
                        const int gr = rc + ( node / 4 );
                        for ( int d = 0; d < BlockSize; ++d )
                            local_res( node * BlockSize + d ) = static_cast< StorageScalarType >(
                                b_data( local_subdomain, gx, gy, gr, d ) -
                                ax_data( local_subdomain, gx, gy, gr, d ) );
                    }

                    const auto local_corr = inv_cells( local_subdomain, xc, yc, rc ) * local_res;

                    for ( int node = 0; node < NumNodesPerCell; ++node )
                    {
                        const int gx = xc + ( node % 2 );
                        const int gy = yc + ( ( node / 2 ) % 2 );
                        const int gr = rc + ( node / 4 );
                        for ( int d = 0; d < BlockSize; ++d )
                            x_data( local_subdomain, gx, gy, gr, d ) +=
                                omega * static_cast< ScalarType >( local_corr( node * BlockSize + d ) );
                    }
                } );
        }

        Kokkos::fence();
    }

    InverseCellMatricesType inverse_cell_matrices_;
    int                     iterations_;
    SolutionVectorType      tmp_;
    ScalarType              omega_;
    const grid::shell::DistributedDomain* domain_;
};

} // namespace terra::linalg::solvers
