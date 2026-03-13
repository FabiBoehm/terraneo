#pragma once

#include "cell_vanka.hpp"

#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"

namespace terra::linalg::solvers {

/// @brief Cell-based additive Vanka smoother for the full Stokes saddle-point system.
///
/// Generalizes CellVanka to include both velocity and pressure DOFs in the local system.
/// For each fine hex cell, the local system is a (VelDim + PresDim) x (VelDim + PresDim) saddle-point matrix:
/// \f[
///   \begin{bmatrix} A & B^T \\ B & 0 \end{bmatrix}
/// \f]
/// where A is the velocity-velocity coupling (24x24 for 3D), B^T is the gradient coupling (24x8),
/// and B is the divergence coupling (8x24).
///
/// Pressure DOFs are taken from the coarse grid cell at (xc/2, yc/2, rc/2) for a fine cell at (xc, yc, rc).
///
/// @tparam OperatorT Full Stokes operator type (must satisfy OperatorLike, with Block2VectorLike src/dst).
/// @tparam VecDim Number of velocity components per node (default 3).
template < OperatorLike OperatorT, int VecDim = 3 >
class CellVankaStokes
{
  public:
    /// @brief Operator type to be solved (full Stokes).
    using OperatorType = OperatorT;
    /// @brief Solution vector type (VectorQ1IsoQ2Q1).
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type (VectorQ1IsoQ2Q1).
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Scalar type for computations.
    using ScalarType = SolutionVectorType::ScalarType;

    /// @brief Number of velocity nodes per hex cell (fine grid).
    static constexpr int NumVelNodesPerCell = 8;
    /// @brief Number of pressure nodes per hex cell (coarse grid).
    static constexpr int NumPresNodesPerCell = 8;

    /// @brief Velocity dimension in the local system (8 nodes * VecDim).
    static constexpr int VelDim = NumVelNodesPerCell * VecDim;
    /// @brief Pressure dimension in the local system (8 nodes * 1).
    static constexpr int PresDim = NumPresNodesPerCell;
    /// @brief Total dimension of the cell-local saddle-point system.
    static constexpr int CellDim = VelDim + PresDim;

    /// @brief Dense cell matrix type.
    using CellMatrixType = dense::Mat< ScalarType, CellDim, CellDim >;

    /// @brief Kokkos view storing one inverse cell matrix per fine hex cell.
    using InverseCellMatricesType = Kokkos::View< CellMatrixType****, grid::Layout >;

    static_assert( VecDim > 0, "VecDim must be positive." );

    /// @brief Construct a CellVankaStokes solver.
    /// @param inverse_cell_matrices Pre-computed inverse Stokes cell Vanka matrices.
    /// @param iterations Number of smoothing iterations.
    /// @param tmp Temporary vector for workspace (residual).
    /// @param correction Temporary vector for workspace (accumulated correction).
    /// @param omega Relaxation parameter.
    /// @param domain_fine Fine grid domain for velocity correction communication.
    /// @param domain_coarse Coarse grid domain for pressure correction communication.
    CellVankaStokes(
        const InverseCellMatricesType& inverse_cell_matrices,
        const int                      iterations,
        const SolutionVectorType&      tmp,
        const SolutionVectorType&      correction,
        const ScalarType               omega        = static_cast< ScalarType >( 1.0 / 8.0 ),
        const grid::shell::DistributedDomain* domain_fine   = nullptr,
        const grid::shell::DistributedDomain* domain_coarse = nullptr )
    : inverse_cell_matrices_( inverse_cell_matrices )
    , iterations_( iterations )
    , tmp_( tmp )
    , correction_( correction )
    , omega_( omega )
    , domain_fine_( domain_fine )
    , domain_coarse_( domain_coarse )
    {}

    /// @brief Solve the linear system using additive Stokes cell Vanka iteration.
    void solve_impl( OperatorType& K, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iteration = 0; iteration < iterations_; ++iteration )
        {
            // tmp = K * x
            apply( K, x, tmp_ );

            // tmp = b - K * x (residual for both velocity and pressure)
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );

            // Zero correction only on first iteration; subsequent iterations reuse
            // the zeroing done by update_x_and_zero_correction.
            if ( iteration == 0 )
                linalg::assign( correction_, 0.0 );

            // Apply cell Vanka with coupled velocity-pressure local systems.
            apply_cell_vanka( tmp_ );

            // Communicate corrections across subdomain boundaries.
            if ( domain_fine_ )
            {
                communicate_velocity_correction();
            }
            if ( domain_coarse_ )
            {
                communicate_pressure_correction();
            }

            // x += omega * correction, then zero correction for next iteration.
            update_x_and_zero_correction( x );
        }
    }

    /// @brief Access the inverse cell matrices data.
    InverseCellMatricesType& get_inverse_cell_matrices() { return inverse_cell_matrices_; }

  private:
    /// @brief Apply additive Stokes cell Vanka to accumulate corrections.
    ///
    /// Gathers velocity residual from fine grid and pressure residual from coarse grid,
    /// multiplies by inverse cell matrix, and scatters corrections.
    /// Uses 8-coloring for velocity (non-overlapping), atomic adds for pressure
    /// (same-color fine cells may share coarse pressure nodes).
    ///
    /// Each color launches only the exact number of threads needed for cells of that
    /// color, avoiding wasted threads from filtering.
    void apply_cell_vanka( const SolutionVectorType& residual )
    {
        auto vel_res   = residual.block_1().grid_data();
        auto pres_res  = residual.block_2().grid_data();
        auto vel_corr  = correction_.block_1().grid_data();
        auto pres_corr = correction_.block_2().grid_data();
        auto inv_cells = inverse_cell_matrices_;

        const auto num_subdomains = static_cast< int >( inv_cells.extent( 0 ) );
        const auto num_cells_x    = static_cast< int >( inv_cells.extent( 1 ) );
        const auto num_cells_y    = static_cast< int >( inv_cells.extent( 2 ) );
        const auto num_cells_r    = static_cast< int >( inv_cells.extent( 3 ) );

        for ( int color = 0; color < 8; ++color )
        {
            const int color_x = color % 2;
            const int color_y = ( color / 2 ) % 2;
            const int color_r = color / 4;

            // Compute the number of cells of this color in each dimension.
            const int half_cells_x = ( num_cells_x - color_x + 1 ) / 2;
            const int half_cells_y = ( num_cells_y - color_y + 1 ) / 2;
            const int half_cells_r = ( num_cells_r - color_r + 1 ) / 2;

            if ( half_cells_x <= 0 || half_cells_y <= 0 || half_cells_r <= 0 )
                continue;

            Kokkos::parallel_for(
                "CellVankaStokes::apply_color",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 },
                    { num_subdomains, half_cells_x, half_cells_y, half_cells_r } ),
                KOKKOS_LAMBDA( int local_subdomain, int hx, int hy, int hr ) {
                    // Map half-indices back to actual cell coordinates.
                    const int xc = 2 * hx + color_x;
                    const int yc = 2 * hy + color_y;
                    const int rc = 2 * hr + color_r;

                    dense::Vec< ScalarType, CellDim > local_res;

                    // Gather velocity residual (VelDim = 24 entries).
                    for ( int node = 0; node < NumVelNodesPerCell; ++node )
                    {
                        const int gx = xc + ( node % 2 );
                        const int gy = yc + ( ( node / 2 ) % 2 );
                        const int gr = rc + ( node / 4 );

                        for ( int d = 0; d < VecDim; ++d )
                        {
                            local_res( node * VecDim + d ) = vel_res( local_subdomain, gx, gy, gr, d );
                        }
                    }

                    // Gather pressure residual (PresDim = 8 entries).
                    const int xc_c = xc / 2;
                    const int yc_c = yc / 2;
                    const int rc_c = rc / 2;

                    for ( int node = 0; node < NumPresNodesPerCell; ++node )
                    {
                        const int gxp = xc_c + ( node % 2 );
                        const int gyp = yc_c + ( ( node / 2 ) % 2 );
                        const int grp = rc_c + ( node / 4 );

                        local_res( VelDim + node ) = pres_res( local_subdomain, gxp, gyp, grp );
                    }

                    // Multiply by inverse cell matrix.
                    const auto local_corr = inv_cells( local_subdomain, xc, yc, rc ) * local_res;

                    // Scatter velocity correction (no atomic: same-color cells don't share velocity nodes).
                    for ( int node = 0; node < NumVelNodesPerCell; ++node )
                    {
                        const int gx = xc + ( node % 2 );
                        const int gy = yc + ( ( node / 2 ) % 2 );
                        const int gr = rc + ( node / 4 );

                        for ( int d = 0; d < VecDim; ++d )
                        {
                            vel_corr( local_subdomain, gx, gy, gr, d ) +=
                                local_corr( node * VecDim + d );
                        }
                    }

                    // Scatter pressure correction (atomic: same-color fine cells may share coarse pressure nodes).
                    for ( int node = 0; node < NumPresNodesPerCell; ++node )
                    {
                        const int gxp = xc_c + ( node % 2 );
                        const int gyp = yc_c + ( ( node / 2 ) % 2 );
                        const int grp = rc_c + ( node / 4 );

                        Kokkos::atomic_add(
                            &pres_corr( local_subdomain, gxp, gyp, grp ),
                            local_corr( VelDim + node ) );
                    }
                } );
        }

        Kokkos::fence();
    }

    /// @brief Fused kernel: x += omega * correction and zero correction for next iteration.
    ///
    /// Combines the update of x and the zeroing of the correction vector into a single
    /// set of kernel launches, saving kernel launches per iteration compared to separate
    /// lincomb + assign calls. Velocity and pressure blocks are updated in separate kernels
    /// due to different data layouts (5D vs 4D views).
    void update_x_and_zero_correction( SolutionVectorType& x )
    {
        auto omega = omega_;

        // Update velocity block.
        {
            auto x_data    = x.block_1().grid_data();
            auto corr_data = correction_.block_1().grid_data();

            Kokkos::parallel_for(
                "CellVankaStokes::update_vel_and_zero",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 },
                    { static_cast< int >( x_data.extent( 0 ) ),
                      static_cast< int >( x_data.extent( 1 ) ),
                      static_cast< int >( x_data.extent( 2 ) ),
                      static_cast< int >( x_data.extent( 3 ) ) } ),
                KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                    for ( int d = 0; d < VecDim; ++d )
                    {
                        x_data( s, i, j, k, d ) += omega * corr_data( s, i, j, k, d );
                        corr_data( s, i, j, k, d ) = ScalarType( 0 );
                    }
                } );
        }

        // Update pressure block (scalar, no component loop).
        {
            auto x_data    = x.block_2().grid_data();
            auto corr_data = correction_.block_2().grid_data();

            Kokkos::parallel_for(
                "CellVankaStokes::update_pres_and_zero",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 },
                    { static_cast< int >( x_data.extent( 0 ) ),
                      static_cast< int >( x_data.extent( 1 ) ),
                      static_cast< int >( x_data.extent( 2 ) ),
                      static_cast< int >( x_data.extent( 3 ) ) } ),
                KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                    x_data( s, i, j, k ) += omega * corr_data( s, i, j, k );
                    corr_data( s, i, j, k ) = ScalarType( 0 );
                } );
        }

        Kokkos::fence();
    }

    /// @brief Additively communicate velocity correction across subdomain boundaries.
    void communicate_velocity_correction()
    {
        auto corr_data = correction_.block_1().grid_data();
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, VecDim > send_buf( *domain_fine_ );
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, VecDim > recv_buf( *domain_fine_ );
        communication::shell::pack_send_and_recv_local_subdomain_boundaries( *domain_fine_, corr_data, send_buf, recv_buf );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( *domain_fine_, corr_data, recv_buf );
    }

    /// @brief Additively communicate pressure correction across subdomain boundaries.
    void communicate_pressure_correction()
    {
        auto corr_data = correction_.block_2().grid_data();
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > send_buf( *domain_coarse_ );
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > recv_buf( *domain_coarse_ );
        communication::shell::pack_send_and_recv_local_subdomain_boundaries( *domain_coarse_, corr_data, send_buf, recv_buf );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( *domain_coarse_, corr_data, recv_buf );
    }

    InverseCellMatricesType inverse_cell_matrices_;  ///< Inverse cell Vanka matrices.
    int                     iterations_;              ///< Number of iterations.
    SolutionVectorType      tmp_;                     ///< Temporary workspace (residual).
    SolutionVectorType      correction_;              ///< Temporary workspace (correction).
    ScalarType              omega_;                   ///< Relaxation parameter.
    const grid::shell::DistributedDomain* domain_fine_;   ///< Fine grid domain for velocity communication.
    const grid::shell::DistributedDomain* domain_coarse_; ///< Coarse grid domain for pressure communication.
};

/// @brief Compute the inverse cell Vanka matrices for the full Stokes saddle-point system.
///
/// For each fine hex cell C at (xc, yc, rc), assembles the (VelDim+PresDim) x (VelDim+PresDim) matrix:
/// @code
///   [ A    B^T ]
///   [ B     0  ]
/// @endcode
/// where:
///   - A is the velocity-velocity coupling (from the viscous operator, same as velocity-only Vanka)
///   - B^T is the gradient coupling (velocity-pressure, computed from FE shape functions)
///   - B is the divergence coupling (pressure-velocity, B = (B^T)^T)
///
/// Velocity DOFs: 8 fine-grid nodes * VecDim components = VelDim
/// Pressure DOFs: 8 coarse-grid nodes of coarse cell (xc/2, yc/2, rc/2) = PresDim
///
/// @tparam ViscousT Viscous operator type (velocity block, must provide get_local_matrix()).
/// @tparam ScalarT Scalar type.
/// @tparam VecDim Number of velocity components per node (default 3).
/// @param A Viscous operator.
/// @param domain_fine Fine grid domain (velocity).
/// @param domain_coarse Coarse grid domain (pressure).
/// @param grid Fine grid physical coordinates.
/// @param radii Fine grid radial coordinates.
/// @param treat_boundary Whether to apply boundary treatment to gradient coupling.
template < typename ViscousT, typename ScalarT, int VecDim = 3 >
Kokkos::View< dense::Mat< ScalarT, 8 * VecDim + 8, 8 * VecDim + 8 >****, grid::Layout >
compute_stokes_cell_vanka_matrices(
    const ViscousT&                            A,
    const grid::shell::DistributedDomain&      domain_fine,
    const grid::shell::DistributedDomain&      domain_coarse,
    const grid::Grid3DDataVec< ScalarT, 3 >&   grid,
    const grid::Grid2DDataScalar< ScalarT >&   radii,
    const bool                                 treat_boundary )
{
    using CellMatrixType = dense::Mat< ScalarT, 8 * VecDim + 8, 8 * VecDim + 8 >;

    constexpr int num_nodes_per_wedge = 6;
    constexpr int VelDim              = 8 * VecDim;
    constexpr int CellDim             = VelDim + 8;

    const auto num_subdomains = domain_fine.subdomains().size();
    const auto num_cells_x    = domain_fine.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto num_cells_y    = domain_fine.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto num_cells_r    = domain_fine.domain_info().subdomain_num_nodes_radially() - 1;

    Kokkos::View< CellMatrixType****, grid::Layout > cell_matrices(
        "stokes_cell_vanka_matrices", num_subdomains, num_cells_x, num_cells_y, num_cells_r );

    // Wedge local node to hex node offset mapping (same as cell_vanka.hpp).
    constexpr int w0_dx[6] = { 0, 1, 0, 0, 1, 0 };
    constexpr int w0_dy[6] = { 0, 0, 1, 0, 0, 1 };
    constexpr int w0_dr[6] = { 0, 0, 0, 1, 1, 1 };
    constexpr int w1_dx[6] = { 1, 0, 1, 1, 0, 1 };
    constexpr int w1_dy[6] = { 1, 1, 0, 1, 1, 0 };
    constexpr int w1_dr[6] = { 0, 0, 0, 1, 1, 1 };

    const int nc_x = static_cast< int >( num_cells_x );
    const int nc_y = static_cast< int >( num_cells_y );
    const int nc_r = static_cast< int >( num_cells_r );

    auto grid_data  = grid;
    auto radii_data = radii;

    // -----------------------------------------------------------------------
    // Assemble cell matrices.
    // -----------------------------------------------------------------------
    Kokkos::parallel_for(
        "CellVankaStokes::assemble",
        grid::shell::local_domain_md_range_policy_cells( domain_fine ),
        KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
            CellMatrixType V;
            V.fill( ScalarT( 0 ) );

            const int xc_c = xc / 2;
            const int yc_c = yc / 2;
            const int rc_c = rc / 2;

            // ===== VELOCITY-VELOCITY BLOCK (rows 0..VelDim-1, cols 0..VelDim-1) =====
            // Iterate over 3x3x3 neighborhood of fine hex cells.
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

                        for ( int wedge = 0; wedge < 2; ++wedge )
                        {
                            const auto local_mat =
                                A.get_local_matrix( local_subdomain, xn, yn, rn, wedge );

                            for ( int a = 0; a < num_nodes_per_wedge; ++a )
                            {
                                const int gxa = xn + ( wedge == 0 ? w0_dx[a] : w1_dx[a] );
                                const int gya = yn + ( wedge == 0 ? w0_dy[a] : w1_dy[a] );
                                const int gra = rn + ( wedge == 0 ? w0_dr[a] : w1_dr[a] );

                                const int dxa = gxa - xc;
                                const int dya = gya - yc;
                                const int dra = gra - rc;
                                if ( dxa < 0 || dxa > 1 || dya < 0 || dya > 1 || dra < 0 || dra > 1 )
                                    continue;

                                const int ia = dxa + 2 * dya + 4 * dra;

                                for ( int b = 0; b < num_nodes_per_wedge; ++b )
                                {
                                    const int gxb = xn + ( wedge == 0 ? w0_dx[b] : w1_dx[b] );
                                    const int gyb = yn + ( wedge == 0 ? w0_dy[b] : w1_dy[b] );
                                    const int grb = rn + ( wedge == 0 ? w0_dr[b] : w1_dr[b] );

                                    const int dxb = gxb - xc;
                                    const int dyb = gyb - yc;
                                    const int drb = grb - rc;
                                    if ( dxb < 0 || dxb > 1 || dyb < 0 || dyb > 1 || drb < 0 || drb > 1 )
                                        continue;

                                    const int ib = dxb + 2 * dyb + 4 * drb;

                                    for ( int di = 0; di < VecDim; ++di )
                                    {
                                        for ( int dj = 0; dj < VecDim; ++dj )
                                        {
                                            V( ia * VecDim + di, ib * VecDim + dj ) +=
                                                local_mat(
                                                    a + di * num_nodes_per_wedge,
                                                    b + dj * num_nodes_per_wedge );
                                        }
                                    }
                                }
                            }

                            // ===== GRADIENT / DIVERGENCE COUPLING =====
                            // Compute gradient local matrix for this fine element.
                            // Maps velocity nodes (fine) to pressure nodes (coarse).

                            const int xn_c = xn / 2;
                            const int yn_c = yn / 2;
                            const int rn_c = rn / 2;

                            // Compute gradient local element matrix (18 x 6).
                            dense::Vec< ScalarT, 3 >
                                wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                              [fe::wedge::num_nodes_per_wedge_surface] = {};
                            fe::wedge::wedge_surface_physical_coords(
                                wedge_phy_surf, grid_data, local_subdomain, xn, yn );

                            const ScalarT r_1 = radii_data( local_subdomain, rn );
                            const ScalarT r_2 = radii_data( local_subdomain, rn + 1 );

                            constexpr auto num_quad_points =
                                fe::wedge::quadrature::quad_felippa_1x1_num_quad_points;
                            dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
                            ScalarT                  quad_weights[num_quad_points];
                            fe::wedge::quadrature::quad_felippa_1x1_quad_points( quad_points );
                            fe::wedge::quadrature::quad_felippa_1x1_quad_weights( quad_weights );

                            const int fine_radial_wedge_index  = rn % 2;
                            const int fine_lateral_wedge_index =
                                fe::wedge::fine_lateral_wedge_idx( xn, yn, wedge );

                            dense::Mat< ScalarT, 18, 6 > G_e;
                            G_e.fill( ScalarT( 0 ) );

                            for ( int q = 0; q < num_quad_points; ++q )
                            {
                                const auto J = fe::wedge::jac(
                                    wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                                const auto det       = Kokkos::abs( J.det() );
                                const auto J_inv_T   = J.inv().transposed();

                                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                                {
                                    const auto grad_i =
                                        fe::wedge::grad_shape( i, quad_points[q] );

                                    for ( int j = 0; j < num_nodes_per_wedge; ++j )
                                    {
                                        const auto shape_j = fe::wedge::shape_coarse(
                                            j,
                                            fine_radial_wedge_index,
                                            fine_lateral_wedge_index,
                                            quad_points[q] );

                                        for ( int d = 0; d < 3; ++d )
                                        {
                                            G_e( d * 6 + i, j ) += quad_weights[q] *
                                                ( -( ( J_inv_T * grad_i )( d ) * shape_j ) * det );
                                        }
                                    }
                                }
                            }

                            // Apply boundary treatment to gradient matrix.
                            if ( treat_boundary )
                            {
                                const int num_radii_fine =
                                    static_cast< int >( radii_data.extent( 1 ) );

                                if ( rn == 0 )
                                {
                                    // Inner boundary: zero velocity rows for inner nodes (i < 3).
                                    for ( int d = 0; d < 3; ++d )
                                        for ( int i = 0; i < 6; ++i )
                                            for ( int j = 0; j < 6; ++j )
                                                if ( i < 3 )
                                                    G_e( 6 * d + i, j ) = ScalarT( 0 );
                                }
                                if ( rn + 1 == num_radii_fine - 1 )
                                {
                                    // Outer boundary: zero velocity rows for outer nodes (i >= 3).
                                    for ( int d = 0; d < 3; ++d )
                                        for ( int i = 0; i < 6; ++i )
                                            for ( int j = 0; j < 6; ++j )
                                                if ( i >= 3 )
                                                    G_e( 6 * d + i, j ) = ScalarT( 0 );
                                }
                            }

                            // Map gradient entries to cell matrix.
                            // For each (velocity node a, pressure node j) pair where
                            // velocity node is in target fine cell and pressure node is in target coarse cell.
                            for ( int a = 0; a < num_nodes_per_wedge; ++a )
                            {
                                const int gxa = xn + ( wedge == 0 ? w0_dx[a] : w1_dx[a] );
                                const int gya = yn + ( wedge == 0 ? w0_dy[a] : w1_dy[a] );
                                const int gra = rn + ( wedge == 0 ? w0_dr[a] : w1_dr[a] );

                                const int dxa = gxa - xc;
                                const int dya = gya - yc;
                                const int dra = gra - rc;
                                if ( dxa < 0 || dxa > 1 || dya < 0 || dya > 1 || dra < 0 || dra > 1 )
                                    continue;

                                const int iv = dxa + 2 * dya + 4 * dra;

                                for ( int j = 0; j < num_nodes_per_wedge; ++j )
                                {
                                    // Pressure node j -> coarse grid coords.
                                    const int gxp = xn_c + ( wedge == 0 ? w0_dx[j] : w1_dx[j] );
                                    const int gyp = yn_c + ( wedge == 0 ? w0_dy[j] : w1_dy[j] );
                                    const int grp = rn_c + ( wedge == 0 ? w0_dr[j] : w1_dr[j] );

                                    const int dxp = gxp - xc_c;
                                    const int dyp = gyp - yc_c;
                                    const int drp = grp - rc_c;
                                    if ( dxp < 0 || dxp > 1 || dyp < 0 || dyp > 1 || drp < 0 || drp > 1 )
                                        continue;

                                    const int ip = dxp + 2 * dyp + 4 * drp;

                                    for ( int d = 0; d < VecDim; ++d )
                                    {
                                        // B^T block (gradient): velocity row, pressure column.
                                        V( iv * VecDim + d, VelDim + ip ) +=
                                            G_e( d * 6 + a, j );
                                        // B block (divergence = G^T): pressure row, velocity column.
                                        V( VelDim + ip, iv * VecDim + d ) +=
                                            G_e( d * 6 + a, j );
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
    // Fix velocity diagonal blocks with cross-subdomain communication.
    // Same approach as in compute_cell_vanka_matrices (cell_vanka.hpp).
    // -----------------------------------------------------------------------

    using BlockMatrixType = dense::Mat< ScalarT, VecDim, VecDim >;

    const auto num_nodes_x = domain_fine.domain_info().subdomain_num_nodes_per_side_laterally();
    const auto num_nodes_y = domain_fine.domain_info().subdomain_num_nodes_per_side_laterally();
    const auto num_nodes_r = domain_fine.domain_info().subdomain_num_nodes_radially();

    // 1. Assemble block diagonal from local elements.
    Kokkos::View< BlockMatrixType****, grid::Layout > diag_blocks(
        "stokes_vanka_diag_blocks", num_subdomains, num_nodes_x, num_nodes_y, num_nodes_r );

    Kokkos::parallel_for(
        "CellVankaStokes::extract_diag",
        grid::shell::local_domain_md_range_policy_cells( domain_fine ),
        KOKKOS_LAMBDA( int local_subdomain, int x_cell, int y_cell, int r_cell ) {
            for ( int wedge = 0; wedge < 2; ++wedge )
            {
                const auto local_mat = A.get_local_matrix( local_subdomain, x_cell, y_cell, r_cell, wedge );

                for ( int n = 0; n < num_nodes_per_wedge; ++n )
                {
                    const int gx = x_cell + ( wedge == 0 ? w0_dx[n] : w1_dx[n] );
                    const int gy = y_cell + ( wedge == 0 ? w0_dy[n] : w1_dy[n] );
                    const int gr = r_cell + ( wedge == 0 ? w0_dr[n] : w1_dr[n] );

                    for ( int di = 0; di < VecDim; ++di )
                    {
                        for ( int dj = 0; dj < VecDim; ++dj )
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

    // 2. Communicate block diagonal additively across subdomain boundaries.
    for ( int row = 0; row < VecDim; ++row )
    {
        grid::Grid4DDataVec< ScalarT, VecDim > row_data(
            "stokes_vanka_diag_row", num_subdomains, num_nodes_x, num_nodes_y, num_nodes_r );

        Kokkos::parallel_for(
            "CellVankaStokes::pack_diag_row",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { static_cast< long long >( num_subdomains ),
                  static_cast< long long >( num_nodes_x ),
                  static_cast< long long >( num_nodes_y ),
                  static_cast< long long >( num_nodes_r ) } ),
            KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                for ( int col = 0; col < VecDim; ++col )
                {
                    row_data( s, i, j, k, col ) = diag_blocks( s, i, j, k )( row, col );
                }
            } );
        Kokkos::fence();

        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buf( domain_fine );
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buf( domain_fine );
        communication::shell::pack_send_and_recv_local_subdomain_boundaries( domain_fine, row_data, send_buf, recv_buf );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_fine, row_data, recv_buf );

        Kokkos::parallel_for(
            "CellVankaStokes::unpack_diag_row",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { static_cast< long long >( num_subdomains ),
                  static_cast< long long >( num_nodes_x ),
                  static_cast< long long >( num_nodes_y ),
                  static_cast< long long >( num_nodes_r ) } ),
            KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                for ( int col = 0; col < VecDim; ++col )
                {
                    diag_blocks( s, i, j, k )( row, col ) = row_data( s, i, j, k, col );
                }
            } );
        Kokkos::fence();
    }

    // 3. Replace velocity diagonal blocks in cell matrices.
    Kokkos::parallel_for(
        "CellVankaStokes::fix_diag",
        grid::shell::local_domain_md_range_policy_cells( domain_fine ),
        KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
            for ( int node = 0; node < 8; ++node )
            {
                const int gx = xc + ( node % 2 );
                const int gy = yc + ( ( node / 2 ) % 2 );
                const int gr = rc + ( node / 4 );

                for ( int di = 0; di < VecDim; ++di )
                {
                    for ( int dj = 0; dj < VecDim; ++dj )
                    {
                        cell_matrices( local_subdomain, xc, yc, rc )(
                            node * VecDim + di, node * VecDim + dj ) =
                            diag_blocks( local_subdomain, gx, gy, gr )( di, dj );
                    }
                }
            }
        } );
    Kokkos::fence();

    // -----------------------------------------------------------------------
    // Invert all cell matrices.
    // -----------------------------------------------------------------------
    Kokkos::parallel_for(
        "CellVankaStokes::invert",
        grid::shell::local_domain_md_range_policy_cells( domain_fine ),
        KOKKOS_LAMBDA( int local_subdomain, int xc, int yc, int rc ) {
            cell_matrices( local_subdomain, xc, yc, rc ) =
                cell_matrices( local_subdomain, xc, yc, rc ).inv();
        } );
    Kokkos::fence();

    return cell_matrices;
}

} // namespace terra::linalg::solvers
