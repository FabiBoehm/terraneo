#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"
#include "warp_tile.hpp"

#include <cstdlib>

namespace terra::fe::wedge::operators::shell {

using grid::shell::get_boundary_condition_flag;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using terra::grid::shell::BoundaryConditionFlag;
using terra::grid::shell::BoundaryConditions;
using terra::grid::shell::ShellBoundaryFlag;
using terra::linalg::trafo::trafo_mat_cartesian_to_normal_tangential;

/// Minimum resident teams per SM the warp paths of Gradient and Divergence are
/// compiled for (the register cap is 65536 / (64 * this)).
#ifndef TERRA_GRADDIV_MIN_BLOCKS
#define TERRA_GRADDIV_MIN_BLOCKS 8
#endif

// CoordScalarT is the precision of the STORED node coordinates and radii. It
// defaults to ScalarT, so existing instantiations are unchanged. Pass double to
// decouple it: the Jacobian is formed by differencing coordinates, which loses the
// geometry entirely in 16-bit and is already marginal in single. ScalarT governs
// the stored fields src/dst and the coefficient -- where the memory traffic is.
// ArithScalarT is the precision of the warp path's staged pressure tile and of its
// per-cell contraction; it defaults to ScalarT.
//
// Two paths: the per-cell path (one thread per hex cell, local 18x6 matrices,
// atomics; every backend) and the warp-owner path (warp_tile.hpp: a team per 8x8
// column tile, the geometry recomputed from the staged coordinates, the coarse
// pressure interpolated at the wedge centroid, the result scattered through
// register carry and warp shuffles; CUDA/HIP). Both evaluate the same bilinear form
// with the one-point rule, so they agree to roundoff.
template < typename ScalarT, typename CoordScalarT = ScalarT, typename ArithScalarT = ScalarT >
class Gradient
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Vec< ScalarT, 3 >;
    using ScalarType    = ScalarT;
    using GeomT         = CoordScalarT;
    using ArithT        = ArithScalarT;
    /// Jacobian arithmetic: the geometry precision widened to at least float.
    using GeoT = std::conditional_t< ( sizeof( GeomT ) >= 4 ), GeomT, float >;
    /// Element type of the staged coarse-pressure tile.
    using StageT = ArithT;
    /// Accumulation and contraction: never 16-bit (see EpsilonDivDivKerngen::AccT).
    using AccT = std::conditional_t< ( sizeof( ArithT ) >= 4 ), ArithT, float >;
    using ConT = std::conditional_t< ( sizeof( ArithT ) >= 4 ), ArithT, GeoT >;
    using Team = warp_tile::Team;

    static constexpr int warp_min_blocks = TERRA_GRADDIV_MIN_BLOCKS;

  private:
    grid::shell::DistributedDomain domain_fine_;
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid3DDataVec< CoordScalarT, 3 >                        grid_fine_;
    grid::Grid2DDataScalar< CoordScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_fine_;
    BoundaryConditions                                       bcs_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, 3 > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, 3 > recv_buffers_;
    communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataVec< ScalarType, 3 > > comm_plan_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataVec< ScalarType, 3 > dst_;

    // Warp path.
    warp_tile::Tiling tiling_;
    bool              fast_enabled_ = true;
    int               cmb_mode_     = 0; ///< 0 none/Neumann, 1 Dirichlet, 2 free-slip
    int               surf_mode_    = 0;

    static int bc_mode( const BoundaryConditionFlag f )
    {
        return ( f == DIRICHLET ) ? 1 : ( f == FREESLIP ) ? 2 : 0;
    }

  public:
    Gradient(
        const grid::shell::DistributedDomain&                           domain_fine,
        const grid::shell::DistributedDomain&                           domain_coarse,
        const grid::Grid3DDataVec< CoordScalarT, 3 >&                        grid_fine,
        const grid::Grid2DDataScalar< CoordScalarT >&                        radii_fine,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_fine,
        BoundaryConditions                                              bcs,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_fine_( domain_fine )
    , domain_coarse_( domain_coarse )
    , grid_fine_( grid_fine )
    , radii_( radii_fine )
    , boundary_mask_fine_( boundary_mask_fine )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain_fine )
    , recv_buffers_( domain_fine )
    , comm_plan_( domain_fine )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];

        cmb_mode_  = bc_mode( get_boundary_condition_flag( bcs_, CMB ) );
        surf_mode_ = bc_mode( get_boundary_condition_flag( bcs_, SURFACE ) );

        // The run length the staged tile allows under the launch bounds.
        int r_block = warp_tile::default_r_block( domain_fine );
        if ( r_block == 16 && shmem_bytes< 17 >() > warp_tile::kokkos_shmem_limit( 64, warp_min_blocks ) )
            r_block = 8;
        tiling_ = warp_tile::Tiling( domain_fine, r_block );
        if ( const char* env = std::getenv( "TERRA_GRADDIV_FAST" ) )
            fast_enabled_ = ( std::atoi( env ) != 0 );
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    /// Switch between the warp path and the per-cell path (for tests and measurements).
    void set_fast_path( const bool on ) { fast_enabled_ = on; }

    /// Whether the next apply runs the warp-owner kernel.
    bool fast_path_runs() const
    {
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
            return false;
        else
            return fast_enabled_ && tiling_.usable();
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "gradient_apply" );

        const bool fast = fast_path_runs();

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            if ( fast )
                warp_tile::zero_perimeter( dst.grid_data(), tiling_ );
            else
                assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        util::Timer timer_kernel( "gradient_kernel" );
        if ( fast )
        {
            if ( tiling_.r_block == 16 )
                launch_warp< 17 >();
            else
                launch_warp< 9 >();
        }
        else
        {
            Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_fine_ ), *this );
        }
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "gradient_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    // ------------------------------------------------------------------ warp path

    /// Team scratch of the warp path for NlevC node levels per run: the geometry of
    /// the tile, the coarse pressure of the tile and the inter-warp seam.
    template < int NlevC >
    static constexpr size_t shmem_bytes()
    {
        constexpr int nslot = NlevC + 1;
        constexpr int ncs   = NlevC / 2 + 2;
        return warp_tile::shmem_block_bytes( sizeof( GeomT ) * ( size_t( warp_tile::nxy ) * 3 + nslot ) ) +
               warp_tile::shmem_block_bytes( sizeof( StageT ) * size_t( 25 ) * ncs ) +
               warp_tile::shmem_block_bytes( sizeof( AccT ) * warp_tile::seam_scalars< 3 >() );
    }

    /// Public only because nvcc requires the function enclosing an extended lambda
    /// to be; not part of the interface.
    template < int NlevC >
    void launch_warp()
    {
        Kokkos::TeamPolicy< Kokkos::LaunchBounds< 64, warp_min_blocks > > policy( tiling_.blocks, warp_tile::team_size );
        policy.set_scratch_size( 0, Kokkos::PerTeam( shmem_bytes< NlevC >() ) );
        Kokkos::parallel_for(
            "gradient_warp", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                this->template run_team_warp< NlevC >( team );
            } );
    }

    template < int NlevC >
    KOKKOS_INLINE_FUNCTION void run_team_warp( const Team& team ) const
    {
        namespace wt = warp_tile;
        constexpr int nlev  = NlevC;     // node levels of the run (cells + 1)
        constexpr int nslot = NlevC + 1; // staged: the run plus the level below it (the halo cell)
        constexpr int ncs   = NlevC / 2 + 2; // coarse slots: coarse level R0 - 1 + cs
        constexpr int ncxy  = 25;

        int sub, x0, y0, r0;
        tiling_.decode( team.league_rank(), sub, x0, y0, r0 );
        const int hex_lat = tiling_.hex_lat, hex_rad = tiling_.hex_rad;
        const int rank = team.team_rank();
        const int tx = rank % wt::lat, ty = rank / wt::lat;
        const int x_cell = x0 + tx, y_cell = y0 + ty;
        const int lane = rank % 32, wy = rank / 32, ltx = lane % wt::PW, lty = lane / wt::PW;

        auto* geom_mem = reinterpret_cast< GeomT* >( team.team_shmem().get_shmem(
            wt::shmem_block_bytes( sizeof( GeomT ) * ( size_t( wt::nxy ) * 3 + nslot ) ) ) );
        auto* p_mem    = reinterpret_cast< StageT* >(
            team.team_shmem().get_shmem( wt::shmem_block_bytes( sizeof( StageT ) * size_t( ncxy ) * ncs ) ) );
        auto* seam_mem = reinterpret_cast< AccT* >(
            team.team_shmem().get_shmem( wt::shmem_block_bytes( sizeof( AccT ) * wt::seam_scalars< 3 >() ) ) );

        using Scratch       = typename Team::scratch_memory_space;
        using ScratchCoords = Kokkos::View< GeomT[wt::nxy][3], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        using ScratchR      = Kokkos::View< GeomT[nslot], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        using ScratchP      = Kokkos::View< StageT[ncxy][ncs], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        ScratchCoords coords_sh( geom_mem );
        ScratchR      r_sh( geom_mem + size_t( wt::nxy ) * 3 );
        ScratchP      p_sh( p_mem );

        wt::stage_geometry< nslot, GeomT >( team, coords_sh, r_sh, grid_fine_, radii_, sub, x0, y0, r0, hex_lat, hex_rad );

        // The coarse pressure columns under the tile: 5 x 5, coarse levels R0 - 1 .. .
        const int X0 = x0 / 2, Y0 = y0 / 2, R0 = r0 / 2;
        const int chex_lat = hex_lat / 2, chex_rad = hex_rad / 2;
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, ncxy * ncs ), [&]( int t ) {
            const int node = t / ncs;
            const int cs   = t - node * ncs;
            const int X    = X0 + node % 5;
            const int Y    = Y0 + node / 5;
            const int R    = R0 - 1 + cs;
            p_sh( node, cs ) = ( X <= chex_lat && Y <= chex_lat && R >= 0 && R <= chex_rad )
                                   ? static_cast< StageT >( src_( sub, X, Y, R ) )
                                   : StageT( 0 );
        } );
        team.team_barrier();

        auto node_id = []( int nx, int ny ) -> int { return nx + ( wt::lat + 1 ) * ny; };
        const int corner_node[4] = { node_id( tx, ty ), node_id( tx + 1, ty ), node_id( tx, ty + 1 ), node_id( tx + 1, ty + 1 ) };

        GeoT cx[4], cy[4], cz[4];
#pragma unroll
        for ( int c = 0; c < 4; ++c )
        {
            cx[c] = static_cast< GeoT >( coords_sh( corner_node[c], 0 ) );
            cy[c] = static_cast< GeoT >( coords_sh( corner_node[c], 1 ) );
            cz[c] = static_cast< GeoT >( coords_sh( corner_node[c], 2 ) );
        }

        // The fine column's place in its coarse cell, and the coarse interpolation
        // weights of both wedges (constant along the column).
        const int px = tx & 1, py = ty & 1;
        const int CX = tx >> 1, CY = ty >> 1;
        GeoT lw[2][3];
        int  cnode[2][3]; // coarse node columns (in p_sh) of the three lateral coarse nodes
#pragma unroll
        for ( int w = 0; w < 2; ++w )
        {
            const int idx = wt::fine_sub_triangle( w, px, py );
            const int cw  = wt::coarse_wedge_of( w, px, py );
#pragma unroll
            for ( int c = 0; c < 3; ++c )
            {
                lw[w][c]    = wt::coarse_lateral_weight< GeoT >( idx, c );
                cnode[w][c] = ( CX + wt::wedge_node_off( cw, c, 0 ) ) + 5 * ( CY + wt::wedge_node_off( cw, c, 1 ) );
            }
        }

        const int s_begin = ( r0 > 0 ) ? 0 : 1;

        AccT acc[4][3], up[4][3];
#pragma unroll
        for ( int c = 0; c < 4; ++c )
#pragma unroll
            for ( int d = 0; d < 3; ++d )
                acc[c][d] = AccT( 0 );

        const bool replace_mode = ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace );
        const bool interior     = ( tx > 0 ) && ( ty > 0 );

        int  s_flush = 0;
        auto emit    = [&]( const int ddx, const int ddy, const int d, const AccT val, const bool owned ) {
            const int   rr = r0 - 1 + s_flush;
            ScalarType* p  = &dst_( sub, x_cell + ddx, y_cell + ddy, rr, d );
            if ( owned && interior )
                *p = static_cast< ScalarType >( replace_mode ? val : static_cast< AccT >( *p ) + val );
            else if ( val != AccT( 0 ) )
                wt::global_add( p, val );
        };

        int s_top = s_begin;
        for ( int s = s_begin; s < nlev; ++s )
        {
            const int r_cell = r0 - 1 + s;
            if ( r_cell >= hex_rad ) // uniform over the team
                break;
            s_top = s + 1;

#pragma unroll
            for ( int cc = 0; cc < 4; ++cc )
#pragma unroll
                for ( int d = 0; d < 3; ++d )
                    up[cc][d] = AccT( 0 );

            const GeoT r_0 = static_cast< GeoT >( r_sh( s ) );
            const GeoT r_1 = static_cast< GeoT >( r_sh( s + 1 ) );

            const int  R  = r_cell >> 1, pr = r_cell & 1;
            const int  cs = R - R0 + 1;
            const GeoT wb = wt::coarse_radial_weight_bottom< GeoT >( pr );
            const GeoT wu = static_cast< GeoT >( 1 ) - wb;

            const bool at_cmb     = util::has_flag( boundary_mask_fine_( sub, x_cell, y_cell, r_cell ), CMB );
            const bool at_surface = util::has_flag( boundary_mask_fine_( sub, x_cell, y_cell, r_cell + 1 ), SURFACE );
            const int  lo_mode    = at_cmb ? cmb_mode_ : 0;
            const int  hi_mode    = at_surface ? surf_mode_ : 0;

#pragma unroll
            for ( int w = 0; w < 2; ++w )
            {
                const wt::WedgeGeometry< GeoT > g( cx, cy, cz, wt::wedge_vertex( w, 0 ), wt::wedge_vertex( w, 1 ), wt::wedge_vertex( w, 2 ), r_0, r_1 );

                // Coarse pressure at the wedge centroid.
                GeoT pq = 0;
#pragma unroll
                for ( int c = 0; c < 3; ++c )
                    pq += lw[w][c] * ( wb * static_cast< GeoT >( p_sh( cnode[w][c], cs ) ) +
                                       wu * static_cast< GeoT >( p_sh( cnode[w][c], cs + 1 ) ) );
                const ConT f = static_cast< ConT >( -g.abs_det * pq );

#pragma unroll
                for ( int n = 0; n < 6; ++n )
                {
                    const int mode = ( n < 3 ) ? lo_mode : hi_mode;
                    if ( mode == 1 )
                        continue;
                    ConT v0 = f * static_cast< ConT >( g.a0[n] );
                    ConT v1 = f * static_cast< ConT >( g.a1[n] );
                    ConT v2 = f * static_cast< ConT >( g.a2[n] );
                    const int cc = wt::wedge_node_off( w, n, 0 ) + 2 * wt::wedge_node_off( w, n, 1 );
                    if ( mode == 2 )
                    {
                        // Free-slip: only the tangential part of the row survives.
                        const GeoT nn  = cx[cc] * cx[cc] + cy[cc] * cy[cc] + cz[cc] * cz[cc];
                        const ConT dp  = static_cast< ConT >( ( v0 * cx[cc] + v1 * cy[cc] + v2 * cz[cc] ) / nn );
                        v0 -= dp * static_cast< ConT >( cx[cc] );
                        v1 -= dp * static_cast< ConT >( cy[cc] );
                        v2 -= dp * static_cast< ConT >( cz[cc] );
                    }
                    if ( wt::wedge_node_off( w, n, 2 ) )
                    {
                        up[cc][0] += static_cast< AccT >( v0 );
                        up[cc][1] += static_cast< AccT >( v1 );
                        up[cc][2] += static_cast< AccT >( v2 );
                    }
                    else
                    {
                        acc[cc][0] += static_cast< AccT >( v0 );
                        acc[cc][1] += static_cast< AccT >( v1 );
                        acc[cc][2] += static_cast< AccT >( v2 );
                    }
                }
            }

            // The level under this cell is complete for this team -- except slot 0,
            // the level under the halo cell, which belongs to the team below.
            if ( s >= 1 )
            {
                s_flush = s;
                wt::flush_level< 3 >( team, acc, ltx, lty, wy, seam_mem, s & 1, true, emit );
            }

#pragma unroll
            for ( int cc = 0; cc < 4; ++cc )
#pragma unroll
                for ( int d = 0; d < 3; ++d )
                    acc[cc][d] = up[cc][d];
        }

        // The level on top of the run belongs to the team above -- unless it is the
        // surface, which no team is above.
        if ( r0 - 1 + s_top == hex_rad )
        {
            s_flush = s_top;
            wt::flush_level< 3 >( team, acc, ltx, lty, wy, seam_mem, s_top & 1, true, emit );
        }
    }

    // ------------------------------------------------------------------ per-cell path

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        // Geometry and quadrature in GeomT (the stored coordinate precision).
        dense::Vec< GeomT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_fine_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const GeomT r_1 = radii_( local_subdomain_id, r_cell );
        const GeomT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // Quadrature points.
        constexpr auto num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;

        dense::Vec< GeomT, 3 > quad_points[num_quad_points];
        GeomT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );

        const int fine_radial_wedge_index = r_cell % 2;

        // Compute the local element matrix.
        dense::Mat< ScalarT, 18, 6 > A[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; q++ )
        {
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const int fine_lateral_wedge_index = fine_lateral_wedge_idx( x_cell, y_cell, wedge );

                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto grad_i = grad_shape( i, quad_points[q] );

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto shape_j =
                            shape_coarse( j, fine_radial_wedge_index, fine_lateral_wedge_index, quad_points[q] );

                        for ( int d = 0; d < 3; d++ )
                        {
                            A[wedge]( d * 6 + i, j ) +=
                                quad_weights[q] * ( -( ( J_inv_transposed * grad_i )(d) *shape_j ) * det );
                        }
                    }
                }
            }
        }

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell / 2, y_cell / 2, r_cell / 2, src_ );

        // Boundary treatment
        bool at_cmb = util::has_flag( boundary_mask_fine_( local_subdomain_id, x_cell, y_cell, r_cell ), CMB );
        bool at_surface =
            util::has_flag( boundary_mask_fine_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), SURFACE );

        dense::Mat< ScalarT, 18, 6 > boundary_mask;
        boundary_mask.fill( 1.0 );

        dense::Mat< ScalarT, 18, 18 > R[num_wedges_per_hex_cell];
        // flag to later not go through the hustle of checking the bcs
        bool freeslip_reorder = false;

        if ( at_cmb || at_surface )
        {
            // Inner boundary (CMB).
            ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
            BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );

            if ( bcf == DIRICHLET )
            {
                for ( int dimi = 0; dimi < 3; ++dimi )
                {
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            if ( at_cmb && ( i < 3 ) || at_surface && ( i >= 3 ) )
                            {
                                boundary_mask( dimi * num_nodes_per_wedge + i, j ) = 0.0;
                            }
                        }
                    }
                }
            }
            else if ( bcf == FREESLIP )
            {

                freeslip_reorder                                            = true;
                dense::Mat< ScalarT, 18, 6 > A_tmp[num_wedges_per_hex_cell] = { 0 };

                // reorder source dofs for nodes instead of velocity dims in src vector and local matrix
                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    for ( int node_idxi = 0; node_idxi < num_nodes_per_wedge; node_idxi++ )
                    {
                        for ( int dimi = 0; dimi < 3; ++dimi )
                        {
                            for ( int node_idxj = 0; node_idxj < num_nodes_per_wedge; node_idxj++ )
                            {
                                A_tmp[wedge]( node_idxi * 3 + dimi, node_idxj ) =
                                    A[wedge]( node_idxi + dimi * num_nodes_per_wedge, node_idxj );
                            }
                        }
                    }
                }

                // assemble rotation matrices for boundary nodes
                // e.g. if we are at CMB, we need to rotate DoFs 0, 1, 2 of each wedge
                // at SURFACE, we need to rotate DoFs 3, 4, 5

                constexpr int layer_hex_offset_x[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
                constexpr int layer_hex_offset_y[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };

                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    // make rotation matrix unity
                    for ( int i = 0; i < 18; ++i )
                    {
                        R[wedge]( i, i ) = 1.0;
                    }

                    for ( int boundary_node_idx = 0; boundary_node_idx < 3; boundary_node_idx++ )
                    {
                        // compute normal
                        const auto normal = grid::shell::coords(
                            local_subdomain_id,
                            x_cell + layer_hex_offset_x[wedge][boundary_node_idx],
                            y_cell + layer_hex_offset_y[wedge][boundary_node_idx],
                            r_cell + ( at_cmb ? 0 : 1 ),
                            grid_fine_,
                            radii_ );


                        // compute rotation matrix for DoFs on current node
                        auto R_i = trafo_mat_cartesian_to_normal_tangential( normal );

                        // insert into wedge-local rotation matrix
                        int offset_in_R = at_cmb ? 0 : 9;
                        for ( int dimi = 0; dimi < 3; ++dimi )
                        {
                            for ( int dimj = 0; dimj < 3; ++dimj )
                            {
                                R[wedge](
                                    offset_in_R + boundary_node_idx * 3 + dimi,
                                    offset_in_R + boundary_node_idx * 3 + dimj ) = R_i( dimi, dimj );
                            }
                        }
                    }

                    // transform local matrix to rotated/ normal-tangential space: pre/post multiply with rotation matrices
                    // TODO transpose this way?
                    A[wedge] = R[wedge] * A_tmp[wedge];

                    // eliminate normal components: Dirichlet on the normal-tangential system
                    int node_start = at_surface ? 3 : 0;
                    int node_end   = at_surface ? 6 : 3;

                    for ( int node_idx = node_start; node_idx < node_end; node_idx++ )
                    {
                        int idx = node_idx * 3;
                        for ( int k = 0; k < 6; ++k )
                        {
                            boundary_mask( idx, k ) = 0.0;
                        }
                    }
                }

            }
            else if ( bcf == NEUMANN ) {}
        }

        // apply boundary mask
        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            A[wedge].hadamard_product( boundary_mask );
        }

        dense::Vec< ScalarT, 18 > dst[num_wedges_per_hex_cell];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        if ( freeslip_reorder )
        {

            // transform dst back from nt space
            dense::Vec< ScalarT, 18 > dst_tmp[num_wedges_per_hex_cell];
            dst_tmp[0] = R[0].transposed() * dst[0];
            dst_tmp[1] = R[1].transposed() * dst[1];
            for ( int i = 0; i < 18; ++i )
            {
                dst[0]( i ) = dst_tmp[0]( i );
                dst[1]( i ) = dst_tmp[1]( i );
            }

            // reorder to dimensionwise ordering
            reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst[0] );
            reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst[1] );

        }

        for ( int d = 0; d < 3; d++ )
        {
            dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
            dst_d[0] = dst[0].template slice< 6 >( d * 6 );
            dst_d[1] = dst[1].template slice< 6 >( d * 6 );

            atomically_add_local_wedge_vector_coefficients(
                dst_, local_subdomain_id, x_cell, y_cell, r_cell, d, dst_d );
        }
    }
};

static_assert( linalg::OperatorLike< Gradient< double > > );

} // namespace terra::fe::wedge::operators::shell
