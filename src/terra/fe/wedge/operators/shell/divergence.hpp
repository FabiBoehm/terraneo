#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/deterministic.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
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

#ifndef TERRA_GRADDIV_MIN_BLOCKS
#define TERRA_GRADDIV_MIN_BLOCKS 8
#endif

// CoordScalarT is the precision of the STORED node coordinates and radii. It
// defaults to ScalarT, so existing instantiations are unchanged. Pass double to
// decouple it: the Jacobian is formed by differencing coordinates, which loses the
// geometry entirely in 16-bit and is already marginal in single. ScalarT governs
// the stored fields src/dst and the coefficient -- where the memory traffic is.
// ArithScalarT is the precision of the warp path's staged velocity tile and of its
// per-cell contraction; it defaults to ScalarT.
//
// Two paths, as in Gradient (gradient.hpp). The warp path's output is the coarse
// pressure grid: a fine cell's two wedges contribute to the 8 corners of the coarse
// cell around it, so the owner reduction first sums the 2 x 2 fine columns of each
// coarse cell with warp shuffles and then runs the corner reduction on the coarse
// columns. A coarse level is complete after the two fine cells above it; the run
// has no halo cell, so the levels at both ends of a run are reduced into global
// memory (every run end is a coarse level, which Replace mode zeroes).
template < typename ScalarT, typename CoordScalarT = ScalarT, typename ArithScalarT = ScalarT >
class Divergence
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, 3 >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;
    using GeomT         = CoordScalarT;
    using ArithT        = ArithScalarT;
    using GeoT          = std::conditional_t< ( sizeof( GeomT ) >= 4 ), GeomT, float >;
    using StageT        = ArithT;
    using AccT          = std::conditional_t< ( sizeof( ArithT ) >= 4 ), ArithT, float >;
    using ConT          = std::conditional_t< ( sizeof( ArithT ) >= 4 ), ArithT, GeoT >;
    using Team          = warp_tile::Team;

    static constexpr int warp_min_blocks = TERRA_GRADDIV_MIN_BLOCKS;

  private:
    grid::shell::DistributedDomain domain_fine_;
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid3DDataVec< CoordScalarT, 3 >                        grid_fine_;
    grid::Grid2DDataScalar< CoordScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_fine_;

    BoundaryConditions bcs_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;
    communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataScalar< ScalarType > > comm_plan_;

    grid::Grid4DDataVec< ScalarType, 3 > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    // Warp path.
    warp_tile::Tiling tiling_;
    bool              fast_enabled_ = true;
    int               min_blocks_   = warp_min_blocks;
    /// Two-phase deterministic scatter (warp_tile.hpp): TERRA_DETERMINISTIC_SCATTER=1.
    bool                               det_ = warp_tile::deterministic_scatter_requested();
    warp_tile::PerimeterBuffer< AccT > pb_;
    int               cmb_mode_     = 0; ///< 0 none/Neumann, 1 Dirichlet, 2 free-slip
    int               surf_mode_    = 0;

    static int bc_mode( const BoundaryConditionFlag f )
    {
        return ( f == DIRICHLET ) ? 1 : ( f == FREESLIP ) ? 2 : 0;
    }

  public:
    Divergence(
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
    , send_buffers_( domain_coarse )
    , recv_buffers_( domain_coarse )
    , comm_plan_( domain_coarse )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];

        cmb_mode_  = bc_mode( get_boundary_condition_flag( bcs_, CMB ) );
        surf_mode_ = bc_mode( get_boundary_condition_flag( bcs_, SURFACE ) );

        // Occupancy target: 8 or 16 resident teams per SM (128 or 64 registers), a
        // measurement switch via the environment.
        if ( const char* env = std::getenv( "TERRA_GRADDIV_MIN_BLOCKS" ) )
            min_blocks_ = ( std::atoi( env ) >= 16 ) ? 16 : 8;
        int r_block = warp_tile::default_r_block( domain_fine );
        if ( r_block == 16 && shmem_bytes< 17 >() > warp_tile::kokkos_shmem_limit( 64, min_blocks_ ) )
            r_block = 8;
        // A tile whose scratch does not fit the higher occupancy target runs at the default one.
        if ( min_blocks_ > warp_min_blocks && shmem_bytes< 9 >() > warp_tile::kokkos_shmem_limit( 64, min_blocks_ ) )
        {
            min_blocks_ = warp_min_blocks;
            r_block     = warp_tile::default_r_block( domain_fine );
            if ( r_block == 16 && shmem_bytes< 17 >() > warp_tile::kokkos_shmem_limit( 64, min_blocks_ ) )
                r_block = 8;
        }
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

    void set_fast_path( const bool on ) { fast_enabled_ = on; }
    void set_deterministic_scatter( const bool on ) { det_ = on; }
    bool deterministic_scatter_runs() const { return det_ && fast_path_runs(); }

    /// Whether the next apply runs the warp-owner kernel. The coarse owner logic
    /// needs whole coarse cells per run (an even run length) and a lateral tile of
    /// whole coarse cells, which the 8-wide tile is.
    bool fast_path_runs() const
    {
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
            return false;
        else
            return fast_enabled_ && tiling_.usable() && ( tiling_.hex_rad % 2 == 0 );
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "divergence_apply" );

        const bool fast = fast_path_runs();

        // The coarse grid has an eighth of the fine nodes: zeroing all of it costs
        // less than the warp path's own reads, so Replace mode just does that.
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        // The two-phase scatter on the coarse grid: coarse tiles of lat / 2 cells, and
        // since the runs are not level-complete (no halo cell), two edge halves and
        // the slab of the team below's owner values at every run-boundary level.
        const bool det = fast && det_;
        if ( det && !pb_.matches( warp_tile::lat / 2, tiling_.nsub, tiling_.hex_lat / 2, tiling_.hex_rad / 2, 1 ) )
            pb_ = warp_tile::PerimeterBuffer< AccT >(
                warp_tile::lat / 2, tiling_.nsub, tiling_.hex_lat / 2, tiling_.hex_rad / 2, 1, 2, tiling_.r_block / 2 );

        util::Timer timer_kernel( "divergence_kernel" );
        if ( fast )
        {
            if ( min_blocks_ >= 16 )
                tiling_.r_block == 16 ? launch_warp< 17, 16 >() : launch_warp< 9, 16 >();
            else
                tiling_.r_block == 16 ? launch_warp< 17, 8 >() : launch_warp< 9, 8 >();
            if ( det )
            {
                // dst was zeroed (Replace) or holds the previous value (Add): always the base.
                warp_tile::slab_reduce( dst_, pb_ );
                warp_tile::perimeter_reduce( dst_, pb_, false );
            }
        }
        else
        {
            det::for_each_colour( det_, [&]( const int c ) {
            colour_ = c;
            Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_fine_ ), *this );
        } );
        }
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "divergence_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    // ------------------------------------------------------------------ warp path

    /// Team scratch for NlevC node levels per run: geometry, the staged velocity of
    /// the run's levels and the inter-warp seam (one coarse scalar, two corners).
    template < int NlevC >
    static constexpr size_t shmem_bytes()
    {
        return warp_tile::shmem_block_bytes( sizeof( GeomT ) * ( size_t( warp_tile::nxy ) * 3 + NlevC ) ) +
               warp_tile::shmem_block_bytes( sizeof( StageT ) * size_t( warp_tile::nxy ) * 3 * NlevC ) +
               warp_tile::shmem_block_bytes( sizeof( AccT ) * warp_tile::seam_scalars< 1 >() );
    }

    template < int NlevC, int MinBlocks >
    void launch_warp()
    {
        Kokkos::TeamPolicy< Kokkos::LaunchBounds< 64, MinBlocks > > policy( tiling_.blocks, warp_tile::team_size );
        policy.set_scratch_size( 0, Kokkos::PerTeam( shmem_bytes< NlevC >() ) );
        Kokkos::parallel_for(
            "divergence_warp", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                this->template run_team_warp< NlevC >( team );
            } );
    }

    template < int NlevC >
    KOKKOS_INLINE_FUNCTION void run_team_warp( const Team& team ) const
    {
        namespace wt = warp_tile;
        constexpr int nlev = NlevC; // node levels of the run (cells + 1), no halo

        int sub, x0, y0, r0;
        tiling_.decode( team.league_rank(), sub, x0, y0, r0 );
        const int hex_lat = tiling_.hex_lat, hex_rad = tiling_.hex_rad;
        const int rank = team.team_rank();
        const int tx = rank % wt::lat, ty = rank / wt::lat;
        const int x_cell = x0 + tx, y_cell = y0 + ty;
        const int lane = rank % 32, wy = rank / 32, ltx = lane % wt::PW, lty = lane / wt::PW;

        auto* geom_mem = reinterpret_cast< GeomT* >( team.team_shmem().get_shmem(
            wt::shmem_block_bytes( sizeof( GeomT ) * ( size_t( wt::nxy ) * 3 + nlev ) ) ) );
        auto* field_mem = reinterpret_cast< StageT* >( team.team_shmem().get_shmem(
            wt::shmem_block_bytes( sizeof( StageT ) * size_t( wt::nxy ) * 3 * nlev ) ) );
        auto* seam_mem = reinterpret_cast< AccT* >(
            team.team_shmem().get_shmem( wt::shmem_block_bytes( sizeof( AccT ) * wt::seam_scalars< 1 >() ) ) );

        using Scratch       = typename Team::scratch_memory_space;
        using ScratchCoords = Kokkos::View< GeomT[wt::nxy][3], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        using ScratchR      = Kokkos::View< GeomT[nlev], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        using ScratchSrc = Kokkos::View< StageT[wt::nxy][3][nlev], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        ScratchCoords coords_sh( geom_mem );
        ScratchR      r_sh( geom_mem + size_t( wt::nxy ) * 3 );
        ScratchSrc    src_sh( field_mem );

        // Slot s holds level r0 + s (stage_geometry's slot 0 is r0 - 1, hence r0 + 1).
        wt::stage_geometry< nlev, GeomT >( team, coords_sh, r_sh, grid_fine_, radii_, sub, x0, y0, r0 + 1, hex_lat, hex_rad );

        constexpr bool AsyncStaging =
            std::is_same_v< StageT, ScalarType > && ( sizeof( ScalarType ) == 4 || sizeof( ScalarType ) == 8 );
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, wt::nxy * nlev ), [&]( int t ) {
            const int node = t / nlev;
            const int lvl  = t - node * nlev;
            const int xi   = x0 + node % ( wt::lat + 1 );
            const int yi   = y0 + node / ( wt::lat + 1 );
            const int rr   = r0 + lvl;
            if ( xi <= hex_lat && yi <= hex_lat && rr <= hex_rad )
            {
                if constexpr ( AsyncStaging )
                {
                    util::cp_async( &src_sh( node, 0, lvl ), &src_( sub, xi, yi, rr, 0 ) );
                    util::cp_async( &src_sh( node, 1, lvl ), &src_( sub, xi, yi, rr, 1 ) );
                    util::cp_async( &src_sh( node, 2, lvl ), &src_( sub, xi, yi, rr, 2 ) );
                }
                else
                {
                    src_sh( node, 0, lvl ) = static_cast< StageT >( src_( sub, xi, yi, rr, 0 ) );
                    src_sh( node, 1, lvl ) = static_cast< StageT >( src_( sub, xi, yi, rr, 1 ) );
                    src_sh( node, 2, lvl ) = static_cast< StageT >( src_( sub, xi, yi, rr, 2 ) );
                }
            }
            else
            {
                src_sh( node, 0, lvl ) = src_sh( node, 1, lvl ) = src_sh( node, 2, lvl ) = StageT( 0 );
            }
        } );
        if constexpr ( AsyncStaging )
            util::cp_async_wait_all();
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

        // The fine column's place in its coarse cell; for each wedge the coarse corner
        // (of the coarse hex cell) and lateral weight of its three coarse nodes.
        const int px = tx & 1, py = ty & 1;
        const int CX = tx >> 1, CY = ty >> 1;
        GeoT lw[2][3];
        int  ccorner[2][3];
#pragma unroll
        for ( int w = 0; w < 2; ++w )
        {
            const int idx = wt::fine_sub_triangle( w, px, py );
            const int cw  = wt::coarse_wedge_of( w, px, py );
#pragma unroll
            for ( int c = 0; c < 3; ++c )
            {
                lw[w][c]      = wt::coarse_lateral_weight< GeoT >( idx, c );
                ccorner[w][c] = wt::wedge_node_off( cw, c, 0 ) + 2 * wt::wedge_node_off( cw, c, 1 );
            }
        }

        // Field values of the 4 columns at the cell's lower and upper level.
        StageT sl[4][3], su[4][3];
#pragma unroll
        for ( int c = 0; c < 4; ++c )
#pragma unroll
            for ( int d = 0; d < 3; ++d )
                sl[c][d] = src_sh( corner_node[c], d, 0 );

        // Coarse corner contributions to the coarse level under the current coarse
        // cell (acc) and above it (up).
        AccT acc[4], up[4];
#pragma unroll
        for ( int c = 0; c < 4; ++c )
            acc[c] = up[c] = AccT( 0 );

        const bool replace_mode = ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace );
        const int  X0 = x0 / 2, Y0 = y0 / 2, R0 = r0 / 2, chex_rad = hex_rad / 2;
        const bool coarse_lane = ( px == 0 ) && ( py == 0 ); // the lane that owns coarse column (CX, CY)
        const int  clty        = lty >> 1;                  // coarse row inside the warp patch

        // One coarse level: sum the 2 x 2 fine columns of every coarse cell, reduce the
        // 4 coarse corners to the owner lane, write once.
        auto flush_coarse = [&]( AccT ( &v )[4], const int R, const bool complete ) {
            AccT t[4], u[4];
#pragma unroll
            for ( int c = 0; c < 4; ++c )
                t[c] = v[c] + util::warp_shfl_down( v[c], 1 );
#pragma unroll
            for ( int c = 0; c < 4; ++c )
                u[c] = t[c] + util::warp_shfl_down( t[c], wt::PW );

            const AccT from_left  = util::warp_shfl_up( u[1], 2 );
            const AccT from_below = util::warp_shfl_up( u[2], 2 * wt::PW );
            const AccT from_diag  = util::warp_shfl_up( u[3], 2 * wt::PW + 2 );
            AccT       own        = u[0];
            own += ( CX > 0 ) ? from_left : AccT( 0 );
            own += ( clty > 0 ) ? from_below : AccT( 0 );
            own += ( CX > 0 && clty > 0 ) ? from_diag : AccT( 0 );

            // Seam: the top coarse row of warp w hands its from-below corners to the
            // bottom coarse row of warp w + 1.
            const int par = R & 1;
            auto      seam = [&]( int p, int w, int col, int corner ) -> AccT& {
                return seam_mem[( ( p * ( wt::NW - 1 ) + w ) * wt::PW + col ) * 2 + corner];
            };
            if constexpr ( wt::NW > 1 )
            {
                if ( coarse_lane && clty == wt::PH / 2 - 1 && wy < wt::NW - 1 )
                {
                    seam( par, wy, CX, 0 ) = u[2];
                    seam( par, wy, CX, 1 ) = u[3];
                }
                team.team_barrier();
                if ( coarse_lane && clty == 0 && wy > 0 )
                {
                    own += seam( par, wy - 1, CX, 0 );
                    own += ( CX > 0 ) ? seam( par, wy - 1, CX - 1, 1 ) : AccT( 0 );
                }
            }

            if ( coarse_lane )
            {
                auto emit = [&]( const int ddx, const int ddy, const AccT val, const bool owned ) {
                    ScalarType* p = &dst_( sub, X0 + CX + ddx, Y0 + CY + ddy, R );
                    if ( det_ )
                    {
                        // Two-phase scatter. The team whose run contains the level
                        // owns it (its bottom level included); the top flush of a
                        // run another team continues is "below": its owner value
                        // goes to the slab, its edges to the second half.
                        const bool below = !complete && R != R0;
                        if ( owned && !below )
                            *p = static_cast< ScalarType >( replace_mode ? val : static_cast< AccT >( *p ) + val );
                        else if ( owned )
                            pb_.slab( sub, 0, X0 + CX, Y0 + CY, R ) = val;
                        else
                            pb_.put_edge( sub, below ? 1 : 0, ddx, ddy, 0, X0 + CX, Y0 + CY, R, val );
                    }
                    else if ( owned && complete && CX > 0 && CY > 0 )
                        *p = static_cast< ScalarType >( replace_mode ? val : static_cast< AccT >( *p ) + val );
                    else if ( val != AccT( 0 ) )
                        wt::global_add( p, val );
                };
                emit( 0, 0, own, true );
                if ( CX == wt::lat / 2 - 1 )
                {
                    emit( 1, 0, u[1], false );
                    emit( 1, 1, u[3], false );
                }
                if ( clty == wt::PH / 2 - 1 && wy == wt::NW - 1 )
                {
                    emit( 0, 1, u[2], false );
                    if ( CX != wt::lat / 2 - 1 )
                        emit( 1, 1, u[3], false );
                }
            }
        };

        int R_last = -1; // the last coarse cell completed in this run
        for ( int s = 0; s < nlev - 1; ++s )
        {
            const int r_cell = r0 + s;
            if ( r_cell >= hex_rad ) // uniform over the team
                break;
            const int R = r_cell >> 1, pr = r_cell & 1;

#pragma unroll
            for ( int cc = 0; cc < 4; ++cc )
#pragma unroll
                for ( int d = 0; d < 3; ++d )
                    su[cc][d] = src_sh( corner_node[cc], d, s + 1 );

            const GeoT r_0 = static_cast< GeoT >( r_sh( s ) );
            const GeoT r_1 = static_cast< GeoT >( r_sh( s + 1 ) );
            const GeoT wb  = wt::coarse_radial_weight_bottom< GeoT >( pr );
            const GeoT wu  = static_cast< GeoT >( 1 ) - wb;

            const bool at_cmb     = util::has_flag( boundary_mask_fine_( sub, x_cell, y_cell, r_cell ), CMB );
            const bool at_surface = util::has_flag( boundary_mask_fine_( sub, x_cell, y_cell, r_cell + 1 ), SURFACE );
            const int  lo_mode    = at_cmb ? cmb_mode_ : 0;
            const int  hi_mode    = at_surface ? surf_mode_ : 0;

#pragma unroll
            for ( int w = 0; w < 2; ++w )
            {
                const wt::WedgeGeometry< GeoT > g( cx, cy, cz, wt::wedge_vertex( w, 0 ), wt::wedge_vertex( w, 1 ), wt::wedge_vertex( w, 2 ), r_0, r_1 );

                ConT div_u = 0;
#pragma unroll
                for ( int n = 0; n < 6; ++n )
                {
                    const int mode = ( n < 3 ) ? lo_mode : hi_mode;
                    if ( mode == 1 )
                        continue;
                    const int cc = wt::wedge_node_off( w, n, 0 ) + 2 * wt::wedge_node_off( w, n, 1 );
                    ConT      u0 = static_cast< ConT >( wt::wedge_node_off( w, n, 2 ) ? su[cc][0] : sl[cc][0] );
                    ConT      u1 = static_cast< ConT >( wt::wedge_node_off( w, n, 2 ) ? su[cc][1] : sl[cc][1] );
                    ConT      u2 = static_cast< ConT >( wt::wedge_node_off( w, n, 2 ) ? su[cc][2] : sl[cc][2] );
                    if ( mode == 2 )
                    {
                        // Free-slip: only the tangential velocity enters.
                        const GeoT nn = cx[cc] * cx[cc] + cy[cc] * cy[cc] + cz[cc] * cz[cc];
                        const ConT dp = static_cast< ConT >( ( u0 * cx[cc] + u1 * cy[cc] + u2 * cz[cc] ) / nn );
                        u0 -= dp * static_cast< ConT >( cx[cc] );
                        u1 -= dp * static_cast< ConT >( cy[cc] );
                        u2 -= dp * static_cast< ConT >( cz[cc] );
                    }
                    div_u += static_cast< ConT >( g.a0[n] ) * u0 + static_cast< ConT >( g.a1[n] ) * u1 +
                             static_cast< ConT >( g.a2[n] ) * u2;
                }
                const ConT val = static_cast< ConT >( -g.abs_det ) * div_u;
#pragma unroll
                for ( int c = 0; c < 3; ++c )
                {
                    const ConT lv = val * static_cast< ConT >( lw[w][c] );
                    acc[ccorner[w][c]] += static_cast< AccT >( lv * static_cast< ConT >( wb ) );
                    up[ccorner[w][c]] += static_cast< AccT >( lv * static_cast< ConT >( wu ) );
                }
            }

            if ( pr == 1 )
            {
                // The coarse cell R is done: the coarse level under it got this coarse
                // cell's lower contributions and (unless the run starts here) the
                // previous one's upper.
                flush_coarse( acc, R, ( R > R0 ) || ( R == 0 ) );
                R_last = R;
#pragma unroll
                for ( int c = 0; c < 4; ++c )
                {
                    acc[c] = up[c];
                    up[c]  = AccT( 0 );
                }
            }

#pragma unroll
            for ( int cc = 0; cc < 4; ++cc )
#pragma unroll
                for ( int d = 0; d < 3; ++d )
                    sl[cc][d] = su[cc][d];
        }

        // The level on top of the run: complete only at the surface.
        if ( R_last >= 0 )
            flush_coarse( acc, R_last + 1, R_last + 1 == chex_rad );
    }

    // ------------------------------------------------------------------ per-cell path

    /// Deterministic colouring (fe/wedge/deterministic.hpp): the launch only scatters
    /// the cells of this parity colour, -1 all of them.
    int colour_ = -1;

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        if ( det::skip( colour_, x_cell, y_cell, r_cell ) )
            return;
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
        dense::Mat< ScalarT, 6, 18 > A[num_wedges_per_hex_cell] = {};

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
                    const auto shape_i =
                        shape_coarse( i, fine_radial_wedge_index, fine_lateral_wedge_index, quad_points[q] );

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto grad_j = grad_shape( j, quad_points[q] );

                        for ( int d = 0; d < 3; d++ )
                        {
                            A[wedge]( i, d * 6 + j ) +=
                                quad_weights[q] * ( -( J_inv_transposed * grad_j )(d) *shape_i * det );
                        }
                    }
                }
            }
        }

        bool at_cmb = util::has_flag( boundary_mask_fine_( local_subdomain_id, x_cell, y_cell, r_cell ), CMB );
        bool at_surface =
            util::has_flag( boundary_mask_fine_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), SURFACE );

        dense::Vec< ScalarT, 18 > src[num_wedges_per_hex_cell];
        for ( int d = 0; d < 3; d++ )
        {
            dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
            extract_local_wedge_vector_coefficients( src_d, local_subdomain_id, x_cell, y_cell, r_cell, d, src_ );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    src[wedge]( d * 6 + i ) = src_d[wedge]( i );
                }
            }
        }

        // Boundary treatment
        dense::Mat< ScalarT, 6, 18 > boundary_mask;
        boundary_mask.fill( 1.0 );

        dense::Mat< ScalarT, 18, 18 > R[num_wedges_per_hex_cell];

        if ( at_cmb || at_surface )
        {
            // Inner boundary (CMB).
            ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
            BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );

            if ( bcf == DIRICHLET )
            {
                for ( int dimj = 0; dimj < 3; ++dimj )
                {
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            if ( at_cmb && ( j < 3 ) || at_surface && ( j >= 3 ) )
                            {
                                boundary_mask( i, dimj * num_nodes_per_wedge + j ) = 0.0;
                            }
                        }
                    }
                }
            }
            else if ( bcf == FREESLIP )
            {

                dense::Mat< ScalarT, 6, 18 > A_tmp[num_wedges_per_hex_cell] = { 0 };

                // reorder source dofs for nodes instead of velocity dims in src vector and local matrix
                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    for ( int node_idxi = 0; node_idxi < num_nodes_per_wedge; node_idxi++ )
                    {
                        for ( int dimj = 0; dimj < 3; ++dimj )
                        {
                            for ( int node_idxj = 0; node_idxj < num_nodes_per_wedge; node_idxj++ )
                            {
                                A_tmp[wedge]( node_idxi, node_idxj * 3 + dimj ) =
                                    A[wedge]( node_idxi, node_idxj + dimj * num_nodes_per_wedge );
                            }
                        }
                    }
                    reorder_local_dofs( DoFOrdering::DIMENSIONWISE, DoFOrdering::NODEWISE, src[wedge] );
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
                    A[wedge] = A_tmp[wedge] * R[wedge].transposed();
                    // transform source dofs to nt-space
                    auto src_tmp = R[wedge] * src[wedge];
                    for ( int i = 0; i < 18; ++i )
                    {
                        src[wedge]( i ) = src_tmp( i );
                    }

                    // eliminate normal components: Dirichlet on the normal-tangential system
                    int node_start = at_surface ? 3 : 0;
                    int node_end   = at_surface ? 6 : 3;
                    for ( int node_idx = node_start; node_idx < node_end; node_idx++ )
                    {
                        int idx = node_idx * 3;
                        for ( int k = 0; k < 6; ++k )
                        {
                            boundary_mask( k, idx ) = 0.0;
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

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        // no need to reorder or post trafo the pressure:
        // independent of dof ordering, div ops map to the same 6 coarse-grid pressure dofs in the same orderinng

        atomically_add_local_wedge_scalar_coefficients(
            dst_, local_subdomain_id, x_cell / 2, y_cell / 2, r_cell / 2, dst );
    }
};

static_assert( linalg::OperatorLike< Divergence< double > > );

} // namespace terra::fe::wedge::operators::shell
