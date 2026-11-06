#pragma once

// Fast Dirichlet/Neumann path of EpsilonDivDivKerngen without any shared-memory
// atomics: the warp-owner scatter.
//
// Why: on sm_90 a floating-point reduction into shared memory is not a native
// instruction. The tile buffer's red.shared.add.f32 compiles to an
// LDS / FADD / ATOMS.CAST.SPIN / ISETP / BRA compare-and-swap loop, and the per-
// instruction profile of the float kernel at level 8 put those loops and their branch
// bookkeeping at ~40% of the executed instructions and ~45% of the stall samples.
//
// How: one thread per lateral column (team = lat x lat threads, r_tile = 1) walks its
// radial run of cells. The contributions of a cell go to the 8 corners of the hex
// cell, i.e. 4 columns x 2 levels: the upper level is carried in registers into the
// next cell, where it meets that cell's lower contributions, so a node level is
// complete in the thread's registers before anything is written. The 4 lateral
// corners are then reduced to their owner lane with warp shuffles -- a lane owns
// column (tx, ty); the lanes tx-1, ty-1 and (tx-1, ty-1) hold the other three
// corner contributions -- and the owner writes the node once: a plain store for a
// node interior to the team's tile, a global reduction where another team also
// contributes (the tile's boundary and both ends of the radial run). The warp patch
// is PW x PH columns with PW = lat and PH = 32 / lat; lanes on a patch's right edge
// send their outward contributions with a global reduction instead of a shuffle,
// lanes on its top row hand them to the warp above through a small shared-memory
// seam (double buffered over the level parity, one team barrier per level), and
// only the tile's own top row sends them with a global reduction.
//
// Each team also computes the cell below its run (a halo cell, one of 8 or 16
// redundant), so the run's bottom level is complete inside the team and the level
// on top of the run is left to the team above: no level-boundary atomics, and in
// Replace mode only the tile perimeters need zeroing beforehand.
//
// The gather side profits as well: the 4 columns' field values at the upper level
// are the lower level of the next cell, so each cell loads 4 x (3 + 1) values from
// the staged tile instead of 2 x 6 x (3 + 1).

#include "../epsilon_divdiv_kerngen.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
template < int LatTileC, int NlevC >
KOKKOS_INLINE_FUNCTION void
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::operator_fast_dn_warp_path(
    const Team& team,
    const int   local_subdomain_id,
    const int   x0,
    const int   y0,
    const int   r0,
    const int   tx,
    const int   ty,
    const int   x_cell,
    const int   y_cell ) const
{
    static_assert( LatTileC > 0 && NlevC > 1, "the warp-owner path has compile-time tile shapes only" );
    static_assert( 32 % LatTileC == 0 && LatTileC * LatTileC >= 32, "a warp must tile the lateral patch" );

    constexpr int lat_tile = LatTileC;
    constexpr int nlev     = NlevC;     // node levels of the run (cells + 1)
    constexpr int nslot    = NlevC + 1; // staged: the run plus the level below it (the halo cell)
    constexpr int nxy      = ( lat_tile + 1 ) * ( lat_tile + 1 );
    constexpr int PW       = LatTileC;      // warp patch width (columns in x)
    constexpr int PH       = 32 / LatTileC; // warp patch height (columns in y)

    // ---------------- staging (same tile as the tile-buffer path, no dst buffer) ----------------
    auto* geom_mem = reinterpret_cast< GeomT* >( team.team_shmem().get_shmem(
        shmem_block_bytes( sizeof( GeomT ) * ( size_t( nxy ) * 3 + size_t( nslot ) ) ) ) );
    auto* field_mem = reinterpret_cast< StageT* >( team.team_shmem().get_shmem( shmem_block_bytes(
        sizeof( StageT ) * ( size_t( nxy ) * 3 * nslot + size_t( nxy ) * nslot ) ) ) );

    constexpr int NW = ( lat_tile * lat_tile ) / 32; // warps per team
    AccT*         seam_mem = nullptr;
    if constexpr ( NW > 1 )
        seam_mem = reinterpret_cast< AccT* >(
            team.team_shmem().get_shmem( shmem_block_bytes( sizeof( AccT ) * size_t( 2 ) * ( NW - 1 ) * PW * 6 ) ) );
    // seam( parity, warp, column, component, corner ): corner 0 = straight above, 1 = above-right
    auto seam = [&]( int par, int w, int col, int d, int corner ) -> AccT& {
        return seam_mem[( ( ( par * ( NW - 1 ) + w ) * PW + col ) * 3 + d ) * 2 + corner];
    };

    using Scratch       = typename Team::scratch_memory_space;
    using ScratchCoords = Kokkos::View< GeomT[nxy][3], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
    using ScratchR      = Kokkos::View< GeomT[nslot], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
    using ScratchSrc = Kokkos::View< StageT[nxy][3][nslot], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
    using ScratchK   = Kokkos::View< StageT[nxy][nslot], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;

    ScratchCoords coords_sh( geom_mem );
    ScratchR      r_sh( geom_mem + size_t( nxy ) * 3 );
    ScratchSrc    src_sh( field_mem );
    ScratchK      k_sh( field_mem + size_t( nxy ) * 3 * nslot );

    auto node_id = []( int nx, int ny ) -> int { return nx + ( lat_tile + 1 ) * ny; };

    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy ), [&]( int n ) {
        const int xi = x0 + n % ( lat_tile + 1 );
        const int yi = y0 + n / ( lat_tile + 1 );
        if ( xi <= hex_lat_ && yi <= hex_lat_ )
        {
            coords_sh( n, 0 ) = grid_( local_subdomain_id, xi, yi, 0 );
            coords_sh( n, 1 ) = grid_( local_subdomain_id, xi, yi, 1 );
            coords_sh( n, 2 ) = grid_( local_subdomain_id, xi, yi, 2 );
        }
        else
        {
            coords_sh( n, 0 ) = coords_sh( n, 1 ) = coords_sh( n, 2 ) = 0.0;
        }
    } );

    // Slot s holds level r0 - 1 + s: slot 0 is the level below the run (its cell is
    // the halo), slots 1 .. nlev the run itself.
    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nslot ), [&]( int s ) {
        const int rr = r0 - 1 + s;
        r_sh( s ) = ( rr >= 0 && rr <= hex_rad_ ) ? static_cast< double >( radii_( local_subdomain_id, rr ) ) : 0.0;
    } );

    constexpr bool AsyncStaging =
        std::is_same_v< ArithT, ScalarType > && ( sizeof( ScalarType ) == 4 || sizeof( ScalarType ) == 8 );

    if constexpr ( AsyncStaging )
    {
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy * nslot ), [&]( int t ) {
            const int node = t / nslot;
            const int lvl  = t - node * nslot;
            const int xi   = x0 + node % ( lat_tile + 1 );
            const int yi   = y0 + node / ( lat_tile + 1 );
            const int rr   = r0 - 1 + lvl;

            if ( xi <= hex_lat_ && yi <= hex_lat_ && rr >= 0 && rr <= hex_rad_ )
            {
                util::cp_async( &k_sh( node, lvl ), &k_( local_subdomain_id, xi, yi, rr ) );
                util::cp_async( &src_sh( node, 0, lvl ), &src_( local_subdomain_id, xi, yi, rr, 0 ) );
                util::cp_async( &src_sh( node, 1, lvl ), &src_( local_subdomain_id, xi, yi, rr, 1 ) );
                util::cp_async( &src_sh( node, 2, lvl ), &src_( local_subdomain_id, xi, yi, rr, 2 ) );
            }
            else
            {
                k_sh( node, lvl )      = 0.0;
                src_sh( node, 0, lvl ) = src_sh( node, 1, lvl ) = src_sh( node, 2, lvl ) = 0.0;
            }
        } );
        util::cp_async_wait_all();
    }
    else
    {
        // Synchronous (16-bit) loads, a batch of independent ones per thread before any
        // is consumed: the scratch pointers are generic, so the compiler could not hoist
        // a load over the previous item's shared store, and every item paid the full
        // latency (ncu: the conversion waiting on the load was 13 % of the stall samples).
        constexpr int StageBatch = 8;
        constexpr int total      = nxy * nslot;
        const int     ts         = team.team_size();
        const int     rank       = team.team_rank();
        // Uniform per team: a tile whose nodes and levels all lie in the grid needs no
        // per-item guard (ncu: the guarded form's selects and compares were 2.6e8 of
        // the 16-bit kernel's instructions; only the boundary tiles need them).
        const bool inside = ( x0 + lat_tile <= hex_lat_ ) && ( y0 + lat_tile <= hex_lat_ ) && ( r0 >= 1 ) &&
                            ( r0 - 1 + nslot - 1 <= hex_rad_ );
        for ( int base = rank; base < total; base += ts * StageBatch )
        {
            ScalarType vk[StageBatch], v0[StageBatch], v1[StageBatch], v2[StageBatch];
            bool       ok[StageBatch];
#pragma unroll
            for ( int u = 0; u < StageBatch; ++u )
            {
                const int t    = base + u * ts;
                const int node = t / nslot;
                const int lvl  = t - node * nslot;
                const int xi   = x0 + node % ( lat_tile + 1 );
                const int yi   = y0 + node / ( lat_tile + 1 );
                const int rr   = r0 - 1 + lvl;
                ok[u] = ( t < total ) && ( inside || ( xi <= hex_lat_ && yi <= hex_lat_ && rr >= 0 && rr <= hex_rad_ ) );
                if ( ok[u] )
                {
                    vk[u] = k_( local_subdomain_id, xi, yi, rr );
                    v0[u] = src_( local_subdomain_id, xi, yi, rr, 0 );
                    v1[u] = src_( local_subdomain_id, xi, yi, rr, 1 );
                    v2[u] = src_( local_subdomain_id, xi, yi, rr, 2 );
                }
            }
            if ( inside )
            {
#pragma unroll
                for ( int u = 0; u < StageBatch; ++u )
                {
                    const int t = base + u * ts;
                    if ( t < total )
                    {
                        const int node = t / nslot;
                        const int lvl  = t - node * nslot;
                        k_sh( node, lvl )      = static_cast< StageT >( vk[u] );
                        src_sh( node, 0, lvl ) = static_cast< StageT >( v0[u] );
                        src_sh( node, 1, lvl ) = static_cast< StageT >( v1[u] );
                        src_sh( node, 2, lvl ) = static_cast< StageT >( v2[u] );
                    }
                }
            }
            else
            {
#pragma unroll
                for ( int u = 0; u < StageBatch; ++u )
                {
                    const int t = base + u * ts;
                    if ( t < total )
                    {
                        const int node = t / nslot;
                        const int lvl  = t - node * nslot;
                        k_sh( node, lvl )      = ok[u] ? static_cast< StageT >( vk[u] ) : StageT( 0 );
                        src_sh( node, 0, lvl ) = ok[u] ? static_cast< StageT >( v0[u] ) : StageT( 0 );
                        src_sh( node, 1, lvl ) = ok[u] ? static_cast< StageT >( v1[u] ) : StageT( 0 );
                        src_sh( node, 2, lvl ) = ok[u] ? static_cast< StageT >( v2[u] ) : StageT( 0 );
                    }
                }
            }
        }
    }
    team.team_barrier();

    // ---------------- per-thread column walk ----------------
    // Every lane of the warp runs the whole walk (the shuffles need all 32), a lane
    // whose column lies outside the grid just contributes zeros.
    const bool cell_active = ( x_cell < hex_lat_ && y_cell < hex_lat_ );
    const int  lane        = team.team_rank() % 32;
    const int  wy          = team.team_rank() / 32; // warp index = row block of the tile
    const int  ltx         = lane % PW;             // == tx for lat_tile == PW
    const int  lty         = lane / PW;             // row inside the warp patch
    (void) tx;

    constexpr GeoT ONE_THIRD = static_cast< GeoT >( 1.0 / 3.0 );
    constexpr GeoT ONE_SIXTH = static_cast< GeoT >( 1.0 / 6.0 );

    constexpr GeoT dN_ref[6][3] = { { -0.5, -0.5, -ONE_SIXTH }, { 0.5, 0.0, -ONE_SIXTH }, { 0.0, 0.5, -ONE_SIXTH },
                                    { -0.5, -0.5, ONE_SIXTH },  { 0.5, 0.0, ONE_SIXTH },  { 0.0, 0.5, ONE_SIXTH } };

    // Corner c = ddx + 2 ddy of the hex cell; the wedge tables give (ddx, ddy, ddr).
    constexpr int WEDGE_NODE_OFF[2][6][3] = {
        { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
        { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

    const int n00 = node_id( tx, ty );
    const int n10 = node_id( tx + 1, ty );
    const int n01 = node_id( tx, ty + 1 );
    const int n11 = node_id( tx + 1, ty + 1 );
    const int corner_node[4] = { n00, n10, n01, n11 };

    // Column geometry, loaded once: corner c holds coords of column c.
    GeoT cx[4], cy[4], cz[4];
#pragma unroll
    for ( int c = 0; c < 4; ++c )
    {
        cx[c] = coords_sh( corner_node[c], 0 );
        cy[c] = coords_sh( corner_node[c], 1 );
        cz[c] = coords_sh( corner_node[c], 2 );
    }

    // The walk starts at the halo cell (slot 0, the cell below the run) when there is
    // one: with it, the run's bottom level is complete inside this team, and the level
    // on top of the run belongs to the team above. Runs at the CMB start at slot 1.
    const int s_begin = ( r0 > 0 ) ? 0 : 1;

    // Field values of the 4 columns at the cell's lower (sl, kl) and upper (su, ku) level.
    StageT sl[4][3], su[4][3], kl[4], ku[4];
#pragma unroll
    for ( int c = 0; c < 4; ++c )
    {
        kl[c] = k_sh( corner_node[c], s_begin );
#pragma unroll
        for ( int d = 0; d < 3; ++d )
            sl[c][d] = src_sh( corner_node[c], d, s_begin );
    }

    // Contributions to the node level under the current cell (acc: carried in from
    // the previous cell) and above it (up).
    AccT acc[4][3], up[4][3];
#pragma unroll
    for ( int c = 0; c < 4; ++c )
#pragma unroll
        for ( int d = 0; d < 3; ++d )
            acc[c][d] = AccT( 0 );

    const bool replace_mode = ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace );

    // One node level, complete in registers across the warp: reduce the 4 corner
    // contributions to the owner lane, write once, send the outward edge
    // contributions of the patch with global reductions.
    auto flush_level = [&]( AccT ( &v )[4][3], const int s ) {
        const int rr  = r0 - 1 + s;
        const int par = s & 1;

        // Owner reduction. shfl_up leaves the first `delta` lanes with their own
        // value, so those take 0 explicitly; the first row of a warp gets its
        // from-below values through the seam instead.
        AccT own[3];
#pragma unroll
        for ( int d = 0; d < 3; ++d )
        {
            const AccT from_left  = util::warp_shfl_up( v[1][d], 1 );
            const AccT from_below = util::warp_shfl_up( v[2][d], PW );
            const AccT from_diag  = util::warp_shfl_up( v[3][d], PW + 1 );
            own[d]                = v[0][d];
            own[d] += ( ltx > 0 ) ? from_left : AccT( 0 );
            own[d] += ( lty > 0 ) ? from_below : AccT( 0 );
            own[d] += ( ltx > 0 && lty > 0 ) ? from_diag : AccT( 0 );
        }

        if constexpr ( NW > 1 )
        {
            if ( lty == PH - 1 && wy < NW - 1 )
            {
#pragma unroll
                for ( int d = 0; d < 3; ++d )
                {
                    seam( par, wy, ltx, d, 0 ) = v[2][d];
                    seam( par, wy, ltx, d, 1 ) = v[3][d];
                }
            }
            team.team_barrier();
            if ( lty == 0 && wy > 0 )
            {
#pragma unroll
                for ( int d = 0; d < 3; ++d )
                {
                    own[d] += seam( par, wy - 1, ltx, d, 0 );
                    own[d] += ( ltx > 0 ) ? seam( par, wy - 1, ltx - 1, d, 1 ) : AccT( 0 );
                }
            }
        }

        if ( cell_active && rr <= hex_rad_ && det_scatter_ )
        {
            // Two-phase scatter: the owner stores its node, the outward edge
            // contributions go to the side buffer (one writer per slot), and
            // perimeter_reduce adds them afterwards in a fixed order.
#pragma unroll
            for ( int d = 0; d < 3; ++d )
            {
                ScalarType* p = &dst_( local_subdomain_id, x_cell, y_cell, rr, d );
                *p = static_cast< ScalarType >( replace_mode ? own[d] : static_cast< AccT >( *p ) + own[d] );
                if ( ltx == PW - 1 )
                {
                    pb_.put_edge( local_subdomain_id, 0, 1, 0, d, x_cell, y_cell, rr, v[1][d] );
                    pb_.put_edge( local_subdomain_id, 0, 1, 1, d, x_cell, y_cell, rr, v[3][d] );
                }
                if ( lty == PH - 1 && wy == NW - 1 )
                {
                    pb_.put_edge( local_subdomain_id, 0, 0, 1, d, x_cell, y_cell, rr, v[2][d] );
                    if ( ltx != PW - 1 )
                        pb_.put_edge( local_subdomain_id, 0, 1, 1, d, x_cell, y_cell, rr, v[3][d] );
                }
            }
        }
        else if ( cell_active && rr <= hex_rad_ )
        {
            // Interior node of the tile: only this team touches it, and inside the
            // team only this lane.
            const bool interior = ( tx > 0 ) && ( ty > 0 ) && replace_mode;
#pragma unroll
            for ( int d = 0; d < 3; ++d )
            {
                if ( interior )
                    dst_( local_subdomain_id, x_cell, y_cell, rr, d ) = static_cast< ScalarType >( own[d] );
                else if ( own[d] != AccT( 0 ) )
                    global_add( &dst_( local_subdomain_id, x_cell, y_cell, rr, d ), own[d] );

                // Tile edges: right column goes to column tx+1 of the next tile, the
                // tile's top row to row ty+1.
                if ( ltx == PW - 1 )
                {
                    if ( v[1][d] != AccT( 0 ) )
                        global_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, rr, d ), v[1][d] );
                    if ( v[3][d] != AccT( 0 ) )
                        global_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, rr, d ), v[3][d] );
                }
                if ( lty == PH - 1 && wy == NW - 1 )
                {
                    if ( v[2][d] != AccT( 0 ) )
                        global_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, rr, d ), v[2][d] );
                    if ( ltx != PW - 1 && v[3][d] != AccT( 0 ) )
                        global_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, rr, d ), v[3][d] );
                }
            }
        }
    };

    // Slot of the level on top of the last computed cell (uniform over the team).
    int s_top = s_begin;
    for ( int s = s_begin; s < nlev; ++s )
    {
        const int r_cell = r0 - 1 + s;
        if ( r_cell >= hex_rad_ ) // uniform over the team
            break;
        s_top = s + 1;

#pragma unroll
        for ( int cc = 0; cc < 4; ++cc )
        {
            ku[cc] = k_sh( corner_node[cc], s + 1 );
#pragma unroll
            for ( int d = 0; d < 3; ++d )
            {
                su[cc][d] = src_sh( corner_node[cc], d, s + 1 );
                up[cc][d] = AccT( 0 );
            }
        }

        if ( cell_active )
        {
            const GeoT r_0 = static_cast< GeoT >( r_sh( s ) );
            const GeoT r_1 = static_cast< GeoT >( r_sh( s + 1 ) );

            const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
            const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );
            const bool at_boundary = at_cmb || at_surface;
            bool       treat_boundary_dirichlet = false;
            if ( at_boundary )
            {
                const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
                treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
            }
            const int cmb_shift     = ( at_boundary && treat_boundary_dirichlet && at_cmb ) ? 3 : 0;
            const int surface_shift = ( at_boundary && treat_boundary_dirichlet && at_surface ) ? 3 : 0;

            const GeoT half_dr = static_cast< GeoT >( 0.5 ) * ( r_1 - r_0 );
            const GeoT r_mid   = static_cast< GeoT >( 0.5 ) * ( r_0 + r_1 );

#pragma unroll
            for ( int w = 0; w < 2; ++w )
            {
                // Wedge vertices as corners: w = 0 -> (0,0),(1,0),(0,1); w = 1 -> (1,1),(0,1),(1,0).
                constexpr int V0[2] = { 0, 3 }, V1[2] = { 1, 2 }, V2[2] = { 2, 1 };
                const int v0 = V0[w], v1 = V1[w], v2 = V2[w];

                // Register-resident node access: corner and level of wedge node n.
                auto S = [&]( const int n, const int d ) -> StageT {
                    const int cc = WEDGE_NODE_OFF[w][n][0] + 2 * WEDGE_NODE_OFF[w][n][1];
                    return WEDGE_NODE_OFF[w][n][2] ? su[cc][d] : sl[cc][d];
                };
                auto K = [&]( const int n ) -> StageT {
                    const int cc = WEDGE_NODE_OFF[w][n][0] + 2 * WEDGE_NODE_OFF[w][n][1];
                    return WEDGE_NODE_OFF[w][n][2] ? ku[cc] : kl[cc];
                };
                auto add = [&]( const int n, const int d, const ConT val ) {
                    const int cc = WEDGE_NODE_OFF[w][n][0] + 2 * WEDGE_NODE_OFF[w][n][1];
                    if ( WEDGE_NODE_OFF[w][n][2] )
                        up[cc][d] += static_cast< AccT >( val );
                    else
                        acc[cc][d] += static_cast< AccT >( val );
                };

                StageT k_sum = 0.0;
#pragma unroll
                for ( int n = 0; n < 6; ++n )
                    k_sum += K( n );
                const StageT k_eval = static_cast< StageT >( static_cast< double >( k_sum ) * ( 1.0 / 6.0 ) );

                const GeoT J_0_0 = r_mid * ( -cx[v0] + cx[v1] );
                const GeoT J_0_1 = r_mid * ( -cx[v0] + cx[v2] );
                const GeoT J_0_2 = half_dr * ( ONE_THIRD * ( cx[v0] + cx[v1] + cx[v2] ) );
                const GeoT J_1_0 = r_mid * ( -cy[v0] + cy[v1] );
                const GeoT J_1_1 = r_mid * ( -cy[v0] + cy[v2] );
                const GeoT J_1_2 = half_dr * ( ONE_THIRD * ( cy[v0] + cy[v1] + cy[v2] ) );
                const GeoT J_2_0 = r_mid * ( -cz[v0] + cz[v1] );
                const GeoT J_2_1 = r_mid * ( -cz[v0] + cz[v2] );
                const GeoT J_2_2 = half_dr * ( ONE_THIRD * ( cz[v0] + cz[v1] + cz[v2] ) );

                const GeoT J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                   J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

                const ConT kwJ       = static_cast< ConT >( static_cast< GeoT >( k_eval ) * Kokkos::abs( J_det ) );
                const GeoT   inv_det = static_cast< GeoT >( 1.0 ) / J_det;

                const GeoT i00 = inv_det * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                const GeoT i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                const GeoT i02 = inv_det * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                const GeoT i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                const GeoT i11 = inv_det * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                const GeoT i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                const GeoT i20 = inv_det * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                const GeoT i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                const GeoT i22 = inv_det * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                // ---- phase 1: gradient of the field at the one quadrature point ----
                ConT gu00 = 0.0, gu10 = 0.0, gu11 = 0.0, gu20 = 0.0, gu21 = 0.0, gu22 = 0.0, div_u = 0.0;
                ConT a0n[6], a1n[6], a2n[6];
#pragma unroll
                for ( int n = 0; n < 6; ++n )
                {
                    const GeoT gx = dN_ref[n][0], gy = dN_ref[n][1], gz = dN_ref[n][2];
                    a0n[n]        = static_cast< ConT >( i00 * gx + i01 * gy + i02 * gz );
                    a1n[n]        = static_cast< ConT >( i10 * gx + i11 * gy + i12 * gz );
                    a2n[n]        = static_cast< ConT >( i20 * gx + i21 * gy + i22 * gz );
                    if ( n < cmb_shift || n >= 6 - surface_shift )
                        continue;
                    const ConT half = static_cast< ConT >( 0.5 );
                    const ConT s0 = static_cast< ConT >( S( n, 0 ) ), s1 = static_cast< ConT >( S( n, 1 ) ),
                               s2 = static_cast< ConT >( S( n, 2 ) );
                    gu00 += a0n[n] * s0;
                    gu11 += a1n[n] * s1;
                    gu22 += a2n[n] * s2;
                    gu10 += half * ( a1n[n] * s0 + a0n[n] * s1 );
                    gu20 += half * ( a2n[n] * s0 + a0n[n] * s2 );
                    gu21 += half * ( a2n[n] * s1 + a1n[n] * s2 );
                    div_u += a0n[n] * s0 + a1n[n] * s1 + a2n[n] * s2;
                }

                // ---- phase 2: scatter into the corner registers ----
                {
                    constexpr double NEG_TWO_THIRDS = -0.66666666666666663;
                    const ConT       two            = static_cast< ConT >( 2.0 );
                    const ConT       ntt            = static_cast< ConT >( NEG_TWO_THIRDS );
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        if ( n < cmb_shift || n >= 6 - surface_shift )
                            continue;
                        const ConT a0 = a0n[n], a1 = a1n[n], a2 = a2n[n];
                        add( n, 0, kwJ * ( two * ( a0 * gu00 + a1 * gu10 + a2 * gu20 ) + ntt * a0 * div_u ) );
                        add( n, 1, kwJ * ( two * ( a0 * gu10 + a1 * gu11 + a2 * gu21 ) + ntt * a1 * div_u ) );
                        add( n, 2, kwJ * ( two * ( a0 * gu20 + a1 * gu21 + a2 * gu22 ) + ntt * a2 * div_u ) );
                    }
                }

                // ---- hourglass stabilization (see the tile-buffer path) ----
                if ( stab_c_ > 0.0 )
                {
                    constexpr double G1[6]   = { ONE_THIRD, -2.0 * ONE_THIRD, ONE_THIRD, -ONE_THIRD, 2.0 * ONE_THIRD, -ONE_THIRD };
                    constexpr double G2[6]   = { ONE_THIRD, ONE_THIRD, -2.0 * ONE_THIRD, -ONE_THIRD, -ONE_THIRD, 2.0 * ONE_THIRD };
                    constexpr double NORM2_G = 4.0 / 3.0;

                    GeoT sum_gg = 0;
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        const GeoT b0 = static_cast< GeoT >( a0n[n] ), b1 = static_cast< GeoT >( a1n[n] ),
                                   b2 = static_cast< GeoT >( a2n[n] );
                        sum_gg += b0 * b0 + b1 * b1 + b2 * b2;
                    }
                    const GeoT   diag_mean = static_cast< GeoT >( kwJ ) * static_cast< GeoT >( 10.0 / 54.0 ) * sum_gg;
                    const ConT eps =
                        static_cast< ConT >( static_cast< GeoT >( stab_c_ ) * diag_mean / static_cast< GeoT >( NORM2_G ) );

#pragma unroll
                    for ( int d = 0; d < 3; ++d )
                    {
                        ConT c1 = 0.0, c2 = 0.0;
#pragma unroll
                        for ( int n = 0; n < 6; ++n )
                        {
                            if ( n < cmb_shift || n >= 6 - surface_shift )
                                continue;
                            const ConT u = static_cast< ConT >( S( n, d ) );
                            c1 += static_cast< ConT >( G1[n] ) * u;
                            c2 += static_cast< ConT >( G2[n] ) * u;
                        }
#pragma unroll
                        for ( int n = 0; n < 6; ++n )
                        {
                            if ( n < cmb_shift || n >= 6 - surface_shift )
                                continue;
                            add( n, d, eps * ( static_cast< ConT >( G1[n] ) * c1 + static_cast< ConT >( G2[n] ) * c2 ) );
                        }
                    }
                }

                // ---- Dirichlet rows: the diagonal entry stands in for the row ----
                if ( treat_boundary_dirichlet && at_boundary )
                {
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        if ( n < surface_shift || n >= 6 - cmb_shift )
                            continue;
                        const GeoT g0 = static_cast< GeoT >( a0n[n] ), g1 = static_cast< GeoT >( a1n[n] ),
                                   g2 = static_cast< GeoT >( a2n[n] );
                        const GeoT   gg = g0 * g0 + g1 * g1 + g2 * g2;
                        const ConT d0 = static_cast< ConT >( gg + ONE_THIRD * g0 * g0 );
                        const ConT d1 = static_cast< ConT >( gg + ONE_THIRD * g1 * g1 );
                        const ConT d2 = static_cast< ConT >( gg + ONE_THIRD * g2 * g2 );
                        add( n, 0, kwJ * S( n, 0 ) * d0 );
                        add( n, 1, kwJ * S( n, 1 ) * d1 );
                        add( n, 2, kwJ * S( n, 2 ) * d2 );
                    }
                }
            } // wedge loop
        }

        // The level under this cell is complete for this team: it got the previous
        // cell's upper and this cell's lower contributions -- except slot 0, the level
        // under the halo cell, which belongs to the team below.
        if ( s >= 1 )
            flush_level( acc, s );

#pragma unroll
        for ( int cc = 0; cc < 4; ++cc )
        {
            kl[cc] = ku[cc];
#pragma unroll
            for ( int d = 0; d < 3; ++d )
            {
                sl[cc][d]  = su[cc][d];
                acc[cc][d] = up[cc][d];
            }
        }
    }

    // The level on top of the run belongs to the team above (its halo cell completes
    // it) -- unless it is the surface, which no team is above.
    if ( r0 - 1 + s_top == hex_rad_ )
        flush_level( acc, s_top );
}

} // namespace terra::fe::wedge::operators::shell
