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
// node interior to the team's tile, a global reduction where another team or warp
// also contributes (the tile's boundary and the first row of every warp patch,
// whose lower neighbour is a different warp). The warp patch is PW x PH columns
// with PW = lat and PH = 32 / lat; lanes on a patch's right or top edge send their
// outward contributions with a global reduction instead of a shuffle.
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
    constexpr int nlev     = NlevC;
    constexpr int nxy      = ( lat_tile + 1 ) * ( lat_tile + 1 );
    constexpr int PW       = LatTileC;      // warp patch width (columns in x)
    constexpr int PH       = 32 / LatTileC; // warp patch height (columns in y)

    // ---------------- staging (same tile as the tile-buffer path, no dst buffer) ----------------
    auto* geom_mem = reinterpret_cast< GeomT* >( team.team_shmem().get_shmem(
        shmem_block_bytes( sizeof( GeomT ) * ( size_t( nxy ) * 3 + size_t( nlev ) ) ) ) );
    auto* field_mem = reinterpret_cast< ArithT* >( team.team_shmem().get_shmem( shmem_block_bytes(
        sizeof( ArithT ) * ( size_t( nxy ) * 3 * nlev + size_t( nxy ) * nlev ) ) ) );

    using Scratch       = typename Team::scratch_memory_space;
    using ScratchCoords = Kokkos::View< GeomT[nxy][3], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
    using ScratchR      = Kokkos::View< GeomT[nlev], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
    using ScratchSrc = Kokkos::View< ArithT[nxy][3][nlev], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
    using ScratchK   = Kokkos::View< ArithT[nxy][nlev], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;

    ScratchCoords coords_sh( geom_mem );
    ScratchR      r_sh( geom_mem + size_t( nxy ) * 3 );
    ScratchSrc    src_sh( field_mem );
    ScratchK      k_sh( field_mem + size_t( nxy ) * 3 * nlev );

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

    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nlev ), [&]( int lvl ) {
        const int rr = r0 + lvl;
        r_sh( lvl )  = ( rr <= hex_rad_ ) ? static_cast< double >( radii_( local_subdomain_id, rr ) ) : 0.0;
    } );

    constexpr bool AsyncStaging =
        std::is_same_v< ArithT, ScalarType > && ( sizeof( ScalarType ) == 4 || sizeof( ScalarType ) == 8 );

    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy * nlev ), [&]( int t ) {
        const int node = t / nlev;
        const int lvl  = t - node * nlev;
        const int xi   = x0 + node % ( lat_tile + 1 );
        const int yi   = y0 + node / ( lat_tile + 1 );
        const int rr   = r0 + lvl;

        if ( xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_ )
        {
            if constexpr ( AsyncStaging )
            {
                util::cp_async( &k_sh( node, lvl ), &k_( local_subdomain_id, xi, yi, rr ) );
                util::cp_async( &src_sh( node, 0, lvl ), &src_( local_subdomain_id, xi, yi, rr, 0 ) );
                util::cp_async( &src_sh( node, 1, lvl ), &src_( local_subdomain_id, xi, yi, rr, 1 ) );
                util::cp_async( &src_sh( node, 2, lvl ), &src_( local_subdomain_id, xi, yi, rr, 2 ) );
            }
            else
            {
                k_sh( node, lvl )      = static_cast< ArithT >( k_( local_subdomain_id, xi, yi, rr ) );
                src_sh( node, 0, lvl ) = static_cast< ArithT >( src_( local_subdomain_id, xi, yi, rr, 0 ) );
                src_sh( node, 1, lvl ) = static_cast< ArithT >( src_( local_subdomain_id, xi, yi, rr, 1 ) );
                src_sh( node, 2, lvl ) = static_cast< ArithT >( src_( local_subdomain_id, xi, yi, rr, 2 ) );
            }
        }
        else
        {
            k_sh( node, lvl )      = 0.0;
            src_sh( node, 0, lvl ) = src_sh( node, 1, lvl ) = src_sh( node, 2, lvl ) = 0.0;
        }
    } );

    if constexpr ( AsyncStaging )
        util::cp_async_wait_all();
    team.team_barrier();

    // ---------------- per-thread column walk ----------------
    // Every lane of the warp runs the whole walk (the shuffles need all 32), a lane
    // whose column lies outside the grid just contributes zeros.
    const bool cell_active = ( x_cell < hex_lat_ && y_cell < hex_lat_ );
    const int  lane        = team.team_rank() % 32;
    const int  ltx         = lane % PW; // == tx for lat_tile == PW
    const int  lty         = lane / PW; // row inside the warp patch
    (void) tx;
    (void) ty;

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

    // Field values of the 4 columns at the cell's lower (sl, kl) and upper (su, ku) level.
    ArithT sl[4][3], su[4][3], kl[4], ku[4];
#pragma unroll
    for ( int c = 0; c < 4; ++c )
    {
        kl[c] = k_sh( corner_node[c], 0 );
#pragma unroll
        for ( int d = 0; d < 3; ++d )
            sl[c][d] = src_sh( corner_node[c], d, 0 );
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
    auto flush_level = [&]( AccT ( &v )[4][3], const int lvl, const bool level_boundary ) {
        const int rr = r0 + lvl;
#pragma unroll
        for ( int d = 0; d < 3; ++d )
        {
            // Owner reduction. shfl_up leaves the first `delta` lanes with their own
            // value, so those take 0 explicitly.
            const AccT from_left  = util::warp_shfl_up( v[1][d], 1 );
            const AccT from_below = util::warp_shfl_up( v[2][d], PW );
            const AccT from_diag  = util::warp_shfl_up( v[3][d], PW + 1 );
            AccT own = v[0][d];
            own += ( ltx > 0 ) ? from_left : AccT( 0 );
            own += ( lty > 0 ) ? from_below : AccT( 0 );
            own += ( ltx > 0 && lty > 0 ) ? from_diag : AccT( 0 );

            if ( cell_active && rr <= hex_rad_ )
            {
                // Interior node of the tile: only this team touches it, and inside the
                // team only this lane -- unless its lower row is another warp's.
                const bool interior = ( tx > 0 ) && ( lty > 0 ) && !level_boundary && replace_mode;
                if ( interior )
                    dst_( local_subdomain_id, x_cell, y_cell, rr, d ) = static_cast< ScalarType >( own );
                else if ( own != AccT( 0 ) )
                    global_add( &dst_( local_subdomain_id, x_cell, y_cell, rr, d ), own );

                // Patch edges: right column goes to column tx+1 (another warp or the
                // tile halo), top row to row ty+1.
                if ( ltx == PW - 1 )
                {
                    if ( v[1][d] != AccT( 0 ) )
                        global_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, rr, d ), v[1][d] );
                    if ( v[3][d] != AccT( 0 ) )
                        global_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, rr, d ), v[3][d] );
                }
                if ( lty == PH - 1 )
                {
                    if ( v[2][d] != AccT( 0 ) )
                        global_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, rr, d ), v[2][d] );
                    if ( ltx != PW - 1 && v[3][d] != AccT( 0 ) )
                        global_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, rr, d ), v[3][d] );
                }
            }
        }
    };

    for ( int c = 0; c < nlev - 1; ++c )
    {
        const int r_cell = r0 + c;
        if ( r_cell >= hex_rad_ ) // uniform over the team
            break;

#pragma unroll
        for ( int cc = 0; cc < 4; ++cc )
        {
            ku[cc] = k_sh( corner_node[cc], c + 1 );
#pragma unroll
            for ( int d = 0; d < 3; ++d )
            {
                su[cc][d] = src_sh( corner_node[cc], d, c + 1 );
                up[cc][d] = AccT( 0 );
            }
        }

        if ( cell_active )
        {
            const GeoT r_0 = static_cast< GeoT >( r_sh( c ) );
            const GeoT r_1 = static_cast< GeoT >( r_sh( c + 1 ) );

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
                auto S = [&]( const int n, const int d ) -> ArithT {
                    const int cc = WEDGE_NODE_OFF[w][n][0] + 2 * WEDGE_NODE_OFF[w][n][1];
                    return WEDGE_NODE_OFF[w][n][2] ? su[cc][d] : sl[cc][d];
                };
                auto K = [&]( const int n ) -> ArithT {
                    const int cc = WEDGE_NODE_OFF[w][n][0] + 2 * WEDGE_NODE_OFF[w][n][1];
                    return WEDGE_NODE_OFF[w][n][2] ? ku[cc] : kl[cc];
                };
                auto add = [&]( const int n, const int d, const ArithT val ) {
                    const int cc = WEDGE_NODE_OFF[w][n][0] + 2 * WEDGE_NODE_OFF[w][n][1];
                    if ( WEDGE_NODE_OFF[w][n][2] )
                        up[cc][d] += static_cast< AccT >( val );
                    else
                        acc[cc][d] += static_cast< AccT >( val );
                };

                ArithT k_sum = 0.0;
#pragma unroll
                for ( int n = 0; n < 6; ++n )
                    k_sum += K( n );
                const ArithT k_eval = static_cast< ArithT >( static_cast< double >( k_sum ) * ( 1.0 / 6.0 ) );

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

                const ArithT kwJ     = static_cast< ArithT >( static_cast< GeoT >( k_eval ) * Kokkos::abs( J_det ) );
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
                ArithT gu00 = 0.0, gu10 = 0.0, gu11 = 0.0, gu20 = 0.0, gu21 = 0.0, gu22 = 0.0, div_u = 0.0;
                ArithT a0n[6], a1n[6], a2n[6];
#pragma unroll
                for ( int n = 0; n < 6; ++n )
                {
                    const GeoT gx = dN_ref[n][0], gy = dN_ref[n][1], gz = dN_ref[n][2];
                    a0n[n]        = static_cast< ArithT >( i00 * gx + i01 * gy + i02 * gz );
                    a1n[n]        = static_cast< ArithT >( i10 * gx + i11 * gy + i12 * gz );
                    a2n[n]        = static_cast< ArithT >( i20 * gx + i21 * gy + i22 * gz );
                    if ( n < cmb_shift || n >= 6 - surface_shift )
                        continue;
                    const ArithT half = static_cast< ArithT >( 0.5 );
                    const ArithT s0 = S( n, 0 ), s1 = S( n, 1 ), s2 = S( n, 2 );
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
                    const ArithT     two            = static_cast< ArithT >( 2.0 );
                    const ArithT     ntt            = static_cast< ArithT >( NEG_TWO_THIRDS );
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        if ( n < cmb_shift || n >= 6 - surface_shift )
                            continue;
                        const ArithT a0 = a0n[n], a1 = a1n[n], a2 = a2n[n];
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
                    const ArithT eps =
                        static_cast< ArithT >( static_cast< GeoT >( stab_c_ ) * diag_mean / static_cast< GeoT >( NORM2_G ) );

#pragma unroll
                    for ( int d = 0; d < 3; ++d )
                    {
                        ArithT c1 = 0.0, c2 = 0.0;
#pragma unroll
                        for ( int n = 0; n < 6; ++n )
                        {
                            if ( n < cmb_shift || n >= 6 - surface_shift )
                                continue;
                            const ArithT u = S( n, d );
                            c1 += static_cast< ArithT >( G1[n] ) * u;
                            c2 += static_cast< ArithT >( G2[n] ) * u;
                        }
#pragma unroll
                        for ( int n = 0; n < 6; ++n )
                        {
                            if ( n < cmb_shift || n >= 6 - surface_shift )
                                continue;
                            add( n, d, eps * ( static_cast< ArithT >( G1[n] ) * c1 + static_cast< ArithT >( G2[n] ) * c2 ) );
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
                        const ArithT d0 = static_cast< ArithT >( gg + ONE_THIRD * g0 * g0 );
                        const ArithT d1 = static_cast< ArithT >( gg + ONE_THIRD * g1 * g1 );
                        const ArithT d2 = static_cast< ArithT >( gg + ONE_THIRD * g2 * g2 );
                        add( n, 0, kwJ * S( n, 0 ) * d0 );
                        add( n, 1, kwJ * S( n, 1 ) * d1 );
                        add( n, 2, kwJ * S( n, 2 ) * d2 );
                    }
                }
            } // wedge loop
        }

        // Node level c is complete for this team: it got the previous cell's upper
        // and this cell's lower contributions. Level 0 of the run also belongs to
        // the radial tile below, so it is a boundary level.
        flush_level( acc, c, c == 0 );

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

    // The run's top level: shared with the radial tile above.
    {
        int last = nlev - 1;
        if ( r0 + last > hex_rad_ )
            last = hex_rad_ - r0;
        flush_level( acc, last, true );
    }
}

} // namespace terra::fe::wedge::operators::shell
