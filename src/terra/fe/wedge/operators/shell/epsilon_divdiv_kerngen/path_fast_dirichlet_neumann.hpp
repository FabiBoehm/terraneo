#pragma once

// Fast Dirichlet/Neumann path of EpsilonDivDivKerngen: matrix-free, team-tiled, one
//
// Jacobian per wedge in GeoT, contraction in ArithT, tile-local scatter for narrow storage.

#include "../epsilon_divdiv_kerngen.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
template < bool Diagonal, bool TileScatter, int LatTileC, int NlevC >
KOKKOS_INLINE_FUNCTION void
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::operator_fast_dirichlet_neumann_path(
    const Team& team,
    const int   local_subdomain_id,
    const int   x0,
    const int   y0,
    const int   r0,
    const int   tx,
    const int   ty,
    const int   tr,
    const int   x_cell,
    const int   y_cell ) const
{
    // Tile shape as immediates when this instantiation has one (LatTileC, NlevC > 0):
    // every scratch stride, the node numbering and the staging loops' div/mod then
    // fold; the (0, 0) instantiation keeps the runtime shape for any other tile.
    constexpr bool StaticTile = ( LatTileC > 0 && NlevC > 0 );
    const int lat_tile = StaticTile ? LatTileC : lat_tile_;
    const int nlev     = StaticTile ? NlevC : r_tile_block_ + 1;
    const int nxy      = ( lat_tile + 1 ) * ( lat_tile + 1 );
    constexpr int NxyC = ( LatTileC + 1 ) * ( LatTileC + 1 );

    // Geometry and fields come from two separate scratch blocks so they can have
    // different element types (see team_shmem_size_dn).
    auto* geom_mem = reinterpret_cast< GeomT* >( team.team_shmem().get_shmem(
        shmem_block_bytes( sizeof( GeomT ) * ( size_t( nxy ) * 3 + size_t( nlev ) ) ) ) );
    auto* field_mem = reinterpret_cast< ArithT* >( team.team_shmem().get_shmem( shmem_block_bytes(
        sizeof( ArithT ) * ( size_t( nxy ) * 3 * nlev + size_t( nxy ) * nlev ) ) ) );

    auto* acc_mem = reinterpret_cast< AccT* >( team.team_shmem().get_shmem(
        shmem_block_bytes( sizeof( AccT ) * ( TileScatter ? size_t( nxy ) * 3 * nlev : 0 ) ) ) );

    using Scratch = typename Team::scratch_memory_space;
    using ScratchCoords = std::conditional_t<
        StaticTile,
        Kokkos::View< GeomT[NxyC][3], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >,
        Kokkos::View< GeomT**, Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged > >;
    using ScratchR = std::conditional_t<
        StaticTile,
        Kokkos::View< GeomT[NlevC], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >,
        Kokkos::View< GeomT*, Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged > >;
    using ScratchSrc = std::conditional_t<
        StaticTile,
        Kokkos::View< ArithT[NxyC][3][NlevC], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >,
        Kokkos::View< ArithT***, Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged > >;
    using ScratchK = std::conditional_t<
        StaticTile,
        Kokkos::View< ArithT[NxyC][NlevC], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >,
        Kokkos::View< ArithT**, Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged > >;

    auto make_coords = [&]() {
        if constexpr ( StaticTile ) return ScratchCoords( geom_mem );
        else return ScratchCoords( geom_mem, nxy, 3 );
    };
    auto make_r = [&]() {
        if constexpr ( StaticTile ) return ScratchR( geom_mem + size_t( nxy ) * 3 );
        else return ScratchR( geom_mem + size_t( nxy ) * 3, nlev );
    };
    auto make_src = [&]() {
        if constexpr ( StaticTile ) return ScratchSrc( field_mem );
        else return ScratchSrc( field_mem, nxy, 3, nlev );
    };
    auto make_k = [&]() {
        if constexpr ( StaticTile ) return ScratchK( field_mem + size_t( nxy ) * 3 * nlev );
        else return ScratchK( field_mem + size_t( nxy ) * 3 * nlev, nxy, nlev );
    };
    ScratchCoords coords_sh = make_coords();
    ScratchR      r_sh      = make_r();
    ScratchSrc    src_sh    = make_src();
    ScratchK      k_sh      = make_k();

    // Tile-local scatter buffer: every cell this team owns writes only to nodes the
    // team already stages, so the whole tile accumulates here and reaches global
    // memory as one atomic per node instead of one per wedge-node.
    using ScratchDst = std::conditional_t<
        StaticTile && TileScatter,
        Kokkos::View< AccT[NxyC][3][NlevC], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >,
        Kokkos::View< AccT***, Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged > >;
    auto make_dst = [&]() {
        if constexpr ( StaticTile && TileScatter ) return ScratchDst( acc_mem );
        else return ScratchDst( acc_mem, TileScatter ? nxy : 0, 3, TileScatter ? nlev : 0 );
    };
    ScratchDst dst_sh = make_dst();

    auto node_id = [&]( int nx, int ny ) -> int { return nx + ( lat_tile + 1 ) * ny; };

    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy ), [&]( int n ) {
        const int dxn = n % ( lat_tile + 1 );
        const int dyn = n / ( lat_tile + 1 );
        const int xi  = x0 + dxn;
        const int yi  = y0 + dyn;

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

    const int total_pairs = nxy * nlev;
    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, total_pairs ), [&]( int t ) {
        const int node = t / nlev;
        const int lvl  = t - node * nlev;

        const int dxn = node % ( lat_tile + 1 );
        const int dyn = node / ( lat_tile + 1 );

        const int xi = x0 + dxn;
        const int yi = y0 + dyn;
        const int rr = r0 + lvl;

        if ( xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_ )
        {
            // Straight into the tile's own type. Going through double here is a
            // leftover from when the tiles were double whatever ScalarT was; with
            // ArithT tiles it makes a 16-bit gather a half->double->half round
            // trip, four per node and level. The 16-bit Kokkos types only convert
            // out of themselves explicitly, hence the casts.
            k_sh( node, lvl )      = static_cast< ArithT >( k_( local_subdomain_id, xi, yi, rr ) );
            src_sh( node, 0, lvl ) = static_cast< ArithT >( src_( local_subdomain_id, xi, yi, rr, 0 ) );
            src_sh( node, 1, lvl ) = static_cast< ArithT >( src_( local_subdomain_id, xi, yi, rr, 1 ) );
            src_sh( node, 2, lvl ) = static_cast< ArithT >( src_( local_subdomain_id, xi, yi, rr, 2 ) );
        }
        else
        {
            k_sh( node, lvl )      = 0.0;
            src_sh( node, 0, lvl ) = src_sh( node, 1, lvl ) = src_sh( node, 2, lvl ) = 0.0;
        }

        if constexpr ( TileScatter )
            dst_sh( node, 0, lvl ) = dst_sh( node, 1, lvl ) = dst_sh( node, 2, lvl ) = AccT( 0 );
    } );

    team.team_barrier();

    // Not a return: every thread must reach the team barrier before the tile
    // buffer is flushed, so an out-of-range cell skips the work instead of leaving.
    const bool cell_active = ( x_cell < hex_lat_ && y_cell < hex_lat_ );

    constexpr GeoT ONE_THIRD = static_cast< GeoT >( 1.0 / 3.0 );
    constexpr GeoT ONE_SIXTH = static_cast< GeoT >( 1.0 / 6.0 );

    static constexpr GeoT dN_ref[6][3] = {
        { -0.5, -0.5, -ONE_SIXTH },
        { 0.5, 0.0, -ONE_SIXTH },
        { 0.0, 0.5, -ONE_SIXTH },
        { -0.5, -0.5, ONE_SIXTH },
        { 0.5, 0.0, ONE_SIXTH },
        { 0.0, 0.5, ONE_SIXTH } };

    static constexpr int WEDGE_NODE_OFF[2][6][3] = {
        { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
        { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };
    // Read through a select on the wedge: with the node index an immediate (the loops
    // have constant trip counts) both operands are constants and nothing is loaded.
    auto wedge_off = [&]( const int w, const int n, const int k ) -> int {
        return w == 0 ? WEDGE_NODE_OFF[0][n][k] : WEDGE_NODE_OFF[1][n][k];
    };

    const int n00 = node_id( tx, ty );
    const int n01 = node_id( tx, ty + 1 );
    const int n10 = node_id( tx + 1, ty );
    const int n11 = node_id( tx + 1, ty + 1 );

    for ( int pass = 0; cell_active && pass < r_passes_; ++pass )
    {
        const int lvl0   = pass * r_tile_ + tr;
        const int r_cell = r0 + lvl0;

        if ( r_cell >= hex_rad_ )
            break;

        const GeoT r_0 = static_cast< GeoT >( r_sh( lvl0 ) );
        const GeoT r_1 = static_cast< GeoT >( r_sh( lvl0 + 1 ) );

        const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

        const bool at_boundary              = at_cmb || at_surface;
        bool       treat_boundary_dirichlet = false;
        if ( at_boundary )
        {
            const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
            treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
        }

        const int cmb_shift = ( ( at_boundary && treat_boundary_dirichlet && ( !Diagonal ) && at_cmb ) ? 3 : 0 );
        const int surface_shift =
            ( ( at_boundary && treat_boundary_dirichlet && ( !Diagonal ) && at_surface ) ? 3 : 0 );

        for ( int w = 0; w < 2; ++w )
        {
            const int v0 = w == 0 ? n00 : n11;
            const int v1 = w == 0 ? n10 : n01;
            const int v2 = w == 0 ? n01 : n10;

            ArithT k_sum = 0.0;
#pragma unroll
            for ( int node = 0; node < 6; ++node )
            {
                const int nid = node_id( tx + wedge_off( w, node, 0 ), ty + wedge_off( w, node, 1 ) );
                k_sum += k_sh( nid, lvl0 + wedge_off( w, node, 2 ) );
            }
            const ArithT k_eval = static_cast< ArithT >( static_cast< double >( k_sum ) * ( 1.0 / 6.0 ) );

            ArithT kwJ;

            // ==== Phase 1: Jacobian + Gather (gu tensor) ====
            // The inverse Jacobian stays live into the scatter: recomputing it there saved
            // registers, but the kernel is issue-bound and the ~100 instructions cost more.
            ArithT gu00 = 0.0;
            ArithT gu10 = 0.0, gu11 = 0.0;
            ArithT gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
            ArithT div_u = 0.0;
            const GeoT half_dr = static_cast< GeoT >( 0.5 ) * ( r_1 - r_0 );
            const GeoT r_mid   = static_cast< GeoT >( 0.5 ) * ( r_0 + r_1 );

            const GeoT J_0_0 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
            const GeoT J_0_1 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
            const GeoT J_0_2 =
                half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );

            const GeoT J_1_0 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
            const GeoT J_1_1 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
            const GeoT J_1_2 =
                half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );

            const GeoT J_2_0 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
            const GeoT J_2_1 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
            const GeoT J_2_2 =
                half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

            const GeoT J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 -
                                 J_0_1 * J_1_0 * J_2_2 + J_0_1 * J_1_2 * J_2_0 +
                                 J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

            // |J| is geometry, so form the weight in GeoT and narrow once.
            kwJ = static_cast< ArithT >( static_cast< GeoT >( k_eval ) * Kokkos::abs( J_det ) );

            const GeoT inv_det = static_cast< GeoT >( 1.0 ) / J_det;

            const GeoT i00 = inv_det * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
            const GeoT i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
            const GeoT i02 = inv_det * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
            const GeoT i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
            const GeoT i11 = inv_det * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
            const GeoT i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
            const GeoT i20 = inv_det * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
            const GeoT i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
            const GeoT i22 = inv_det * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

            if ( !Diagonal )
            {
#pragma unroll
                for ( int n = 0; n < 6; ++n )
                {
                    // Constant trip count so the node tables fold into immediates; the
                    // Dirichlet rows are skipped by predicate instead.
                    if ( n < cmb_shift || n >= 6 - surface_shift )
                        continue;
                    const GeoT gx = dN_ref[n][0];
                    const GeoT gy = dN_ref[n][1];
                    const GeoT gz = dN_ref[n][2];
                    // Shape-function gradients are geometry: formed in GeoT from
                    // the inverse Jacobian, then narrowed once so the contraction
                    // against the field values runs entirely in ArithT.
                    const GeoT g0 = i00 * gx + i01 * gy + i02 * gz;
                    const GeoT g1 = i10 * gx + i11 * gy + i12 * gz;
                    const GeoT g2 = i20 * gx + i21 * gy + i22 * gz;
                    const ArithT a0 = static_cast< ArithT >( g0 );
                    const ArithT a1 = static_cast< ArithT >( g1 );
                    const ArithT a2 = static_cast< ArithT >( g2 );
                    const ArithT half = static_cast< ArithT >( 0.5 );

                    const int ddx = wedge_off( w, n, 0 );
                    const int ddy = wedge_off( w, n, 1 );
                    const int ddr = wedge_off( w, n, 2 );
                    const int nid = node_id( tx + ddx, ty + ddy );
                    const int lvl = lvl0 + ddr;

                    const ArithT s0 = src_sh( nid, 0, lvl );
                    const ArithT s1 = src_sh( nid, 1, lvl );
                    const ArithT s2 = src_sh( nid, 2, lvl );

                    gu00  += a0 * s0;
                    gu11  += a1 * s1;
                    gu22  += a2 * s2;
                    gu10  += half * ( a1 * s0 + a0 * s1 );
                    gu20  += half * ( a2 * s0 + a0 * s2 );
                    gu21  += half * ( a2 * s1 + a1 * s2 );
                    div_u += a0 * s0 + a1 * s1 + a2 * s2;
                }
            }

            // ==== Phase 2: Scatter, with the same inverse Jacobian ====
            {
                if ( !Diagonal )
                {
                    constexpr double NEG_TWO_THIRDS = -0.66666666666666663;
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        // Constant trip count so the node tables fold into immediates; the
                        // Dirichlet rows are skipped by predicate instead.
                        if ( n < cmb_shift || n >= 6 - surface_shift )
                            continue;
                        const GeoT gx = dN_ref[n][0];
                        const GeoT gy = dN_ref[n][1];
                        const GeoT gz = dN_ref[n][2];
                        const GeoT g0 = i00 * gx + i01 * gy + i02 * gz;
                        const GeoT g1 = i10 * gx + i11 * gy + i12 * gz;
                        const GeoT g2 = i20 * gx + i21 * gy + i22 * gz;
                        const ArithT a0  = static_cast< ArithT >( g0 );
                        const ArithT a1  = static_cast< ArithT >( g1 );
                        const ArithT a2  = static_cast< ArithT >( g2 );
                        const ArithT two = static_cast< ArithT >( 2.0 );
                        const ArithT ntt = static_cast< ArithT >( NEG_TWO_THIRDS );

                        const int ddx = wedge_off( w, n, 0 );
                        const int ddy = wedge_off( w, n, 1 );
                        const int ddr = wedge_off( w, n, 2 );
                        tile_accumulate< TileScatter >(
                            scatter_target< TileScatter >( dst_sh, node_id( tx + ddx, ty + ddy ), 0, lvl0 + ddr, local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr ),
                            static_cast< ScalarType >(
                                kwJ * ( two * ( a0 * gu00 + a1 * gu10 + a2 * gu20 ) + ntt * a0 * div_u ) ) );
                        tile_accumulate< TileScatter >(
                            scatter_target< TileScatter >( dst_sh, node_id( tx + ddx, ty + ddy ), 1, lvl0 + ddr, local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr ),
                            static_cast< ScalarType >(
                                kwJ * ( two * ( a0 * gu10 + a1 * gu11 + a2 * gu21 ) + ntt * a1 * div_u ) ) );
                        tile_accumulate< TileScatter >(
                            scatter_target< TileScatter >( dst_sh, node_id( tx + ddx, ty + ddy ), 2, lvl0 + ddr, local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr ),
                            static_cast< ScalarType >(
                                kwJ * ( two * ( a0 * gu20 + a1 * gu21 + a2 * gu22 ) + ntt * a2 * div_u ) ) );
                    }

                // ---- hourglass stabilization (rank-6 gamma correction) ----
                // One-point quadrature on the wedge leaves two zero-energy modes
                // per element. They are invisible to the operator, so the A-block
                // has a near-null space the V-cycle cannot touch and the outer
                // residual stops descending well above the requested tolerance.
                //
                // Same gamma vectors and the same eps = stab_c * mean(diag)/|g|^2
                // as EpsilonDivDivSimpleStab, but applied matrix-free: the
                // correction is rank 2 per component, so it costs two dot products
                // and a scatter instead of an 18x18 assembly.
                //
                // mean(diag) has a closed form here. The local diagonal entry for
                // node n, component d is kwJ * (gg_n + g_{n,d}^2 / 3), and summing
                // over d gives (10/3) gg_n, so the mean over the 18 entries is
                // kwJ * (10/54) * sum_n gg_n -- no assembly needed for it either.
                if ( stab_c_ > 0.0 )
                {
                    constexpr double G1[6]   = { ONE_THIRD,  -2.0 * ONE_THIRD, ONE_THIRD,
                                                 -ONE_THIRD, 2.0 * ONE_THIRD,  -ONE_THIRD };
                    constexpr double G2[6]   = { ONE_THIRD,  ONE_THIRD,  -2.0 * ONE_THIRD,
                                                 -ONE_THIRD, -ONE_THIRD, 2.0 * ONE_THIRD };
                    constexpr double NORM2_G = 4.0 / 3.0;

                    GeoT sum_gg = 0;
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        const GeoT gx = dN_ref[n][0];
                        const GeoT gy = dN_ref[n][1];
                        const GeoT gz = dN_ref[n][2];
                        const GeoT b0 = i00 * gx + i01 * gy + i02 * gz;
                        const GeoT b1 = i10 * gx + i11 * gy + i12 * gz;
                        const GeoT b2 = i20 * gx + i21 * gy + i22 * gz;
                        sum_gg += b0 * b0 + b1 * b1 + b2 * b2;
                    }
                    // The scaling is geometry, so it is formed in GeoT and
                    // narrowed once; everything that touches the field values then
                    // runs in ArithT like the rest of this kernel.
                    const GeoT diag_mean = static_cast< GeoT >( kwJ ) * static_cast< GeoT >( 10.0 / 54.0 ) * sum_gg;
                    const ArithT eps       = static_cast< ArithT >( static_cast< GeoT >( stab_c_ ) * diag_mean / static_cast< GeoT >( NORM2_G ) );

                    for ( int d = 0; d < 3; ++d )
                    {
                        ArithT c1 = 0.0, c2 = 0.0;
#pragma unroll
                        for ( int n = 0; n < 6; ++n )
                        {
                            // Constant trip count so the node tables fold into immediates; the
                            // Dirichlet rows are skipped by predicate instead.
                            if ( n < cmb_shift || n >= 6 - surface_shift )
                                continue;
                            const int nid =
                                node_id( tx + wedge_off( w, n, 0 ), ty + wedge_off( w, n, 1 ) );
                            const int    lvl = lvl0 + wedge_off( w, n, 2 );
                            const ArithT u   = src_sh( nid, d, lvl );
                            c1 += static_cast< ArithT >( G1[n] ) * u;
                            c2 += static_cast< ArithT >( G2[n] ) * u;
                        }
#pragma unroll
                        for ( int n = 0; n < 6; ++n )
                        {
                            // Constant trip count so the node tables fold into immediates; the
                            // Dirichlet rows are skipped by predicate instead.
                            if ( n < cmb_shift || n >= 6 - surface_shift )
                                continue;
                            const int ddx = wedge_off( w, n, 0 );
                            const int ddy = wedge_off( w, n, 1 );
                            const int ddr = wedge_off( w, n, 2 );
                            tile_accumulate< TileScatter >(
                                scatter_target< TileScatter >( dst_sh, node_id( tx + ddx, ty + ddy ), d, lvl0 + ddr, local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr ),
                                static_cast< ScalarType >( eps * ( static_cast< ArithT >( G1[n] ) * c1 +
                                                                   static_cast< ArithT >( G2[n] ) * c2 ) ) );
                        }
                    }
                }
                }

                if ( Diagonal || ( treat_boundary_dirichlet && at_boundary ) )
                {
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        if ( n < surface_shift || n >= 6 - cmb_shift )
                            continue;
                        const GeoT gx = dN_ref[n][0];
                        const GeoT gy = dN_ref[n][1];
                        const GeoT gz = dN_ref[n][2];
                        const GeoT g0 = i00 * gx + i01 * gy + i02 * gz;
                        const GeoT g1 = i10 * gx + i11 * gy + i12 * gz;
                        const GeoT g2 = i20 * gx + i21 * gy + i22 * gz;
                        const GeoT gg = g0 * g0 + g1 * g1 + g2 * g2;
                        // Purely geometric row factors, narrowed once each.
                        const ArithT d0 = static_cast< ArithT >( gg + ONE_THIRD * g0 * g0 );
                        const ArithT d1 = static_cast< ArithT >( gg + ONE_THIRD * g1 * g1 );
                        const ArithT d2 = static_cast< ArithT >( gg + ONE_THIRD * g2 * g2 );

                        const int nid = node_id( tx + wedge_off( w, n, 0 ), ty + wedge_off( w, n, 1 ) );
                        const int lvl = lvl0 + wedge_off( w, n, 2 );

                        const ArithT sv0 = src_sh( nid, 0, lvl );
                        const ArithT sv1 = src_sh( nid, 1, lvl );
                        const ArithT sv2 = src_sh( nid, 2, lvl );

                        const int ddx = wedge_off( w, n, 0 );
                        const int ddy = wedge_off( w, n, 1 );
                        const int ddr = wedge_off( w, n, 2 );
                        tile_accumulate< TileScatter >(
                            scatter_target< TileScatter >( dst_sh, node_id( tx + ddx, ty + ddy ), 0, lvl0 + ddr, local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr ),
                            static_cast< ScalarType >( kwJ * sv0 * d0 ) );
                        tile_accumulate< TileScatter >(
                            scatter_target< TileScatter >( dst_sh, node_id( tx + ddx, ty + ddy ), 1, lvl0 + ddr, local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr ),
                            static_cast< ScalarType >( kwJ * sv1 * d1 ) );
                        tile_accumulate< TileScatter >(
                            scatter_target< TileScatter >( dst_sh, node_id( tx + ddx, ty + ddy ), 2, lvl0 + ddr, local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr ),
                            static_cast< ScalarType >( kwJ * sv2 * d2 ) );
                    }

                    // Diagonal of the stabilization itself, so the Chebyshev
                    // inverse diagonal matches the operator it smooths.
                    if ( Diagonal && stab_c_ > 0.0 )
                    {
                        constexpr double G1[6]   = { ONE_THIRD,  -2.0 * ONE_THIRD, ONE_THIRD,
                                                     -ONE_THIRD, 2.0 * ONE_THIRD,  -ONE_THIRD };
                        constexpr double G2[6]   = { ONE_THIRD,  ONE_THIRD,  -2.0 * ONE_THIRD,
                                                     -ONE_THIRD, -ONE_THIRD, 2.0 * ONE_THIRD };
                        constexpr double NORM2_G = 4.0 / 3.0;

                        GeoT sum_gg = 0;
#pragma unroll
                        for ( int n = 0; n < 6; ++n )
                        {
                            const GeoT gx = dN_ref[n][0];
                            const GeoT gy = dN_ref[n][1];
                            const GeoT gz = dN_ref[n][2];
                            const GeoT b0 = i00 * gx + i01 * gy + i02 * gz;
                            const GeoT b1 = i10 * gx + i11 * gy + i12 * gz;
                            const GeoT b2 = i20 * gx + i21 * gy + i22 * gz;
                            sum_gg += b0 * b0 + b1 * b1 + b2 * b2;
                        }
                        const GeoT diag_mean = static_cast< GeoT >( kwJ ) * static_cast< GeoT >( 10.0 / 54.0 ) * sum_gg;
                        const ArithT eps       = static_cast< ArithT >( static_cast< GeoT >( stab_c_ ) * diag_mean / static_cast< GeoT >( NORM2_G ) );
#pragma unroll
                        for ( int n = 0; n < 6; ++n )
                        {
                            if ( n < surface_shift || n >= 6 - cmb_shift )
                                continue;
                            const ArithT gdiag =
                                static_cast< ArithT >( G1[n] * G1[n] + G2[n] * G2[n] );
                            const int ddx = wedge_off( w, n, 0 );
                            const int ddy = wedge_off( w, n, 1 );
                            const int ddr = wedge_off( w, n, 2 );
                            for ( int d = 0; d < 3; ++d )
                                tile_accumulate< TileScatter >(
                                    scatter_target< TileScatter >( dst_sh, node_id( tx + ddx, ty + ddy ), d, lvl0 + ddr, local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr ),
                                    static_cast< ScalarType >( eps * gdiag ) );
                        }
                    }
                }
            }

        } // end wedge loop
    }

    // Flush the tile. One global atomic per node and component, against one per
    // wedge-node before: for a 4/8/1 tile that is 675 instead of ~4600. It is also
    // the only place a 16-bit field precision is narrowed on the way out.
    if constexpr ( TileScatter )
    {
        team.team_barrier();

        if constexpr ( sizeof( ScalarType ) == 2 )
        {
            // A 2-byte field has no lock-free scalar atomic, and the level axis is the
            // fastest index of each component view, so consecutive levels share one
            // 32-bit word: a scalar CAS on one half fights the thread on the other
            // half (measured 5x the requests of one per node, 70% branch efficiency).
            // So the flush walks the words of each node's level run instead -- one
            // native packed reduction per word, a zero for any half outside the tile
            // (the one before the run at an odd start, the one after it at the end:
            // both belong to a neighbour and adding zero leaves them as they are).
            const int nwords = nlev / 2 + 1;
            Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy * 3 * nwords ), [&]( int t ) {
                const int w    = t % nwords;
                const int d    = ( t / nwords ) % 3;
                const int node = t / ( 3 * nwords );
                const int xi   = x0 + node % ( lat_tile + 1 );
                const int yi   = y0 + node / ( lat_tile + 1 );

                if ( xi > hex_lat_ || yi > hex_lat_ )
                    return;

                const int   nlev_valid = Kokkos::min( nlev, hex_rad_ - r0 + 1 );
                ScalarType* run        = &dst_( local_subdomain_id, xi, yi, r0, d );
                const int   odd        = ( reinterpret_cast< std::uintptr_t >( run ) & 2u ) ? 1 : 0;
                const int   l0         = 2 * w - odd;
                if ( l0 >= nlev_valid )
                    return;

                const AccT v0 = ( l0 >= 0 ) ? dst_sh( node, d, l0 ) : AccT( 0 );
                const AccT v1 = ( l0 + 1 < nlev_valid ) ? dst_sh( node, d, l0 + 1 ) : AccT( 0 );
                if ( v0 == AccT( 0 ) && v1 == AccT( 0 ) )
                    return;

                util::atomic_add_packed_pair( run + l0, static_cast< float >( v0 ), static_cast< float >( v1 ) );
            } );
        }
        else
        {
            Kokkos::parallel_for( Kokkos::TeamThreadRange( team, total_pairs ), [&]( int t ) {
                const int node = t / nlev;
                const int lvl  = t - node * nlev;
                const int xi   = x0 + node % ( lat_tile + 1 );
                const int yi   = y0 + node / ( lat_tile + 1 );
                const int rr   = r0 + lvl;

                if ( xi > hex_lat_ || yi > hex_lat_ || rr > hex_rad_ )
                    return;

                for ( int d = 0; d < 3; ++d )
                {
                    const AccT v = dst_sh( node, d, lvl );
                    if ( v != AccT( 0 ) )
                        util::atomic_add_lockfree( &dst_( local_subdomain_id, xi, yi, rr, d ),
                                                   static_cast< ScalarType >( v ) );
                }
            } );
        }
    }
}

} // namespace terra::fe::wedge::operators::shell
