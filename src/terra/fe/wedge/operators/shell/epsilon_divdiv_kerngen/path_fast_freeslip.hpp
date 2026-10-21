#pragma once

// Fast free-slip path of EpsilonDivDivKerngen: matrix-free with the normal/tangential
//
// projection applied on the fly at free-slip boundary nodes. Scratch tiles are double.

#include "../epsilon_divdiv_kerngen.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, int VecDim, typename CoordScalarT, typename ArithScalarT, typename MatrixScalarT >
template < bool Diagonal >
KOKKOS_INLINE_FUNCTION void
EpsilonDivDivKerngen< ScalarT, VecDim, CoordScalarT, ArithScalarT, MatrixScalarT >::operator_fast_freeslip_path(
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
    const int nlev = r_tile_block_ + 1;
    const int nxy  = ( lat_tile_ + 1 ) * ( lat_tile_ + 1 );

    double* shmem =
        reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size( team.team_size() ) ) );

    using ScratchCoords =
        Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
    using ScratchSrc = Kokkos::
        View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
    using ScratchK =
        Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

    ScratchCoords coords_sh( shmem, nxy, 3 );
    shmem += nxy * 3;

    // Normalized normals in shared memory — avoids 12 persistent register-doubles.
    ScratchCoords normals_sh( shmem, nxy, 3 );
    shmem += nxy * 3;

    ScratchSrc src_sh( shmem, nxy, 3, nlev );
    shmem += nxy * 3 * nlev;

    ScratchK k_sh( shmem, nxy, nlev );
    shmem += nxy * nlev;

    auto r_sh =
        Kokkos::View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >(
            shmem, nlev );

    auto node_id = [&]( int nx, int ny ) -> int { return nx + ( lat_tile_ + 1 ) * ny; };

    // Preload coords + compute normalized normals cooperatively.
    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy ), [&]( int n ) {
        const int dxn = n % ( lat_tile_ + 1 );
        const int dyn = n / ( lat_tile_ + 1 );
        const int xi  = x0 + dxn;
        const int yi  = y0 + dyn;

        if ( xi <= hex_lat_ && yi <= hex_lat_ )
        {
            // This free-slip path stages into double scratch whatever the
            // template precisions are, so the only conversion needed is widening
            // the GeomT grid values on the way in -- the 16-bit wrappers have no
            // implicit conversion to double.
            const double cx = static_cast< double >( grid_( local_subdomain_id, xi, yi, 0 ) );
            const double cy = static_cast< double >( grid_( local_subdomain_id, xi, yi, 1 ) );
            const double cz = static_cast< double >( grid_( local_subdomain_id, xi, yi, 2 ) );
            coords_sh( n, 0 ) = cx;
            coords_sh( n, 1 ) = cy;
            coords_sh( n, 2 ) = cz;

            const double n2 = cx * cx + cy * cy + cz * cz;
            if ( n2 > 0.0 )
            {
                const double invn = 1.0 / Kokkos::sqrt( n2 );
                normals_sh( n, 0 ) = cx * invn;
                normals_sh( n, 1 ) = cy * invn;
                normals_sh( n, 2 ) = cz * invn;
            }
            else
            {
                normals_sh( n, 0 ) = normals_sh( n, 1 ) = normals_sh( n, 2 ) = 0.0;
            }
        }
        else
        {
            coords_sh( n, 0 ) = coords_sh( n, 1 ) = coords_sh( n, 2 ) = 0.0;
            normals_sh( n, 0 ) = normals_sh( n, 1 ) = normals_sh( n, 2 ) = 0.0;
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

        const int dxn = node % ( lat_tile_ + 1 );
        const int dyn = node / ( lat_tile_ + 1 );

        const int xi = x0 + dxn;
        const int yi = y0 + dyn;
        const int rr = r0 + lvl;

        if ( xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_ )
        {
            // The team scratch tiles are double regardless of ScalarT (see
            // team_shmem_size), so the gather from the stored fields is the one
            // place the field precision is widened. The 16-bit Kokkos types only
            // convert out of themselves explicitly, hence the casts.
            k_sh( node, lvl )      = static_cast< double >( k_( local_subdomain_id, xi, yi, rr ) );
            src_sh( node, 0, lvl ) = static_cast< double >( src_( local_subdomain_id, xi, yi, rr, 0 ) );
            src_sh( node, 1, lvl ) = static_cast< double >( src_( local_subdomain_id, xi, yi, rr, 1 ) );
            src_sh( node, 2, lvl ) = static_cast< double >( src_( local_subdomain_id, xi, yi, rr, 2 ) );
        }
        else
        {
            k_sh( node, lvl )      = 0.0;
            src_sh( node, 0, lvl ) = src_sh( node, 1, lvl ) = src_sh( node, 2, lvl ) = 0.0;
        }
    } );

    team.team_barrier();

    if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ )
        return;

    for ( int pass = 0; pass < r_passes_; ++pass )
    {
    const int    lvl0   = pass * r_tile_ + tr;
    const int    r_cell = r0 + lvl0;

    if ( r_cell >= hex_rad_ )
        break;

    const double r_0  = r_sh( lvl0 );
    const double r_1  = r_sh( lvl0 + 1 );

    const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
    const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

    const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
    const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );

    const bool cmb_freeslip      = at_cmb && ( cmb_bc == FREESLIP );
    const bool surf_freeslip     = at_surface && ( surface_bc == FREESLIP );
    const bool cmb_dirichlet     = at_cmb && ( cmb_bc == DIRICHLET );
    const bool surface_dirichlet = at_surface && ( surface_bc == DIRICHLET );

    const int cmb_shift     = ( ( cmb_dirichlet && ( !Diagonal ) ) ? 3 : 0 );
    const int surface_shift = ( ( surface_dirichlet && ( !Diagonal ) ) ? 3 : 0 );

    static constexpr int WEDGE_NODE_OFF[2][6][3] = {
        { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
        { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

    static constexpr int WEDGE_TO_UNIQUE[2][6] = {
        { 0, 1, 2, 3, 4, 5 },
        { 6, 2, 1, 7, 5, 4 }
    };

    constexpr double ONE_THIRD      = 1.0 / 3.0;
    constexpr double ONE_SIXTH      = 1.0 / 6.0;
    constexpr double NEG_TWO_THIRDS = -0.66666666666666663;

    static constexpr double dN_ref[6][3] = {
        { -0.5, -0.5, -ONE_SIXTH },
        { 0.5, 0.0, -ONE_SIXTH },
        { 0.0, 0.5, -ONE_SIXTH },
        { -0.5, -0.5, ONE_SIXTH },
        { 0.5, 0.0, ONE_SIXTH },
        { 0.0, 0.5, ONE_SIXTH } };

    const int n00 = node_id( tx, ty );
    const int n01 = node_id( tx, ty + 1 );
    const int n10 = node_id( tx + 1, ty );
    const int n11 = node_id( tx + 1, ty + 1 );

    // Corner-to-shared-memory-node mapping: 0→n00, 1→n10, 2→n01, 3→n11.
    const int corner_node[4] = { n00, n10, n01, n11 };
    // Corner-to-unique-node for CMB (r=0) and surface (r=1) layers.
    static constexpr int CMB_CORNER_TO_UNIQUE[4]  = { 0, 1, 2, 6 };
    static constexpr int SURF_CORNER_TO_UNIQUE[4] = { 3, 4, 5, 7 };

    double dst8[3][8] = { { 0.0 } };

    // Scalar Ann accumulators per corner.  Accumulated inside the test-side loop
    // (merged from the former separate Ann loop to reuse already-computed gradients).
    double Ann_acc_cmb[4]  = {};
    double Ann_acc_surf[4] = {};

    for ( int w = 0; w < 2; ++w )
    {
        const int v0 = w == 0 ? n00 : n11;
        const int v1 = w == 0 ? n10 : n01;
        const int v2 = w == 0 ? n01 : n10;

        double k_sum = 0.0;
#pragma unroll
        for ( int node = 0; node < 6; ++node )
        {
            const int ddx = WEDGE_NODE_OFF[w][node][0];
            const int ddy = WEDGE_NODE_OFF[w][node][1];
            const int ddr = WEDGE_NODE_OFF[w][node][2];

            const int nid = node_id( tx + ddx, ty + ddy );
            const int lvl = lvl0 + ddr;
            k_sum += k_sh( nid, lvl );
        }
        const double k_eval = ONE_SIXTH * k_sum;

        double wJ = 0.0;
        double i00, i01, i02;
        double i10, i11, i12;
        double i20, i21, i22;

        {
            const double half_dr = 0.5 * ( r_1 - r_0 );
            const double r_mid   = 0.5 * ( r_0 + r_1 );

            const double J_0_0 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
            const double J_0_1 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
            const double J_0_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );

            const double J_1_0 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
            const double J_1_1 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
            const double J_1_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );

            const double J_2_0 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
            const double J_2_1 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
            const double J_2_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

            const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                 J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

            const double invJ = 1.0 / J_det;

            i00 = invJ * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
            i01 = invJ * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
            i02 = invJ * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );

            i10 = invJ * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
            i11 = invJ * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
            i12 = invJ * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );

            i20 = invJ * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
            i21 = invJ * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
            i22 = invJ * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

            wJ = Kokkos::abs( J_det );
        }

        const double kwJ = k_eval * wJ;

        // ---- Fused trial + test side with merged Ann accumulation ----
        static constexpr int CMB_NODE_TO_CORNER[2][3] = { { 0, 1, 2 }, { 3, 2, 1 } };

        double gu00 = 0.0;
        double gu10 = 0.0, gu11 = 0.0;
        double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
        double div_u = 0.0;

        if ( !Diagonal )
        {
            // Trial side: accumulate symmetric gradient of u (fused dim loops).
            // Read directly from shared memory with inline tangential projection for freeslip nodes.
#pragma unroll
            for ( int n = cmb_shift; n < 6 - surface_shift; ++n )
            {
                const double gx = dN_ref[n][0];
                const double gy = dN_ref[n][1];
                const double gz = dN_ref[n][2];
                const double g0 = i00 * gx + i01 * gy + i02 * gz;
                const double g1 = i10 * gx + i11 * gy + i12 * gz;
                const double g2 = i20 * gx + i21 * gy + i22 * gz;

                const int nid = node_id( tx + WEDGE_NODE_OFF[w][n][0], ty + WEDGE_NODE_OFF[w][n][1] );
                const int lvl = lvl0 + WEDGE_NODE_OFF[w][n][2];

                double s0 = src_sh( nid, 0, lvl );
                double s1 = src_sh( nid, 1, lvl );
                double s2 = src_sh( nid, 2, lvl );

                // Inline tangential projection for freeslip boundary nodes.
                if ( cmb_freeslip && n < 3 )
                {
                    const double nx  = normals_sh( nid, 0 );
                    const double ny  = normals_sh( nid, 1 );
                    const double nz  = normals_sh( nid, 2 );
                    const double dot = nx * s0 + ny * s1 + nz * s2;
                    s0 -= dot * nx;
                    s1 -= dot * ny;
                    s2 -= dot * nz;
                }
                if ( surf_freeslip && n >= 3 )
                {
                    const double nx  = normals_sh( nid, 0 );
                    const double ny  = normals_sh( nid, 1 );
                    const double nz  = normals_sh( nid, 2 );
                    const double dot = nx * s0 + ny * s1 + nz * s2;
                    s0 -= dot * nx;
                    s1 -= dot * ny;
                    s2 -= dot * nz;
                }

                gu00  += g0 * s0;
                gu11  += g1 * s1;
                gu22  += g2 * s2;
                gu10  += 0.5 * ( g1 * s0 + g0 * s1 );
                gu20  += 0.5 * ( g2 * s0 + g0 * s2 );
                gu21  += 0.5 * ( g2 * s1 + g1 * s2 );
                div_u += g0 * s0 + g1 * s1 + g2 * s2;
            }

            // Test side + merged Ann accumulation.
            // Ann uses the same gradient already computed — no separate loop needed.
#pragma unroll
            for ( int n = cmb_shift; n < 6 - surface_shift; ++n )
            {
                const double gx = dN_ref[n][0];
                const double gy = dN_ref[n][1];
                const double gz = dN_ref[n][2];
                const double g0 = i00 * gx + i01 * gy + i02 * gz;
                const double g1 = i10 * gx + i11 * gy + i12 * gz;
                const double g2 = i20 * gx + i21 * gy + i22 * gz;

                const int uid = WEDGE_TO_UNIQUE[w][n];
                dst8[0][uid] +=
                    kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) + NEG_TWO_THIRDS * g0 * div_u );
                dst8[1][uid] +=
                    kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) + NEG_TWO_THIRDS * g1 * div_u );
                dst8[2][uid] +=
                    kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) + NEG_TWO_THIRDS * g2 * div_u );

                // Accumulate Ann for freeslip CMB nodes (n < 3) and surface nodes (n >= 3).
                if ( cmb_freeslip && n < 3 )
                {
                    const int    corner = CMB_NODE_TO_CORNER[w][n];
                    const int    cn     = corner_node[corner];
                    const double nxu    = normals_sh( cn, 0 );
                    const double nyu    = normals_sh( cn, 1 );
                    const double nzu    = normals_sh( cn, 2 );
                    const double gg     = g0 * g0 + g1 * g1 + g2 * g2;
                    const double ng     = nxu * g0 + nyu * g1 + nzu * g2;
                    Ann_acc_cmb[corner] += kwJ * ( gg + ONE_THIRD * ng * ng );
                }
                if ( surf_freeslip && n >= 3 )
                {
                    const int    corner = CMB_NODE_TO_CORNER[w][n - 3];
                    const int    cn     = corner_node[corner];
                    const double nxu    = normals_sh( cn, 0 );
                    const double nyu    = normals_sh( cn, 1 );
                    const double nzu    = normals_sh( cn, 2 );
                    const double gg     = g0 * g0 + g1 * g1 + g2 * g2;
                    const double ng     = nxu * g0 + nyu * g1 + nzu * g2;
                    Ann_acc_surf[corner] += kwJ * ( gg + ONE_THIRD * ng * ng );
                }
            }
        }

        // ---- Diagonal / Dirichlet boundary handling (fused) ----
        if ( Diagonal || cmb_dirichlet || surface_dirichlet )
        {
            // Fused diagonal: kwJ * s_d * (|g|^2 + (1/3) * g_d^2)
#pragma unroll
            for ( int n = surface_shift; n < 6 - cmb_shift; ++n )
            {
                if ( Diagonal && cmb_freeslip && n < 3 )
                    continue;
                if ( Diagonal && surf_freeslip && n >= 3 )
                    continue;

                const double gx = dN_ref[n][0];
                const double gy = dN_ref[n][1];
                const double gz = dN_ref[n][2];
                const double g0 = i00 * gx + i01 * gy + i02 * gz;
                const double g1 = i10 * gx + i11 * gy + i12 * gz;
                const double g2 = i20 * gx + i21 * gy + i22 * gz;
                const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                const int    uid = WEDGE_TO_UNIQUE[w][n];
                const int    nid = node_id( tx + WEDGE_NODE_OFF[w][n][0], ty + WEDGE_NODE_OFF[w][n][1] );
                const int    lvl = lvl0 + WEDGE_NODE_OFF[w][n][2];
                const double s0  = src_sh( nid, 0, lvl );
                const double s1  = src_sh( nid, 1, lvl );
                const double s2  = src_sh( nid, 2, lvl );

                dst8[0][uid] += kwJ * s0 * ( gg + ONE_THIRD * g0 * g0 );
                dst8[1][uid] += kwJ * s1 * ( gg + ONE_THIRD * g1 * g1 );
                dst8[2][uid] += kwJ * s2 * ( gg + ONE_THIRD * g2 * g2 );
            }

            // For free-slip boundary nodes in diagonal mode: compute R^T diag(R A_3x3 R^T) R src.
            // Normals loaded from shared memory, u_n recomputed from src_sh (original, unprojected).
            if ( Diagonal )
            {
                static constexpr int FS_CORNER_MAP[2][3] = { { 0, 1, 2 }, { 3, 2, 1 } };

                auto apply_rotated_diag =
                    [&]( const int ni, const int node_idx, const int src_lvl ) {
                        const int    corner = FS_CORNER_MAP[w][ni];
                        const int    cn     = corner_node[corner];
                        const double nxu    = normals_sh( cn, 0 );
                        const double nyu    = normals_sh( cn, 1 );
                        const double nzu    = normals_sh( cn, 2 );
                        const int    u      = WEDGE_TO_UNIQUE[w][node_idx];

                        const double gx = dN_ref[node_idx][0];
                        const double gy = dN_ref[node_idx][1];
                        const double gz = dN_ref[node_idx][2];

                        const double g0     = i00 * gx + i01 * gy + i02 * gz;
                        const double g1     = i10 * gx + i11 * gy + i12 * gz;
                        const double g2     = i20 * gx + i21 * gy + i22 * gz;
                        const double gg_loc = g0 * g0 + g1 * g1 + g2 * g2;

                        dense::Vec< double, 3 > n_vec;
                        n_vec( 0 )      = nxu;
                        n_vec( 1 )      = nyu;
                        n_vec( 2 )      = nzu;
                        const auto R_rot = trafo_mat_cartesian_to_normal_tangential< double >( n_vec );

                        // Read original (unprojected) src directly from shared memory.
                        // (Previously: projected src8 + u_n*n = original, now simplified.)
                        const double s0 = src_sh( cn, 0, src_lvl );
                        const double s1 = src_sh( cn, 1, src_lvl );
                        const double s2 = src_sh( cn, 2, src_lvl );

                        const double Rg0 = R_rot( 0, 0 ) * g0 + R_rot( 0, 1 ) * g1 + R_rot( 0, 2 ) * g2;
                        const double Rg1 = R_rot( 1, 0 ) * g0 + R_rot( 1, 1 ) * g1 + R_rot( 1, 2 ) * g2;
                        const double Rg2 = R_rot( 2, 0 ) * g0 + R_rot( 2, 1 ) * g1 + R_rot( 2, 2 ) * g2;
                        const double Rs0 = R_rot( 0, 0 ) * s0 + R_rot( 0, 1 ) * s1 + R_rot( 0, 2 ) * s2;
                        const double Rs1 = R_rot( 1, 0 ) * s0 + R_rot( 1, 1 ) * s1 + R_rot( 1, 2 ) * s2;
                        const double Rs2 = R_rot( 2, 0 ) * s0 + R_rot( 2, 1 ) * s1 + R_rot( 2, 2 ) * s2;

                        const double v0 = kwJ * ( gg_loc + ONE_THIRD * Rg0 * Rg0 ) * Rs0;
                        const double v1 = kwJ * ( gg_loc + ONE_THIRD * Rg1 * Rg1 ) * Rs1;
                        const double v2 = kwJ * ( gg_loc + ONE_THIRD * Rg2 * Rg2 ) * Rs2;

                        dst8[0][u] += R_rot( 0, 0 ) * v0 + R_rot( 1, 0 ) * v1 + R_rot( 2, 0 ) * v2;
                        dst8[1][u] += R_rot( 0, 1 ) * v0 + R_rot( 1, 1 ) * v1 + R_rot( 2, 1 ) * v2;
                        dst8[2][u] += R_rot( 0, 2 ) * v0 + R_rot( 1, 2 ) * v1 + R_rot( 2, 2 ) * v2;
                    };

                if ( cmb_freeslip )
                {
                    for ( int ni = 0; ni < 3; ++ni )
                        apply_rotated_diag( ni, ni, lvl0 );
                }
                if ( surf_freeslip )
                {
                    for ( int ni = 0; ni < 3; ++ni )
                        apply_rotated_diag( ni, ni + 3, lvl0 + 1 );
                }
            }
        }
    }

    // Test-side projection for free-slip (P A P) — normals loaded from shared memory.
    if ( !Diagonal && cmb_freeslip )
    {
        for ( int c = 0; c < 4; ++c )
        {
            const double nx = normals_sh( corner_node[c], 0 );
            const double ny = normals_sh( corner_node[c], 1 );
            const double nz = normals_sh( corner_node[c], 2 );
            const int    u  = CMB_CORNER_TO_UNIQUE[c];
            const double dot = nx * dst8[0][u] + ny * dst8[1][u] + nz * dst8[2][u];
            dst8[0][u] -= dot * nx;
            dst8[1][u] -= dot * ny;
            dst8[2][u] -= dot * nz;
        }
    }
    if ( !Diagonal && surf_freeslip )
    {
        for ( int c = 0; c < 4; ++c )
        {
            const double nx = normals_sh( corner_node[c], 0 );
            const double ny = normals_sh( corner_node[c], 1 );
            const double nz = normals_sh( corner_node[c], 2 );
            const int    u  = SURF_CORNER_TO_UNIQUE[c];
            const double dot = nx * dst8[0][u] + ny * dst8[1][u] + nz * dst8[2][u];
            dst8[0][u] -= dot * nx;
            dst8[1][u] -= dot * ny;
            dst8[2][u] -= dot * nz;
        }
    }

    // Add back normal correction: Ann_acc[c] * u_n[c] * n_c.
    // u_n recomputed from original (unprojected) src in shared memory.
    if ( !Diagonal && cmb_freeslip )
    {
        for ( int c = 0; c < 4; ++c )
        {
            const int    cn  = corner_node[c];
            const double nx  = normals_sh( cn, 0 );
            const double ny  = normals_sh( cn, 1 );
            const double nz  = normals_sh( cn, 2 );
            const double os0 = src_sh( cn, 0, lvl0 );
            const double os1 = src_sh( cn, 1, lvl0 );
            const double os2 = src_sh( cn, 2, lvl0 );
            const double u_n_val = nx * os0 + ny * os1 + nz * os2;
            const double corr    = Ann_acc_cmb[c] * u_n_val;
            const int    u       = CMB_CORNER_TO_UNIQUE[c];
            dst8[0][u] += corr * nx;
            dst8[1][u] += corr * ny;
            dst8[2][u] += corr * nz;
        }
    }
    if ( !Diagonal && surf_freeslip )
    {
        for ( int c = 0; c < 4; ++c )
        {
            const int    cn  = corner_node[c];
            const double nx  = normals_sh( cn, 0 );
            const double ny  = normals_sh( cn, 1 );
            const double nz  = normals_sh( cn, 2 );
            const double os0 = src_sh( cn, 0, lvl0 + 1 );
            const double os1 = src_sh( cn, 1, lvl0 + 1 );
            const double os2 = src_sh( cn, 2, lvl0 + 1 );
            const double u_n_val = nx * os0 + ny * os1 + nz * os2;
            const double corr    = Ann_acc_surf[c] * u_n_val;
            const int    u       = SURF_CORNER_TO_UNIQUE[c];
            dst8[0][u] += corr * nx;
            dst8[1][u] += corr * ny;
            dst8[2][u] += corr * nz;
        }
    }

    // Scatter accumulated hex-cell contributions to global memory.
#pragma unroll
    for ( int dim_add = 0; dim_add < 3; ++dim_add )
    {
        util::atomic_add_lockfree( &dst_( local_subdomain_id, x_cell, y_cell, r_cell, dim_add ), dst8[dim_add][0] );
        util::atomic_add_lockfree( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell, dim_add ), dst8[dim_add][1] );
        util::atomic_add_lockfree( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell, dim_add ), dst8[dim_add][2] );
        util::atomic_add_lockfree( &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1, dim_add ), dst8[dim_add][3] );
        util::atomic_add_lockfree(
            &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim_add ), dst8[dim_add][4] );
        util::atomic_add_lockfree(
            &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][5] );
        util::atomic_add_lockfree(
            &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim_add ), dst8[dim_add][6] );
        util::atomic_add_lockfree(
            &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][7] );
    }

    } // end r_passes loop
}

} // namespace terra::fe::wedge::operators::shell
