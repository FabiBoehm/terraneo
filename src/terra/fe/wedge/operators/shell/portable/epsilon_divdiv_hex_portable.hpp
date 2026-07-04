// Portable hex 2x2x2 Gauss Dirichlet/Neumann kernel for EpsilonDivDivKerngen.
//
// This file is included from inside the class body of
// `terra::fe::wedge::operators::shell::EpsilonDivDivKerngen<ScalarT, VecDim>`,
// so the declaration below is a class member.
//
// It is a backend-agnostic (CUDA / host / any Kokkos backend) re-expression of
// the AMD-wavefront kernel in hip/epsilon_divdiv_hex.hpp.  Same physics: full
// 2x2x2 Gauss integration of the eps-div-div (deviatoric viscous) operator on
// the trilinear Q1 hex (no wedge subdivision), Dirichlet/Neumann boundaries.
//
// The HIP kernel maps the 8 hex corners onto 8 wavefront lanes and forms the
// strain eps(u)(q) with `wave_reduce8` cross-lane sums.  Here a single thread
// owns one hex cell and the per-corner sums are ordinary serial loops, so there
// are no __shfl_xor calls and no 64-lane assumption.  One thread per hex cell;
// shared corner nodes are accumulated with atomic_add, exactly as the wave path
// scatters.  Slower than the hand-tuned wave kernel, but correct everywhere and
// sufficient for convergence studies.
//
// Quadrature: Gauss-Legendre 2-pt on [0,1] (nodes 1/2 +/- 1/(2 sqrt 3), weight
// 1/2); the 2x2x2 product rule gives 8 points of weight 1/8 on [0,1]^3.

#pragma once

template < bool Diagonal >
void run_hex_portable() const
{
    const int    n_cells    = local_subdomains_ * hex_lat_ * hex_lat_ * hex_rad_;
    const int    per_subdom = hex_lat_ * hex_lat_ * hex_rad_;
    const int    lat        = hex_lat_;
    const int    rad        = hex_rad_;

    Kokkos::parallel_for(
        "epsilon_divdiv_apply_kernel_hex_portable",
        Kokkos::RangePolicy<>( 0, n_cells ),
        KOKKOS_CLASS_LAMBDA( const int idx ) {
            // ---------- decode flat index -> (subdomain, x_cell, y_cell, r_cell) ----------
            int       rk                 = idx;
            const int local_subdomain_id = rk / per_subdom;
            rk                           = rk % per_subdom;
            const int r_cell             = rk % rad;
            rk                           = rk / rad;
            const int y_cell             = rk % lat;
            const int x_cell             = rk / lat;

            // ---------- gather the 4 lateral corner unit-sphere coords ----------
            double cs[4][3];
            for ( int n = 0; n < 4; ++n )
            {
                const int dx = n & 1;
                const int dy = ( n >> 1 ) & 1;
                cs[n][0]     = grid_( local_subdomain_id, x_cell + dx, y_cell + dy, 0 );
                cs[n][1]     = grid_( local_subdomain_id, x_cell + dx, y_cell + dy, 1 );
                cs[n][2]     = grid_( local_subdomain_id, x_cell + dx, y_cell + dy, 2 );
            }

            const double r_0 = radii_( local_subdomain_id, r_cell );
            const double r_1 = radii_( local_subdomain_id, r_cell + 1 );
            const double dr  = r_1 - r_0;

            // ---------- gather the 8 hex-corner src + coefficient ----------
            // corner node index: bit0 = dxn, bit1 = dyn, bit2 = dzn
            double s[8][3];
            double kc[8];
            for ( int node = 0; node < 8; ++node )
            {
                const int dxn = node & 1;
                const int dyn = ( node >> 1 ) & 1;
                const int dzn = ( node >> 2 ) & 1;
                const int xi  = x_cell + dxn;
                const int yi  = y_cell + dyn;
                const int rr  = r_cell + dzn;
                s[node][0]    = src_( local_subdomain_id, xi, yi, rr, 0 );
                s[node][1]    = src_( local_subdomain_id, xi, yi, rr, 1 );
                s[node][2]    = src_( local_subdomain_id, xi, yi, rr, 2 );
                kc[node]      = k_( local_subdomain_id, xi, yi, rr );
            }

            // ---------- boundary state (uniform across the cell's corners) ----------
            const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
            const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );
            const bool at_boundary = at_cmb || at_surface;
            bool       treat_boundary_dirichlet = false;
            if ( at_boundary )
            {
                const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
                treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
            }

            // per-corner masks: CMB face = dzn==0, SURFACE face = dzn==1
            bool matvec_in_range[8];
            bool bdry_in_range[8];
            for ( int node = 0; node < 8; ++node )
            {
                const bool on_cmb_face = ( ( ( node >> 2 ) & 1 ) == 0 );
                const bool on_sur_face = ( ( ( node >> 2 ) & 1 ) == 1 );
                matvec_in_range[node]  = !( treat_boundary_dirichlet && at_cmb && on_cmb_face ) &&
                                        !( treat_boundary_dirichlet && at_surface && on_sur_face );
                bdry_in_range[node] = treat_boundary_dirichlet &&
                                      ( ( at_cmb && on_cmb_face ) || ( at_surface && on_sur_face ) );
            }

            // ---------- accumulators per corner ----------
            double val[8][3]  = {};
            double diag[8][3] = {};

            constexpr double ONE_THIRD      = 1.0 / 3.0;
            constexpr double NEG_TWO_THIRDS = -0.66666666666666663;
            constexpr double GAUSS_A        = 0.21132486540518713;
            constexpr double GAUSS_B        = 0.78867513459481287;

            // Quadrature selected by single_quadpoint_:
            //   false -> 2x2x2 Gauss (8 points, weight 1/8 each)  [full rank]
            //   true  -> single cell-centre point (weight 1)      [RANK-DEFICIENT:
            //            1-pt Q1-hex integration has hourglass modes; for comparison only]
            const int    n_per_dim = single_quadpoint_ ? 1 : 2;
            const double qcoord0   = single_quadpoint_ ? 0.5 : GAUSS_A;
            const double w_quad    = single_quadpoint_ ? 1.0 : 0.125;

            // ============ Gauss point loop ============
            for ( int qzi = 0; qzi < n_per_dim; ++qzi )
            {
                const double zeta = ( qzi == 0 ) ? qcoord0 : GAUSS_B;
                const double r_q  = ( 1.0 - zeta ) * r_0 + zeta * r_1;

                for ( int qyi = 0; qyi < n_per_dim; ++qyi )
                {
                    const double eta = ( qyi == 0 ) ? qcoord0 : GAUSS_B;

                    const double cx_xi = ( 1.0 - eta ) * ( cs[1][0] - cs[0][0] ) + eta * ( cs[3][0] - cs[2][0] );
                    const double cy_xi = ( 1.0 - eta ) * ( cs[1][1] - cs[0][1] ) + eta * ( cs[3][1] - cs[2][1] );
                    const double cz_xi = ( 1.0 - eta ) * ( cs[1][2] - cs[0][2] ) + eta * ( cs[3][2] - cs[2][2] );

                    const double J_0_0 = r_q * cx_xi;
                    const double J_1_0 = r_q * cy_xi;
                    const double J_2_0 = r_q * cz_xi;

                    for ( int qxi = 0; qxi < n_per_dim; ++qxi )
                    {
                        const double xi = ( qxi == 0 ) ? qcoord0 : GAUSS_B;

                        const double cx_eta = ( 1.0 - xi ) * ( cs[2][0] - cs[0][0] ) + xi * ( cs[3][0] - cs[1][0] );
                        const double cy_eta = ( 1.0 - xi ) * ( cs[2][1] - cs[0][1] ) + xi * ( cs[3][1] - cs[1][1] );
                        const double cz_eta = ( 1.0 - xi ) * ( cs[2][2] - cs[0][2] ) + xi * ( cs[3][2] - cs[1][2] );

                        const double xa = 1.0 - xi;
                        const double xb = xi;
                        const double ya = 1.0 - eta;
                        const double yb = eta;
                        const double cx_c =
                            xa * ya * cs[0][0] + xb * ya * cs[1][0] + xa * yb * cs[2][0] + xb * yb * cs[3][0];
                        const double cy_c =
                            xa * ya * cs[0][1] + xb * ya * cs[1][1] + xa * yb * cs[2][1] + xb * yb * cs[3][1];
                        const double cz_c =
                            xa * ya * cs[0][2] + xb * ya * cs[1][2] + xa * yb * cs[2][2] + xb * yb * cs[3][2];

                        const double J_0_1 = r_q * cx_eta;
                        const double J_1_1 = r_q * cy_eta;
                        const double J_2_1 = r_q * cz_eta;
                        const double J_0_2 = dr * cx_c;
                        const double J_1_2 = dr * cy_c;
                        const double J_2_2 = dr * cz_c;

                        const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                             J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;
                        const double inv_det = 1.0 / J_det;
                        const double abs_det = Kokkos::abs( J_det );

                        // per-corner physical gradient + shape value, and k at q
                        double g[8][3];
                        double k_eval = 0.0;
                        for ( int node = 0; node < 8; ++node )
                        {
                            const int    dxn = node & 1;
                            const int    dyn = ( node >> 1 ) & 1;
                            const int    dzn = ( node >> 2 ) & 1;
                            const double s_x = ( dxn == 0 ) ? -1.0 : 1.0;
                            const double s_y = ( dyn == 0 ) ? -1.0 : 1.0;
                            const double s_z = ( dzn == 0 ) ? -1.0 : 1.0;
                            // 1D linear shape value at the quad coord for this corner:
                            //   corner bit 0 -> N = 1 - coord ;  corner bit 1 -> N = coord
                            const double M_x = ( dxn != 0 ) ? xi : ( 1.0 - xi );
                            const double M_y = ( dyn != 0 ) ? eta : ( 1.0 - eta );
                            const double M_z = ( dzn != 0 ) ? zeta : ( 1.0 - zeta );

                            const double gx_ref = s_x * M_y * M_z;
                            const double gy_ref = M_x * s_y * M_z;
                            const double gz_ref = M_x * M_y * s_z;

                            g[node][0] = inv_det * ( ( J_1_1 * J_2_2 - J_1_2 * J_2_1 ) * gx_ref +
                                                     ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 ) * gy_ref +
                                                     ( J_1_0 * J_2_1 - J_1_1 * J_2_0 ) * gz_ref );
                            g[node][1] = inv_det * ( ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 ) * gx_ref +
                                                     ( J_0_0 * J_2_2 - J_0_2 * J_2_0 ) * gy_ref +
                                                     ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 ) * gz_ref );
                            g[node][2] = inv_det * ( ( J_0_1 * J_1_2 - J_0_2 * J_1_1 ) * gx_ref +
                                                     ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 ) * gy_ref +
                                                     ( J_0_0 * J_1_1 - J_0_1 * J_1_0 ) * gz_ref );

                            k_eval += ( M_x * M_y * M_z ) * kc[node];
                        }

                        const double kwJ = w_quad * k_eval * abs_det;

                        // strain eps(u)(q): sum over corners (Dirichlet-constrained corners excluded)
                        double gu00 = 0.0, gu10 = 0.0, gu11 = 0.0, gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
                        if constexpr ( !Diagonal )
                        {
                            for ( int node = 0; node < 8; ++node )
                            {
                                if ( !matvec_in_range[node] )
                                    continue;
                                const double g0 = g[node][0], g1 = g[node][1], g2 = g[node][2];
                                const double s0 = s[node][0], s1 = s[node][1], s2 = s[node][2];
                                gu00 += g0 * s0;
                                gu10 += 0.5 * ( g1 * s0 + g0 * s1 );
                                gu11 += g1 * s1;
                                gu20 += 0.5 * ( g2 * s0 + g0 * s2 );
                                gu21 += 0.5 * ( g2 * s1 + g1 * s2 );
                                gu22 += g2 * s2;
                            }
                        }
                        const double div_u = gu00 + gu11 + gu22;

                        // per-corner contributions at this quad point
                        for ( int node = 0; node < 8; ++node )
                        {
                            const double g0 = g[node][0], g1 = g[node][1], g2 = g[node][2];
                            const double s0 = s[node][0], s1 = s[node][1], s2 = s[node][2];

                            if constexpr ( !Diagonal )
                            {
                                if ( matvec_in_range[node] )
                                {
                                    val[node][0] += kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) +
                                                            NEG_TWO_THIRDS * g0 * div_u );
                                    val[node][1] += kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) +
                                                            NEG_TWO_THIRDS * g1 * div_u );
                                    val[node][2] += kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) +
                                                            NEG_TWO_THIRDS * g2 * div_u );
                                }
                                if ( bdry_in_range[node] )
                                {
                                    const double gg = g0 * g0 + g1 * g1 + g2 * g2;
                                    diag[node][0] += kwJ * s0 * ( gg + ONE_THIRD * g0 * g0 );
                                    diag[node][1] += kwJ * s1 * ( gg + ONE_THIRD * g1 * g1 );
                                    diag[node][2] += kwJ * s2 * ( gg + ONE_THIRD * g2 * g2 );
                                }
                            }
                            else
                            {
                                const double gg = g0 * g0 + g1 * g1 + g2 * g2;
                                diag[node][0] += kwJ * s0 * ( gg + ONE_THIRD * g0 * g0 );
                                diag[node][1] += kwJ * s1 * ( gg + ONE_THIRD * g1 * g1 );
                                diag[node][2] += kwJ * s2 * ( gg + ONE_THIRD * g2 * g2 );
                            }
                        }
                    } // qx
                } // qy
            } // qz

            // ============ scatter (one atomic_add set per corner) ============
            for ( int node = 0; node < 8; ++node )
            {
                const int dxn = node & 1;
                const int dyn = ( node >> 1 ) & 1;
                const int dzn = ( node >> 2 ) & 1;
                const int xo  = x_cell + dxn;
                const int yo  = y_cell + dyn;
                const int ro  = r_cell + dzn;

                if constexpr ( !Diagonal )
                {
                    if ( matvec_in_range[node] )
                    {
                        Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 0 ), val[node][0] );
                        Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 1 ), val[node][1] );
                        Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 2 ), val[node][2] );
                    }
                    if ( bdry_in_range[node] )
                    {
                        Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 0 ), diag[node][0] );
                        Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 1 ), diag[node][1] );
                        Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 2 ), diag[node][2] );
                    }
                }
                else
                {
                    Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 0 ), diag[node][0] );
                    Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 1 ), diag[node][1] );
                    Kokkos::atomic_add( &dst_( local_subdomain_id, xo, yo, ro, 2 ), diag[node][2] );
                }
            }
        } );
}
