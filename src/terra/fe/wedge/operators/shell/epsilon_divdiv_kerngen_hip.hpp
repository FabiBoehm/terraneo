/**
 * @file epsilon_divdiv_kerngen_hip.hpp
 * @brief Native HIP kernel for the fast Dirichlet/Neumann path of EpsilonDivDivKerngen.
 *
 * This is a direct port of operator_fast_dirichlet_neumann_path<false> (matvec)
 * from the Kokkos TeamPolicy version, eliminating Kokkos kernel launch overhead
 * and giving the compiler full control over register allocation.
 *
 * Usage:
 *   Call launch_epsdivdiv_dn_hip() from the host instead of the Kokkos parallel_for.
 *   All data must already reside in HIP device memory (Kokkos Views expose .data() pointers).
 *
 * Design choices:
 *   - One HIP thread block = one team (same tiling as Kokkos version)
 *   - __shared__ memory replaces Kokkos scratch memory
 *   - atomicAdd replaces Kokkos::atomic_add (with -munsafe-fp-atomics for native FP64 atomics)
 *   - Grid4DDataVec is SoA: 3 separate 4D arrays for x,y,z components
 *   - Grid data uses LayoutRight: last index (r) is contiguous
 */

#pragma once

#include <hip/hip_runtime.h>
#include <cmath>

namespace terra::fe::wedge::operators::shell::hip_kernels {

// ============================================================================
// Shared memory layout for the DN kernel
// Must match: coords(nxy,3) + src(nxy,3,nlev) + k(nxy,nlev) + r(nlev)
// ============================================================================
struct DNShmemLayout
{
    int nxy;
    int nlev;

    // Offsets (in doubles) into the shared memory buffer
    __device__ int coords_offset() const { return 0; }
    __device__ int src_offset()    const { return nxy * 3; }
    __device__ int k_offset()      const { return nxy * 3 + nxy * 3 * nlev; }
    __device__ int r_offset()      const { return nxy * 3 + nxy * 3 * nlev + nxy * nlev; }
    __device__ int total()         const { return nxy * 3 + nxy * 3 * nlev + nxy * nlev + nlev; }

    // LayoutRight accessors: last index is contiguous
    __device__ double& coords( double* base, int node, int dim ) const
    {
        return base[coords_offset() + node * 3 + dim];
    }
    __device__ double& src( double* base, int node, int dim, int lvl ) const
    {
        return base[src_offset() + ( node * 3 + dim ) * nlev + lvl];
    }
    __device__ double& k( double* base, int node, int lvl ) const
    {
        return base[k_offset() + node * nlev + lvl];
    }
    __device__ double& r( double* base, int lvl ) const
    {
        return base[r_offset() + lvl];
    }
};

// ============================================================================
// Kernel parameters (POD struct, passed by value to avoid pointer chasing)
// ============================================================================
struct DNKernelParams
{
    // Grid3DDataVec<double,3> = View<double***[3], LayoutRight>: (sd, x, y, [3])
    const double* grid;       // single pointer, [3] interleaved as last dim
    int           nx_grid;    // extent(1) of grid view
    int           ny_grid;    // extent(2) of grid view

    const double* radii;      // Grid2DDataScalar: (sd, r) LayoutRight
    int           nr_radii;   // extent(1) of radii view

    const double* k;          // Grid4DDataScalar: (sd, x, y, r) LayoutRight
    // SoA source/dest: 3 separate Grid4DDataScalar views
    const double* src_0;
    const double* src_1;
    const double* src_2;
    double*       dst_0;
    double*       dst_1;
    double*       dst_2;

    // All 4D scalar views share the same extents
    int           nx;         // extent(1)
    int           ny;         // extent(2)
    int           nr;         // extent(3)

    // Boundary mask: Grid4DDataScalar<uint8_t>: (sd, x, y, r) LayoutRight
    const uint8_t* mask;

    // Boundary conditions
    uint8_t       bc_cmb;
    uint8_t       bc_surface;

    // Tiling
    int lat_tile;
    int r_tile;
    int r_passes;
    int r_tile_block;

    // Grid extents (in cells)
    int hex_lat;
    int hex_rad;
    int lat_tiles;
    int r_tiles;

    // ShellBoundaryFlag encoding (from bit_masks.hpp)
    static constexpr uint8_t CMB_FLAG     = 0x06;  // BOUNDARY | (1<<2)
    static constexpr uint8_t SURFACE_FLAG = 0x0A;  // BOUNDARY | (1<<3)
    static constexpr uint8_t DIRICHLET    = 1;     // BoundaryConditionFlag::DIRICHLET
};

// ============================================================================
// LayoutRight index helpers — flat index from extents
// ============================================================================

// Grid3DDataVec: View<double***[3], LayoutRight> → flat = ((sd*nx + x)*ny + y)*3 + dim
__device__ __forceinline__
double grid_val( const DNKernelParams& p, int sd, int x, int y, int dim )
{
    return p.grid[( ( sd * p.nx_grid + x ) * p.ny_grid + y ) * 3 + dim];
}

// Grid4DDataScalar: View<double****, LayoutRight> → flat = ((sd*nx + x)*ny + y)*nr + r
__device__ __forceinline__
int idx4( const DNKernelParams& p, int sd, int x, int y, int r )
{
    return ( ( sd * p.nx + x ) * p.ny + y ) * p.nr + r;
}

// ============================================================================
// Main kernel: Fast Dirichlet/Neumann matvec path
// ============================================================================
__global__
__attribute__((amdgpu_flat_work_group_size(64, 1024)))
__attribute__((amdgpu_waves_per_eu(4)))
void epsdivdiv_dn_matvec_kernel( const DNKernelParams p )
{
    // --- Decode block (team) indices ---
    int tmp = blockIdx.x;
    const int r_tile_id    = tmp % p.r_tiles;    tmp /= p.r_tiles;
    const int lat_y_id     = tmp % p.lat_tiles;  tmp /= p.lat_tiles;
    const int lat_x_id     = tmp % p.lat_tiles;  tmp /= p.lat_tiles;
    const int sd           = tmp;  // local subdomain id

    const int x0 = lat_x_id * p.lat_tile;
    const int y0 = lat_y_id * p.lat_tile;
    const int r0 = r_tile_id * p.r_tile_block;

    // --- Decode thread indices within team ---
    // Thread layout: tw (fastest) | tr | tx | ty (slowest)
    // Each thread handles one wedge (w = 0 or 1) of one hex cell.
    const int tid = threadIdx.x;
    const int tw  = tid & 1;                               // wedge: 0 or 1
    const int tr  = ( tid >> 1 ) % p.r_tile;
    const int tx  = ( tid >> 1 ) / p.r_tile % p.lat_tile;
    const int ty  = ( tid >> 1 ) / ( p.r_tile * p.lat_tile );

    const int x_cell = x0 + tx;
    const int y_cell = y0 + ty;

    // --- Shared memory ---
    extern __shared__ double shmem[];

    const int nlev = p.r_tile_block + 1;
    const int n    = p.lat_tile + 1;
    const int nxy  = n * n;

    DNShmemLayout sh{ nxy, nlev };

    auto node_id = [&]( int nx, int ny ) -> int { return nx + n * ny; };

    // --- Phase 0: Cooperative load into shared memory ---

    // Load coordinates (nxy items, team_size threads)
    for ( int i = tid; i < nxy; i += blockDim.x )
    {
        const int dxn = i % n;
        const int dyn = i / n;
        const int xi  = x0 + dxn;
        const int yi  = y0 + dyn;

        if ( xi <= p.hex_lat && yi <= p.hex_lat )
        {
            sh.coords( shmem, i, 0 ) = grid_val( p, sd, xi, yi, 0 );
            sh.coords( shmem, i, 1 ) = grid_val( p, sd, xi, yi, 1 );
            sh.coords( shmem, i, 2 ) = grid_val( p, sd, xi, yi, 2 );
        }
        else
        {
            sh.coords( shmem, i, 0 ) = sh.coords( shmem, i, 1 ) = sh.coords( shmem, i, 2 ) = 0.0;
        }
    }

    // Load radii (nlev items)
    for ( int i = tid; i < nlev; i += blockDim.x )
    {
        const int rr = r0 + i;
        sh.r( shmem, i ) = ( rr <= p.hex_rad ) ? p.radii[sd * p.nr_radii + rr] : 0.0;
    }

    // Load src and k (nxy * nlev items)
    const int total_pairs = nxy * nlev;
    for ( int t = tid; t < total_pairs; t += blockDim.x )
    {
        const int node = t / nlev;
        const int lvl  = t - node * nlev;
        const int dxn  = node % n;
        const int dyn  = node / n;
        const int xi   = x0 + dxn;
        const int yi   = y0 + dyn;
        const int rr   = r0 + lvl;

        if ( xi <= p.hex_lat && yi <= p.hex_lat && rr <= p.hex_rad )
        {
            const int idx = idx4( p, sd, xi, yi, rr );
            sh.k( shmem, node, lvl )      = p.k[idx];
            sh.src( shmem, node, 0, lvl )  = p.src_0[idx];
            sh.src( shmem, node, 1, lvl )  = p.src_1[idx];
            sh.src( shmem, node, 2, lvl )  = p.src_2[idx];
        }
        else
        {
            sh.k( shmem, node, lvl )     = 0.0;
            sh.src( shmem, node, 0, lvl ) = sh.src( shmem, node, 1, lvl ) = sh.src( shmem, node, 2, lvl ) = 0.0;
        }
    }

    __syncthreads();

    // --- Early exit for out-of-bounds threads ---
    if ( x_cell >= p.hex_lat || y_cell >= p.hex_lat )
        return;
    if ( tr >= p.r_tile )
        return;

    // --- Constants ---
    constexpr double ONE_THIRD       = 1.0 / 3.0;
    constexpr double ONE_SIXTH       = 1.0 / 6.0;
    constexpr double NEG_TWO_THIRDS  = -2.0 / 3.0;

    constexpr double dN_ref[6][3] = {
        { -0.5, -0.5, -ONE_SIXTH },
        {  0.5,  0.0, -ONE_SIXTH },
        {  0.0,  0.5, -ONE_SIXTH },
        { -0.5, -0.5,  ONE_SIXTH },
        {  0.5,  0.0,  ONE_SIXTH },
        {  0.0,  0.5,  ONE_SIXTH } };

    constexpr int WEDGE_NODE_OFF[2][6][3] = {
        { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
        { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

    const int n00 = node_id( tx, ty );
    const int n01 = node_id( tx, ty + 1 );
    const int n10 = node_id( tx + 1, ty );
    const int n11 = node_id( tx + 1, ty + 1 );

    // --- Per-cell work across radial passes ---
    for ( int pass = 0; pass < p.r_passes; ++pass )
    {
        const int lvl0   = pass * p.r_tile + tr;
        const int r_cell = r0 + lvl0;

        if ( r_cell >= p.hex_rad )
            break;

        const double r_0 = sh.r( shmem, lvl0 );
        const double r_1 = sh.r( shmem, lvl0 + 1 );

        // Boundary condition checks (mask uses same extents as 4D scalar views)
        const int mask_base = ( ( sd * p.nx + x_cell ) * p.ny + y_cell ) * p.nr;
        const uint8_t mask_val    = p.mask[mask_base + r_cell];
        const uint8_t mask_val_p1 = p.mask[mask_base + r_cell + 1];
        const bool at_cmb      = mask_val    == DNKernelParams::CMB_FLAG;
        const bool at_surface  = mask_val_p1 == DNKernelParams::SURFACE_FLAG;
        const bool at_boundary = at_cmb || at_surface;

        bool treat_boundary_dirichlet = false;
        if ( at_boundary )
        {
            if ( at_cmb )
                treat_boundary_dirichlet = ( p.bc_cmb == DNKernelParams::DIRICHLET );
            else
                treat_boundary_dirichlet = ( p.bc_surface == DNKernelParams::DIRICHLET );
        }

        const int cmb_shift     = ( at_boundary && treat_boundary_dirichlet && at_cmb )     ? 3 : 0;
        const int surface_shift = ( at_boundary && treat_boundary_dirichlet && at_surface )  ? 3 : 0;

        // --- One wedge per thread (tw is this thread's wedge: 0 or 1) ---
        {
            const int w  = tw;
            const int v0 = w == 0 ? n00 : n11;
            const int v1 = w == 0 ? n10 : n01;
            const int v2 = w == 0 ? n01 : n10;

            // Coefficient evaluation
            double k_sum = 0.0;
            for ( int node = 0; node < 6; ++node )
            {
                const int nid = node_id( tx + WEDGE_NODE_OFF[w][node][0], ty + WEDGE_NODE_OFF[w][node][1] );
                k_sum += sh.k( shmem, nid, lvl0 + WEDGE_NODE_OFF[w][node][2] );
            }
            const double k_eval = ONE_SIXTH * k_sum;

            double kwJ;

            // ==== Phase 1: Jacobian + Gather (strain tensor) ====
            double gu00 = 0.0, gu10 = 0.0, gu11 = 0.0;
            double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
            double div_u = 0.0;
            {
                const double half_dr = 0.5 * ( r_1 - r_0 );
                const double r_mid   = 0.5 * ( r_0 + r_1 );

                const double J_0_0 = r_mid * ( -sh.coords( shmem, v0, 0 ) + sh.coords( shmem, v1, 0 ) );
                const double J_0_1 = r_mid * ( -sh.coords( shmem, v0, 0 ) + sh.coords( shmem, v2, 0 ) );
                const double J_0_2 = half_dr * ONE_THIRD * ( sh.coords( shmem, v0, 0 ) + sh.coords( shmem, v1, 0 ) + sh.coords( shmem, v2, 0 ) );

                const double J_1_0 = r_mid * ( -sh.coords( shmem, v0, 1 ) + sh.coords( shmem, v1, 1 ) );
                const double J_1_1 = r_mid * ( -sh.coords( shmem, v0, 1 ) + sh.coords( shmem, v2, 1 ) );
                const double J_1_2 = half_dr * ONE_THIRD * ( sh.coords( shmem, v0, 1 ) + sh.coords( shmem, v1, 1 ) + sh.coords( shmem, v2, 1 ) );

                const double J_2_0 = r_mid * ( -sh.coords( shmem, v0, 2 ) + sh.coords( shmem, v1, 2 ) );
                const double J_2_1 = r_mid * ( -sh.coords( shmem, v0, 2 ) + sh.coords( shmem, v2, 2 ) );
                const double J_2_2 = half_dr * ONE_THIRD * ( sh.coords( shmem, v0, 2 ) + sh.coords( shmem, v1, 2 ) + sh.coords( shmem, v2, 2 ) );

                const double J_det = J_0_0 * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 )
                                   - J_0_1 * ( J_1_0 * J_2_2 - J_1_2 * J_2_0 )
                                   + J_0_2 * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );

                kwJ = k_eval * fabs( J_det );
                const double inv_det = 1.0 / J_det;

                const double i00 = inv_det * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                const double i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                const double i02 = inv_det * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                const double i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                const double i11 = inv_det * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                const double i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                const double i20 = inv_det * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                const double i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                const double i22 = inv_det * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                for ( int nn = cmb_shift; nn < 6 - surface_shift; ++nn )
                {
                    const double g0 = i00 * dN_ref[nn][0] + i01 * dN_ref[nn][1] + i02 * dN_ref[nn][2];
                    const double g1 = i10 * dN_ref[nn][0] + i11 * dN_ref[nn][1] + i12 * dN_ref[nn][2];
                    const double g2 = i20 * dN_ref[nn][0] + i21 * dN_ref[nn][1] + i22 * dN_ref[nn][2];

                    const int nid = node_id( tx + WEDGE_NODE_OFF[w][nn][0], ty + WEDGE_NODE_OFF[w][nn][1] );
                    const int lvl = lvl0 + WEDGE_NODE_OFF[w][nn][2];

                    const double s0 = sh.src( shmem, nid, 0, lvl );
                    const double s1 = sh.src( shmem, nid, 1, lvl );
                    const double s2 = sh.src( shmem, nid, 2, lvl );

                    gu00  += g0 * s0;
                    gu11  += g1 * s1;
                    gu22  += g2 * s2;
                    gu10  += 0.5 * ( g1 * s0 + g0 * s1 );
                    gu20  += 0.5 * ( g2 * s0 + g0 * s2 );
                    gu21  += 0.5 * ( g2 * s1 + g1 * s2 );
                    div_u += g0 * s0 + g1 * s1 + g2 * s2;
                }
            }
            // invJ out of scope — registers reclaimed

            // ==== Phase 2: Recompute Jacobian + Scatter ====
            {
                const double half_dr = 0.5 * ( r_1 - r_0 );
                const double r_mid   = 0.5 * ( r_0 + r_1 );

                const double J_0_0 = r_mid * ( -sh.coords( shmem, v0, 0 ) + sh.coords( shmem, v1, 0 ) );
                const double J_0_1 = r_mid * ( -sh.coords( shmem, v0, 0 ) + sh.coords( shmem, v2, 0 ) );
                const double J_0_2 = half_dr * ONE_THIRD * ( sh.coords( shmem, v0, 0 ) + sh.coords( shmem, v1, 0 ) + sh.coords( shmem, v2, 0 ) );

                const double J_1_0 = r_mid * ( -sh.coords( shmem, v0, 1 ) + sh.coords( shmem, v1, 1 ) );
                const double J_1_1 = r_mid * ( -sh.coords( shmem, v0, 1 ) + sh.coords( shmem, v2, 1 ) );
                const double J_1_2 = half_dr * ONE_THIRD * ( sh.coords( shmem, v0, 1 ) + sh.coords( shmem, v1, 1 ) + sh.coords( shmem, v2, 1 ) );

                const double J_2_0 = r_mid * ( -sh.coords( shmem, v0, 2 ) + sh.coords( shmem, v1, 2 ) );
                const double J_2_1 = r_mid * ( -sh.coords( shmem, v0, 2 ) + sh.coords( shmem, v2, 2 ) );
                const double J_2_2 = half_dr * ONE_THIRD * ( sh.coords( shmem, v0, 2 ) + sh.coords( shmem, v1, 2 ) + sh.coords( shmem, v2, 2 ) );

                const double J_det = J_0_0 * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 )
                                   - J_0_1 * ( J_1_0 * J_2_2 - J_1_2 * J_2_0 )
                                   + J_0_2 * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );

                const double inv_det = 1.0 / J_det;

                const double i00 = inv_det * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                const double i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                const double i02 = inv_det * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                const double i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                const double i11 = inv_det * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                const double i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                const double i20 = inv_det * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                const double i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                const double i22 = inv_det * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                for ( int nn = cmb_shift; nn < 6 - surface_shift; ++nn )
                {
                    const double g0 = i00 * dN_ref[nn][0] + i01 * dN_ref[nn][1] + i02 * dN_ref[nn][2];
                    const double g1 = i10 * dN_ref[nn][0] + i11 * dN_ref[nn][1] + i12 * dN_ref[nn][2];
                    const double g2 = i20 * dN_ref[nn][0] + i21 * dN_ref[nn][1] + i22 * dN_ref[nn][2];

                    const int ddx = WEDGE_NODE_OFF[w][nn][0];
                    const int ddy = WEDGE_NODE_OFF[w][nn][1];
                    const int ddr = WEDGE_NODE_OFF[w][nn][2];

                    const int di = idx4( p, sd, x_cell + ddx, y_cell + ddy, r_cell + ddr );

                    atomicAdd( &p.dst_0[di], kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) + NEG_TWO_THIRDS * g0 * div_u ) );
                    atomicAdd( &p.dst_1[di], kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) + NEG_TWO_THIRDS * g1 * div_u ) );
                    atomicAdd( &p.dst_2[di], kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) + NEG_TWO_THIRDS * g2 * div_u ) );
                }

                // Diagonal contribution at Dirichlet boundaries
                if ( treat_boundary_dirichlet && at_boundary )
                {
                    for ( int nn = surface_shift; nn < 6 - cmb_shift; ++nn )
                    {
                        const double g0 = i00 * dN_ref[nn][0] + i01 * dN_ref[nn][1] + i02 * dN_ref[nn][2];
                        const double g1 = i10 * dN_ref[nn][0] + i11 * dN_ref[nn][1] + i12 * dN_ref[nn][2];
                        const double g2 = i20 * dN_ref[nn][0] + i21 * dN_ref[nn][1] + i22 * dN_ref[nn][2];
                        const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                        const int nid = node_id( tx + WEDGE_NODE_OFF[w][nn][0], ty + WEDGE_NODE_OFF[w][nn][1] );
                        const int lvl = lvl0 + WEDGE_NODE_OFF[w][nn][2];

                        const double sv0 = sh.src( shmem, nid, 0, lvl );
                        const double sv1 = sh.src( shmem, nid, 1, lvl );
                        const double sv2 = sh.src( shmem, nid, 2, lvl );

                        const int ddx = WEDGE_NODE_OFF[w][nn][0];
                        const int ddy = WEDGE_NODE_OFF[w][nn][1];
                        const int ddr = WEDGE_NODE_OFF[w][nn][2];

                        const int di = idx4( p, sd, x_cell + ddx, y_cell + ddy, r_cell + ddr );
                        atomicAdd( &p.dst_0[di], kwJ * sv0 * ( gg + ONE_THIRD * g0 * g0 ) );
                        atomicAdd( &p.dst_1[di], kwJ * sv1 * ( gg + ONE_THIRD * g1 * g1 ) );
                        atomicAdd( &p.dst_2[di], kwJ * sv2 * ( gg + ONE_THIRD * g2 * g2 ) );
                    }
                }
            }
        } // end wedge loop
    } // end r_passes loop
}

// ============================================================================
// Host-side launch function
// ============================================================================
inline void launch_epsdivdiv_dn_matvec(
    const DNKernelParams& params,
    int                   num_blocks,
    int                   team_size,
    size_t                shmem_bytes,
    hipStream_t           stream = 0 )
{
    hipLaunchKernelGGL(
        epsdivdiv_dn_matvec_kernel,
        dim3( num_blocks ),
        dim3( team_size ),
        shmem_bytes,
        stream,
        params );
}

} // namespace terra::fe::wedge::operators::shell::hip_kernels
