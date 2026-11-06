#pragma once

// The warp-owner tile: what the fast Dirichlet/Neumann path of EpsilonDivDivKerngen
// (epsilon_divdiv_kerngen/path_fast_dn_warp.hpp) does to scatter without atomics,
// factored out so the gradient, divergence and energy operators can run the same way.
//
// One thread per lateral column of an 8 x 8 cell tile (a team of 64 threads = two
// warps), each walking a radial run of cells. A cell's contributions go to the 8
// corners of the hex cell: 4 columns x 2 levels. The upper level is carried in
// registers into the next cell, so a node level is complete in the thread's
// registers before anything is written; the 4 lateral corners are then reduced to
// their owner lane with warp shuffles -- the lane that owns column (tx, ty) gets the
// contributions of lanes (tx-1, ty), (tx, ty-1), (tx-1, ty-1) -- and the owner writes
// the node once: a plain store (or read-modify-write in Add mode) for a node only
// this team touches, a global reduction on the tile's perimeter where another team
// contributes too. The second warp gets its from-below corners through a small
// shared-memory seam. Each team also computes the cell below its run (the halo
// cell), so the run's bottom level is complete inside the team and the level on top
// belongs to the team above: no level-boundary atomics, and in Replace mode only the
// tile perimeters need zeroing beforehand.

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "util/floating_point.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

/// Shared memory per SM the launch bounds' minimum blocks must fit in (bytes): H100 228 KiB.
#ifndef TERRA_EPSDIVDIV_SM_SHMEM
#define TERRA_EPSDIVDIV_SM_SHMEM ( 228 * 1024 )
#endif

namespace terra::fe::wedge::operators::shell::warp_tile {

/// Lateral tile edge in cells; one thread per column.
constexpr int lat       = 8;
constexpr int team_size = lat * lat;
constexpr int PW        = lat;      ///< warp patch width (columns in x)
constexpr int PH        = 32 / lat; ///< warp patch height (columns in y)
constexpr int NW        = team_size / 32;
constexpr int nxy       = ( lat + 1 ) * ( lat + 1 ); ///< node columns of a tile

using Team = Kokkos::TeamPolicy<>::member_type;

/// Corner c = ddx + 2 ddy of the hex cell; (ddx, ddy, ddr) of node n of wedge w (the
/// convention of kernel_helpers.hpp). A function with a local table rather than a
/// namespace-scope array: nvcc does not let device code read the latter.
KOKKOS_INLINE_FUNCTION constexpr int wedge_node_off( const int w, const int n, const int k )
{
    constexpr int off[2][6][3] = { { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
                                   { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };
    return off[w][n][k];
}

/// Corner index (ddx + 2 ddy) of lateral vertex v in {0, 1, 2} of wedge w:
/// w = 0 -> (0,0),(1,0),(0,1); w = 1 -> (1,1),(0,1),(1,0).
KOKKOS_INLINE_FUNCTION constexpr int wedge_vertex( const int w, const int v )
{
    constexpr int vert[2][3] = { { 0, 1, 2 }, { 3, 2, 1 } };
    return vert[w][v];
}

/// get_shmem hands out aligned chunks; round each block up so two requests in one
/// team allocation cannot overlap whatever the element sizes are.
KOKKOS_INLINE_FUNCTION constexpr size_t shmem_block_bytes( size_t bytes )
{
    return ( bytes + 7u ) & ~size_t( 7u );
}

/// Kokkos (5.1, CUDA) launches a team kernel only if its scratch fits one block
/// (48 KiB, no opt-in) and LaunchBounds' minimum blocks per SM fit the SM's shared
/// memory. See EpsilonDivDivKerngen::kokkos_shmem_limit.
constexpr size_t kokkos_shmem_limit( int team, int min_blocks )
{
    const size_t per_block = 48u * 1024u;
    const size_t per_sm    = size_t( TERRA_EPSDIVDIV_SM_SHMEM );
    const size_t cap       = per_sm / size_t( min_blocks ) < per_block ? per_sm / size_t( min_blocks ) : per_block;
    return cap - size_t( team + 2 ) * sizeof( double ) - 256u;
}

/// The team grid: tiles of lat x lat columns and r_block cells, r fastest in the
/// league rank (the order EpsilonDivDivKerngen uses).
struct Tiling
{
    int nsub      = 0;
    int hex_lat   = 0; ///< cells per side of a subdomain
    int hex_rad   = 0; ///< radial cells of a subdomain
    int r_block   = 0; ///< cells per radial run
    int lat_tiles = 0;
    int r_tiles   = 0;
    int blocks    = 0;

    Tiling() = default;

    Tiling( const grid::shell::DistributedDomain& domain, const int r_block_ )
    {
        const auto& info = domain.domain_info();
        nsub             = static_cast< int >( domain.subdomains().size() );
        hex_lat          = info.subdomain_num_nodes_per_side_laterally() - 1;
        hex_rad          = info.subdomain_num_nodes_radially() - 1;
        r_block          = r_block_;
        lat_tiles        = ( hex_lat + lat - 1 ) / lat;
        r_tiles          = ( hex_rad + r_block - 1 ) / r_block;
        blocks           = nsub * lat_tiles * lat_tiles * r_tiles;
    }

    /// The warp path wants whole tiles (the perimeter fill and the owner logic assume
    /// every column of a tile exists) and at least one cell per run.
    bool usable() const { return hex_lat >= lat && hex_lat % lat == 0 && hex_rad >= 1; }

    KOKKOS_INLINE_FUNCTION void decode( int league, int& sub, int& x0, int& y0, int& r0 ) const
    {
        const int rt = league % r_tiles;
        league /= r_tiles;
        const int yt = league % lat_tiles;
        league /= lat_tiles;
        const int xt = league % lat_tiles;
        sub          = league / lat_tiles;
        x0           = xt * lat;
        y0           = yt * lat;
        r0           = rt * r_block;
    }
};

/// Radial run length per team: 16 cells where the subdomain allows it, 8 otherwise.
inline int default_r_block( const grid::shell::DistributedDomain& domain )
{
    const int hex_rad = domain.domain_info().subdomain_num_nodes_radially() - 1;
    return ( hex_rad % 16 == 0 ) ? 16 : 8;
}

/// Zero the node columns on the tiles' perimeters (x or y a multiple of lat): in
/// Replace mode the only nodes a warp-owner kernel with a halo cell accumulates into
/// (it stores the rest). Consecutive threads take consecutive levels, so the writes
/// coalesce along r. Works for scalar and vector 4-D grid data.
template < typename GridData >
void zero_perimeter( GridData dst, const Tiling& t )
{
    constexpr int VecDim = grid::grid_data_vec_dim< GridData >();
    using S              = typename GridData::value_type;
    const int nx         = t.hex_lat + 1;        // columns per side
    const int nr         = t.hex_rad + 1;        // levels
    const int nmult      = t.hex_lat / lat + 1;  // multiples of lat in [0, hex_lat]
    const int ncols_x    = nmult * nx;           // x % lat == 0, any y
    const int nx_rest    = nx - nmult;           // x % lat != 0
    const int ncols      = ncols_x + nmult * nx_rest; // + y % lat == 0, x % lat != 0
    // The index decode, shared by the scalar and the vector kernel (nvcc does not
    // allow an extended lambda's first capture inside an if constexpr).
    auto decode = KOKKOS_LAMBDA( const size_t i, int& s, int& xi, int& yi, int& r ) {
        r             = int( i % size_t( nr ) );
        const int col = int( ( i / size_t( nr ) ) % size_t( ncols ) );
        s             = int( i / size_t( nr ) / size_t( ncols ) );
        if ( col < ncols_x )
        {
            xi = ( col / nx ) * lat;
            yi = col % nx;
        }
        else
        {
            const int c  = col - ncols_x;
            const int xr = c % nx_rest;
            yi           = ( c / nx_rest ) * lat;
            xi           = xr + 1 + xr / ( lat - 1 ); // skips the multiples of lat
        }
    };
    const Kokkos::RangePolicy<> policy( 0, size_t( t.nsub ) * size_t( ncols ) * size_t( nr ) );
    if constexpr ( VecDim == 1 )
    {
        Kokkos::parallel_for(
            "warp_tile_zero_perimeter", policy, KOKKOS_LAMBDA( const size_t i ) {
                int s, xi, yi, r;
                decode( i, s, xi, yi, r );
                dst( s, xi, yi, r ) = S( 0 );
            } );
    }
    else
    {
        Kokkos::parallel_for(
            "warp_tile_zero_perimeter", policy, KOKKOS_LAMBDA( const size_t i ) {
                int s, xi, yi, r;
                decode( i, s, xi, yi, r );
                for ( int d = 0; d < VecDim; ++d )
                    dst( s, xi, yi, r, d ) = S( 0 );
            } );
    }
}

// ---- deterministic two-phase scatter ----
//
// The warp-owner kernels store every node only their own team touches and reduce
// the rest -- the columns on the tiles' perimeters -- with global atomics, whose
// arrival order differs from launch to launch, so two applies of the same operator
// on the same input differ in the last bit. With the two-phase scatter nothing is
// reduced in phase 1: the owner lane stores its node (all of them, perimeter or
// not), and the three outward edge contributions of a perimeter node -- the
// (1,0), (0,1) and (1,1) corners from the tile to its left, below it and diagonally
// below -- go into their own slot of a side buffer, one writer per slot. Phase 2
// adds the slots to the owner's value in a fixed order. Deterministic for a given
// tiling, at the cost of the side buffer's traffic (26 % of the nodes at lat 8).
//
// Requested for all operators with TERRA_DETERMINISTIC_SCATTER=1.

/// Whether the environment asks for the deterministic scatter.
inline bool deterministic_scatter_requested()
{
    static const bool v = []() {
        const char* env = std::getenv( "TERRA_DETERMINISTIC_SCATTER" );
        return env != nullptr && std::atoi( env ) != 0;
    }();
    return v;
}

/// One device workspace per element type, grown on demand and shared by every
/// operator (applies are ordered on the default execution space). Freed at finalize.
template < typename T >
Kokkos::View< T* > perimeter_workspace( const size_t n )
{
    static Kokkos::View< T* >* ws = nullptr;
    if ( ws == nullptr )
    {
        ws = new Kokkos::View< T* >( Kokkos::view_alloc( Kokkos::WithoutInitializing, "warp_tile_perimeter_ws" ), n );
        Kokkos::push_finalize_hook( []() {
            delete ws;
            ws = nullptr;
        } );
    }
    else if ( ws->extent( 0 ) < n )
    {
        *ws = Kokkos::View< T* >( Kokkos::view_alloc( Kokkos::WithoutInitializing, "warp_tile_perimeter_ws" ), n );
    }
    return *ws;
}

/// The side buffer of the two-phase scatter: the perimeter node columns of a grid
/// tiled with `tile` x `tile` cell tiles (x or y a multiple of tile, the index map of
/// zero_perimeter), all levels, NC components, three edge slots (1 = from the left,
/// 2 = from below, 3 = diagonal, i.e. the corner ddx + 2 ddy the contribution came
/// from), two halves (0 = the team whose run contains the level, 1 = the team below
/// whose run ends there; kernels whose runs are level-complete use half 0 only) and
/// an optional slab for kernels whose runs are not level-complete: the owner value
/// of the team below for every column at every run-boundary level.
template < typename AccT >
struct PerimeterBuffer
{
    Kokkos::View< AccT* > buf;
    int                   tile = 0, nsub = 0, hex_lat = 0, nx = 0, nr = 0, nc = 0;
    int                   nmult = 0, ncols_x = 0, nx_rest = 0, ncols = 0;
    int                   halves = 1;
    int                   r_block = 0, nslab = 0; ///< slab: run length in cells and number of boundary levels
    size_t                slab_off = 0;

    PerimeterBuffer() = default;

    /// @param halves_  1 for level-complete runs (EpsDivDiv, Gradient, SUPG), 2 otherwise
    /// @param r_block_ run length in cells of the grid the buffer is on (0: no slab)
    PerimeterBuffer( const int tile_, const int nsub_, const int hex_lat_, const int hex_rad, const int nc_, const int halves_, const int r_block_ )
    : tile( tile_ )
    , nsub( nsub_ )
    , hex_lat( hex_lat_ )
    , nx( hex_lat_ + 1 )
    , nr( hex_rad + 1 )
    , nc( nc_ )
    , nmult( hex_lat_ / tile_ + 1 )
    , ncols_x( ( hex_lat_ / tile_ + 1 ) * ( hex_lat_ + 1 ) )
    , nx_rest( hex_lat_ + 1 - ( hex_lat_ / tile_ + 1 ) )
    , ncols( ( hex_lat_ / tile_ + 1 ) * ( hex_lat_ + 1 ) + ( hex_lat_ / tile_ + 1 ) * ( hex_lat_ + 1 - ( hex_lat_ / tile_ + 1 ) ) )
    , halves( halves_ )
    , r_block( r_block_ )
    {
        // Run-boundary levels: r = k r_block, 0 < r < hex_rad.
        nslab    = ( r_block_ > 0 ) ? ( hex_rad - 1 ) / r_block_ : 0;
        slab_off = size_t( nsub ) * halves * 3 * nc * size_t( ncols ) * size_t( nr );
        const size_t slab = size_t( nsub ) * nslab * nc * size_t( nx ) * size_t( nx );
        buf               = perimeter_workspace< AccT >( slab_off + slab );
    }

    bool matches( const int tile_, const int nsub_, const int hex_lat_, const int hex_rad, const int nc_ ) const
    {
        return buf.extent( 0 ) > 0 && tile == tile_ && nsub == nsub_ && hex_lat == hex_lat_ && nr == hex_rad + 1 && nc == nc_;
    }

    /// Perimeter column index of node column (x, y); x % tile == 0 or y % tile == 0.
    KOKKOS_INLINE_FUNCTION int col_of( const int x, const int y ) const
    {
        if ( x % tile == 0 )
            return ( x / tile ) * nx + y;
        return ncols_x + ( y / tile ) * nx_rest + ( x - 1 - x / tile );
    }

    /// Node column of perimeter column index col.
    KOKKOS_INLINE_FUNCTION void node_of( const int col, int& x, int& y ) const
    {
        if ( col < ncols_x )
        {
            x = ( col / nx ) * tile;
            y = col % nx;
        }
        else
        {
            const int c  = col - ncols_x;
            const int xr = c % nx_rest;
            y            = ( c / nx_rest ) * tile;
            x            = xr + 1 + xr / ( tile - 1 );
        }
    }

    /// Edge slot (1..3) of component d of perimeter column col at level r.
    KOKKOS_INLINE_FUNCTION AccT& slot( const int sub, const int half, const int s, const int d, const int col, const int r ) const
    {
        return buf( ( ( ( size_t( sub ) * halves + half ) * 3 + ( s - 1 ) ) * nc + d ) * size_t( ncols ) * nr + size_t( col ) * nr + r );
    }

    /// Slab entry of component d of column (x, y) at run-boundary level r (r = k r_block, k >= 1).
    KOKKOS_INLINE_FUNCTION AccT& slab( const int sub, const int d, const int x, const int y, const int r ) const
    {
        const int k = r / r_block - 1;
        return buf( slab_off + ( ( size_t( sub ) * nslab + k ) * nc + d ) * size_t( nx ) * nx + size_t( x ) * nx + y );
    }

    /// Write the outward edge contributions of a lane at a flush: what the owner
    /// kernels hand to their `emit` with owned = false, keyed by the corner.
    KOKKOS_INLINE_FUNCTION void
        put_edge( const int sub, const int half, const int ddx, const int ddy, const int d, const int x_cell, const int y_cell, const int r, const AccT v ) const
    {
        slot( sub, half, ddx + 2 * ddy, d, col_of( x_cell + ddx, y_cell + ddy ), r ) = v;
    }
};

/// Entry (s, x, y, r[, d]) of scalar or vector 4-D grid data, so a device lambda can
/// use one expression for both (nvcc: an extended lambda cannot first-capture a
/// variable inside an if constexpr).
template < typename GridData >
KOKKOS_INLINE_FUNCTION typename GridData::value_type&
    entry( const GridData& g, const int s, const int x, const int y, const int r, const int d )
{
    if constexpr ( grid::grid_data_vec_dim< GridData >() == 1 )
        return g( s, x, y, r );
    else
        return g( s, x, y, r, d );
}

/// Phase 2: add the slots to the perimeter nodes in a fixed order. `replace` resets
/// the nodes no owner lane stores (x == hex_lat or y == hex_lat); the others start
/// from the owner's value. Scalar and vector 4-D grid data.
template < typename GridData, typename AccT >
void perimeter_reduce( GridData dst, const PerimeterBuffer< AccT >& pb, const bool replace )
{
    constexpr int VecDim = grid::grid_data_vec_dim< GridData >();
    using S              = typename GridData::value_type;
    const int tile = pb.tile, hex_lat = pb.hex_lat, nr = pb.nr, ncols = pb.ncols, halves = pb.halves, rb = pb.r_block;
    const Kokkos::RangePolicy<> policy( 0, size_t( pb.nsub ) * size_t( ncols ) * size_t( nr ) );
    Kokkos::parallel_for(
        "warp_tile_perimeter_reduce", policy, KOKKOS_LAMBDA( const size_t i ) {
            const int r   = int( i % size_t( nr ) );
            const int col = int( ( i / size_t( nr ) ) % size_t( ncols ) );
            const int sub = int( i / size_t( nr ) / size_t( ncols ) );
            int       x, y;
            pb.node_of( col, x, y );
            const bool owner = ( x < hex_lat ) && ( y < hex_lat );
            const bool s1    = ( x % tile == 0 ) && x > 0 && y < hex_lat;
            const bool s2    = ( y % tile == 0 ) && y > 0 && x < hex_lat;
            const bool s3    = x > 0 && y > 0;
            // A level a run below ends at has a second set of edges, from that team.
            const bool two = ( halves > 1 ) && rb > 0 && r > 0 && r < nr - 1 && ( r % rb == 0 );
            for ( int d = 0; d < VecDim; ++d )
            {
                AccT v = ( owner || !replace ) ? static_cast< AccT >( entry( dst, sub, x, y, r, d ) ) : AccT( 0 );
                if ( s1 )
                    v += pb.slot( sub, 0, 1, d, col, r );
                if ( s2 )
                    v += pb.slot( sub, 0, 2, d, col, r );
                if ( s3 )
                    v += pb.slot( sub, 0, 3, d, col, r );
                if ( two )
                {
                    if ( s1 )
                        v += pb.slot( sub, 1, 1, d, col, r );
                    if ( s2 )
                        v += pb.slot( sub, 1, 2, d, col, r );
                    if ( s3 )
                        v += pb.slot( sub, 1, 3, d, col, r );
                }
                entry( dst, sub, x, y, r, d ) = static_cast< S >( v );
            }
        } );
}

/// Phase 2 for runs that are not level-complete: add the team-below owner values of
/// the slab to every column (x, y < hex_lat) of every run-boundary level. Runs before
/// perimeter_reduce, so a perimeter column at such a level gets slab, then edges.
template < typename GridData, typename AccT >
void slab_reduce( GridData dst, const PerimeterBuffer< AccT >& pb )
{
    constexpr int VecDim = grid::grid_data_vec_dim< GridData >();
    using S              = typename GridData::value_type;
    if ( pb.nslab == 0 )
        return;
    const int hex_lat = pb.hex_lat, nslab = pb.nslab, rb = pb.r_block;
    const Kokkos::RangePolicy<> policy( 0, size_t( pb.nsub ) * size_t( nslab ) * size_t( hex_lat ) * size_t( hex_lat ) );
    Kokkos::parallel_for(
        "warp_tile_slab_reduce", policy, KOKKOS_LAMBDA( const size_t i ) {
            const int y   = int( i % size_t( hex_lat ) );
            const int x   = int( ( i / size_t( hex_lat ) ) % size_t( hex_lat ) );
            const int k   = int( ( i / size_t( hex_lat ) / size_t( hex_lat ) ) % size_t( nslab ) );
            const int sub = int( i / size_t( hex_lat ) / size_t( hex_lat ) / size_t( nslab ) );
            const int r   = ( k + 1 ) * rb;
            for ( int d = 0; d < VecDim; ++d )
            {
                S& e = entry( dst, sub, x, y, r, d );
                e    = static_cast< S >( static_cast< AccT >( e ) + pb.slab( sub, d, x, y, r ) );
            }
        } );
}

/// Shared-memory seam between the warps of a team: the top row of warp w hands its
/// from-below corner contributions to the bottom row of warp w + 1. Double buffered
/// over the level parity so one barrier per level suffices.
template < int NC >
KOKKOS_INLINE_FUNCTION constexpr size_t seam_scalars()
{
    return size_t( 2 ) * ( NW - 1 ) * PW * NC * 2;
}

/// One node level, complete in registers across the warp: reduce the 4 corner
/// contributions v[c][d] (c = ddx + 2 ddy) to the owner lane and hand every value to
/// `emit( ddx, ddy, d, value, owned )`: the owner's node (0, 0, owned = true) and
/// the outward edge contributions of the patch (owned = false, to be reduced into
/// another lane's or team's node). Must be called by every lane of the warp.
template < int NC, typename AccT, typename Emit >
KOKKOS_INLINE_FUNCTION void flush_level(
    const Team& team,
    AccT ( &v )[4][NC],
    const int   ltx,
    const int   lty,
    const int   wy,
    AccT*       seam_mem,
    const int   par,
    const bool  active,
    Emit&&      emit )
{
    auto seam = [&]( int p, int w, int col, int d, int corner ) -> AccT& {
        return seam_mem[( ( ( p * ( NW - 1 ) + w ) * PW + col ) * NC + d ) * 2 + corner];
    };

    // Owner reduction. shfl_up leaves the first `delta` lanes with their own value,
    // so those take 0 explicitly; the first row of a warp gets its from-below values
    // through the seam instead.
    AccT own[NC];
#pragma unroll
    for ( int d = 0; d < NC; ++d )
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
            for ( int d = 0; d < NC; ++d )
            {
                seam( par, wy, ltx, d, 0 ) = v[2][d];
                seam( par, wy, ltx, d, 1 ) = v[3][d];
            }
        }
        team.team_barrier();
        if ( lty == 0 && wy > 0 )
        {
#pragma unroll
            for ( int d = 0; d < NC; ++d )
            {
                own[d] += seam( par, wy - 1, ltx, d, 0 );
                own[d] += ( ltx > 0 ) ? seam( par, wy - 1, ltx - 1, d, 1 ) : AccT( 0 );
            }
        }
    }

    if ( active )
    {
#pragma unroll
        for ( int d = 0; d < NC; ++d )
        {
            emit( 0, 0, d, own[d], true );
            // Tile edges: right column goes to column tx+1 of the next tile, the
            // tile's top row to row ty+1.
            if ( ltx == PW - 1 )
            {
                emit( 1, 0, d, v[1][d], false );
                emit( 1, 1, d, v[3][d], false );
            }
            if ( lty == PH - 1 && wy == NW - 1 )
            {
                emit( 0, 1, d, v[2][d], false );
                if ( ltx != PW - 1 )
                    emit( 1, 1, d, v[3][d], false );
            }
        }
    }
}

/// One contribution straight into the global field, in the accumulation type. A
/// 2-byte field has no lock-free scalar atomic: the add goes through the packed pair
/// of the 32-bit word the entry lives in, with 0 for the neighbour (branch-free).
template < typename ScalarType, typename V >
KOKKOS_INLINE_FUNCTION void global_add( ScalarType* p, const V v )
{
    if constexpr ( sizeof( ScalarType ) == 2 )
    {
        const int   odd = int( ( reinterpret_cast< std::uintptr_t >( p ) & 2u ) != 0 );
        const float fv  = static_cast< float >( v );
        util::atomic_add_packed_pair( p - odd, odd ? 0.0f : fv, odd ? fv : 0.0f );
    }
    else
    {
        util::atomic_add_global( p, v );
    }
}

/// Stage the node coordinates of the tile's (lat + 1)^2 columns and the radii of the
/// slots [r0 - 1, r0 - 1 + nslot) into team scratch. Slot s holds level r0 - 1 + s:
/// slot 0 is the level below the run (its cell is the halo), slots 1 .. the run.
template < int nslot, typename GeomT, typename CoordsSh, typename RSh, typename Grid, typename Radii >
KOKKOS_INLINE_FUNCTION void stage_geometry(
    const Team& team,
    CoordsSh&   coords_sh,
    RSh&        r_sh,
    const Grid& grid,
    const Radii& radii,
    const int   sub,
    const int   x0,
    const int   y0,
    const int   r0,
    const int   hex_lat,
    const int   hex_rad )
{
    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy ), [&]( int n ) {
        const int xi = x0 + n % ( lat + 1 );
        const int yi = y0 + n / ( lat + 1 );
        if ( xi <= hex_lat && yi <= hex_lat )
        {
            coords_sh( n, 0 ) = grid( sub, xi, yi, 0 );
            coords_sh( n, 1 ) = grid( sub, xi, yi, 1 );
            coords_sh( n, 2 ) = grid( sub, xi, yi, 2 );
        }
        else
        {
            coords_sh( n, 0 ) = coords_sh( n, 1 ) = coords_sh( n, 2 ) = GeomT( 0 );
        }
    } );
    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nslot ), [&]( int s ) {
        const int rr = r0 - 1 + s;
        r_sh( s )    = ( rr >= 0 && rr <= hex_rad ) ? radii( sub, rr ) : GeomT( 0 );
    } );
}

/// The geometry of one wedge at the one-point quadrature rule (the wedge centroid),
/// from the unit-sphere coordinates of its three columns and the two radii: the
/// physical gradients a_dn of the six shape functions and |det J|. GeoT throughout.
template < typename GeoT >
struct WedgeGeometry
{
    GeoT a0[6], a1[6], a2[6];
    GeoT abs_det;

    KOKKOS_INLINE_FUNCTION WedgeGeometry(
        const GeoT ( &cx )[4],
        const GeoT ( &cy )[4],
        const GeoT ( &cz )[4],
        const int  v0,
        const int  v1,
        const int  v2,
        const GeoT r_0,
        const GeoT r_1 )
    {
        constexpr GeoT ONE_THIRD = static_cast< GeoT >( 1.0 / 3.0 );
        constexpr GeoT ONE_SIXTH = static_cast< GeoT >( 1.0 / 6.0 );
        constexpr GeoT dN_ref[6][3] = { { -0.5, -0.5, -ONE_SIXTH }, { 0.5, 0.0, -ONE_SIXTH }, { 0.0, 0.5, -ONE_SIXTH },
                                        { -0.5, -0.5, ONE_SIXTH },  { 0.5, 0.0, ONE_SIXTH },  { 0.0, 0.5, ONE_SIXTH } };

        const GeoT half_dr = static_cast< GeoT >( 0.5 ) * ( r_1 - r_0 );
        const GeoT r_mid   = static_cast< GeoT >( 0.5 ) * ( r_0 + r_1 );

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
        abs_det            = Kokkos::abs( J_det );
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

#pragma unroll
        for ( int n = 0; n < 6; ++n )
        {
            const GeoT gx = dN_ref[n][0], gy = dN_ref[n][1], gz = dN_ref[n][2];
            a0[n]         = i00 * gx + i01 * gy + i02 * gz;
            a1[n]         = i10 * gx + i11 * gy + i12 * gz;
            a2[n]         = i20 * gx + i21 * gy + i22 * gz;
        }
    }
};


// ---- the P1-isoP2 / P1 coupling: a fine wedge inside its coarse wedge ----

/// Which coarse wedge the pressure of fine wedge w of a fine cell with parities
/// (px, py) is read from. The matrix-free per-cell paths of Gradient and Divergence
/// pair fine wedge w with coarse wedge w. Geometrically the two centre sub-triangles
/// are the other way round: fine wedge 0 of the (1,1) cell lies in coarse wedge 1 and
/// fine wedge 1 of the (0,0) cell in coarse wedge 0 (kernel_helpers.hpp's
/// fine_lateral_wedge_idx gives both the centre index 3, whose lateral weights are the
/// coarse wedge's barycentre). Reproducing the per-cell path is the default, so the
/// two paths agree to roundoff and every result stays what it was; the geometric
/// pairing is one constant away.
constexpr bool legacy_centre_pairing = true;

KOKKOS_INLINE_FUNCTION constexpr int coarse_wedge_of( const int w, const int px, const int py )
{
    if constexpr ( legacy_centre_pairing )
        return w;
    else
        return ( w == 0 ) ? ( ( px && py ) ? 1 : 0 ) : ( ( px || py ) ? 1 : 0 );
}

/// Lateral weights of the three coarse lateral shape functions at the centroid of the
/// fine sub-triangle with index idx in {0, 1, 2, 3} (kernel_helpers.hpp numbering):
/// 2/3 on the shared vertex and 1/6 on the other two for the three corner triangles,
/// 1/3 each for the centre one. shape_lat_coarse( c, idx, 1/3, 1/3 ).
template < typename T >
KOKKOS_INLINE_FUNCTION constexpr T coarse_lateral_weight( const int idx, const int c )
{
    return ( idx == 3 ) ? static_cast< T >( 1.0 / 3.0 )
                        : ( ( c == idx ) ? static_cast< T >( 2.0 / 3.0 ) : static_cast< T >( 1.0 / 6.0 ) );
}

/// Sub-triangle index of fine wedge w in a fine cell with parities (px, py), as
/// kernel_helpers.hpp's fine_lateral_wedge_idx gives it.
KOKKOS_INLINE_FUNCTION constexpr int fine_sub_triangle( const int w, const int px, const int py )
{
    constexpr int indices[2][2][2] = { { { 0, 1 }, { 2, 3 } }, { { 3, 2 }, { 1, 0 } } };
    return indices[w][py][px];
}

/// Radial weight of a coarse bottom node (j < 3) at the centroid (zeta = 0) of the
/// lower (pr = 0) or upper (pr = 1) fine cell of a coarse cell; the top node gets
/// 1 - this. shape_rad_coarse( j, pr, 0 ).
template < typename T >
KOKKOS_INLINE_FUNCTION constexpr T coarse_radial_weight_bottom( const int pr )
{
    return pr ? static_cast< T >( 0.25 ) : static_cast< T >( 0.75 );
}

} // namespace terra::fe::wedge::operators::shell::warp_tile
