#pragma once

// Deterministic scatter of the per-cell kernels: cell colouring.
//
// A per-cell kernel scatters each cell's contributions to its eight nodes with
// atomics, so the order in which the cells around a node add to it depends on the
// scheduling, and two applies of the same operator differ in the last bit. With
// TERRA_DETERMINISTIC_SCATTER=1 every such kernel is launched eight times instead,
// once per parity colour (x, y, r mod 2) of the cell, and a launch skips the cells
// of the other colours. No two cells of one colour share a node, so each node gets
// exactly one contribution per launch and the sum is accumulated in colour order:
// reproducible. Each launch iterates the cells of its colour only (for_cells): a
// launch over the whole grid that skips seven eighths of the cells costs as much as
// the full launch on a GPU, the lanes of a warp wait for the one that works, which
// made the slow EpsDivDiv path of the Galerkin-coarsened levels eight times slower.
// The warp-owner kernels have their own two-phase scatter (warp_tile.hpp).

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <string>

namespace terra::fe::wedge::det {

namespace detail {
inline bool& override_flag()
{
    static bool v = false;
    return v;
}
inline bool& override_value()
{
    static bool v = false;
    return v;
}
} // namespace detail

/// Whether the deterministic scatter is requested: TERRA_DETERMINISTIC_SCATTER=1 in
/// the environment, or set_requested().
inline bool requested()
{
    if ( detail::override_flag() )
        return detail::override_value();
    static const bool env = []() {
        const char* e = std::getenv( "TERRA_DETERMINISTIC_SCATTER" );
        return e != nullptr && std::atoi( e ) != 0;
    }();
    return env;
}

/// Override the environment (tests).
inline void set_requested( const bool on )
{
    detail::override_flag()  = true;
    detail::override_value() = on;
}

/// Whether cell (x, y, r) is not of `colour` (-1: no colouring, nothing is skipped).
/// `bits` = 1: parity colouring, 8 colours, for a kernel whose cells scatter to their
/// own nodes; 2: x, y, r mod 4, 64 colours, for a fine-cell kernel that scatters to
/// the coarse nodes (the two fine cells of a coarse cell share them, and so do the
/// fine cells of neighbouring coarse cells).
KOKKOS_INLINE_FUNCTION bool skip( const int colour, const int x, const int y, const int r, const int bits = 1 )
{
    const int m = ( 1 << bits ) - 1;
    return colour >= 0 && ( ( x & m ) | ( ( y & m ) << bits ) | ( ( r & m ) << ( 2 * bits ) ) ) != colour;
}

/// Lateral parity colouring (4 colours) for a kernel whose thread handles a whole
/// column of cells in sequence.
KOKKOS_INLINE_FUNCTION bool skip_lat( const int colour, const int x, const int y )
{
    return colour >= 0 && ( ( x & 1 ) | ( ( y & 1 ) << 1 ) ) != colour;
}

/// Number of indices in [0, n) congruent to `rem` modulo 2^bits.
inline long long colour_extent( const long long n, const int rem, const int bits = 1 )
{
    const long long s = 1 << bits;
    return n > rem ? ( n - rem + s - 1 ) / s : 0;
}

/// A cell functor f( subdomain, x, y, r ) over the compact index space of one colour:
/// index i along x stands for the cell x0 + (i << bits).
template < typename F >
struct ColouredCells
{
    F   f;
    int s0, x0, y0, r0, bits;

    KOKKOS_INLINE_FUNCTION void operator()( const int s, const int xi, const int yi, const int ri ) const
    {
        f( s0 + s, x0 + ( xi << bits ), y0 + ( yi << bits ), r0 + ( ri << bits ) );
    }
};

/// Run f( subdomain, x, y, r ) over the cells of `cells` (a rank-4 MDRangePolicy,
/// subdomain first), or, when deterministic, once per colour over the cells of that
/// colour (parity relative to the policy's lower bound), in colour order.
template < typename Policy, typename F >
void for_cells( const std::string& name, const Policy& cells, const bool deterministic, const F& f, const int bits = 1 )
{
    if ( !deterministic )
    {
        Kokkos::parallel_for( name, cells, f );
        return;
    }
    const int  m  = ( 1 << bits ) - 1;
    const auto lo = cells.m_lower;
    const auto hi = cells.m_upper;
    for ( int c = 0; c < ( 1 << ( 3 * bits ) ); ++c )
    {
        const int cx = c & m, cy = ( c >> bits ) & m, cr = ( c >> ( 2 * bits ) ) & m;
        typename Policy::point_type l{}, u{};
        u[0] = hi[0] - lo[0];
        u[1] = colour_extent( hi[1] - lo[1], cx, bits );
        u[2] = colour_extent( hi[2] - lo[2], cy, bits );
        u[3] = colour_extent( hi[3] - lo[3], cr, bits );
        if ( u[0] <= 0 || u[1] <= 0 || u[2] <= 0 || u[3] <= 0 )
            continue;
        Kokkos::parallel_for(
            name,
            Policy( l, u ),
            ColouredCells< F >{
                f, int( lo[0] ), int( lo[1] ) + cx, int( lo[2] ) + cy, int( lo[3] ) + cr, bits } );
    }
}

} // namespace terra::fe::wedge::det
