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
// reproducible, at the cost of eight launches over the whole grid (the kernels that
// take this path are the coarse levels, the mass operators and the grid transfers,
// all cheap). The warp-owner kernels have their own two-phase scatter (warp_tile.hpp).

#include <Kokkos_Core.hpp>

#include <cstdlib>

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

/// Run launch(-1) once, or, when deterministic, launch(c) for every colour c.
template < typename F >
void for_each_colour( const bool deterministic, F&& launch, const int bits = 1 )
{
    if ( !deterministic )
        launch( -1 );
    else
        for ( int c = 0; c < ( 1 << ( 3 * bits ) ); ++c )
            launch( c );
}

} // namespace terra::fe::wedge::det
