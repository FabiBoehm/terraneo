#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "kokkos/kokkos_wrapper.hpp"

namespace terra::util {

/// @concept FloatingPoint
/// @brief Concept for types usable as a field precision.
///
/// This is `std::floating_point` widened by the two Kokkos 16-bit types. They are
/// class types wrapping `__half` / `__hip_bfloat16`, so `std::floating_point` rejects
/// them even though they provide the full arithmetic interface. On a backend without
/// a native 16-bit type Kokkos aliases both to `float`, where `std::floating_point`
/// already holds and the extra alternatives are redundant -- they are what makes the
/// kernel templates instantiable on a backend where the types are distinct.
/// @tparam T The scalar type to check.
template < typename T >
concept FloatingPoint = std::floating_point< T > || std::is_same_v< T, Kokkos::Experimental::half_t > ||
                        std::is_same_v< T, Kokkos::Experimental::bhalf_t >;

/// @brief Accumulator type for reductions over a field of precision @p T.
///
/// ScalarType for the built-in types, double for the 16-bit ones. A sum over a whole
/// level in 16 bits is not usable: half_t stops representing consecutive integers at
/// 2048 and saturates at 65504, so a dot product either drops its low-order
/// contributions or overflows to inf, which is how a 16-bit V-cycle returns NaN even
/// when every stored field is comfortably in range. Reductions therefore accumulate
/// here and narrow once, on return.
template < FloatingPoint T >
using Accumulator = std::conditional_t< std::floating_point< T >, T, double >;

/// @brief An atomic add that never falls back to a lock.
///
/// Kokkos has no lock-free atomic_add for the 16-bit types: desul guards
/// Lock_Based_Fetch_Op on `!device_atomic_always_lock_free<T>`, which holds for every
/// 2-byte type, and that implementation ballots the warp and spins on a per-address
/// lock, so the lanes serialize. The cost is severe wherever the 4- and 8-byte adds
/// map to a native hardware atomic: on H100 the EpsDivDiv scatter runs 10.5x slower
/// in 16-bit than in double, and the penalty tracks the *storage* type alone -- the
/// arithmetic precision does not matter.
///
/// For a narrow type we therefore run a compare-and-swap loop on the aligned 32-bit
/// word that contains the value. That is lock-free, needs no warp-wide agreement, and
/// contends only with lanes touching the same word rather than with the whole warp.
///
/// @note The enclosing word may reach two bytes past @p dest. Device allocators pad
///       well beyond that, and a Kokkos View base is at least 256-byte aligned, so
///       the access stays inside the allocation; it can touch the neighbouring
///       element, which the CAS retry handles.
/// @tparam T Field precision.
/// @param dest Address to add into.
/// @param value Summand, in the field precision.
template < FloatingPoint T >
KOKKOS_INLINE_FUNCTION void atomic_add_lockfree( T* dest, T value )
{
    if constexpr ( sizeof( T ) >= 4 )
    {
        Kokkos::atomic_add( dest, value );
    }
    else
    {
        static_assert( sizeof( T ) == 2, "expected a 2-byte field precision" );
        using Word = unsigned int;

        const auto      addr  = reinterpret_cast< std::uintptr_t >( dest );
        Word*           word  = reinterpret_cast< Word* >( addr & ~std::uintptr_t( 3 ) );
        const int       shift = static_cast< int >( ( addr & std::uintptr_t( 3 ) ) * 8 );
        const Word      mask  = Word( 0xffffu ) << shift;

        Word old = Kokkos::atomic_load( word );
        Word assumed;
        do
        {
            assumed = old;

            const auto bits = static_cast< std::uint16_t >( ( assumed >> shift ) & 0xffffu );
            const T    cur  = static_cast< T >( Kokkos::bit_cast< T >( bits ) + value );
            const auto out  = Kokkos::bit_cast< std::uint16_t >( cur );
            const Word updated = ( assumed & ~mask ) | ( Word( out ) << shift );

            old = Kokkos::atomic_compare_exchange( word, assumed, updated );
        } while ( old != assumed );
    }
}

/// @brief Unit roundoff of @p T, as a double.
///
/// Kokkos reports the numeric traits of the 16-bit types as bit_comparison_type, and
/// converting one to double on the host goes through the wrapper, which is exactly the
/// path that cannot be relied on with ROCm. The two mantissa lengths are fixed by the
/// formats (fp16 is 1/5/10, bf16 is 1/8/7), so state them.
template < FloatingPoint T >
constexpr double epsilon_of()
{
    if constexpr ( std::floating_point< T > )
        return static_cast< double >( Kokkos::Experimental::epsilon_v< T > );
    else if constexpr ( std::is_same_v< T, Kokkos::Experimental::half_t > )
        return 0x1p-10;
    else
        return 0x1p-7;
}

/// @brief Narrow @p v to @p T, correctly on the host as well as inside a kernel.
///
/// hipcc (ROCm 6.3.4, gfx90a, -O3) miscompiles the host-side double -> half_t
/// conversion: a value it cannot constant-fold comes out as the bit pattern 0x0000.
/// A literal is folded and therefore right, which is what makes this so quiet -- a
/// Chebyshev or FGMRES coefficient computed at runtime on the host silently becomes
/// zero, the V-cycle returns nothing, and the outer solver reports NaN from the
/// preconditioner. bf16 (__hip_bfloat16) is unaffected, as is every conversion that
/// happens on the device.
///
/// So encode IEEE-754 binary16 by hand on that one path, round-to-nearest-even.
/// Everywhere else this is the plain static_cast it looks like.
template < FloatingPoint T >
KOKKOS_INLINE_FUNCTION T narrow( double v )
{
#if defined( KOKKOS_ENABLE_HIP ) && !defined( __HIP_DEVICE_COMPILE__ )
    if constexpr ( std::is_same_v< T, Kokkos::Experimental::half_t > && !std::floating_point< T > )
    {
        static_assert( sizeof( T ) == 2 );

        const float f = static_cast< float >( v );
        uint32_t    b = 0;
        std::memcpy( &b, &f, sizeof( b ) );

        const uint32_t sign      = ( b >> 16 ) & 0x8000u;
        const uint32_t biased    = ( b >> 23 ) & 0xFFu;
        const uint32_t mantissa  = b & 0x7FFFFFu;
        const int32_t  exponent  = static_cast< int32_t >( biased ) - 127 + 15;
        uint16_t       bits      = 0;

        if ( biased == 0xFFu ) // inf or NaN
            bits = static_cast< uint16_t >( sign | 0x7C00u | ( mantissa ? 0x200u : 0u ) );
        else if ( exponent >= 0x1F ) // overflows binary16
            bits = static_cast< uint16_t >( sign | 0x7C00u );
        else if ( exponent <= 0 ) // subnormal, or too small to represent at all
        {
            if ( exponent < -10 )
                bits = static_cast< uint16_t >( sign );
            else
            {
                const uint32_t m     = mantissa | 0x800000u;
                const int      shift = 14 - exponent;
                uint32_t       q     = m >> shift;
                if ( ( m >> ( shift - 1 ) ) & 1u )
                    ++q;
                bits = static_cast< uint16_t >( sign | q );
            }
        }
        else
        {
            uint32_t q = ( static_cast< uint32_t >( exponent ) << 10 ) | ( mantissa >> 13 );
            if ( ( mantissa >> 12 ) & 1u )
                ++q;
            bits = static_cast< uint16_t >( sign | q );
        }

        T out;
        std::memcpy( &out, &bits, sizeof( bits ) );
        return out;
    }
#endif
    return static_cast< T >( v );
}

} // namespace terra::util
