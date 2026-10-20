#pragma once

#include <concepts>
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

} // namespace terra::util
