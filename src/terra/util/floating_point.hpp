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

} // namespace terra::util
