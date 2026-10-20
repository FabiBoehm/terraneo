#pragma once

#include "util/floating_point.hpp"
#include "terra/linalg/operator.hpp"

namespace terra::linalg::solvers {

/// @brief Power iteration to estimate the largest eigenvalue of a
/// row-normalized operator D^{-1}A, where D^{-1} is the inverted diagonal of A.
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
template < terra::linalg::OperatorLike OperatorT >
//template<typename OperatorT>
double power_iteration(
    OperatorT&          op,
    SrcOf< OperatorT >& tmpIt,
    SrcOf< OperatorT >& tmpAux,
    //VectorQ1Scalar< double>& tmpIt,
    //VectorQ1Scalar< double>& tmpAux,
    const int iterations )
{
    // TODO typecheck on src dst

    // Coefficients are formed in double and narrowed once: the 16-bit Kokkos types
    // only convert out of themselves explicitly, and clang rejects an implicit
    // double -> float narrowing inside a braced initializer list.
    using ScalarType = ScalarOf< SrcOf< OperatorT > >;

    // randomize start
    randomize( tmpIt );

    // normalize
    auto norm = linalg::norm_2( tmpIt );
    lincomb( tmpIt, { util::narrow< ScalarType >( 1.0 / norm ) }, { tmpIt }, util::narrow< ScalarType >( 0.0 ) );

    // apply operator
    apply( op, tmpIt, tmpAux );

    double radius = 0.0;
    for ( int iteration = 0; iteration < iterations; ++iteration )
    {
        // normalize
        norm = linalg::norm_2( tmpAux );
        lincomb( tmpIt, { util::narrow< ScalarType >( 1.0 / norm ) }, { tmpAux }, util::narrow< ScalarType >( 0.0 ) );

        // apply operator
        apply( op, tmpIt, tmpAux );

        // compute radius
        radius = static_cast< double >( dot( tmpIt, tmpAux ) );
    }
    return radius;
}

} // namespace terra::linalg::solvers