#pragma once

#include "../kokkos/kokkos_wrapper.hpp"
#include "./mat.hpp"
#include "./vec.hpp"

namespace terra::dense {

/// @brief Packed symmetric matrix storing the lower triangle of M.
///
/// For an N×N symmetric matrix M, stores only the lower triangle (including diagonal),
/// using N*(N+1)/2 entries instead of N*N. For N=24 (Vanka cell): 300 vs 576 doubles.
///
/// The matrix-vector product exploits symmetry in two passes over the packed data:
/// 1. Lower triangle pass: for each row i, dot product with columns 0..i
/// 2. Transpose contribution: add M(j,i) * v(j) for j > i (reads from lower triangle)
///
/// @tparam T Scalar type.
/// @tparam N Matrix dimension.
template < typename T, int N >
struct PackedSymMat
{
    static constexpr int dim  = N;
    static constexpr int size = N * ( N + 1 ) / 2;

    T data[size] = {};

    /// @brief Access M(i, j) where j <= i (lower triangle).
    KOKKOS_INLINE_FUNCTION
    T& operator()( int i, int j ) { return data[i * ( i + 1 ) / 2 + j]; }

    KOKKOS_INLINE_FUNCTION
    const T& operator()( int i, int j ) const { return data[i * ( i + 1 ) / 2 + j]; }

    /// @brief Symmetric matrix-vector product: result = M * v.
    ///
    /// Supports mixed precision: storage type T may differ from vector type U.
    /// When T=float and U=double, matrix entries are promoted to double for computation,
    /// halving memory bandwidth while maintaining double-precision arithmetic.
    ///
    /// Single-pass algorithm: reads each packed entry exactly once.
    /// For each row i and column j <= i:
    ///   - result[i] += M(i,j) * v(j)      (lower triangle)
    ///   - result[j] += M(i,j) * v(i)      (upper triangle, j < i only)
    /// This halves reads of the packed data compared to a two-pass approach
    /// (300 vs 576 reads for N=24).
    template < typename U = T >
    KOKKOS_INLINE_FUNCTION
    Vec< U, N > operator*( const Vec< U, N >& v ) const
    {
        Vec< U, N > result;
        for ( int i = 0; i < N; ++i )
            result( i ) = U( 0 );

        for ( int i = 0; i < N; ++i )
        {
            const int base = i * ( i + 1 ) / 2;
            const U   vi   = v( i );
            U         sum  = U( 0 );

            // Off-diagonal entries: contribute to both result[i] and result[j].
            for ( int j = 0; j < i; ++j )
            {
                const U mij = static_cast< U >( data[base + j] );
                sum += mij * v( j );
                result( j ) += mij * vi;
            }

            // Diagonal entry: contributes only to result[i].
            sum += static_cast< U >( data[base + i] ) * vi;
            result( i ) += sum;
        }

        return result;
    }

    /// @brief Pack the lower triangle of a full symmetric matrix.
    ///
    /// @param M Full symmetric matrix (only lower triangle is read).
    /// @return Packed lower triangle.
    KOKKOS_INLINE_FUNCTION
    static PackedSymMat from_symmetric( const Mat< T, N, N >& M )
    {
        PackedSymMat P;

        for ( int i = 0; i < N; ++i )
        {
            const int base = i * ( i + 1 ) / 2;
            for ( int j = 0; j <= i; ++j )
            {
                P.data[base + j] = M.data[i][j];
            }
        }

        return P;
    }

    /// @brief Pack the lower triangle of a full symmetric matrix with type conversion.
    ///
    /// Converts from source scalar type U to storage type T during packing.
    /// Used for mixed-precision storage (e.g., invert in double, store as float).
    ///
    /// @tparam U Source scalar type (e.g., double).
    /// @param M Full symmetric matrix in type U.
    /// @return Packed lower triangle in storage type T.
    template < typename U >
    KOKKOS_INLINE_FUNCTION
    static PackedSymMat from_symmetric( const Mat< U, N, N >& M )
    {
        PackedSymMat P;

        for ( int i = 0; i < N; ++i )
        {
            const int base = i * ( i + 1 ) / 2;
            for ( int j = 0; j <= i; ++j )
            {
                P.data[base + j] = static_cast< T >( M.data[i][j] );
            }
        }

        return P;
    }
};

} // namespace terra::dense
