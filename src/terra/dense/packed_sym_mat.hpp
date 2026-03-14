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
    /// Uses the lower triangle to compute both M[i][j]*v[j] and M[j][i]*v[i] contributions.
    /// Pass 1 computes the lower-triangular part (sequential packed access).
    /// Pass 2 adds the strictly-upper-triangular part (transpose of lower triangle).
    KOKKOS_INLINE_FUNCTION
    Vec< T, N > operator*( const Vec< T, N >& v ) const
    {
        Vec< T, N > result;

        // Pass 1: Lower triangle contribution (row-wise, sequential packed access).
        // result[i] = sum_{j=0..i} M(i,j) * v(j)
        for ( int i = 0; i < N; ++i )
        {
            T sum = T( 0 );
            const int base = i * ( i + 1 ) / 2;
            for ( int j = 0; j <= i; ++j )
            {
                sum += data[base + j] * v( j );
            }
            result( i ) = sum;
        }

        // Pass 2: Strictly-upper-triangle contribution (column-wise).
        // result[i] += sum_{j=i+1..N-1} M(j,i) * v(j)
        // Process column by column: for column i, accumulate M(j,i)*v(j) for j > i.
        for ( int j = 1; j < N; ++j )
        {
            const T      vj   = v( j );
            const int    base = j * ( j + 1 ) / 2;
            for ( int i = 0; i < j; ++i )
            {
                result( i ) += data[base + i] * vj;
            }
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
};

} // namespace terra::dense
