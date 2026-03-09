#pragma once

#include "solver.hpp"
#include "linalg/operator.hpp"

namespace terra::linalg::solvers {

namespace detail {

template < typename ScalarT >
KOKKOS_INLINE_FUNCTION void invert_3x3( const ScalarT* M, ScalarT* Minv )
{
    const ScalarT det = M[0] * ( M[4] * M[8] - M[5] * M[7] ) - M[1] * ( M[3] * M[8] - M[5] * M[6] ) +
                        M[2] * ( M[3] * M[7] - M[4] * M[6] );

    if ( det == ScalarT( 0 ) )
    {
        for ( int i = 0; i < 9; i++ )
            Minv[i] = ScalarT( 0 );
        return;
    }

    const ScalarT inv_det = ScalarT( 1 ) / det;
    Minv[0]               = ( M[4] * M[8] - M[5] * M[7] ) * inv_det;
    Minv[1]               = ( M[2] * M[7] - M[1] * M[8] ) * inv_det;
    Minv[2]               = ( M[1] * M[5] - M[2] * M[4] ) * inv_det;
    Minv[3]               = ( M[5] * M[6] - M[3] * M[8] ) * inv_det;
    Minv[4]               = ( M[0] * M[8] - M[2] * M[6] ) * inv_det;
    Minv[5]               = ( M[2] * M[3] - M[0] * M[5] ) * inv_det;
    Minv[6]               = ( M[3] * M[7] - M[4] * M[6] ) * inv_det;
    Minv[7]               = ( M[1] * M[6] - M[0] * M[7] ) * inv_det;
    Minv[8]               = ( M[0] * M[4] - M[1] * M[3] ) * inv_det;
}

template < typename ScalarT >
KOKKOS_INLINE_FUNCTION void matmul_3x3( const ScalarT* A, const ScalarT* B, ScalarT* C )
{
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
        {
            C[i * 3 + j] = ScalarT( 0 );
            for ( int k = 0; k < 3; k++ )
                C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
        }
}

template < typename ScalarT >
KOKKOS_INLINE_FUNCTION void matvec_3x3( const ScalarT* A, const ScalarT* x, ScalarT* y )
{
    for ( int i = 0; i < 3; i++ )
    {
        y[i] = ScalarT( 0 );
        for ( int j = 0; j < 3; j++ )
            y[i] += A[i * 3 + j] * x[j];
    }
}

} // namespace detail

/// @brief Radial line smoother for vectorial operators on spherical shell grids.
///
/// Precomputes the 3×3 block-tridiagonal system along each radial line (sd, x, y)
/// from the operator's element matrices. In each smoothing sweep, computes the
/// residual r = b - Ax, solves the block-tridiagonal system using the Thomas
/// algorithm (block LU decomposition), and updates x += omega * correction.
///
/// This smoother is particularly effective for problems with radial viscosity
/// jumps (e.g., Lin et al. 2022, Stotz et al. 2017 profiles) because it
/// captures the strong radial coupling exactly.
///
/// Satisfies the SolverLike concept (see solver.hpp).
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike and provide get_local_matrix).
template < OperatorLike OperatorT >
class RadialLineSmoother
{
  public:
    using OperatorType      = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType     = DstOf< OperatorType >;
    using ScalarType        = SolutionVectorType::ScalarType;

    /// @brief Construct a radial line smoother.
    /// @param op Operator to extract radial tridiagonal from (copied by value for device access).
    /// @param iterations Number of smoothing sweeps per solve call.
    /// @param tmp Temporary vector for residual/correction workspace.
    /// @param omega Relaxation parameter (default 1.0).
    RadialLineSmoother(
        OperatorT                 op,
        const int                 iterations,
        const SolutionVectorType& tmp,
        const ScalarType          omega = 1.0 )
    : iterations_( iterations )
    , tmp_( tmp )
    , omega_( omega )
    {
        const auto& domain = op.get_domain();
        num_sd_             = static_cast< int >( domain.subdomains().size() );
        nx_                 = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        ny_                 = nx_;
        nr_                 = domain.domain_info().subdomain_num_nodes_radially();

        diag_blocks_  = Kokkos::View< ScalarType*****, Kokkos::LayoutRight >( "rls_diag", num_sd_, nx_, ny_, nr_, 9 );
        lower_blocks_ = Kokkos::View< ScalarType*****, Kokkos::LayoutRight >( "rls_lower", num_sd_, nx_, ny_, nr_, 9 );
        upper_blocks_ = Kokkos::View< ScalarType*****, Kokkos::LayoutRight >( "rls_upper", num_sd_, nx_, ny_, nr_, 9 );

        precompute_tridiagonal( op );
    }

    /// @brief Solve: apply one or more radial-line smoothing sweeps.
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iter = 0; iter < iterations_; iter++ )
        {
            apply( A, x, tmp_ );
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );
            thomas_solve();
            lincomb( x, { 1.0, omega_ }, { x, tmp_ } );
        }
    }

    // Public for CUDA extended lambda compatibility (cannot use KOKKOS_LAMBDA in private methods).
    void precompute_tridiagonal( OperatorT op )
    {
        auto diag  = diag_blocks_;
        auto lower = lower_blocks_;
        auto upper = upper_blocks_;

        const int ncx = nx_ - 1;
        const int ncy = ny_ - 1;
        const int ncr = nr_ - 1;
        const int nr  = nr_;

        auto local_idx_in_wedge = KOKKOS_LAMBDA( int dx, int dy, int dr, int w ) -> int
        {
            if ( w == 0 )
            {
                if ( dx + dy > 1 )
                    return -1;
                int base = dr * 3;
                if ( dx == 0 && dy == 0 )
                    return base;
                if ( dx == 1 && dy == 0 )
                    return base + 1;
                if ( dx == 0 && dy == 1 )
                    return base + 2;
                return -1;
            }
            else
            {
                if ( dx + dy < 1 )
                    return -1;
                int base = dr * 3;
                if ( dx == 1 && dy == 1 )
                    return base;
                if ( dx == 0 && dy == 1 )
                    return base + 1;
                if ( dx == 1 && dy == 0 )
                    return base + 2;
                return -1;
            }
        };

        Kokkos::parallel_for(
            "precompute_radial_tridiag",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sd_, nx_, ny_, nr_ } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                // Boundary nodes: identity diagonal, zero off-diagonals.
                if ( r == 0 || r == nr - 1 )
                {
                    for ( int i = 0; i < 9; i++ )
                    {
                        diag( sd, x, y, r, i )  = ( i == 0 || i == 4 || i == 8 ) ? ScalarType( 1 ) : ScalarType( 0 );
                        lower( sd, x, y, r, i ) = ScalarType( 0 );
                        upper( sd, x, y, r, i ) = ScalarType( 0 );
                    }
                    return;
                }

                // Interior nodes: assemble from element matrices.
                ScalarType D[9] = {}, L[9] = {}, U[9] = {};

                for ( int dhx = 0; dhx <= 1; dhx++ )
                {
                    for ( int dhy = 0; dhy <= 1; dhy++ )
                    {
                        for ( int dhr = 0; dhr <= 1; dhr++ )
                        {
                            const int hx = x - dhx;
                            const int hy = y - dhy;
                            const int hr = r - dhr;
                            if ( hx < 0 || hx >= ncx || hy < 0 || hy >= ncy || hr < 0 || hr >= ncr )
                                continue;

                            for ( int w = 0; w < 2; w++ )
                            {
                                const int fine_lidx = local_idx_in_wedge( dhx, dhy, dhr, w );
                                if ( fine_lidx < 0 )
                                    continue;

                                auto A = op.get_local_matrix( sd, hx, hy, hr, w );

                                // Diagonal block: self-coupling.
                                for ( int d1 = 0; d1 < 3; d1++ )
                                    for ( int d2 = 0; d2 < 3; d2++ )
                                        D[d1 * 3 + d2] += A( fine_lidx + d1 * 6, fine_lidx + d2 * 6 );

                                // Lower block: coupling to (x,y,r-1).
                                // Only hex cells at hr = r-1 (i.e., dhr == 1) contain both r (at dr=1) and r-1 (at dr=0).
                                if ( dhr == 1 )
                                {
                                    const int lower_lidx = local_idx_in_wedge( dhx, dhy, 0, w );
                                    if ( lower_lidx >= 0 )
                                    {
                                        for ( int d1 = 0; d1 < 3; d1++ )
                                            for ( int d2 = 0; d2 < 3; d2++ )
                                                L[d1 * 3 + d2] +=
                                                    A( fine_lidx + d1 * 6, lower_lidx + d2 * 6 );
                                    }
                                }

                                // Upper block: coupling to (x,y,r+1).
                                // Only hex cells at hr = r (i.e., dhr == 0) contain both r (at dr=0) and r+1 (at dr=1).
                                if ( dhr == 0 )
                                {
                                    const int upper_lidx = local_idx_in_wedge( dhx, dhy, 1, w );
                                    if ( upper_lidx >= 0 )
                                    {
                                        for ( int d1 = 0; d1 < 3; d1++ )
                                            for ( int d2 = 0; d2 < 3; d2++ )
                                                U[d1 * 3 + d2] +=
                                                    A( fine_lidx + d1 * 6, upper_lidx + d2 * 6 );
                                    }
                                }
                            }
                        }
                    }
                }

                for ( int i = 0; i < 9; i++ )
                {
                    diag( sd, x, y, r, i )  = D[i];
                    lower( sd, x, y, r, i ) = L[i];
                    upper( sd, x, y, r, i ) = U[i];
                }
            } );

        Kokkos::fence();
    }

    void thomas_solve()
    {
        auto diag     = diag_blocks_;
        auto lower    = lower_blocks_;
        auto upper    = upper_blocks_;
        auto tmp_data = tmp_.grid_data();
        const int nr  = nr_;

        Kokkos::parallel_for(
            "thomas_solve_radial",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_sd_, nx_, ny_ } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y ) {
                // Work arrays: modified diagonal blocks and RHS.
                // Max 64 radial nodes should be sufficient for any practical level.
                constexpr int MAX_NR = 64;
                ScalarType    D_mod[MAX_NR * 9];
                ScalarType    r_mod[MAX_NR * 3];

                // Initialize from precomputed blocks and residual.
                for ( int r = 0; r < nr; r++ )
                {
                    for ( int i = 0; i < 9; i++ )
                        D_mod[r * 9 + i] = diag( sd, x, y, r, i );
                    for ( int d = 0; d < 3; d++ )
                        r_mod[r * 3 + d] = tmp_data( sd, x, y, r, d );
                }

                // Forward sweep: eliminate lower diagonal.
                for ( int r = 1; r < nr; r++ )
                {
                    // Dinv = D'^{-1}_{r-1}
                    ScalarType Dinv[9];
                    detail::invert_3x3( &D_mod[( r - 1 ) * 9], Dinv );

                    // m = L_r * Dinv
                    ScalarType L_r[9];
                    for ( int i = 0; i < 9; i++ )
                        L_r[i] = lower( sd, x, y, r, i );
                    ScalarType m[9];
                    detail::matmul_3x3( L_r, Dinv, m );

                    // D'_r -= m * U_{r-1}
                    ScalarType U_rm1[9];
                    for ( int i = 0; i < 9; i++ )
                        U_rm1[i] = upper( sd, x, y, r - 1, i );
                    ScalarType mU[9];
                    detail::matmul_3x3( m, U_rm1, mU );
                    for ( int i = 0; i < 9; i++ )
                        D_mod[r * 9 + i] -= mU[i];

                    // r'_r -= m * r'_{r-1}
                    ScalarType mr[3];
                    detail::matvec_3x3( m, &r_mod[( r - 1 ) * 3], mr );
                    for ( int d = 0; d < 3; d++ )
                        r_mod[r * 3 + d] -= mr[d];
                }

                // Back substitution.
                {
                    ScalarType Dinv[9];
                    detail::invert_3x3( &D_mod[( nr - 1 ) * 9], Dinv );
                    ScalarType e[3];
                    detail::matvec_3x3( Dinv, &r_mod[( nr - 1 ) * 3], e );
                    for ( int d = 0; d < 3; d++ )
                        tmp_data( sd, x, y, nr - 1, d ) = e[d];
                }

                for ( int r = nr - 2; r >= 0; r-- )
                {
                    // rhs = r'_r - U_r * e_{r+1}
                    ScalarType U_r[9];
                    for ( int i = 0; i < 9; i++ )
                        U_r[i] = upper( sd, x, y, r, i );
                    ScalarType e_next[3];
                    for ( int d = 0; d < 3; d++ )
                        e_next[d] = tmp_data( sd, x, y, r + 1, d );
                    ScalarType Ue[3];
                    detail::matvec_3x3( U_r, e_next, Ue );
                    ScalarType rhs[3];
                    for ( int d = 0; d < 3; d++ )
                        rhs[d] = r_mod[r * 3 + d] - Ue[d];

                    ScalarType Dinv[9];
                    detail::invert_3x3( &D_mod[r * 9], Dinv );
                    ScalarType e[3];
                    detail::matvec_3x3( Dinv, rhs, e );
                    for ( int d = 0; d < 3; d++ )
                        tmp_data( sd, x, y, r, d ) = e[d];
                }
            } );

        Kokkos::fence();
    }

    int                iterations_;
    SolutionVectorType tmp_;
    ScalarType         omega_;

    Kokkos::View< ScalarType*****, Kokkos::LayoutRight > diag_blocks_;
    Kokkos::View< ScalarType*****, Kokkos::LayoutRight > lower_blocks_;
    Kokkos::View< ScalarType*****, Kokkos::LayoutRight > upper_blocks_;

    int num_sd_, nx_, ny_, nr_;
};

/// @brief Hybrid Jacobi + Radial Line smoother.
///
/// Each smoothing sweep applies one Jacobi iteration (handles lateral coupling)
/// followed by one radial line solve (handles strong radial coupling).
/// This combination is effective for problems with radial viscosity jumps.
///
/// Satisfies the SolverLike concept (see solver.hpp).
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike and provide get_local_matrix).
template < OperatorLike OperatorT >
class HybridJacobiRadialLineSmoother
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;
    using ScalarType         = SolutionVectorType::ScalarType;

    /// @brief Construct a hybrid Jacobi + radial line smoother.
    /// @param op Operator to extract radial tridiagonal from.
    /// @param inverse_diagonal Inverse of the diagonal of the operator.
    /// @param iterations Number of hybrid sweeps per solve call.
    /// @param tmp Temporary vector for workspace.
    /// @param omega_jacobi Relaxation parameter for Jacobi (from power iteration).
    /// @param omega_line Relaxation parameter for the line solve (default 1.0).
    HybridJacobiRadialLineSmoother(
        OperatorT                 op,
        const SolutionVectorType& inverse_diagonal,
        const int                 iterations,
        const SolutionVectorType& tmp,
        const ScalarType          omega_jacobi,
        const ScalarType          omega_line = 1.0 )
    : iterations_( iterations )
    , inverse_diagonal_( inverse_diagonal )
    , tmp_( tmp )
    , omega_jacobi_( omega_jacobi )
    , omega_line_( omega_line )
    {
        const auto& domain = op.get_domain();
        num_sd_             = static_cast< int >( domain.subdomains().size() );
        nx_                 = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        ny_                 = nx_;
        nr_                 = domain.domain_info().subdomain_num_nodes_radially();

        diag_blocks_  = Kokkos::View< ScalarType*****, Kokkos::LayoutRight >( "rls_diag", num_sd_, nx_, ny_, nr_, 9 );
        lower_blocks_ = Kokkos::View< ScalarType*****, Kokkos::LayoutRight >( "rls_lower", num_sd_, nx_, ny_, nr_, 9 );
        upper_blocks_ = Kokkos::View< ScalarType*****, Kokkos::LayoutRight >( "rls_upper", num_sd_, nx_, ny_, nr_, 9 );

        precompute_tridiagonal( op );
    }

    /// @brief Solve: apply hybrid Jacobi + radial line sweeps.
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iter = 0; iter < iterations_; iter++ )
        {
            // 1. Jacobi sweep: x += omega_j * D^{-1} * (b - Ax)
            apply( A, x, tmp_ );
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );
            scale_in_place( tmp_, inverse_diagonal_ );
            lincomb( x, { 1.0, omega_jacobi_ }, { x, tmp_ } );

            // 2. Radial line sweep: x += omega_l * M^{-1} * (b - Ax)
            apply( A, x, tmp_ );
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );
            thomas_solve();
            lincomb( x, { 1.0, omega_line_ }, { x, tmp_ } );
        }
    }

    SolutionVectorType& get_inverse_diagonal() { return inverse_diagonal_; }

    // Public for CUDA extended lambda compatibility.
    void precompute_tridiagonal( OperatorT op )
    {
        auto diag  = diag_blocks_;
        auto lower = lower_blocks_;
        auto upper = upper_blocks_;

        const int ncx = nx_ - 1;
        const int ncy = ny_ - 1;
        const int ncr = nr_ - 1;
        const int nr  = nr_;

        auto local_idx_in_wedge = KOKKOS_LAMBDA( int dx, int dy, int dr, int w ) -> int
        {
            if ( w == 0 )
            {
                if ( dx + dy > 1 )
                    return -1;
                int base = dr * 3;
                if ( dx == 0 && dy == 0 )
                    return base;
                if ( dx == 1 && dy == 0 )
                    return base + 1;
                if ( dx == 0 && dy == 1 )
                    return base + 2;
                return -1;
            }
            else
            {
                if ( dx + dy < 1 )
                    return -1;
                int base = dr * 3;
                if ( dx == 1 && dy == 1 )
                    return base;
                if ( dx == 0 && dy == 1 )
                    return base + 1;
                if ( dx == 1 && dy == 0 )
                    return base + 2;
                return -1;
            }
        };

        Kokkos::parallel_for(
            "precompute_radial_tridiag",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sd_, nx_, ny_, nr_ } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                if ( r == 0 || r == nr - 1 )
                {
                    for ( int i = 0; i < 9; i++ )
                    {
                        diag( sd, x, y, r, i )  = ( i == 0 || i == 4 || i == 8 ) ? ScalarType( 1 ) : ScalarType( 0 );
                        lower( sd, x, y, r, i ) = ScalarType( 0 );
                        upper( sd, x, y, r, i ) = ScalarType( 0 );
                    }
                    return;
                }

                ScalarType D[9] = {}, L[9] = {}, U[9] = {};

                for ( int dhx = 0; dhx <= 1; dhx++ )
                {
                    for ( int dhy = 0; dhy <= 1; dhy++ )
                    {
                        for ( int dhr = 0; dhr <= 1; dhr++ )
                        {
                            const int hx = x - dhx;
                            const int hy = y - dhy;
                            const int hr = r - dhr;
                            if ( hx < 0 || hx >= ncx || hy < 0 || hy >= ncy || hr < 0 || hr >= ncr )
                                continue;

                            for ( int w = 0; w < 2; w++ )
                            {
                                const int fine_lidx = local_idx_in_wedge( dhx, dhy, dhr, w );
                                if ( fine_lidx < 0 )
                                    continue;

                                auto A = op.get_local_matrix( sd, hx, hy, hr, w );

                                for ( int d1 = 0; d1 < 3; d1++ )
                                    for ( int d2 = 0; d2 < 3; d2++ )
                                        D[d1 * 3 + d2] += A( fine_lidx + d1 * 6, fine_lidx + d2 * 6 );

                                if ( dhr == 1 )
                                {
                                    const int lower_lidx = local_idx_in_wedge( dhx, dhy, 0, w );
                                    if ( lower_lidx >= 0 )
                                    {
                                        for ( int d1 = 0; d1 < 3; d1++ )
                                            for ( int d2 = 0; d2 < 3; d2++ )
                                                L[d1 * 3 + d2] +=
                                                    A( fine_lidx + d1 * 6, lower_lidx + d2 * 6 );
                                    }
                                }

                                if ( dhr == 0 )
                                {
                                    const int upper_lidx = local_idx_in_wedge( dhx, dhy, 1, w );
                                    if ( upper_lidx >= 0 )
                                    {
                                        for ( int d1 = 0; d1 < 3; d1++ )
                                            for ( int d2 = 0; d2 < 3; d2++ )
                                                U[d1 * 3 + d2] +=
                                                    A( fine_lidx + d1 * 6, upper_lidx + d2 * 6 );
                                    }
                                }
                            }
                        }
                    }
                }

                for ( int i = 0; i < 9; i++ )
                {
                    diag( sd, x, y, r, i )  = D[i];
                    lower( sd, x, y, r, i ) = L[i];
                    upper( sd, x, y, r, i ) = U[i];
                }
            } );

        Kokkos::fence();
    }

    void thomas_solve()
    {
        auto diag     = diag_blocks_;
        auto lower    = lower_blocks_;
        auto upper    = upper_blocks_;
        auto tmp_data = tmp_.grid_data();
        const int nr  = nr_;

        Kokkos::parallel_for(
            "thomas_solve_radial",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_sd_, nx_, ny_ } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y ) {
                constexpr int MAX_NR = 64;
                ScalarType    D_mod[MAX_NR * 9];
                ScalarType    r_mod[MAX_NR * 3];

                for ( int r = 0; r < nr; r++ )
                {
                    for ( int i = 0; i < 9; i++ )
                        D_mod[r * 9 + i] = diag( sd, x, y, r, i );
                    for ( int d = 0; d < 3; d++ )
                        r_mod[r * 3 + d] = tmp_data( sd, x, y, r, d );
                }

                for ( int r = 1; r < nr; r++ )
                {
                    ScalarType Dinv[9];
                    detail::invert_3x3( &D_mod[( r - 1 ) * 9], Dinv );

                    ScalarType L_r[9];
                    for ( int i = 0; i < 9; i++ )
                        L_r[i] = lower( sd, x, y, r, i );
                    ScalarType m[9];
                    detail::matmul_3x3( L_r, Dinv, m );

                    ScalarType U_rm1[9];
                    for ( int i = 0; i < 9; i++ )
                        U_rm1[i] = upper( sd, x, y, r - 1, i );
                    ScalarType mU[9];
                    detail::matmul_3x3( m, U_rm1, mU );
                    for ( int i = 0; i < 9; i++ )
                        D_mod[r * 9 + i] -= mU[i];

                    ScalarType mr[3];
                    detail::matvec_3x3( m, &r_mod[( r - 1 ) * 3], mr );
                    for ( int d = 0; d < 3; d++ )
                        r_mod[r * 3 + d] -= mr[d];
                }

                {
                    ScalarType Dinv[9];
                    detail::invert_3x3( &D_mod[( nr - 1 ) * 9], Dinv );
                    ScalarType e[3];
                    detail::matvec_3x3( Dinv, &r_mod[( nr - 1 ) * 3], e );
                    for ( int d = 0; d < 3; d++ )
                        tmp_data( sd, x, y, nr - 1, d ) = e[d];
                }

                for ( int r = nr - 2; r >= 0; r-- )
                {
                    ScalarType U_r[9];
                    for ( int i = 0; i < 9; i++ )
                        U_r[i] = upper( sd, x, y, r, i );
                    ScalarType e_next[3];
                    for ( int d = 0; d < 3; d++ )
                        e_next[d] = tmp_data( sd, x, y, r + 1, d );
                    ScalarType Ue[3];
                    detail::matvec_3x3( U_r, e_next, Ue );
                    ScalarType rhs[3];
                    for ( int d = 0; d < 3; d++ )
                        rhs[d] = r_mod[r * 3 + d] - Ue[d];

                    ScalarType Dinv[9];
                    detail::invert_3x3( &D_mod[r * 9], Dinv );
                    ScalarType e[3];
                    detail::matvec_3x3( Dinv, rhs, e );
                    for ( int d = 0; d < 3; d++ )
                        tmp_data( sd, x, y, r, d ) = e[d];
                }
            } );

        Kokkos::fence();
    }

    int                iterations_;
    SolutionVectorType inverse_diagonal_;
    SolutionVectorType tmp_;
    ScalarType         omega_jacobi_;
    ScalarType         omega_line_;

    Kokkos::View< ScalarType*****, Kokkos::LayoutRight > diag_blocks_;
    Kokkos::View< ScalarType*****, Kokkos::LayoutRight > lower_blocks_;
    Kokkos::View< ScalarType*****, Kokkos::LayoutRight > upper_blocks_;

    int num_sd_, nx_, ny_, nr_;
};

} // namespace terra::linalg::solvers
