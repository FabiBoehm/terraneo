#pragma once

#include "solver.hpp"

#include <cmath>

namespace terra::linalg::solvers {

/// @brief Jacobi iterative solver for linear systems.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Uses a diagonal preconditioner and supports relaxation.
/// The update rule is:
/// \f[ x^{(k+1)} = x^{(k)} + \omega D^{-1} (b - Ax^{(k)}) \f]
/// where \f$ D \f$ is the diagonal of \f$ A \f$ and \f$ \omega \f$ is the relaxation parameter.
///
/// When chebyshev_lambda_max > 0, uses Chebyshev polynomial acceleration (roots form):
/// each step applies x += (omega/t_j) * D^{-1}(b - Ax) where t_j are roots of the
/// Chebyshev polynomial on [lambda_min, lambda_max].
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
template < OperatorLike OperatorT >
class Jacobi
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Scalar type for computations.
    using ScalarType = SolutionVectorType::ScalarType;

    /// @brief Construct a Jacobi solver.
    /// @param inverse_diagonal Inverse of the diagonal of the operator.
    /// @param iterations Number of Jacobi iterations to perform.
    /// @param tmp Temporary vector for workspace.
    /// @param omega Relaxation parameter (default 1.0).
    /// @param chebyshev_lambda_max Max eigenvalue of omega*D^{-1}*A (0 = disabled).
    Jacobi(
        const SolutionVectorType& inverse_diagonal,
        const int                 iterations,
        const SolutionVectorType& tmp,
        const ScalarType          omega                = 1.0,
        const ScalarType          chebyshev_lambda_max = ScalarType( 0 ) )
    : inverse_diagonal_( inverse_diagonal )
    , iterations_( iterations )
    , tmp_( tmp )
    , omega_( omega )
    , chebyshev_lambda_max_( chebyshev_lambda_max )
    {}

    /// @brief Solve the linear system using Jacobi iteration.
    /// Applies the update rule for the specified number of iterations.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        if ( chebyshev_lambda_max_ > ScalarType( 0 ) )
        {
            solve_chebyshev( A, x, b );
        }
        else
        {
            for ( int iteration = 0; iteration < iterations_; ++iteration )
            {
                apply( A, x, tmp_ );
                lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );
                scale_in_place( tmp_, inverse_diagonal_ );
                lincomb( x, { 1.0, omega_ }, { x, tmp_ } );
            }
        }
    }

    SolutionVectorType& get_inverse_diagonal() {
      return inverse_diagonal_;
    }

  private:
    /// @brief Chebyshev polynomial smoother (roots form).
    void solve_chebyshev( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        const int        degree          = iterations_;
        const ScalarType lambda_min_ratio = ScalarType( std::max( 3, 2 * degree * degree ) );
        const ScalarType lambda_max       = chebyshev_lambda_max_;
        const ScalarType lambda_min       = lambda_max / lambda_min_ratio;
        const ScalarType center           = ( lambda_max + lambda_min ) / ScalarType( 2 );
        const ScalarType half_width       = ( lambda_max - lambda_min ) / ScalarType( 2 );

        for ( int j = 0; j < degree; ++j )
        {
            const ScalarType root = center + half_width * std::cos(
                ScalarType( M_PI ) * ScalarType( 2 * j + 1 ) / ScalarType( 2 * degree ) );

            apply( A, x, tmp_ );
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );
            scale_in_place( tmp_, inverse_diagonal_ );
            lincomb( x, { 1.0, omega_ / root }, { x, tmp_ } );
        }
    }

    SolutionVectorType inverse_diagonal_; ///< Inverse diagonal vector.
    int                iterations_;       ///< Number of iterations.
    SolutionVectorType tmp_;              ///< Temporary workspace vector.
    ScalarType         omega_;            ///< Relaxation parameter.
    ScalarType         chebyshev_lambda_max_; ///< Max eigenvalue of omega*D^{-1}*A (0 = disabled).
};

/// @brief Static assertion: Jacobi satisfies SolverLike concept.
static_assert( SolverLike< Jacobi< linalg::detail::DummyConcreteOperator > > );

} // namespace terra::linalg::solvers
