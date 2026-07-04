#pragma once

#include <cmath>
#include <string>
#include <utility>

#include "linalg/operator.hpp"
#include "linalg/solvers/solver.hpp"
#include "linalg/vector.hpp"
#include "util/logging.hpp"

namespace terra::mantlecirculation {

/// @brief Thin SolverLike wrapper that runs an inner velocity-block solver and,
///        when `enabled`, logs `||b - A x|| / ||b||` after every application.
///
/// Held by value in `BlockTriangularPreconditioner2x2` (so copyable); the inner
/// solver is referenced by raw pointer since its lifetime is owned elsewhere
/// (typically a `unique_ptr` on the orchestrating context).  The residual
/// buffer `tmp_` is a shallow-handle copy of a velocity vector, so wrapper
/// copies share the underlying buffer — fine for the single-threaded outer
/// Krylov loop where this is used.
template < linalg::OperatorLike OperatorT, linalg::solvers::SolverLike Inner >
class DiagnosticVelocityBlockSolver
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;
    using ScalarType         = typename SolutionVectorType::ScalarType;

    DiagnosticVelocityBlockSolver(
        Inner&             inner,
        SolutionVectorType tmp,
        bool               enabled = false,
        std::string        label   = "viscous_pc" )
    : inner_( &inner )
    , tmp_( std::move( tmp ) )
    , enabled_( enabled )
    , label_( std::move( label ) )
    {}

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        if ( !enabled_ )
        {
            linalg::solvers::solve( *inner_, A, x, b );
            return;
        }

        const ScalarType b_norm = std::sqrt( linalg::dot( b, b ) );

        linalg::solvers::solve( *inner_, A, x, b );

        // tmp_ := A x   then   tmp_ := b - A x
        linalg::apply( A, x, tmp_ );
        linalg::lincomb( tmp_, { ScalarType( 1 ), ScalarType( -1 ) }, { b, tmp_ } );
        const ScalarType r_norm = std::sqrt( linalg::dot( tmp_, tmp_ ) );
        const ScalarType rel    = ( b_norm > ScalarType( 0 ) ) ? r_norm / b_norm : ScalarType( 0 );

        util::logroot << "[" << label_ << "] ||b - A x|| / ||b|| = " << rel
                      << "  (||b|| = " << b_norm << ", ||r|| = " << r_norm << ")"
                      << std::endl;
    }

  private:
    Inner*             inner_;
    SolutionVectorType tmp_;
    bool               enabled_;
    std::string        label_;
};

} // namespace terra::mantlecirculation
