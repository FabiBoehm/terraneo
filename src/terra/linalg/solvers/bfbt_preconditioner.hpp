#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

/// @brief BFBT (BFBt) preconditioner for the Schur complement of the Stokes system.
///
/// Approximates the inverse of the Schur complement \f$ S = B A^{-1} B^T \f$ via:
/// \f[
///     S_{\text{BFBT}}^{-1} r = (B D^{-1} B^T)^{-1} (B D^{-1} A D^{-1} B^T) (B D^{-1} B^T)^{-1} r
/// \f]
/// where \f$ D \f$ is a diagonal approximation (e.g., lumped velocity mass).
///
/// The application of \f$ S_{\text{BFBT}}^{-1} \f$ to a vector \f$ r \f$ requires:
/// 1. Solve \f$ (B D^{-1} B^T) t_1 = r \f$
/// 2. Compute \f$ t_2 = (B D^{-1} A D^{-1} B^T) t_1 \f$
/// 3. Solve \f$ (B D^{-1} B^T) x = t_2 \f$
///
/// @tparam SchurOperatorT     The "nominal" Schur complement operator type (pressure -> pressure).
/// @tparam BDinvBTT           Type of the composed \f$ B D^{-1} B^T \f$ operator.
/// @tparam BDinvADinvBTT      Type of the composed \f$ B D^{-1} A D^{-1} B^T \f$ operator.
/// @tparam InnerSolverT       Solver for \f$ (B D^{-1} B^T) \f$ systems (e.g., PCG).
template <
    OperatorLike SchurOperatorT,
    OperatorLike BDinvBTT,
    OperatorLike BDinvADinvBTT,
    SolverLike   InnerSolverT >
class BFBTPreconditioner
{
  public:
    using OperatorType      = SchurOperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType     = DstOf< OperatorType >;

    static_assert( std::is_same_v< SrcOf< BDinvBTT >, SolutionVectorType > );
    static_assert( std::is_same_v< DstOf< BDinvBTT >, SolutionVectorType > );
    static_assert( std::is_same_v< SrcOf< BDinvADinvBTT >, SolutionVectorType > );
    static_assert( std::is_same_v< DstOf< BDinvADinvBTT >, SolutionVectorType > );

    /// @brief Construct a BFBT preconditioner.
    /// @param bdinvbt       The composed \f$ B D^{-1} B^T \f$ operator.
    /// @param bdinvadinvbt  The composed \f$ B D^{-1} A D^{-1} B^T \f$ operator.
    /// @param inner_solver  Solver for \f$ (B D^{-1} B^T) \f$ systems.
    /// @param t1            Temporary pressure vector.
    /// @param t2            Temporary pressure vector.
    /// @param omega         Relaxation parameter (default 1.0).
    BFBTPreconditioner(
        BDinvBTT&          bdinvbt,
        BDinvADinvBTT&     bdinvadinvbt,
        InnerSolverT&      inner_solver,
        SolutionVectorType& t1,
        SolutionVectorType& t2,
        double              omega = 1.0 )
    : bdinvbt_( bdinvbt )
    , bdinvadinvbt_( bdinvadinvbt )
    , inner_solver_( inner_solver )
    , t1_( t1 )
    , t2_( t2 )
    , omega_( omega )
    {}

    /// @brief Apply the BFBT preconditioner: \f$ x = \omega \, S_{\text{BFBT}}^{-1} b \f$.
    /// @param A  Ignored (the BFBT preconditioner uses its own internal operators).
    /// @param x  Solution vector (output).
    /// @param b  Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        // Step 1: solve (B D^{-1} B^T) t1 = b
        assign( t1_, 0.0 );
        solve( inner_solver_, bdinvbt_, t1_, b );

        // Step 2: t2 = (B D^{-1} A D^{-1} B^T) t1
        apply( bdinvadinvbt_, t1_, t2_ );

        // Step 3: solve (B D^{-1} B^T) x = t2
        assign( x, 0.0 );
        solve( inner_solver_, bdinvbt_, x, t2_ );

        // Step 4: relaxation
        if ( omega_ != 1.0 )
        {
            lincomb( x, { omega_ }, { x } );
        }
    }

  private:
    BDinvBTT&           bdinvbt_;
    BDinvADinvBTT&      bdinvadinvbt_;
    InnerSolverT&       inner_solver_;
    SolutionVectorType& t1_;
    SolutionVectorType& t2_;
    double              omega_;
};

} // namespace terra::linalg::solvers
