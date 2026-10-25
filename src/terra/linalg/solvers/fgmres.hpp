#pragma once

#include "util/timer.hpp"

#include "util/floating_point.hpp"
#include "eigen/eigen_wrapper.hpp"
#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"

#include <chrono>
#include <functional>

namespace terra::linalg::solvers {

template < typename ScalarType >
struct FGMRESOptions
{
    /// @brief Number of inner iterations before restart (FGMRES(m)).
    int restart = 30;

    /// @brief Relative residual tolerance for convergence.
    ///
    /// Held in util::Accumulator< ScalarType > rather than ScalarType: a tolerance of
    /// 1e-6 is not representable as a normal fp16 number, and comparing residuals is
    /// host-side work that has no reason to run in the storage precision.
    util::Accumulator< ScalarType > relative_residual_tolerance = 1e-6;

    /// @brief Absolute residual tolerance for convergence.
    util::Accumulator< ScalarType > absolute_residual_tolerance = 1e-6;

    /// @brief Maximum number of total iterations (across all restarts).
    int max_iterations = 100;
};

/// @brief Flexible GMRES (FGMRES) iterative solver for nonsymmetric linear systems.
///
/// FGMRES allows for varying (flexible) preconditioning at each iteration,
/// making it suitable for use with inexact or variable preconditioners.
///
/// Reference:
/// @code
/// Saad, Y. (1993).
/// A flexible inner-outer preconditioned GMRES algorithm.
/// SIAM Journal on Scientific Computing, 14(2), 461-469.
/// @endcode
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Supports optional right preconditioning via the flexible framework.
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam PreconditionerT Preconditioner type (must satisfy SolverLike, defaults to IdentitySolver).
template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class FGMRES
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;

    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;

    /// @brief Right-hand side vector type.
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Scalar type of the Krylov vectors.
    using ScalarType = typename SolutionVectorType::ScalarType;

    /// @brief Precision of the small dense problem: the Hessenberg matrix, the Givens
    ///        rotations, the back-substitution and the residual bookkeeping.
    ///
    /// util::Accumulator< ScalarType >, so ScalarType for float and double and double
    /// for the 16-bit types. The least-squares problem is (restart+1)^2 scalars on the
    /// host, so carrying it in double costs nothing next to the Krylov vectors, and it
    /// keeps the orthogonalization out of a 9.8e-4 unit roundoff. Only the vectors are
    /// stored in ScalarType.
    using DenseScalar = util::Accumulator< ScalarType >;

    /// @brief Construct an FGMRES solver with a custom preconditioner.
    ///
    /// @param tmp Temporary vectors for workspace. Must contain at least 2*restart + 4 vectors:
    ///            - tmp[0]: residual vector r
    ///            - tmp[1 .. restart+1]: Arnoldi basis vectors V_0..V_restart
    ///            - tmp[restart+2 .. 2*restart+1]: preconditioned directions Z_0..Z_{restart-1}
    ///            - tmp[2*restart+2]: work vector w for A*z
    ///            - tmp[2*restart+3]: accumulator vector for solution update
    /// @param options FGMRES solver parameters (restart, tolerances, max iterations).
    /// @param statistics Shared pointer to statistics table for logging iteration progress (optional).
    /// @param preconditioner Preconditioner solver (defaults to identity).
    FGMRES(
        const std::vector< SolutionVectorType >& tmp,
        const FGMRESOptions< ScalarType >&       options        = {},
        const std::shared_ptr< util::Table >&    statistics     = nullptr,
        const PreconditionerT                    preconditioner = IdentitySolver< OperatorT >() )
    : tag_( "fgmres_solver" )
    , tmp_( tmp )
    , options_( options )
    , statistics_( statistics )
    , preconditioner_( preconditioner )
    , skip_preconditioner_in_case_of_nan_or_infs_( true )
    {}

    /// @brief Set a tag string for statistics output identification.
    /// @param tag Tag string to identify this solver instance in statistics.
    void set_tag( const std::string& tag ) { tag_ = tag; }

    /// @brief Set the number of inner iterations before restart (FGMRES(m)).
    /// @param m Restart parameter (must be at least 1).
    void set_restart( int m ) { options_.restart = std::max( 1, m ); }

    /// Called after every restart cycle, once the iterate has been updated and the
    /// true residual recomputed, with (total iterations, iterate, relative residual).
    /// Returning false stops the solve there: a caller that measures something the
    /// solver cannot see (the error against a known solution, say) decides when the
    /// iterate is good enough.
    using RestartCallback = std::function< bool( int, const SolutionVectorType&, double ) >;
    void set_restart_callback( RestartCallback callback ) { restart_callback_ = std::move( callback ); }

    /// @brief Solve the linear system \f$ Ax = b \f$ using flexible GMRES with restarts.
    ///
    /// Uses right preconditioning: solves \f$ A M^{-1} y = b \f$ where \f$ x = M^{-1} y \f$.
    /// The preconditioner M can vary at each iteration (flexibility).
    ///
    /// The method builds an Arnoldi basis via modified Gram-Schmidt orthogonalization,
    /// applies Givens rotations to minimize the least-squares residual in the Krylov subspace,
    /// and updates the solution. If convergence is not achieved within the restart window,
    /// the process restarts with the updated solution until max_iterations is reached or
    /// convergence criteria are satisfied.
    ///
    /// @param A Operator (matrix) to solve with.
    /// @param x Solution vector (input: initial guess, output: final solution).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        // Wall-clock time since entry, recorded with every history row (time to
        // solution per tolerance). Every row follows a host-side reduction, so the
        // device is in sync at that point.
        const auto t_solve0 = std::chrono::steady_clock::now();
        auto       elapsed_s = [&]() {
            return std::chrono::duration< double >( std::chrono::steady_clock::now() - t_solve0 ).count();
        };
        // Compute initial residual r = b - A*x
        auto& r = tmp_[0];
        apply( A, x, r );
        lincomb( r, { 1.0, -1.0 }, { b, r } );

        DenseScalar       beta0            = Kokkos::sqrt( dot( r, r ) );
        const DenseScalar initial_residual = beta0;

        if ( statistics_ )
        {
            statistics_->add_row(
                { { "tag", tag_ },
                  { "iteration", 0 },
                  { "elapsed_s", elapsed_s() },
                  { "relative_residual", 1.0 },
                  { "absolute_residual", static_cast< double >( beta0 ) } } );
        }

        if ( beta0 <= options_.absolute_residual_tolerance )
        {
            return;
        }

        // Validate workspace size.
        // Required: 1 (r) + (m+1) (V) + m (Z) + 1 (w) + 1 (acc) = 2m + 4
        const int nTmp   = static_cast< int >( tmp_.size() );
        const int m_req  = options_.restart;
        const int needed = 2 * m_req + 4;
        if ( nTmp < needed )
        {
            std::cerr << "FGMRES: insufficient tmp vectors. Provided " << nTmp
                      << ", required at least (2*restart + 4) = " << needed << " for restart m = " << m_req
                      << std::endl;
            Kokkos::abort( "FGMRES: insufficient tmp vectors" );
        }

        // Workspace layout offsets:
        const int offV   = 1;                    // V_0 .. V_m stored in tmp[1 .. m+1]
        const int offZ   = offV + ( m_req + 1 ); // Z_0 .. Z_{m-1} stored in tmp[offV + m+1 .. offV + 2m]
        const int idxW   = offZ + m_req;         // w = A*z_j
        const int idxAcc = idxW + 1;             // accumulator for solution update

        // Allocate small dense arrays for Hessenberg matrix and Givens rotations.
        Eigen::Matrix< DenseScalar, Eigen::Dynamic, Eigen::Dynamic > H( m_req + 1, m_req );
        Eigen::Matrix< DenseScalar, Eigen::Dynamic, 1 >              g( m_req + 1 );
        std::vector< DenseScalar >                                   cs( m_req + 1, 0 ), sn( m_req + 1, 0 );

        int total_iters = 0;

        // A breakdown or a stagnation leaves the Krylov basis unusable, so restarting
        // from the same residual reproduces it exactly: the inner `break` skips the
        // loop's ++total_iters, the outer condition is unchanged, and the solver spins
        // forever. Both safe breaks below therefore set this, and we leave after the
        // least-squares solve and solution update have been applied.
        bool hard_stop = false;

        // Outer restart loop
        while ( total_iters < options_.max_iterations )
        {
            const int iters_at_cycle_start = total_iters;

            // A 16-bit basis cannot normalise a residual whose inverse norm overflows
            // the storage type (fp16: beta0 < 1.5e-5 gives 1/beta0 = Inf, V_0 = NaN and
            // the preconditioner returns NaN on the next restart). Such a residual is at
            // the precision's floor: leave with the solution as it stands.
            if ( 1.0 / static_cast< double >( beta0 ) > util::max_of< ScalarType >() )
            {
                util::logroot << "FGMRES[" << tag_ << "]: residual norm " << double( beta0 )
                              << " cannot be normalised in the basis precision; stopping. (total_iters = "
                              << total_iters << ")" << std::endl;
                break;
            }

            // Initialize Arnoldi: V_0 = r / ||r||
            auto& V0 = tmp_[offV + 0];
            lincomb( V0, { util::narrow< ScalarType >( 1.0 / beta0 ) }, { r } );

            // Reset dense workspace for this restart cycle
            H.setZero();
            g.setZero();
            std::fill( cs.begin(), cs.end(), DenseScalar( 0 ) );
            std::fill( sn.begin(), sn.end(), DenseScalar( 0 ) );
            g( 0 ) = beta0;

            int inner_its = 0;

            // Inner Arnoldi iteration (build Krylov subspace up to dimension m_req)
            for ( int j = 0; j < m_req && total_iters < options_.max_iterations; ++j, ++total_iters )
            {
                // Apply preconditioner: z_j = M^{-1} v_j
                auto& vj = tmp_[offV + j];
                auto& zj = tmp_[offZ + j];
                assign( zj, 0 );
                {
                    util::Timer t( "fgmres_preconditioner" );
                    solve( preconditioner_, A, zj, vj );
                }

                const bool preconditioner_result_contains_nan_or_inf = has_nan_or_inf( zj );

                if ( skip_preconditioner_in_case_of_nan_or_infs_ && preconditioner_result_contains_nan_or_inf )
                {
                    util::logroot
                        << "FGMRES[" << tag_
                        << "]: The preconditioner appears to have produced a vector that has NaN or Inf entries.\n"
                           "        This may be a result of the preconditioner or of the input provided by FGMRES.\n"
                           "        To at least provide a fix for the first case, we keep the input to the preconditioner\n"
                           "        and write it to the output (equivalent of skipping the preconditioner in this iteration).\n"
                           "        (Details: total_iters = "
                        << total_iters << ", j = " << j << ", beta0 = " << double( beta0 )
                        << ", nan/inf in v_j = " << has_nan_or_inf( vj ) << ", in x = " << has_nan_or_inf( x )
                        << ", in r = " << has_nan_or_inf( r ) << ")" << std::endl;

                    assign( zj, vj );
                }

                // Apply operator: w = A * z_j
                auto& w = tmp_[idxW];
                {
                    util::Timer t( "fgmres_operator" );
                    apply( A, zj, w );
                }

                util::Timer t_orth( "fgmres_orthogonalization" );
                // Modified Gram-Schmidt orthogonalization against V_0..V_j
                for ( int i = 0; i <= j; ++i )
                {
                    auto&            vi  = tmp_[offV + i];
                    const DenseScalar hij = dot( w, vi );
                    H( i, j )             = hij;
                    lincomb( w, { util::narrow< ScalarType >( 1.0 ), util::narrow< ScalarType >( -hij ) }, { w, vi } );
                }

                // Compute h_{j+1,j} and normalize to get v_{j+1}
                const DenseScalar h_jp1j = Kokkos::sqrt( dot( w, w ) );
                t_orth.stop();

                // SAFE BREAK 1: Arnoldi Breakdown. The threshold is the unit roundoff of
                // DenseScalar, not of ScalarType: H, g and the Givens rotations are all
                // formed and stored in DenseScalar, so that is what limits how small a
                // residual the least-squares problem can resolve. Using ScalarType here
                // made the test useless for a 16-bit basis -- epsilon 2^-10 (fp16) or
                // 2^-7 (bf16) against an O(1e2) initial residual declares a breakdown for
                // any h_{j+1,j} below ~0.2, i.e. on the very first Arnoldi step.
                if ( h_jp1j < util::epsilon_of< DenseScalar >() * initial_residual ||
                     1.0 / static_cast< double >( h_jp1j ) > util::max_of< ScalarType >() )
                {
                    H( j + 1, j ) = 0;
                    inner_its     = j + 1;
                    hard_stop     = true;
                    ++total_iters; // `break` skips the loop increment

                    util::logroot << "FGMRES[" << tag_ << "]: Arnoldi breakdown. Terminating.\n"
                                     "        (Details: total_iters = "
                                  << total_iters << ", j = " << j << ", h_jp1j = " << double( h_jp1j )
                                  << ", threshold = " << util::epsilon_of< DenseScalar >() * double( initial_residual )
                                  << ")" << std::endl;

                    break; // Subspace is "full" or converged; exit to solve current least squares
                }

                H( j + 1, j ) = h_jp1j;

                if ( h_jp1j > DenseScalar( 0 ) )
                {
                    auto& vjp1 = tmp_[offV + j + 1];
                    lincomb( vjp1, { util::narrow< ScalarType >( 1.0 / h_jp1j ) }, { w } );
                }

                // Apply previous Givens rotations to column j of H
                for ( int i = 0; i < j; ++i )
                {
                    const DenseScalar temp = cs[i] * H( i, j ) + sn[i] * H( i + 1, j );
                    H( i + 1, j )         = -sn[i] * H( i, j ) + cs[i] * H( i + 1, j );
                    H( i, j )             = temp;
                }

                // Compute new Givens rotation to zero H(j+1, j)
                {
                    const DenseScalar a    = H( j, j );
                    const DenseScalar b    = H( j + 1, j );
                    const DenseScalar r_ab = Kokkos::hypot( a, b );
                    cs[j]                  = ( r_ab == DenseScalar( 0 ) ) ? DenseScalar( 1 ) : a / r_ab;
                    sn[j]                  = ( r_ab == DenseScalar( 0 ) ) ? DenseScalar( 0 ) : b / r_ab;
                    H( j, j )             = r_ab;
                    H( j + 1, j )         = 0;

                    // SAFE BREAK 2: Stagnation / Singularity. DenseScalar roundoff for the
                    // same reason as SAFE BREAK 1 -- H( j, j ) is a DenseScalar.
                    if ( Kokkos::abs( H( j, j ) ) < util::epsilon_of< DenseScalar >() )
                    {
                        inner_its = j; // Don't include this column in the back-solve
                        hard_stop = true;
                        ++total_iters; // `break` skips the loop increment

                        util::logroot << "FGMRES[" << tag_ << "]: Stagnation after Givens rotation. Terminating.\n"
                                         "        (Details: total_iters = "
                                      << total_iters << ", j = " << j << ", |H(j,j)| = "
                                      << double( Kokkos::abs( H( j, j ) ) ) << ")" << std::endl;

                        break;
                    }

                    // Apply new Givens rotation to g
                    const DenseScalar gj  = g( j );
                    const DenseScalar gj1 = g( j + 1 );
                    g( j )               = cs[j] * gj + sn[j] * gj1;
                    g( j + 1 )           = -sn[j] * gj + cs[j] * gj1;
                }

                // Estimate residual norm from least-squares problem
                const DenseScalar abs_res = Kokkos::abs( g( j + 1 ) );
                const DenseScalar rel_res = abs_res / initial_residual;

                if ( statistics_ )
                {
                    statistics_->add_row(
                        { { "tag", tag_ },
                          { "iteration", total_iters + 1 },
                          { "elapsed_s", elapsed_s() },
                          { "relative_residual", static_cast< double >( rel_res ) },
                          { "absolute_residual", static_cast< double >( abs_res ) } } );
                }

                inner_its = j + 1;

                // Check for convergence
                if ( rel_res <= options_.relative_residual_tolerance || abs_res < options_.absolute_residual_tolerance )
                {
                    ++total_iters; // `break` skips the loop increment
                    break;
                }
            }

            // Solve upper-triangular system R*y = g(0..inner_its-1) via back-substitution
            Eigen::Matrix< DenseScalar, Eigen::Dynamic, 1 > y( inner_its );
            y.setZero();
            for ( int row = inner_its - 1; row >= 0; --row )
            {
                DenseScalar sum = g( row );
                for ( int col = row + 1; col < inner_its; ++col )
                    sum -= H( row, col ) * y( col );
                y( row ) = sum / H( row, row );
            }

            // The least-squares coefficients are formed in DenseScalar; a 16-bit basis
            // can leave H ill-conditioned far above DenseScalar's roundoff, and a
            // coefficient beyond the storage type's range would narrow to Inf and put
            // NaN into x. Such a basis has nothing representable left to contribute.
            {
                double y_max = 0.0;
                for ( int i = 0; i < inner_its; ++i )
                    y_max = std::max( y_max, std::abs( static_cast< double >( y( i ) ) ) );
                if ( !( y_max <= util::max_of< ScalarType >() ) )
                {
                    util::logroot << "FGMRES[" << tag_ << "]: least-squares coefficient " << y_max
                                  << " not representable in the basis precision; stopping without this update.\n"
                                     "        (Details: total_iters = "
                                  << total_iters << ", inner_its = " << inner_its << ", beta0 = " << double( beta0 )
                                  << ")" << std::endl;
                    break;
                }
            }

            // Update solution: x += sum_{i=0}^{inner_its-1} y_i * z_i
            util::Timer t_upd( "fgmres_update_and_restart" );
            auto& acc = tmp_[idxAcc];
            assign( acc, 0 );
            for ( int i = 0; i < inner_its; ++i )
            {
                auto& zi = tmp_[offZ + i];
                lincomb( acc, { util::narrow< ScalarType >( 1.0 ), util::narrow< ScalarType >( y( i ) ) },
                         { acc, zi } );
            }
            lincomb( x, { 1.0, 1.0 }, { x, acc } );

            // Recompute residual r = b - A*x for next restart
            apply( A, x, r );
            lincomb( r, { 1.0, -1.0 }, { b, r } );
            beta0 = Kokkos::sqrt( dot( r, r ) );

            if ( restart_callback_ &&
                 !restart_callback_( total_iters, x, static_cast< double >( beta0 / initial_residual ) ) )
            {
                return;
            }

            // Check for final convergence
            if ( beta0 <= options_.absolute_residual_tolerance ||
                 beta0 / initial_residual <= options_.relative_residual_tolerance )
            {
                return;
            }

            // A safe break has already contributed everything this basis can give.
            if ( hard_stop )
            {
                return;
            }

            // Defence in depth. Nothing above should be able to leave the inner loop
            // without counting an iteration any more, but when that happened the outer
            // condition never moved and the solver re-ran an identical cycle forever --
            // silently, for hours, since the statistics table only prints at the end.
            // Make that state terminal and loud rather than invisible.
            if ( total_iters == iters_at_cycle_start )
            {
                util::logroot << "FGMRES[" << tag_
                              << "]: restart cycle advanced no iteration. Terminating to avoid an "
                                 "infinite loop.\n        (Details: total_iters = "
                              << total_iters << ", beta0 = " << double( beta0 )
                              << ", rel = " << double( beta0 / initial_residual ) << ")" << std::endl;
                return;
            }
        }
    }

  private:
    std::string tag_; ///< Tag for statistics output identification.

    std::vector< SolutionVectorType > tmp_; ///< Temporary workspace vectors.

    FGMRESOptions< ScalarType > options_; ///< Solver parameters (restart, tolerances, max_iterations).

    std::shared_ptr< util::Table > statistics_; ///< Statistics table for iteration logging (optional).
    RestartCallback                restart_callback_; ///< Per-restart observer, may stop the solve (optional).

    PreconditionerT preconditioner_; ///< Preconditioner solver.

    bool skip_preconditioner_in_case_of_nan_or_infs_;
};

/// @brief Static assertion: FGMRES satisfies SolverLike concept.
static_assert(
    SolverLike<
        FGMRES< linalg::detail::
                    DummyOperator< linalg::detail::DummyVector< double >, linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers