#pragma once

#include <memory>

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"

namespace terra::linalg::solvers {

/// @brief Type-erased preconditioner with a fixed (outer) operator/vector type.
///
/// Satisfies SolverLike< OperatorT >, but forwards solve_impl to a polymorphic
/// inner implementation chosen at runtime. This lets a fixed-precision outer
/// solver (e.g. the double block-triangular Stokes preconditioner) drive an inner
/// preconditioner — typically a multigrid V-cycle — whose internal storage/working
/// precision is selected at construction (double / float / half). The inner impl is
/// responsible for any precision conversion at its own boundary.
template < OperatorLike OperatorT >
class VelocityPrecHandle
{
  public:
    using OperatorType  = OperatorT;
    using SrcVectorType = SrcOf< OperatorT >;
    using DstVectorType = DstOf< OperatorT >;

    /// @brief Polymorphic inner preconditioner with the outer (double) interface.
    struct Impl
    {
        virtual ~Impl()                                                                      = default;
        virtual void solve_impl( OperatorType& A, SrcVectorType& x, const DstVectorType& b ) = 0;
    };

    VelocityPrecHandle() = default;
    explicit VelocityPrecHandle( std::shared_ptr< Impl > impl )
    : impl_( std::move( impl ) )
    {}

    void solve_impl( OperatorType& A, SrcVectorType& x, const DstVectorType& b ) { impl_->solve_impl( A, x, b ); }

  private:
    std::shared_ptr< Impl > impl_;
};

/// @brief Inner impl that forwards to a same-precision inner solver (e.g. the double
/// multigrid). Behaviour-identical to using the inner solver directly.
template < OperatorLike OperatorT, typename InnerSolverT >
class ForwardingPrecImpl : public VelocityPrecHandle< OperatorT >::Impl
{
  public:
    using SrcVectorType = SrcOf< OperatorT >;
    using DstVectorType = DstOf< OperatorT >;

    explicit ForwardingPrecImpl( InnerSolverT& inner )
    : inner_( inner )
    {}

    void solve_impl( OperatorT& A, SrcVectorType& x, const DstVectorType& b ) override { solve( inner_, A, x, b ); }

  private:
    InnerSolverT& inner_;
};

/// @brief Inner impl that applies no preconditioner at all: x <- b.
///
/// The control arm. Everything the (1,1) multigrid buys, at any precision, is the
/// difference between some other impl and this one, so it is worth being able to
/// select it the same way rather than inferring it.
template < OperatorLike OperatorT >
class IdentityPrecImpl : public VelocityPrecHandle< OperatorT >::Impl
{
  public:
    using SrcVectorType = SrcOf< OperatorT >;
    using DstVectorType = DstOf< OperatorT >;

    void solve_impl( OperatorT& /*A*/, SrcVectorType& x, const DstVectorType& b ) override { assign( x, b ); }
};

} // namespace terra::linalg::solvers
