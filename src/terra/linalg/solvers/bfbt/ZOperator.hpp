#pragma once

#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"

namespace terra::linalg {

template < typename ScalarT >
class ZOperator
{
  public:
    using ViscT = terra::fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarT, 3 >;
    using GradT = terra::fe::wedge::operators::shell::Gradient< ScalarT >;
    using DivT  = terra::fe::wedge::operators::shell::Divergence< ScalarT >;

    using SrcVectorType          = SrcOf< GradT >;
    using DstVectorType          = DstOf< DivT >;
    using IntermediateVectorType = DstOf< ViscT >;
    using ScalarType             = ScalarOf< DstVectorType >;

  private:
    DivT                   div_;
    GradT                  grad_;
    IntermediateVectorType viscInvDiag_;
    IntermediateVectorType tmp_;

  public:
    explicit ZOperator(
        const DivT&                   div,
        const GradT&                  grad,
        const IntermediateVectorType& viscInvDiag,
        const IntermediateVectorType& tmp )
    : div_( div )
    , grad_( grad )
    , viscInvDiag_( viscInvDiag )
    , tmp_( tmp ) {};

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        grad_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );

        grad_.set_treat_boundary(true);
        apply( grad_, src, tmp_ );

        scale_in_place( tmp_, viscInvDiag_ );

        div_.set_treat_boundary(true);
        div_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );

        apply( div_, tmp_, dst );
    }
};

static_assert( linalg::OperatorLike< ZOperator< double > > );

} // namespace terra::linalg