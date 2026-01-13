#pragma once

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/bfbt/ZOperator.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/linalg/solvers/solver.hpp"

namespace terra::linalg {

template < typename ScalarT, solvers::SolverLike ZSolverT = solvers::PCG< ZOperator< ScalarT > > >
class BFBTOperator
{
  public:
    using ViscT = terra::fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarT, 3 >;
    using GradT = terra::fe::wedge::operators::shell::Gradient< ScalarT >;
    using DivT  = terra::fe::wedge::operators::shell::Divergence< ScalarT >;

    using PressureVectorType = DstOf< DivT >;
    using VelocityVectorType = DstOf< ViscT >;
    using SrcVectorType      = PressureVectorType;
    using DstVectorType      = PressureVectorType;
    using ScalarType         = ScalarOf< VelocityVectorType >;

  private:
    DivT                 div_;
    GradT                grad_;
    ViscT                visc_;
    ZOperator< ScalarT > zop_;
    VelocityVectorType   viscInvDiag_;
    VelocityVectorType   vtmp1_;
    VelocityVectorType   vtmp2_;
    PressureVectorType   ptmp_;
    ZSolverT             zsolver_;

  public:
    explicit BFBTOperator(
        const DivT&                 div,
        const GradT&                grad,
        const ViscT&                visc,
        const ZOperator< ScalarT >& zop,
        const VelocityVectorType&   viscInvDiag,
        const VelocityVectorType&   vtmp1,
        const VelocityVectorType&   vtmp2,
        const PressureVectorType&   ptmp,
        const ZSolverT              zsolver )
    : div_( div )
    , grad_( grad )
    , visc_( visc )
    , zop_( zop )
    , viscInvDiag_( viscInvDiag )
    , vtmp1_( vtmp1 )
    , vtmp2_( vtmp2 )
    , ptmp_( ptmp )
    , zsolver_( zsolver ) {};

    void apply_impl( const PressureVectorType& src, PressureVectorType& dst )
    {
        randomize(ptmp_);
        solve( zsolver_, zop_, ptmp_, src );
      
        grad_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );

        grad_.set_treat_boundary(true);
        apply( grad_, ptmp_, vtmp1_ );

        scale_in_place( vtmp1_, viscInvDiag_ );

        visc_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );

        visc_.set_treat_boundary(true);
        apply( visc_, vtmp1_, vtmp2_ );

        scale_in_place( vtmp2_, viscInvDiag_ );

        div_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );

        div_.set_treat_boundary(true);
        apply( div_, vtmp2_, ptmp_ );

        //randomize(dst);
        assign(dst, 0);
        solve( zsolver_, zop_, dst, ptmp_ );
        
    }
};

static_assert( linalg::OperatorLike< BFBTOperator< double, solvers::PCG< ZOperator< double > > > > );
static_assert( linalg::OperatorLike< BFBTOperator< double, solvers::FGMRES< ZOperator< double > > > > );

} // namespace terra::linalg