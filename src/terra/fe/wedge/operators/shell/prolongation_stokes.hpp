
#pragma once

#include "prolongation_constant.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"

namespace terra::fe::wedge::operators::shell {

/// @brief Block prolongation for Stokes-type saddle-point systems (VectorQ1IsoQ2Q1).
///
/// Applies ProlongationVecConstant to the velocity block and ProlongationConstant to the pressure block.
/// Satisfies OperatorLike.
template < typename ScalarT, int VecDim = 3 >
class ProlongationStokes
{
  public:
    using SrcVectorType = linalg::VectorQ1IsoQ2Q1< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1IsoQ2Q1< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    ProlongationVecConstant< ScalarT, VecDim > P_vel_;
    ProlongationConstant< ScalarT >            P_pre_;

  public:
    explicit ProlongationStokes(
        linalg::OperatorApplyMode operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : P_vel_( operator_apply_mode )
    , P_pre_( operator_apply_mode )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        P_vel_.apply_impl( src.block_1(), dst.block_1() );
        P_pre_.apply_impl( src.block_2(), dst.block_2() );
    }
};

} // namespace terra::fe::wedge::operators::shell
