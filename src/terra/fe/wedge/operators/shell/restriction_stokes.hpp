
#pragma once

#include "restriction_constant.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"

namespace terra::fe::wedge::operators::shell {

/// @brief Block restriction for Stokes-type saddle-point systems (VectorQ1IsoQ2Q1).
///
/// Applies RestrictionVecConstant to the velocity block and RestrictionConstant to the pressure block.
/// Satisfies OperatorLike.
template < typename ScalarT, int VecDim = 3 >
class RestrictionStokes
{
  public:
    using SrcVectorType = linalg::VectorQ1IsoQ2Q1< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1IsoQ2Q1< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    RestrictionVecConstant< ScalarT, VecDim > R_vel_;
    RestrictionConstant< ScalarT >            R_pre_;

  public:
    RestrictionStokes(
        const grid::shell::DistributedDomain& domain_coarse_vel,
        const grid::shell::DistributedDomain& domain_coarse_pre,
        linalg::OperatorApplyMode             operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : R_vel_( domain_coarse_vel, operator_apply_mode )
    , R_pre_( domain_coarse_pre, operator_apply_mode )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        R_vel_.apply_impl( src.block_1(), dst.block_1() );
        R_pre_.apply_impl( src.block_2(), dst.block_2() );
    }
};

} // namespace terra::fe::wedge::operators::shell
