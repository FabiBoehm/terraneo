#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/vector.hpp"
#include "terra/linalg/vector_q1.hpp"

namespace terra::linalg {

/// @brief Composed operator \f$ B \, \mathrm{diag}(d)^{-1} \, B^T \f$ mapping pressure to pressure.
///
/// Here \f$ B \f$ is the divergence operator, \f$ B^T \f$ the gradient operator,
/// and \f$ d \f$ is a velocity-space diagonal (e.g., lumped velocity mass, or operator diagonal).
/// The inverse diagonal must be precomputed and passed in.
///
/// @tparam GradientT  Gradient operator type (pressure -> velocity, i.e., \f$ B^T \f$).
/// @tparam DivergenceT Divergence operator type (velocity -> pressure, i.e., \f$ B \f$).
template < OperatorLike GradientT, OperatorLike DivergenceT >
class ComposedBDinvBT
{
  public:
    using VelocityVectorType = DstOf< GradientT >;
    using PressureVectorType = SrcOf< GradientT >;

    using SrcVectorType = PressureVectorType;
    using DstVectorType = PressureVectorType;
    using ScalarType    = ScalarOf< PressureVectorType >;

    static_assert( std::is_same_v< SrcOf< DivergenceT >, VelocityVectorType > );
    static_assert( std::is_same_v< DstOf< DivergenceT >, PressureVectorType > );

    /// @brief Construct the composed operator.
    /// @param gradient   Gradient operator \f$ B^T \f$ (pressure -> velocity).
    /// @param divergence Divergence operator \f$ B \f$ (velocity -> pressure).
    /// @param inv_diag   Precomputed inverse diagonal vector in velocity space.
    /// @param tmp_vel    Temporary velocity-space vector for intermediate results.
    ComposedBDinvBT(
        GradientT&          gradient,
        DivergenceT&        divergence,
        VelocityVectorType& inv_diag,
        VelocityVectorType& tmp_vel )
    : gradient_( gradient )
    , divergence_( divergence )
    , inv_diag_( inv_diag )
    , tmp_vel_( tmp_vel )
    {}

    /// @brief Apply \f$ y = B \, D^{-1} \, B^T \, x \f$.
    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        // tmp = B^T * src  (gradient: pressure -> velocity)
        apply( gradient_, src, tmp_vel_ );

        // tmp *= D^{-1}
        scale_in_place( tmp_vel_, inv_diag_ );

        // dst = B * tmp  (divergence: velocity -> pressure)
        apply( divergence_, tmp_vel_, dst );
    }

  private:
    GradientT&          gradient_;
    DivergenceT&        divergence_;
    VelocityVectorType& inv_diag_;
    VelocityVectorType& tmp_vel_;
};

/// @brief Composed operator \f$ B \, D^{-1} \, A \, D^{-1} \, B^T \f$ mapping pressure to pressure.
///
/// Here \f$ A \f$ is the viscous (velocity-velocity) operator,
/// \f$ B \f$ is the divergence, \f$ B^T \f$ is the gradient,
/// and \f$ D^{-1} \f$ is a precomputed inverse diagonal in velocity space.
///
/// @tparam GradientT   Gradient operator type (pressure -> velocity).
/// @tparam DivergenceT Divergence operator type (velocity -> pressure).
/// @tparam ViscousT    Viscous operator type (velocity -> velocity).
template < OperatorLike GradientT, OperatorLike DivergenceT, OperatorLike ViscousT >
class ComposedBDinvADinvBT
{
  public:
    using VelocityVectorType = DstOf< GradientT >;
    using PressureVectorType = SrcOf< GradientT >;

    using SrcVectorType = PressureVectorType;
    using DstVectorType = PressureVectorType;
    using ScalarType    = ScalarOf< PressureVectorType >;

    static_assert( std::is_same_v< SrcOf< DivergenceT >, VelocityVectorType > );
    static_assert( std::is_same_v< DstOf< DivergenceT >, PressureVectorType > );
    static_assert( std::is_same_v< SrcOf< ViscousT >, VelocityVectorType > );
    static_assert( std::is_same_v< DstOf< ViscousT >, VelocityVectorType > );

    /// @brief Construct the composed operator.
    /// @param gradient   Gradient operator \f$ B^T \f$.
    /// @param divergence Divergence operator \f$ B \f$.
    /// @param viscous    Viscous operator \f$ A \f$.
    /// @param inv_diag   Precomputed inverse diagonal in velocity space.
    /// @param tmp_vel_1  First temporary velocity vector.
    /// @param tmp_vel_2  Second temporary velocity vector.
    ComposedBDinvADinvBT(
        GradientT&          gradient,
        DivergenceT&        divergence,
        ViscousT&           viscous,
        VelocityVectorType& inv_diag,
        VelocityVectorType& tmp_vel_1,
        VelocityVectorType& tmp_vel_2 )
    : gradient_( gradient )
    , divergence_( divergence )
    , viscous_( viscous )
    , inv_diag_( inv_diag )
    , tmp_vel_1_( tmp_vel_1 )
    , tmp_vel_2_( tmp_vel_2 )
    {}

    /// @brief Apply \f$ y = B \, D^{-1} \, A \, D^{-1} \, B^T \, x \f$.
    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        // tmp1 = B^T * src  (gradient: pressure -> velocity)
        apply( gradient_, src, tmp_vel_1_ );

        // tmp1 *= D^{-1}
        scale_in_place( tmp_vel_1_, inv_diag_ );

        // tmp2 = A * tmp1  (viscous: velocity -> velocity)
        apply( viscous_, tmp_vel_1_, tmp_vel_2_ );

        // tmp2 *= D^{-1}
        scale_in_place( tmp_vel_2_, inv_diag_ );

        // dst = B * tmp2  (divergence: velocity -> pressure)
        apply( divergence_, tmp_vel_2_, dst );
    }

  private:
    GradientT&          gradient_;
    DivergenceT&        divergence_;
    ViscousT&           viscous_;
    VelocityVectorType& inv_diag_;
    VelocityVectorType& tmp_vel_1_;
    VelocityVectorType& tmp_vel_2_;
};

} // namespace terra::linalg
