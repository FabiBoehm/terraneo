#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/util/timer.hpp"

namespace terra::linalg::solvers {

/// @brief Working precision of one multigrid level.
///
/// Half and BHalf are accepted by the enum but a level built with them only works
/// for operators whose geometry survives 16-bit coordinate differencing; the
/// EpsDivDiv operator does not (see mg_prec_supported_for_operator below).
enum class MGPrec
{
    Double,
    Float,
    Half,
    BHalf
};

inline const char* mg_prec_name( MGPrec p )
{
    switch ( p )
    {
    case MGPrec::Float: return "float";
    case MGPrec::Half:  return "fp16";
    case MGPrec::BHalf: return "bf16";
    default:            return "double";
    }
}

/// @brief Parse one precision token: d|double, f|float|single, h|fp16|half, b|bf16|bhalf.
inline std::optional< MGPrec > parse_mg_prec( std::string_view s )
{
    if ( s == "d" || s == "double" ) return MGPrec::Double;
    if ( s == "f" || s == "float" || s == "single" ) return MGPrec::Float;
    if ( s == "h" || s == "fp16" || s == "half" || s == "half_t" ) return MGPrec::Half;
    if ( s == "b" || s == "bf16" || s == "bfloat16" || s == "bhalf" ) return MGPrec::BHalf;
    return std::nullopt;
}

/// @brief Parse a per-level precision list, coarsest first: "d,d,f,f".
///
/// A single token is broadcast to every level ("f" == all levels float). A list
/// shorter than num_levels is padded with its last entry. Returns nullopt on a
/// bad token.
inline std::optional< std::vector< MGPrec > > parse_mg_precision_list( std::string_view spec, int num_levels )
{
    std::vector< MGPrec > out;
    size_t                pos = 0;
    while ( pos <= spec.size() )
    {
        const size_t comma = spec.find( ',', pos );
        const auto   tok   = spec.substr( pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos );
        const auto   p     = parse_mg_prec( tok );
        if ( !p ) return std::nullopt;
        out.push_back( *p );
        if ( comma == std::string_view::npos ) break;
        pos = comma + 1;
    }
    if ( out.empty() ) return std::nullopt;
    while ( static_cast< int >( out.size() ) < num_levels )
        out.push_back( out.back() );
    out.resize( static_cast< size_t >( num_levels ) );
    return out;
}

/// @brief True when Kokkos silently substitutes float for this 16-bit type on the
///        current backend, in which case a level nominally at 16 bit has the same
///        footprint as single and the measurement means something else.
inline bool mg_prec_is_emulated( MGPrec p )
{
#if defined( KOKKOS_HALF_T_IS_FLOAT ) && KOKKOS_HALF_T_IS_FLOAT
    if ( p == MGPrec::Half ) return true;
#endif
#if defined( KOKKOS_BHALF_T_IS_FLOAT ) && KOKKOS_BHALF_T_IS_FLOAT
    if ( p == MGPrec::BHalf ) return true;
#endif
    return false;
}

/// @brief Bytes per scalar actually stored for a level at this precision.
inline int mg_prec_bytes( MGPrec p )
{
    switch ( p )
    {
    case MGPrec::Double: return 8;
    case MGPrec::Float:  return 4;
    default:             return mg_prec_is_emulated( p ) ? 4 : 2;
    }
}

// -----------------------------------------------------------------------------
// Type-erased level interface
// -----------------------------------------------------------------------------

/// @brief One multigrid cycle at a single level, in that level's own vector type.
///
/// Type erasure is per level rather than for the hierarchy as a whole: the
/// interface is templated on the level's vector type, and a level holds a pointer
/// to its coarser neighbour whose interface is templated on the *coarser* vector
/// type. So each level's precision is independent, chosen at construction, and the
/// number of distinct instantiations is bounded by the number of (fine, coarse)
/// precision pairs actually used -- not by the number of level/precision products.
template < class VectorT >
class MGCycle
{
  public:
    using VectorType = VectorT;

    virtual ~MGCycle() = default;

    /// @brief One cycle on this level: x <- x + C (b - A x).
    virtual void cycle( VectorType& x, const VectorType& b ) = 0;

    /// @brief Working precision of this level.
    virtual MGPrec precision() const = 0;
};

/// @brief Coarsest level: hand the problem to a coarse solver in this level's precision.
template < OperatorLike OperatorT, SolverLike CoarseSolverT >
class MGCoarseLevel final : public MGCycle< SrcOf< OperatorT > >
{
  public:
    using VectorType = SrcOf< OperatorT >;

    MGCoarseLevel( OperatorT A, CoarseSolverT solver, MGPrec prec )
    : A_( std::move( A ) )
    , solver_( std::move( solver ) )
    , prec_( prec )
    {}

    void cycle( VectorType& x, const VectorType& b ) override
    {
        util::Timer t( "mgmp_coarse_solve" );
        solve( solver_, A_, x, b );
    }

    MGPrec precision() const override { return prec_; }

  private:
    OperatorT     A_;
    CoarseSolverT solver_;
    MGPrec        prec_;
};

/// @brief One multigrid level at its own precision, above a coarser level that may
///        run at a different one.
///
/// Every component of the level is independently typed: the operator, the pre- and
/// post-smoothers, the restriction and the prolongation are separate template
/// parameters, so each can be instantiated at a different precision as long as the
/// vector types meet at the interfaces.
///
/// Precision changes are applied on the COARSE side of each transfer: the
/// restriction writes into a coarse vector in *this* level's precision, which is
/// then converted down for the coarser level, and the returned correction is
/// converted back up before prolongation. A coarse level holds roughly an eighth of
/// the unknowns of the level above it, so a conversion pass there costs about an
/// eighth of what converting the fine vectors would, and the fine-level traffic --
/// which is what the reduced precision is meant to cut -- is untouched.
///
/// When the two levels share a precision the conversion disappears entirely
/// (the branch is `if constexpr`) and no coarse-side buffers are allocated.
template <
    OperatorLike OperatorT,
    SolverLike   SmootherT,
    OperatorLike RestrictionT,
    OperatorLike ProlongationT,
    class        CoarseVectorT >
class MGLevel final : public MGCycle< SrcOf< OperatorT > >
{
  public:
    using VectorType = SrcOf< OperatorT >;
    using ScalarType = typename VectorType::ScalarType;

    /// Coarse vector in THIS level's precision: what the restriction writes.
    using CoarseLocalType = DstOf< RestrictionT >;
    using CoarseScalar    = typename CoarseVectorT::ScalarType;

    /// No conversion needed when the coarser level already speaks our coarse type.
    static constexpr bool kSamePrec = std::is_same_v< CoarseLocalType, CoarseVectorT >;

    static_assert( std::is_same_v< SrcOf< ProlongationT >, CoarseLocalType >,
        "prolongation source must be the coarse vector in this level's precision" );
    static_assert( std::is_same_v< DstOf< ProlongationT >, VectorType >,
        "prolongation destination must be this level's vector type" );
    static_assert( std::is_same_v< SrcOf< RestrictionT >, VectorType >,
        "restriction source must be this level's vector type" );

    /// @param conv_buffers Two coarse vectors in the COARSER level's precision
    ///        (restricted residual, returned correction). Pass an empty vector when
    ///        this level and the coarser one share a precision.
    MGLevel(
        OperatorT                                      A,
        SmootherT                                      smoother_pre,
        SmootherT                                      smoother_post,
        RestrictionT                                   R,
        ProlongationT                                  P,
        VectorType                                     r_fine,
        CoarseLocalType                                r_coarse_local,
        CoarseLocalType                                e_coarse_local,
        std::shared_ptr< MGCycle< CoarseVectorT > >    coarser,
        std::vector< CoarseVectorT >                   conv_buffers,
        MGPrec                                         prec,
        int                                            level )
    : A_( std::move( A ) )
    , smoother_pre_( std::move( smoother_pre ) )
    , smoother_post_( std::move( smoother_post ) )
    , R_( std::move( R ) )
    , P_( std::move( P ) )
    , r_fine_( std::move( r_fine ) )
    , r_coarse_local_( std::move( r_coarse_local ) )
    , e_coarse_local_( std::move( e_coarse_local ) )
    , coarser_( std::move( coarser ) )
    , conv_( std::move( conv_buffers ) )
    , prec_( prec )
    , suffix_( "_L" + std::to_string( level ) )
    {
        if constexpr ( !kSamePrec )
        {
            if ( conv_.size() < 2 )
                Kokkos::abort( "MGLevel: a precision change across this level needs 2 conversion buffers" );
        }
    }

    void cycle( VectorType& x, const VectorType& b ) override
    {
        {
            util::Timer t( "mgmp_smooth_pre" + suffix_ );
            solve( smoother_pre_, A_, x, b );
        }

        {
            util::Timer t( "mgmp_residual_restrict" + suffix_ );
            apply( A_, x, r_fine_ );
            lincomb( r_fine_, { ScalarType( 1 ), ScalarType( -1 ) }, { b, r_fine_ }, ScalarType( 0 ) );
            apply( R_, r_fine_, r_coarse_local_ );
        }

        if constexpr ( kSamePrec )
        {
            assign( e_coarse_local_, ScalarType( 0 ) );
            coarser_->cycle( e_coarse_local_, r_coarse_local_ );
        }
        else
        {
            util::Timer t( "mgmp_convert" + suffix_ );
            convert( r_coarse_local_, conv_[0] );
            assign( conv_[1], CoarseScalar( 0 ) );
            coarser_->cycle( conv_[1], conv_[0] );
            convert( conv_[1], e_coarse_local_ );
        }

        {
            // Additive prolongation: x <- x + P e.
            util::Timer t( "mgmp_prolong" + suffix_ );
            apply( P_, e_coarse_local_, x );
        }

        {
            util::Timer t( "mgmp_smooth_post" + suffix_ );
            solve( smoother_post_, A_, x, b );
        }
    }

    MGPrec precision() const override { return prec_; }

  private:
    OperatorT       A_;
    SmootherT       smoother_pre_;
    SmootherT       smoother_post_;
    RestrictionT    R_;
    ProlongationT   P_;
    VectorType      r_fine_;
    CoarseLocalType r_coarse_local_;
    CoarseLocalType e_coarse_local_;

    std::shared_ptr< MGCycle< CoarseVectorT > > coarser_;
    std::vector< CoarseVectorT >                conv_;

    MGPrec      prec_;
    std::string suffix_;
};

/// @brief Driver: repeats cycles on the finest level and monitors the residual.
///
/// Satisfies SolverLike, so it can be used directly as a preconditioner (for
/// instance behind VelocityPrecHandle) in place of Multigrid.
template < OperatorLike OperatorT >
class MultigridMP
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorT >;
    using RHSVectorType      = DstOf< OperatorT >;
    using ScalarType         = typename SolutionVectorType::ScalarType;

    /// @param relative_residual_threshold Stop early below this; <= 0 disables the
    ///        residual check, which also skips its extra operator apply per cycle.
    MultigridMP(
        std::shared_ptr< MGCycle< SolutionVectorType > > finest,
        SolutionVectorType                              tmp,
        int                                             num_cycles,
        ScalarType                                      relative_residual_threshold )
    : finest_( std::move( finest ) )
    , tmp_( std::move( tmp ) )
    , num_cycles_( num_cycles )
    , rel_tol_( relative_residual_threshold )
    {}

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        util::Timer t( "mgmp_solve" );

        ScalarType initial = ScalarType( 0 );
        if ( rel_tol_ > ScalarType( 0 ) )
        {
            initial = residual_norm( A, x, b );
            if ( !( initial > ScalarType( 0 ) ) ) return;
        }

        for ( int c = 0; c < num_cycles_; ++c )
        {
            finest_->cycle( x, b );

            if ( rel_tol_ > ScalarType( 0 ) )
            {
                const ScalarType r = residual_norm( A, x, b );
                if ( r / initial <= rel_tol_ ) break;
            }
        }
    }

    /// @brief Precision of the finest level.
    MGPrec precision() const { return finest_->precision(); }

  private:
    ScalarType residual_norm( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        apply( A, x, tmp_ );
        lincomb( tmp_, { ScalarType( 1 ), ScalarType( -1 ) }, { b, tmp_ }, ScalarType( 0 ) );
        return norm_2( tmp_ );
    }

    std::shared_ptr< MGCycle< SolutionVectorType > > finest_;
    SolutionVectorType                              tmp_;
    int                                             num_cycles_;
    ScalarType                                      rel_tol_;
};

} // namespace terra::linalg::solvers
