// Mixed-precision study of the Stokes solver.
//
// Three precisions are selectable at runtime and all default to double:
//   --ablock-prec double|float|bf16|fp16   the working precision of the whole
//       velocity V-cycle -- level vectors, coarse operators, Chebyshev smoothers,
//       transfers, inverse diagonals and the coarse FGMRES.
//   --ablock-geom double|low   the node coordinates and the Jacobian formed from
//       them (CoordScalarT). "low" is float, never 16 bits: the Jacobian is formed
//       by differencing the coordinates and that loses the geometry at 16 bits.
//   --krylov-precision double|single|bf16|fp16   the storage precision of the outer
//       FGMRES Krylov basis (via FGMRESLowMem).
// Everything else -- the outer FGMRES, the Stokes operator, the pressure mass matrix
// and the Schur block -- is double throughout, so each run measures what one
// reduced-precision component costs in outer iterations.
//
// --hourglass C adds the rank-6 gamma correction to the EpsDivDiv operator on every
// level. One-point quadrature leaves two zero-energy modes per wedge, so without it
// the velocity block has a near-null space the V-cycle cannot resolve and the outer
// residual stops descending well above the requested tolerance. --outer-tol sets how
// far the outer FGMRES is asked to go.
//
// It doubles as the original A-block convergence study under increasing viscosity
// contrast, with and without Galerkin coarse approximation (--gca).
//
// Unlike test_epsilon_divdiv_stokes_viscosity_contrast.cpp -- which wraps the
// viscous-block MG as the (1,1) preconditioner inside the full Stokes saddle-
// point FGMRES -- this test isolates the A-block: it builds *only* the viscous
// block A = K.block_11() (free-slip), an MG hierarchy on it (Cheby-Jacobi
// smoothers + coarse FGMRES), and runs the multigrid V-cycles *as the solver*
// for  A u = f  (f = strong-enforced radial buoyancy force).  The recorded
// per-cycle relative residual is the diagnostic.
//
// The point is to measure how A-block MG convergence degrades as the
// Frank-Kamenetskii viscosity contrast rmu grows, and whether GCA-assembled
// coarse operators (P^T A P, built top-down via TwoGridGCA from the Neumann
// fine block) recover convergence where re-discretized coarse operators (DCA)
// stall/diverge.
//
// Geometry, viscosity law and buoyancy mirror the A7 benchmark's initial state
// (Ra=7e3, free-slip both shells, Y_3^2 perturbation), same as the contrast
// test -- only the contrast rmu is swept here instead of fixed at 1e5.

#include <cstring>

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/strong_algebraic_freeslip_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp"

#include "terra/geophysics/viscosity/viscosity_interpolation.hpp"
#include "terra/shell/radial_profiles.hpp"

#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/fgmres_lowmem.hpp"
#include "util/timer.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/gca/gca_elements_collector.hpp"

#include "gca_fixed.hpp"  // copy of gca.hpp with a searchable crossover-DoF remap (gcafix::TwoGridGCA)
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/velocity_prec_handle.hpp"
#include "terra/linalg/solvers/chebyshev.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"

// Full-Stokes outer solve: w-BFBT Schur preconditioner (from the mc app).

#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"

#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::local_domain_md_range_policy_nodes;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::solvers::TwoGridGCA;
using terra::grid::shell::BoundaryConditions;

// Krylov-basis storage precision for the full-Stokes outer FGMRES. DOUBLE = plain
// FGMRES (basis in solution precision); FLOAT/BF16/FP16 = FGMRESLowMem with the
// basis stored in single / bfloat16 / binary16 (mirrors the mc app's reduced-
// precision Krylov-basis path in stokes_solver.hpp).
//
// BF16 and FP16 are both 2 B/dof; they differ in where they fail.
//   bhalf_t (8 exp / 7 mant): range of float, unit roundoff 3.9e-3.
//   half_t  (5 exp / 10 mant): unit roundoff 4.9e-4 -- 8x finer -- but min normal
//           6.1e-5, and a unit-2-norm basis vector over N dofs has entries of
//           order 1/sqrt(N) (~1.8e-4 at 3e7 dofs), so its smaller entries land in
//           half_t's denormals. Which of the two wins is the measurement.
// The basis is store-only (converted to and from the double scratch vectors and
// never operated on), so neither type instantiates 16-bit arithmetic and neither
// needs the half reduction identities that only exist in Kokkos >= 5.1.0.
enum class KrylovPrec { DOUBLE, FLOAT, BF16, FP16 };

inline const char* krylov_prec_name( KrylovPrec p )
{
    switch ( p )
    {
    case KrylovPrec::FP16:  return "fp16";
    case KrylovPrec::BF16:  return "bf16";
    case KrylovPrec::FLOAT: return "single";
    default:                return "double";
    }
}

// Kokkos maps half_t/bhalf_t to the backend's native type where one exists and
// silently falls back to float otherwise (e.g. bhalf_t on HIP before Kokkos
// 5.1.0). In the fallback case a nominally 16-bit basis has the same footprint as
// single and the run measures something else entirely, so state it in the header.
inline const char* krylov_prec_storage_note( KrylovPrec p )
{
    if ( p == KrylovPrec::FP16 )
#if defined( KOKKOS_HALF_T_IS_FLOAT ) && KOKKOS_HALF_T_IS_FLOAT
        return " [EMULATED: half_t falls back to float on this backend, 4 B/dof]";
#else
        return " [native half_t, 2 B/dof]";
#endif
    if ( p == KrylovPrec::BF16 )
#if defined( KOKKOS_BHALF_T_IS_FLOAT ) && KOKKOS_BHALF_T_IS_FLOAT
        return " [EMULATED: bhalf_t falls back to float on this backend, 4 B/dof]";
#else
        return " [native bhalf_t, 2 B/dof]";
#endif
    return "";
}

// A7 benchmark geometry + buoyancy parameters.  rmu (viscosity contrast) is now
// a runtime parameter -- it is what this test sweeps.
constexpr double kRm             = 1.22;
constexpr double kRp             = 2.22;
constexpr double kRayleigh       = 7.0e3;
constexpr double kPerturbDefault = 0.01;

constexpr auto BOUNDARY = static_cast< grid::shell::ShellBoundaryFlag >(
    static_cast< uint8_t >( CMB ) | static_cast< uint8_t >( SURFACE ) );

// Real spherical harmonic Y_3^2(theta, phi) = sqrt(105/(32pi)) sin^2(theta) cos(theta) cos(2 phi).
KOKKOS_INLINE_FUNCTION
double Y32( const double theta, const double phi )
{
    constexpr double c  = 1.0219854764332823;
    const double     st = Kokkos::sin( theta );
    const double     ct = Kokkos::cos( theta );
    return c * st * st * ct * Kokkos::cos( 2.0 * phi );
}

// A7 initial T(r, theta, phi) clamped to [0, 1].
KOKKOS_INLINE_FUNCTION
double a7_initial_temperature( const double cx, const double cy, const double cz, double perturb_amp )
{
    const double rad = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    double       T_lateral = 0.0;
    if ( perturb_amp != 0.0 && rad > 0.0 )
    {
        const double theta = Kokkos::acos( cz / rad );
        const double phi   = Kokkos::atan2( cy, cx );
        T_lateral          = perturb_amp * Y32( theta, phi );
    }
    return Kokkos::clamp( ( kRp - rad ) / ( kRp - kRm ) + T_lateral, 0.0, 1.0 );
}

// eta(T) = rmu^(0.5 - T), same as apps/mantlecirculation/src/interpolators.hpp.
struct FrankKamenetskiiViscosityInterpolator
{
    Grid3DDataVec< double, 3 >       grid_;
    Grid2DDataScalar< double >       radii_;
    grid::Grid4DDataScalar< double > data_;
    double                           rmu_;
    double                           perturb_amp_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double T = a7_initial_temperature( c( 0 ), c( 1 ), c( 2 ), perturb_amp_ );
        data_( sd, x, y, r ) = Kokkos::pow( rmu_, 0.5 - T );
    }
};

// Radial low-viscosity BAND: eta = eta_band (low) inside |rad - r_center| < half_width,
// eta = 1 elsewhere — a weak channel of fixed physical width.  The band is wide
// enough to be resolved on the fine levels but sub-cell on the coarse levels, so DCA
// mis-resolves it while GCA's P^T A P inherits the fine-resolved band.
struct RadialBandViscosityInterpolator
{
    Grid3DDataVec< double, 3 >       grid_;
    Grid2DDataScalar< double >       radii_;
    grid::Grid4DDataScalar< double > data_;
    double                           eta_band_;
    double                           r_center_;
    double                           half_width_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c   = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double                  rad = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
        data_( sd, x, y, r )              = ( Kokkos::fabs( rad - r_center_ ) < half_width_ ) ? eta_band_ : 1.0;
    }
};

// Radial buoyancy body force f(x) = -Ra * T(x) * r_hat.
struct RadialBuoyancyRHSInterpolator
{
    Grid3DDataVec< double, 3 >       grid_;
    Grid2DDataScalar< double >       radii_;
    grid::Grid4DDataVec< double, 3 > data_;
    double                           rayleigh_;
    double                           perturb_amp_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double rad = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
        if ( rad <= 0.0 )
        {
            for ( int d = 0; d < 3; ++d )
                data_( sd, x, y, r, d ) = 0.0;
            return;
        }
        const double T        = a7_initial_temperature( c( 0 ), c( 1 ), c( 2 ), perturb_amp_ );
        const double f_radial = -rayleigh_ * T;
        for ( int d = 0; d < 3; ++d )
            data_( sd, x, y, r, d ) = f_radial * c( d ) / rad;
    }
};

// Inverse-diagonal-as-preconditioner for the coarse FGMRES (same pattern as the
// mc app's coarse solver in stokes_solver.hpp).
template < linalg::OperatorLike OperatorT >
struct InverseDiagonalPreconditioner
{
    using OperatorType       = OperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;

    SolutionVectorType inv_diag_;

    explicit InverseDiagonalPreconditioner( const SolutionVectorType& d )
    : inv_diag_( d )
    {}

    void solve_impl( OperatorType& /*A*/, SolutionVectorType& x, const RHSVectorType& b )
    {
        linalg::assign( x, b );
        linalg::scale_in_place( x, inv_diag_ );
    }
};


// =============================================================================
// Reduced-precision A-block V-cycle (the velocity preconditioner).
// =============================================================================

/// Working precision of the velocity-block V-cycle. The outer FGMRES and the Stokes
/// operator always stay in double; only this hierarchy changes.
enum class ABlockPrec { DOUBLE, FLOAT, FP16, BF16, NONE };

/// Arithmetic precision inside the EpsDivDiv fast Dirichlet/Neumann kernel.
///
/// LOW is the default and the sensible end state: one precision knob per component,
/// so --ablock-prec fp16 means an fp16 V-cycle rather than fp16 storage feeding
/// double arithmetic. Only the geometry (coordinates, radii, Jacobian and the
/// shape-function gradients derived from it) stays in double.
///
/// DOUBLE exists because the two are worth separating when a low-precision arm
/// misbehaves: it reproduces what the kernel did unconditionally before, 2-byte
/// storage with 8-byte arithmetic, which is a pure memory-traffic experiment. Running
/// both tells you whether a degradation came from the bus or from the contraction.
enum class ABlockArith { DOUBLE, LOW };

/// Geometry precision of the A-block V-cycle: the stored coordinates and radii and,
/// through GeoT, the Jacobian arithmetic. LOW is float (16-bit geometry is never
/// offered: the coordinate differencing would be left with three decimal digits).
enum class ABlockGeom { DOUBLE, LOW };

/// Outer iteration. FGMRES is the Krylov method; IR is pure iterative refinement,
/// x <- x + M^-1 (b - A x), with the residual and the update in double and M^-1 one
/// application of the block preconditioner.
///
/// IR is the sharper probe of a reduced-precision V-cycle: its convergence rate is
/// exactly the preconditioner's contraction factor, with nothing in between. FGMRES
/// builds a Krylov space that can compensate for a mediocre preconditioner, which is
/// what makes a low-precision arm there look better than the V-cycle really is.
enum class OuterSolver { FGMRES, IR };

inline const char* outer_solver_name( OuterSolver o ) { return o == OuterSolver::IR ? "ir" : "fgmres"; }

inline const char* ablock_arith_name( ABlockArith a )
{
    return a == ABlockArith::LOW ? "low" : "double";
}

inline const char* ablock_geom_name( ABlockGeom g )
{
    return g == ABlockGeom::LOW ? "low" : "double";
}

inline const char* ablock_prec_name( ABlockPrec p )
{
    switch ( p )
    {
    case ABlockPrec::NONE:  return "none";
    case ABlockPrec::FLOAT: return "float";
    case ABlockPrec::FP16:  return "fp16";
    case ABlockPrec::BF16:  return "bf16";
    default:                return "double";
    }
}

/// Kokkos substitutes float for a 16-bit type when the backend has no native one, so
/// report which case this build got rather than let a 4-byte run read as 2-byte.
inline const char* ablock_prec_note( ABlockPrec p )
{
#if defined( KOKKOS_HALF_T_IS_FLOAT ) && KOKKOS_HALF_T_IS_FLOAT
    if ( p == ABlockPrec::FP16 ) return " [EMULATED: half_t is float on this backend]";
#endif
#if defined( KOKKOS_BHALF_T_IS_FLOAT ) && KOKKOS_BHALF_T_IS_FLOAT
    if ( p == ABlockPrec::BF16 ) return " [EMULATED: bhalf_t is float on this backend]";
#endif
    return "";
}

/// @brief The entire velocity-block V-cycle built in precision M, driven by the
///        double-precision outer solver through VelocityPrecHandle.
///
/// Geometry is GeomM (CoordScalarT): the coordinates and radii are regenerated per
/// level in that precision, which is also what the Jacobian is formed in. M governs
/// the stored fields: the level vectors, the viscosity coefficient, the inverse
/// diagonals and the coarse solver's Krylov workspace. convert() runs at the
/// double<->M boundary on entry and exit, which is two passes over one fine-level
/// vector per application -- the V-cycle itself never leaves M.
template < typename M, typename ArithM = double, typename GeomM = double >
class LowPrecABlock final
: public linalg::solvers::VelocityPrecHandle<
      fe::wedge::operators::shell::EpsilonDivDivKerngen< double > >::Impl
{
  public:
    using OuterOp  = fe::wedge::operators::shell::EpsilonDivDivKerngen< double >;
    using OuterVec = linalg::VectorQ1Vec< double, 3 >;

    // < field precision, VecDim, coordinate precision, fast-path arithmetic precision >
    using ViscM  = fe::wedge::operators::shell::EpsilonDivDivKerngen< M, 3, GeomM, ArithM >;
    using ProlM  = fe::wedge::operators::shell::ProlongationVecConstant< M >;
    using RestM  = fe::wedge::operators::shell::RestrictionVecConstant< M >;
    using SmooM  = linalg::solvers::Chebyshev< ViscM >;
    using CPrecM = InverseDiagonalPreconditioner< ViscM >;
    using CSolvM = linalg::solvers::FGMRES< ViscM, CPrecM >;
    using MGM    = linalg::solvers::Multigrid< ViscM, ProlM, RestM, SmooM, CSolvM >;
    using VecM   = linalg::VectorQ1Vec< M, 3 >;
    using ScalM  = linalg::VectorQ1Scalar< M >;

    LowPrecABlock(
        const std::vector< DistributedDomain >&                                  domains,
        const std::vector< Grid3DDataVec< double, 3 > >&                         coords_shell,
        const std::vector< Grid2DDataScalar< double > >&                         coords_radii,
        const std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >&        mask_data,
        const std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > >& boundary_mask_data,
        const std::vector< VectorQ1Scalar< double > >&                           eta,
        const std::vector< VectorQ1Vec< double, 3 > >&                           inverse_diagonals,
        BoundaryConditions&                                                      bcs,
        int                                                                      cheby_order,
        int                                                                      chebyshev_prepost,
        double                                                                   coarse_tol,
        double                                                                   hourglass_c )
    {
        const size_t num_levels = domains.size();
        const size_t fine       = num_levels - 1;

        // Viscosity and inverse diagonals converted once, down to M.
        for ( size_t l = 0; l < num_levels; ++l )
        {
            eta_m_.emplace_back( "eta_m_" + std::to_string( l ), domains[l], mask_data[l] );
            linalg::convert( eta[l], eta_m_.back() );
            inv_diag_m_.emplace_back( "inv_diag_m_" + std::to_string( l ), domains[l], mask_data[l] );
            linalg::convert( inverse_diagonals[l], inv_diag_m_.back() );
        }

        // The coordinates are a function of the domain alone, so the GeomM copies are
        // regenerated here rather than converted from the double ones handed in.
        for ( size_t l = 0; l < num_levels; ++l )
        {
            coords_g_.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< GeomM >( domains[l] ) );
            radii_g_.push_back( grid::shell::subdomain_shell_radii< GeomM >( domains[l] ) );
        }

        // Coarse operators and transfers in M, geometry in GeomM.
        for ( size_t l = 0; l < num_levels; ++l )
        {
            tmp_.emplace_back( "mp_tmp_" + std::to_string( l ), domains[l], mask_data[l] );
            if ( l < num_levels - 1 )
            {
                tmp_r_.emplace_back( "mp_tmp_r_" + std::to_string( l ), domains[l], mask_data[l] );
                tmp_e_.emplace_back( "mp_tmp_e_" + std::to_string( l ), domains[l], mask_data[l] );
                a_c_.emplace_back( domains[l], coords_g_[l], radii_g_[l],
                                   boundary_mask_data[l], eta_m_[l].grid_data(), bcs, false );
                a_c_.back().set_hourglass_coefficient( hourglass_c );
                p_.emplace_back( linalg::OperatorApplyMode::Add );
                r_.emplace_back( domains[l] );
            }
        }

        // Finest-level operator in M: Multigrid takes the fine operator at solve time,
        // and the one the outer solver passes is double, so this hierarchy owns its own.
        a_fine_ = std::make_unique< ViscM >( domains[fine], coords_g_[fine], radii_g_[fine],
                                             boundary_mask_data[fine], eta_m_[fine].grid_data(), bcs, false );
        a_fine_->set_hourglass_coefficient( hourglass_c );

        for ( size_t l = 0; l < num_levels; ++l )
        {
            std::vector< VecM > cheby_tmps;
            cheby_tmps.emplace_back( "mp_cheby_0_" + std::to_string( l ), domains[l], mask_data[l] );
            cheby_tmps.emplace_back( "mp_cheby_1_" + std::to_string( l ), domains[l], mask_data[l] );
            smoothers_.emplace_back( cheby_order, inv_diag_m_[l], cheby_tmps, chebyshev_prepost );
        }

        constexpr int coarse_restart = 30;
        for ( int i = 0; i < 2 * coarse_restart + 4; ++i )
            coarse_tmps_.emplace_back( "mp_coarse_tmp", domains[0], mask_data[0] );
        linalg::solvers::FGMRESOptions< M > copts;
        copts.restart                     = coarse_restart;
        copts.max_iterations              = ( coarse_tol < 1e-7 ) ? 300 : 50;
        // The tolerances live in FGMRES's dense precision, not in M: 1e-6 is not a
        // normal fp16 number and the comparison is host-side anyway.
        //
        // They are also floored at the working precision's unit roundoff. A coarse
        // solve cannot be driven below what M can represent: asking bf16 (eps 7.8e-3)
        // for 1e-6 only burns the iteration budget and trips the Arnoldi breakdown
        // guard on every restart -- 613 times in one run before this. Each precision
        // is asked for the best it can actually deliver, which leaves double at 1e-6
        // and costs float essentially nothing.
        coarse_tol_eff_                   = std::max( coarse_tol, 10.0 * util::epsilon_of< M >() );
        copts.relative_residual_tolerance = coarse_tol_eff_;
        copts.absolute_residual_tolerance = 1e-16;
        coarse_table_  = std::make_shared< util::Table >();
        coarse_solver_ = std::make_unique< CSolvM >( coarse_tmps_, copts, coarse_table_,
                                                     CPrecM( inv_diag_m_[0] ) );
        coarse_solver_->set_tag( "mp_coarse_fgmres" );

        // One V-cycle per application, no internal residual check.
        mg_ = std::make_unique< MGM >( p_, r_, a_c_, tmp_r_, tmp_e_, tmp_, smoothers_, smoothers_,
                                       *coarse_solver_, 1, util::narrow< M >( 1e-16 ) );

        x_m_      = std::make_unique< VecM >( "mp_x", domains[fine], mask_data[fine] );
        b_m_      = std::make_unique< VecM >( "mp_b", domains[fine], mask_data[fine] );
        b_scaled_ = std::make_unique< OuterVec >( "mp_b_scaled", domains[fine], mask_data[fine] );

        // Report the two coefficient fields against what M can represent. Anything
        // that overflows on conversion becomes an inf and the V-cycle returns NaN, so
        // this is the first thing to look at when a low-precision arm diverges.
        double eta_max = 0.0, inv_diag_max = 0.0;
        bool   eta_overflow = false, inv_diag_overflow = false;
        for ( size_t l = 0; l < num_levels; ++l )
        {
            eta_max      = std::max( eta_max, static_cast< double >( linalg::norm_inf( eta[l] ) ) );
            inv_diag_max = std::max( inv_diag_max, static_cast< double >( linalg::norm_inf( inverse_diagonals[l] ) ) );
            eta_overflow      = eta_overflow || linalg::has_nan_or_inf( eta_m_[l] );
            inv_diag_overflow = inv_diag_overflow || linalg::has_nan_or_inf( inv_diag_m_[l] );
        }
        util::logroot << "[LowPrecABlock] coarse rel-tol = " << coarse_tol_eff_ << " (requested " << coarse_tol
                      << ", floor 10*eps = " << 10.0 * util::epsilon_of< M >() << ")\n"
                      << "                M finite_max = "
                      // The numeric traits of the 16-bit types are bit_comparison_type, not M,
                      // so they go through M's constructor before they can reach double.
                      << static_cast< double >( M( Kokkos::Experimental::finite_max_v< M > ) )
                      << ", norm_min = " << static_cast< double >( M( Kokkos::Experimental::norm_min_v< M > ) )
                      << "\n                max|eta| = " << eta_max << ( eta_overflow ? "  [OVERFLOWS M]" : "" )
                      << ", max|diag^-1| = " << inv_diag_max << ( inv_diag_overflow ? "  [OVERFLOWS M]" : "" )
                      << std::endl;
    }

    void solve_impl( OuterOp& /*A_double*/, OuterVec& x, const OuterVec& b ) override
    {
        // The preconditioner is applied to whatever Krylov vector the outer FGMRES
        // hands it, whose magnitude follows the outer residual -- here O(1e2) at the
        // start. fp16 saturates at 65504 and has no normals below 6.1e-5, so the
        // input is normalized to unit max-norm before it is narrowed and the result is
        // scaled back afterwards. A V-cycle is linear, so this is exact up to the two
        // double multiplications.
        const double scale = static_cast< double >( linalg::norm_inf( b ) );
        if ( !( scale > 0.0 ) )
        {
            linalg::assign( x, 0.0 );
            return;
        }

        linalg::lincomb( *b_scaled_, { 1.0 / scale }, { b } );
        linalg::convert( *b_scaled_, *b_m_ );
        linalg::assign( *x_m_, util::narrow< M >( 0.0 ) );

        if ( probe_ )
        {
            probe_stages();
            probe_ = false;
        }

        {
            util::Timer t_vcycle( "mp_ablock_vcycle" );
            linalg::solvers::solve( *mg_, *a_fine_, *x_m_, *b_m_ );
        }
        linalg::convert( *x_m_, x );
        linalg::lincomb( x, { scale }, { x } );
    }

  public:
    // nvcc rejects an extended __host__ __device__ lambda whose enclosing
    // function has private or protected access, and probe_stages() runs a
    // KOKKOS_LAMBDA reduction, so it stays public. clang/HIP accepts either.
    /// @brief Walk the V-cycle one stage at a time on the first application and report
    ///        magnitudes and NaN/Inf, so a diverging low-precision arm can be pinned to
    ///        a stage instead of guessed at. Runs once, on scratch vectors.
    void probe_stages()
    {
        const size_t fine = tmp_.size() - 1;
        auto         rep  = [&]( const char* what, auto& v ) {
            util::logroot << "[probe] " << what << ": max|.| = " << static_cast< double >( linalg::norm_inf( v ) )
                          << ( linalg::has_nan_or_inf( v ) ? "   <-- NaN/Inf" : "" ) << std::endl;
        };

        // Separate "the 16-bit field cannot hold the value" from "the reduction over it
        // is wrong": write a known constant, read one entry back on the host, and
        // max-reduce the same view once in M (what norm_inf does) and once in double.
        {
            linalg::assign( *x_m_, util::narrow< M >( 3.5 ) );
            const auto& view = x_m_->grid_data().comp_[0];
            auto host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, view );
            double mx_double = 0.0;
            Kokkos::parallel_reduce(
                "probe_max",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 }, { view.extent( 0 ), view.extent( 1 ), view.extent( 2 ), view.extent( 3 ) } ),
                KOKKOS_LAMBDA( int a, int b2, int c, int d, double& acc ) {
                    const double v = Kokkos::abs( static_cast< double >( view( a, b2, c, d ) ) );
                    acc            = v > acc ? v : acc;
                },
                Kokkos::Max< double >( mx_double ) );
            Kokkos::fence();
            // Compare raw bit patterns, so a broken host-side conversion cannot hide a
            // correct value (or fake one). 3.5 is 0x4300 in fp16 and 0x4060 in bf16.
            const M  nrm = linalg::norm_inf( *x_m_ );
            const M  ref = static_cast< M >( 3.5 );
            uint16_t nrm_bits = 0, ref_bits = 0;
            if constexpr ( sizeof( M ) == 2 )
            {
                std::memcpy( &nrm_bits, &nrm, 2 );
                std::memcpy( &ref_bits, &ref, 2 );
            }
            // Host-side round trip on a value the compiler cannot fold away. 3.5 is 0x4300
            // in fp16 and 0x4060 in bf16; a wrong bit pattern means double -> M is
            // broken on the host, a wrong read-back means M -> double is.
            // A/B the two host-side narrowings on the SAME unfoldable input, reading the
            // bits of each with the same memcpy. If both give 0x4300 then static_cast is
            // fine here and util::narrow is redundant; if they differ, narrow is load
            // bearing and we have a toolchain problem to report.
            volatile double rt_in = 3.5;
            const M         rt_cast   = static_cast< M >( rt_in );
            const M         rt_narrow = util::narrow< M >( rt_in );
            uint16_t        bits_cast = 0, bits_narrow = 0;
            if constexpr ( sizeof( M ) == 2 )
            {
                std::memcpy( &bits_cast, &rt_cast, 2 );
                std::memcpy( &bits_narrow, &rt_narrow, 2 );
            }
            util::logroot << "[probe] host narrowing of a volatile 3.5:  static_cast -> 0x" << std::hex << bits_cast
                          << ", util::narrow -> 0x" << bits_narrow << std::dec
                          << "   (fp16 expects 0x4300, bf16 0x4060)" << std::endl;

            util::logroot << "[probe] assign 3.5: host mirror[0,1,1,1] = " << static_cast< double >( host( 0, 1, 1, 1 ) )
                          << ", Max<double> over the view = " << mx_double << "\n"
                          << "[probe]   host M(3.5) -> double = " << static_cast< double >( ref ) << ", bits = 0x"
                          << std::hex << ref_bits << std::dec << "\n"
                          << "[probe]   norm_inf (Max<M>) -> double = " << static_cast< double >( nrm )
                          << ", bits = 0x" << std::hex << nrm_bits << std::dec << std::endl;
            linalg::assign( *x_m_, util::narrow< M >( 0.0 ) );
        }

        rep( "b (converted to M)       ", *b_m_ );
        rep( "diag^-1 (fine, in M)     ", inv_diag_m_[fine] );

        linalg::apply( *a_fine_, *b_m_, tmp_[fine] );
        rep( "A_fine * b               ", tmp_[fine] );

        // The eigenvalue bound the Chebyshev smoother will use, computed exactly as it
        // does internally -- this is the step that consumes a full-level inner product.
        {
            linalg::DiagonallyScaledOperator< ViscM > dsa( *a_fine_, inv_diag_m_[fine] );
            const double lambda = linalg::solvers::power_iteration( dsa, tmp_[fine], *x_m_, 50 );
            util::logroot << "[probe] power_iteration lambda_max = " << lambda << std::endl;
            linalg::assign( *x_m_, util::narrow< M >( 0.0 ) );
        }

        linalg::solvers::solve( smoothers_[fine], *a_fine_, *x_m_, *b_m_ );
        rep( "after one fine smoother  ", *x_m_ );
        linalg::assign( *x_m_, util::narrow< M >( 0.0 ) );

        linalg::assign( tmp_e_.empty() ? *x_m_ : tmp_e_[0], util::narrow< M >( 0.0 ) );
        if ( !tmp_r_.empty() )
        {
            linalg::assign( tmp_r_[0], util::narrow< M >( 1.0 ) );
            linalg::solvers::solve( *coarse_solver_, a_c_[0], tmp_e_[0], tmp_r_[0] );
            rep( "after coarse solve       ", tmp_e_[0] );
        }
    }

  private:

    std::vector< ScalM > eta_m_;
    std::vector< VecM >  inv_diag_m_;
    std::vector< VecM >  tmp_, tmp_r_, tmp_e_;
    std::vector< Grid3DDataVec< GeomM, 3 > > coords_g_;
    std::vector< Grid2DDataScalar< GeomM > >  radii_g_;
    std::vector< ViscM > a_c_;
    std::vector< ProlM > p_;
    std::vector< RestM > r_;
    std::vector< SmooM > smoothers_;
    std::vector< VecM >  coarse_tmps_;

    std::shared_ptr< util::Table > coarse_table_;
    std::unique_ptr< ViscM >       a_fine_;
    std::unique_ptr< CSolvM >      coarse_solver_;
    std::unique_ptr< MGM >         mg_;
    std::unique_ptr< VecM >        x_m_, b_m_;
    std::unique_ptr< OuterVec >    b_scaled_;
    bool                           probe_ = true;
    double                         coarse_tol_eff_ = 0.0;
};

enum class BCKind { DIRICHLET, FREESLIP };

struct RunResult
{
    int    cycles;              // number of MG V-cycles performed
    double rel_residual_final;  // final relative residual
    double asymptotic_rate;     // residual_convergence_rate of the last cycle
    bool   converged;
};

// Build the viscous A-block MG and run it as the solver for  A u = f.
// gca: 0 = re-discretized coarse operators (DCA),
//      1 = full GCA (P^T A P on every coarse element),
//      2 = adaptive GCA (only on elements flagged by the gradient collector).
RunResult run_ablock_mg( int    min_level,
                         int    max_level,
                         int    cheby_order,
                         double perturb_amp,
                         double rmu,
                         int    gca,
                         bool   no_stokes_prec, // --no-stokes-prec: bypass PrecStokes entirely
                         int    max_cycles,
                         BCKind bc,
                         linalg::solvers::InterpolationMode gca_interp,
                         int    visc_profile,   // 0 = Frank-Kamenetskii (smooth), 1 = sharp radial layer
                         double coarse_tol,     // coarse FGMRES relative tolerance
                         bool   full_stokes,    // false = solve A-block with MG; true = full Stokes outer FGMRES
                         KrylovPrec krylov_prec, // Krylov-basis precision for the full-Stokes outer FGMRES
                         ABlockPrec  ablock_prec, // working precision of the A-block V-cycle preconditioner
                         ABlockArith ablock_arith, // fast-path arithmetic precision in that V-cycle
                         ABlockGeom  ablock_geom,  // geometry precision in that V-cycle
                         double      hourglass_c, // rank-6 hourglass stabilization coefficient (0 = off)
                         double      outer_tol,   // outer relative residual tolerance
                         OuterSolver outer_solver, // fgmres or pure iterative refinement
                         int         outer_restart_in ) // outer FGMRES restart length (<=0 = default 50)
{
    using ScalarType = double;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const int idx = level - min_level;
        domains.push_back( DistributedDomain::create_uniform( level, level, kRm, kRp, 0, 0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const size_t num_levels     = domains.size();
    const size_t velocity_level = num_levels - 1;
    const size_t pressure_level = num_levels - 2;

    const auto bc_flag = ( bc == BCKind::DIRICHLET ) ? DIRICHLET : FREESLIP;
    BoundaryConditions bcs = {
        { CMB, bc_flag },
        { SURFACE, bc_flag },
    };
    // GCA descent always coarsens from the Neumann block (gca.hpp does not
    // transfer Dirichlet/free-slip BCs; they are imposed on the fine operator
    // and -- for Dirichlet -- algebraically on the RHS).
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    using Stokes       = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous      = Stokes::Block11Type;
    using Gradient     = Stokes::Block12Type;
    using Divergence   = Stokes::Block21Type;
    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;
    using ViscousMass  = fe::wedge::operators::shell::VectorMass< ScalarType >;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    // Low-viscosity band: centre r=2.0, width 0.1 (half-width 0.05), eta = 1/rmu inside.
    const double r_center   = 2.0;
    const double half_width = 0.05;
    const double eta_band   = 1.0 / rmu;  // low-viscosity band, contrast = rmu

    // Per-level viscosity fields.
    std::vector< VectorQ1Scalar< ScalarType > > eta;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        eta.emplace_back( "eta_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( visc_profile == 2 || visc_profile == 3 )
        {
            // Published radial viscosity profile (min-normalized), interpolated by radius
            // onto this level's nodes.  T-independent.  2 = Stotz 2017 (~12000), 3 = Lin 2022 (~1057).
            const std::string csv =
                ( visc_profile == 2 )
                    ? "/home/hpc/iwia/iwia054h/terraneo/data/radialprofiles/ViscosityProfile_Stotz_et_al_2017.csv"
                    : "/home/hpc/iwia/iwia054h/terraneo/data/radialprofiles/ViscosityProfile_Lin_et_al_2022.csv";
            auto profile = shell::interpolate_radial_profile_into_subdomains_from_csv< ScalarType >(
                csv, "radius_normalized_1p22_2p22", "viscosity_scaled_by_min", coords_radii[level] );
            geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType >( profile, 1.0 )
                .interpolate( eta[level].grid_data() );
        }
        else if ( visc_profile == 1 )
        {
            Kokkos::parallel_for(
                "layer_viscosity_level_" + std::to_string( level ),
                local_domain_md_range_policy_nodes( domains[level] ),
                RadialBandViscosityInterpolator{
                    coords_shell[level], coords_radii[level], eta[level].grid_data(), eta_band, r_center, half_width } );
        }
        else
        {
            Kokkos::parallel_for(
                "fk_viscosity_level_" + std::to_string( level ),
                local_domain_md_range_policy_nodes( domains[level] ),
                FrankKamenetskiiViscosityInterpolator{
                    coords_shell[level], coords_radii[level], eta[level].grid_data(), rmu, perturb_amp } );
        }
        Kokkos::fence();
    }

    // Fine free-slip viscous block (the operator the MG solves), plus the
    // Neumann copy used as the GCA descent's finest source (matches the mc app:
    // GCA coarsens from K_neumann_->block_11()).
    Stokes K_op(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], eta[velocity_level].grid_data(),
        bcs, false );

    Stokes K_op_neumann(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], eta[velocity_level].grid_data(),
        bcs_neumann, false );

    // The hourglass correction has to be on the operator the V-cycle actually solves
    // as well as on its coarse levels, otherwise the smoother and the fine residual
    // disagree about what the operator is.
    K_op.block_11().set_hourglass_coefficient( hourglass_c );
    K_op_neumann.block_11().set_hourglass_coefficient( hourglass_c );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // GCAElements map on the coarsest grid: which coarsest-grid elements get GCA.
    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", domains[0], mask_data[0] );
    if ( gca == 2 )
    {
        linalg::assign( GCAElements, 0 );
        terra::linalg::solvers::GCAElementsCollector< ScalarType >(
            domains[velocity_level], eta[velocity_level].grid_data(),
            max_level - min_level, GCAElements.grid_data() );
    }
    else if ( gca == 1 )
    {
        linalg::assign( GCAElements, 1 );
    }

    // Coarse viscous operators + transfer operators.
    std::vector< Viscous >                   A_c;
    std::vector< Prolongation >              P;
    std::vector< Restriction >               R;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg, tmp_mg_r, tmp_mg_e;

    for ( size_t level = 0; level < num_levels; ++level )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );

            A_c.emplace_back(
                domains[level], coords_shell[level], coords_radii[level],
                boundary_mask_data[level], eta[level].grid_data(), bcs, false );
            A_c.back().set_hourglass_coefficient( hourglass_c );

            if ( gca == 1 )
            {
                A_c.back().set_stored_matrix_mode(
                    linalg::OperatorStoredMatrixMode::Full, static_cast< int >( level ), GCAElements.grid_data() );
            }
            else if ( gca == 2 )
            {
                A_c.back().set_stored_matrix_mode(
                    linalg::OperatorStoredMatrixMode::Selective, static_cast< int >( level ), GCAElements.grid_data() );
            }

            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // GCA assembly, top-down: coarsest source is the Neumann fine block, then
    // each coarse op is coarsened from the next-finer coarse op.
    if ( gca > 0 )
    {
        for ( int level = static_cast< int >( num_levels ) - 2; level >= 0; --level )
        {
            TwoGridGCA< ScalarType, Viscous >(
                ( level == static_cast< int >( num_levels ) - 2 ) ? K_op_neumann.block_11() : A_c[level + 1],
                A_c[level],
                level,
                GCAElements.grid_data(),
                /*treat_boundary=*/true,
                gca_interp );
        }
    }

    // Inverse diagonals per level -- computed AFTER GCA assembly so the coarse
    // diagonals reflect the stored (P^T A P) matrices.
    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        inverse_diagonals.emplace_back(
            "inv_diag_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > ones(
            "inv_diag_ones_" + std::to_string( level ), domains[level], mask_data[level] );
        linalg::assign( ones, 1.0 );

        if ( level == velocity_level )
        {
            K_op.block_11().set_diagonal( true );
            linalg::apply( K_op.block_11(), ones, inverse_diagonals.back() );
            K_op.block_11().set_diagonal( false );
        }
        else
        {
            A_c[level].set_diagonal( true );
            linalg::apply( A_c[level], ones, inverse_diagonals.back() );
            A_c[level].set_diagonal( false );
        }
        linalg::invert_entries( inverse_diagonals.back() );
    }

    // Chebyshev smoothers per level.
    using Smoother = linalg::solvers::Chebyshev< Viscous >;
    std::vector< Smoother > smoothers;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        std::vector< VectorQ1Vec< ScalarType > > cheby_tmps;
        cheby_tmps.emplace_back( "cheby_tmp_0_" + std::to_string( level ), domains[level], mask_data[level] );
        cheby_tmps.emplace_back( "cheby_tmp_1_" + std::to_string( level ), domains[level], mask_data[level] );

        constexpr int chebyshev_prepost = 3;
        smoothers.emplace_back( cheby_order, inverse_diagonals[level], cheby_tmps, chebyshev_prepost );
    }

    // Coarse FGMRES (matches the mc app's coarse solver).
    auto coarse_table = std::make_shared< util::Table >();
    using CoarseGridPrec   = InverseDiagonalPreconditioner< Viscous >;
    using CoarseGridSolver = linalg::solvers::FGMRES< Viscous, CoarseGridPrec >;
    constexpr int coarse_restart = 30;
    const int     coarse_max     = ( coarse_tol < 1e-7 ) ? 300 : 50;  // allow more iters for a tight coarse tol
    std::vector< VectorQ1Vec< ScalarType > > coarse_tmps;
    for ( int i = 0; i < 2 * coarse_restart + 4; ++i )
        coarse_tmps.emplace_back( "coarse_tmp", domains[0], mask_data[0] );
    linalg::solvers::FGMRESOptions< ScalarType > coarse_opts;
    coarse_opts.restart                     = coarse_restart;
    coarse_opts.max_iterations              = coarse_max;
    // Same unit-roundoff floor as the reduced-precision hierarchies (see
    // LowPrecABlock); inactive at double, where 10*eps is 2.2e-15.
    coarse_opts.relative_residual_tolerance = std::max( coarse_tol, 10.0 * util::epsilon_of< ScalarType >() );
    coarse_opts.absolute_residual_tolerance = 1e-16;
    CoarseGridSolver coarse_solver( coarse_tmps, coarse_opts, coarse_table, CoarseGridPrec( inverse_diagonals[0] ) );
    coarse_solver.set_tag( "coarse_grid_fgmres" );

    // Multigrid: A-block solver (max_cycles V-cycles) OR a single V-cycle when used
    // as the (1,1) preconditioner inside the full-Stokes outer FGMRES.
    using MG = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    auto mg_table = std::make_shared< util::Table >();
    MG  mg( P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_solver,
            full_stokes ? 1 : max_cycles, full_stokes ? 1e-16 : 1e-10 );
    mg.collect_statistics( mg_table );

    // RHS: M * (-Ra * T * r_hat), strong-enforced for free-slip.  Built on a
    // full IsoQ2Q1 vector to reuse the free-slip enforcement, then the velocity
    // block is the A-block RHS.
    VectorQ1IsoQ2Q1< ScalarType > f_full(
        "f_full", domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
    VectorQ1IsoQ2Q1< ScalarType > tmp_full(
        "tmp_full", domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );

    Kokkos::parallel_for(
        "radial_buoyancy_rhs",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        RadialBuoyancyRHSInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level],
            tmp_full.block_1().grid_data(), kRayleigh, perturb_amp } );
    Kokkos::fence();

    linalg::apply( M, tmp_full.block_1(), f_full.block_1() );
    if ( bc == BCKind::FREESLIP )
    {
        fe::strong_algebraic_freeslip_enforcement_in_place(
            f_full, coords_shell[velocity_level], boundary_mask_data[velocity_level], BOUNDARY );
    }
    else
    {
        // Homogeneous no-slip: zero the velocity RHS on both shells.
        fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
            f_full, boundary_mask_data[velocity_level], BOUNDARY );
    }

    if ( !full_stokes )
    {
        // ---- A-block: solve  A u = f  with multigrid as the solver. ----
        VectorQ1Vec< ScalarType > u_vel( "u_vel", domains[velocity_level], mask_data[velocity_level] );
        linalg::assign( u_vel, 0.0 );
        linalg::solvers::solve( mg, K_op.block_11(), u_vel, f_full.block_1() );

        mg_table->query_rows_equals( "tag", "multigrid" )
            .select_columns( { "absolute_residual", "relative_residual", "residual_convergence_rate" } )
            .print_pretty();

        const auto rows = mg_table->query_rows_equals( "tag", "multigrid" ).rows();
        const int    cycles = rows.empty() ? 0 : static_cast< int >( rows.size() ) - 1;
        const double rel_residual_final =
            rows.empty() ? 1.0 : std::get< double >( rows.back().at( "relative_residual" ) );
        double asymptotic_rate = std::numeric_limits< double >::quiet_NaN();
        if ( rows.size() >= 2 && rows.back().count( "residual_convergence_rate" ) )
            asymptotic_rate = std::get< double >( rows.back().at( "residual_convergence_rate" ) );
        return { cycles, rel_residual_final, asymptotic_rate, rel_residual_final <= 1e-6 };
    }

    // ---- Full Stokes: outer FGMRES on the saddle point, with the MG (DCA/GCA) as
    //      the (1,1) preconditioner and a w-BFBT Schur preconditioner. ----
    // Pressure mass on the pressure level, scaled by 1/eta (the (2,2) PrecStokes block).
    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[pressure_level], mask_data[pressure_level] );
    linalg::assign( k_pm, eta[pressure_level] );
    linalg::invert_entries( k_pm );
    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level], k_pm.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    // Schur approximation: the inverse of the lumped pressure mass, whose coefficient
    // is already 1/eta (k_pm above). The lumped matrix applied to a vector of ones is
    // its own diagonal, so one apply gives it.
    //
    // This replaces the w-BFBT Schur preconditioner, which was a second complete
    // solver -- an explicit K_w pressure Poisson solve capped at 200 iterations plus a
    // weighted lumped velocity mass -- running in double on every outer iteration. It
    // is a better Schur approximation under strong viscosity contrast, but it is not
    // what this test measures, and its cost and its own precision would confound the
    // A-block comparison.
    VectorQ1Scalar< ScalarType > pmass_inv_diag(
        "pmass_inv_diag", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > ones_p( "ones_p", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( ones_p, 1.0 );
        linalg::apply( pmass, ones_p, pmass_inv_diag );
        linalg::invert_entries( pmass_inv_diag );
    }

    using PrecSchur = InverseDiagonalPreconditioner< PressureMass >;
    PrecSchur prec_schur( pmass_inv_diag );

    // The (1,1) velocity preconditioner is type-erased so its internal precision --
    // the whole V-cycle hierarchy -- can be chosen at runtime via --ablock-prec, while
    // the outer FGMRES, the Stokes operator and the Schur approximation stay in double.
    using VelPrecHandle = linalg::solvers::VelocityPrecHandle< Viscous >;
    using PrecStokes    = linalg::solvers::BlockTriangularPreconditioner2x2<
        Stokes, Viscous, PressureMass, Gradient, VelPrecHandle, PrecSchur >;
    VectorQ1IsoQ2Q1< ScalarType > tri_tmp(
        "tri_tmp", domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
    std::shared_ptr< typename VelPrecHandle::Impl > vel_impl;
    // One instantiation per (storage, arithmetic, geometry) triple; the constructor
    // arguments are identical, so the body lives in a templated lambda. A reduced
    // geometry is only offered alongside reduced arithmetic: it is the arm "everything
    // the V-cycle can do in M", not a dimension to be crossed with the others.
    using ImplPtr  = std::shared_ptr< typename VelPrecHandle::Impl >;
    auto make_impl = [&]< typename M, typename ArithM, typename GeomM >() -> ImplPtr {
        return std::make_shared< LowPrecABlock< M, ArithM, GeomM > >(
            domains, coords_shell, coords_radii, mask_data, boundary_mask_data, eta,
            inverse_diagonals, bcs, cheby_order, 3, coarse_tol, hourglass_c );
    };
    using Half           = Kokkos::Experimental::half_t;
    using BHalf          = Kokkos::Experimental::bhalf_t;
    const bool low_arith = ( ablock_arith == ABlockArith::LOW );
    const bool low_geom  = low_arith && ( ablock_geom == ABlockGeom::LOW );
    switch ( ablock_prec )
    {
    case ABlockPrec::FLOAT:
        vel_impl = low_geom    ? make_impl.template operator()< float, float, float >()
                   : low_arith ? make_impl.template operator()< float, float, double >()
                               : make_impl.template operator()< float, double, double >();
        break;
    case ABlockPrec::FP16:
        vel_impl = low_geom    ? make_impl.template operator()< Half, Half, float >()
                   : low_arith ? make_impl.template operator()< Half, Half, double >()
                               : make_impl.template operator()< Half, double, double >();
        break;
    case ABlockPrec::BF16:
        vel_impl = low_geom    ? make_impl.template operator()< BHalf, BHalf, float >()
                   : low_arith ? make_impl.template operator()< BHalf, BHalf, double >()
                               : make_impl.template operator()< BHalf, double, double >();
        break;
    case ABlockPrec::NONE:
        // Control: no velocity preconditioner at all, so the outer FGMRES sees only
        // the Schur block. Everything the V-cycle buys is measured against this.
        vel_impl = std::make_shared< linalg::solvers::IdentityPrecImpl< Viscous > >();
        break;
    case ABlockPrec::DOUBLE:
    default:
        vel_impl = std::make_shared< linalg::solvers::ForwardingPrecImpl< Viscous, MG > >( mg );
        break;
    }
    VelPrecHandle vel_prec( vel_impl );
    PrecStokes prec_stokes( K_op.block_11(), pmass, K_op.block_12(), tri_tmp, vel_prec, prec_schur );

    // Restart length of the outer FGMRES. Setting it to max_cycles disables restarting
    // entirely (full GMRES), which costs 2*restart+4 Stokes vectors of workspace but
    // keeps the whole Krylov space -- the way to tell a genuine preconditioner floor
    // from a restart artifact.
    const int outer_restart = ( outer_restart_in > 0 ) ? outer_restart_in : 50;
    linalg::solvers::FGMRESOptions< ScalarType > outer_opts;
    outer_opts.restart                     = outer_restart;
    outer_opts.max_iterations              = max_cycles;  // reuse --max-cycles as outer iteration cap
    outer_opts.relative_residual_tolerance = outer_tol;
    outer_opts.absolute_residual_tolerance = 1e-16;
    auto outer_table = std::make_shared< util::Table >();

    VectorQ1IsoQ2Q1< ScalarType > u_stokes(
        "u_stokes", domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
    linalg::assign( u_stokes, 0.0 );

    // Outer FGMRES with the Krylov basis stored in double / single / bf16 / fp16.
    //   double           -> plain FGMRES (basis == solution precision),
    //   float/bf16/fp16  -> FGMRESLowMem (3 full-precision scratch + 2*restart+1
    //                       reduced-precision basis vectors),
    // mirroring the mc app's reduced-precision Krylov-basis workspace in
    // stokes_solver.hpp.
    if ( outer_solver == OuterSolver::IR )
    {
        // Pure iterative refinement. The residual and the update are formed in double;
        // the correction comes from one application of the block preconditioner, whose
        // (1,1) block runs at the precision under test. No Krylov acceleration, so the
        // per-iteration factor is the preconditioner's own.
        //
        // Rows go into the same table under the same tag as the FGMRES path, so the
        // convergence history is read out identically.
        VectorQ1IsoQ2Q1< ScalarType > r_ir( "r_ir", domains[velocity_level], domains[pressure_level],
                                            mask_data[velocity_level], mask_data[pressure_level] );
        VectorQ1IsoQ2Q1< ScalarType > d_ir( "d_ir", domains[velocity_level], domains[pressure_level],
                                            mask_data[velocity_level], mask_data[pressure_level] );

        const auto t_ir0      = std::chrono::steady_clock::now();
        auto       elapsed_ir = [&]() {
            return std::chrono::duration< double >( std::chrono::steady_clock::now() - t_ir0 ).count();
        };
        linalg::apply( K_op, u_stokes, r_ir );
        linalg::lincomb( r_ir, { 1.0, -1.0 }, { f_full, r_ir } );
        const double r0 = static_cast< double >( linalg::norm_2( r_ir ) );

        outer_table->add_row( { { "tag", std::string( "stokes_fgmres" ) },
                                { "iteration", 0 },
                                { "elapsed_s", elapsed_ir() },
                                { "relative_residual", 1.0 },
                                { "absolute_residual", r0 } } );

        for ( int k = 1; k <= max_cycles; ++k )
        {
            linalg::assign( d_ir, 0.0 );
            linalg::solvers::solve( prec_stokes, K_op, d_ir, r_ir );

            if ( linalg::has_nan_or_inf( d_ir ) )
            {
                util::logroot << "IR: the preconditioner returned NaN/Inf at iteration " << k
                              << "; stopping (no Krylov space to fall back on)." << std::endl;
                break;
            }

            linalg::lincomb( u_stokes, { 1.0, 1.0 }, { u_stokes, d_ir } );
            linalg::apply( K_op, u_stokes, r_ir );
            linalg::lincomb( r_ir, { 1.0, -1.0 }, { f_full, r_ir } );

            const double rk  = static_cast< double >( linalg::norm_2( r_ir ) );
            const double rel = ( r0 > 0.0 ) ? rk / r0 : 0.0;
            outer_table->add_row( { { "tag", std::string( "stokes_fgmres" ) },
                                    { "iteration", k },
                                    { "elapsed_s", elapsed_ir() },
                                    { "relative_residual", rel },
                                    { "absolute_residual", rk } } );
            if ( rel <= outer_tol )
                break;
        }
    }
    else if ( krylov_prec == KrylovPrec::DOUBLE )
    {
        std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
        for ( int i = 0; i < 2 * outer_restart + 4; ++i )
            tmp_fgmres.emplace_back( "tmp_fgmres_" + std::to_string( i ),
                domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
        if ( no_stokes_prec )
        {
            // Genuinely unpreconditioned: FGMRES's default preconditioner is
            // IdentitySolver, so the whole block-triangular operator is bypassed.
            // Distinct from --ablock-prec none, which keeps the Schur block and only
            // makes the velocity block identity.
            linalg::solvers::FGMRES< Stokes > fgmres( tmp_fgmres, outer_opts, outer_table );
            // Same tag as the preconditioned branch: the table is printed and
            // reduced via query_rows_equals( "tag", "stokes_fgmres" ), so a
            // distinct tag here silently yields an empty convergence table.
            fgmres.set_tag( "stokes_fgmres" );
            linalg::solvers::solve( fgmres, K_op, u_stokes, f_full );
        }
        else
        {
            linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( tmp_fgmres, outer_opts, outer_table, prec_stokes );
            fgmres.set_tag( "stokes_fgmres" );
            linalg::solvers::solve( fgmres, K_op, u_stokes, f_full );
        }
    }
    else
    {
        std::vector< VectorQ1IsoQ2Q1< ScalarType > > work;  // FGMRESLowMem aliases r/w => 3 scratch
        for ( int i = 0; i < 3; ++i )
            work.emplace_back( "work_fgmres_" + std::to_string( i ),
                domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
        // One instantiation per basis scalar; the solve is otherwise identical, so
        // the body lives in a templated lambda rather than three copies.
        auto solve_lowmem = [&]< typename BasisScalar >()
        {
            using Basis = VectorQ1IsoQ2Q1< BasisScalar >;
            std::vector< Basis > basis;
            for ( int i = 0; i < 2 * outer_restart + 1; ++i )
                basis.emplace_back( "basis_fgmres_" + std::to_string( i ),
                    domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
            linalg::solvers::FGMRESLowMem< Stokes, Basis, PrecStokes > fgmres(
                work, basis, outer_opts, outer_table, prec_stokes );
            fgmres.set_tag( "stokes_fgmres" );
            linalg::solvers::solve( fgmres, K_op, u_stokes, f_full );
        };

        if ( krylov_prec == KrylovPrec::FLOAT )
            solve_lowmem.template operator()< float >();
        else if ( krylov_prec == KrylovPrec::BF16 )
            solve_lowmem.template operator()< Kokkos::Experimental::bhalf_t >();
        else  // FP16
            solve_lowmem.template operator()< Kokkos::Experimental::half_t >();
    }

    outer_table->query_rows_equals( "tag", "stokes_fgmres" )
        .select_columns( { "absolute_residual", "relative_residual", "iteration", "elapsed_s" } )
        .print_pretty();

    const auto rows    = outer_table->query_rows_equals( "tag", "stokes_fgmres" ).rows();
    const int  iters   = rows.empty() ? 0 : static_cast< int >( rows.size() ) - 1;
    const double rel_residual_final =
        rows.empty() ? 1.0 : std::get< double >( rows.back().at( "relative_residual" ) );
    return { iters, rel_residual_final, std::numeric_limits< double >::quiet_NaN(), rel_residual_final <= 1e-6 };
}

// Direct fidelity check: does GCA's stored-matrix coarse operator reproduce the
// TRUE Galerkin action R(A(P(x)))?  Builds a single coarse/fine level pair, the
// Neumann fine block A (the GCA descent's source), coarsens it into A_c via
// TwoGridGCA, then compares  A_c·x  against  R(A(P(x)))  using the MG's own
// constant prolongation P and restriction R.  Also checks constant-mode
// reproduction (A_c·1 vs R(A(P(1)))).  Neumann BCs throughout so no apply-time
// boundary masking muddies the interior comparison.
void verify_gca_rap( int max_level, double rmu, linalg::solvers::InterpolationMode interp )
{
    using ScalarType = double;
    using Stokes     = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous    = Stokes::Block11Type;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    const int cl = max_level - 1;  // coarse level
    const int fl = max_level;      // fine level

    auto make = [&]( int lvl ) {
        auto dom = DistributedDomain::create_uniform( lvl, lvl, kRm, kRp, 0, 0 );
        return dom;
    };
    DistributedDomain dom_c = make( cl ), dom_f = make( fl );
    auto coords_c = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_c );
    auto coords_f = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_f );
    auto radii_c  = grid::shell::subdomain_shell_radii< ScalarType >( dom_c );
    auto radii_f  = grid::shell::subdomain_shell_radii< ScalarType >( dom_f );
    auto mask_c   = grid::setup_node_ownership_mask_data( dom_c );
    auto mask_f   = grid::setup_node_ownership_mask_data( dom_f );
    auto bmask_c  = grid::shell::setup_boundary_mask_data( dom_c );
    auto bmask_f  = grid::shell::setup_boundary_mask_data( dom_f );

    BoundaryConditions bcs_neumann = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };

    // eta on both levels (FK law).
    VectorQ1Scalar< ScalarType > eta_c( "eta_c", dom_c, mask_c ), eta_f( "eta_f", dom_f, mask_f );
    Kokkos::parallel_for( "eta_c", local_domain_md_range_policy_nodes( dom_c ),
        FrankKamenetskiiViscosityInterpolator{ coords_c, radii_c, eta_c.grid_data(), rmu, kPerturbDefault } );
    Kokkos::parallel_for( "eta_f", local_domain_md_range_policy_nodes( dom_f ),
        FrankKamenetskiiViscosityInterpolator{ coords_f, radii_f, eta_f.grid_data(), rmu, kPerturbDefault } );
    Kokkos::fence();

    // Fine Neumann viscous block (GCA source) and coarse op (Neumann, stored).
    Viscous A_fine( dom_f, coords_f, radii_f, bmask_f, eta_f.grid_data(), bcs_neumann, false );
    Viscous A_c( dom_c, coords_c, radii_c, bmask_c, eta_c.grid_data(), bcs_neumann, false );

    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", dom_c, mask_c );
    linalg::assign( GCAElements, 1 );
    A_c.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Full, 0, GCAElements.grid_data() );

    TwoGridGCA< ScalarType, Viscous >( A_fine, A_c, 0, GCAElements.grid_data(), /*treat_boundary=*/true, interp );

    // A_self: the SAME fine operator, but in stored mode populated with its own
    // assemble_local_matrix (the matrices GCA reads), applied element-wise via the
    // slow path.  Comparing A_self vs the live A_fine isolates Q1 (assemble vs
    // apply); comparing A_c vs R*A_self*P isolates Q2 (the GCA triple product).
    Viscous A_self( dom_f, coords_f, radii_f, bmask_f, eta_f.grid_data(), bcs_neumann, false );
    VectorQ1Scalar< ScalarType > GCAElements_f( "GCAElements_f", dom_f, mask_f );
    linalg::assign( GCAElements_f, 1 );
    A_self.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Full, 0, GCAElements_f.grid_data() );
    {
        const auto& di = dom_f.domain_info();
        Kokkos::parallel_for(
            "populate_self",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { static_cast< long long >( dom_f.subdomains().size() ),
                  di.subdomain_num_nodes_per_side_laterally() - 1,
                  di.subdomain_num_nodes_per_side_laterally() - 1,
                  di.subdomain_num_nodes_radially() - 1 } ),
            KOKKOS_LAMBDA( const int sd, const int xx, const int yy, const int rr ) {
                A_self.set_local_matrix( sd, xx, yy, rr, 0, A_self.assemble_local_matrix( sd, xx, yy, rr, 0 ) );
                A_self.set_local_matrix( sd, xx, yy, rr, 1, A_self.assemble_local_matrix( sd, xx, yy, rr, 1 ) );
            } );
        Kokkos::fence();
    }

    Prolongation P( linalg::OperatorApplyMode::Replace );
    Restriction  R( dom_c );

    VectorQ1Vec< ScalarType > xc( "xc", dom_c, mask_c ), y_gca( "y_gca", dom_c, mask_c ),
        y_rap( "y_rap", dom_c, mask_c ), y_rap_self( "y_rap_self", dom_c, mask_c ), diff( "diff", dom_c, mask_c );
    VectorQ1Vec< ScalarType > xf( "xf", dom_f, mask_f ), axf( "axf", dom_f, mask_f ),
        axf_self( "axf_self", dom_f, mask_f ), fdiff( "fdiff", dom_f, mask_f );

    const char* interp_name = ( interp == linalg::solvers::InterpolationMode::Linear ) ? "linear" : "constant";
    util::logroot << "\n=== GCA fidelity check  (coarse lvl " << cl << " <- fine lvl " << fl
                  << ", rmu=" << rmu << ", interp=" << interp_name << ") ===\n";

    // Test vectors: a smooth field, and the constant field (constant-reproduction).
    for ( int tc = 0; tc < 2; ++tc )
    {
        if ( tc == 0 )
        {
            auto d = xc.grid_data();
            Kokkos::parallel_for( "fill_smooth", local_domain_md_range_policy_nodes( dom_c ),
                KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                    for ( int c = 0; c < 3; ++c )
                        d( sd, x, y, r, c ) =
                            Kokkos::sin( 0.7 * ( x + 1 ) + 0.3 * ( y + 1 ) + 0.5 * ( r + 1 ) + c );
                } );
            Kokkos::fence();
        }
        else
        {
            linalg::assign( xc, 1.0 );
        }

        linalg::apply( A_c, xc, y_gca );          // GCA stored-matrix action
        linalg::apply( P, xc, xf );               // prolongate coarse -> fine
        linalg::apply( A_fine, xf, axf );         // fine operator (live apply)
        linalg::apply( R, axf, y_rap );           // restrict fine -> coarse
        linalg::apply( A_self, xf, axf_self );    // fine operator (assemble_local_matrix, slow)
        linalg::apply( R, axf_self, y_rap_self ); // restrict fine -> coarse

        linalg::lincomb( diff, { 1.0, -1.0 }, { y_gca, y_rap } );
        const double n_gca  = linalg::norm_2( y_gca );
        const double n_rap  = linalg::norm_2( y_rap );
        const double n_diff = linalg::norm_2( diff );
        const double rel    = ( n_rap > 0 ) ? n_diff / n_rap : n_diff;

        util::logroot << "  [" << ( tc == 0 ? "smooth x" : "constant x" ) << "]  "
                      << "||A_c x|| = " << n_gca
                      << "   ||R A P x|| = " << n_rap
                      << "   ||A_c x - R A P x|| = " << n_diff
                      << "   rel = " << rel << "\n";

        if ( tc == 0 )
        {
            // Q1: live apply vs assemble_local_matrix (on the fine grid).
            linalg::lincomb( fdiff, { 1.0, -1.0 }, { axf_self, axf } );
            const double q1 = linalg::norm_2( axf ) > 0 ? linalg::norm_2( fdiff ) / linalg::norm_2( axf ) : 0.0;
            // Q2: GCA triple product vs true Galerkin of the SAME element matrices.
            linalg::lincomb( diff, { 1.0, -1.0 }, { y_gca, y_rap_self } );
            const double q2 = linalg::norm_2( y_rap_self ) > 0
                                  ? linalg::norm_2( diff ) / linalg::norm_2( y_rap_self ) : 0.0;
            util::logroot << "    Q1 (assemble_local_matrix vs live apply): rel = " << q1 << "\n"
                          << "    Q2 (GCA triple-product vs R*A_self*P):    rel = " << q2 << "\n";
        }
    }
    util::logroot << "  (rel ~ 0  => GCA faithfully assembles R A P; large => assembly bug)\n";

    // -------- Search the crossover-DoF permutation that makes gcafix == R*A_self*P --------
    // Reference: true Galerkin of the assemble_local_matrix operator, on smooth x.
    {
        auto d = xc.grid_data();
        Kokkos::parallel_for( "fill_smooth2", local_domain_md_range_policy_nodes( dom_c ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                for ( int c = 0; c < 3; ++c )
                    d( sd, x, y, r, c ) = Kokkos::sin( 0.7 * ( x + 1 ) + 0.3 * ( y + 1 ) + 0.5 * ( r + 1 ) + c );
            } );
        Kokkos::fence();
    }
    linalg::apply( P, xc, xf );
    linalg::apply( A_self, xf, axf_self );
    linalg::apply( R, axf_self, y_rap_self );
    const double ref_norm = linalg::norm_2( y_rap_self );

    Viscous A_c_fix( dom_c, coords_c, radii_c, bmask_c, eta_c.grid_data(), bcs_neumann, false );
    A_c_fix.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Full, 0, GCAElements.grid_data() );

    const auto interp_fix = ( interp == linalg::solvers::InterpolationMode::Linear )
                                ? linalg::solvers::gcafix::InterpolationMode::Linear
                                : linalg::solvers::gcafix::InterpolationMode::Constant;

    int    perm[6] = { 0, 1, 2, 3, 4, 5 };
    int    best_perm[6];
    double best_rel = 1e300, identity_rel = -1.0;
    do
    {
        linalg::solvers::gcafix::TwoGridGCA< ScalarType, Viscous >(
            A_fine, A_c_fix, 0, GCAElements.grid_data(), /*treat_boundary=*/true, interp_fix, perm );
        linalg::apply( A_c_fix, xc, y_gca );
        linalg::lincomb( diff, { 1.0, -1.0 }, { y_gca, y_rap_self } );
        const double rel = ref_norm > 0 ? linalg::norm_2( diff ) / ref_norm : 0.0;
        const bool   is_identity =
            ( perm[0] == 0 && perm[1] == 1 && perm[2] == 2 && perm[3] == 3 && perm[4] == 4 && perm[5] == 5 );
        if ( is_identity )
            identity_rel = rel;
        if ( rel < best_rel )
        {
            best_rel = rel;
            for ( int i = 0; i < 6; ++i )
                best_perm[i] = perm[i];
        }
    } while ( std::next_permutation( perm, perm + 6 ) );

    util::logroot << "  [crossover-perm search]  identity rel = " << identity_rel
                  << "   best rel = " << best_rel << "   best perm = ["
                  << best_perm[0] << " " << best_perm[1] << " " << best_perm[2] << " "
                  << best_perm[3] << " " << best_perm[4] << " " << best_perm[5] << "]\n";

    // -------- Linear-reproduction test (first-order consistency of the coarse op) --------
    // x_lin = position field (a global linear/constant-strain field). For eta=1 Neumann
    // operators, A * x_lin = 0 at INTERIOR nodes (constant stress, zero divergence).
    // P_mg reproduces linears, so R*A_self*P*x_lin is ~0 in the interior.  If A_c (GCA)
    // does NOT annihilate x_lin in the interior, GCA's interpolation is not first-order
    // consistent -> that is the root cause of the 26% mismatch.
    {
        auto       d   = xc.grid_data();
        const auto g   = coords_c;
        const auto rad = radii_c;
        Kokkos::parallel_for( "fill_linear", local_domain_md_range_policy_nodes( dom_c ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, g, rad );
                for ( int comp = 0; comp < 3; ++comp )
                    d( sd, x, y, r, comp ) = c( comp );
            } );
        Kokkos::fence();
    }
    linalg::apply( A_c, xc, y_gca );
    linalg::apply( P, xc, xf );
    linalg::apply( A_self, xf, axf_self );
    linalg::apply( R, axf_self, y_rap_self );

    // Zero the boundary (CMB/SURFACE) coarse nodes so we measure the INTERIOR only.
    auto zero_boundary = [&]( VectorQ1Vec< ScalarType >& v ) {
        auto       vd = v.grid_data();
        const auto bm = bmask_c;
        Kokkos::parallel_for( "zero_bnd", local_domain_md_range_policy_nodes( dom_c ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                if ( util::has_flag( bm( sd, x, y, r ), CMB ) || util::has_flag( bm( sd, x, y, r ), SURFACE ) )
                    for ( int comp = 0; comp < 3; ++comp )
                        vd( sd, x, y, r, comp ) = 0.0;
            } );
        Kokkos::fence();
    };
    const double full_gca = linalg::norm_2( y_gca );
    const double full_ref = linalg::norm_2( y_rap_self );
    zero_boundary( y_gca );
    zero_boundary( y_rap_self );
    util::logroot << "  [linear x_lin]  interior ||A_c x_lin|| = " << linalg::norm_2( y_gca )
                  << "   interior ||R A P x_lin|| = " << linalg::norm_2( y_rap_self )
                  << "   (full: " << full_gca << " vs " << full_ref << ")\n"
                  << "    (interior ref ~0 & GCA large => GCA P does NOT reproduce linears => root cause)\n";
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int    min_level   = 2;
    int    max_level   = 5;
    int    cheby_order = 4;
    int    max_cycles  = 50;
    double perturb_amp = kPerturbDefault;
    int    single_gca  = -1;     // -1 => sweep schemes; otherwise run that gca only
    double single_rmu  = -1.0;   // <0 => sweep the rmu ladder; otherwise single rmu
    BCKind bc          = BCKind::DIRICHLET;  // default: full (no-slip) Dirichlet
    using linalg::solvers::InterpolationMode;
    InterpolationMode single_interp = InterpolationMode::Constant;  // for --gca single runs
    int    visc_profile = 0;  // 0 = Frank-Kamenetskii (smooth), 1 = sharp radial layer
    double coarse_tol   = 1e-6;  // coarse FGMRES relative tolerance
    bool   full_stokes  = false; // --solve stokes: outer FGMRES on full saddle point
    KrylovPrec krylov_prec = KrylovPrec::DOUBLE; // --krylov-precision double|single|bf16|fp16 (full-Stokes only)
    ABlockPrec  ablock_prec  = ABlockPrec::DOUBLE;  // --ablock-prec double|float|fp16|bf16 (full-Stokes only)
    ABlockArith ablock_arith = ABlockArith::LOW; // --ablock-arith low|double (low = follow --ablock-prec)
    ABlockGeom  ablock_geom  = ABlockGeom::DOUBLE; // --ablock-geom double|low (low = float coordinates and Jacobian)
    double      hourglass_c  = 0.0;              // --hourglass C (0 = off, 0.3 as in EpsDivDivSimpleStab)
    bool        no_stokes_prec = false;         // --no-stokes-prec (unpreconditioned control)
    double      outer_tol    = 1e-6;             // --outer-tol T
    OuterSolver outer_solver = OuterSolver::FGMRES; // --outer-solver fgmres|ir
    int         outer_restart_in = 0;               // --outer-restart N (0 = default 50)
    for ( int i = 1; i + 1 < argc; ++i )
    {
        const std::string a = argv[i];
        if ( a == "--min-level" )           min_level   = std::atoi( argv[i + 1] );
        else if ( a == "--max-level" )      max_level   = std::atoi( argv[i + 1] );
        else if ( a == "--cheby-order" )    cheby_order = std::atoi( argv[i + 1] );
        else if ( a == "--max-cycles" )     max_cycles  = std::atoi( argv[i + 1] );
        else if ( a == "--perturb-amp" )    perturb_amp = std::atof( argv[i + 1] );
        else if ( a == "--gca" )            single_gca  = std::atoi( argv[i + 1] );
        else if ( a == "--rmu" )            single_rmu  = std::atof( argv[i + 1] );
        else if ( a == "--bc" )
            bc = ( std::string( argv[i + 1] ) == "freeslip" ) ? BCKind::FREESLIP : BCKind::DIRICHLET;
        else if ( a == "--gca-interp" )
            single_interp = ( std::string( argv[i + 1] ) == "linear" ) ? InterpolationMode::Linear
                                                                       : InterpolationMode::Constant;
        else if ( a == "--visc-profile" )
            visc_profile = ( std::string( argv[i + 1] ) == "layer" )   ? 1
                           : ( std::string( argv[i + 1] ) == "stotz" )  ? 2
                           : ( std::string( argv[i + 1] ) == "lin" )    ? 3
                                                                        : 0;
        else if ( a == "--coarse-tol" )    coarse_tol  = std::atof( argv[i + 1] );
        else if ( a == "--solve" )         full_stokes = ( std::string( argv[i + 1] ) == "stokes" );
        else if ( a == "--outer-solver" )
        {
            const std::string q = argv[i + 1];
            outer_solver        = ( q == "ir" || q == "refinement" ) ? OuterSolver::IR : OuterSolver::FGMRES;
        }
        else if ( a == "--hourglass" )      hourglass_c = std::atof( argv[i + 1] );
        else if ( a == "--outer-tol" )      outer_tol   = std::atof( argv[i + 1] );
        else if ( a == "--outer-restart" )  outer_restart_in = std::atoi( argv[i + 1] );
        else if ( a == "--ablock-arith" )
        {
            const std::string q = argv[i + 1];
            ablock_arith        = ( q == "low" || q == "field" || q == "reduced" ) ? ABlockArith::LOW
                                                                                  : ABlockArith::DOUBLE;
        }
        else if ( a == "--ablock-geom" )
        {
            const std::string q = argv[i + 1];
            ablock_geom         = ( q == "low" || q == "float" || q == "reduced" ) ? ABlockGeom::LOW
                                                                                  : ABlockGeom::DOUBLE;
        }
        else if ( a == "--ablock-prec" )
        {
            const std::string q = argv[i + 1];
            ablock_prec = ( q == "none" || q == "identity" )              ? ABlockPrec::NONE
                          : ( q == "fp16" || q == "half" )                ? ABlockPrec::FP16
                          : ( q == "bf16" || q == "bfloat16" )            ? ABlockPrec::BF16
                          : ( q == "float" || q == "single" )             ? ABlockPrec::FLOAT
                                                                          : ABlockPrec::DOUBLE;
        }
        else if ( a == "--krylov-precision" )
        {
            const std::string p = argv[i + 1];
            // NOTE: "half" now means binary16 (half_t), not bfloat16. It used to
            // alias bf16, which was misleading once both types are selectable.
            krylov_prec = ( p == "bf16" || p == "bfloat16" || p == "bhalf" ) ? KrylovPrec::BF16
                          : ( p == "fp16" || p == "half" || p == "half_t" )  ? KrylovPrec::FP16
                          : ( p == "single" || p == "float" )                ? KrylovPrec::FLOAT
                                                                             : KrylovPrec::DOUBLE;
        }
    }
    // Valueless flag, so it gets its own scan: the loop above advances by one and
    // stops at i + 1 < argc, which would miss it in the final argument position.
    for ( int i = 1; i < argc; ++i )
        if ( std::string( argv[i] ) == "--no-stokes-prec" ) no_stokes_prec = true;

    bool verify_gca = false;
    for ( int i = 1; i < argc; ++i )
        if ( std::string( argv[i] ) == "--verify-gca" ) verify_gca = true;

    if ( verify_gca )
    {
        const double r = ( single_rmu > 0.0 ) ? single_rmu : 1.0;
        verify_gca_rap( max_level, r, InterpolationMode::Constant );
        verify_gca_rap( max_level, r, InterpolationMode::Linear );
        return 0;
    }
    const char* bc_name = ( bc == BCKind::DIRICHLET ) ? "dirichlet/dirichlet (no-slip)" : "freeslip/freeslip";

    // Published radial profiles (Stotz/Lin) are fixed (rmu irrelevant) => single run.
    std::vector< double > rmus = ( visc_profile == 2 || visc_profile == 3 )
                                     ? std::vector< double >{ 1.0 }
                                     : ( single_rmu > 0.0 )
                                           ? std::vector< double >{ single_rmu }
                                           : std::vector< double >{ 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6 };

    // A scheme = (gca mode, GCA interpolation, label).  Default sweep compares
    // re-discretized coarse ops against full GCA with constant vs linear P.
    struct Scheme { int gca; InterpolationMode interp; const char* name; };
    std::vector< Scheme > schemes;
    if ( single_gca >= 0 )
        schemes = { { single_gca, single_interp,
                      ( single_gca == 0 )
                          ? "DCA(re-discretized)"
                          : ( single_interp == InterpolationMode::Linear ? "GCA(linear)" : "GCA(constant)" ) } };
    else
        schemes = {
            { 0, InterpolationMode::Constant, "DCA(re-discretized)" },
            { 1, InterpolationMode::Constant, "GCA(constant)" },
        };

    util::logroot << "=== " << ( full_stokes ? "FULL STOKES outer-FGMRES (w-BFBT Schur) " : "A-block (viscous) MG " )
                  << "convergence vs viscosity contrast ===\n"
                  << "    solve = " << ( full_stokes ? "stokes (cycles = outer FGMRES iters)" : "ablock (cycles = MG V-cycles)" )
                  << ( full_stokes ? std::string( ", outer = " ) + outer_solver_name( outer_solver )
                                           + ", krylov-basis = " + krylov_prec_name( krylov_prec )
                                           + krylov_prec_storage_note( krylov_prec )
                                           + ( no_stokes_prec ? std::string( ", UNPRECONDITIONED (PrecStokes bypassed)" )
                                                              : std::string() )
                                           + ", A-block prec = " + ablock_prec_name( ablock_prec )
                                           + ablock_prec_note( ablock_prec )
                                           + ", A-block arith = " + ablock_arith_name( ablock_arith )
                                           + ", A-block geom = " + ablock_geom_name( ablock_geom )
                                           + ", hourglass = " + std::to_string( hourglass_c )
                                           + ", outer tol = " + std::to_string( outer_tol )
                                           + ", outer restart = "
                                           + std::to_string( outer_restart_in > 0 ? outer_restart_in : 50 )
                                   : std::string() )
                  << ", BC = " << bc_name
                  << ", viscosity = "
                  << ( visc_profile == 3   ? "Lin et al. 2022 radial profile (contrast ~1057)"
                       : visc_profile == 2 ? "Stotz et al. 2017 radial profile (contrast ~12000)"
                       : visc_profile == 1 ? "low-viscosity band (r=2.0, width 0.1)"
                                           : "Frank-Kamenetskii (smooth)" )
                  << ", levels " << min_level << ".." << max_level
                  << ", cheby_order=" << cheby_order
                  << ", max_cycles=" << max_cycles
                  << ", coarse_tol=" << coarse_tol
                  << ", perturb_amp=" << perturb_amp << "\n";

    auto summary = std::make_shared< util::Table >();

    for ( double rmu : rmus )
    {
        for ( const auto& s : schemes )
        {
            util::logroot << "\n--- rmu = " << rmu << ", " << s.name << " ---\n";

            const auto res =
                run_ablock_mg( min_level, max_level, cheby_order, perturb_amp, rmu, s.gca, no_stokes_prec,
                               max_cycles, bc, s.interp,
                               visc_profile, coarse_tol, full_stokes, krylov_prec, ablock_prec, ablock_arith, ablock_geom,
                               hourglass_c, outer_tol, outer_solver, outer_restart_in );

            summary->add_row(
                { { "rmu", rmu },
                  { "gca", s.gca },
                  { "krylov", std::string( krylov_prec_name( krylov_prec ) ) },
                  { "ablock_prec",
                    no_stokes_prec ? std::string( "unpreconditioned" )
                                   : std::string( ablock_prec_name( ablock_prec ) ) },
                  { "ablock_arith", std::string( ablock_arith_name( ablock_arith ) ) },
                  { "ablock_geom", std::string( ablock_geom_name( ablock_geom ) ) },
                  { "hourglass", hourglass_c },
                  { "outer_solver", std::string( outer_solver_name( outer_solver ) ) },
                  { "scheme", std::string( s.name ) },
                  { "cycles", res.cycles },
                  { "final_rel_res", res.rel_residual_final },
                  { "asympt_rate", res.asymptotic_rate },
                  { "converged", res.converged ? 1 : 0 } } );
        }
    }

    util::logroot << "\n=== SUMMARY: A-block MG convergence ===\n";
    summary->select_columns( { "rmu", "krylov", "scheme", "cycles", "final_rel_res", "asympt_rate", "converged" } )
        .print_pretty();

    // Where the wall clock went (per-rank timer tree: fgmres_*, stokes_apply, mgmp_*,
    // epsilon_divdiv_*), for the time-to-solution breakdown.
    util::logroot << "\n=== TIMERS (json) ===\n" << util::TimerTree::instance().json() << std::endl;

    return 0;
}
