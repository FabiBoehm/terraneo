#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "fe/wedge/operators/shell/div_k_grad.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "grid/bit_masks.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/bit_masks.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"

#include "wbfbt_pressure_poisson.hpp"

namespace terra::mantlecirculation {

/// @brief K_w solver: re-discretized DivKGrad on continuous Q1 pressure space
///        with coefficient k = 1/sqrt(eta), inverted by a scalar-Q1 multigrid
///        V-cycle (Chebyshev smoothers + PCG coarse solve).
///
/// The MG hierarchy lives on the existing StokesContext domains at the
/// pressure level and below.  The fine operator and coarse operators all
/// reference the same per-level `inv_sqrt_eta_` grids — updating those in
/// place (during `refresh()`) is enough to repropagate new viscosity into
/// every level's operator without rebuilding the operators or the MG stack.
///
/// Boundary conditions: Neumann (consistent with the B*C_w^-1*B^T action).
/// The resulting K_w has a 1D constant nullspace; `solve()` projects it out
/// of the rhs and solution.
///
/// @note v1 simplifications, documented for future cleanup:
///   - No MG agglomeration support yet.
///   - On coarse levels the inv_sqrt_eta coefficient is restricted from the
///     finest pressure level via the same RestrictionConstant used in the MG
///     hierarchy (arithmetic-like, mass-lumped).  This is approximate; a
///     proper L2 projection (as in Rudi's HMG) could be added later.
///   - `refresh()` re-computes inverse diagonals and forces Chebyshev to
///     re-estimate max eigenvalues on the next solve.
template < typename ScalarType >
class DivKGradMGPressurePoissonSolver final : public WBFBTPressurePoissonSolver< ScalarType >
{
  public:
    using PressureVector = linalg::VectorQ1Scalar< ScalarType >;

    using DivKGrad         = fe::wedge::operators::shell::DivKGrad< ScalarType >;
    using Prolongation     = fe::wedge::operators::shell::ProlongationConstant< ScalarType >;
    using Restriction      = fe::wedge::operators::shell::RestrictionConstant< ScalarType >;
    using Smoother         = linalg::solvers::Chebyshev< DivKGrad >;
    using CoarseGridSolver = linalg::solvers::PCG< DivKGrad >;
    using MG               = linalg::solvers::Multigrid< DivKGrad, Prolongation, Restriction, Smoother, CoarseGridSolver >;

    /// @param domains            One DistributedDomain per level, levels 0..pressure_level inclusive (same as StokesContext::domains_).
    /// @param coords_shell       Per-level shell coords (unit sphere coordinates per node).
    /// @param coords_radii       Per-level radial coords.
    /// @param ownership_mask     Per-level node-ownership masks.
    /// @param boundary_mask      Per-level boundary masks (used by DivKGrad to ignore Dirichlet rows; we use treat_boundary=false → Neumann here).
    /// @param pressure_level     Index of the finest level used by this solver.
    /// @param chebyshev_order    Chebyshev polynomial order for smoothers.
    /// @param prepost_smoothing  Number of pre/post smoothing applications per V-cycle.
    /// @param num_vcycles        Number of V-cycles per `solve()` call.
    /// @param relative_tol       Outer relative-residual tolerance for the multigrid wrapper (loose by default; we cap iters via num_vcycles).
    /// @param table              Optional table for solver statistics; pass a fresh instance if you do not need diagnostics.
    DivKGradMGPressurePoissonSolver(
        const std::vector< std::shared_ptr< grid::shell::DistributedDomain > >&        domains,
        const std::vector< grid::Grid3DDataVec< ScalarType, 3 > >&                     coords_shell,
        const std::vector< grid::Grid2DDataScalar< ScalarType > >&                     coords_radii,
        const std::vector< grid::Grid4DDataScalar< grid::NodeOwnershipFlag > >&        ownership_mask,
        const std::vector< grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > >& boundary_mask,
        int                                                                            pressure_level,
        int                                                                            chebyshev_order   = 2,
        int                                                                            prepost_smoothing = 3,
        int                                                                            num_vcycles       = 1,
        ScalarType                                                                     relative_tol      = 1e-6,
        std::shared_ptr< util::Table >                                                 table             = nullptr )
    : pressure_level_( pressure_level )
    , num_levels_( pressure_level + 1 )
    , table_( table ? table : std::make_shared< util::Table >() )
    {
        if ( static_cast< int >( domains.size() ) < num_levels_ )
            throw std::runtime_error( "DivKGradMGPressurePoissonSolver: domains.size() must be >= pressure_level+1" );

        // Copy the inputs we need to retain as references for the operators.
        domains_.reserve( num_levels_ );
        for ( int level = 0; level < num_levels_; ++level )
            domains_.push_back( domains[level] );
        coords_shell_.assign( coords_shell.begin(), coords_shell.begin() + num_levels_ );
        coords_radii_.assign( coords_radii.begin(), coords_radii.begin() + num_levels_ );
        ownership_mask_.assign( ownership_mask.begin(), ownership_mask.begin() + num_levels_ );
        boundary_mask_.assign( boundary_mask.begin(), boundary_mask.begin() + num_levels_ );

        // num_dofs at finest level (for null-space projection at solve time).
        num_dofs_finest_ = kernels::common::count_masked< long >(
            ownership_mask_[pressure_level_], grid::NodeOwnershipFlag::OWNED );

        allocate_per_level_state_();
        build_operators_();
        build_mg_();

        chebyshev_order_   = chebyshev_order;
        prepost_smoothing_ = prepost_smoothing;
        num_vcycles_       = num_vcycles;
        relative_tol_      = relative_tol;
    }

    /// Apply K_w^-1 via the multigrid V-cycle, with constant-mode projection
    /// on rhs and solution.
    void solve( const PressureVector& rhs, PressureVector& sol ) override
    {
        // 1) Make a projected copy of the rhs (subtract its mean to remove the
        //    constant component, which is in the operator's nullspace under
        //    free-slip-everywhere Stokes BCs).
        linalg::assign( rhs_proj_, rhs );
        project_out_constant_( rhs_proj_ );

        // 2) Zero initial guess.  TODO: try warm-starting from the previous
        //    solve to reduce inner V-cycle iterations.
        linalg::assign( sol, ScalarType( 0 ) );

        // 3) MG solve.
        linalg::solvers::solve( *mg_, A_fine_(), sol, rhs_proj_ );

        // 4) Project the constant out of the solution.
        project_out_constant_( sol );
    }

    /// Refresh the K_w coefficient k = 1/sqrt(eta) from a new finest-level
    /// `sqrt_eta_pressure_finest`.  Coarser levels are populated by
    /// restriction.  Operator instances are not rebuilt — the underlying
    /// Kokkos view is updated in place, so subsequent `solve()` calls see
    /// the new coefficient automatically.
    void refresh( const PressureVector& sqrt_eta_pressure_finest ) override
    {
        // 1) Copy fine sqrt(eta) into the per-level slot.
        linalg::assign( sqrt_eta_[pressure_level_], sqrt_eta_pressure_finest );

        // 2) Restrict to coarser levels.  R_[L] restricts from level L+1 to L.
        for ( int level = pressure_level_ - 1; level >= 0; --level )
        {
            // R_[level] writes into level's grid from level+1.
            linalg::apply( R_[level], sqrt_eta_[level + 1], sqrt_eta_[level] );
        }

        // 3) Compute k = 1/sqrt(eta) per level, into inv_sqrt_eta_ which the
        //    DivKGrad ops reference.
        for ( int level = 0; level < num_levels_; ++level )
        {
            linalg::assign( inv_sqrt_eta_[level], sqrt_eta_[level] );
            linalg::invert_entries( inv_sqrt_eta_[level] );
        }

        // 4) Recompute inverse diagonals and rebuild smoothers.  Fresh
        //    smoothers have need_max_ev_estimation_=true so they re-estimate
        //    the spectrum against the updated coefficient on next solve.
        recompute_inverse_diagonals_();
        reconstruct_smoothers_();

        // 5) Rebuild the MG.  Multigrid stores smoothers by value
        //    (multigrid.hpp:78), so reconstructing our local `smoothers_`
        //    alone never reaches the MG's internal copies — without this
        //    rebuild, every K_w solve after the first would keep the stale
        //    spectrum estimate baked in at construction time.
        rebuild_mg_();
    }

  private:
    // -----------------------------------------------------------------------
    // One-time allocation.
    // -----------------------------------------------------------------------

    void allocate_per_level_state_()
    {
        sqrt_eta_.reserve( num_levels_ );
        inv_sqrt_eta_.reserve( num_levels_ );
        tmp_.reserve( num_levels_ );
        smoother_tmp_a_.reserve( num_levels_ );
        smoother_tmp_b_.reserve( num_levels_ );
        inverse_diagonals_.reserve( num_levels_ );
        for ( int level = 0; level < num_levels_; ++level )
        {
            sqrt_eta_.emplace_back(
                "wbfbt_sqrt_eta_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
            inv_sqrt_eta_.emplace_back(
                "wbfbt_inv_sqrt_eta_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
            tmp_.emplace_back(
                "wbfbt_tmp_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
            smoother_tmp_a_.emplace_back(
                "wbfbt_smoother_tmp_a_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
            smoother_tmp_b_.emplace_back(
                "wbfbt_smoother_tmp_b_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
            inverse_diagonals_.emplace_back(
                "wbfbt_inv_diag_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
        }
        rhs_proj_ = PressureVector(
            "wbfbt_rhs_proj", *domains_[pressure_level_], ownership_mask_[pressure_level_] );

        // Coarse-grid PCG temporaries (4 vectors, on the coarsest level).
        coarse_grid_tmps_.reserve( 4 );
        for ( int i = 0; i < 4; ++i )
        {
            coarse_grid_tmps_.emplace_back(
                "wbfbt_coarse_tmp_" + std::to_string( i ),
                *domains_[0], ownership_mask_[0] );
        }

        tmp_r_c_.reserve( num_levels_ - 1 );
        tmp_e_c_.reserve( num_levels_ - 1 );
        for ( int level = 0; level < num_levels_ - 1; ++level )
        {
            tmp_r_c_.emplace_back(
                "wbfbt_tmp_r_c_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
            tmp_e_c_.emplace_back(
                "wbfbt_tmp_e_c_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
        }
    }

    void build_operators_()
    {
        // Initialise inv_sqrt_eta_ to 1 so the operators are well-defined
        // before refresh() is called.
        for ( int level = 0; level < num_levels_; ++level )
            linalg::assign( inv_sqrt_eta_[level], ScalarType( 1 ) );

        // DivKGrad operators: Neumann (treat_boundary=false), apply (take_diagonal=false).
        // The view inv_sqrt_eta_[level].grid_data() is captured; in-place updates of
        // the underlying data are seen by the operator without rebuild.
        A_c_.reserve( num_levels_ - 1 );
        for ( int level = 0; level < num_levels_ - 1; ++level )
        {
            A_c_.emplace_back(
                *domains_[level],
                coords_shell_[level],
                coords_radii_[level],
                boundary_mask_[level],
                inv_sqrt_eta_[level].grid_data(),
                /* treat_boundary= */ false,
                /* take_diagonal= */ false );
        }
        A_fine_op_ = std::make_unique< DivKGrad >(
            *domains_[pressure_level_],
            coords_shell_[pressure_level_],
            coords_radii_[pressure_level_],
            boundary_mask_[pressure_level_],
            inv_sqrt_eta_[pressure_level_].grid_data(),
            /* treat_boundary= */ false,
            /* take_diagonal= */ false );

        // Diagonal-only views for cheap inverse-diagonal extraction.
        A_c_diag_.reserve( num_levels_ - 1 );
        for ( int level = 0; level < num_levels_ - 1; ++level )
        {
            A_c_diag_.emplace_back(
                *domains_[level],
                coords_shell_[level],
                coords_radii_[level],
                boundary_mask_[level],
                inv_sqrt_eta_[level].grid_data(),
                /* treat_boundary= */ false,
                /* take_diagonal= */ true );
        }
        A_fine_diag_ = std::make_unique< DivKGrad >(
            *domains_[pressure_level_],
            coords_shell_[pressure_level_],
            coords_radii_[pressure_level_],
            boundary_mask_[pressure_level_],
            inv_sqrt_eta_[pressure_level_].grid_data(),
            /* treat_boundary= */ false,
            /* take_diagonal= */ true );

        // Prolongation (additive: y += P x) / Restriction.
        P_.reserve( num_levels_ - 1 );
        R_.reserve( num_levels_ - 1 );
        for ( int level = 0; level < num_levels_ - 1; ++level )
        {
            P_.emplace_back( linalg::OperatorApplyMode::Add );
            R_.emplace_back( *domains_[level] );
        }
    }

    void build_mg_()
    {
        recompute_inverse_diagonals_();
        reconstruct_smoothers_();

        // Coarse-grid PCG.  Built once; PCG has no cached spectral state, so
        // coefficient changes are absorbed automatically through the operator
        // view it captured at construction time.
        linalg::solvers::IterativeSolverParameters coarse_params{ 200, 1e-7, 1e-12 };
        coarse_grid_solver_ = std::make_unique< CoarseGridSolver >(
            coarse_params, table_, coarse_grid_tmps_ );

        rebuild_mg_();
    }

    void rebuild_mg_()
    {
        mg_ = std::make_unique< MG >(
            P_, R_, A_c_, tmp_r_c_, tmp_e_c_, tmp_,
            smoothers_, smoothers_,
            *coarse_grid_solver_,
            num_vcycles_,
            relative_tol_ );
    }

    void recompute_inverse_diagonals_()
    {
        PressureVector one(
            "wbfbt_one", *domains_[pressure_level_], ownership_mask_[pressure_level_] );
        for ( int level = 0; level < num_levels_; ++level )
        {
            PressureVector ones_lvl(
                "wbfbt_ones_lvl_" + std::to_string( level ),
                *domains_[level], ownership_mask_[level] );
            linalg::assign( ones_lvl, ScalarType( 1 ) );
            if ( level < num_levels_ - 1 )
                linalg::apply( A_c_diag_[level], ones_lvl, inverse_diagonals_[level] );
            else
                linalg::apply( *A_fine_diag_, ones_lvl, inverse_diagonals_[level] );
            linalg::invert_entries( inverse_diagonals_[level] );
        }
    }

    void reconstruct_smoothers_()
    {
        smoothers_.clear();
        smoothers_.reserve( num_levels_ );
        for ( int level = 0; level < num_levels_; ++level )
        {
            std::vector< PressureVector > smoother_tmps{ smoother_tmp_a_[level], smoother_tmp_b_[level] };
            smoothers_.emplace_back(
                chebyshev_order_,
                inverse_diagonals_[level],
                smoother_tmps,
                prepost_smoothing_,
                /* max_ev_power_iterations= */ 50 );
        }
    }

    DivKGrad& A_fine_() { return *A_fine_op_; }

    void project_out_constant_( PressureVector& v )
    {
        const ScalarType s = kernels::common::masked_sum(
            v.grid_data(), v.mask_data(), grid::NodeOwnershipFlag::OWNED );
        const ScalarType mean = s / static_cast< ScalarType >( num_dofs_finest_ );
        linalg::lincomb( v, { ScalarType( 1 ) }, { v }, -mean );
    }

    // -----------------------------------------------------------------------
    // Members (order matters: vectors holding views referenced by operators
    // must outlive those operators on destruction).
    // -----------------------------------------------------------------------

    int                                                                          pressure_level_{};
    int                                                                          num_levels_{};
    long                                                                         num_dofs_finest_{ 0 };
    int                                                                          chebyshev_order_{ 2 };
    int                                                                          prepost_smoothing_{ 3 };
    int                                                                          num_vcycles_{ 1 };
    ScalarType                                                                   relative_tol_{ 1e-6 };

    std::vector< std::shared_ptr< grid::shell::DistributedDomain > >             domains_;
    std::vector< grid::Grid3DDataVec< ScalarType, 3 > >                          coords_shell_;
    std::vector< grid::Grid2DDataScalar< ScalarType > >                          coords_radii_;
    std::vector< grid::Grid4DDataScalar< grid::NodeOwnershipFlag > >             ownership_mask_;
    std::vector< grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > >      boundary_mask_;

    // Per-level scalar fields.  inv_sqrt_eta_'s underlying view is referenced
    // by every DivKGrad operator at its level, so updating it in place is
    // enough to propagate new viscosity into the V-cycle.
    std::vector< PressureVector > sqrt_eta_;
    std::vector< PressureVector > inv_sqrt_eta_;
    std::vector< PressureVector > tmp_;
    std::vector< PressureVector > smoother_tmp_a_;
    std::vector< PressureVector > smoother_tmp_b_;
    std::vector< PressureVector > inverse_diagonals_;
    std::vector< PressureVector > tmp_r_c_;
    std::vector< PressureVector > tmp_e_c_;
    std::vector< PressureVector > coarse_grid_tmps_;
    PressureVector                rhs_proj_;

    // Operators: fine on top, A_c_ for the coarse levels (indices 0..num_levels-2).
    std::unique_ptr< DivKGrad >   A_fine_op_;
    std::vector< DivKGrad >       A_c_;
    std::unique_ptr< DivKGrad >   A_fine_diag_;
    std::vector< DivKGrad >       A_c_diag_;
    std::vector< Prolongation >   P_;
    std::vector< Restriction >    R_;

    std::vector< Smoother >       smoothers_;

    std::shared_ptr< util::Table > table_;
    std::unique_ptr< CoarseGridSolver > coarse_grid_solver_;
    std::unique_ptr< MG >           mg_;
};

} // namespace terra::mantlecirculation
