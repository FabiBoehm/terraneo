#pragma once
//
// Dynamic remeshing for time-dependent mantle convection on the block-octree AMR.
//
// Every `remesh_every` timesteps we rebuild the adaptive forest from a fresh error estimator on the current
// temperature, load-balance the new forest across ranks, transfer the state (T and u) from the old mesh onto
// the new one, and re-setup the operators / MG. The old and new meshes are built at the SAME intra-block LDR
// resolution but on DIFFERENT forests (some blocks subdivided, some coarsened, some unchanged) -- so the
// transfer is a FOREST-TO-FOREST remap, distinct from the MG prolongation/restriction (which change LDR on a
// fixed forest).
//
// TRANSFER STRATEGY (this revision): geometric prolong/restrict/copy applied IDENTICALLY to every field
// (T and u), selected per changed block by the old<->new subdivision relation:
//   * unchanged block         -> copy nodal data
//   * new finer  (subdivided) -> PROLONG (Q1 interpolation of the old coarse block onto the new children)
//   * new coarser (coarsened) -> RESTRICT/inject (Q1 coarse nodes are a subset of the fine nodes)
// The per-field strategy is chosen through a TransferPolicy so a conservative mass-matrix L2 projection (or any
// other mapping) can be plugged in later WITHOUT touching the orchestration -- see FieldRemapper::remap.
//
// Design principles carried over from the single-GPU AMR investigation (branch `amr`):
//   * INTERFACE-PRESERVING refinement: dilate the refine set by a MARGIN so the 2:1 hanging interface lands in
//     smooth/flat-coefficient regions (hanging on a coefficient/solution gradient is the MG iteration penalty --
//     a known, expected effect of coefficient-blind geometric transfer, not a bug).
//   * GLOBAL refinement decision: the per-block indicator is all-reduced so every rank agrees on the new forest
//     (a replicated-forest invariant; local decisions diverge the forest at np>1 and blow up the solve).
//   * ON-DEVICE reconciliation: reuse apply_exchange_device / apply_constraint_device (no host round-trips).
//
#include "terra/grid/shell/adaptive_distribute.hpp"
#include "terra/grid/shell/adaptive_forest.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

#include <map>
#include <vector>

namespace terra::grid::shell::amr {

// =========================================================================================================
//  Options + carried state
// =========================================================================================================
struct RemeshOptions
{
    int    remesh_every = 20;    // remesh every N timesteps (0 = never)
    int    min_subdiv   = 0;     // coarsest block subdivision kept anywhere
    int    max_subdiv   = 8;     // finest block subdivision allowed (the M cap of AdaptiveForest)
    double refine_frac  = 0.30;  // refine leaf i if indicator[i] >= refine_frac * max_indicator
    double coarsen_frac = 0.05;  // coarsen leaf i if indicator[i] <  coarsen_frac * max_indicator
    int    margin_rings = 2;     // INTERFACE-PRESERVING: grow the refine set by this many block layers
    double imbalance_tol = 0.05; // rebalance only if load imbalance exceeds this fraction
};

// Everything the time loop carries across a remesh. Fields live on `mesh`; after remesh() they live on the new
// mesh with identical physical meaning. Operators / MG / constraint tables are rebuilt by the caller-supplied
// RebuildOperators callback (they are not part of this struct).
template < typename ScalarT >
struct AdaptiveState
{
    AdaptiveForest                    forest; // the block-octree (replicated structure)
    DistributedAdaptiveMesh           mesh;   // owned-only distributed instance of `forest`
    Grid3DDataVec< ScalarT, 3 >       coords; // unit-sphere node coords for `mesh`
    Grid2DDataScalar< ScalarT >       radii;
    linalg::VectorQ1Scalar< ScalarT > T;      // temperature (the advected STATE)
    linalg::VectorQ1Vec< ScalarT, 3 > u;      // velocity (recomputed each step; warm-start only)
};

inline bool should_remesh( int step, const RemeshOptions& o )
{
    return o.remesh_every > 0 && step > 0 && ( step % o.remesh_every == 0 );
}

// =========================================================================================================
//  Transfer abstraction  (the extensibility point)
// =========================================================================================================
//
// How a single field crosses old_mesh -> new_mesh. `Geometric` is the only kind implemented now (prolong /
// restrict / copy, coefficient-blind, warm-start quality). Future kinds (conservative mass-matrix L2 for T,
// etc.) add an enum value and a branch in FieldRemapper::remap -- the plan and orchestration stay unchanged.
enum class TransferKind
{
    Geometric,       // Q1 prolong (refine) / inject (coarsen) / copy (same) -- implemented
    ConservativeL2,  // mass-matrix L2 projection (M_new x = P_cross^T M_old x_old) -- TODO, heat-conserving
};

struct TransferPolicy
{
    TransferKind kind = TransferKind::Geometric;
    // future knobs: lumped-vs-consistent mass for ConservativeL2, etc.
};

// Per-new-leaf relation to the old forest, classified once per remesh and reused for every field.
enum class BlockRelation
{
    Same,     // old block at identical subdivision covers this new block  -> copy
    Refined,  // one old (coarser) block covers this new block             -> prolong
    Coarsened // several old (finer) blocks tile this new block            -> restrict / inject
};

struct BlockMapEntry
{
    int                new_local = -1;     // local subdomain index in the NEW mesh
    BlockRelation      relation  = BlockRelation::Same;
    int                old_local = -1;     // covering old block (Same/Refined); one representative for Coarsened
    std::vector< int > old_children;       // covering old blocks (Coarsened): the finer tiles
    // np>1: the owning ranks of old_local / old_children live here too (populated by the comm-plan step).
};

// The remap plan: the block-to-block map plus (np>1) the cross-rank communication lists. Computed ONCE per
// remesh from the two forests, then applied to T and u (and any other field) via FieldRemapper::remap.
struct RemeshPlan
{
    std::vector< BlockMapEntry > blocks; // one per new-mesh local subdomain
    // TODO(np>1): send/recv descriptors (which old block array goes to which rank's new block).
};

// =========================================================================================================
//  (1) ERROR ESTIMATOR  -- per-block scalar indicator on the current field.
// =========================================================================================================
//
// For advection-dominated mantle flow the features are thermal plumes / boundary layers -- where T has large
// gradient/curvature. A gradient-recovery (ZZ) estimator on T is a good oracle-free choice; it is MASS-WEIGHTED
// (volume-consistent) so it is comparable across subdivision levels. (An explicit residual estimator is
// viscosity-weighted and over-peaks on high-mu cores -- avoid it, per the investigation.)
template < typename ScalarT >
std::vector< double > compute_indicator( const AdaptiveState< ScalarT >& s );

// =========================================================================================================
//  (2) REBUILD FOREST  -- GLOBAL decision + interface-preserving margin + 2:1 balance.
// =========================================================================================================
// `cur_mesh` supplies the mapping local-block-index -> finest anchor so the per-local indicator can be
// scattered to the global leaf order and all-reduced (the replicated-forest invariant).
AdaptiveForest rebuild_forest( const AdaptiveForest& cur, const DistributedAdaptiveMesh& cur_mesh,
                               const std::vector< double >& ind_local, MPI_Comm comm, const RemeshOptions& o );

// =========================================================================================================
//  (3) LOAD BALANCE  -- weighted rank assignment (SFC).  Passthrough for now.
// =========================================================================================================
inline SubdomainToRankDistributionFunction load_balance( const AdaptiveForest& /*forest*/, MPI_Comm /*comm*/,
                                                         const RemeshOptions& /*o*/ )
{
    terra::util::Timer _t( "remesh_load_balance" );
    // TODO(np>1): weight(leaf) = nodes-per-block at leaf.subdivision; sort leaves along a space-filling curve
    //   over their octree keys; prefix-sum weights into nprocs equal-weight segments; accept only if it beats
    //   imbalance_tol. For now reuse the existing static diamond partition (correct at np=1, decent at np>1).
    return grid::shell::subdomain_to_rank_by_diamond;
}

// =========================================================================================================
//  (4) FIELD TRANSFER  old_mesh -> new_mesh.
// =========================================================================================================
//
// Two steps, separated so the (forest-only) plan is built once and reused for every field:
//   plan_remesh(old_mesh, new_mesh)              -> RemeshPlan   (block map + comm lists)
//   FieldRemapper(old_mesh, new_mesh, plan).remap(src, dst, policy)   for T, then u, ...
RemeshPlan plan_remesh( const DistributedAdaptiveMesh& old_mesh, const DistributedAdaptiveMesh& new_mesh );

template < typename ScalarT >
class FieldRemapper
{
  public:
    FieldRemapper( const DistributedAdaptiveMesh& old_mesh, const DistributedAdaptiveMesh& new_mesh,
                   const RemeshPlan& plan )
    : old_( &old_mesh ), new_( &new_mesh ), plan_( &plan )
    {}

    // Remap a scalar field (e.g. temperature).
    void remap( const linalg::VectorQ1Scalar< ScalarT >& src, linalg::VectorQ1Scalar< ScalarT >& dst,
                const TransferPolicy& policy );

    // Remap a vector field (e.g. velocity) component-by-component.
    template < int C >
    void remap( const linalg::VectorQ1Vec< ScalarT, C >& src, linalg::VectorQ1Vec< ScalarT, C >& dst,
                const TransferPolicy& policy );

  private:
    // The field-independent workhorse: fill one destination component view from one source component view,
    // using the block map. All fields share this; policy.kind selects the per-block rule.
    void remap_component( const Grid4DDataScalar< ScalarT >& src_view,
                          const Grid4DDataScalar< ScalarT >& dst_view, const TransferPolicy& policy );

    const DistributedAdaptiveMesh* old_;
    const DistributedAdaptiveMesh* new_;
    const RemeshPlan*              plan_;
};

// =========================================================================================================
//  (5) ORCHESTRATION.  Call once every remesh_every steps, right after the temperature update.
// =========================================================================================================
template < typename ScalarT, typename RebuildOperators >
void remesh( AdaptiveState< ScalarT >& s, const RemeshOptions& o, MPI_Comm comm, int ldr, int s_rad,
             const std::vector< ScalarT >& radii_nodes, RebuildOperators rebuild_ops )
{
    terra::util::Timer _t( "remesh" );

    const auto ind      = compute_indicator( s );                                   // (1) where is the error now?
    auto       nforest  = rebuild_forest( s.forest, s.mesh, ind, comm, o );          // (2) new octree
    auto       new_part = load_balance( nforest, comm, o );                          // (3) rank assignment

    // (4) build the new distributed mesh + coords/radii, then transfer the state onto it.
    DistributedAdaptiveMesh nmesh   = build_distributed_adaptive_mesh( comm, ldr, radii_nodes, nforest, new_part );
    auto                    ncoords = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarT >( nmesh.domain );
    auto                    nradii  = grid::shell::subdomain_shell_radii< ScalarT >( nmesh.domain );
    auto                    nmask   = adaptive_ownership_mask( nmesh );

    linalg::VectorQ1Scalar< ScalarT > T_new( "T", nmesh.domain, nmask );
    linalg::VectorQ1Vec< ScalarT, 3 > u_new( "u", nmesh.domain, nmask );

    const RemeshPlan         plan = plan_remesh( s.mesh, nmesh );
    FieldRemapper< ScalarT > remap( s.mesh, nmesh, plan );
    remap.remap( s.T, T_new, TransferPolicy{ TransferKind::Geometric } );            // all vars: geometric
    remap.template remap< 3 >( s.u, u_new, TransferPolicy{ TransferKind::Geometric } );

    // (5) swap in the new mesh + state, then rebuild all mesh-dependent operators / MG / constraint tables.
    s.forest = std::move( nforest );
    s.mesh   = std::move( nmesh );
    s.coords = std::move( ncoords );
    s.radii  = std::move( nradii );
    s.T      = std::move( T_new );
    s.u      = std::move( u_new );
    rebuild_ops( s );
}

// =========================================================================================================
//  TIME-LOOP INTEGRATION (sketch, in the mantle-convection driver):
//
//    AdaptiveState<double> st = initial_adaptive_state(...);
//    auto rebuild_ops = [&]( AdaptiveState<double>& s ){ /* build Stokes, adv-diff, MG on s.mesh */ };
//    rebuild_ops( st );
//    for ( int step = 0; step < nsteps; ++step )
//    {
//        stokes_solve( st );          // u from buoyancy(T)   (warm-started by the transferred u)
//        advect_diffuse( st, dt );    // T^{n+1} from T^n, u
//        if ( should_remesh( step, opts ) )
//            remesh( st, opts, MPI_COMM_WORLD, LDR, S_rad, radii_nodes, rebuild_ops );
//    }
// =========================================================================================================

} // namespace terra::grid::shell::amr

#include "adaptive_remesh_impl.hpp"
