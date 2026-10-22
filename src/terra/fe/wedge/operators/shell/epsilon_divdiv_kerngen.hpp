#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "impl/Kokkos_Profiling.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/gca/local_matrix_storage.hpp"
#include "grid/bit_masks.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/floating_point.hpp"
#include "util/timer.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace terra::fe::wedge::operators::shell {

using grid::shell::get_boundary_condition_flag;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using terra::grid::shell::BoundaryConditionFlag;
using terra::grid::shell::BoundaryConditions;
using terra::grid::shell::ShellBoundaryFlag;
using terra::linalg::trafo::trafo_mat_cartesian_to_normal_tangential;

/**
 * @brief Matrix-free / matrix-based epsilon-div-div operator on wedge elements in a spherical shell.
 *
 * This class supports three execution paths:
 *
 * 1) Slow path (local matrix path)
 *    - Used if local matrices are stored (full or selective)
 *    - Reuses assembled/stored 18x18 local matrices
 *
 * 2) Fast path: Dirichlet/Neumann
 *    - Matrix-free
 *    - Handles Dirichlet by skipping constrained face-node couplings (and diagonal treatment if requested)
 *    - Neumann is naturally included
 *
 * 3) Fast path: Free-slip
 *    - Matrix-free
 *    - Applies trial and test-side tangential projection on free-slip boundaries
 *    - Preserves behavior consistent with the slow path for iterative solves
 *
 * IMPORTANT DESIGN CHANGE:
 * ------------------------
 * The path decision is made on the HOST and cached in `kernel_path_`.
 * `apply_impl()` dispatches to a different kernel launch per path.
 * This avoids a runtime branch inside the hot device kernel.
 */
// CoordScalarT is the precision of the STORED node coordinates and radii. It
// defaults to ScalarT, so existing instantiations are unchanged. Pass double to
// decouple it: the Jacobian is formed by differencing coordinates, which loses the
// geometry entirely in 16-bit and is already marginal in single. ScalarT governs
// the stored fields src/dst and the coefficient -- where the memory traffic is.
// ArithScalarT is the precision of the ARITHMETIC in the fast Dirichlet/Neumann
// kernel: the team-scratch field tiles and the local contraction. It defaults to
// double, which is exactly what this kernel did unconditionally before, so every
// existing instantiation is unchanged. Pass ScalarT to run the contraction at the
// field precision -- the geometry (coordinates, Jacobian, reference gradients) stays
// in CoordScalarT either way.
// Occupancy target for the fast Dirichlet/Neumann and free-slip team policies. The
// maximum team size caps the tile shape (lat_tile^2 * r_tile) and the minimum waves
// per execution unit caps the register budget: at 5 waves/SIMD the compiler gets
// 48 VGPRs + 48 AGPRs on gfx90a and spills, at 4 it gets 64 + 64. Both are
// overridable from the build so an occupancy/spill trade can be measured without
// editing this header.
#ifndef TERRA_EPSDIVDIV_MAX_TEAM
#define TERRA_EPSDIVDIV_MAX_TEAM 128
#endif
#ifndef TERRA_EPSDIVDIV_MIN_WAVES
#define TERRA_EPSDIVDIV_MIN_WAVES 5
#endif
/// Shared memory per SM the launch bounds' minimum blocks must fit in (bytes): H100 228 KiB.
#ifndef TERRA_EPSDIVDIV_SM_SHMEM
#define TERRA_EPSDIVDIV_SM_SHMEM ( 228 * 1024 )
#endif

template < typename ScalarT,
           int      VecDim        = 3,
           typename CoordScalarT  = ScalarT,
           typename ArithScalarT  = double,
           typename MatrixScalarT = ScalarT >
class EpsilonDivDivKerngen
{
  public:
    using SrcVectorType                 = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType                 = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType                    = ScalarT;
    /// Precision of geometry: stored coordinates, radii, quadrature, Jacobians and
    /// everything derived from them. Decoupled from ScalarType so a reduced field
    /// precision never reaches the coordinate differencing in the Jacobian.
    using GeomT                         = CoordScalarT;
    /// Precision of the fast-path scratch field tiles and the local contraction.
    /// Independent of GeomT so bandwidth and arithmetic can be varied separately.
    using ArithT                        = ArithScalarT;
    /// Precision the stored local matrices are kept in (stored-local-matrix path only).
    using MatT                          = MatrixScalarT;
    /// Precision of the Jacobian arithmetic: assembly, determinant, inverse and the
    /// reference-to-physical gradient transform. GeomT, widened to at least float:
    /// a 16-bit Jacobian would put the coordinate differencing at three decimal digits.
    using GeoT = std::conditional_t< ( sizeof( GeomT ) >= 4 ), GeomT, float >;
    /// Precision the tile-local scatter accumulates in. Never 16-bit: a 2-byte
    /// atomic_add has no lock-free implementation and falls back to a warp spin lock,
    /// which measured as the single largest cost of a reduced field precision.
    ///
    /// float rather than double for the 16-bit arms. A node takes on the order of
    /// twenty contributions carrying at most an 11-bit mantissa, and the sum is
    /// narrowed back to ScalarType on the way out, so float holds it comfortably;
    /// double would only double the scratch these arms carry, and with it cost
    /// occupancy -- which is the whole reason the buffer is off for wide storage.
    using AccT = std::conditional_t< ( sizeof( ArithT ) >= 4 ), ArithT, float >;
    /// Precision of the per-cell contraction (shape-function gradients times field
    /// values, the quadrature weight, the hourglass correction). For 16-bit ArithT it
    /// is GeoT: the gradient of the smoothed iterate scales like 1/h^2 and passes
    /// 65504 at level 7, and the weight k*|det J| ~ h^3 is an fp16 subnormal from
    /// level 6 on -- the fp16 V-cycle returned Inf/NaN there. Staging and the
    /// coefficient average stay in ArithT.
    using ConT = std::conditional_t< ( sizeof( ArithT ) >= 4 ), ArithT, GeoT >;
    /// Element type of the warp path's staged field tiles. 16-bit fields are widened
    /// once, at staging, instead of at every one of the ~3x more reads the cell loop
    /// makes (ncu: the 16-bit kernels ran the ALU pipe at 43 % vs 26 % for float, the
    /// half->float conversions and 16-bit extractions). The tile then has the float
    /// kernel's footprint, which the warp launch allows.
    /// Measured: widening at staging bought nothing (the reads were not the cost), so
    /// the tile stays in ArithT and its half-size footprint buys occupancy instead.
    using StageT = ArithT;
    /// Whether the scatter goes through the team's tile buffer instead of straight to
    /// global memory. With the buffer's accumulation a native shared-space reduction
    /// (tile_accumulate), H100 level 8 at the swept tile measured, buffer off -> on:
    /// double 10.16 -> 9.78, float 12.01 -> 12.17, bf16 2.49 -> 8.71, fp16 3.35 -> 8.36
    /// GDoF/s. The 16-bit arms need it (a 2-byte global atomic has no lock-free form),
    /// float gains a little, and double -- whose buffer costs twice the scratch, hence
    /// occupancy -- loses, so the default is on for 4-byte and narrower storage.
    /// Scatter modes of the fast Dirichlet/Neumann path: 0 straight to global memory,
    /// 1 through the team's tile buffer, 2 the warp-owner reduction (no shared-memory
    /// atomics at all; see path_fast_dn_warp.hpp). Mode 2 needs an 8-wide lateral
    /// tile with r_tile = 1 and a compiled level count; set_tile_scatter() retiles.
    /// The warp-owner scatter needs warp shuffles and native global reductions.
    static constexpr bool warp_scatter_backend =
#if defined( KOKKOS_ENABLE_CUDA ) || defined( KOKKOS_ENABLE_HIP )
        true;
#else
        false;
#endif
    /// Fast DN scatter: 2 = warp-owner (4-byte and 16-bit storage on a GPU backend),
    /// 1 = red.shared tile buffer, 0 = direct atomics (double: its atomics are cheap
    /// relative to its bandwidth and the buffer would halve the occupancy).
    static constexpr int default_tile_scatter = ( sizeof( ScalarType ) <= 4 ) ? ( warp_scatter_backend ? 2 : 1 ) : 0;
    static constexpr int LocalMatrixDim = 18;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 >;
    using LocalMatrixStorage      = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >;
    using Team                    = Kokkos::TeamPolicy<>::member_type;

  private:
    // Optional element-local matrix storage (GCA/coarsening/explicit local-matrix mode)
    LocalMatrixStorage local_matrix_storage_;

    // Domain and geometry / coefficients
    grid::shell::DistributedDomain                           domain_;
    grid::Grid3DDataVec< CoordScalarT, 3 >                        grid_;   ///< Lateral shell geometry (unit sphere coords)
    grid::Grid2DDataScalar< CoordScalarT >                        radii_;  ///< Radial coordinates per local subdomain
    grid::Grid4DDataScalar< ScalarType >                     k_;      ///< Scalar coefficient field
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;   ///< Boundary flags per cell/node
    BoundaryConditions                                       bcs_;    ///< CMB and SURFACE boundary conditions

    bool diagonal_; ///< If true, apply diagonal-only action
    int tile_scatter_ = default_tile_scatter; ///< Fast DN path scatter mode (see default_tile_scatter)

    /// Hourglass (rank-6 gamma) stabilization coefficient; 0 disables it entirely.
    /// One-point quadrature on the wedge leaves two zero-energy modes per element,
    /// so the assembled A-block has a near-null space that no amount of multigrid
    /// resolves -- the outer residual simply stops descending. Same coefficient and
    /// scaling as EpsilonDivDivSimpleStab.
    double stab_c_ = 0.0;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;
    linalg::OperatorStoredMatrixMode  operator_stored_matrix_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim >                    recv_buffers_;
    terra::communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataVec< ScalarType, VecDim > > comm_plan_;

    // Device views captured by kernels
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;
    grid::Grid4DDataVec< ScalarType, VecDim > src_;

    // Quadrature (Felippa 1x1 on wedge)
    const int                num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;
    dense::Vec< GeomT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
    GeomT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];

    // Local subdomain extents (in cells)
    int local_subdomains_;
    int hex_lat_;
    int hex_rad_;
    int lat_refinement_level_;

    // Team-policy tiling (one team handles a slab lat_tile x lat_tile x r_tile cells)
    /// Tile counts, team size and block count implied by the current tile shape.
    void recompute_tiling()
    {
        r_tile_block_ = r_tile_ * r_passes_;

        lat_tiles_ = ( hex_lat_ + lat_tile_ - 1 ) / lat_tile_;
        r_tiles_   = ( hex_rad_ + r_tile_block_ - 1 ) / r_tile_block_;

        team_size_ = lat_tile_ * lat_tile_ * r_tile_;
        blocks_    = local_subdomains_ * lat_tiles_ * lat_tiles_ * r_tiles_;
    }

    int lat_tile_;
    int r_tile_;
    int r_passes_;
    int r_tile_block_;
    int lat_tiles_;
    int r_tiles_;
    int team_size_;
    int blocks_;

    ScalarT r_max_;
    ScalarT r_min_;

    /**
     * @brief Kernel path selected on host.
     *
     * This value is computed whenever BCs or stored-matrix mode change and is then
     * used by `apply_impl()` to select which kernel launch to issue.
     */
    enum class KernelPath
    {
        Slow,
        FastDirichletNeumann,
        FastFreeslip,
        StoredLocalMatrices,
    };
    KernelPath kernel_path_ = KernelPath::FastDirichletNeumann;

    static const char* kernel_path_name( KernelPath p )
    {
        switch ( p )
        {
        case KernelPath::Slow:
            return "slow";
        case KernelPath::FastFreeslip:
            return "fast-freeslip";
        case KernelPath::StoredLocalMatrices:
            return "stored-local-matrices";
        default:
            return "fast-dirichlet-neumann";
        }
    }

    // ---- Stored-local-matrix path (epsilon_divdiv_kerngen/path_stored_local_matrices.hpp)
    /// Packed symmetric 18x18: the upper triangle, row-major from the diagonal.
    static constexpr int PackedLocalMatrixSize = LocalMatrixDim * ( LocalMatrixDim + 1 ) / 2;
    /// Entry-major, stored_entry( entry, cell ): 2 * PackedLocalMatrixSize entries per hex
    /// cell. One flat view, indexed in size_t by hand: at level 7 the matrices run past
    /// 2^32 entries, which is more than a rank-2 offset is guaranteed to carry.
    Kokkos::View< MatT* > stored_matrices_;
    int  num_cells_                = 0; ///< extent of the stored matrices' cell axis
    bool use_stored_matrices_      = false;
    bool stored_matrices_valid_    = false;
    bool stored_matrices_hourglass_ = false;

    // Rotation null-space penalty (active only for fs/fs)
    bool                                              penalty_active_  = false;
    ScalarT                                           penalty_epsilon_ = 1.0;
    grid::Grid4DDataVec< ScalarType, VecDim >         null_modes_[3];
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > ownership_mask_;

  private:
    /**
     * @brief Recompute the kernel path on host.
     *
     * Rules:
     * - Any stored local matrix mode => slow path
     * - Else if any free-slip BC on CMB or SURFACE => fast free-slip path
     * - Else => fast Dirichlet/Neumann path
     */
    /// The path that recomputes the geometry on every apply: what the stored-local-
    /// matrix path stands in for, and what it falls back to for the diagonal.
    KernelPath matrix_free_kernel_path() const
    {
        // Serial backend: always use slow path (fast paths require shared memory / larger teams).
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
            return KernelPath::Slow;

        const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
        const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );

        const bool has_freeslip = ( cmb_bc == FREESLIP ) || ( surface_bc == FREESLIP );
        return has_freeslip ? KernelPath::FastFreeslip : KernelPath::FastDirichletNeumann;
    }

    void update_kernel_path_flag_host_only()
    {
        const bool has_stored_matrices = ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off );

        if ( has_stored_matrices )
            kernel_path_ = KernelPath::Slow;
        else if ( use_stored_matrices_ )
            kernel_path_ = KernelPath::StoredLocalMatrices;
        else
            kernel_path_ = matrix_free_kernel_path();
    }


  public:
    EpsilonDivDivKerngen(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< CoordScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< CoordScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::Grid4DDataScalar< ScalarT >&                        k,
        BoundaryConditions                                              bcs,
        bool                                                            diagonal,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode operator_stored_matrix_mode = linalg::OperatorStoredMatrixMode::Off )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , k_( k )
    , mask_( mask )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , operator_stored_matrix_mode_( operator_stored_matrix_mode )
    , recv_buffers_( domain )
    , comm_plan_( domain )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];

        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );

        const grid::shell::DomainInfo& domain_info = domain_.domain_info();
        local_subdomains_                          = domain_.subdomains().size();
        hex_lat_                                   = domain_info.subdomain_num_nodes_per_side_laterally() - 1;
        hex_rad_                                   = domain_info.subdomain_num_nodes_radially() - 1;
        lat_refinement_level_                      = domain_info.diamond_lateral_refinement_level();

        // On Serial backend, team_size must be 1 => use 1x1x1 tiles.
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
        {
            lat_tile_     = 1;
            r_tile_       = 1;
            r_passes_     = 1;
        }
        else
        {
            lat_tile_     = 4;
            r_tile_       = 8;
            r_passes_     = 2;
        }
        recompute_tiling();

        r_min_ = domain_info.radii()[0];
        r_max_ = domain_info.radii()[domain_info.radii().size() - 1];

        // Host-side path selection (no in-kernel path branching)
        update_kernel_path_flag_host_only();

        // The default scatter mode decides the default tile (the warp-owner path has
        // its own shapes); applied here, once the kernel path is known.
        set_tile_scatter( tile_scatter_ );

        // Measurement switch without a code change: TERRA_EPSDIVDIV_TILE_SCATTER=0|1|2.
        if ( const char* env = std::getenv( "TERRA_EPSDIVDIV_TILE_SCATTER" ) )
        {
            set_tile_scatter( std::atoi( env ) );
        }

        util::logroot << "[EpsilonDivDiv] tile size (x,y,r)=(" << lat_tile_ << "," << lat_tile_ << "," << r_tile_
                      << "), r_passes=" << r_passes_ << std::endl;
        util::logroot << "[EpsilonDivDiv] number of tiles (x,y,r)=(" << lat_tiles_ << "," << lat_tiles_ << ","
                      << r_tiles_ << "), team_size=" << team_size_ << ", blocks=" << blocks_ << std::endl;
        util::logroot << "[EpsilonDivDiv] kernel path = " << kernel_path_name( kernel_path_ ) << std::endl;

        init_penalty();
    }

    void set_penalty_epsilon( ScalarT eps ) { penalty_epsilon_ = eps; }

    /// @brief Override the team-tile shape at runtime, for tuning.
    ///
    /// Call before the first apply(). Two limits are checked here rather than left to
    /// fail at launch: the fast Dirichlet/Neumann path is dispatched with
    /// Kokkos::LaunchBounds< TERRA_EPSDIVDIV_MAX_TEAM, ... >, so a team of more than
    /// TERRA_EPSDIVDIV_MAX_TEAM threads cannot be launched at all, and the staging slab grows as
    /// ( r_tile * r_passes + 1 ) * ( lat_tile + 1 )^2, which has to fit in 64 KiB of
    /// shared memory. On rejection the previous tiling is restored and false returned,
    /// so a tuning sweep can simply skip the configuration.
    bool set_tiling( int lat_tile, int r_tile, int r_passes, bool verbose = true )
    {
        if ( lat_tile < 1 || r_tile < 1 || r_passes < 1 )
        {
            util::logroot << "[EpsilonDivDiv] set_tiling rejected: tiles must be >= 1" << std::endl;
            return false;
        }
        const int max_team = TERRA_EPSDIVDIV_MAX_TEAM;
        if ( lat_tile * lat_tile * r_tile > max_team )
        {
            util::logroot << "[EpsilonDivDiv] set_tiling rejected (" << lat_tile << "," << lat_tile << ","
                          << r_tile << "): team_size " << lat_tile * lat_tile * r_tile
                          << " exceeds the LaunchBounds of " << max_team << std::endl;
            return false;
        }

        const int saved_lat = lat_tile_, saved_r = r_tile_, saved_p = r_passes_;
        lat_tile_ = lat_tile;
        r_tile_   = r_tile;
        r_passes_ = r_passes;
        recompute_tiling();

        // The warp-owner path has its own launch bounds (teams of 64, eight per SM);
        // when it is the path that will run this tile, its limits are the ones that
        // count. In mode 2 a tile that is not a warp shape runs through the tile buffer.
        const bool warp_only = ( tile_scatter_ == 2 ) && warp_shape_ok() &&
                               ( matrix_free_kernel_path() == KernelPath::FastDirichletNeumann );

        size_t shmem     = 0;
        size_t max_shmem = 0;
        if ( warp_only )
        {
            max_shmem = warp_shmem_limit();
            shmem     = team_shmem_size_dn( team_size_ );
        }
        else
        {
            max_shmem = kokkos_shmem_limit( team_size_, TERRA_EPSDIVDIV_MIN_WAVES );
            shmem     = std::max( team_shmem_size( team_size_ ), team_shmem_size_dn( team_size_ ) );
        }
        if ( shmem > max_shmem )
        {
            util::logroot << "[EpsilonDivDiv] set_tiling rejected (" << lat_tile << "," << lat_tile << ","
                          << r_tile << "," << r_passes << "): " << shmem << " B of scratch exceeds "
                          << max_shmem / 1024 << " KiB" << std::endl;
            lat_tile_ = saved_lat;
            r_tile_   = saved_r;
            r_passes_ = saved_p;
            recompute_tiling();
            return false;
        }

        if ( verbose )
        {
            util::logroot << "[EpsilonDivDiv] retiled to (x,y,r)=(" << lat_tile_ << "," << lat_tile_ << ","
                          << r_tile_ << "), r_passes=" << r_passes_ << ", team_size=" << team_size_
                          << ", blocks=" << blocks_ << ", scratch=" << shmem << " B" << std::endl;
        }
        return true;
    }

    int team_size() const { return team_size_; }

    /**
     * @brief Initialize penalty for fs/fs null-space regularization.
     *
     * Builds 3 orthonormal rotation-mode vectors (ê_i × r) with free-slip
     * enforcement, then Gram-Schmidt orthonormalizes them.
     * Only called when both CMB and SURFACE have FREESLIP BCs.
     */
    void init_penalty()
    {
        const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
        const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );

        penalty_active_ = ( cmb_bc == FREESLIP ) && ( surface_bc == FREESLIP );
        if ( !penalty_active_ )
            return;

        util::logroot << "[EpsilonDivDiv] Initializing fs/fs null-space penalty (epsilon="
                      << penalty_epsilon_ << ")" << std::endl;

        ownership_mask_ = grid::setup_node_ownership_mask_data( domain_ );

        const int nsub = domain_.subdomains().size();
        const int nlat = domain_.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nrad = domain_.domain_info().subdomain_num_nodes_radially();

        auto grid_local  = grid_;
        auto radii_local = radii_;
        auto mask_local  = mask_;

        // Build 3 rotation modes: axis i => ê_i × r
        for ( int axis = 0; axis < 3; ++axis )
        {
            null_modes_[axis] = grid::Grid4DDataVec< ScalarType, VecDim >(
                "penalty_null_mode_" + std::to_string( axis ), nsub, nlat, nlat, nrad );

            auto nm = null_modes_[axis];

            Kokkos::parallel_for(
                "penalty_fill_mode_" + std::to_string( axis ),
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 }, { nsub, nlat, nlat, nrad } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    const auto c = grid::shell::coords( s, x, y, r, grid_local, radii_local );
                    // ê_axis × r: cyclic cross product
                    // axis=0: (0, z, -y), axis=1: (-z, 0, x), axis=2: (y, -x, 0)
                    const int a1 = ( axis + 1 ) % 3;
                    const int a2 = ( axis + 2 ) % 3;
                    for ( int d = 0; d < VecDim; ++d )
                        nm( s, x, y, r, d ) = 0.0;
                    // c is GeomT; the rotation modes are stored in the field precision.
                    nm( s, x, y, r, a1 ) = static_cast< ScalarType >( c( a2 ) );
                    nm( s, x, y, r, a2 ) = static_cast< ScalarType >( -c( a1 ) );
                } );
            Kokkos::fence();

            // Free-slip enforce: transform to n-t, zero normal, transform back
            // Do this for both CMB and SURFACE boundaries
            auto freeslip_boundary_mask =
                static_cast< grid::shell::ShellBoundaryFlag >(
                    static_cast< uint8_t >( CMB ) | static_cast< uint8_t >( SURFACE ) );

            Kokkos::parallel_for(
                "penalty_fs_enforce_" + std::to_string( axis ),
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 }, { nsub, nlat, nlat, nrad } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    if ( !util::has_flag( mask_local( s, x, y, r ), freeslip_boundary_mask ) )
                        return;

                    dense::Vec< ScalarType, 3 > v;
                    for ( int d = 0; d < 3; ++d )
                        v( d ) = nm( s, x, y, r, d );

                    dense::Vec< ScalarType, 3 > normal;
                    for ( int d = 0; d < 3; ++d )
                        normal( d ) = static_cast< ScalarType >( grid_local( s, x, y, d ) );

                    // R * v -> normal-tangential
                    const auto R  = trafo_mat_cartesian_to_normal_tangential( normal );
                    auto       nt = R * v;

                    // Zero normal component
                    nt( 0 ) = 0.0;

                    // R^T * nt -> back to Cartesian
                    const auto Rt   = R.transposed();
                    const auto cart = Rt * nt;

                    for ( int d = 0; d < 3; ++d )
                        nm( s, x, y, r, d ) = cart( d );
                } );
            Kokkos::fence();
        }

        // Gram-Schmidt orthonormalization
        for ( int i = 0; i < 3; ++i )
        {
            // Subtract projections of previous modes
            for ( int j = 0; j < i; ++j )
            {
                ScalarType dot_ij = kernels::common::masked_dot_product(
                    null_modes_[i], null_modes_[j], ownership_mask_, grid::NodeOwnershipFlag::OWNED );

                auto ni = null_modes_[i];
                auto nj = null_modes_[j];
                Kokkos::parallel_for(
                    "penalty_gs_subtract",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                        { 0, 0, 0, 0 }, { nsub, nlat, nlat, nrad } ),
                    KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                        for ( int d = 0; d < VecDim; ++d )
                            ni( s, x, y, r, d ) -= dot_ij * nj( s, x, y, r, d );
                    } );
                Kokkos::fence();
            }

            // Normalize
            ScalarType norm_sq = kernels::common::masked_dot_product(
                null_modes_[i], null_modes_[i], ownership_mask_, grid::NodeOwnershipFlag::OWNED );
            ScalarType inv_norm = 1.0 / Kokkos::sqrt( norm_sq );

            auto ni = null_modes_[i];
            Kokkos::parallel_for(
                "penalty_gs_normalize",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 }, { nsub, nlat, nlat, nrad } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    for ( int d = 0; d < VecDim; ++d )
                        ni( s, x, y, r, d ) *= inv_norm;
                } );
            Kokkos::fence();
        }

        util::logroot << "[EpsilonDivDiv] Null-space penalty initialized (3 modes)" << std::endl;
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief Set the hourglass stabilization coefficient (0 disables; 0.3 is the
    ///        value EpsilonDivDivSimpleStab uses).
    void set_hourglass_coefficient( double c )
    {
        stab_c_                = c;
        stored_matrices_valid_ = false;
    }

    /// Route the Dirichlet/Neumann scatter through the team's tile buffer (one global
    /// atomic per node) instead of straight to global memory (one per wedge-node).
    /// Defaults to on for 16-bit storage only; both variants are compiled, so this is
    /// a measurement switch as much as a setting. Call before the first apply().
    void set_tile_scatter( int mode )
    {
        tile_scatter_ = mode;
        if ( mode == 2 && !warp_scatter_available() )
        {
            util::logroot << "[EpsilonDivDiv] warp-owner scatter needs a CUDA or HIP backend; using the tile buffer"
                          << std::endl;
            tile_scatter_ = 1;
            return;
        }
        if ( mode == 2 && matrix_free_kernel_path() != KernelPath::FastDirichletNeumann )
        {
            // The warp-owner path exists for the fast DN kernel only; the other paths
            // keep the tile buffer and their default tile.
            tile_scatter_ = 1;
            if ( warp_shape_ok() )
                set_tiling( 4, 4, 8, false );
            return;
        }
        if ( mode == 2 )
        {
            // Re-validate the current tile under the warp path's limits (shape, scratch);
            // one that does not pass is replaced by 8/1/16, or 8/1/8 if that does not fit.
            const bool keep = warp_shape_ok() && set_tiling( lat_tile_, r_tile_, r_passes_, false );
            if ( !keep && ( hex_lat_ < 8 || ( !set_tiling( 8, 1, 16, false ) && !set_tiling( 8, 1, 8, false ) ) ) )
            {
                util::logroot << "[EpsilonDivDiv] warp-owner scatter: no usable tile at this size; using the tile buffer"
                              << std::endl;
                tile_scatter_ = 1;
            }
        }
    }
    int tile_scatter() const { return tile_scatter_; }

    static constexpr bool warp_scatter_available() { return warp_scatter_backend; }

    /// True when the next apply runs the warp-owner kernel. Its owner lanes store every
    /// node only their team touches, so in Replace mode only the tile perimeters (the
    /// atomics' targets) need zeroing; the perimeter fill below needs hex_lat_ to be a
    /// multiple of the tile.
    bool warp_path_runs() const
    {
        return !diagonal_ && kernel_path_ == KernelPath::FastDirichletNeumann && tile_scatter_ == 2 &&
               warp_shape_ok() && hex_lat_ % lat_tile_ == 0;
    }

    /// Zero the columns on the warp tiles' perimeters (x or y a multiple of lat_tile_):
    /// the only nodes the warp-owner kernel accumulates into (it stores the rest).
    /// Consecutive threads take consecutive levels, so the writes coalesce along r.
    void zero_warp_perimeter( grid::Grid4DDataVec< ScalarType, VecDim > dst )
    {
        const int lat     = lat_tile_;
        const int nx      = hex_lat_ + 1;       // columns per side
        const int nr      = hex_rad_ + 1;       // levels
        const int nmult   = hex_lat_ / lat + 1; // multiples of lat in [0, hex_lat]
        const int ncols_x = nmult * nx;         // x % lat == 0, any y
        const int nx_rest = nx - nmult;         // x % lat != 0
        const int ncols   = ncols_x + nmult * nx_rest; // + y % lat == 0, x % lat != 0
        Kokkos::parallel_for(
            "epsilon_divdiv_zero_warp_perimeter",
            Kokkos::RangePolicy<>( 0, size_t( local_subdomains_ ) * size_t( ncols ) * size_t( nr ) ),
            KOKKOS_LAMBDA( const size_t i ) {
                const int r   = int( i % size_t( nr ) );
                const int col = int( ( i / size_t( nr ) ) % size_t( ncols ) );
                const int s   = int( i / size_t( nr ) / size_t( ncols ) );
                int       xi, yi;
                if ( col < ncols_x )
                {
                    xi = ( col / nx ) * lat;
                    yi = col % nx;
                }
                else
                {
                    const int c  = col - ncols_x;
                    const int xr = c % nx_rest;
                    yi           = ( c / nx_rest ) * lat;
                    xi           = xr + 1 + xr / ( lat - 1 ); // skips the multiples of lat
                }
                for ( int d = 0; d < VecDim; ++d )
                    dst( s, xi, yi, r, d ) = ScalarType( 0 );
            } );
    }

    /// The warp-owner path is compiled for (lat, levels per team) = (8, 9) and (8, 17):
    /// tiles 8/1/8 and 8/1/16, one thread per column, teams of 64 (two warps). It also
    /// stages the level below the run (halo cell); a longer run would not fit the launch.
    bool warp_shape_ok() const
    {
        return lat_tile_ == 8 && r_tile_ == 1 && ( r_tile_block_ == 8 || r_tile_block_ == 16 );
    }

    /// Kokkos (5.1, CUDA) launches a team kernel only if its scratch fits one block
    /// (48 KiB, no opt-in) and LaunchBounds' minimum blocks per SM fit the SM's shared
    /// memory (TERRA_EPSDIVDIV_SM_SHMEM, 228 KiB on H100). A request past either throws
    /// at launch -- after Kokkos acquired one of its ten team-scratch slots, which the
    /// exception leaks, so the tenth failed launch hangs the process. set_tiling()
    /// therefore predicts the limit exactly (minus Kokkos' (team+2) doubles and a margin).
    static constexpr size_t kokkos_shmem_limit( int team, int min_blocks )
    {
        const size_t per_block = 48u * 1024u;
        const size_t per_sm    = size_t( TERRA_EPSDIVDIV_SM_SHMEM );
        const size_t cap       = per_sm / size_t( min_blocks ) < per_block ? per_sm / size_t( min_blocks ) : per_block;
        return cap - size_t( team + 2 ) * sizeof( double ) - 256u;
    }
    /// Warp teams launch with LaunchBounds< 64, warp_min_blocks >: 8 for 4/8-byte
    /// storage (128 registers), 10 for 16-bit (its tile is half the size, 102 registers).
    static constexpr int    warp_min_blocks = ( sizeof( ScalarType ) <= 2 ) ? 10 : 8;
    static constexpr size_t warp_shmem_limit() { return kokkos_shmem_limit( 64, warp_min_blocks ); }
    double hourglass_coefficient() const { return stab_c_; }

    void set_boundary_conditions( BoundaryConditions bcs )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];
        stored_matrices_valid_ = false;

        // Recompute host-side dispatch mode whenever BCs change, and with it the
        // scatter mode and tile (the warp-owner path exists for the fast DN kernel only).
        update_kernel_path_flag_host_only();
        set_tile_scatter( tile_scatter_ == 1 && default_tile_scatter == 2 ? 2 : tile_scatter_ );
    }

    // ---- Stored-local-matrix path --------------------------------------------------
    /// Select the stored-local-matrix path (matvec only; the diagonal keeps using the
    /// matrix-free path). The matrices are built on the first apply, or explicitly
    /// with precompute_stored_local_matrices(). They reflect the coefficient field,
    /// boundary conditions and hourglass coefficient at that time: changing the
    /// coefficient afterwards needs invalidate_stored_local_matrices().
    void set_stored_local_matrices( bool on );
    bool stored_local_matrices() const { return use_stored_matrices_; }
    void precompute_stored_local_matrices();
    void invalidate_stored_local_matrices() { stored_matrices_valid_ = false; }
    size_t stored_local_matrices_bytes() const;

    int num_cells() const { return local_subdomains_ * hex_lat_ * hex_lat_ * hex_rad_; }

    /// Cell numbering of the stored matrices: r fastest, so consecutive threads are
    /// radial neighbours and their entries sit next to each other in memory.
    KOKKOS_INLINE_FUNCTION
    void decode_cell_index( const int cell, int& local_subdomain_id, int& x_cell, int& y_cell, int& r_cell ) const
    {
        int tmp            = cell;
        r_cell             = tmp % hex_rad_;
        tmp               /= hex_rad_;
        y_cell             = tmp % hex_lat_;
        tmp               /= hex_lat_;
        x_cell             = tmp % hex_lat_;
        local_subdomain_id = tmp / hex_lat_;
    }

    KOKKOS_INLINE_FUNCTION
    MatT& stored_entry( const int entry, const int cell ) const
    {
        return stored_matrices_( size_t( entry ) * size_t( num_cells_ ) + size_t( cell ) );
    }

    /// Position of ( i, j ), i <= j, in the packed upper triangle.
    static KOKKOS_INLINE_FUNCTION constexpr int packed_index( const int i, const int j )
    {
        return i * LocalMatrixDim - ( i * ( i - 1 ) ) / 2 + ( j - i );
    }

    template < typename T >
    static KOKKOS_INLINE_FUNCTION void dirichlet_boundary_mask(
        bool at_cmb, bool at_surface, dense::Mat< T, LocalMatrixDim, LocalMatrixDim >& mask );

    template < typename T >
    KOKKOS_INLINE_FUNCTION void freeslip_rotation(
        int                                              wedge,
        int                                              local_subdomain_id,
        int                                              x_cell,
        int                                              y_cell,
        int                                              r_cell,
        bool                                             at_cmb,
        dense::Mat< T, LocalMatrixDim, LocalMatrixDim >& R ) const;

    KOKKOS_INLINE_FUNCTION
    dense::Mat< GeoT, LocalMatrixDim, LocalMatrixDim > effective_local_matrix(
        int local_subdomain_id, int x_cell, int y_cell, int r_cell, int wedge, bool at_cmb, bool at_surface ) const;

    KOKKOS_INLINE_FUNCTION
    void run_stored_local_matrices( int cell ) const;

    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }
    const grid::shell::DistributedDomain&       get_domain() const { return domain_; }
    grid::Grid2DDataScalar< CoordScalarT >           get_radii() const { return radii_; }
    grid::Grid3DDataVec< CoordScalarT, 3 >           get_grid() { return grid_; }

    KOKKOS_INLINE_FUNCTION
    bool has_flag(
        const int                      local_subdomain_id,
        const int                      x_cell,
        const int                      y_cell,
        const int                      r_cell,
        grid::shell::ShellBoundaryFlag flag ) const
    {
        return util::has_flag( mask_( local_subdomain_id, x_cell, y_cell, r_cell ), flag );
    }

    void set_stored_matrix_mode(
        linalg::OperatorStoredMatrixMode     operator_stored_matrix_mode,
        int                                  level_range,
        grid::Grid4DDataScalar< ScalarType > GCAElements )
    {
        operator_stored_matrix_mode_ = operator_stored_matrix_mode;
        stored_matrices_valid_       = false;

        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            local_matrix_storage_ = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >(
                domain_, operator_stored_matrix_mode_, level_range, GCAElements );
        }

        // Recompute host-side dispatch mode whenever storage mode changes
        update_kernel_path_flag_host_only();

        util::logroot << "[EpsilonDivDiv] (set_stored_matrix_mode) kernel path = " << kernel_path_name( kernel_path_ )
                      << std::endl;
    }

    linalg::OperatorStoredMatrixMode get_stored_matrix_mode() { return operator_stored_matrix_mode_; }

    KOKKOS_INLINE_FUNCTION
    void set_local_matrix(
        const int                                                    local_subdomain_id,
        const int                                                    x_cell,
        const int                                                    y_cell,
        const int                                                    r_cell,
        const int                                                    wedge,
        const dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim >& mat ) const
    {
        KOKKOS_ASSERT( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off );
        local_matrix_storage_.set_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge, mat );
    }

    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            if ( !local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) )
            {
                Kokkos::abort( "No matrix found at that spatial index." );
            }
            return local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
        }
        return assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
    }

    /**
     * @brief Apply the operator: dst <- Op(src) (or additive, depending on mode).
     *
     * Host-side responsibilities:
     * - Handle replace/add mode initialization
     * - Cache src/dst views for kernel capture
     * - Build team policy
     * - Dispatch to exactly one kernel variant based on `kernel_path_`
     * - Optional halo communication accumulation
     *
     * This is where the path decision now lives (host), so device code does not branch
     * on `kernel_path_` in the hot path.
     */
    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "epsilon_divdiv_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            if ( warp_path_runs() )
                zero_warp_perimeter( dst.grid_data() );
            else
                assign( dst, 0 );
        }

        dst_ = dst.grid_data();
        src_ = src.grid_data();

        util::Timer          timer_kernel( "epsilon_divdiv_kernel" );
        Kokkos::TeamPolicy<> policy( blocks_, team_size_ );

        // The stored matrices serve the matvec; the diagonal is defined by the
        // matrix-free path and falls back to it.
        const bool       stored = ( kernel_path_ == KernelPath::StoredLocalMatrices ) && !diagonal_;
        const KernelPath path   = ( kernel_path_ == KernelPath::StoredLocalMatrices ) ? matrix_free_kernel_path()
                                                                                       : kernel_path_;

        // Fast paths use shared-memory slab staging
        if ( path != KernelPath::Slow )
        {
            policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );
        }

        // Host-side dispatch => no per-team path branching in device code
        if ( stored )
        {
            if ( !stored_matrices_valid_ )
                precompute_stored_local_matrices();

            Kokkos::parallel_for(
                "epsilon_divdiv_apply_kernel_stored_local_matrices",
                Kokkos::RangePolicy<>( 0, num_cells() ),
                KOKKOS_CLASS_LAMBDA( const int cell ) { this->run_stored_local_matrices( cell ); } );
        }
        else if ( path == KernelPath::Slow )
        {
            Kokkos::parallel_for(
                "epsilon_divdiv_apply_kernel_slow", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->run_team_slow( team );
                } );
        }
        else if ( path == KernelPath::FastFreeslip )
        {
            if ( diagonal_ )
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_fs_diag",
                    policy,
                    KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_freeslip< true >( team );
                    } );
            }
            else
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_fs_matvec",
                    policy,
                    KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_freeslip< false >( team );
                    } );
            }
        }
        else
        {
            Kokkos::TeamPolicy< Kokkos::LaunchBounds< TERRA_EPSDIVDIV_MAX_TEAM, TERRA_EPSDIVDIV_MIN_WAVES > > dn_policy( blocks_, team_size_ );
            dn_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size_dn( team_size_ ) ) );
            // Scatter variants: matvec or diagonal, through the tile buffer or not, or
            // the warp-owner reduction (matvec only; the diagonal is computed once).
            const bool tile_buffer = ( tile_scatter_ == 1 ) || ( tile_scatter_ == 2 && !warp_shape_ok() );
            if ( diagonal_ && tile_scatter_ == 2 )
            {
                // The diagonal is computed once and has no warp-owner variant: run it
                // through the plain fast path on a tile that path can launch (the warp
                // tile's full staging would not fit), then restore the warp tiling. The
                // launch lambda copies *this, so the members may change afterwards.
                const int saved_lat = lat_tile_, saved_r = r_tile_, saved_p = r_passes_;
                tile_scatter_ = 0;
                set_tiling( 4, 4, 8, false );
                Kokkos::TeamPolicy< Kokkos::LaunchBounds< TERRA_EPSDIVDIV_MAX_TEAM, TERRA_EPSDIVDIV_MIN_WAVES > >
                    diag_policy( blocks_, team_size_ );
                diag_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size_dn( team_size_ ) ) );
                launch_fast_dirichlet_neumann< true, false >( diag_policy );
                Kokkos::fence();
                tile_scatter_ = 2;
                set_tiling( saved_lat, saved_r, saved_p, false );
            }
            else if ( diagonal_ )
                tile_buffer ? launch_fast_dirichlet_neumann< true, true >( dn_policy )
                            : launch_fast_dirichlet_neumann< true, false >( dn_policy );
            else if ( tile_scatter_ == 2 && warp_shape_ok() )
            {
                // 128 registers per thread, eight 64-thread teams per SM.
                Kokkos::TeamPolicy< Kokkos::LaunchBounds< 64, warp_min_blocks > > warp_policy( blocks_, team_size_ );
                warp_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size_dn( team_size_ ) ) );
                launch_fast_dn_warp( warp_policy );
            }
            else
                tile_buffer ? launch_fast_dirichlet_neumann< false, true >( dn_policy )
                            : launch_fast_dirichlet_neumann< false, false >( dn_policy );
        }

        Kokkos::fence();
        timer_kernel.stop();

        // Null-space penalty for fs/fs: dst += epsilon * sum_i (n_i^T src) n_i
        if ( penalty_active_ && !diagonal_ )
        {
            for ( int i = 0; i < 3; ++i )
            {
                ScalarType alpha = penalty_epsilon_ *
                    kernels::common::masked_dot_product(
                        null_modes_[i], src_, ownership_mask_, grid::NodeOwnershipFlag::OWNED );

                auto nm        = null_modes_[i];
                auto dst_local = dst_;
                Kokkos::parallel_for(
                    "penalty_axpy",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                        { 0, 0, 0, 0 },
                        { dst_local.extent( 0 ), dst_local.extent( 1 ),
                          dst_local.extent( 2 ), dst_local.extent( 3 ) } ),
                    KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                        for ( int d = 0; d < VecDim; ++d )
                            dst_local( s, x, y, r, d ) += alpha * nm( s, x, y, r, d );
                    } );
            }
            Kokkos::fence( "penalty" );
        }

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "epsilon_divdiv_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    /**
     * @brief Convert one gradient column into symmetric-gradient and div contributions.
     *
     * Given grad(phi e_dim), this helper computes:
     * - diagonal entries (E00,E11,E22)
     * - off-diagonal symmetric entries (sym01,sym02,sym12)
     * - divergence contribution `gdd`
     *
     * The fast kernels use this repeatedly in fused operator application.
     */
    KOKKOS_INLINE_FUNCTION
    void column_grad_to_sym(
        const int    dim,
        const double g0,
        const double g1,
        const double g2,
        double&      E00,
        double&      E11,
        double&      E22,
        double&      sym01,
        double&      sym02,
        double&      sym12,
        double&      gdd ) const
    {
        E00 = E11 = E22 = sym01 = sym02 = sym12 = gdd = 0.0;

        switch ( dim )
        {
        case 0:
            E00   = g0;
            gdd   = g0;
            sym01 = 0.5 * g1;
            sym02 = 0.5 * g2;
            break;
        case 1:
            E11   = g1;
            gdd   = g1;
            sym01 = 0.5 * g0;
            sym12 = 0.5 * g2;
            break;
        default:
            E22   = g2;
            gdd   = g2;
            sym02 = 0.5 * g0;
            sym12 = 0.5 * g1;
            break;
        }
    }

    /**
     * @brief Team scratch memory size for fast paths.
     *
     * Layout per team:
     *   [coords_sh | src_sh | k_sh | r_sh]
     */
    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size( const int ts ) const
    {
        const int nlev = r_tile_block_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;

        // coords_sh(nxy,3) + normals_sh(nxy,3) + src_sh(nxy,3,nlev) + k_sh(nxy,nlev) + r_sh(nlev) + 1
        const size_t nscalars =
            size_t( nxy ) * 3 + size_t( nxy ) * 3 + size_t( nxy ) * 3 * nlev + size_t( nxy ) * nlev + size_t( nlev ) + 1;

        // The team scratch tiles (coords_sh/src_sh/k_sh/r_sh) are stored in DOUBLE
        // regardless of ScalarType, so the byte size must use sizeof(double). Using
        // sizeof(ScalarType) under-allocates by 2x in a single-precision build, so the
        // double tiles overrun/overlap and the operator apply reads garbage -> NaN.
        return sizeof( double ) * nscalars;
    }

    /// get_shmem hands out aligned chunks; round each block up so two requests in one
    /// team allocation cannot overlap whatever the element sizes are.
    static KOKKOS_INLINE_FUNCTION constexpr size_t shmem_block_bytes( size_t bytes )
    {
        return ( bytes + 7u ) & ~size_t( 7u );
    }

    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size_dn( const int /* ts */ ) const
    {
        const int nlev = r_tile_block_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;

        // Two blocks, because the tiles no longer share one element type: the geometry
        // (coords_sh, r_sh) is GeomT and the fields (src_sh, k_sh) are ArithT. Each is
        // sized with its own sizeof and rounded up, so they can never overlap -- a
        // single sizeof(double) * nscalars silently under-allocates the moment one tile
        // stops being double, and the apply then reads garbage and produces NaN.
        // Mode 2 (warp-owner): the run's levels plus the one below (halo cell) in both
        // blocks, and the seam between the warps of a team (two parities x (warps - 1)
        // rows x lat columns x 3 components x 2 corners).
        const bool   warp          = ( tile_scatter_ == 2 ) && warp_shape_ok();
        const int    nstaged       = warp ? nlev + 1 : nlev;
        const size_t geom_scalars  = size_t( nxy ) * 3 + size_t( nstaged );
        const size_t field_scalars = size_t( nxy ) * 3 * nstaged + size_t( nxy ) * nstaged;
        const int    nwarps      = ( lat_tile_ * lat_tile_ ) / 32;
        const bool   buffer      = ( tile_scatter_ == 1 ) || ( tile_scatter_ == 2 && !warp );
        const size_t acc_scalars = buffer                 ? size_t( nxy ) * 3 * nlev
                                   : ( warp && nwarps > 1 ) ? size_t( 2 ) * ( nwarps - 1 ) * lat_tile_ * 6
                                                            : 0;

        return shmem_block_bytes( sizeof( GeomT ) * geom_scalars ) +
               shmem_block_bytes( ( warp ? sizeof( StageT ) : sizeof( ArithT ) ) * field_scalars ) +
               shmem_block_bytes( sizeof( AccT ) * acc_scalars );
    }

  private:
    /**
     * @brief Decode a team league rank / team rank into:
     * - tile origin (x0,y0,r0)
     * - intra-tile thread coordinates (tx,ty,tr)
     * - target cell (x_cell,y_cell,r_cell)
     */
    KOKKOS_INLINE_FUNCTION
    void decode_team_indices(
        const Team& team,
        int&        local_subdomain_id,
        int&        x0,
        int&        y0,
        int&        r0,
        int&        tx,
        int&        ty,
        int&        tr,
        int&        x_cell,
        int&        y_cell,
        int&        r_cell ) const
    {
        int tmp = team.league_rank();

        const int r_tile_id = tmp % r_tiles_;
        tmp /= r_tiles_;

        const int lat_y_id = tmp % lat_tiles_;
        tmp /= lat_tiles_;

        const int lat_x_id = tmp % lat_tiles_;
        tmp /= lat_tiles_;

        local_subdomain_id = tmp;

        x0 = lat_x_id * lat_tile_;
        y0 = lat_y_id * lat_tile_;
        r0 = r_tile_id * r_tile_block_;

        const int tid = team.team_rank();
        tr            = tid % r_tile_;
        tx            = ( tid / r_tile_ ) % lat_tile_;
        ty            = tid / ( r_tile_ * lat_tile_ );

        x_cell = x0 + tx;
        y_cell = y0 + ty;
        r_cell = r0 + tr;
    }

    // -------------------------------------------------------------------------
    // Path-specific team entry points (NO path branching)
    // These are the functions called directly by host-dispatched kernel launches.
    // -------------------------------------------------------------------------

    /**
     * @brief Team entry for slow (local-matrix) path.
     *
     * This wrapper performs only:
     * - team index decoding
     * - boundary flag queries
     * - forwarding to operator_slow_path()
     *
     * No dispatch branching happens here.
     */
    KOKKOS_INLINE_FUNCTION
    void run_team_slow( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        for ( int pass = 0; pass < r_passes_; ++pass )
        {
            const int r_cell_pass = r0 + pass * r_tile_ + tr;
            if ( r_cell_pass >= hex_rad_ )
                break;

            const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_pass, CMB );
            const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_pass + 1, SURFACE );

            operator_slow_path(
                team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell_pass, at_cmb, at_surface );
        }
    }

  public:
    /// Public only because nvcc requires the function enclosing an extended lambda
    /// to be; not part of the interface.
    template < bool Diagonal, bool TileScatter, int LatTileC, int NlevC, typename Policy >
    void launch_fast_dirichlet_neumann_shape( const Policy& dn_policy )
    {
        Kokkos::parallel_for(
            Diagonal ? "epsilon_divdiv_apply_kernel_fast_dn_diag" : "epsilon_divdiv_apply_kernel_fast_dn_matvec",
            dn_policy,
            KOKKOS_CLASS_LAMBDA( const Team& team ) {
                this->template run_team_fast_dirichlet_neumann< Diagonal, TileScatter, LatTileC, NlevC >( team );
            } );
    }

    /// The matvec has compile-time instantiations for the tile shapes the sweeps found
    /// best -- (lat, levels per team) = (4, 33) for 4/8/4 and the 4/4/8 default,
    /// (8, 9) for 8/2/4, (8, 17) for 8/2/8 -- and a runtime-shape fallback for every
    /// other tile. The diagonal is computed once and only uses the fallback.
    template < bool Diagonal, bool TileScatter, typename Policy >
    void launch_fast_dirichlet_neumann( const Policy& dn_policy )
    {
        const int nlev = r_tile_block_ + 1;
        if constexpr ( !Diagonal )
        {
            if ( lat_tile_ == 4 && nlev == 33 )
                return launch_fast_dirichlet_neumann_shape< Diagonal, TileScatter, 4, 33 >( dn_policy );
            if ( lat_tile_ == 8 && nlev == 9 )
                return launch_fast_dirichlet_neumann_shape< Diagonal, TileScatter, 8, 9 >( dn_policy );
            if ( lat_tile_ == 8 && nlev == 17 )
                return launch_fast_dirichlet_neumann_shape< Diagonal, TileScatter, 8, 17 >( dn_policy );
        }
        launch_fast_dirichlet_neumann_shape< Diagonal, TileScatter, 0, 0 >( dn_policy );
    }

    /// Public for the same nvcc reason as launch_fast_dirichlet_neumann_shape.
    template < int LatTileC, int NlevC, typename Policy >
    void launch_fast_dn_warp_shape( const Policy& policy )
    {
        Kokkos::parallel_for(
            "epsilon_divdiv_apply_kernel_fast_dn_warp", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                this->template run_team_fast_dn_warp< LatTileC, NlevC >( team );
            } );
    }

    template < typename Policy >
    void launch_fast_dn_warp( const Policy& policy )
    {
        if ( r_tile_block_ == 8 )
            return launch_fast_dn_warp_shape< 8, 9 >( policy );
        launch_fast_dn_warp_shape< 8, 17 >( policy );
    }

  private:
    template < int LatTileC, int NlevC >
    KOKKOS_INLINE_FUNCTION void run_team_fast_dn_warp( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );
        operator_fast_dn_warp_path< LatTileC, NlevC >( team, local_subdomain_id, x0, y0, r0, tx, ty, x_cell, y_cell );
    }

    /// One contribution straight into the global field, in the accumulation type.
    /// A 2-byte field has no lock-free scalar atomic: the add goes through the
    /// packed pair of the 32-bit word the entry lives in, with 0 for the neighbour.
    template < typename V >
    KOKKOS_INLINE_FUNCTION void global_add( ScalarType* p, const V v ) const
    {
        if constexpr ( sizeof( ScalarType ) == 2 )
        {
            // Branch-free: as an if/else on the parity, a warp with lanes of both
            // kinds executed both reductions (ncu: REDG, BRA, BSYNC, LOP3 and SHF
            // each +6.3e7 against the float kernel, one per 16-bit global add).
            const int   odd = int( ( reinterpret_cast< std::uintptr_t >( p ) & 2u ) != 0 );
            const float fv  = static_cast< float >( v );
            util::atomic_add_packed_pair( p - odd, odd ? 0.0f : fv, odd ? fv : 0.0f );
        }
        else
        {
            util::atomic_add_global( p, v );
        }
    }

    /**
     * @brief Team entry for fast Dirichlet/Neumann matrix-free path.
     *
     * Templated on Diagonal so the compiler can dead-code-eliminate the
     * unused matvec or diagonal-only path, reducing register pressure.
     */
    template < bool Diagonal, bool TileScatter, int LatTileC, int NlevC >
    KOKKOS_INLINE_FUNCTION
    void run_team_fast_dirichlet_neumann( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        operator_fast_dirichlet_neumann_path< Diagonal, TileScatter, LatTileC, NlevC >(
            team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
    }

    /**
     * @brief Team entry for fast free-slip matrix-free path.
     *
     * Templated on Diagonal so the compiler can dead-code-eliminate the
     * unused matvec or diagonal-only path, reducing register pressure.
     */
    template < bool Diagonal >
    KOKKOS_INLINE_FUNCTION
    void run_team_fast_freeslip( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        operator_fast_freeslip_path< Diagonal >(
            team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
    }

    // ===================== SLOW PATH (epsilon_divdiv_kerngen/path_slow.hpp) =====================
    KOKKOS_INLINE_FUNCTION
    void operator_slow_path(
        const Team& team,
        const int   local_subdomain_id,
        const int   x0,
        const int   y0,
        const int   r0,
        const int   tx,
        const int   ty,
        const int   tr,
        const int   x_cell,
        const int   y_cell,
        const int   r_cell,
        const bool  at_cmb,
        const bool  at_surface ) const;

    // ===================== FAST DIRICHLET/NEUMANN PATH =====================
    /// Where one node contribution is accumulated: the team's tile buffer when the
    /// storage is narrow, global memory otherwise. The returned pointer differs in
    /// type between the two, which is why this is a selector rather than a branch at
    /// each of the eight scatter sites.
    /// DIAGNOSTIC ONLY. Replaces the tile accumulation's atomic with a plain +=,
    /// which races between threads owning adjacent cells and so produces wrong
    /// results. It exists to bound what removing the LDS atomic could be worth,
    /// before paying for a per-cell register accumulation that may or may not stay
    /// in registers. Never define this for a run whose numbers matter.
#ifdef TERRA_EPSDIVDIV_DIAG_NONATOMIC
    template < bool TileScatter, typename T, typename V >
    static KOKKOS_INLINE_FUNCTION void tile_accumulate( T* p, const V& v )
    {
        *p += static_cast< T >( v );
    }
#else
    /// One contribution into the tile buffer (TileScatter) or global memory. On CUDA
    /// the tile buffer gets a shared-space reduction: through the generic pointer
    /// Kokkos' scratch hands out the compiler can only emit a generic ATOM that the
    /// hardware resolves at run time, where red.shared is the native instruction.
    template < bool TileScatter, typename T, typename V >
    static KOKKOS_INLINE_FUNCTION void tile_accumulate( T* p, const V& v )
    {
#if defined( __CUDA_ARCH__ )
        if constexpr ( TileScatter && std::is_same_v< T, float > )
        {
            const unsigned s = static_cast< unsigned >( __cvta_generic_to_shared( p ) );
            asm volatile( "red.shared.add.f32 [%0], %1;" ::"r"( s ), "f"( static_cast< float >( v ) ) : "memory" );
            return;
        }
        else if constexpr ( TileScatter && std::is_same_v< T, double > )
        {
            const unsigned s = static_cast< unsigned >( __cvta_generic_to_shared( p ) );
            asm volatile( "red.shared.add.f64 [%0], %1;" ::"r"( s ), "d"( static_cast< double >( v ) ) : "memory" );
            return;
        }
#endif
        util::atomic_add_lockfree( p, v );
    }
#endif

    template < bool TileScatter, typename Buf >
    KOKKOS_INLINE_FUNCTION auto* scatter_target( const Buf& tile,
                                                 const int  lnode,
                                                 const int  d,
                                                 const int  llvl,
                                                 const int  sub,
                                                 const int  xi,
                                                 const int  yi,
                                                 const int  rr ) const
    {
        if constexpr ( TileScatter )
        {
            return &tile( lnode, d, llvl );
        }
        else
        {
            return &dst_( sub, xi, yi, rr, d );
        }
    }

    template < bool Diagonal, bool TileScatter, int LatTileC, int NlevC >
    KOKKOS_INLINE_FUNCTION
    void operator_fast_dirichlet_neumann_path(
        const Team& team,
        const int   local_subdomain_id,
        const int   x0,
        const int   y0,
        const int   r0,
        const int   tx,
        const int   ty,
        const int   tr,
        const int   x_cell,
        const int   y_cell ) const;

    template < int LatTileC, int NlevC >
    KOKKOS_INLINE_FUNCTION void operator_fast_dn_warp_path(
        const Team& team,
        const int   local_subdomain_id,
        const int   x0,
        const int   y0,
        const int   r0,
        const int   tx,
        const int   ty,
        const int   x_cell,
        const int   y_cell ) const;

    // ===================== FAST FREESLIP PATH =====================
    KOKKOS_INLINE_FUNCTION
    void normalize3( double& x, double& y, double& z ) const
    {
        const double n2 = x * x + y * y + z * z;
        if ( n2 > 0.0 )
        {
            const double invn = 1.0 / Kokkos::sqrt( n2 );
            x *= invn;
            y *= invn;
            z *= invn;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void project_tangential_inplace(
        const double nx,
        const double ny,
        const double nz,
        double&      ux,
        double&      uy,
        double&      uz ) const
    {
        const double dot = nx * ux + ny * uy + nz * uz;
        ux -= dot * nx;
        uy -= dot * ny;
        uz -= dot * nz;
    }

    template < bool Diagonal >
    KOKKOS_INLINE_FUNCTION
    void operator_fast_freeslip_path(
        const Team& team,
        const int   local_subdomain_id,
        const int   x0,
        const int   y0,
        const int   r0,
        const int   tx,
        const int   ty,
        const int   tr,
        const int   x_cell,
        const int   y_cell ) const;

  public:
    /**
     * @brief Legacy generic team operator.
     *
     * Kept for compatibility/debugging, but no longer used by apply_impl().
     * The host now dispatches directly to path-specific kernels.
     *
     * This function still works, but it reintroduces a branch on `kernel_path_`
     * and should therefore be avoided in performance-critical use.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        if ( kernel_path_ == KernelPath::Slow )
        {
            for ( int pass = 0; pass < r_passes_; ++pass )
            {
                const int r_cell_pass = r0 + pass * r_tile_ + tr;
                if ( r_cell_pass >= hex_rad_ )
                    break;
                const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_pass, CMB );
                const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_pass + 1, SURFACE );
                operator_slow_path(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell_pass, at_cmb, at_surface );
            }
        }
        else if ( kernel_path_ == KernelPath::FastFreeslip )
        {
            if ( diagonal_ )
                operator_fast_freeslip_path< true >(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
            else
                operator_fast_freeslip_path< false >(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
        }
        else
        {
            if ( diagonal_ )
                operator_fast_dirichlet_neumann_path< true, ( default_tile_scatter == 1 ), 0, 0 >(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
            else
                operator_fast_dirichlet_neumann_path< false, ( default_tile_scatter == 1 ), 0, 0 >(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
        }
    }

    KOKKOS_INLINE_FUNCTION
    void assemble_trial_test_vecs(
        const int                            wedge,
        const dense::Vec< GeomT, VecDim >&   quad_point,
        const GeomT                          quad_weight,
        const GeomT                          r_1,
        const GeomT                          r_2,
        dense::Vec< GeomT, 3 > ( *wedge_phy_surf )[3],
        const dense::Vec< ScalarT, 6 >*   k_local_hex,
        const int                         dimi,
        const int                         dimj,
        dense::Mat< GeomT, VecDim, VecDim >* sym_grad_i,
        dense::Mat< GeomT, VecDim, VecDim >* sym_grad_j,
        GeomT&                               jdet_keval_quadweight ) const;

    /**
     * @brief Assemble one wedge-local 18x18 matrix (slow path / on-demand assembly).
     */
    /// @tparam OutT precision the assembled matrix is returned in; the assembly itself runs in GeomT.
    template < typename OutT = ScalarT >
    KOKKOS_INLINE_FUNCTION
    dense::Mat< OutT, LocalMatrixDim, LocalMatrixDim > assemble_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const;
};


static_assert( linalg::GCACapable< EpsilonDivDivKerngen< float > > );
static_assert( linalg::GCACapable< EpsilonDivDivKerngen< double > > );
// Reduced field precision over double-precision geometry. GCA is intentionally not
// claimed for these: the assembled-matrix path stores in ScalarT, which is a
// separate question from the matrix-free apply.
static_assert( linalg::OperatorLike< EpsilonDivDivKerngen< float, 3, double > > );

} // namespace terra::fe::wedge::operators::shell

// The kernel paths, one file each: out-of-line member definitions, so they must
// follow the class and sit outside its namespace block.
#include "epsilon_divdiv_kerngen/path_slow.hpp"
#include "epsilon_divdiv_kerngen/path_fast_dirichlet_neumann.hpp"
#include "epsilon_divdiv_kerngen/path_fast_dn_warp.hpp"
#include "epsilon_divdiv_kerngen/path_fast_freeslip.hpp"
#include "epsilon_divdiv_kerngen/path_stored_local_matrices.hpp"
