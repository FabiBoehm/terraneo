#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"
#include "warp_tile.hpp"

#include <cstdlib>

namespace terra::fe::wedge::operators::shell {

/// Minimum resident teams per SM the SUPG warp path is compiled for.
#ifndef TERRA_SUPG_MIN_BLOCKS
#define TERRA_SUPG_MIN_BLOCKS 8
#endif

// CoordScalarT is the precision of the STORED node coordinates and radii. It
// defaults to ScalarT, so existing instantiations are unchanged. Pass double to
// decouple it: the Jacobian is formed by differencing coordinates, which loses the
// geometry entirely in 16-bit and is already marginal in single. ScalarT governs
// the stored fields src/dst and the coefficient -- where the memory traffic is.
template < typename ScalarT, typename CoordScalarT = ScalarT >
KOKKOS_INLINE_FUNCTION ScalarT
    supg_tau( const ScalarT vel_norm, const ScalarT kappa, const ScalarT h, const ScalarT Pe_tol = 1e-8 )
{
    const ScalarT kappa_min     = ScalarT( 1e-8 );
    const ScalarT kappa_virtual = Kokkos::max( kappa, kappa_min );

    // Guarding against zero velocity.
    const ScalarT eps_vel = ScalarT( 1e-12 );

    if ( vel_norm <= eps_vel )
    {
        return 0.0;
    }

    // For small Peclet numbers (diffusion-dominated flow) set tau to 0 (no stabilization).
    const ScalarT Pe = vel_norm * h / ( 2.0 * kappa_virtual );

    if ( Pe <= Pe_tol )
    {
        return 0.0;
    }

    // Clamp tau to avoid huge values
    const ScalarT tau_max = ScalarT( 10.0 ) * h / Kokkos::max( vel_norm, eps_vel );

    // Finally, compute tau.
    // Note: coth(Pe) = 1/tanh(Pe)
    const ScalarT coth_term = ScalarT( 1.0 ) / Kokkos::tanh( Pe ) - ScalarT( 1.0 ) / Pe;
    const ScalarT tau       = ( h / ( 2.0 * vel_norm ) ) * coth_term;

    return ( tau > tau_max ) ? tau_max : tau;
    // return tau;
}

/// \brief Linear operator for a method-of-lines discretization of the unsteady advection-diffusion equation with SUPG
/// (streamline upwind Petrov-Galerkin) stabilization.
///
/// # Continuous problem
///
/// The unsteady advection-diffusion equation is given by
/// \f[
///   \frac{\partial}{\partial t}T + \mathbf{u} \cdot \nabla T - \nabla \cdot (\kappa \nabla T) = f
/// \f]
/// where \f$ T \f$ is the scalar temperature solution, \f$ \mathbf{u} \f$ a given velocity field,
/// \f$ f \f$ a given source term, and \f$ \kappa \f$ a given diffusivity function.
///
/// # Space discretization
///
/// \note We assume here that we have \f$ \kappa|_E = \mathrm{const} \f$ on each element \f$E\f$ which simplifies the
/// implementation of the SUPG stabilization for linear finite elements as certain terms drop. Currently, \f$\kappa =
/// \mathrm{const}\f$ globally, but once that is changed, we will need to average on every element it in the kernel or
/// pass it in as an elementwise constant (FE) function.
///
/// We first discretize in space, then in time (method of lines). After discretization in space, we get the system of
/// ODEs in time
/// \f[
///   M \frac{d}{dt}T + (C + K + G)T = F + F_\mathrm{SUPG}
/// \f]
/// where
///
/// Term         | Description  | Bilinear form
/// -------------|--------------|--------------
/// \f$M_{ij}\f$ | mass         | \f$ \int \phi_i \phi_j \f$
/// \f$C_{ij}\f$ | advection    | \f$ \int \phi_i (\mathbf{u} \cdot \nabla \phi_j) \f$
/// \f$K_{ij}\f$ | diffusion    | \f$ \int \nabla \phi_i \cdot (\kappa \nabla \phi_j) \f$
/// \f$G_{ij}\f$ | SUPG adv-adv | \f$ \sum_E \int_E \tau_E (\mathbf{u} \cdot \nabla \phi_i) (\mathbf{u} \cdot \nabla \phi_j) \f$
/// \f$F_i\f$    | forcing      | \f$ \int \phi_i f \f$
/// \f$(F_\mathrm{SUPG})_{i}\f$ | SUPG forcing | \f$ \sum_E \int_E \tau_E (\mathbf{u} \cdot \nabla \phi_i) f \f$.
///
/// \note After brief testing, it seems that in general the term \f$F_\mathrm{SUPG}\f$ does not always improve the
///       computed solutions. It even appears to slightly increase the error sometimes. So setting
///       \f$F_\mathrm{SUPG} = 0\f$ can really be just fine in certain settings.
///
/// # Time discretization
///
/// For the time discretization, we employ implicit BDF schemes. A general formula is
///
/// \f[
///   (\alpha_0 M + \Delta t A) T^{n+1} = - \sum_{j=1}^k \alpha_j M T^{n+1-j} + \Delta t R^{n+1}
/// \f]
///
/// where \f$A = (C + K + G)\f$ and \f$R = F + F_{\mathrm{SUPG}}\f$.
///
/// We recover the common BDF1 (backward or implicit Euler) and BDF2 schemes by choosing:
///
/// Scheme                          | \f$k\f$ | \f$\alpha\f$                         | full equation
/// --------------------------------|---------|--------------------------------------|--------------
/// BDF1 (backward/implicit Euler)  | 1       | \f$[1, -1]\f$                        | \f$(M + \Delta t A) T^{n+1} = M T^{n} + \Delta t R^{n+1}\f$
/// BDF2                            | 2       | \f$[\frac{3}{2}, -2, \frac{1}{2}]\f$ | \f$(\frac{3}{2} M + \Delta t A) T^{n+1} = 2 M T^{n} - \frac{1}{2} M T^{n-1} + \Delta t R^{n+1}\f$
///
/// The RHS term must be built with appropriate other classes/function. This class is only concerned with the
/// matrix-free evaluation of the LHS system matrix
/// \f[
///    \alpha_0 M + \Delta t A.
/// \f]
///
/// The parameters \f$\alpha_0\f$ and \f$\Delta t\f$ are set through the constructor via `mass_scaling` and `dt`
/// respectively.
///
/// # SUPG stabilization
///
/// Several choices for the stabilization parameter \f$ \tau_E \f$ are possible.
/// As it is commonly selected in the literature (see, e.g., John and Knobloch (2006)), we set on element \f$E\f$
/// \f[
///  \tau_E = \frac{h_E}{2 \|\mathbf{u}\|} \left(\coth(\mathrm{Pe_E}) - \frac{1}{\mathrm{Pe_E}}\right)
/// \f]
/// where
/// \f[
///  \mathrm{Pe_E} = \frac{\|\mathbf{u}\|h_E}{2 \kappa}
/// \f]
/// with some precautionary measures to avoid edge cases that would result in division by zero etc.
///
/// # Kernel paths
///
/// The per-cell path (one thread per hex cell, 6x6 local matrices, atomics; every
/// backend) and the warp-owner path (warp_tile.hpp; CUDA/HIP): the temperature and
/// the velocity are staged per 8x8 column tile, the six-point rule is evaluated per
/// wedge from the staged coordinates, and the result is scattered through register
/// carry and warp shuffles like the EpsDivDiv fast path. ArithScalarT is the staging
/// and contraction precision of the warp path.
template < typename ScalarT, int VelocityVecDim = 3, typename CoordScalarT = ScalarT, typename ArithScalarT = ScalarT >
class UnsteadyAdvectionDiffusionSUPG
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;
    using GeomT         = CoordScalarT;
    using ArithT        = ArithScalarT;
    using GeoT          = std::conditional_t< ( sizeof( GeomT ) >= 4 ), GeomT, float >;
    using StageT        = ArithT;
    using AccT          = std::conditional_t< ( sizeof( ArithT ) >= 4 ), ArithT, float >;
    using Team          = warp_tile::Team;

    static constexpr int warp_min_blocks = TERRA_SUPG_MIN_BLOCKS;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< CoordScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< CoordScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    linalg::VectorQ1Vec< ScalarT, VelocityVecDim > velocity_;

    ScalarT diffusivity_;
    ScalarT dt_;

    bool    treat_boundary_;
    bool    diagonal_;
    ScalarT mass_scaling_;
    bool    lumped_mass_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;
    communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataScalar< ScalarType > > comm_plan_;

    grid::Grid4DDataScalar< ScalarType >              src_;
    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

    // Warp path.
    warp_tile::Tiling tiling_;
    bool              fast_enabled_ = true;

  public:
    UnsteadyAdvectionDiffusionSUPG(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< CoordScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< CoordScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const linalg::VectorQ1Vec< ScalarT, VelocityVecDim >&           velocity,
        const ScalarT                                                   diffusivity,
        const ScalarT                                                   dt,
        bool                                                            treat_boundary,
        bool                                                            diagonal     = false,
        ScalarT                                                         mass_scaling = 1.0,
        bool                                                            lumped_mass  = false,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , boundary_mask_( boundary_mask )
    , velocity_( velocity )
    , diffusivity_( diffusivity )
    , dt_( dt )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , mass_scaling_( mass_scaling )
    , lumped_mass_( lumped_mass )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    , comm_plan_( domain )
    {
        int r_block = warp_tile::default_r_block( domain );
        if ( r_block == 16 && shmem_bytes< 17 >() > warp_tile::kokkos_shmem_limit( 64, warp_min_blocks ) )
            r_block = 8;
        tiling_ = warp_tile::Tiling( domain, r_block );
        if ( const char* env = std::getenv( "TERRA_SUPG_FAST" ) )
            fast_enabled_ = ( std::atoi( env ) != 0 );
    }

    ScalarT&       dt() { return dt_; }
    const ScalarT& dt() const { return dt_; }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void set_fast_path( const bool on ) { fast_enabled_ = on; }

    bool fast_path_runs() const
    {
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > || VelocityVecDim != 3 )
            return false;
        else
            return fast_enabled_ && tiling_.usable();
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "ad_supg_apply" );

        const bool fast = fast_path_runs();

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            if ( fast )
                warp_tile::zero_perimeter( dst.grid_data(), tiling_ );
            else
                assign( dst, 0 );
        }

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        vel_grid_ = velocity_.grid_data();

        util::Timer timer_kernel( "ad_supg_kernel" );
        if ( fast )
        {
            if ( tiling_.r_block == 16 )
                launch_warp< 17 >();
            else
                launch_warp< 9 >();
        }
        else
        {
            Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        }
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "ad_supg_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    // ------------------------------------------------------------------ warp path

    /// Team scratch for NlevC node levels per run (plus the halo level): geometry,
    /// the temperature and velocity tiles, the inter-warp seam.
    template < int NlevC >
    static constexpr size_t shmem_bytes()
    {
        constexpr int nslot = NlevC + 1;
        return warp_tile::shmem_block_bytes( sizeof( GeomT ) * ( size_t( warp_tile::nxy ) * 3 + nslot ) ) +
               warp_tile::shmem_block_bytes( sizeof( StageT ) * size_t( warp_tile::nxy ) * 4 * nslot ) +
               warp_tile::shmem_block_bytes( sizeof( AccT ) * warp_tile::seam_scalars< 1 >() );
    }

    template < int NlevC >
    void launch_warp()
    {
        Kokkos::TeamPolicy< Kokkos::LaunchBounds< 64, warp_min_blocks > > policy( tiling_.blocks, warp_tile::team_size );
        policy.set_scratch_size( 0, Kokkos::PerTeam( shmem_bytes< NlevC >() ) );
        Kokkos::parallel_for(
            "ad_supg_warp", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                this->template run_team_warp< NlevC >( team );
            } );
    }

    template < int NlevC >
    KOKKOS_INLINE_FUNCTION void run_team_warp( const Team& team ) const
    {
        namespace wt = warp_tile;
        constexpr int nlev  = NlevC;
        constexpr int nslot = NlevC + 1;

        int sub, x0, y0, r0;
        tiling_.decode( team.league_rank(), sub, x0, y0, r0 );
        const int hex_lat = tiling_.hex_lat, hex_rad = tiling_.hex_rad;
        const int rank = team.team_rank();
        const int tx = rank % wt::lat, ty = rank / wt::lat;
        const int x_cell = x0 + tx, y_cell = y0 + ty;
        const int lane = rank % 32, wy = rank / 32, ltx = lane % wt::PW, lty = lane / wt::PW;

        auto* geom_mem = reinterpret_cast< GeomT* >( team.team_shmem().get_shmem(
            wt::shmem_block_bytes( sizeof( GeomT ) * ( size_t( wt::nxy ) * 3 + nslot ) ) ) );
        auto* field_mem = reinterpret_cast< StageT* >( team.team_shmem().get_shmem(
            wt::shmem_block_bytes( sizeof( StageT ) * size_t( wt::nxy ) * 4 * nslot ) ) );
        auto* seam_mem = reinterpret_cast< AccT* >(
            team.team_shmem().get_shmem( wt::shmem_block_bytes( sizeof( AccT ) * wt::seam_scalars< 1 >() ) ) );

        using Scratch       = typename Team::scratch_memory_space;
        using ScratchCoords = Kokkos::View< GeomT[wt::nxy][3], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        using ScratchR      = Kokkos::View< GeomT[nslot], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        using ScratchF = Kokkos::View< StageT[wt::nxy][4][nslot], Kokkos::LayoutRight, Scratch, Kokkos::MemoryUnmanaged >;
        ScratchCoords coords_sh( geom_mem );
        ScratchR      r_sh( geom_mem + size_t( wt::nxy ) * 3 );
        ScratchF      f_sh( field_mem ); // (node, 0..2 velocity / 3 temperature, slot)

        wt::stage_geometry< nslot, GeomT >( team, coords_sh, r_sh, grid_, radii_, sub, x0, y0, r0, hex_lat, hex_rad );

        constexpr bool AsyncStaging =
            std::is_same_v< StageT, ScalarType > && ( sizeof( ScalarType ) == 4 || sizeof( ScalarType ) == 8 );
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, wt::nxy * nslot ), [&]( int t ) {
            const int node = t / nslot;
            const int lvl  = t - node * nslot;
            const int xi   = x0 + node % ( wt::lat + 1 );
            const int yi   = y0 + node / ( wt::lat + 1 );
            const int rr   = r0 - 1 + lvl;
            if ( xi <= hex_lat && yi <= hex_lat && rr >= 0 && rr <= hex_rad )
            {
                if constexpr ( AsyncStaging )
                {
                    util::cp_async( &f_sh( node, 0, lvl ), &vel_grid_( sub, xi, yi, rr, 0 ) );
                    util::cp_async( &f_sh( node, 1, lvl ), &vel_grid_( sub, xi, yi, rr, 1 ) );
                    util::cp_async( &f_sh( node, 2, lvl ), &vel_grid_( sub, xi, yi, rr, 2 ) );
                    util::cp_async( &f_sh( node, 3, lvl ), &src_( sub, xi, yi, rr ) );
                }
                else
                {
                    f_sh( node, 0, lvl ) = static_cast< StageT >( vel_grid_( sub, xi, yi, rr, 0 ) );
                    f_sh( node, 1, lvl ) = static_cast< StageT >( vel_grid_( sub, xi, yi, rr, 1 ) );
                    f_sh( node, 2, lvl ) = static_cast< StageT >( vel_grid_( sub, xi, yi, rr, 2 ) );
                    f_sh( node, 3, lvl ) = static_cast< StageT >( src_( sub, xi, yi, rr ) );
                }
            }
            else
            {
                f_sh( node, 0, lvl ) = f_sh( node, 1, lvl ) = f_sh( node, 2, lvl ) = f_sh( node, 3, lvl ) = StageT( 0 );
            }
        } );
        if constexpr ( AsyncStaging )
            util::cp_async_wait_all();
        team.team_barrier();

        auto node_id = []( int nx, int ny ) -> int { return nx + ( wt::lat + 1 ) * ny; };
        const int corner_node[4] = { node_id( tx, ty ), node_id( tx + 1, ty ), node_id( tx, ty + 1 ), node_id( tx + 1, ty + 1 ) };

        GeoT cx[4], cy[4], cz[4];
#pragma unroll
        for ( int c = 0; c < 4; ++c )
        {
            cx[c] = static_cast< GeoT >( coords_sh( corner_node[c], 0 ) );
            cy[c] = static_cast< GeoT >( coords_sh( corner_node[c], 1 ) );
            cz[c] = static_cast< GeoT >( coords_sh( corner_node[c], 2 ) );
        }

        // The Felippa 3x2 rule: three lateral points x two radial, weight 1/6 each.
        // Lateral shape values at the three lateral points, radial at the two zetas.
        constexpr GeoT TWO_THIRDS = static_cast< GeoT >( 0.6666666666666666 );
        constexpr GeoT ONE_SIXTH  = static_cast< GeoT >( 0.1666666666666667 );
        constexpr GeoT ZETA       = static_cast< GeoT >( 0.5773502691896257 );
        constexpr GeoT HALF       = static_cast< GeoT >( 0.5 );
        // (point, lateral node): 1-xi-eta, xi, eta at (2/3,1/6), (1/6,2/3), (1/6,1/6)
        constexpr GeoT LATW[3][3] = { { ONE_SIXTH, TWO_THIRDS, ONE_SIXTH },
                                      { ONE_SIXTH, ONE_SIXTH, TWO_THIRDS },
                                      { TWO_THIRDS, ONE_SIXTH, ONE_SIXTH } };
        constexpr GeoT QXI[3]  = { TWO_THIRDS, ONE_SIXTH, ONE_SIXTH };
        constexpr GeoT QETA[3] = { ONE_SIXTH, TWO_THIRDS, ONE_SIXTH };
        // (zeta side, bottom/top): 0.5 (1 -+ zeta)
        constexpr GeoT RADW[2][2]  = { { HALF * ( 1 + ZETA ), HALF * ( 1 - ZETA ) }, { HALF * ( 1 - ZETA ), HALF * ( 1 + ZETA ) } };
        constexpr GeoT ZETAS[2]    = { -ZETA, ZETA };
        constexpr GeoT DLAT_XI[3]  = { -1, 1, 0 };
        constexpr GeoT DLAT_ETA[3] = { -1, 0, 1 };
        constexpr GeoT QW          = ONE_SIXTH;

        const GeoT kappa = static_cast< GeoT >( diffusivity_ );
        const GeoT dt    = static_cast< GeoT >( dt_ );
        const GeoT ms    = static_cast< GeoT >( mass_scaling_ );

        const int s_begin = ( r0 > 0 ) ? 0 : 1;

        // Field values of the 4 columns at the cell's lower and upper level: velocity 0..2, T 3.
        StageT fl[4][4], fu[4][4];
#pragma unroll
        for ( int c = 0; c < 4; ++c )
#pragma unroll
            for ( int d = 0; d < 4; ++d )
                fl[c][d] = f_sh( corner_node[c], d, s_begin );

        AccT acc[4][1], up[4][1];
#pragma unroll
        for ( int c = 0; c < 4; ++c )
            acc[c][0] = AccT( 0 );

        const bool replace_mode = ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace );
        const bool interior     = ( tx > 0 ) && ( ty > 0 );

        int  s_flush = 0;
        auto emit    = [&]( const int ddx, const int ddy, const int /*d*/, const AccT val, const bool owned ) {
            const int   rr = r0 - 1 + s_flush;
            ScalarType* p  = &dst_( sub, x_cell + ddx, y_cell + ddy, rr );
            if ( owned && interior )
                *p = static_cast< ScalarType >( replace_mode ? val : static_cast< AccT >( *p ) + val );
            else if ( val != AccT( 0 ) )
                wt::global_add( p, val );
        };

        int s_top = s_begin;
        for ( int s = s_begin; s < nlev; ++s )
        {
            const int r_cell = r0 - 1 + s;
            if ( r_cell >= hex_rad ) // uniform over the team
                break;
            s_top = s + 1;

#pragma unroll
            for ( int cc = 0; cc < 4; ++cc )
            {
                up[cc][0] = AccT( 0 );
#pragma unroll
                for ( int d = 0; d < 4; ++d )
                    fu[cc][d] = f_sh( corner_node[cc], d, s + 1 );
            }

            const GeoT r_1     = static_cast< GeoT >( r_sh( s ) );
            const GeoT r_2     = static_cast< GeoT >( r_sh( s + 1 ) );
            const GeoT h       = r_2 - r_1;
            const GeoT half_dr = HALF * h;

            const bool at_cmb = treat_boundary_ && util::has_flag( boundary_mask_( sub, x_cell, y_cell, r_cell ),
                                                                   grid::shell::ShellBoundaryFlag::CMB );
            const bool at_surface = treat_boundary_ && util::has_flag( boundary_mask_( sub, x_cell, y_cell, r_cell + 1 ),
                                                                       grid::shell::ShellBoundaryFlag::SURFACE );

#pragma unroll
            for ( int w = 0; w < 2; ++w )
            {
                const int v0 = wt::wedge_vertex( w, 0 ), v1 = wt::wedge_vertex( w, 1 ), v2 = wt::wedge_vertex( w, 2 );

                // Nodal values of the wedge: velocity and temperature.
                GeoT un[6][3], Tn[6];
                bool bnd[6]; // boundary node: its row is diagonal-only, its column is dropped elsewhere
#pragma unroll
                for ( int n = 0; n < 6; ++n )
                {
                    const int  cc  = wt::wedge_node_off( w, n, 0 ) + 2 * wt::wedge_node_off( w, n, 1 );
                    const bool top = wt::wedge_node_off( w, n, 2 );
#pragma unroll
                    for ( int d = 0; d < 3; ++d )
                        un[n][d] = static_cast< GeoT >( top ? fu[cc][d] : fl[cc][d] );
                    Tn[n]  = static_cast< GeoT >( top ? fu[cc][3] : fl[cc][3] );
                    bnd[n] = ( n < 3 ) ? at_cmb : at_surface;
                }

                // Lateral edges of the wedge on the unit sphere.
                const GeoT e1x = cx[v1] - cx[v0], e1y = cy[v1] - cy[v0], e1z = cz[v1] - cz[v0];
                const GeoT e2x = cx[v2] - cx[v0], e2y = cy[v2] - cy[v0], e2z = cz[v2] - cz[v0];

                // Velocity at the six quadrature points and the wedge's tau (the
                // volume-weighted average with equal weights is the plain mean).
                GeoT uq[6][3];
                GeoT tau = 0;
#pragma unroll
                for ( int q = 0; q < 6; ++q )
                {
                    const int l = q % 3, z = q / 3;
                    uq[q][0] = uq[q][1] = uq[q][2] = 0;
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        const GeoT N = LATW[l][n % 3] * RADW[z][n / 3];
                        uq[q][0] += N * un[n][0];
                        uq[q][1] += N * un[n][1];
                        uq[q][2] += N * un[n][2];
                    }
                    const GeoT un2 = uq[q][0] * uq[q][0] + uq[q][1] * uq[q][1] + uq[q][2] * uq[q][2];
                    tau += supg_tau< GeoT >( Kokkos::sqrt( un2 ), kappa, h, static_cast< GeoT >( 1e-08 ) );
                }
                tau *= ONE_SIXTH;

                AccT out[6];
#pragma unroll
                for ( int n = 0; n < 6; ++n )
                    out[n] = AccT( 0 );

#pragma unroll
                for ( int q = 0; q < 6; ++q )
                {
                    const int  l   = q % 3, z = q / 3;
                    const GeoT xi  = QXI[l], eta = QETA[l];
                    const GeoT r_q = r_1 + half_dr * ( 1 + ZETAS[z] );
                    const GeoT lx  = cx[v0] + xi * e1x + eta * e2x;
                    const GeoT ly  = cy[v0] + xi * e1y + eta * e2y;
                    const GeoT lz  = cz[v0] + xi * e1z + eta * e2z;

                    const GeoT J00 = e1x * r_q, J01 = e2x * r_q, J02 = lx * half_dr;
                    const GeoT J10 = e1y * r_q, J11 = e2y * r_q, J12 = ly * half_dr;
                    const GeoT J20 = e1z * r_q, J21 = e2z * r_q, J22 = lz * half_dr;
                    const GeoT det = J00 * J11 * J22 - J00 * J12 * J21 - J01 * J10 * J22 + J01 * J12 * J20 +
                                     J02 * J10 * J21 - J02 * J11 * J20;
                    const GeoT inv_det = static_cast< GeoT >( 1.0 ) / det;
                    // Entries of J^-T, as in WedgeGeometry.
                    const GeoT i00 = inv_det * ( J11 * J22 - J12 * J21 );
                    const GeoT i01 = inv_det * ( -J10 * J22 + J12 * J20 );
                    const GeoT i02 = inv_det * ( J10 * J21 - J11 * J20 );
                    const GeoT i10 = inv_det * ( -J01 * J22 + J02 * J21 );
                    const GeoT i11 = inv_det * ( J00 * J22 - J02 * J20 );
                    const GeoT i12 = inv_det * ( -J00 * J21 + J01 * J20 );
                    const GeoT i20 = inv_det * ( J01 * J12 - J02 * J11 );
                    const GeoT i21 = inv_det * ( -J00 * J12 + J02 * J10 );
                    const GeoT i22 = inv_det * ( J00 * J11 - J01 * J10 );
                    const GeoT wdet = QW * Kokkos::abs( det );

                    // Shape values, physical gradients, and the interpolated T and grad T
                    // with the boundary columns dropped.
                    GeoT N[6], g0[6], g1[6], g2[6];
                    GeoT Tq = 0, gT0 = 0, gT1 = 0, gT2 = 0;
#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        const GeoT nl = LATW[l][n % 3], nr = RADW[z][n / 3];
                        N[n]          = nl * nr;
                        const GeoT dx = DLAT_XI[n % 3] * nr, dy = DLAT_ETA[n % 3] * nr, dz = nl * ( ( n / 3 ) ? HALF : -HALF );
                        g0[n]         = i00 * dx + i01 * dy + i02 * dz;
                        g1[n]         = i10 * dx + i11 * dy + i12 * dz;
                        g2[n]         = i20 * dx + i21 * dy + i22 * dz;
                        const GeoT tj = bnd[n] ? GeoT( 0 ) : Tn[n];
                        Tq += N[n] * tj;
                        gT0 += g0[n] * tj;
                        gT1 += g1[n] * tj;
                        gT2 += g2[n] * tj;
                    }
                    const GeoT u0 = uq[q][0], u1 = uq[q][1], u2 = uq[q][2];
                    const GeoT ugT = u0 * gT0 + u1 * gT1 + u2 * gT2;

#pragma unroll
                    for ( int n = 0; n < 6; ++n )
                    {
                        const GeoT ugn = u0 * g0[n] + u1 * g1[n] + u2 * g2[n];
                        GeoT       val;
                        if ( diagonal_ || bnd[n] )
                        {
                            // The diagonal entry stands in for the row (the lumped mass is the
                            // row sum before masking, i.e. sum_j N_j = 1).
                            const GeoT mass = ms * N[n] * ( lumped_mass_ ? GeoT( 1 ) : N[n] );
                            const GeoT gg   = g0[n] * g0[n] + g1[n] * g1[n] + g2[n] * g2[n];
                            val = ( mass + dt * ( kappa * gg + N[n] * ugn + tau * ugn * ugn ) ) * Tn[n];
                        }
                        else
                        {
                            const GeoT mass = ms * N[n] * ( lumped_mass_ ? Tn[n] : Tq );
                            const GeoT diff = kappa * ( g0[n] * gT0 + g1[n] * gT1 + g2[n] * gT2 );
                            val = mass + dt * ( diff + N[n] * ugT + tau * ugn * ugT );
                        }
                        out[n] += static_cast< AccT >( wdet * val );
                    }
                }

#pragma unroll
                for ( int n = 0; n < 6; ++n )
                {
                    const int cc = wt::wedge_node_off( w, n, 0 ) + 2 * wt::wedge_node_off( w, n, 1 );
                    if ( wt::wedge_node_off( w, n, 2 ) )
                        up[cc][0] += out[n];
                    else
                        acc[cc][0] += out[n];
                }
            }

            if ( s >= 1 )
            {
                s_flush = s;
                wt::flush_level< 1 >( team, acc, ltx, lty, wy, seam_mem, s & 1, true, emit );
            }

#pragma unroll
            for ( int cc = 0; cc < 4; ++cc )
            {
                acc[cc][0] = up[cc][0];
#pragma unroll
                for ( int d = 0; d < 4; ++d )
                    fl[cc][d] = fu[cc][d];
            }
        }

        if ( r0 - 1 + s_top == hex_rad )
        {
            s_flush = s_top;
            wt::flush_level< 1 >( team, acc, ltx, lty, wy, seam_mem, s_top & 1, true, emit );
        }
    }

    // ------------------------------------------------------------------ per-cell path

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // Quadrature points.
        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        // Interpolating velocity into quad points.

        dense::Vec< ScalarT, VelocityVecDim > vel_interp[num_wedges_per_hex_cell][num_quad_points];
        dense::Vec< ScalarT, 6 >              vel_coeffs[VelocityVecDim][num_wedges_per_hex_cell];

        for ( int d = 0; d < VelocityVecDim; d++ )
        {
            extract_local_wedge_vector_coefficients(
                vel_coeffs[d], local_subdomain_id, x_cell, y_cell, r_cell, d, vel_grid_ );
        }

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    for ( int d = 0; d < VelocityVecDim; d++ )
                    {
                        vel_interp[wedge][q]( d ) += vel_coeffs[d][wedge]( i ) * shape_i;
                    }
                }
            }
        }

        // Let's compute the streamline diffusivity.

        ScalarT streamline_diffusivity[num_wedges_per_hex_cell];

        // Far from accurate but for now assume h = r.
        const auto h = r_2 - r_1;

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            ScalarT tau_accum = 0.0;
            ScalarT waccum    = 0.0;

            for ( int q = 0; q < num_quad_points; ++q )
            {
                // get velocity at this quad point
                const auto&   uq         = vel_interp[wedge][q];
                const ScalarT vel_norm_q = uq.norm();

                const ScalarT tau_q = supg_tau< ScalarT >( vel_norm_q, diffusivity_, h, 1e-08 );

                // quadrature weight for this point (if you have weights)
                const ScalarT wq = quad_weights[q]; // if not available, use 1.0
                tau_accum += tau_q * wq;
                waccum += wq;
            }

            // final cell/wedge tau: volume-weighted average
            ScalarT tau_cell              = ( waccum > 0.0 ) ? ( tau_accum / waccum ) : 0.0;
            streamline_diffusivity[wedge] = tau_cell;
        }

        // Compute the local element matrix.
        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};
        dense::Mat< ScalarT, 6, 6 > M[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; q++ )
        {
            const auto w = quad_weights[q];

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                const auto vel = vel_interp[wedge][q];

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    const auto grad_i  = J_inv_transposed * grad_shape( i, quad_points[q] );

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto shape_j = shape( j, quad_points[q] );
                        const auto grad_j  = J_inv_transposed * grad_shape( j, quad_points[q] );

                        const auto mass      = shape_i * shape_j;
                        const auto diffusion = diffusivity_ * ( grad_i ).dot( grad_j );
                        const auto advection = ( vel.dot( grad_j ) ) * shape_i;
                        const auto streamline_diffusion =
                            streamline_diffusivity[wedge] * ( vel.dot( grad_j ) ) * ( vel.dot( grad_i ) );

                        M[wedge]( i, j ) += w * mass_scaling_ * mass * det;
                        A[wedge]( i, j ) += w * dt_ * ( diffusion + advection + streamline_diffusion ) * det;
                    }
                }
            }
        }

        if ( lumped_mass_ )
        {
            dense::Vec< ScalarT, 6 > ones;
            ones.fill( 1.0 );
            M[0] = dense::Mat< ScalarT, 6, 6 >::diagonal_from_vec( M[0] * ones );
            M[1] = dense::Mat< ScalarT, 6, 6 >::diagonal_from_vec( M[1] * ones );
        }

        if ( treat_boundary_ )
        {
            const int at_cmb_boundary = util::has_flag(
                boundary_mask_( local_subdomain_id, x_cell, y_cell, r_cell ), grid::shell::ShellBoundaryFlag::CMB );
            const int at_surface_boundary = util::has_flag(
                boundary_mask_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ),
                grid::shell::ShellBoundaryFlag::SURFACE );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                dense::Mat< ScalarT, 6, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( at_cmb_boundary )
                {
                    // Inner boundary (CMB).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i < 3 || j < 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                if ( at_surface_boundary )
                {
                    // Outer boundary (surface).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i >= 3 || j >= 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                M[wedge].hadamard_product( boundary_mask );
                A[wedge].hadamard_product( boundary_mask );
            }
        }

        if ( diagonal_ )
        {
            M[0] = M[0].diagonal();
            M[1] = M[1].diagonal();
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        dst[0] = ( M[0] + A[0] ) * src[0];
        dst[1] = ( M[1] + A[1] ) * src[1];

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }
};

static_assert( linalg::OperatorLike< UnsteadyAdvectionDiffusionSUPG< double > > );

} // namespace terra::fe::wedge::operators::shell
