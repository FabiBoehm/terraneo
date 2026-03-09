

#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv.hpp"
#include "fe/wedge/operators/shell/prolongation_linear.hpp"
#include "fe/wedge/shell/grid_transfer_constant.hpp"
#include "fe/wedge/shell/grid_transfer_linear.hpp"
#include "grid/grid_types.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/gca/gca_elements_collector.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

using terra::fe::wedge::num_nodes_per_wedge;
using terra::fe::wedge::num_wedges_per_hex_cell;

using terra::grid::shell::ShellBoundaryFlag::CMB;
using terra::grid::shell::ShellBoundaryFlag::SURFACE;
using terra::util::has_flag;
namespace terra::linalg::solvers {

/// @brief Modes for choosing interpolation weights.
enum class InterpolationMode
{
    Constant,
    Linear,
    OperatorDependent,
};

/// @brief: Galerkin coarse approximation (GCA).
/// TwoGridGCA takes a coarser and a finer operator. Each thread assembles a
/// coarse-grid gca matrix in the coarser operator on a single hex. To do this, it loops
/// the finer hexes of the coarse hex and its respective wedges. It computes the interpolation
/// matrix P mapping from coarse wedge to the current fine wedge, computes the
/// triple-product P^TAP with the fine-operator local matrix A and adds the resulting gca matrix
/// up for all fine wedges comprising the coarse wedge. Finally, it stores the result in the
/// wedge-wise matrix storage of the coarse operator.
template < typename ScalarT, terra::linalg::GCACapable Operator >
class TwoGridGCA
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< double >;
    using DstVectorType = linalg::VectorQ1Scalar< double >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain    domain_fine_;
    Operator                          fine_op_;
    Operator                          coarse_op_;
    grid::Grid3DDataVec< ScalarT, 3 > grid_fine_;
    grid::Grid2DDataScalar< ScalarT > radii_fine_;
    grid::Grid2DDataScalar< ScalarT > radii_coarse_;
    bool                              treat_boundary_;

    int                                  level_range_;
    grid::Grid4DDataScalar< ScalarType > GCAElements_;
    InterpolationMode                    interpolation_mode_;

    // Precomputed AMG weights for OperatorDependent mode.
    grid::Grid4DDataScalar< ScalarT > parent_weight_0_;
    grid::Grid4DDataScalar< ScalarT > parent_weight_1_;
    grid::Grid4DDataScalar< ScalarT > parent_weight_2_;
    grid::Grid4DDataScalar< ScalarT > parent_weight_3_;

    int fine_num_cells_x_;
    int fine_num_cells_y_;
    int fine_num_cells_r_;

  public:
    /// @brief GCA Ctor
    /// Assembles Galerkin coarse-grid operators in the coarse-op passed.
    /// @param fine_op: operator on the finer grid to derive the coarse-grid operators from
    /// @param coarse_op: operator on the coarser grid to store the coarse-grid operators in
    /// @param level_range: max_level - min_level range used in the app: required check whether a certain element
    ///                     is a child of a GCA element.
    /// @param GCAElements: map of coarsest-grid elements, on which GCA should be used. Using this and level_range,
    ///                     the GCA can check for a certain element whether it is a child of a marked coarsest-grid 
    ///                     element. If that is the case, GCA is applied to it.
    explicit TwoGridGCA(
        Operator                             fine_op,
        Operator                             coarse_op,
        int                                  level_range,
        grid::Grid4DDataScalar< ScalarType > GCAElements,
        bool                                 treat_boundary     = true,
        InterpolationMode                    interpolation_mode = InterpolationMode::Constant )
    : domain_fine_( fine_op.get_domain() )
    , fine_op_( fine_op )
    , coarse_op_( coarse_op )
    , grid_fine_( fine_op.get_grid() )
    , radii_fine_( fine_op.get_radii() )
    , radii_coarse_( coarse_op.get_radii() )
    , GCAElements_( GCAElements )
    , level_range_( level_range )
    , treat_boundary_( treat_boundary )
    , interpolation_mode_( interpolation_mode )
    , fine_num_cells_x_( fine_op.get_domain().domain_info().subdomain_num_nodes_per_side_laterally() - 1 )
    , fine_num_cells_y_( fine_op.get_domain().domain_info().subdomain_num_nodes_per_side_laterally() - 1 )
    , fine_num_cells_r_( fine_op.get_domain().domain_info().subdomain_num_nodes_radially() - 1 )
    {
        // assert( coarse_op_.get_stored_matrix_mode() != linalg::OperatorStoredMatrixMode::Off );

        // this probably cant not happen
        if ( coarse_op.get_domain().subdomains().size() != domain_fine_.subdomains().size() )
        {
            throw std::runtime_error( "Prolongation: src and dst must have a compatible number of subdomains." );
        }

        if ( 2 * ( coarse_op.get_domain().domain_info().subdomain_num_nodes_per_side_laterally() - 1 ) !=
             domain_fine_.domain_info().subdomain_num_nodes_per_side_laterally() - 1 )
        {
            throw std::runtime_error( "Prolongation: src and dst must have a compatible number of lateral cells." );
        }
        if ( 2 * ( coarse_op.get_domain().domain_info().subdomain_num_nodes_radially() - 1 ) !=
             domain_fine_.domain_info().subdomain_num_nodes_radially() - 1 )
        {
            throw std::runtime_error( "Prolongation: src and dst must have a compatible number of radial cells." );
        }

        if ( interpolation_mode == InterpolationMode::OperatorDependent )
        {
            if ( Operator::LocalMatrixDim != 6 )
            {
                throw std::runtime_error(
                    "OperatorDependent interpolation currently only supported for scalar operators (LocalMatrixDim == 6)." );
            }
            precompute_amg_weights();
        }

        // Looping over the coarse grid.
        Kokkos::parallel_for(
            "gca_coarsening",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                {
                    static_cast< long long >( coarse_op.get_domain().subdomains().size() ),
                    coarse_op.get_domain().domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                    coarse_op.get_domain().domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                    coarse_op.get_domain().domain_info().subdomain_num_nodes_radially() - 1,
                } ),
            *this );

        Kokkos::fence();
    }

    /// @brief Precomputes operator-dependent interpolation weights.
    ///
    /// Uses a ratio-corrected approach: computes standard AMG weights (w_amg)
    /// for the actual operator and reference AMG weights (w_ref) for a unit-
    /// coefficient operator. The ratio w_amg/w_ref isolates the coefficient
    /// effect, which is then applied as a correction to geometric linear weights.
    void precompute_amg_weights()
    {
        const auto& domain = domain_fine_;
        const int   num_sd = static_cast< int >( domain.subdomains().size() );
        const int   nx     = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        const int   ny     = nx;
        const int   nr     = domain.domain_info().subdomain_num_nodes_radially();
        const int   ncx    = fine_num_cells_x_;
        const int   ncy    = fine_num_cells_y_;
        const int   ncr    = fine_num_cells_r_;

        // Allocate fields for stiffness entries and diagonal-normalized reference.
        grid::Grid4DDataScalar< ScalarT > a_ip_0( "a_ip_0", num_sd, nx, ny, nr );
        grid::Grid4DDataScalar< ScalarT > a_ip_1( "a_ip_1", num_sd, nx, ny, nr );
        grid::Grid4DDataScalar< ScalarT > a_ip_2( "a_ip_2", num_sd, nx, ny, nr );
        grid::Grid4DDataScalar< ScalarT > a_ip_3( "a_ip_3", num_sd, nx, ny, nr );
        grid::Grid4DDataScalar< ScalarT > a_denom( "a_denom", num_sd, nx, ny, nr );

        grid::Grid4DDataScalar< ScalarT > a_ip_ref_0( "a_ip_ref_0", num_sd, nx, ny, nr );
        grid::Grid4DDataScalar< ScalarT > a_ip_ref_1( "a_ip_ref_1", num_sd, nx, ny, nr );
        grid::Grid4DDataScalar< ScalarT > a_ip_ref_2( "a_ip_ref_2", num_sd, nx, ny, nr );
        grid::Grid4DDataScalar< ScalarT > a_ip_ref_3( "a_ip_ref_3", num_sd, nx, ny, nr );
        grid::Grid4DDataScalar< ScalarT > a_denom_ref( "a_denom_ref", num_sd, nx, ny, nr );

        parent_weight_0_ = grid::Grid4DDataScalar< ScalarT >( "pw0", num_sd, nx, ny, nr );
        parent_weight_1_ = grid::Grid4DDataScalar< ScalarT >( "pw1", num_sd, nx, ny, nr );
        parent_weight_2_ = grid::Grid4DDataScalar< ScalarT >( "pw2", num_sd, nx, ny, nr );
        parent_weight_3_ = grid::Grid4DDataScalar< ScalarT >( "pw3", num_sd, nx, ny, nr );

        auto pw0 = parent_weight_0_;
        auto pw1 = parent_weight_1_;
        auto pw2 = parent_weight_2_;
        auto pw3 = parent_weight_3_;

        auto fine_op    = fine_op_;
        auto grid_fine  = grid_fine_;
        auto radii_fine = radii_fine_;

        // Phase 1: Assemble stiffness entries. For the diagonal-proxy reference,
        // each element's off-diagonal entries are normalized by its diagonal entry
        // A(i,j)/A(i,i), removing element-volume scaling while preserving connectivity.
        Kokkos::parallel_for(
            "amg_weights_phase1",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sd, nx, ny, nr } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                const bool x_even = ( x % 2 == 0 );
                const bool y_even = ( y % 2 == 0 );
                const bool r_even = ( r % 2 == 0 );
                if ( x_even && y_even && r_even )
                    return;

                const int num_parents = ( x_even && y_even ) ? 2 : 4;

                // Determine parent coordinates.
                int r_bot = r / 2;
                int r_top = r_bot + 1;

                int px[4], py[4], pr[4];
                if ( num_parents == 2 )
                {
                    px[0] = x;  py[0] = y;  pr[0] = 2 * r_bot;
                    px[1] = x;  py[1] = y;  pr[1] = 2 * r_top;
                    px[2] = 0;  py[2] = 0;  pr[2] = 0;
                    px[3] = 0;  py[3] = 0;  pr[3] = 0;
                }
                else
                {
                    int x0, y0, x1, y1;
                    if ( x_even )
                    {
                        x0 = x / 2;  x1 = x / 2;
                        y0 = y / 2;  y1 = y / 2 + 1;
                    }
                    else if ( y_even )
                    {
                        x0 = x / 2;  x1 = x / 2 + 1;
                        y0 = y / 2;  y1 = y / 2;
                    }
                    else
                    {
                        x0 = x / 2 + 1;  x1 = x / 2;
                        y0 = y / 2;      y1 = y / 2 + 1;
                    }
                    px[0] = 2 * x0;  py[0] = 2 * y0;  pr[0] = 2 * r_bot;
                    px[1] = 2 * x1;  py[1] = 2 * y1;  pr[1] = 2 * r_bot;
                    px[2] = 2 * x0;  py[2] = 2 * y0;  pr[2] = 2 * r_top;
                    px[3] = 2 * x1;  py[3] = 2 * y1;  pr[3] = 2 * r_top;
                }

                ScalarT aii       = 0;
                ScalarT aip[4]    = {};
                ScalarT aweak     = 0;
                ScalarT aii_ref   = 0;
                ScalarT aip_ref[4] = {};
                ScalarT aweak_ref = 0;

                // Iterate over ALL fine hexes containing this node.
                for ( int dhx = -1; dhx <= 0; dhx++ )
                for ( int dhy = -1; dhy <= 0; dhy++ )
                for ( int dhr = -1; dhr <= 0; dhr++ )
                {
                    int hx = x + dhx;
                    int hy = y + dhy;
                    int hr = r + dhr;
                    if ( hx < 0 || hx >= ncx || hy < 0 || hy >= ncy || hr < 0 || hr >= ncr )
                        continue;

                    int dx = x - hx;
                    int dy = y - hy;
                    int dr = r - hr;

                    for ( int w = 0; w < 2; w++ )
                    {
                        int lidx = local_index_in_wedge( dx, dy, dr, w );
                        if ( lidx < 0 )
                            continue;

                        auto A = fine_op.get_local_matrix( sd, hx, hy, hr, w );

                        // Diagonal-proxy: normalize off-diagonals by this element's diagonal.
                        ScalarT diag     = A( lidx, lidx );
                        ScalarT inv_diag = ( Kokkos::abs( diag ) > ScalarT( 1e-15 ) )
                                               ? ScalarT( 1 ) / diag
                                               : ScalarT( 0 );

                        // Get vertex coords for this wedge.
                        dense::Vec< int, 4 > hex_idx = { sd, hx, hy, hr };
                        dense::Vec< int, 4 > verts[6];
                        if ( w == 0 )
                        {
                            verts[0] = hex_idx;
                            verts[1] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 0 } );
                            verts[2] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 0 } );
                            verts[3] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 0, 1 } );
                            verts[4] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 1 } );
                            verts[5] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 1 } );
                        }
                        else
                        {
                            verts[0] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 1, 0 } );
                            verts[1] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 0 } );
                            verts[2] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 0 } );
                            verts[3] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 1, 1 } );
                            verts[4] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 1 } );
                            verts[5] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 1 } );
                        }

                        aii     += A( lidx, lidx );
                        aii_ref += ScalarT( 1 );  // diagonal-normalized diagonal = A(i,i)/A(i,i) = 1
                        for ( int col = 0; col < 6; col++ )
                        {
                            if ( col == lidx )
                                continue;

                            ScalarT a_val     = A( lidx, col );
                            ScalarT a_val_ref = a_val * inv_diag;

                            bool is_parent = false;
                            for ( int p = 0; p < num_parents; p++ )
                            {
                                if ( verts[col]( 1 ) == px[p] && verts[col]( 2 ) == py[p] &&
                                     verts[col]( 3 ) == pr[p] )
                                {
                                    aip[p]     += a_val;
                                    aip_ref[p] += a_val_ref;
                                    is_parent = true;
                                    break;
                                }
                            }
                            if ( !is_parent )
                            {
                                aweak     += a_val;
                                aweak_ref += a_val_ref;
                            }
                        }
                    }
                }

                a_ip_0( sd, x, y, r ) = aip[0];
                a_ip_1( sd, x, y, r ) = aip[1];
                a_ip_2( sd, x, y, r ) = aip[2];
                a_ip_3( sd, x, y, r ) = aip[3];
                a_denom( sd, x, y, r ) = aii + aweak;

                a_ip_ref_0( sd, x, y, r ) = aip_ref[0];
                a_ip_ref_1( sd, x, y, r ) = aip_ref[1];
                a_ip_ref_2( sd, x, y, r ) = aip_ref[2];
                a_ip_ref_3( sd, x, y, r ) = aip_ref[3];
                a_denom_ref( sd, x, y, r ) = aii_ref + aweak_ref;
            } );

        Kokkos::fence();

        // Phase 2: Communicate across subdomain boundaries (SUM reduction).
        communication::shell::send_recv( domain, a_ip_0, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_ip_1, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_ip_2, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_ip_3, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_denom, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_ip_ref_0, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_ip_ref_1, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_ip_ref_2, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_ip_ref_3, communication::CommunicationReduction::SUM );
        communication::shell::send_recv( domain, a_denom_ref, communication::CommunicationReduction::SUM );

        // Phase 3: Compute ratio-corrected weights using diagonal proxy.
        // w_amg = -a_ip / (a_ii + a_weak): standard AMG weights (biased by geometry).
        // w_ref = -a_ip_ref / (a_ii_ref + a_weak_ref): diagonal-normalized AMG weights
        //   (geometry removed per-element, approximates k=1 weights).
        // correction = w_amg / w_ref: isolates coefficient effect.
        // w_final = w_lin * correction: applies coefficient effect to geometric weights.
        Kokkos::parallel_for(
            "amg_weights_phase3",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sd, nx, ny, nr } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                const bool x_even = ( x % 2 == 0 );
                const bool y_even = ( y % 2 == 0 );
                const bool r_even = ( r % 2 == 0 );
                if ( x_even && y_even && r_even )
                    return;

                const int num_parents = ( x_even && y_even ) ? 2 : 4;

                // Compute standard AMG weights.
                ScalarT denom     = a_denom( sd, x, y, r );
                ScalarT denom_ref = a_denom_ref( sd, x, y, r );

                ScalarT w_amg[4], w_ref[4];
                if ( denom != ScalarT( 0 ) )
                {
                    w_amg[0] = -a_ip_0( sd, x, y, r ) / denom;
                    w_amg[1] = -a_ip_1( sd, x, y, r ) / denom;
                    w_amg[2] = ( num_parents > 2 ) ? -a_ip_2( sd, x, y, r ) / denom : ScalarT( 0 );
                    w_amg[3] = ( num_parents > 2 ) ? -a_ip_3( sd, x, y, r ) / denom : ScalarT( 0 );
                }
                else
                {
                    for ( int p = 0; p < 4; p++ ) w_amg[p] = ScalarT( 0 );
                }

                if ( denom_ref != ScalarT( 0 ) )
                {
                    w_ref[0] = -a_ip_ref_0( sd, x, y, r ) / denom_ref;
                    w_ref[1] = -a_ip_ref_1( sd, x, y, r ) / denom_ref;
                    w_ref[2] = ( num_parents > 2 ) ? -a_ip_ref_2( sd, x, y, r ) / denom_ref : ScalarT( 0 );
                    w_ref[3] = ( num_parents > 2 ) ? -a_ip_ref_3( sd, x, y, r ) / denom_ref : ScalarT( 0 );
                }
                else
                {
                    for ( int p = 0; p < 4; p++ ) w_ref[p] = ScalarT( 0 );
                }

                // Compute geometric linear weights.
                int r_bot = r / 2;
                ScalarT w_lin[4];

                if ( num_parents == 2 )
                {
                    auto weights_lin = fe::wedge::shell::prolongation_linear_weights(
                        dense::Vec< int, 4 >{ sd, x, y, r },
                        dense::Vec< int, 4 >{ sd, x / 2, y / 2, r_bot },
                        grid_fine, radii_fine );
                    w_lin[0] = weights_lin( 0 );
                    w_lin[1] = weights_lin( 1 );
                    w_lin[2] = ScalarT( 0 );
                    w_lin[3] = ScalarT( 0 );
                }
                else
                {
                    int x0, y0, x1, y1;
                    if ( x_even )
                    {
                        x0 = x / 2;  x1 = x / 2;
                        y0 = y / 2;  y1 = y / 2 + 1;
                    }
                    else if ( y_even )
                    {
                        x0 = x / 2;  x1 = x / 2 + 1;
                        y0 = y / 2;  y1 = y / 2;
                    }
                    else
                    {
                        x0 = x / 2 + 1;  x1 = x / 2;
                        y0 = y / 2;      y1 = y / 2 + 1;
                    }
                    auto weights_lin = fe::wedge::shell::prolongation_linear_weights(
                        dense::Vec< int, 4 >{ sd, x, y, r },
                        dense::Vec< int, 4 >{ sd, x0, y0, r_bot },
                        dense::Vec< int, 4 >{ sd, x1, y1, r_bot },
                        grid_fine, radii_fine );
                    w_lin[0] = weights_lin( 0 );  // bot_0
                    w_lin[1] = weights_lin( 0 );  // bot_1 (same radial layer)
                    w_lin[2] = weights_lin( 1 );  // top_0
                    w_lin[3] = weights_lin( 1 );  // top_1
                }

                // Apply ratio correction: w_final = w_lin * (w_amg / w_ref).
                // Per-weight fallback: if w_ref is near zero, correction = 1 (use w_lin).
                constexpr ScalarT eps = ScalarT( 1e-12 );
                ScalarT w[4];
                for ( int p = 0; p < num_parents; p++ )
                {
                    ScalarT correction = ( Kokkos::abs( w_ref[p] ) > eps )
                                             ? w_amg[p] / w_ref[p]
                                             : ScalarT( 1 );
                    w[p] = w_lin[p] * correction;
                }
                for ( int p = num_parents; p < 4; p++ )
                    w[p] = ScalarT( 0 );

                // Clamp negative weights to zero and rescale.
                ScalarT sum = 0;
                for ( int p = 0; p < num_parents; p++ )
                {
                    if ( w[p] < ScalarT( 0 ) )
                        w[p] = 0;
                    sum += w[p];
                }
                if ( sum > ScalarT( 0 ) )
                {
                    for ( int p = 0; p < num_parents; p++ )
                        w[p] /= sum;
                }

                // Limit extreme weight ratios: clamp each weight to [w_min_clamp, 1].
                constexpr ScalarT w_min_clamp = 0.05;
                bool              needs_reclamp = false;
                for ( int p = 0; p < num_parents; p++ )
                {
                    if ( w[p] > ScalarT( 0 ) && w[p] < w_min_clamp )
                    {
                        w[p]          = w_min_clamp;
                        needs_reclamp = true;
                    }
                }
                if ( needs_reclamp )
                {
                    sum = 0;
                    for ( int p = 0; p < num_parents; p++ )
                        sum += w[p];
                    if ( sum > ScalarT( 0 ) )
                    {
                        for ( int p = 0; p < num_parents; p++ )
                            w[p] /= sum;
                    }
                }

                pw0( sd, x, y, r ) = w[0];
                pw1( sd, x, y, r ) = w[1];
                pw2( sd, x, y, r ) = w[2];
                pw3( sd, x, y, r ) = w[3];
            } );

        Kokkos::fence();
    }

    /// @brief Computes indices of vertices associated to a wedge in a hex cell.
    /// @param coarse_hex_idx  [in] global index of the hex cell
    /// @param wedge  [in] wedge index (local index 0 or 1)
    /// @param wedge_local_vertex_indices  [out] global indices of the vertices located on the wedge
    KOKKOS_INLINE_FUNCTION void wedge_vertex_indices(
        dense::Vec< int, 4 > hex_idx,
        int                  wedge,
        dense::Vec< int, 4 > ( &wedge_local_vertex_indices )[6] ) const
    {
        if ( wedge == 0 )
        {
            wedge_local_vertex_indices[0] = hex_idx;
            wedge_local_vertex_indices[1] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 0 } );
            wedge_local_vertex_indices[2] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 0 } );
            wedge_local_vertex_indices[3] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 0, 1 } );
            wedge_local_vertex_indices[4] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 1 } );
            wedge_local_vertex_indices[5] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 1 } );
        }
        else
        {
            wedge_local_vertex_indices[0] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 1, 0 } );
            wedge_local_vertex_indices[1] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 0 } );
            wedge_local_vertex_indices[2] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 0 } );
            wedge_local_vertex_indices[3] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 1, 1 } );
            wedge_local_vertex_indices[4] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 1 } );
            wedge_local_vertex_indices[5] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 1 } );
        }
    }

    /// @brief Finds the local index (0-5) of a node within a wedge given its relative position in the hex.
    /// @param dx, dy, dr: relative position of the node within the hex (0 or 1 each)
    /// @param wedge: wedge index (0 or 1)
    /// @return local index 0-5, or -1 if the node is not in this wedge
    KOKKOS_INLINE_FUNCTION
    static int local_index_in_wedge( int dx, int dy, int dr, int wedge )
    {
        if ( wedge == 0 )
        {
            if ( dx + dy > 1 )
                return -1;
            int base = dr * 3;
            if ( dx == 0 && dy == 0 )
                return base;
            if ( dx == 1 && dy == 0 )
                return base + 1;
            if ( dx == 0 && dy == 1 )
                return base + 2;
            return -1;
        }
        else
        {
            if ( dx + dy < 1 )
                return -1;
            int base = dr * 3;
            if ( dx == 1 && dy == 1 )
                return base;
            if ( dx == 0 && dy == 1 )
                return base + 1;
            if ( dx == 1 && dy == 0 )
                return base + 2;
            return -1;
        }
    }

    /// @brief Computes AMG-style interpolation weights at a fine node.
    /// Assembles the stiffness row at the fine node from all surrounding wedges
    /// within the coarse hex and computes weights as w_j = -a_ij / (a_ii + sum_weak).
    /// Note: contributions from wedges in adjacent coarse hexes are not included;
    /// they are implicitly lumped into the modified diagonal.
    KOKKOS_INLINE_FUNCTION
    void compute_amg_weights_at_node(
        int                                        local_subdomain_id,
        dense::Vec< int, 4 >                       fine_node_idx,
        dense::Vec< int, 4 >                       coarse_hex_base_fine,
        int                                         num_parents,
        const dense::Vec< int, 4 > ( &parents )[4],
        ScalarT ( &weights )[4] ) const
    {
        ScalarT a_ii    = 0;
        ScalarT a_ij[4] = {};
        ScalarT a_weak  = 0;

        dense::Vec< int, 4 > fhs[8] = {
            { 0, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 1, 1, 0 },
            { 0, 0, 0, 1 }, { 0, 1, 0, 1 }, { 0, 0, 1, 1 }, { 0, 1, 1, 1 },
        };

        for ( int fh = 0; fh < 8; fh++ )
        {
            auto fhi = coarse_hex_base_fine + fhs[fh];
            int  dx  = fine_node_idx( 1 ) - fhi( 1 );
            int  dy  = fine_node_idx( 2 ) - fhi( 2 );
            int  dr  = fine_node_idx( 3 ) - fhi( 3 );
            if ( dx < 0 || dx > 1 || dy < 0 || dy > 1 || dr < 0 || dr > 1 )
                continue;

            for ( int w = 0; w < 2; w++ )
            {
                int lidx = local_index_in_wedge( dx, dy, dr, w );
                if ( lidx < 0 )
                    continue;

                auto A = fine_op_.get_local_matrix( local_subdomain_id, fhi( 1 ), fhi( 2 ), fhi( 3 ), w );

                dense::Vec< int, 4 > verts[6];
                wedge_vertex_indices( fhi, w, verts );

                a_ii += A( lidx, lidx );
                for ( int col = 0; col < num_nodes_per_wedge; col++ )
                {
                    if ( col == lidx )
                        continue;

                    bool found = false;
                    for ( int p = 0; p < num_parents; p++ )
                    {
                        if ( verts[col]( 1 ) == parents[p]( 1 ) && verts[col]( 2 ) == parents[p]( 2 ) &&
                             verts[col]( 3 ) == parents[p]( 3 ) )
                        {
                            a_ij[p] += A( lidx, col );
                            found = true;
                            break;
                        }
                    }
                    if ( !found )
                    {
                        a_weak += A( lidx, col );
                    }
                }
            }
        }

        ScalarT denom = a_ii + a_weak;
        for ( int p = 0; p < num_parents; p++ )
        {
            weights[p] = ( denom != ScalarT( 0 ) ) ? -a_ij[p] / denom : ScalarT( 0 );
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(
        const int local_subdomain_id,
        const int x_coarse_idx,
        const int y_coarse_idx,
        const int r_coarse_idx ) const
    {
        int x_cell_coarsest = map_to_coarse_element( x_coarse_idx, level_range_ );
        int y_cell_coarsest = map_to_coarse_element( y_coarse_idx, level_range_ );
        int r_cell_coarsest = map_to_coarse_element( r_coarse_idx, level_range_ );
        if ( GCAElements_( local_subdomain_id, x_cell_coarsest, y_cell_coarsest, r_cell_coarsest ) > 0 )
        {
            dense::Vec< int, 4 > fine_hex_shifts[8] = {
                { 0, 0, 0, 0 },
                { 0, 1, 0, 0 },
                { 0, 0, 1, 0 },
                { 0, 1, 1, 0 },
                { 0, 0, 0, 1 },
                { 0, 1, 0, 1 },
                { 0, 0, 1, 1 },
                { 0, 1, 1, 1 },
            };

            dense::Vec< int, 4 > coarse_hex_idx = { local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx };
            (void) coarse_hex_idx; // unused

            dense::Vec< int, 4 > coarse_hex_idx_fine = {
                local_subdomain_id, 2 * x_coarse_idx, 2 * y_coarse_idx, 2 * r_coarse_idx };

            dense::Mat< ScalarT, Operator::LocalMatrixDim, Operator::LocalMatrixDim >
                A_coarse[num_wedges_per_hex_cell] = {};
            A_coarse[0].fill( 0 );
            A_coarse[1].fill( 0 );
            // loop finer hexes of our coarse hex
            for ( int fine_hex_lidx = 0; fine_hex_lidx < 8; fine_hex_lidx++ )
            {
                auto fine_hex_idx = coarse_hex_idx_fine + fine_hex_shifts[fine_hex_lidx];

                // two wedges per fine hex
                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    dense::Mat< ScalarT, 6, 6 > P = {};

                    // obtain vertex indices of the current fine wedge
                    dense::Vec< int, 4 > wedge_local_vertex_indices_fine[6];
                    wedge_vertex_indices( fine_hex_idx, wedge, wedge_local_vertex_indices_fine );

                    // compute local (fully-assembled!) interpolation matrices mapping from the coarse DoFs in the hex to the current fine wedge DoFs

                    // loop destination of the interpolation (row dim of P): fine DoFs
                    for ( int fine_dof_lidx = 0; fine_dof_lidx < num_nodes_per_wedge; fine_dof_lidx++ )
                    {
                        auto fine_dof_idx = wedge_local_vertex_indices_fine[fine_dof_lidx];

                        // fine dof is on coarse dof
                        if ( fine_dof_idx( 1 ) % 2 == 0 && fine_dof_idx( 2 ) % 2 == 0 && fine_dof_idx( 3 ) % 2 == 0 )
                        {
                            // local index of destination fine DoF == local index of source coarse DoF
                            P( fine_dof_lidx, fine_dof_lidx ) = 1.0;
                            continue;
                        }

                        // else: need radial direction bot (>=) and top (<=) of current fine DoF
                        int r_idx_coarse_bot = fine_dof_idx( 3 ) < radii_fine_.extent( 1 ) - 1 ?
                                                   fine_dof_idx( 3 ) / 2 :
                                                   fine_dof_idx( 3 ) / 2 - 1;
                        int r_idx_coarse_top = r_idx_coarse_bot + 1;
                        (void) r_idx_coarse_top; // unused

                        // fine dof is radially aligned: x and y index match with coarse DoFs
                        // interpolate on the line in radial direction (coarse DoF bot -- fine DoF -- coarse DoF top)
                        if ( fine_dof_idx( 1 ) % 2 == 0 && fine_dof_idx( 2 ) % 2 == 0 )
                        {
                            // x, y on coarse, so we can just divide by 2 to obtain coarse indices
                            const auto fine_dof_x_idx_coarse = fine_dof_idx( 1 ) / 2;
                            const auto fine_dof_y_idx_coarse = fine_dof_idx( 2 ) / 2;

                            dense::Vec< ScalarType, 2 > weights{};
                            if ( interpolation_mode_ == InterpolationMode::Linear )
                            {
                                // actual weight computation
                                weights = fe::wedge::shell::prolongation_linear_weights(
                                    dense::Vec< int, 4 >{
                                        local_subdomain_id, fine_dof_idx( 1 ), fine_dof_idx( 2 ), fine_dof_idx( 3 ) },
                                    dense::Vec< int, 4 >{
                                        local_subdomain_id,
                                        fine_dof_x_idx_coarse,
                                        fine_dof_y_idx_coarse,
                                        r_idx_coarse_bot },
                                    grid_fine_,
                                    radii_fine_ );
                            }
                            else if ( interpolation_mode_ == InterpolationMode::Constant )
                            {
                                weights( 0 ) = 0.5;
                                weights( 1 ) = 0.5;
                            }
                            else if ( interpolation_mode_ == InterpolationMode::OperatorDependent )
                            {
                                weights( 0 ) = parent_weight_0_( local_subdomain_id, fine_dof_idx( 1 ),
                                                                  fine_dof_idx( 2 ), fine_dof_idx( 3 ) );
                                weights( 1 ) = parent_weight_1_( local_subdomain_id, fine_dof_idx( 1 ),
                                                                  fine_dof_idx( 2 ), fine_dof_idx( 3 ) );
                            }
                            else
                            {
                                Kokkos::abort( "Unknown interpolation mode." );
                            }

                            if ( fine_dof_lidx == 2 or fine_dof_lidx == 5 )
                            {
                                P( fine_dof_lidx, 2 ) = weights( 0 );
                                P( fine_dof_lidx, 5 ) = weights( 1 );
                            }
                            else if ( fine_dof_lidx == 0 or fine_dof_lidx == 3 )
                            {
                                P( fine_dof_lidx, 0 ) = weights( 0 );
                                P( fine_dof_lidx, 3 ) = weights( 1 );
                            }
                            else if ( fine_dof_lidx == 1 or fine_dof_lidx == 4 )
                            {
                                P( fine_dof_lidx, 1 ) = weights( 0 );
                                P( fine_dof_lidx, 4 ) = weights( 1 );
                            }
                            continue;
                        }

                        // else: we interpolate fine DoF from the plane of 4 coarse DoFs that contains the fine DoF

                        // for the two botting coarse DoFs
                        int x0_idx_coarse = -1;
                        int x1_idx_coarse = -1;
                        int y0_idx_coarse = -1;
                        int y1_idx_coarse = -1;

                        // local indices of the 4 coarse DoFs in the plane
                        int coarse_dof_lindices[4] = { -1 };

                        if ( fine_dof_idx( 1 ) % 2 == 0 )
                        {
                            // "Vertical" edge.
                            x0_idx_coarse = fine_dof_idx( 1 ) / 2;
                            x1_idx_coarse = fine_dof_idx( 1 ) / 2;

                            y0_idx_coarse = fine_dof_idx( 2 ) / 2;
                            y1_idx_coarse = fine_dof_idx( 2 ) / 2 + 1;

                            coarse_dof_lindices[0] = 0;
                            coarse_dof_lindices[1] = 2;
                            coarse_dof_lindices[2] = 3;
                            coarse_dof_lindices[3] = 5;
                        }
                        else if ( fine_dof_idx( 2 ) % 2 == 0 )
                        {
                            // "Horizontal" edge.
                            x0_idx_coarse = fine_dof_idx( 1 ) / 2;
                            x1_idx_coarse = fine_dof_idx( 1 ) / 2 + 1;

                            y0_idx_coarse = fine_dof_idx( 2 ) / 2;
                            y1_idx_coarse = fine_dof_idx( 2 ) / 2;

                            coarse_dof_lindices[0] = 0;
                            coarse_dof_lindices[1] = 1;
                            coarse_dof_lindices[2] = 3;
                            coarse_dof_lindices[3] = 4;
                        }
                        else
                        {
                            // "Diagonal" edge.
                            x0_idx_coarse = fine_dof_idx( 1 ) / 2 + 1;
                            x1_idx_coarse = fine_dof_idx( 1 ) / 2;

                            y0_idx_coarse = fine_dof_idx( 2 ) / 2;
                            y1_idx_coarse = fine_dof_idx( 2 ) / 2 + 1;

                            coarse_dof_lindices[0] = 1;
                            coarse_dof_lindices[1] = 2;
                            coarse_dof_lindices[2] = 4;
                            coarse_dof_lindices[3] = 5;
                        }

                        if ( interpolation_mode_ == InterpolationMode::Linear )
                        {
                            const auto weights = fe::wedge::shell::prolongation_linear_weights(
                                dense::Vec< int, 4 >{
                                    local_subdomain_id, fine_dof_idx( 1 ), fine_dof_idx( 2 ), fine_dof_idx( 3 ) },
                                dense::Vec< int, 4 >{
                                    local_subdomain_id, x0_idx_coarse, y0_idx_coarse, r_idx_coarse_bot },
                                dense::Vec< int, 4 >{
                                    local_subdomain_id, x1_idx_coarse, y1_idx_coarse, r_idx_coarse_bot },
                                grid_fine_,
                                radii_fine_ );

                            P( fine_dof_lidx, coarse_dof_lindices[0] ) = weights( 0 );
                            P( fine_dof_lidx, coarse_dof_lindices[1] ) = weights( 0 );
                            P( fine_dof_lidx, coarse_dof_lindices[2] ) = weights( 1 );
                            P( fine_dof_lidx, coarse_dof_lindices[3] ) = weights( 1 );
                        }
                        else if ( interpolation_mode_ == InterpolationMode::Constant )
                        {
                            P( fine_dof_lidx, coarse_dof_lindices[0] ) =
                                terra::fe::wedge::shell::prolongation_constant_weight< ScalarType >(
                                    fine_dof_idx( 1 ),
                                    fine_dof_idx( 2 ),
                                    fine_dof_idx( 3 ),
                                    x0_idx_coarse,
                                    y0_idx_coarse,
                                    r_idx_coarse_bot );
                            P( fine_dof_lidx, coarse_dof_lindices[1] ) =
                                terra::fe::wedge::shell::prolongation_constant_weight< ScalarType >(
                                    fine_dof_idx( 1 ),
                                    fine_dof_idx( 2 ),
                                    fine_dof_idx( 3 ),
                                    x1_idx_coarse,
                                    y1_idx_coarse,
                                    r_idx_coarse_bot );
                            P( fine_dof_lidx, coarse_dof_lindices[2] ) =
                                terra::fe::wedge::shell::prolongation_constant_weight< ScalarType >(
                                    fine_dof_idx( 1 ),
                                    fine_dof_idx( 2 ),
                                    fine_dof_idx( 3 ),
                                    x0_idx_coarse,
                                    y0_idx_coarse,
                                    r_idx_coarse_top );
                            P( fine_dof_lidx, coarse_dof_lindices[3] ) =
                                terra::fe::wedge::shell::prolongation_constant_weight< ScalarType >(
                                    fine_dof_idx( 1 ),
                                    fine_dof_idx( 2 ),
                                    fine_dof_idx( 3 ),
                                    x1_idx_coarse,
                                    y1_idx_coarse,
                                    r_idx_coarse_top );
                        }
                        else if ( interpolation_mode_ == InterpolationMode::OperatorDependent )
                        {
                            P( fine_dof_lidx, coarse_dof_lindices[0] ) = parent_weight_0_(
                                local_subdomain_id, fine_dof_idx( 1 ), fine_dof_idx( 2 ), fine_dof_idx( 3 ) );
                            P( fine_dof_lidx, coarse_dof_lindices[1] ) = parent_weight_1_(
                                local_subdomain_id, fine_dof_idx( 1 ), fine_dof_idx( 2 ), fine_dof_idx( 3 ) );
                            P( fine_dof_lidx, coarse_dof_lindices[2] ) = parent_weight_2_(
                                local_subdomain_id, fine_dof_idx( 1 ), fine_dof_idx( 2 ), fine_dof_idx( 3 ) );
                            P( fine_dof_lidx, coarse_dof_lindices[3] ) = parent_weight_3_(
                                local_subdomain_id, fine_dof_idx( 1 ), fine_dof_idx( 2 ), fine_dof_idx( 3 ) );
                        }
                        else
                        {
                            Kokkos::abort( "Unknown interpolation mode." );
                        }
                    }

                    dense::Mat< ScalarT, Operator::LocalMatrixDim, Operator::LocalMatrixDim > A_fine =
                        fine_op_.get_local_matrix(
                            local_subdomain_id, fine_hex_idx( 1 ), fine_hex_idx( 2 ), fine_hex_idx( 3 ), wedge );

                    // core part: assemble local gca matrix by mapping from coarse wedge to current fine wedge,
                    // applying the corresponding local operator and mapping back.
                    dense::Mat< ScalarT, Operator::LocalMatrixDim, Operator::LocalMatrixDim > P_vec = { 0 };
                    if constexpr ( Operator::LocalMatrixDim == 18 )
                    {
                        // in a vectorial operator we need to setup a vectorial interpolation
                        for ( int dim = 0; dim < 3; ++dim )
                        {
                            for ( int i = 0; i < 6; ++i )
                            {
                                for ( int j = 0; j < 6; ++j )
                                {
                                    P_vec( i + dim * num_nodes_per_wedge, j + dim * num_nodes_per_wedge ) = P( i, j );
                                }
                            }
                        }
                    }
                    else
                    {
                        // for scalar operators we just use the scalar interpolation
                        P_vec = P;
                    }
                    dense::Mat< ScalarT, Operator::LocalMatrixDim, Operator::LocalMatrixDim > PTAP =
                        P_vec.transposed() * A_fine * P_vec;

                    // correctly add to gca coarsened matrix
                    // depending on the fine hex and wedge, we are located on the coarse 0 or 1 wedge
                    // and need to add to the corresponding coarse matrix
                    if ( ( wedge == 0 && ( fine_hex_lidx == 0 || fine_hex_lidx == 1 || fine_hex_lidx == 2 ||
                                           fine_hex_lidx == 4 || fine_hex_lidx == 5 || fine_hex_lidx == 6 ) ) or
                         ( wedge == 1 && ( fine_hex_lidx == 0 || fine_hex_lidx == 4 ) ) )
                    {
                        A_coarse[0] += PTAP;
                    }
                    else if (
                        ( wedge == 1 && ( fine_hex_lidx == 1 || fine_hex_lidx == 2 || fine_hex_lidx == 3 ||
                                          fine_hex_lidx == 5 || fine_hex_lidx == 6 || fine_hex_lidx == 7 ) ) or
                        ( wedge == 0 && ( fine_hex_lidx == 3 || fine_hex_lidx == 7 ) ) )
                    {
                        A_coarse[1] += PTAP;
                    }
                    else
                    {
                        Kokkos::abort( "Unexpected path." );
                    }
                }
            }

            // bc treatment moved to ops, will be revisited during freeslip impl
            if ( false )
            {
                dense::Mat< ScalarT, Operator::LocalMatrixDim, Operator::LocalMatrixDim > boundary_mask;
                boundary_mask.fill( 1.0 );

                if constexpr ( Operator::LocalMatrixDim == 18 )
                {
                    for ( int dimi = 0; dimi < 3; ++dimi )
                    {
                        for ( int dimj = 0; dimj < 3; ++dimj )
                        {
                            if ( coarse_op_.has_flag(
                                     local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx, CMB ) )
                            {
                                // Inner boundary (CMB).
                                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                                {
                                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                                    {
                                        if ( ( dimi == dimj && i != j && ( i < 3 || j < 3 ) ) or
                                             ( dimi != dimj && ( i < 3 || j < 3 ) ) )
                                        {
                                            boundary_mask(
                                                i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) = 0.0;
                                        }
                                    }
                                }
                            }

                            if ( coarse_op_.has_flag(
                                     local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx + 1, SURFACE ) )
                            {
                                // Outer boundary (surface).
                                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                                {
                                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                                    {
                                        if ( ( dimi == dimj && i != j && ( i >= 3 || j >= 3 ) ) or
                                             ( dimi != dimj && ( i >= 3 || j >= 3 ) ) )
                                        {
                                            boundary_mask(
                                                i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) = 0.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if ( coarse_op_.has_flag( local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx, CMB ) )
                    {
                        // Inner boundary (CMB).
                        for ( int i = 0; i < num_nodes_per_wedge; i++ )
                        {
                            for ( int j = 0; j < num_nodes_per_wedge; j++ )
                            {
                                if ( i != j && ( i < 3 || j < 3 ) )
                                {
                                    boundary_mask( i, j ) = 0.0;
                                }
                            }
                        }
                    }

                    if ( coarse_op_.has_flag(
                             local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx + 1, SURFACE ) )
                    {
                        // Outer boundary (surface).
                        for ( int i = 0; i < num_nodes_per_wedge; i++ )
                        {
                            for ( int j = 0; j < num_nodes_per_wedge; j++ )
                            {
                                if ( i != j && ( i >= 3 || j >= 3 ) )
                                {
                                    boundary_mask( i, j ) = 0.0;
                                }
                            }
                        }
                    }
                }
                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    A_coarse[wedge].hadamard_product( boundary_mask );
                }
            }

            // store coarse matrices
            coarse_op_.set_local_matrix( local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx, 0, A_coarse[0] );
            coarse_op_.set_local_matrix( local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx, 1, A_coarse[1] );
        }
    }
};
} // namespace terra::linalg::solvers