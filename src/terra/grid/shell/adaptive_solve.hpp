#pragma once

// Solve-side glue for AMR: adaptive ownership + boundary masks and the constrained operator wrapper.
//
// The wrapper composes a COMMUNICATION-FREE local operator A_loc (any terra shell operator constructed
// with OperatorCommunicationMode::SkipCommunication) with the AMR tables into the constrained operator
//
//     A_c = C^T A_loc C :   x -> constrain(x)  ->  y = A_loc x  ->  exchange(y)
//
// where C interpolates hanging DoFs from their coarse parents (apply_constraint_device) and C^T is the
// assembly exchange (class sums + hanging P^T scatter + broadcast, apply_exchange_device). The result
// satisfies linalg::OperatorLike, so terra's Krylov solvers use it unchanged. Genuine-DoF dot products
// come from the adaptive ownership mask (exactly one OWNED copy per physical node; hanging copies are
// never owned).
//
// The uniform mask setups in grid/bit_masks.hpp cannot be used on adaptive domains (they communicate
// over the uniform neighborhood, which is empty here, and key radial boundaries on uniform subdomain
// indices) -- the two functions below are their adaptive replacements.

#include "../../kernels/common/grid_operations.hpp"
#include "../../linalg/operator.hpp"
#include "../bit_masks.hpp"
#include "adaptive_2to1_kokkos.hpp"
#include "bit_masks.hpp"

namespace terra::grid::shell::amr {

// Exactly one OWNED copy per genuine physical node: canonical class members + genuine singletons.
// Non-canonical class members and hanging copies are not owned.
inline Grid4DDataScalar< grid::NodeOwnershipFlag >
    adaptive_node_ownership_mask( const DistributedDomain& dom, const TwoToOneTables& t )
{
    auto mask = allocate_scalar_grid< grid::NodeOwnershipFlag >( "amr_ownership_mask", dom );
    auto h    = Kokkos::create_mirror_view( mask );
    for ( std::size_t s = 0; s < mask.extent( 0 ); ++s )
        for ( std::size_t x = 0; x < mask.extent( 1 ); ++x )
            for ( std::size_t y = 0; y < mask.extent( 2 ); ++y )
                for ( std::size_t r = 0; r < mask.extent( 3 ); ++r )
                    h( s, x, y, r ) = grid::NodeOwnershipFlag::OWNED;
    for ( std::size_t c = 0; c + 1 < t.cls_offsets.size(); ++c )
        for ( int m = t.cls_offsets[c] + 1; m < t.cls_offsets[c + 1]; ++m ) // non-canonical members
        {
            const Idx4& i        = t.cls_members[m];
            h( i.s, i.x, i.y, i.r ) = grid::NodeOwnershipFlag::NO_FLAG;
        }
    for ( const auto& d : t.con_dst ) // hanging copies
        h( d.s, d.x, d.y, d.r ) = grid::NodeOwnershipFlag::NO_FLAG;
    Kokkos::deep_copy( mask, h );
    return mask;
}

// CMB/SURFACE flags from each block's position and subdivision: a block touches the CMB iff its
// finest-frame radial anchor is 0, and the surface iff anchor + span reaches the outer shell.
inline Grid4DDataScalar< ShellBoundaryFlag > adaptive_boundary_mask( const DistributedDomain& dom )
{
    auto mask = allocate_scalar_grid< ShellBoundaryFlag >( "amr_boundary_mask", dom );
    auto h    = Kokkos::create_mirror_view( mask );
    for ( std::size_t s = 0; s < mask.extent( 0 ); ++s )
        for ( std::size_t x = 0; x < mask.extent( 1 ); ++x )
            for ( std::size_t y = 0; y < mask.extent( 2 ); ++y )
                for ( std::size_t r = 0; r < mask.extent( 3 ); ++r )
                    h( s, x, y, r ) = ShellBoundaryFlag::INNER;

    const int M          = dom.max_subdivision();
    const int nr         = static_cast< int >( mask.extent( 3 ) );
    const int radial_ext = dom.domain_info().num_subdomains_in_radial_direction() << M;
    for ( const auto& [anchor, tup] : dom.subdomains() )
    {
        const int sub  = std::get< 0 >( tup );
        const int k    = dom.subdivision_of( anchor );
        const int span = 1 << ( M - k );
        if ( anchor.subdomain_r() == 0 )
            for ( std::size_t x = 0; x < mask.extent( 1 ); ++x )
                for ( std::size_t y = 0; y < mask.extent( 2 ); ++y )
                    h( sub, x, y, 0 ) = ShellBoundaryFlag::CMB;
        if ( anchor.subdomain_r() + span == radial_ext )
            for ( std::size_t x = 0; x < mask.extent( 1 ); ++x )
                for ( std::size_t y = 0; y < mask.extent( 2 ); ++y )
                    h( sub, x, y, nr - 1 ) = ShellBoundaryFlag::SURFACE;
    }
    Kokkos::deep_copy( mask, h );
    return mask;
}

// The constrained operator C^T A_loc C, optionally with Dirichlet elimination on the ASSEMBLED-
// CONSTRAINED system.
//
// LocalOp must be constructed with SkipCommunication -- its apply must be the pure block-local element
// loop; the tables provide the assembly (C^T) and the hanging-node interpolation (C).
//
// WHY the elimination lives HERE and not (only) in the operator: kerngen's element kernels do treat
// Dirichlet correctly via the node mask flags (adaptive-safe). But when a hanging node h has a
// Dirichlet node P among its constraint parents, the folding C^T A C REGENERATES a coupling
// A_sys(i,P) += A(i,h) * w that element-level elimination cannot see (h is unflagged) -- while the
// standard RHS lifting subtracts the same coupling, producing an O(1) inconsistency in every interior
// row adjacent to boundary-parented hanging nodes (measured: 2x larger l2 error on a boundary-touching
// refined block). Dirichlet on constrained AMR systems must therefore be imposed POST-folding:
//     A_sys = P (C^T A_loc C) P + I_boundary,   b = P (b_0 - C^T A_loc C g) + I_boundary g,
// with P zeroing the flagged DoFs. This is symmetric, and on hanging-free meshes it agrees with the
// element-level treatment to solver accuracy (verified against the uniform pipeline on H100).
// Pass no mask for the plain constrained operator (e.g. the mass, or A applied to the lift g).
// `tmp` is scratch with the same layout as the operator's vectors.
template < linalg::OperatorLike LocalOp >
class AdaptiveConstrainedOperator
{
  public:
    using ScalarType    = typename LocalOp::ScalarType;
    using SrcVectorType = typename LocalOp::SrcVectorType;
    using DstVectorType = typename LocalOp::DstVectorType;

    AdaptiveConstrainedOperator( LocalOp& op, const TwoToOneTablesDevice& tables,
                                 const SrcVectorType& tmp )
    : op_( op )
    , t_( tables )
    , tmp_( tmp )
    {}

    // Dirichlet-eliminated variant: DoFs flagged `dirichlet_flag` in `boundary_mask` are projected out
    // of the constrained operator and get identity rows.
    AdaptiveConstrainedOperator( LocalOp& op, const TwoToOneTablesDevice& tables,
                                 const SrcVectorType&                         tmp,
                                 const Grid4DDataScalar< ShellBoundaryFlag >& boundary_mask,
                                 ShellBoundaryFlag                            dirichlet_flag )
    : op_( op )
    , t_( tables )
    , tmp_( tmp )
    , boundary_mask_( boundary_mask )
    , dirichlet_flag_( dirichlet_flag )
    , eliminate_dirichlet_( true )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        amr_deep_copy( tmp_.grid_data(), src.grid_data() );
        if ( eliminate_dirichlet_ ) // P: zero Dirichlet DoFs (before constraining, so hanging DoFs
                                    // with boundary parents interpolate the zeroed values consistently)
            kernels::common::assign_masked_else_keep_old(
                tmp_.grid_data(), ScalarType( 0 ), boundary_mask_, dirichlet_flag_ );
        apply_constraint_device( t_, tmp_.grid_data() ); // conforming input (C)
        linalg::apply( op_, tmp_, dst );                 // block-local element apply (A_loc)
        apply_exchange_device( t_, dst.grid_data() );    // assembly (C^T)
        if ( eliminate_dirichlet_ )                      // identity rows on the Dirichlet DoFs
            kernels::common::assign_masked_else_keep_old(
                dst.grid_data(), src.grid_data(), boundary_mask_, dirichlet_flag_ );
    }

  private:
    LocalOp&                              op_;
    TwoToOneTablesDevice                  t_;
    SrcVectorType                         tmp_;
    Grid4DDataScalar< ShellBoundaryFlag > boundary_mask_{};
    ShellBoundaryFlag                     dirichlet_flag_      = ShellBoundaryFlag::NO_FLAG;
    bool                                  eliminate_dirichlet_ = false;
};

} // namespace terra::grid::shell::amr
