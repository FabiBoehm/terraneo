#pragma once

// Distributed (multi-rank) adaptive assembly -- the surface-halo layer over the local class assembly.
//
// After Stage 0 the constraint and hanging-P^T are fully local to each fine block, so the ONLY thing
// that spans ranks is the genuine node classes (coincident copies across blocks). Those are a
// generalized additive halo: each rank sums its local class members (the existing local assembler),
// then a cross-rank reduce sends partials to the class owner, who broadcasts the assembled value back.
// No ghost blocks; only shared node values move.
//
// The comm plan is derived from the GLOBAL classes: every rank replicates the forest, builds an
// all-on-one-rank domain + its tables to get the global class structure (host, no Kokkos grids), tags
// each member with its owner rank via the partition function, and keeps the cross-rank classes. Owner
// = min member rank, computed identically on every rank (deterministic). The plan itself is MPI-free,
// so it is testable serially by looping over ranks on one process.
//
// This first version stages the assembly through the host (device -> host mirror -> MPI -> device);
// packing only the boundary nodes on-device is a later optimization.

#include <functional>
#include <map>
#include <set>
#include <vector>

#include "adaptive_2to1_kokkos.hpp"
#include "adaptive_exchange.hpp"
#include "adaptive_solve.hpp"
#include "mpi/mpi.hpp"

namespace terra::grid::shell::amr {

// Contiguous buffer of local node addresses exchanged with one partner rank (aligned across ranks
// because both sides iterate the global classes in the same order).
struct RankBuffer
{
    int                 partner;
    std::vector< Idx4 > nodes;
};

struct DistributedPlan
{
    int my_rank = 0, nprocs = 1;

    std::vector< RankBuffer > reduce_send; // I am a non-owner member: send my local partial to owner
    std::vector< RankBuffer > reduce_recv; // I am the owner: receive & ADD a member's partial
    std::vector< RankBuffer > bcast_send;  // I am the owner: send the assembled value to a member
    std::vector< RankBuffer > bcast_recv;  // I am a non-owner member: receive & OVERWRITE

    std::vector< Idx4 > rebroadcast_src;   // after comm, copy the representative ...
    std::vector< Idx4 > rebroadcast_dst;   // ... to my other local members of a cross-rank class

    std::vector< Idx4 > not_owned;         // my local copies NOT owned by me (unmark from the mask):
                                           // every class copy except the min-rank owner's representative
};

using AnchorRankFn = std::function< int( const SubdomainInfo& ) >; // finest anchor -> owner rank

// Build the plan for `my_rank` (MPI-free). `dom_all` is an all-on-one-rank adaptive domain (all leaves)
// and `t_all` its tables -- they carry the global class structure.
inline DistributedPlan build_distributed_plan( const amr::AdaptiveForest& forest,
                                               const DistributedDomain&   dom_all,
                                               const TwoToOneTables&      t_all, const AnchorRankFn& rank_of,
                                               int my_rank, int nprocs )
{
    DistributedPlan plan;
    plan.my_rank = my_rank;
    plan.nprocs  = nprocs;

    // anchor -> (owner rank, local slot in that rank's owned-only domain). The slot order must match
    // create_adaptive_for_rank (same leaf order, same rank filter).
    std::map< SubdomainInfo, std::pair< int, int > > owner_slot;
    std::vector< int >                               slot_counter( nprocs, 0 );
    for ( const auto& leaf : forest.leaves() )
    {
        const SubdomainInfo a = forest.finest_anchor( leaf );
        const int           r = rank_of( a );
        owner_slot[a]         = { r, slot_counter[r]++ };
    }

    // a global-class member (slot in dom_all) -> (rank, local Idx4 in that rank's domain)
    auto to_local = [&]( const Idx4& m ) {
        const SubdomainInfo a  = dom_all.subdomain_info_from_local_idx( m.s );
        const auto&         rs = owner_slot.at( a );
        return std::pair< int, Idx4 >{ rs.first, Idx4{ rs.second, m.x, m.y, m.r } };
    };

    std::map< int, std::vector< Idx4 > > red_send, red_recv, bc_send, bc_recv;

    for ( std::size_t c = 0; c + 1 < t_all.cls_offsets.size(); ++c )
    {
        std::map< int, std::vector< Idx4 > > by_rank; // rank -> its local members (first = representative)
        for ( int m = t_all.cls_offsets[c]; m < t_all.cls_offsets[c + 1]; ++m )
        {
            const auto lr = to_local( t_all.cls_members[m] );
            by_rank[lr.first].push_back( lr.second );
        }
        const int owner = by_rank.begin()->first; // min rank
        if ( by_rank.find( my_rank ) == by_rank.end() )
            continue; // not my class

        // ownership: only the owner's representative is owned; every other copy (mine included when I
        // am not the owner) is unmarked. Applies to LOCAL-ONLY classes too (owner == me, keep rep).
        const auto& mine = by_rank.at( my_rank );
        for ( std::size_t i = ( owner == my_rank ? 1 : 0 ); i < mine.size(); ++i )
            plan.not_owned.push_back( mine[i] );

        if ( by_rank.size() < 2 )
            continue; // local-only class -> no cross-rank comm

        const Idx4 rep = mine.front();
        if ( my_rank == owner )
        {
            for ( const auto& [r, members] : by_rank )
                if ( r != owner )
                {
                    red_recv[r].push_back( rep ); // add each member's partial into my representative
                    bc_send[r].push_back( rep );  // send the assembled representative to each member
                }
        }
        else
        {
            red_send[owner].push_back( rep );
            bc_recv[owner].push_back( rep );
        }
        // local re-broadcast: representative -> my other local members of this cross-rank class
        for ( std::size_t i = 1; i < mine.size(); ++i )
        {
            plan.rebroadcast_src.push_back( rep );
            plan.rebroadcast_dst.push_back( mine[i] );
        }
    }

    auto flatten = [&]( std::map< int, std::vector< Idx4 > >& m, std::vector< RankBuffer >& out ) {
        for ( auto& [p, nodes] : m )
            out.push_back( RankBuffer{ p, std::move( nodes ) } );
    };
    flatten( red_send, plan.reduce_send );
    flatten( red_recv, plan.reduce_recv );
    flatten( bc_send, plan.bcast_send );
    flatten( bc_recv, plan.bcast_recv );
    return plan;
}

// Everything a rank needs for the distributed solve: its owned-only domain, local tables (host +
// device), and the comm plan.
struct DistributedAdaptiveMesh
{
    DistributedDomain    domain;   // owned-only
    TwoToOneTables       t_local;  // built on `domain`
    TwoToOneTablesDevice t_local_d;
    DistributedPlan      plan;
    MPI_Comm             comm = MPI_COMM_WORLD;
    int                  nx = 0, ny = 0, nr = 0;
};

inline DistributedAdaptiveMesh build_distributed_mesh( MPI_Comm comm, int lateral_diamond_refinement_level,
                                                       const std::vector< double >&               radii,
                                                       const amr::AdaptiveForest&                 forest,
                                                       const SubdomainToRankDistributionFunction& part )
{
    DistributedAdaptiveMesh mesh;
    mesh.comm            = comm;
    const int my_rank    = static_cast< int >( mpi::rank( comm ) );
    const int nprocs     = static_cast< int >( mpi::num_processes( comm ) );

    mesh.domain = DistributedDomain::create_adaptive_for_rank( lateral_diamond_refinement_level, radii,
                                                               forest, part, my_rank );
    mesh.domain.set_comm( comm );
    mesh.nx = mesh.domain.domain_info().subdomain_num_nodes_per_side_laterally();
    mesh.nr = mesh.domain.domain_info().subdomain_num_nodes_radially();
    mesh.ny = mesh.nx;
    // global class structure (replicated, host only). This all-leaves build DOES run the corner guard,
    // so it validates the mesh; the owned-only local build below skips it (remote seam partners look
    // "missing" there and would false-trip the guard).
    const auto dom_all = DistributedDomain::create_adaptive_for_rank(
        lateral_diamond_refinement_level, radii, forest, subdomain_to_rank_all_root, 0 );
    const auto t_all = build_2to1_tables( dom_all, mesh.nx, mesh.ny, mesh.nr, /*guard_corners=*/true );

    mesh.t_local   = build_2to1_tables( mesh.domain, mesh.nx, mesh.ny, mesh.nr, /*guard_corners=*/false );
    mesh.t_local_d = upload_2to1_tables( mesh.t_local );

    const int S_lat = forest.lateral_subdomains_per_diamond();
    const int S_rad = forest.radial_subdomains();
    const AnchorRankFn rank_of = [&]( const SubdomainInfo& a ) {
        return static_cast< int >( part( a, S_lat, S_rad ) );
    };
    mesh.plan = build_distributed_plan( forest, dom_all, t_all, rank_of, my_rank, nprocs );
    return mesh;
}

// ---- host-side MPI reduce/broadcast over a single scalar host field --------------------------------
template < typename HostField >
inline void mpi_reduce_broadcast( const DistributedPlan& plan, HostField& hf, MPI_Comm comm )
{
    auto at = [&]( const Idx4& i ) -> double& { return hf( i.s, i.x, i.y, i.r ); };

    // ---- reduce: members -> owner (owner ADDS) ----
    {
        std::vector< std::vector< double > > sbuf( plan.reduce_send.size() ), rbuf( plan.reduce_recv.size() );
        std::vector< MPI_Request >           reqs;
        for ( std::size_t i = 0; i < plan.reduce_recv.size(); ++i )
        {
            rbuf[i].resize( plan.reduce_recv[i].nodes.size() );
            reqs.emplace_back();
            MPI_Irecv( rbuf[i].data(), (int) rbuf[i].size(), MPI_DOUBLE, plan.reduce_recv[i].partner, 71,
                       comm, &reqs.back() );
        }
        for ( std::size_t i = 0; i < plan.reduce_send.size(); ++i )
        {
            sbuf[i].resize( plan.reduce_send[i].nodes.size() );
            for ( std::size_t k = 0; k < sbuf[i].size(); ++k )
                sbuf[i][k] = at( plan.reduce_send[i].nodes[k] );
            reqs.emplace_back();
            MPI_Isend( sbuf[i].data(), (int) sbuf[i].size(), MPI_DOUBLE, plan.reduce_send[i].partner, 71,
                       comm, &reqs.back() );
        }
        MPI_Waitall( (int) reqs.size(), reqs.data(), MPI_STATUSES_IGNORE );
        for ( std::size_t i = 0; i < plan.reduce_recv.size(); ++i )
            for ( std::size_t k = 0; k < rbuf[i].size(); ++k )
                at( plan.reduce_recv[i].nodes[k] ) += rbuf[i][k];
    }
    // ---- broadcast: owner -> members (member OVERWRITES) ----
    {
        std::vector< std::vector< double > > sbuf( plan.bcast_send.size() ), rbuf( plan.bcast_recv.size() );
        std::vector< MPI_Request >           reqs;
        for ( std::size_t i = 0; i < plan.bcast_recv.size(); ++i )
        {
            rbuf[i].resize( plan.bcast_recv[i].nodes.size() );
            reqs.emplace_back();
            MPI_Irecv( rbuf[i].data(), (int) rbuf[i].size(), MPI_DOUBLE, plan.bcast_recv[i].partner, 72,
                       comm, &reqs.back() );
        }
        for ( std::size_t i = 0; i < plan.bcast_send.size(); ++i )
        {
            sbuf[i].resize( plan.bcast_send[i].nodes.size() );
            for ( std::size_t k = 0; k < sbuf[i].size(); ++k )
                sbuf[i][k] = at( plan.bcast_send[i].nodes[k] );
            reqs.emplace_back();
            MPI_Isend( sbuf[i].data(), (int) sbuf[i].size(), MPI_DOUBLE, plan.bcast_send[i].partner, 72,
                       comm, &reqs.back() );
        }
        MPI_Waitall( (int) reqs.size(), reqs.data(), MPI_STATUSES_IGNORE );
        for ( std::size_t i = 0; i < plan.bcast_recv.size(); ++i )
            for ( std::size_t k = 0; k < rbuf[i].size(); ++k )
                at( plan.bcast_recv[i].nodes[k] ) = rbuf[i][k];
    }
    // local re-broadcast to my other copies of cross-rank classes
    for ( std::size_t i = 0; i < plan.rebroadcast_dst.size(); ++i )
        at( plan.rebroadcast_dst[i] ) = at( plan.rebroadcast_src[i] );
}

// Distributed assembly of one scalar host field: local assembly (existing) then the cross-rank halo.
template < typename HostField >
inline void assemble_host( const DistributedAdaptiveMesh& mesh, HostField& hf )
{
    apply_exchange_tables( mesh.t_local, hf ); // local: hanging P^T + local class sums + local broadcast
    mpi_reduce_broadcast( mesh.plan, hf, mesh.comm );
}

// Distributed assembly on a device scalar grid (host-staged).
template < typename ScalarT >
inline void assemble_distributed( const DistributedAdaptiveMesh& mesh,
                                  const grid::Grid4DDataScalar< ScalarT >& field )
{
    auto h = Kokkos::create_mirror_view( field );
    Kokkos::deep_copy( h, field );
    assemble_host( mesh, h );
    Kokkos::deep_copy( field, h );
}

// Distributed assembly on a device vector grid (per component).
template < typename ScalarT, int VecDim >
inline void assemble_distributed( const DistributedAdaptiveMesh& mesh,
                                  const grid::Grid4DDataVec< ScalarT, VecDim >& field )
{
    for ( int d = 0; d < VecDim; ++d )
        assemble_distributed( mesh, field.comp_[d] );
}

// Ownership mask for the distributed domain: every physical node OWNED by exactly one rank globally
// (the min-rank owner's representative). Unmark my non-owned class copies (plan.not_owned) and my
// hanging DoFs (slaves). Genuine block-interior singletons stay owned. So masked dot products count
// each node once across the whole communicator.
inline Grid4DDataScalar< grid::NodeOwnershipFlag >
    distributed_ownership_mask( const DistributedAdaptiveMesh& mesh )
{
    auto mask = allocate_scalar_grid< grid::NodeOwnershipFlag >( "amr_dist_ownership", mesh.domain );
    auto h    = Kokkos::create_mirror_view( mask );
    for ( std::size_t s = 0; s < mask.extent( 0 ); ++s )
        for ( std::size_t x = 0; x < mask.extent( 1 ); ++x )
            for ( std::size_t y = 0; y < mask.extent( 2 ); ++y )
                for ( std::size_t r = 0; r < mask.extent( 3 ); ++r )
                    h( s, x, y, r ) = grid::NodeOwnershipFlag::OWNED;
    for ( const Idx4& i : mesh.plan.not_owned )
        h( i.s, i.x, i.y, i.r ) = grid::NodeOwnershipFlag::NO_FLAG;
    for ( const Idx4& d : mesh.t_local.con_dst ) // hanging DoFs are never owned
        h( d.s, d.x, d.y, d.r ) = grid::NodeOwnershipFlag::NO_FLAG;
    Kokkos::deep_copy( mask, h );
    return mask;
}

// Distributed constrained operator: C^T A_loc C with the cross-rank additive halo, and (optionally)
// Dirichlet elimination on the assembled system (same projection as the serial wrapper). LocalOp is
// built with SkipCommunication -- its apply is the pure block-local element loop; the halo does the
// rest. Constraint is local (Stage 0), the assembly is distributed (host-staged).
template < linalg::OperatorLike LocalOp >
class AdaptiveDistributedConstrainedOperator
{
  public:
    using ScalarType    = typename LocalOp::ScalarType;
    using SrcVectorType = typename LocalOp::SrcVectorType;
    using DstVectorType = typename LocalOp::DstVectorType;

    AdaptiveDistributedConstrainedOperator( LocalOp& op, const DistributedAdaptiveMesh& mesh,
                                            const SrcVectorType& tmp )
    : op_( op )
    , mesh_( &mesh )
    , tmp_( tmp )
    {}

    AdaptiveDistributedConstrainedOperator( LocalOp& op, const DistributedAdaptiveMesh& mesh,
                                            const SrcVectorType&                         tmp,
                                            const Grid4DDataScalar< ShellBoundaryFlag >& boundary_mask,
                                            ShellBoundaryFlag                            dirichlet_flag )
    : op_( op )
    , mesh_( &mesh )
    , tmp_( tmp )
    , boundary_mask_( boundary_mask )
    , dirichlet_flag_( dirichlet_flag )
    , eliminate_dirichlet_( true )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        amr_deep_copy( tmp_.grid_data(), src.grid_data() );
        if ( eliminate_dirichlet_ )
            kernels::common::assign_masked_else_keep_old( tmp_.grid_data(), ScalarType( 0 ),
                                                          boundary_mask_, dirichlet_flag_ );
        apply_constraint_device( mesh_->t_local_d, tmp_.grid_data() ); // local (Stage 0)
        linalg::apply( op_, tmp_, dst );                               // local element apply
        assemble_distributed( *mesh_, dst.grid_data() );              // cross-rank additive halo
        if ( eliminate_dirichlet_ )
            kernels::common::assign_masked_else_keep_old( dst.grid_data(), src.grid_data(),
                                                          boundary_mask_, dirichlet_flag_ );
    }

  private:
    LocalOp&                              op_;
    const DistributedAdaptiveMesh*        mesh_;
    SrcVectorType                         tmp_;
    Grid4DDataScalar< ShellBoundaryFlag > boundary_mask_{};
    ShellBoundaryFlag                     dirichlet_flag_      = ShellBoundaryFlag::NO_FLAG;
    bool                                  eliminate_dirichlet_ = false;
};

} // namespace terra::grid::shell::amr
