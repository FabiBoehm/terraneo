#pragma once
//
// Implementation of the dynamic-remeshing pieces declared in adaptive_remesh.hpp:
//   compute_indicator, rebuild_forest, plan_remesh, FieldRemapper::remap.
//
// SCOPE: correct at np == 1 (the regime the whole block-octree AMR has been validated at). The np > 1 pieces
// (cross-rank block redistribution in the transfer, weighted SFC partition) are marked TODO and guarded with a
// log so a multi-rank run does not silently produce a wrong transfer.
//
// The transfer runs on the HOST (mirror views) since a remesh happens only every remesh_every steps -- clarity
// over device throughput here; a device kernel is a later optimization once the algorithm is settled.
//
#include "terra/grid/shell/adaptive_2to1_kokkos.hpp" // apply_constraint_device (reconcile hanging after transfer)
#include "terra/kokkos/kokkos_wrapper.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

namespace terra::grid::shell::amr {

// =========================================================================================================
//  (1) ERROR ESTIMATOR
// =========================================================================================================
// First-cut per-block indicator: the maximum absolute nodal difference of T across each block (~ |grad T| * h).
// It flags blocks where the temperature varies sharply -- plumes / boundary layers -- which is where advective
// error concentrates. A proper mass-weighted gradient-recovery (ZZ) estimator is the intended upgrade; the
// signature and orchestration stay identical when it lands.  // TODO: ZZ gradient recovery, mass-weighted.
template < typename ScalarT >
std::vector< double > compute_indicator( const AdaptiveState< ScalarT >& s )
{
    terra::util::Timer _t( "remesh_indicator" );
    const int             nsub = static_cast< int >( s.mesh.domain.subdomains().size() );
    const int             nx = s.mesh.nx, ny = s.mesh.ny, nr = s.mesh.nr;
    std::vector< double > ind( nsub, 0.0 );

    auto Th = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, s.T.grid_data() );
    for ( int sd = 0; sd < nsub; ++sd )
    {
        double g = 0.0;
        for ( int x = 0; x < nx; ++x )
            for ( int y = 0; y < ny; ++y )
                for ( int r = 0; r < nr; ++r )
                {
                    const double t = Th( sd, x, y, r );
                    if ( x + 1 < nx ) g = std::max( g, std::fabs( Th( sd, x + 1, y, r ) - t ) );
                    if ( y + 1 < ny ) g = std::max( g, std::fabs( Th( sd, x, y + 1, r ) - t ) );
                    if ( r + 1 < nr ) g = std::max( g, std::fabs( Th( sd, x, y, r + 1 ) - t ) );
                }
        ind[sd] = g;
    }
    return ind;
}

// =========================================================================================================
//  (2) REBUILD FOREST
// =========================================================================================================
namespace detail {

// A leaf's unique global key = the packed finest anchor. Cheap, comparable across forests at the same M.
inline std::int64_t leaf_key( const AdaptiveForest& f, const ForestLeaf& l )
{
    return f.finest_anchor( l ).global_id();
}

} // namespace detail

inline AdaptiveForest rebuild_forest( const AdaptiveForest& cur, const DistributedAdaptiveMesh& cur_mesh,
                                      const std::vector< double >& ind_local, MPI_Comm comm,
                                      const RemeshOptions& o )
{
    terra::util::Timer _t( "remesh_rebuild_forest" );
    const auto&        leaves = cur.leaves();
    const std::size_t  n      = leaves.size();

    // ---- scatter the per-LOCAL-block indicator onto the GLOBAL leaf order, then all-reduce so every rank
    // agrees (replicated-forest invariant). cur_mesh gives local-idx -> finest anchor (== leaf key).
    std::map< std::int64_t, double > ind_by_key;
    const int nsub_local = static_cast< int >( cur_mesh.domain.subdomains().size() );
    for ( int sdi = 0; sdi < nsub_local && sdi < static_cast< int >( ind_local.size() ); ++sdi )
        ind_by_key[cur_mesh.domain.subdomain_info_from_local_idx( sdi ).global_id()] = ind_local[sdi];

    std::vector< double > gind( n, 0.0 );
    for ( std::size_t i = 0; i < n; ++i )
    {
        auto it = ind_by_key.find( detail::leaf_key( cur, leaves[i] ) );
        if ( it != ind_by_key.end() ) gind[i] = it->second;
    }
    MPI_Allreduce( MPI_IN_PLACE, gind.data(), static_cast< int >( n ), MPI_DOUBLE, MPI_SUM, comm );

    double emax = 0.0;
    for ( double e : gind ) emax = std::max( emax, e );

    // ---- flag refine / coarsen against the same global emax.
    std::vector< char > refine( n, 0 ), coarsen( n, 0 );
    for ( std::size_t i = 0; i < n; ++i )
    {
        if ( emax > 0.0 && gind[i] >= o.refine_frac * emax && leaves[i].subdivision < o.max_subdiv )
            refine[i] = 1;
        else if ( gind[i] < o.coarsen_frac * emax && leaves[i].subdivision > o.min_subdiv )
            coarsen[i] = 1;
    }

    // key -> leaf index, for margin dilation via face neighbours.
    std::map< std::int64_t, std::size_t > idx_of_key;
    for ( std::size_t i = 0; i < n; ++i ) idx_of_key[detail::leaf_key( cur, leaves[i] )] = i;

    // ---- INTERFACE-PRESERVING margin: grow the refine set by margin_rings block layers so the 2:1 interface
    // lands in smooth regions (avoids the coefficient-blind MG penalty). One face-neighbour ring per pass.
    for ( int ring = 0; ring < o.margin_rings; ++ring )
    {
        std::vector< char > add( n, 0 );
        for ( std::size_t i = 0; i < n; ++i )
        {
            if ( !refine[i] ) continue;
            for ( int fi = 0; fi < 6; ++fi )
            {
                const auto fn = cur.face_neighbors( leaves[i], static_cast< Face >( fi ) );
                for ( const auto& nb : fn.neighbors )
                {
                    auto it = idx_of_key.find( detail::leaf_key( cur, nb.leaf ) );
                    if ( it != idx_of_key.end() && leaves[it->second].subdivision < o.max_subdiv )
                        add[it->second] = 1;
                }
            }
        }
        for ( std::size_t i = 0; i < n; ++i )
            if ( add[i] ) { refine[i] = 1; coarsen[i] = 0; }
    }

    // ---- build coarsen reps: coarsen() takes ONE representative per complete 8-sibling group, and a group can
    // only collapse if ALL eight children are present and flagged. Group by parent key; keep child[0] as rep.
    std::map< std::int64_t, int > coarsen_hits; // parent key -> #flagged children present
    std::map< std::int64_t, ForestLeaf > parent_rep;
    for ( std::size_t i = 0; i < n; ++i )
    {
        if ( !coarsen[i] || leaves[i].subdivision <= 0 ) continue;
        const ForestLeaf par = cur.parent( leaves[i] );
        const std::int64_t pk = detail::leaf_key( cur, par );
        coarsen_hits[pk] += 1;
        parent_rep[pk] = par;
    }
    std::vector< ForestLeaf > to_coarsen;
    for ( const auto& [pk, hits] : coarsen_hits )
        if ( hits == 8 )
        {
            // rep = the first child; coarsen() expects a representative in the sibling group.
            to_coarsen.push_back( cur.children( parent_rep[pk] )[0] );
        }

    std::vector< ForestLeaf > to_refine;
    for ( std::size_t i = 0; i < n; ++i )
        if ( refine[i] ) to_refine.push_back( leaves[i] );

    // ---- apply to a fresh (deep-copied) forest: coarsen, then refine, then 2:1-balance.
    AdaptiveForest next = cur;
    if ( !to_coarsen.empty() ) next.coarsen( to_coarsen );
    if ( !to_refine.empty() ) next.refine( to_refine );
    next.balance_2to1();
    return next;
}

// =========================================================================================================
//  (4a) PLAN: classify every NEW block against the OLD forest by finest-anchor masking.
// =========================================================================================================
namespace detail {

// Covering equal-or-coarser block: mask the finest anchor down subdivision by subdivision and look it up. Return
// the finest covering local index (or -1). Reproduces AdaptiveForest::leaf_at on the mesh's subdomain map.
inline int probe_covering( const std::map< std::int64_t, std::pair< int, int > >& by_key, // key -> (local, subdiv)
                           int M, int diamond, int ax, int ay, int ar, int start_subdiv )
{
    for ( int so = start_subdiv; so >= 0; --so )
    {
        const int sh = M - so;
        const int cx = ( ax >> sh ) << sh, cy = ( ay >> sh ) << sh, cr = ( ar >> sh ) << sh;
        const std::int64_t gid = SubdomainInfo( diamond, cx, cy, cr ).global_id();
        auto it = by_key.find( gid );
        if ( it != by_key.end() && it->second.second <= so ) // exists and equal-or-coarser => covers
            return it->second.first;
    }
    return -1;
}

inline std::map< std::int64_t, std::pair< int, int > > index_by_key( const DistributedAdaptiveMesh& mesh )
{
    std::map< std::int64_t, std::pair< int, int > > m;
    const int nsub = static_cast< int >( mesh.domain.subdomains().size() );
    for ( int s = 0; s < nsub; ++s )
    {
        const auto& info = mesh.domain.subdomain_info_from_local_idx( s );
        m[info.global_id()] = { s, mesh.domain.subdivision_of( info ) };
    }
    return m;
}

} // namespace detail

inline RemeshPlan plan_remesh( const DistributedAdaptiveMesh& old_mesh, const DistributedAdaptiveMesh& new_mesh )
{
    terra::util::Timer _t( "remesh_plan" );
    if ( mpi::num_processes( new_mesh.comm ) > 1 && mpi::rank( new_mesh.comm ) == 0 )
        std::printf( "  [remesh] WARNING: plan_remesh is np==1 only; cross-rank block transfer is TODO.\n" );

    const int M       = new_mesh.domain.max_subdivision();
    const int nsub_ne = static_cast< int >( new_mesh.domain.subdomains().size() );
    const int nsub_ol = static_cast< int >( old_mesh.domain.subdomains().size() );
    const auto old_by = detail::index_by_key( old_mesh );
    const auto new_by = detail::index_by_key( new_mesh );

    RemeshPlan plan;
    plan.blocks.resize( nsub_ne );

    // Pass A: every new block -> Same / Refined / (tentative) Coarsened.
    for ( int s = 0; s < nsub_ne; ++s )
    {
        const auto& info = new_mesh.domain.subdomain_info_from_local_idx( s );
        const int   sn   = new_mesh.domain.subdivision_of( info );
        BlockMapEntry& e = plan.blocks[s];
        e.new_local      = s;
        const int ol = detail::probe_covering( old_by, M, info.diamond_id(), info.subdomain_x(),
                                                info.subdomain_y(), info.subdomain_r(), sn );
        if ( ol >= 0 )
        {
            const int so = old_mesh.domain.subdivision_of( old_mesh.domain.subdomain_info_from_local_idx( ol ) );
            e.relation   = ( so == sn ) ? BlockRelation::Same : BlockRelation::Refined;
            e.old_local  = ol;
        }
        else
        {
            e.relation = BlockRelation::Coarsened; // filled by pass B
        }
    }

    // Pass B: every old block whose covering NEW block is strictly coarser is a child of that coarsened block.
    for ( int s = 0; s < nsub_ol; ++s )
    {
        const auto& info = old_mesh.domain.subdomain_info_from_local_idx( s );
        const int   so   = old_mesh.domain.subdivision_of( info );
        const int   nl   = detail::probe_covering( new_by, M, info.diamond_id(), info.subdomain_x(),
                                                    info.subdomain_y(), info.subdomain_r(), so );
        if ( nl < 0 ) continue;
        const int sc = new_mesh.domain.subdivision_of( new_mesh.domain.subdomain_info_from_local_idx( nl ) );
        if ( sc < so ) // new coarser than this old block => old is one of its finer tiles
        {
            plan.blocks[nl].old_children.push_back( s );
            if ( plan.blocks[nl].old_local < 0 ) plan.blocks[nl].old_local = s;
        }
    }
    return plan;
}

// =========================================================================================================
//  (4b) TRANSFER KERNELS  (host, np==1)
// =========================================================================================================
template < typename ScalarT >
void FieldRemapper< ScalarT >::remap_component( const Grid4DDataScalar< ScalarT >& src_view,
                                                const Grid4DDataScalar< ScalarT >& dst_view,
                                                const TransferPolicy& policy )
{
    if ( policy.kind != TransferKind::Geometric )
        throw std::runtime_error( "FieldRemapper: only TransferKind::Geometric is implemented" );

    const int M  = new_->domain.max_subdivision();
    const int nx = new_->nx, ny = new_->ny, nr = new_->nr;

    auto oh = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, src_view );
    auto nh = Kokkos::create_mirror_view( dst_view );

    auto anchor = []( const DistributedAdaptiveMesh& m, int local, int& sub, int& ax, int& ay, int& ar ) {
        const auto& info = m.domain.subdomain_info_from_local_idx( local );
        sub = m.domain.subdivision_of( info );
        ax  = info.subdomain_x();
        ay  = info.subdomain_y();
        ar  = info.subdomain_r();
    };

    // trilinear sample of the old block `os` at fractional node coordinates (gx,gy,gr).
    auto sample = [&]( int os, double gx, double gy, double gr ) -> double {
        auto clampd = []( double v, int hi ) { return v < 0 ? 0.0 : ( v > hi ? (double) hi : v ); };
        gx = clampd( gx, nx - 1 ); gy = clampd( gy, ny - 1 ); gr = clampd( gr, nr - 1 );
        const int i0 = std::min( (int) std::floor( gx ), nx - 2 < 0 ? 0 : nx - 2 );
        const int j0 = std::min( (int) std::floor( gy ), ny - 2 < 0 ? 0 : ny - 2 );
        const int k0 = std::min( (int) std::floor( gr ), nr - 2 < 0 ? 0 : nr - 2 );
        const int i1 = std::min( i0 + 1, nx - 1 ), j1 = std::min( j0 + 1, ny - 1 ), k1 = std::min( k0 + 1, nr - 1 );
        const double tx = gx - i0, ty = gy - j0, tz = gr - k0;
        auto v = [&]( int i, int j, int k ) { return (double) oh( os, i, j, k ); };
        const double c00 = v( i0, j0, k0 ) * ( 1 - tx ) + v( i1, j0, k0 ) * tx;
        const double c10 = v( i0, j1, k0 ) * ( 1 - tx ) + v( i1, j1, k0 ) * tx;
        const double c01 = v( i0, j0, k1 ) * ( 1 - tx ) + v( i1, j0, k1 ) * tx;
        const double c11 = v( i0, j1, k1 ) * ( 1 - tx ) + v( i1, j1, k1 ) * tx;
        const double c0  = c00 * ( 1 - ty ) + c10 * ty, c1 = c01 * ( 1 - ty ) + c11 * ty;
        return c0 * ( 1 - tz ) + c1 * tz;
    };

    for ( const auto& e : plan_->blocks )
    {
        const int sN = e.new_local;
        int snv, anx, any, anr;
        anchor( *new_, sN, snv, anx, any, anr );
        const long span_n = 1L << ( M - snv );

        if ( e.relation == BlockRelation::Same )
        {
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        nh( sN, x, y, r ) = oh( e.old_local, x, y, r );
        }
        else if ( e.relation == BlockRelation::Refined )
        {
            // old block is coarser and CONTAINS this new block: sample it at the new nodes' fractional
            // positions. old fractional index = (An - Ao)*(n-1)/span_o + node * span_n/span_o.
            int sov, aox, aoy, aor;
            anchor( *old_, e.old_local, sov, aox, aoy, aor );
            const long   span_o = 1L << ( M - sov );
            const double bx = (double) ( anx - aox ) * ( nx - 1 ) / span_o;
            const double by = (double) ( any - aoy ) * ( ny - 1 ) / span_o;
            const double br = (double) ( anr - aor ) * ( nr - 1 ) / span_o;
            const double sc = (double) span_n / span_o;
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                        nh( sN, x, y, r ) = (ScalarT) sample( e.old_local, bx + x * sc, by + y * sc, br + r * sc );
        }
        else // Coarsened: new block is coarser; each new node coincides with a node in one of the finer children.
        {
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                    {
                        bool set = false;
                        for ( int oc : e.old_children )
                        {
                            int sov, aox, aoy, aor;
                            anchor( *old_, oc, sov, aox, aoy, aor );
                            const long span_o = 1L << ( M - sov );
                            // old index (integer if the new node lands on an old node): (An-Ao)(n-1)/span_o + node*span_n/span_o
                            const double oix = (double) ( anx - aox ) * ( nx - 1 ) / span_o + (double) x * span_n / span_o;
                            const double oiy = (double) ( any - aoy ) * ( ny - 1 ) / span_o + (double) y * span_n / span_o;
                            const double oir = (double) ( anr - aor ) * ( nr - 1 ) / span_o + (double) r * span_n / span_o;
                            const int    ix = (int) std::lround( oix ), iy = (int) std::lround( oiy ),
                                      ir = (int) std::lround( oir );
                            if ( ix >= 0 && ix < nx && iy >= 0 && iy < ny && ir >= 0 && ir < nr &&
                                 std::fabs( oix - ix ) < 1e-6 && std::fabs( oiy - iy ) < 1e-6 &&
                                 std::fabs( oir - ir ) < 1e-6 )
                            {
                                nh( sN, x, y, r ) = oh( oc, ix, iy, ir );
                                set               = true;
                                break;
                            }
                        }
                        if ( !set && !e.old_children.empty() )
                        {
                            // fallback: sample the representative child (keeps warm-start sane if a node is
                            // exactly on a child seam and lround missed by rounding).
                            int sov, aox, aoy, aor;
                            anchor( *old_, e.old_children.front(), sov, aox, aoy, aor );
                            const long   span_o = 1L << ( M - sov );
                            const double gx = (double) ( anx - aox ) * ( nx - 1 ) / span_o + (double) x * span_n / span_o;
                            const double gy = (double) ( any - aoy ) * ( ny - 1 ) / span_o + (double) y * span_n / span_o;
                            const double gr = (double) ( anr - aor ) * ( nr - 1 ) / span_o + (double) r * span_n / span_o;
                            nh( sN, x, y, r ) = (ScalarT) sample( e.old_children.front(), gx, gy, gr );
                        }
                    }
        }
    }
    Kokkos::deep_copy( dst_view, nh );
}

template < typename ScalarT >
void FieldRemapper< ScalarT >::remap( const linalg::VectorQ1Scalar< ScalarT >& src,
                                      linalg::VectorQ1Scalar< ScalarT >& dst, const TransferPolicy& policy )
{
    terra::util::Timer _t( "remesh_transfer_scalar" );
    remap_component( src.grid_data(), dst.grid_data(), policy );
    apply_constraint_device( new_->t_local_d, dst.grid_data() ); // reconcile hanging nodes on the new mesh
}

template < typename ScalarT >
template < int C >
void FieldRemapper< ScalarT >::remap( const linalg::VectorQ1Vec< ScalarT, C >& src,
                                      linalg::VectorQ1Vec< ScalarT, C >& dst, const TransferPolicy& policy )
{
    terra::util::Timer _t( "remesh_transfer_vec" );
    for ( int c = 0; c < C; ++c )
        remap_component( src.grid_data().comp_[c], dst.grid_data().comp_[c], policy );
    apply_constraint_device( new_->t_local_d, dst.grid_data() ); // reconcile hanging nodes on the new mesh
}

} // namespace terra::grid::shell::amr
