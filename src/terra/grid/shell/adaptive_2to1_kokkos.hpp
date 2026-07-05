#pragma once

// Kokkos (device) application of the AMR assembly-exchange + hanging-node-constraint tables.
//
// The tables are built on the host (build_2to1_tables in adaptive_exchange.hpp), uploaded once, and
// applied to a Grid4DDataScalar living in device memory. Kernel structure mirrors the host appliers
// exactly (two-phase gather/scatter), so host and device results agree to rounding:
//   exchange   = class-gather -> asm-gather -> class-scatter -> asm-scatter (atomic_add) -> broadcast
//   constraint = con-gather -> con-scatter (unique destinations; no atomics needed)

#include "../../kokkos/kokkos_wrapper.hpp"
#include "../grid_types.hpp"
#include "adaptive_exchange.hpp"
#include "util/timer.hpp" // time the hanging-node continuity operators (C, C^T, assembly exchange)

namespace terra::grid::shell::amr {

struct TwoToOneTablesDevice
{
    Kokkos::View< int*[4] >    asm_dst, asm_src;
    Kokkos::View< double* >    asm_w;
    Kokkos::View< int* >       cls_offsets; // CSR offsets (n_classes + 1)
    Kokkos::View< int*[4] >    cls_members;
    Kokkos::View< int*[4] > con_dst; // one per hanging copy
    Kokkos::View< int* >    con_off; // CSR offsets (n_hanging + 1)
    Kokkos::View< int*[4] > con_par; // flat genuine parents
    Kokkos::View< double* > con_wt;  // flat weights
};

namespace detail {
inline Kokkos::View< int*[4] > upload_idx4( const std::vector< Idx4 >& v, const std::string& label )
{
    Kokkos::View< int*[4] > d( label, v.size() );
    auto                    h = Kokkos::create_mirror_view( d );
    for ( std::size_t i = 0; i < v.size(); ++i )
    {
        h( i, 0 ) = v[i].s;
        h( i, 1 ) = v[i].x;
        h( i, 2 ) = v[i].y;
        h( i, 3 ) = v[i].r;
    }
    Kokkos::deep_copy( d, h );
    return d;
}
} // namespace detail

inline TwoToOneTablesDevice upload_2to1_tables( const TwoToOneTables& t )
{
    TwoToOneTablesDevice d;
    d.asm_dst     = detail::upload_idx4( t.asm_dst, "amr2to1_asm_dst" );
    d.asm_src     = detail::upload_idx4( t.asm_src, "amr2to1_asm_src" );
    d.cls_members = detail::upload_idx4( t.cls_members, "amr2to1_cls_members" );
    d.con_dst     = detail::upload_idx4( t.con_dst, "amr2to1_con_dst" );

    d.asm_w = Kokkos::View< double* >( "amr2to1_asm_w", t.asm_w.size() );
    {
        auto h = Kokkos::create_mirror_view( d.asm_w );
        for ( std::size_t i = 0; i < t.asm_w.size(); ++i )
            h( i ) = t.asm_w[i];
        Kokkos::deep_copy( d.asm_w, h );
    }
    d.cls_offsets = Kokkos::View< int* >( "amr2to1_cls_offsets", t.cls_offsets.size() );
    {
        auto h = Kokkos::create_mirror_view( d.cls_offsets );
        for ( std::size_t i = 0; i < t.cls_offsets.size(); ++i )
            h( i ) = t.cls_offsets[i];
        Kokkos::deep_copy( d.cls_offsets, h );
    }

    d.con_par = detail::upload_idx4( t.con_par, "amr2to1_con_par" );
    d.con_off = Kokkos::View< int* >( "amr2to1_con_off", t.con_off.size() );
    {
        auto h = Kokkos::create_mirror_view( d.con_off );
        for ( std::size_t i = 0; i < t.con_off.size(); ++i )
            h( i ) = t.con_off[i];
        Kokkos::deep_copy( d.con_off, h );
    }
    d.con_wt = Kokkos::View< double* >( "amr2to1_con_wt", t.con_wt.size() );
    {
        auto h = Kokkos::create_mirror_view( d.con_wt );
        for ( std::size_t i = 0; i < t.con_wt.size(); ++i )
            h( i ) = t.con_wt[i];
        Kokkos::deep_copy( d.con_wt, h );
    }
    return d;
}

// Device assembly exchange: class sums -> hanging P^T (atomic) -> broadcast.
template < typename ScalarT >
inline void apply_exchange_device( const TwoToOneTablesDevice& t,
                                   const grid::Grid4DDataScalar< ScalarT >& field )
{
    const auto f    = field;
    const auto co   = t.cls_offsets;
    const auto cm   = t.cls_members;
    const auto ad   = t.asm_dst;
    const auto as   = t.asm_src;
    const auto aw   = t.asm_w;
    const int  ncls = std::max( 0, static_cast< int >( co.extent( 0 ) ) - 1 ); // 0 for empty tables

    Kokkos::View< ScalarT* > cls_tmp( "amr2to1_cls_tmp", ncls );
    Kokkos::View< ScalarT* > asm_tmp( "amr2to1_asm_tmp", ad.extent( 0 ) );

    // gather (both phases read the pre-exchange field)
    Kokkos::parallel_for(
        "amr2to1_cls_gather", ncls, KOKKOS_LAMBDA( const int c ) {
            ScalarT v = 0;
            for ( int m = co( c ); m < co( c + 1 ); ++m )
                v += f( cm( m, 0 ), cm( m, 1 ), cm( m, 2 ), cm( m, 3 ) );
            cls_tmp( c ) = v;
        } );
    Kokkos::parallel_for(
        "amr2to1_asm_gather", ad.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            asm_tmp( i ) = static_cast< ScalarT >( aw( i ) ) *
                           f( as( i, 0 ), as( i, 1 ), as( i, 2 ), as( i, 3 ) );
        } );

    // scatter: canonical = class sum, then += hanging contributions
    Kokkos::parallel_for(
        "amr2to1_cls_scatter", ncls, KOKKOS_LAMBDA( const int c ) {
            const int m = co( c );
            f( cm( m, 0 ), cm( m, 1 ), cm( m, 2 ), cm( m, 3 ) ) = cls_tmp( c );
        } );
    Kokkos::parallel_for(
        "amr2to1_asm_scatter", ad.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            Kokkos::atomic_add( &f( ad( i, 0 ), ad( i, 1 ), ad( i, 2 ), ad( i, 3 ) ), asm_tmp( i ) );
        } );

    // broadcast canonical to the remaining copies
    Kokkos::parallel_for(
        "amr2to1_broadcast", ncls, KOKKOS_LAMBDA( const int c ) {
            const int     m0 = co( c );
            const ScalarT v  = f( cm( m0, 0 ), cm( m0, 1 ), cm( m0, 2 ), cm( m0, 3 ) );
            for ( int m = m0 + 1; m < co( c + 1 ); ++m )
                f( cm( m, 0 ), cm( m, 1 ), cm( m, 2 ), cm( m, 3 ) ) = v;
        } );
    Kokkos::fence();
}

// Device CLASS-SUM only: assemble coincident copies, but do NOT fold hanging DoFs into their parents.
// Intended for the smoother DIAGONAL: folding the hanging block-diagonal into parents (with the linear
// constraint weight, as assemble_distributed does) grossly inflates the parent diagonal in high-viscosity
// bands -- the true diag(C^T A C) self-term is w^2 and is largely cancelled by the negative parent-hanging
// cross term, so the parent's own assembled diagonal A_pp is a much closer, smooth, deterministic estimate.
template < typename ScalarT >
inline void apply_class_sum_device( const TwoToOneTablesDevice&              t,
                                    const grid::Grid4DDataScalar< ScalarT >& field )
{
    const auto f    = field;
    const auto co   = t.cls_offsets;
    const auto cm   = t.cls_members;
    const int  ncls = std::max( 0, static_cast< int >( co.extent( 0 ) ) - 1 );
    Kokkos::View< ScalarT* > cls_tmp( "amr2to1_cls_only_tmp", ncls );
    Kokkos::parallel_for( "amr2to1_cls_gather_only", ncls, KOKKOS_LAMBDA( const int c ) {
        ScalarT v = 0;
        for ( int m = co( c ); m < co( c + 1 ); ++m )
            v += f( cm( m, 0 ), cm( m, 1 ), cm( m, 2 ), cm( m, 3 ) );
        cls_tmp( c ) = v;
    } );
    Kokkos::parallel_for( "amr2to1_cls_bcast_only", ncls, KOKKOS_LAMBDA( const int c ) {
        for ( int m = co( c ); m < co( c + 1 ); ++m )
            f( cm( m, 0 ), cm( m, 1 ), cm( m, 2 ), cm( m, 3 ) ) = cls_tmp( c );
    } );
    Kokkos::fence();
}

template < typename ScalarT, int VecDim >
inline void apply_class_sum_device( const TwoToOneTablesDevice&                   t,
                                    const grid::Grid4DDataVec< ScalarT, VecDim >& field )
{
    for ( int d = 0; d < VecDim; ++d )
        apply_class_sum_device( t, field.comp_[d] );
}

// Broadcast each class's canonical (first-member) value to all its coincident copies. Used to make a
// per-node label (e.g. a coloring) consistent across copies so a probe vector is a clean e_i.
template < typename ScalarT >
inline void apply_class_broadcast_device( const TwoToOneTablesDevice&              t,
                                          const grid::Grid4DDataScalar< ScalarT >& field )
{
    const auto f    = field;
    const auto co   = t.cls_offsets;
    const auto cm   = t.cls_members;
    const int  ncls = std::max( 0, static_cast< int >( co.extent( 0 ) ) - 1 );
    Kokkos::parallel_for( "amr2to1_cls_bcast_canon", ncls, KOKKOS_LAMBDA( const int c ) {
        const int     m0 = co( c );
        const ScalarT v  = f( cm( m0, 0 ), cm( m0, 1 ), cm( m0, 2 ), cm( m0, 3 ) );
        for ( int m = m0 + 1; m < co( c + 1 ); ++m )
            f( cm( m, 0 ), cm( m, 1 ), cm( m, 2 ), cm( m, 3 ) ) = v;
    } );
    Kokkos::fence();
}

// Device hanging-node constraint: gather interpolations, then overwrite the hanging DoFs.
template < typename ScalarT >
inline void apply_constraint_device( const TwoToOneTablesDevice& t,
                                     const grid::Grid4DDataScalar< ScalarT >& field )
{
    const auto f   = field;
    const auto d   = t.con_dst;
    const auto off = t.con_off;
    const auto par = t.con_par;
    const auto w   = t.con_wt;
    Kokkos::View< ScalarT* > tmp( "amr2to1_con_tmp", d.extent( 0 ) );
    Kokkos::parallel_for(
        "amr2to1_con_gather", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            ScalarT v = 0;
            for ( int p = off( i ); p < off( i + 1 ); ++p )
                v += static_cast< ScalarT >( w( p ) ) *
                     f( par( p, 0 ), par( p, 1 ), par( p, 2 ), par( p, 3 ) );
            tmp( i ) = v;
        } );
    Kokkos::parallel_for(
        "amr2to1_con_scatter", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            f( d( i, 0 ), d( i, 1 ), d( i, 2 ), d( i, 3 ) ) = tmp( i );
        } );
    Kokkos::fence();
}

// Vector-field (SoA Grid4DDataVec) variants: the tables are node-based, so each component is exchanged
// / constrained independently with the same tables.
template < typename ScalarT, int VecDim >
inline void apply_exchange_device( const TwoToOneTablesDevice& t,
                                   const grid::Grid4DDataVec< ScalarT, VecDim >& field )
{
    terra::util::Timer _t( "continuity_exchange" ); // C^T assembly (class-sum + hanging fold + broadcast)
    for ( int d = 0; d < VecDim; ++d )
        apply_exchange_device( t, field.comp_[d] );
}

template < typename ScalarT, int VecDim >
inline void apply_constraint_device( const TwoToOneTablesDevice& t,
                                     const grid::Grid4DDataVec< ScalarT, VecDim >& field )
{
    terra::util::Timer _t( "continuity_constraint" ); // C: hanging DoF = interp of coarse parents
    for ( int d = 0; d < VecDim; ++d )
        apply_constraint_device( t, field.comp_[d] );
}

// Transpose of apply_constraint_device (C^T): C sets  h = W*parents  (overwrite hanging, leave parents),
// so C^T folds each hanging DoF's value ADDITIVELY back into its parents (parents += w*h) and zeroes the
// hanging slot (hanging is not a free DoF). This is the fine-grid piece needed to make the geometric
// RestrictionVecConstant the exact transpose of  C_fine o ProlongationVecConstant.
template < typename ScalarT >
inline void apply_constraint_transpose_device( const TwoToOneTablesDevice&              t,
                                               const grid::Grid4DDataScalar< ScalarT >& field )
{
    const auto f   = field;
    const auto d   = t.con_dst;
    const auto off = t.con_off;
    const auto par = t.con_par;
    const auto w   = t.con_wt;
    Kokkos::View< ScalarT* > tmp( "amr2to1_cont_tmp", d.extent( 0 ) );
    Kokkos::parallel_for( // read hanging values, then zero the hanging slot
        "amr2to1_cont_read", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            tmp( i )                              = f( d( i, 0 ), d( i, 1 ), d( i, 2 ), d( i, 3 ) );
            f( d( i, 0 ), d( i, 1 ), d( i, 2 ), d( i, 3 ) ) = 0;
        } );
    Kokkos::parallel_for( // fold each hanging value additively into its parents (atomic: parents shared)
        "amr2to1_cont_scatter", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            for ( int p = off( i ); p < off( i + 1 ); ++p )
                Kokkos::atomic_add( &f( par( p, 0 ), par( p, 1 ), par( p, 2 ), par( p, 3 ) ),
                                    static_cast< ScalarT >( w( p ) ) * tmp( i ) );
        } );
    Kokkos::fence();
}

template < typename ScalarT, int VecDim >
inline void apply_constraint_transpose_device( const TwoToOneTablesDevice&                   t,
                                               const grid::Grid4DDataVec< ScalarT, VecDim >& field )
{
    terra::util::Timer _t( "continuity_constraint_T" ); // C^T: fold hanging value back into parents
    for ( int d = 0; d < VecDim; ++d )
        apply_constraint_transpose_device( t, field.comp_[d] );
}

// deep-copy helpers so solve-side code can treat scalar and SoA-vector grids uniformly
template < typename ScalarT >
inline void amr_deep_copy( const grid::Grid4DDataScalar< ScalarT >& dst,
                           const grid::Grid4DDataScalar< ScalarT >& src )
{
    Kokkos::deep_copy( dst, src );
}

template < typename ScalarT, int VecDim >
inline void amr_deep_copy( const grid::Grid4DDataVec< ScalarT, VecDim >& dst,
                           const grid::Grid4DDataVec< ScalarT, VecDim >& src )
{
    for ( int d = 0; d < VecDim; ++d )
        Kokkos::deep_copy( dst.comp_[d], src.comp_[d] );
}

} // namespace terra::grid::shell::amr
