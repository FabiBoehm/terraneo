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

namespace terra::grid::shell::amr {

struct TwoToOneTablesDevice
{
    Kokkos::View< int*[4] >    asm_dst, asm_src;
    Kokkos::View< double* >    asm_w;
    Kokkos::View< int* >       cls_offsets; // CSR offsets (n_classes + 1)
    Kokkos::View< int*[4] >    cls_members;
    Kokkos::View< int*[4] >    con_dst;
    Kokkos::View< int*[4][4] > con_src; // (entry, parent, index-component)
    Kokkos::View< double*[4] > con_w;
    Kokkos::View< int* >       con_np;
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

    const std::size_t nc = t.con_np.size();
    d.con_src            = Kokkos::View< int*[4][4] >( "amr2to1_con_src", nc );
    d.con_w              = Kokkos::View< double*[4] >( "amr2to1_con_w", nc );
    d.con_np             = Kokkos::View< int* >( "amr2to1_con_np", nc );
    {
        auto hs = Kokkos::create_mirror_view( d.con_src );
        auto hw = Kokkos::create_mirror_view( d.con_w );
        auto hn = Kokkos::create_mirror_view( d.con_np );
        for ( std::size_t i = 0; i < nc; ++i )
        {
            hn( i ) = t.con_np[i];
            for ( int p = 0; p < 4; ++p )
            {
                hs( i, p, 0 ) = t.con_src[i][p].s;
                hs( i, p, 1 ) = t.con_src[i][p].x;
                hs( i, p, 2 ) = t.con_src[i][p].y;
                hs( i, p, 3 ) = t.con_src[i][p].r;
                hw( i, p )    = t.con_w[i][p];
            }
        }
        Kokkos::deep_copy( d.con_src, hs );
        Kokkos::deep_copy( d.con_w, hw );
        Kokkos::deep_copy( d.con_np, hn );
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
    const int  ncls = static_cast< int >( co.extent( 0 ) ) - 1;

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

// Device hanging-node constraint: gather interpolations, then overwrite the hanging DoFs.
template < typename ScalarT >
inline void apply_constraint_device( const TwoToOneTablesDevice& t,
                                     const grid::Grid4DDataScalar< ScalarT >& field )
{
    const auto f  = field;
    const auto d  = t.con_dst;
    const auto s  = t.con_src;
    const auto w  = t.con_w;
    const auto np = t.con_np;
    Kokkos::View< ScalarT* > tmp( "amr2to1_con_tmp", d.extent( 0 ) );
    Kokkos::parallel_for(
        "amr2to1_con_gather", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            ScalarT v = 0;
            for ( int p = 0; p < np( i ); ++p )
                v += static_cast< ScalarT >( w( i, p ) ) *
                     f( s( i, p, 0 ), s( i, p, 1 ), s( i, p, 2 ), s( i, p, 3 ) );
            tmp( i ) = v;
        } );
    Kokkos::parallel_for(
        "amr2to1_con_scatter", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
            f( d( i, 0 ), d( i, 1 ), d( i, 2 ), d( i, 3 ) ) = tmp( i );
        } );
    Kokkos::fence();
}

} // namespace terra::grid::shell::amr
