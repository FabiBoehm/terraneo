#pragma once

// Kokkos (device) application of the 2:1 exchange + hanging-node constraint tables.
//
// The tables are built on the host (build_2to1_tables in adaptive_exchange.hpp), uploaded once, and
// applied to a Grid4DDataScalar living in device memory. Kernel structure mirrors the host appliers
// exactly (two-phase gather/scatter), so host and device results agree to rounding:
//   exchange   = asm-gather -> asm-scatter (atomic_add; several fine nodes may feed one coarse node)
//                -> broadcast -> same-level pair sums
//   constraint = con-gather -> con-scatter (unique destinations; no atomics needed)

#include "../../kokkos/kokkos_wrapper.hpp"
#include "../grid_types.hpp"
#include "adaptive_exchange.hpp"

namespace terra::grid::shell::amr {

struct TwoToOneTablesDevice
{
    Kokkos::View< int*[4] >    asm_dst, asm_src;
    Kokkos::View< double* >    asm_w;
    Kokkos::View< int*[4] >    bc_dst, bc_src;
    Kokkos::View< int*[4] >    pair_a, pair_b;
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
    d.asm_dst = detail::upload_idx4( t.asm_dst, "amr2to1_asm_dst" );
    d.asm_src = detail::upload_idx4( t.asm_src, "amr2to1_asm_src" );
    d.bc_dst  = detail::upload_idx4( t.bc_dst, "amr2to1_bc_dst" );
    d.bc_src  = detail::upload_idx4( t.bc_src, "amr2to1_bc_src" );
    d.pair_a  = detail::upload_idx4( t.pair_a, "amr2to1_pair_a" );
    d.pair_b  = detail::upload_idx4( t.pair_b, "amr2to1_pair_b" );
    d.con_dst = detail::upload_idx4( t.con_dst, "amr2to1_con_dst" );

    d.asm_w = Kokkos::View< double* >( "amr2to1_asm_w", t.asm_w.size() );
    {
        auto h = Kokkos::create_mirror_view( d.asm_w );
        for ( std::size_t i = 0; i < t.asm_w.size(); ++i )
            h( i ) = t.asm_w[i];
        Kokkos::deep_copy( d.asm_w, h );
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

// Device 2:1 exchange: assembly (atomic) -> broadcast -> same-level pair sums.
template < typename ScalarT >
inline void apply_exchange_device( const TwoToOneTablesDevice& t,
                                   const grid::Grid4DDataScalar< ScalarT >& field )
{
    const auto f = field;
    {
        const auto d = t.asm_dst;
        const auto s = t.asm_src;
        const auto w = t.asm_w;
        Kokkos::View< ScalarT* > tmp( "amr2to1_asm_tmp", d.extent( 0 ) );
        Kokkos::parallel_for(
            "amr2to1_asm_gather", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
                tmp( i ) = static_cast< ScalarT >( w( i ) ) *
                           f( s( i, 0 ), s( i, 1 ), s( i, 2 ), s( i, 3 ) );
            } );
        Kokkos::parallel_for(
            "amr2to1_asm_scatter", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
                Kokkos::atomic_add( &f( d( i, 0 ), d( i, 1 ), d( i, 2 ), d( i, 3 ) ), tmp( i ) );
            } );
    }
    {
        const auto d = t.bc_dst;
        const auto s = t.bc_src;
        Kokkos::parallel_for(
            "amr2to1_bc", d.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
                f( d( i, 0 ), d( i, 1 ), d( i, 2 ), d( i, 3 ) ) =
                    f( s( i, 0 ), s( i, 1 ), s( i, 2 ), s( i, 3 ) );
            } );
    }
    {
        const auto a = t.pair_a;
        const auto b = t.pair_b;
        Kokkos::parallel_for(
            "amr2to1_pair", a.extent( 0 ), KOKKOS_LAMBDA( const int i ) {
                const ScalarT sum = f( a( i, 0 ), a( i, 1 ), a( i, 2 ), a( i, 3 ) ) +
                                    f( b( i, 0 ), b( i, 1 ), b( i, 2 ), b( i, 3 ) );
                f( a( i, 0 ), a( i, 1 ), a( i, 2 ), a( i, 3 ) ) = sum;
                f( b( i, 0 ), b( i, 1 ), b( i, 2 ), b( i, 3 ) ) = sum;
            } );
    }
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
