#pragma once
// Analytic Stokes solutions on a spherical shell after Kramer, Sime & Davies, GMD 2021 (the
// "assess" package), generalised to any harmonic degree/order (l, m) and both forcing families:
//
//   smooth:  f = -g (r/Rp)^k Y_lm r_hat                      (solution C^infinity)
//   delta:   f = -g delta(r - r') Y_lm r_hat                 (pressure jumps at r', velocity C^1)
//
// for no-slip/no-slip and free-slip/free-slip shells, viscosity 1. The poloidal potential is
// P = P_l(r) Y_lm with P_l = A r^l + B r^-(l+1) + C r^(l+2) + D r^(-l+1) [+ E r^(k+3)], the pressure
// p = (G r^l + H r^-(l+1) [+ K r^(k+1)]) Y_lm; the delta family has one (A, B, C, D) set per side of r'.
//
// For a VARIABLE viscosity eta(x) the same (u, p) is kept as the exact solution and the body force
// is manufactured:  f = eta (f0 - grad p) + grad p - 2 eps(u) grad eta  (from -div(2 eta eps(u)) =
// -eta div(2 eps(u)) - 2 eps(u) grad eta and -div(2 eps(u)) = f0 - grad p), evaluated pointwise with
// central differences of the analytic fields. The delta family keeps its surface load at r', scaled
// to  -(eta' g Y + (eta' - 1) [p]) r_hat  ([p] = p_hi - p_lo at r', eta' = eta at r'). Free-slip and
// no-slip remain exact: u.n = 0 is kinematic and the tangential traction 2 eta eps(u) r_hat vanishes
// wherever the viscosity-1 traction does.
#ifndef ASSESS_STANDALONE
#include <Kokkos_Core.hpp>
#endif
#include <cmath>

namespace assess_solution
{
constexpr double gravity = 1.0;

enum class Forcing
{
    SMOOTH,
    DELTA
};

struct Params
{
    int     l        = 2;
    int     m        = 2;
    int     k        = 2;     // smooth family only
    Forcing forcing  = Forcing::SMOOTH;
    double  rp       = 1.72;  // delta family: radius of the density anomaly
    bool    freeslip = false; // free-slip/free-slip, otherwise no-slip/no-slip
    double  Rp       = 2.22;
    double  Rm       = 1.22;
};

struct Radial
{
    double A = 0, B = 0, C = 0, D = 0, E = 0, G = 0, H = 0, K = 0;
};

struct Coeffs
{
    Params prm;
    Radial lo, hi; // smooth: lo == hi; delta: lo for r < r', hi for r > r'
};

// Viscosity of the manufactured problem: eta = rmu^(0.5 - T), T = (Rp - r)/(Rp - Rm) + amp Y_32
// (the test's Frank-Kamenetskii law on the A7 initial temperature, without the clamp so that
// eta stays smooth). rmu = 1 is the constant-viscosity analytic problem.
struct Visc
{
    double rmu = 1.0;
    double amp = 0.0;
    double Rp  = 2.22;
    double Rm  = 1.22;
};

// ---------------------------------------------------------------- coefficients (host)
inline double P( double b, int e )
{
    return std::pow( b, static_cast< double >( e ) );
}

inline Radial smooth_fs( double Rp, double Rm, int k, int l, double g, double nu )
{
    const double a = Rm / Rp;
    Radial       c;
    c.A = 0.5 * P( Rp, -l + 3 ) * ( P( a, k + 3 ) - P( a, -l + 1 ) ) * g /
          ( ( P( a, l ) - P( a, -l + 1 ) ) * ( k + l + 2 ) * ( k - l + 3 ) * ( 2 * l + 1 ) * nu );
    c.B = 0.5 * P( Rp, l + 4 ) * ( P( a, k + 4 ) - P( a, l + 3 ) ) * g /
          ( ( P( a, l + 3 ) - 1.0 / P( a, l ) ) * ( k + l + 4 ) * ( k - l + 1 ) * ( 2 * l + 1 ) * nu );
    c.C = -0.5 * P( Rp, -l + 1 ) * ( P( a, k + 4 ) - 1.0 / P( a, l ) ) * g /
          ( ( P( a, l + 3 ) - 1.0 / P( a, l ) ) * ( k + l + 4 ) * ( k - l + 1 ) * ( 2 * l + 1 ) * nu );
    c.D = -0.5 * P( Rp, l + 2 ) * ( P( a, k + 3 ) - P( a, l ) ) * g /
          ( ( P( a, l ) - P( a, -l + 1 ) ) * ( k + l + 2 ) * ( k - l + 3 ) * ( 2 * l + 1 ) * nu );
    c.E = g / ( P( Rp, k ) * ( k + l + 4 ) * ( k + l + 2 ) * ( k - l + 3 ) * ( k - l + 1 ) * nu );
    return c;
}

inline Radial smooth_ns( double Rp, double Rm, int k, int l, double g, double nu )
{
    const double a     = Rm / Rp;
    const double Gamma = ( ( P( a, l + 1 ) + P( a, l - 3 ) ) * ( 2 * l + 1 ) * ( 2 * l + 1 ) -
                           2.0 * P( a, l - 1 ) * ( 2 * l + 3 ) * ( 2 * l - 1 ) - 4.0 * P( a, 3 * l ) -
                           4.0 * P( a, -l - 2 ) ) *
                         ( k + l + 4 ) * ( k + l + 2 ) * ( k - l + 3 ) * ( k - l + 1 );
    Radial c;
    c.A = ( ( P( a, k + 2 ) + P( a, l - 1 ) ) * ( k + l + 2 ) * ( 2 * l + 3 ) -
            ( P( a, k ) + P( a, l + 1 ) ) * ( k + l + 4 ) * ( 2 * l + 1 ) -
            2.0 * ( P( a, k + 2 * l + 3 ) + P( a, -l - 2 ) ) * ( k - l + 1 ) ) *
          P( Rp, -l + 3 ) * g / ( Gamma * nu );
    c.B = ( ( P( a, k + 2 * l + 1 ) + P( a, l + 1 ) ) * ( k - l + 3 ) * ( 2 * l + 1 ) -
            ( P( a, k + 2 * l + 3 ) + P( a, l - 1 ) ) * ( k - l + 1 ) * ( 2 * l - 1 ) -
            2.0 * ( P( a, k + 2 ) + P( a, 3 * l ) ) * ( k + l + 2 ) ) *
          P( Rp, l + 4 ) * g / ( Gamma * nu );
    c.C = -( ( P( a, k + 2 ) + P( a, l - 3 ) ) * ( k + l + 2 ) * ( 2 * l + 1 ) -
             ( P( a, k ) + P( a, l - 1 ) ) * ( k + l + 4 ) * ( 2 * l - 1 ) -
             2.0 * ( P( a, k + 2 * l + 1 ) + P( a, -l - 2 ) ) * ( k - l + 3 ) ) *
          P( Rp, -l + 1 ) * g / ( Gamma * nu );
    c.D = -( ( P( a, k + 2 * l + 1 ) + P( a, l - 1 ) ) * ( k - l + 3 ) * ( 2 * l + 3 ) -
             ( P( a, k + 2 * l + 3 ) + P( a, l - 3 ) ) * ( k - l + 1 ) * ( 2 * l + 1 ) -
             2.0 * ( P( a, k ) + P( a, 3 * l ) ) * ( k + l + 4 ) ) *
          P( Rp, l + 2 ) * g / ( Gamma * nu );
    c.E = g / ( P( Rp, k ) * ( k + l + 4 ) * ( k + l + 2 ) * ( k - l + 3 ) * ( k - l + 1 ) * nu );
    return c;
}

inline Radial delta_fs( double Rp, double Rm, double rp, int l, double g, double nu, int sign )
{
    const double apm = ( sign > 0 ) ? Rp / rp : Rm / rp;
    const double amp = ( sign > 0 ) ? Rm / rp : Rp / rp;
    const double pm  = sign;
    Radial       c;
    c.A = -0.5 * ( P( amp, 2 * l - 1 ) - 1.0 ) * g * pm * P( rp, -l + 2 ) /
          ( ( P( amp, 2 * l - 1 ) - P( apm, 2 * l - 1 ) ) * ( 2 * l + 1 ) * ( 2 * l - 1 ) * nu );
    c.B = -0.5 * ( P( amp, -2 * l - 3 ) - 1.0 ) * g * pm * P( rp, l + 3 ) /
          ( ( P( amp, -2 * l - 3 ) - P( apm, -2 * l - 3 ) ) * ( 2 * l + 3 ) * ( 2 * l + 1 ) * nu );
    c.C = 0.5 * ( P( amp, 2 * l + 3 ) - 1.0 ) * g * pm /
          ( ( P( amp, 2 * l + 3 ) - P( apm, 2 * l + 3 ) ) * ( 2 * l + 3 ) * ( 2 * l + 1 ) * nu * P( rp, l ) );
    c.D = 0.5 * ( P( amp, -2 * l + 1 ) - 1.0 ) * g * pm * P( rp, l + 1 ) /
          ( ( P( amp, -2 * l + 1 ) - P( apm, -2 * l + 1 ) ) * ( 2 * l + 1 ) * ( 2 * l - 1 ) * nu );
    return c;
}

inline Radial delta_ns( double Rp, double Rm, double rp, int l, double g, double nu, int sign )
{
    const double ap = Rp / rp, am = Rm / rp, q = am / ap;
    const double pm = sign, mp = -sign;
    const double Den = ( 2 * l + 1 ) * ( 2 * l + 1 ) * ( am * am / ( ap * ap ) + ap * ap / ( am * am ) ) -
                       2.0 * ( 2 * l + 3 ) * ( 2 * l - 1 ) - 4.0 * P( q, 2 * l + 1 ) - 4.0 * P( q, -2 * l - 1 );
    Radial c;
    c.A = -0.5 *
          ( am * am - ap * ap - ( 2 * l + 1 ) * mp * P( q, 2 * sign * -1 ) / ( 2 * l - 1 ) -
            ( 2 * l + 3 ) * pm / ( 2 * l + 1 ) + 2.0 * ( P( am, -2 * l - 1 ) - P( ap, -2 * l - 1 ) ) / ( 2 * l + 1 ) +
            2.0 * ( am * am * P( ap, -2 * l - 1 ) - P( am, -2 * l - 1 ) * ap * ap ) / ( 2 * l - 1 ) -
            4.0 * pm * P( q, ( 2 * l + 1 ) * sign ) / ( ( 2 * l + 1 ) * ( 2 * l - 1 ) ) ) *
          g * P( rp, -l + 2 ) / ( Den * nu );
    c.B = -0.5 *
          ( am * am - ap * ap - ( 2 * l + 1 ) * mp * P( q, 2 * sign * -1 ) / ( 2 * l + 3 ) -
            ( 2 * l - 1 ) * pm / ( 2 * l + 1 ) -
            2.0 * ( am * am * P( ap, 2 * l + 1 ) - P( am, 2 * l + 1 ) * ap * ap ) / ( 2 * l + 3 ) -
            2.0 * ( P( am, 2 * l + 1 ) - P( ap, 2 * l + 1 ) ) / ( 2 * l + 1 ) -
            4.0 * pm * P( q, ( 2 * l + 1 ) * -sign ) / ( ( 2 * l + 3 ) * ( 2 * l + 1 ) ) ) *
          g * P( rp, l + 3 ) / ( Den * nu );
    c.C = 0.5 *
          ( ( 2 * l + 1 ) * pm * P( q, 2 * sign ) / ( 2 * l + 3 ) + ( 2 * l - 1 ) * mp / ( 2 * l + 1 ) -
            2.0 * ( P( am, -2 * l - 1 ) - P( ap, -2 * l - 1 ) ) / ( 2 * l + 1 ) +
            4.0 * mp * P( q, ( 2 * l + 1 ) * sign ) / ( ( 2 * l + 3 ) * ( 2 * l + 1 ) ) +
            2.0 * ( P( am, -2 * l - 1 ) / ( ap * ap ) - P( ap, -2 * l - 1 ) / ( am * am ) ) / ( 2 * l + 3 ) +
            1.0 / ( am * am ) - 1.0 / ( ap * ap ) ) *
          g / ( Den * nu * P( rp, l ) );
    c.D = 0.5 *
          ( ( 2 * l + 1 ) * pm * P( q, 2 * sign ) / ( 2 * l - 1 ) + ( 2 * l + 3 ) * mp / ( 2 * l + 1 ) +
            2.0 * ( P( am, 2 * l + 1 ) - P( ap, 2 * l + 1 ) ) / ( 2 * l + 1 ) +
            4.0 * mp * P( q, ( 2 * l + 1 ) * -sign ) / ( ( 2 * l + 1 ) * ( 2 * l - 1 ) ) -
            2.0 * ( P( am, 2 * l + 1 ) / ( ap * ap ) - P( ap, 2 * l + 1 ) / ( am * am ) ) / ( 2 * l - 1 ) +
            1.0 / ( am * am ) - 1.0 / ( ap * ap ) ) *
          g * P( rp, l + 1 ) / ( Den * nu );
    return c;
}

inline void finish( Radial& c, const Params& p, bool smooth )
{
    const double nu = 1.0, g = gravity;
    c.G             = -2.0 * nu * ( p.l + 1 ) * ( 2 * p.l + 3 ) * c.C;
    c.H             = -2.0 * nu * p.l * ( 2 * p.l - 1 ) * c.D;
    c.K = smooth ? -g * ( p.k + 2 ) / ( ( p.k + 1 ) * ( p.k + 2 ) - p.l * ( p.l + 1 ) ) / P( p.Rp, p.k ) : 0.0;
}

inline bool valid( const Params& p )
{
    if ( p.l < 1 || p.m < 0 || p.m > p.l )
        return false;
    if ( p.forcing == Forcing::SMOOTH )
        return ( p.k + 1 ) * ( p.k + 2 ) != p.l * ( p.l + 1 ) && ( p.k + 3 ) * ( p.k + 4 ) != p.l * ( p.l + 1 );
    return p.rp > p.Rm && p.rp < p.Rp;
}

inline Coeffs make_coeffs( const Params& p )
{
    Coeffs c;
    c.prm = p;
    if ( p.forcing == Forcing::SMOOTH )
    {
        c.lo = p.freeslip ? smooth_fs( p.Rp, p.Rm, p.k, p.l, gravity, 1.0 ) : smooth_ns( p.Rp, p.Rm, p.k, p.l, gravity, 1.0 );
        finish( c.lo, p, true );
        c.hi = c.lo;
    }
    else
    {
        c.lo = p.freeslip ? delta_fs( p.Rp, p.Rm, p.rp, p.l, gravity, 1.0, -1 ) : delta_ns( p.Rp, p.Rm, p.rp, p.l, gravity, 1.0, -1 );
        c.hi = p.freeslip ? delta_fs( p.Rp, p.Rm, p.rp, p.l, gravity, 1.0, +1 ) : delta_ns( p.Rp, p.Rm, p.rp, p.l, gravity, 1.0, +1 );
        finish( c.lo, p, false );
        finish( c.hi, p, false );
    }
    return c;
}

// ---------------------------------------------------------------- device evaluation
KOKKOS_INLINE_FUNCTION double ipow( double b, int e )
{
    if ( e < 0 )
    {
        b = 1.0 / b;
        e = -e;
    }
    double r = 1.0;
    while ( e )
    {
        if ( e & 1 )
            r *= b;
        b *= b;
        e >>= 1;
    }
    return r;
}

// Q_j^mu = P_j^mu / sin(theta) for mu >= 1 (associated Legendre with Condon-Shortley phase), for
// j and j - 1, by the recurrence (j - mu) P_j = (2j - 1) x P_{j-1} - (j + mu - 1) P_{j-2}.
KOKKOS_INLINE_FUNCTION void legendre_Q( const int j, const int mu, const double xx, const double ss, double& Q_jm1,
                                        double& Q_j )
{
    double q = 1.0; // (-1)^mu (2mu-1)!! s^(mu-1)
    for ( int i = 1; i <= mu; ++i )
        q *= -( 2.0 * i - 1.0 );
    for ( int i = 1; i < mu; ++i )
        q *= ss;
    double prev = 0.0, cur = q; // Q_{mu-1}^mu = 0, Q_mu^mu
    for ( int jj = mu + 1; jj <= j; ++jj )
    {
        const double next = ( ( 2.0 * jj - 1.0 ) * xx * cur - ( jj + mu - 1.0 ) * prev ) / ( jj - mu );
        prev              = cur;
        cur               = next;
    }
    Q_jm1 = ( j > mu ) ? prev : 0.0;
    Q_j   = cur;
}

// Real spherical harmonic Y_lm = N_lm P_l^m(cos theta) cos(m phi) (Condon-Shortley phase, as
// scipy.special.sph_harm_y), with dY/dtheta and (dY/dphi)/sin(theta). Q_l^m = P_l^m / sin(theta)
// is built by the standard recurrence so that the pole limits are finite.
KOKKOS_INLINE_FUNCTION void harmonic( const int l, const int m, const double x, const double s, const double phi,
                                      double& Y, double& dY_dtheta, double& dY_dphi_over_s )
{
    double norm = ( 2.0 * l + 1.0 ) / ( 4.0 * 3.14159265358979323846 );
    for ( int i = l - m + 1; i <= l + m; ++i )
        norm /= static_cast< double >( i );
    norm = Kokkos::sqrt( norm );

    if ( m == 0 )
    {
        double pm1 = 1.0, p = x; // P_0, P_1
        if ( l == 0 )
            p = 1.0;
        for ( int j = 2; j <= l; ++j )
        {
            const double next = ( ( 2.0 * j - 1.0 ) * x * p - ( j - 1.0 ) * pm1 ) / j;
            pm1               = p;
            p                 = next;
        }
        double q1m1, q1;
        legendre_Q( l, 1, x, s, q1m1, q1 ); // P_l^1 = s Q_l^1 = dP_l/dtheta
        Y              = norm * p;
        dY_dtheta      = norm * s * q1;
        dY_dphi_over_s = 0.0;
        return;
    }
    double Qm1, Q;
    legendre_Q( l, m, x, s, Qm1, Q );
    const double P_lm      = s * Q;
    const double dP_dtheta = -( ( l + m ) * Qm1 - l * x * Q ); // (1-x^2) dP/dx = (l+m) P_{l-1} - l x P_l
    const double cm = Kokkos::cos( m * phi ), sm = Kokkos::sin( m * phi );
    Y              = norm * P_lm * cm;
    dY_dtheta      = norm * dP_dtheta * cm;
    dY_dphi_over_s = -m * norm * Q * sm;
}

KOKKOS_INLINE_FUNCTION double Pl( const Radial& c, const int l, const int k, const double r )
{
    return c.A * ipow( r, l ) + c.B * ipow( r, -l - 1 ) + c.C * ipow( r, l + 2 ) + c.D * ipow( r, -l + 1 ) +
           c.E * ipow( r, k + 3 );
}
KOKKOS_INLINE_FUNCTION double dPldr( const Radial& c, const int l, const int k, const double r )
{
    return l * c.A * ipow( r, l - 1 ) + ( -l - 1 ) * c.B * ipow( r, -l - 2 ) + ( l + 2 ) * c.C * ipow( r, l + 1 ) +
           ( -l + 1 ) * c.D * ipow( r, -l ) + ( k + 3 ) * c.E * ipow( r, k + 2 );
}
KOKKOS_INLINE_FUNCTION double p_radial( const Radial& c, const int l, const int k, const double r )
{
    return c.G * ipow( r, l ) + c.H * ipow( r, -l - 1 ) + c.K * ipow( r, k + 1 );
}

// side: -1 / +1 force the lower / upper coefficient set of the delta family, 0 picks by r.
KOKKOS_INLINE_FUNCTION const Radial& radial_for( const Coeffs& c, const double r, const int side )
{
    if ( c.prm.forcing == Forcing::SMOOTH )
        return c.lo;
    const bool lower = ( side < 0 ) || ( side == 0 && r < c.prm.rp );
    return lower ? c.lo : c.hi;
}

KOKKOS_INLINE_FUNCTION void to_spherical( const double cx, const double cy, const double cz, double& r, double& theta,
                                          double& phi )
{
    r     = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    theta = Kokkos::acos( cz / r );
    phi   = Kokkos::atan2( cy, cx );
}

// u_r = -l(l+1) P_l Y / r,  u_theta = -(P_l/r + P_l') dY/dtheta,  u_phi = -(P_l/r + P_l') (dY/dphi)/sin(theta).
KOKKOS_INLINE_FUNCTION void velocity_cartesian( const Coeffs& c, const double cx, const double cy, const double cz,
                                                double& ux, double& uy, double& uz, const int side = 0 )
{
    const int l = c.prm.l, m = c.prm.m, k = c.prm.k;
    double    r, theta, phi;
    to_spherical( cx, cy, cz, r, theta, phi );
    const double  st = Kokkos::sin( theta ), ct = Kokkos::cos( theta );
    const Radial& rad = radial_for( c, r, side );
    const double  Pv = Pl( rad, l, k, r ), dP = dPldr( rad, l, k, r );
    double        Y, dYt, dYp_s;
    harmonic( l, m, ct, st, phi, Y, dYt, dYp_s );
    const double u_r     = -static_cast< double >( l * ( l + 1 ) ) * Pv * Y / r;
    const double radial  = Pv / r + dP;
    const double u_theta = -radial * dYt;
    const double u_phi   = -radial * dYp_s;
    // On the axis (req = 0) any longitude gives the same Cartesian vector: for m = 1 u_theta ~ cos(phi)
    // and u_phi ~ -sin(phi) combine to one horizontal vector, for m != 1 both vanish there.
    const double req  = Kokkos::sqrt( cx * cx + cy * cy );
    const double cphi = ( req > 0.0 ) ? cx / req : 1.0;
    const double sphi = ( req > 0.0 ) ? cy / req : 0.0;
    ux = st * cphi * u_r + ct * cphi * u_theta - sphi * u_phi;
    uy = st * sphi * u_r + ct * sphi * u_theta + cphi * u_phi;
    uz = ct * u_r - st * u_theta;
}

KOKKOS_INLINE_FUNCTION double pressure_cartesian( const Coeffs& c, const double cx, const double cy, const double cz,
                                                  const int side = 0 )
{
    double r, theta, phi;
    to_spherical( cx, cy, cz, r, theta, phi );
    double Y, dYt, dYp_s;
    harmonic( c.prm.l, c.prm.m, Kokkos::cos( theta ), Kokkos::sin( theta ), phi, Y, dYt, dYp_s );
    return p_radial( radial_for( c, r, side ), c.prm.l, c.prm.k, r ) * Y;
}

KOKKOS_INLINE_FUNCTION bool on_interface( const Coeffs& c, const double r )
{
    return c.prm.forcing == Forcing::DELTA && Kokkos::fabs( r - c.prm.rp ) < 1.0e-8 * c.prm.rp;
}

// Y_lm at a point (for the delta surface load).
KOKKOS_INLINE_FUNCTION double Ylm( const Coeffs& c, const double cx, const double cy, const double cz )
{
    double r, theta, phi;
    to_spherical( cx, cy, cz, r, theta, phi );
    double Y, dYt, dYp_s;
    harmonic( c.prm.l, c.prm.m, Kokkos::cos( theta ), Kokkos::sin( theta ), phi, Y, dYt, dYp_s );
    return Y;
}

// Volumetric body force of the viscosity-1 problem: -g (r/Rp)^k Y r_hat (smooth), 0 (delta).
KOKKOS_INLINE_FUNCTION void body_force_cartesian( const Coeffs& c, const double cx, const double cy, const double cz,
                                                  double& fx, double& fy, double& fz )
{
    if ( c.prm.forcing == Forcing::DELTA )
    {
        fx = fy = fz = 0.0;
        return;
    }
    double r, theta, phi;
    to_spherical( cx, cy, cz, r, theta, phi );
    double Y, dYt, dYp_s;
    harmonic( c.prm.l, c.prm.m, Kokkos::cos( theta ), Kokkos::sin( theta ), phi, Y, dYt, dYp_s );
    const double rho      = ipow( r / c.prm.Rp, c.prm.k ) * Y;
    const double f_radial = -gravity * rho / r;
    fx                    = f_radial * cx;
    fy                    = f_radial * cy;
    fz                    = f_radial * cz;
}

// Real Y_3^2 (the A7 perturbation pattern) and the manufactured-problem viscosity.
KOKKOS_INLINE_FUNCTION double y32( const double cx, const double cy, const double cz )
{
    double r, theta, phi;
    to_spherical( cx, cy, cz, r, theta, phi );
    constexpr double n  = 1.0219854764332823; // sqrt(105/(32 pi))
    const double     st = Kokkos::sin( theta );
    return n * st * st * Kokkos::cos( theta ) * Kokkos::cos( 2.0 * phi );
}
KOKKOS_INLINE_FUNCTION double viscosity( const Visc& v, const double cx, const double cy, const double cz )
{
    if ( v.rmu == 1.0 )
        return 1.0;
    const double r = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    const double T = ( v.Rp - r ) / ( v.Rp - v.Rm ) + ( v.amp != 0.0 ? v.amp * y32( cx, cy, cz ) : 0.0 );
    return Kokkos::pow( v.rmu, 0.5 - T );
}

// f = eta (f0 - grad p) + grad p - 2 eps(u) grad eta, one coefficient side, central differences.
KOKKOS_INLINE_FUNCTION void manufactured_side( const Coeffs& c, const Visc& v, const double cx, const double cy,
                                               const double cz, const int side, double& fx, double& fy, double& fz )
{
    constexpr double h = 1.0e-5;
    const double     x[3] = { cx, cy, cz };
    double           gp[3], ge[3], J[3][3];
    for ( int j = 0; j < 3; ++j )
    {
        double xp[3] = { cx, cy, cz }, xm[3] = { cx, cy, cz };
        xp[j] += h;
        xm[j] -= h;
        double up[3], um[3];
        velocity_cartesian( c, xp[0], xp[1], xp[2], up[0], up[1], up[2], side );
        velocity_cartesian( c, xm[0], xm[1], xm[2], um[0], um[1], um[2], side );
        for ( int i = 0; i < 3; ++i )
            J[i][j] = ( up[i] - um[i] ) / ( 2.0 * h );
        gp[j] = ( pressure_cartesian( c, xp[0], xp[1], xp[2], side ) - pressure_cartesian( c, xm[0], xm[1], xm[2], side ) ) /
                ( 2.0 * h );
        ge[j] = ( viscosity( v, xp[0], xp[1], xp[2] ) - viscosity( v, xm[0], xm[1], xm[2] ) ) / ( 2.0 * h );
    }
    const double eta = viscosity( v, x[0], x[1], x[2] );
    double       f0[3];
    body_force_cartesian( c, cx, cy, cz, f0[0], f0[1], f0[2] );
    double f[3];
    for ( int i = 0; i < 3; ++i )
    {
        double eps_ge = 0.0; // (eps(u) grad eta)_i
        for ( int j = 0; j < 3; ++j )
            eps_ge += 0.5 * ( J[i][j] + J[j][i] ) * ge[j];
        f[i] = eta * ( f0[i] - gp[i] ) + gp[i] - 2.0 * eps_ge;
    }
    fx = f[0];
    fy = f[1];
    fz = f[2];
}

// The body force of the (possibly manufactured) problem at a node. On the delta interface the two
// one-sided values are averaged.
KOKKOS_INLINE_FUNCTION void manufactured_body_force( const Coeffs& c, const Visc& v, const double cx, const double cy,
                                                     const double cz, double& fx, double& fy, double& fz )
{
    if ( v.rmu == 1.0 )
    {
        body_force_cartesian( c, cx, cy, cz, fx, fy, fz );
        return;
    }
    const double r = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    if ( on_interface( c, r ) )
    {
        double ax, ay, az, bx, by, bz;
        manufactured_side( c, v, cx, cy, cz, -1, ax, ay, az );
        manufactured_side( c, v, cx, cy, cz, +1, bx, by, bz );
        fx = 0.5 * ( ax + bx );
        fy = 0.5 * ( ay + by );
        fz = 0.5 * ( az + bz );
        return;
    }
    manufactured_side( c, v, cx, cy, cz, 0, fx, fy, fz );
}

// Exact pressure at a node: the average of both sides on the delta interface.
KOKKOS_INLINE_FUNCTION double nodal_pressure( const Coeffs& c, const double cx, const double cy, const double cz )
{
    const double r = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    if ( on_interface( c, r ) )
        return 0.5 * ( pressure_cartesian( c, cx, cy, cz, -1 ) + pressure_cartesian( c, cx, cy, cz, +1 ) );
    return pressure_cartesian( c, cx, cy, cz, 0 );
}

// Radial surface load per unit area on r' (delta family): -(eta' g Y + (eta' - 1) [p]), [p] = p_hi - p_lo.
KOKKOS_INLINE_FUNCTION double surface_load( const Coeffs& c, const Visc& v, const double cx, const double cy,
                                            const double cz )
{
    const double eta  = viscosity( v, cx, cy, cz );
    const double Y    = Ylm( c, cx, cy, cz );
    const double jump = pressure_cartesian( c, cx, cy, cz, +1 ) - pressure_cartesian( c, cx, cy, cz, -1 );
    return -( eta * gravity * Y + ( eta - 1.0 ) * jump );
}
} // namespace assess_solution
