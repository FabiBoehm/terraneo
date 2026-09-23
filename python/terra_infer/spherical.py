"""Spherical-harmonic transform on the TERRA shell, and an SFNO-style spectral branch.

SAOT's global branch is an AFNO block: FFT the field, mix channels with weights that
carry no mode index, inverse FFT. That is the only part of SAOT that can run at another
resolution -- its weights act on *modes*, and refining the mesh adds modes without
changing the weights. The wavelet branch has no such property (one DWT level means scale
2h, and h moves with the mesh).

An FFT is wrong here: the domain is ten curved diamonds on a spherical shell, not a
periodic box. The spherical harmonics are the right basis on that geometry, so this is
the same idea with the correct transform -- which is what SFNO does for the sphere.

The mesh is an exact radial extrusion (verified: lateral directions agree to 2e-16
across shells), so one lateral transform serves every radial shell.

Analysis is done by pseudo-inverse rather than quadrature. Nodes shared between diamonds
are stored once per subdomain, so a quadrature rule would silently double-count the
seams; the least-squares inverse handles duplicates and non-uniform sampling correctly
and is exact for band-limited fields.
"""

import numpy as np
import torch
import torch.nn as nn

__all__ = ["real_sph_harm", "build_transform", "node_quadrature", "SphericalBranch"]


def node_quadrature(coords: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Per-node quadrature weights for the stored nodes of a mesh (S, nx, ny, nr, 3).

    A uniform mean over stored nodes is a Monte-Carlo estimate under the *storage*
    measure, and that measure moves with the level: seam nodes shared between
    subdomains are stored once per subdomain (21% of lateral nodes at level 3, 6% at
    level 5), and thin regions like the boundary shells occupy a shrinking index
    fraction under refinement. Weighting each node by its volume element |det J| of
    the index->space map (trapezoid-halved on faces) and dividing by its storage
    multiplicity turns such a mean into a discretisation of the continuum integral,
    which converges under refinement instead of drifting with it.

    Returns weights shaped (S, nx, ny, nr), normalised to mean 1.
    """
    c = np.asarray(coords, dtype=np.float64)
    ji = np.gradient(c, axis=1)
    jj = np.gradient(c, axis=2)
    jk = np.gradient(c, axis=3)
    w = np.abs(np.einsum("...i,...i->...", np.cross(ji, jj), jk))
    for ax in (1, 2, 3):
        t = np.ones(c.shape[ax])
        t[0] = t[-1] = 0.5
        sh = [1, 1, 1, 1]
        sh[ax] = c.shape[ax]
        w = w * t.reshape(sh)
    _, inv, cnt = np.unique(np.round(c.reshape(-1, 3), 9), axis=0,
                            return_inverse=True, return_counts=True)
    w = w.reshape(-1) / cnt[inv]
    # normalized=False keeps the PHYSICAL node volumes (sum ~ shell volume).
    # That absolute scale is what converts FE load vectors to strong-form
    # fields; the mean-1 convention silently absorbs it into an unknown
    # constant, which is exactly the amplitude a load/strong conversion loses.
    if normalized:
        w = w / w.mean()
    return w.reshape(c.shape[:4])


def real_sph_harm(dirs: np.ndarray, lmax: int) -> np.ndarray:
    """Real spherical harmonics up to ``lmax`` at unit vectors ``dirs`` (n, 3).

    Returns (n, (lmax+1)^2), ordered l = 0..lmax and m = -l..l.
    """
    from scipy.special import lpmv

    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    ct = np.cos(theta)

    cols = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            # normalisation sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!), built as a ratio so
            # the factorials never overflow
            ratio = 1.0
            for k in range(l - am + 1, l + am + 1):
                ratio /= k
            norm = np.sqrt((2 * l + 1) / (4 * np.pi) * ratio)
            p = lpmv(am, l, ct)
            if m == 0:
                cols.append(norm * p)
            elif m > 0:
                cols.append(np.sqrt(2.0) * norm * p * np.cos(m * phi))
            else:
                cols.append(np.sqrt(2.0) * norm * p * np.sin(am * phi))
    return np.stack(cols, axis=1)


def chebyshev(r: np.ndarray, kmax: int) -> np.ndarray:
    """Chebyshev polynomials T_0..T_kmax on radii mapped to [-1, 1]."""
    lo, hi = r.min(), r.max()
    x = 2.0 * (r - lo) / max(hi - lo, 1e-12) - 1.0
    t = np.arccos(np.clip(x, -1.0, 1.0))
    return np.stack([np.cos(k * t) for k in range(kmax + 1)], axis=1)


_TRANSFORM_CACHE = {}


def build_transform(coords: np.ndarray, lmax: int, kmax: int = 0):
    """Transforms for a mesh (S, nx, ny, nr, 3): lateral, and optionally radial.

    Lateral: ``Y`` (n_lat, n_modes) synthesis and ``A`` (n_modes, n_lat) analysis in
    real spherical harmonics. The lateral directions come from the first radial shell,
    valid because the mesh is a radial extrusion.

    Radial: ``Yr`` (n_rad, kmax+1) and ``Ar`` its pseudo-inverse, in Chebyshev
    polynomials of the radius. Spherical harmonics only span the sphere; without a
    radial basis a branch built on them does no radial mixing at all, and the radial
    direction is where a thin shell has its structure. Together they give a fixed
    (n_modes x kmax+1) coefficient tensor whatever the mesh -- which is what makes the
    branch natively resolution-independent in BOTH directions.
    """
    # Many datasets share one mesh (all the L6 sets, all the L5 sets...), and the
    # pseudo-inverse of a 42250 x 1089 matrix is minutes of work -- memoise it so a run
    # that mixes 13 same-level sets pays for it once.
    _k = (coords.shape, lmax, kmax,
          float(coords.reshape(-1)[0]), float(coords.reshape(-1)[-1]),
          float(coords.sum()))
    _hit = _TRANSFORM_CACHE.get(_k)
    if _hit is not None:
        return _hit

    d = coords[:, :, :, 0, :]
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    Y = real_sph_harm(d.reshape(-1, 3), lmax)
    A = np.linalg.pinv(Y)
    out = [torch.as_tensor(Y, dtype=torch.float32),
           torch.as_tensor(A, dtype=torch.float32)]
    if kmax:
        r = np.linalg.norm(coords[0, 0, 0, :, :], axis=-1)
        Yr = chebyshev(r, kmax)
        out += [torch.as_tensor(Yr, dtype=torch.float32),
                torch.as_tensor(np.linalg.pinv(Yr), dtype=torch.float32)]
    res = tuple(out)
    _TRANSFORM_CACHE[_k] = res
    return res
