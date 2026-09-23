r"""3-D wavelet-attention operator for TERRA-NG shell fields.

This is SAOT's wavelet-attention branch lifted to three dimensions, with the
Fourier branch and the gated fusion removed. Reference: Zhou, Chen & Yang,
"SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs"
(AAAI 2026), https://github.com/chenhong-zhou/SAOT. Nothing is imported from
that checkout -- the linear attention and the Haar transforms are reimplemented
here so this module stands alone.

\section why Why the Fourier branch is gone

SAOT's spectral branch is an AFNO-style complex MLP built on ``rfft2``, which
assumes a periodic rectangular grid. On a thick spherical shell that assumption
fails on both axes: laterally the domain is ten curved diamonds with pentagonal
corners, radially it is bounded rather than periodic. It is also 3% of the
module's parameters, and the soft-shrinkage that makes AFNO *adaptive* is not
implemented upstream. The principled replacement would be a spherical harmonic
transform, not an FFT; until that exists there is nothing to gate against, so
the fusion gate goes with it.

\section dwt The 3-D Haar transform

Along each axis a signal splits into lowpass and highpass halves, so in 3-D
there are 2^3 = 8 subbands (LLL, LLH, ... HHH), each at half the extent in every
direction. Channels go up 8x, volume goes down 8x: element count is preserved
exactly and the transform is invertible, which is the whole point -- unlike a
strided convolution it discards nothing.

Paired with a 1x1x1 convolution that first cuts C -> C/8, the composition is a
lossless **8x token reduction at constant channel width**. That is twice the
reduction SAOT gets in 2-D, and it matters more here because attention cost is
what limits volumetric operator learning.

\section subdomains Local transform, global attention

A TERRA-NG field arrives as ten curved diamonds, ``(n_subdomains, nx, ny, nr, C)``. The
wavelet transform has to stay *inside* a subdomain -- it is a strided convolution and
needs a regular grid, and the ten diamonds are not one contiguous block. Attention has
no such constraint: it is permutation-equivariant over tokens and does not care whether
they are laid out contiguously.

So the two are split. ``reduce``/DWT/``filter`` run per subdomain, then every subdomain's
subband tokens are pooled into one sequence and attention runs over all of them at once.
The model therefore sees the whole shell in a single forward pass and can couple across
diamond seams, which a per-subdomain model cannot do at any depth. Token count is
``n_subdomains * N/8`` -- 1250 at level 3, 359k at level 6, both comfortable for linear
attention.

\section radial What this buys over the per-shell adapter

``saot_adapter`` maps each (subdomain, radial shell) to an independent 2-D image,
so the model never sees a radial derivative. Here the whole subdomain volume
``(nx, ny, nr)`` is one sample and the DWT couples all three axes, so radial
structure -- plumes, boundary layers, the viscosity profile -- is inside the
model's receptive field rather than outside it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spherical import RadiusAttention, SliceAttention, SphericalBranch, radius_neighbors

# --------------------------------------------------------------------------- Haar 3-D


def _haar_filters_3d(dtype=torch.float32) -> torch.Tensor:
    """The eight separable Haar analysis filters, shaped ``(8, 1, 2, 2, 2)``.

    Haar is orthogonal, so the same filters serve for synthesis and the
    round trip is the identity -- asserted in :func:`self_test`.
    """
    lo = torch.tensor([1.0, 1.0], dtype=dtype) / np.sqrt(2.0)
    hi = torch.tensor([-1.0, 1.0], dtype=dtype) / np.sqrt(2.0)

    banks = []
    for fx in (lo, hi):
        for fy in (lo, hi):
            for fz in (lo, hi):
                # outer product over the three axes -> a 2x2x2 stencil
                w = fx.view(2, 1, 1) * fy.view(1, 2, 1) * fz.view(1, 1, 2)
                banks.append(w)
    return torch.stack(banks).unsqueeze(1)  # (8, 1, 2, 2, 2)


class DWT3D(nn.Module):
    """One-level 3-D Haar transform: ``(B, C, X, Y, R) -> (B, 8C, X/2, Y/2, R/2)``.

    Implemented as a grouped stride-2 convolution with fixed filters, so it is a
    single cuDNN call with no parameters. Subbands are concatenated on the
    channel axis in the order produced by :func:`_haar_filters_3d`.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("filters", _haar_filters_3d(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        w = self.filters.to(x.dtype).repeat(c, 1, 1, 1, 1)  # (8C, 1, 2, 2, 2)
        y = F.conv3d(x.contiguous(), w, stride=2, groups=c)
        # conv3d with groups=C emits [c0s0..c0s7, c1s0..], i.e. channel-major in c.
        return y.view(b, c, 8, *y.shape[2:]).transpose(1, 2).reshape(b, 8 * c, *y.shape[2:])


class IDWT3D(nn.Module):
    """Inverse of :class:`DWT3D`: ``(B, 8C, X, Y, R) -> (B, C, 2X, 2Y, 2R)``."""

    def __init__(self):
        super().__init__()
        self.register_buffer("filters", _haar_filters_3d(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c8 = x.shape[0], x.shape[1]
        c = c8 // 8
        # back to channel-major so grouped transposed conv sees each channel's 8 subbands
        x = x.view(b, 8, c, *x.shape[2:]).transpose(1, 2).reshape(b, c8, *x.shape[2:])
        w = self.filters.to(x.dtype).repeat(c, 1, 1, 1, 1)  # (8C, 1, 2, 2, 2)
        return F.conv_transpose3d(x.contiguous(), w, stride=2, groups=c)


# --------------------------------------------------------------------- linear attention


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6):
    """Katharopoulos et al. linear attention with the ``elu(x)+1`` feature map.

    ``V' = phi(Q) (phi(K)^T V) / (phi(Q) . sum phi(K))`` -- the N x N matrix is never
    formed, so cost is O(N D^2) rather than O(N^2 D). Inputs are ``(B, N, H, D)``.
    """
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0

    kv = torch.einsum("bnhd,bnhm->bhmd", k, v)
    z = 1.0 / (torch.einsum("bnhd,bhd->bnh", q, k.sum(dim=1)) + eps)
    return torch.einsum("bnhd,bhmd,bnh->bnhm", q, kv, z)


# ------------------------------------------------------------------ wavelet attention 3D


class SpectralMix(nn.Module):
    """Attention on the 3-D Haar subbands of a volumetric feature field.

    Two layouts, selected by ``band_tokens``:

    **channel layout (default).** Subbands live on the channel axis. A token is a spatial
    position carrying all eight bands in its channels, so band identity is dissolved by
    the dense ``qkv`` and cross-scale mixing is implicit in channel mixing. Cheap:
    ``reduce`` (C -> C/8) before the DWT (x8) makes the pair a lossless 8x token
    reduction at constant width.

    **band-token layout.** Each (position, subband) pair is its own token, so attention
    computes an explicit weight between band i at position p and band j at position q --
    genuine cross-scale attention rather than channel mixing. ``reduce`` is dropped so
    tokens keep full width, which costs 8x the token count (1250 -> 10000 per sample at
    level 3). A learned per-band embedding tells each token which scale it came from.

    With ``n_levels > 1`` the transform becomes a **pyramid**: the LLL band is fed back
    through the DWT, classically, and every level's detail bands plus the coarsest LLL
    join the same sequence. Each level costs 1/8 of the previous, so depth is nearly
    free in tokens -- two levels is 1091 tokens per subdomain against 1000 for one,
    a 9% increase for a second scale. A coarse token then attends to a fine token with
    its own weight, which is the property the channel layout cannot express at all.

    ``reduce`` cuts C -> C/8 and the DWT multiplies channels by 8, so the pair is a
    lossless 8x spatial reduction that lands back at C channels. Attention then runs
    on the N/8 subband tokens; the result is inverted back to full resolution,
    concatenated with the untouched input and projected.
    """

    def __init__(self, dim: int, n_heads: int = 8, use_filter: bool = True,
                 band_tokens: bool = False, n_levels: int = 1,
                 attention: str = "linear", spherical: int = 0,
                 wavelet: bool = True, per_degree: bool = False, n_slices: int = 0,
                 sph_couple: bool = False, radial_modes: int = 0,
                 sph_couple_band: int = 0, gno: bool = False,
                 sph_degree_mlp: bool = False):
        super().__init__()
        if attention not in ("linear", "softmax"):
            raise ValueError(f"attention must be linear|softmax, got {attention!r}")
        # Linear attention has no temperature -- scaling phi(q) cancels exactly between
        # numerator and denominator -- and the weights are an inner product of positive
        # bounded vectors, giving a max/median dynamic range of ~3.6 against softmax's
        # ~3200 on the same tokens. That is why the measured attention sits at 0.95-0.99
        # of maximum entropy with no spatial locality, and why neither more kv capacity
        # nor an explicit positional encoding moved it. softmax restores the exponential.
        self.attention = attention
        self.wavelet = wavelet
        self.band_tokens = band_tokens
        self.n_levels = max(1, n_levels)
        if dim % 8 != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by 8 (one channel per subband)")
        if dim % n_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by n_heads {n_heads}")

        self.n_heads = n_heads
        self.dwt = DWT3D()
        self.idwt = IDWT3D()

        self.reduce = nn.Sequential(
            nn.Conv3d(dim, dim // 8, kernel_size=1),
            nn.GroupNorm(1, dim // 8),  # BatchNorm would tie the operator to the batch
        ) if wavelet else None
        self.filter = (
            nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, padding=1), nn.GroupNorm(1, dim))
            if use_filter and wavelet
            else None
        )

        if band_tokens:
            # No `reduce`: tokens keep full width. The filter is depthwise, because a
            # dense 3x3x3 conv on 8C channels would be 7M parameters on its own.
            self.reduce = None
            self.filter = (nn.Sequential(
                nn.Conv3d(dim * 8, dim * 8, 3, padding=1, groups=dim * 8),
                nn.GroupNorm(1, dim * 8)) if use_filter else None)
            # one embedding per (level, band), plus one for the coarsest LLL
            self.band_emb = nn.Parameter(torch.zeros(self.n_levels * 7 + 1, dim))

        # SAOT's second branch was an FFT block; on a spherical shell the correct
        # transform is the spherical harmonics. Its weights carry no mode index, so it
        # is the resolution-independent half of the block -- the wavelet half is tied to
        # the grid through the DWT.
        self.sph = (SphericalBranch(dim, n_blocks=n_heads, lmax=spherical,
                            per_degree=per_degree, couple=sph_couple,
                            n_radial=radial_modes,
                            couple_band=sph_couple_band,
                            degree_mlp=sph_degree_mlp) if spherical else None)
        self.merge = nn.Linear(dim * 2, dim) if spherical else None
        # An additive branch: attention over a fixed set of learned slices, which is
        # invariant where node-token attention cannot be.
        self.slice_attn = SliceAttention(dim, n_slices, n_heads) if n_slices else None
        # Local branch: attention over a fixed physical neighborhood (see
        # RadiusAttention) -- communicates the features the band-limited spectral
        # branch cannot, without the index-space locality that broke the wavelets.
        self.gno = RadiusAttention(dim) if gno else None

        self.qkv = (nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 3))
                    if wavelet else None)
        # concat of the untouched input with the wavelet path
        self.proj = (nn.Linear(dim + (dim if band_tokens else dim // 8), dim)
                     if wavelet else None)

    def _spherical(self, x, shape, sht, b, s_dom, n, c):
        nx, ny, nr = shape
        lat = x.reshape(b, s_dom, nx, ny, nr, c).permute(0, 5, 1, 2, 3, 4)
        lat = lat.reshape(b, c, s_dom * nx * ny, nr)
        out = self.sph(lat, *sht)
        out = out.reshape(b, c, s_dom, nx, ny, nr).permute(0, 2, 3, 4, 5, 1)
        return out.reshape(b, s_dom, n, c)

    def forward(self, x: torch.Tensor, shape: tuple[int, int, int],
                sht=None, node_q: "torch.Tensor | None" = None,
                gno_idx: "torch.Tensor | None" = None) -> torch.Tensor:
        """``x``: (B, S, N, C) -- batch, subdomains, nodes, channels. ``shape``: (nx, ny, nr).

        The transform is per subdomain (S folds into the batch); the attention is over
        every subdomain's tokens at once (S folds into the sequence). ``node_q`` are
        optional per-node quadrature weights for the slice pooling.
        """
        b, s_dom, n, c = x.shape
        nx, ny, nr = shape

        slice_out = None
        if self.slice_attn is not None:
            flat = x.reshape(x.shape[0], -1, x.shape[-1])
            slice_out = self.slice_attn(flat, node_q).reshape(x.shape)
        if self.gno is not None:
            flat = x.reshape(x.shape[0], -1, x.shape[-1])
            # Checkpointed: the chunked gather is cheap to recompute, and holding its
            # activations for all 8 layers is what ran a level-4 batch out of memory.
            if torch.is_grad_enabled() and self.training:
                gno_out = torch.utils.checkpoint.checkpoint(
                    self.gno, flat, gno_idx, use_reentrant=False)
            else:
                gno_out = self.gno(flat, gno_idx)
            gno_out = gno_out.reshape(x.shape)
            slice_out = gno_out if slice_out is None else slice_out + gno_out

        if not self.wavelet and self.sph is None:
            # no branches at all: the block degenerates to a pointwise MLP, which is
            # the control for whether the spectral coupling does anything
            return slice_out if slice_out is not None else torch.zeros_like(x)

        if not self.wavelet:
            out = self.merge(torch.cat(
                [x, self._spherical(x, shape, sht, b, s_dom, n, c)], dim=-1))
            return out if slice_out is None else out + slice_out

        vol = x.reshape(b * s_dom, nx, ny, nr, c).permute(0, 4, 1, 2, 3)  # (B*S, C, X, Y, R)

        if self.band_tokens:
            bs = b * s_dom
            cur, details, geom, pads = vol, [], [], []
            for _ in range(self.n_levels):
                pad = tuple(d % 2 for d in cur.shape[2:])           # DWT halves each axis
                if any(pad):
                    cur = F.pad(cur, (0, pad[2], 0, pad[1], 0, pad[0]), mode="replicate")
                pads.append(pad)
                sub = self.dwt(cur)                                  # (BS, 8C, ...) LLL first
                if self.filter is not None:
                    sub = self.filter(sub)
                geom.append(sub.shape[2:])
                details.append(sub[:, c:])                           # the 7 detail bands
                cur = sub[:, :c]                                     # recurse on LLL

            # every level's detail bands, plus the coarsest LLL, in ONE sequence
            seq, e = [], 0
            for lv, det in enumerate(details):
                npos = int(np.prod(geom[lv]))
                t = det.reshape(bs, 7, c, npos).permute(0, 1, 3, 2)
                seq.append((t + self.band_emb[None, e:e + 7, None, :]).reshape(bs, 7 * npos, c))
                e += 7
            npos_c = int(np.prod(geom[-1]))
            seq.append((cur.reshape(bs, c, npos_c).transpose(1, 2)
                        + self.band_emb[None, e, None, :]))
            lens = [t.shape[1] for t in seq]
            tokens = torch.cat(seq, dim=1).reshape(b, -1, c)

            qkv = self.qkv(tokens).reshape(b, -1, 3, self.n_heads, c // self.n_heads)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            att = linear_attention(q, k, v).reshape(bs, sum(lens), c)

            parts, o = [], 0
            for L in lens:
                parts.append(att[:, o:o + L]); o += L
            cur = parts[-1].transpose(1, 2).reshape(bs, c, *geom[-1])
            for lv in range(self.n_levels - 1, -1, -1):
                npos = int(np.prod(geom[lv]))
                det = parts[lv].reshape(bs, 7, npos, c).permute(0, 1, 3, 2).reshape(bs, 7 * c, *geom[lv])
                cur = self.idwt(torch.cat([cur, det], dim=1))
                px, py, pr = pads[lv]
                if px: cur = cur[:, :, :-px]
                if py: cur = cur[:, :, :, :-py]
                if pr: cur = cur[:, :, :, :, :-pr]

            back = cur.permute(0, 2, 3, 4, 1).reshape(b, s_dom, n, c)
            return self.proj(torch.cat([x, back], dim=-1))

        sub = self.dwt(self.reduce(vol))
        if self.filter is not None:
            sub = self.filter(sub)

        bs, cs, sx, sy, sr = sub.shape
        # (B*S, C, n) -> (B, S*n, C): one sequence spanning the whole shell.
        tokens = sub.reshape(b, s_dom, cs, -1).permute(0, 1, 3, 2).reshape(b, -1, cs)

        qkv = self.qkv(tokens).reshape(b, -1, 3, self.n_heads, cs // self.n_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.attention == "softmax":
            # fused SDPA: the N x N matrix is never materialised, so the cost is the
            # 1.68x per block that was estimated, not a memory blow-up
            att = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            attended = att.transpose(1, 2).reshape(b, s_dom, -1, cs)
        else:
            attended = linear_attention(q, k, v).reshape(b, s_dom, -1, cs)

        cur = attended.permute(0, 1, 3, 2).reshape(bs, cs, sx, sy, sr)
        cur = self.idwt(cur)                                    # (B*S, C/8, ...)
        back = cur.permute(0, 2, 3, 4, 1).reshape(b, s_dom, n, c // 8)
        wave = self.proj(torch.cat([x, back], dim=-1))
        if self.sph is None or sht is None:
            return wave

        # the shell is a radial extrusion, so the lateral index is (subdomain, i, j)
        # and the radial index rides along as an independent axis
        lat = x.reshape(b, s_dom, nx, ny, nr, c).permute(0, 5, 1, 2, 3, 4)
        lat = lat.reshape(b, c, s_dom * nx * ny, nr)
        sph = self._spherical(x, shape, sht, b, s_dom, n, c)
        out = self.merge(torch.cat([wave, sph], dim=-1))
        return out if slice_out is None else out + slice_out


class Block(nn.Module):
    """Pre-norm block: wavelet attention, then an MLP, each with a residual."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 2, use_filter: bool = True,
                 band_tokens: bool = False, n_levels: int = 1, attention: str = "linear",
                 spherical: int = 0, wavelet: bool = True,
                 per_degree: bool = False, n_slices: int = 0,
                 sph_couple: bool = False, radial_modes: int = 0,
                 sph_couple_band: int = 0, gno: bool = False,
                 sph_degree_mlp: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SpectralMix(dim, n_heads, use_filter, band_tokens, n_levels,
                                       attention, spherical, wavelet, per_degree,
                                       n_slices, sph_couple, radial_modes,
                                       sph_couple_band, gno, sph_degree_mlp)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, shape, sht=None, node_q=None, gno_idx=None):
        x = x + self.attn(self.ln1(x), shape, sht, node_q, gno_idx)
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    """Volumetric operator: ``(B, nx, ny, nr, in_ch) -> (B, nx, ny, nr, out_ch)``.

    Normalised (i, j, k) coordinates are concatenated onto the input, matching how
    SAOT feeds ``space_dim`` coordinates to its structured-mesh model. Odd extents --
    TERRA's are 2^L+1, so always odd -- are padded by one and cropped at the end,
    because the DWT halves each axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shape: tuple[int, int, int],
        n_hidden: int = 64,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 2,
        use_filter: bool = True,
        band_tokens: bool = False,
        n_levels: int = 1,
        attention: str = "linear",
        multigrid: bool = False,
        spherical: int = 0,
        radial_modes: int = 0,
        wavelet: bool = True,
        per_degree: bool = False,
        n_slices: int = 0,
        mass_slices: bool = False,
        sph_couple: bool = False,
        sph_couple_band: int = 0,
        sph_couple_shared: bool = False,
        sph_degree_mlp: bool = False,
        gno_radius: float = 0.0,
        gno_k: int = 32,
        head_mlp: bool = False,
        coords: "np.ndarray | None" = None,
        n_subdomains: int = 10,
    ):
        super().__init__()
        self.shape_in = tuple(shape)
        self.pad = tuple(s % 2 for s in shape)
        self.shape = tuple(s + p for s, p in zip(shape, self.pad))

        # Geometry channels. Without `coords` the model only gets normalised *index*
        # coordinates, which are identical in all ten diamonds -- a node at (4,4,4) looks
        # the same wherever it is on the sphere. That blindness shows up as subdomain-
        # shaped blocks in the prediction. Real Cartesian positions plus normalised depth
        # tell it where each node actually is.
        n_geom = 4 if coords is not None else 3
        self.lift = nn.Sequential(
            nn.Linear(in_channels + n_geom, n_hidden * 2), nn.GELU(),
            nn.Linear(n_hidden * 2, n_hidden)
        )
        self.multigrid = multigrid
        self.spherical = spherical
        self.radial_modes = radial_modes
        self.wavelet = wavelet
        self.blocks = nn.ModuleList(
            [Block(n_hidden, n_heads, mlp_ratio, use_filter, band_tokens, n_levels,
                   attention, spherical, wavelet,
                   per_degree, n_slices, sph_couple, radial_modes,
                   sph_couple_band, gno_radius > 0, sph_degree_mlp)
             for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(n_hidden)
        # A single Linear(H -> 4) already gives each output its own row, so "splitting"
        # it changes nothing -- the sharing is in the trunk, not the head. What does
        # differ is giving each field its own NONLINEAR readout.
        if head_mlp and out_channels == 4:
            self.head = None
            self.head_u = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU(),
                                        nn.Linear(n_hidden, 3))
            self.head_p = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU(),
                                        nn.Linear(n_hidden, 1))
        else:
            self.head = nn.Linear(n_hidden, out_channels)

        if coords is None:
            self.register_buffer("geom", self._grid(self.shape), persistent=False)
        else:
            self.register_buffer("geom", self._physical(coords, self.pad), persistent=False)
        if spherical:
            if coords is None:
                raise ValueError("the spherical branch needs the mesh coordinates")
            t = self._build_sht(coords, self.pad)
            for nm, v in zip(("sht_Y", "sht_A", "sht_Yr", "sht_Ar"), t):
                self.register_buffer(nm, v, persistent=False)
        else:
            self.sht_Y = self.sht_A = None
        if not radial_modes:
            self.sht_Yr = self.sht_Ar = None
        # One coupling shared by every layer: the physical mode-coupling operator does
        # not change with depth, and sharing cuts its parameters by n_layers, which is
        # the overfitting margin the free-form version lost by.
        if sph_couple and sph_couple_shared:
            ref = self.blocks[0].attn.sph
            for blk in self.blocks[1:]:
                s = blk.attn.sph
                s.c_ln, s.c_qkv, s.c_out = ref.c_ln, ref.c_qkv, ref.c_out
        self.mass_slices = mass_slices
        if n_slices and mass_slices:
            if coords is None:
                raise ValueError("mass-weighted slices need the mesh coordinates")
            self.register_buffer("node_q", self._node_q(coords, self.pad),
                                 persistent=False)
        else:
            self.node_q = None
        self.gno_radius, self.gno_k = gno_radius, gno_k
        if gno_radius > 0:
            if coords is None:
                raise ValueError("radius attention needs the mesh coordinates")
            self.register_buffer("gno_idx",
                                 self._gno_idx(coords, self.pad, gno_radius, gno_k),
                                 persistent=False)
        else:
            self.gno_idx = None

    @staticmethod
    def _gno_idx(coords, pad, radius, k):
        """Frozen physical-radius neighbor sample on the padded grid."""
        c = np.asarray(coords, dtype=np.float64)
        px, py, pr = pad
        if px:
            c = np.concatenate([c, c[:, -1:]], axis=1)
        if py:
            c = np.concatenate([c, c[:, :, -1:]], axis=2)
        if pr:
            c = np.concatenate([c, c[:, :, :, -1:]], axis=3)
        return torch.as_tensor(radius_neighbors(c, radius, k), dtype=torch.long)

    @staticmethod
    def _node_q(coords, pad):
        """Quadrature weights for the slice pooling; zero on the replicated pad slices."""
        from .spherical import node_quadrature

        q = node_quadrature(coords)
        px, py, pr = pad
        q = np.pad(q, ((0, 0), (0, px), (0, py), (0, pr)))
        return torch.as_tensor(q.reshape(-1), dtype=torch.float32)

    def _build_sht(self, coords, pad):
        """Synthesis/analysis matrices for the padded lateral grid of this mesh."""
        from .spherical import build_transform

        c = np.asarray(coords, dtype=np.float64)
        px, py, pr = pad
        if px:
            c = np.concatenate([c, c[:, -1:]], axis=1)
        if py:
            c = np.concatenate([c, c[:, :, -1:]], axis=2)
        if pr:
            c = np.concatenate([c, c[:, :, :, -1:]], axis=3)
        return build_transform(c, self.spherical, self.radial_modes)

    def set_mesh(self, shape, coords):
        """Point the model at a different mesh, keeping every weight.

        Nothing learned depends on the discretisation -- only the padded extents, the
        geometry buffer and the DWT depth do. Swapping them between batches is what
        allows one model to be trained on several resolutions at once, which is how the
        variable-depth wavelet chain stops being extrapolation at the finer meshes.
        """
        dev = next(self.parameters()).device
        self.shape_in = tuple(shape)
        self.pad = tuple(s % 2 for s in shape)
        self.shape = tuple(s + p for s, p in zip(shape, self.pad))
        geom = (self._physical(coords, self.pad) if coords is not None
                else self._grid(self.shape))
        self.register_buffer("geom", geom.to(dev), persistent=False)
        if self.spherical:
            t = self._build_sht(coords, self.pad)
            for nm, v in zip(("sht_Y", "sht_A", "sht_Yr", "sht_Ar"), t):
                self.register_buffer(nm, v.to(dev), persistent=False)
        if self.node_q is not None:
            self.register_buffer("node_q", self._node_q(coords, self.pad).to(dev),
                                 persistent=False)
        if self.gno_idx is not None:
            self.register_buffer(
                "gno_idx",
                self._gno_idx(coords, self.pad, self.gno_radius, self.gno_k).to(dev),
                persistent=False)
        return self

    @staticmethod
    def _physical(coords, pad):
        """(S, nx, ny, nr, 3) Cartesian nodes -> (1, S, N, 4) geometry features.

        Channels are x, y, z and the normalised depth (r - r_min)/(r_max - r_min).
        Radius is redundant with the Cartesian triple but the network would have to
        learn a square root to recover it, and depth is the direction the physics
        actually varies along.
        """
        c = torch.as_tensor(np.ascontiguousarray(coords), dtype=torch.float32)
        px, py, pr = pad
        # Replicate the edge so the padded volume matches the field's padding. The
        # padded slice is cropped from the output; it only touches the DWT.
        if px:
            c = torch.cat([c, c[:, -1:]], dim=1)
        if py:
            c = torch.cat([c, c[:, :, -1:]], dim=2)
        if pr:
            c = torch.cat([c, c[:, :, :, -1:]], dim=3)

        r = c.norm(dim=-1, keepdim=True)
        r_min, r_max = float(r.min()), float(r.max())
        depth = (r - r_min) / max(r_max - r_min, 1e-12)
        feats = torch.cat([c, depth], dim=-1)                 # (S, X, Y, R, 4)
        s_dom = feats.shape[0]
        return feats.reshape(1, s_dom, -1, 4)

    @staticmethod
    def _grid(shape):
        nx, ny, nr = shape
        i = torch.linspace(0, 1, nx).view(nx, 1, 1, 1).expand(nx, ny, nr, 1)
        j = torch.linspace(0, 1, ny).view(1, ny, 1, 1).expand(nx, ny, nr, 1)
        k = torch.linspace(0, 1, nr).view(1, 1, nr, 1).expand(nx, ny, nr, 1)
        return torch.cat((i, j, k), dim=-1).reshape(1, 1, nx * ny * nr, 3)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        """``fx``: (B, S, nx, ny, nr, C_in) -> (B, S, nx, ny, nr, C_out).

        With ``multigrid`` set, any finer mesh is handled by restricting the input to
        the trained mesh, running the network there, and prolongating the prediction --
        both parameter-free, both applied ONCE at the model boundary. Restricting inside
        the blocks instead was tried four ways and all of them failed (0.62-3.81),
        because the residual stream then gets restricted and re-interpolated eight times
        while the pointwise layers stay at full resolution.

        The meshes are nested (2^L+1 per axis, coarse coordinates agree with the fine
        ones to 0.0), so restriction is a stride and the network sees exactly the
        discretisation it was trained on.

        A field with no batch axis, (S, nx, ny, nr, C) as the solver sends it, is
        accepted and returned in the same shape.
        """
        squeeze = fx.ndim == 5
        if squeeze:
            fx = fx.unsqueeze(0)

        if self.multigrid and tuple(fx.shape[2:5]) != self.shape_in:
            fine = tuple(fx.shape[2:5])
            st = [(f - 1) // (c - 1) for f, c in zip(fine, self.shape_in)]
            if all(s > 1 and (f - 1) % (c - 1) == 0
                   for s, f, c in zip(st, fine, self.shape_in)):
                coarse = fx[:, :, ::st[0], ::st[1], ::st[2]]
                out = self.forward(coarse if not squeeze else coarse[0])
                if squeeze:
                    out = out.unsqueeze(0)
                o = out.permute(0, 1, 5, 2, 3, 4).reshape(-1, out.shape[-1], *self.shape_in)
                o = F.interpolate(o, size=fine, mode="trilinear", align_corners=True)
                o = o.reshape(out.shape[0], out.shape[1], out.shape[-1], *fine)
                o = o.permute(0, 1, 3, 4, 5, 2)
                return o.squeeze(0) if squeeze else o

        b, s_dom = fx.shape[0], fx.shape[1]

        px, py, pr = self.pad
        fx = F.pad(fx, (0, 0, 0, pr, 0, py, 0, px))

        nx, ny, nr = self.shape
        fx = fx.reshape(b, s_dom, nx * ny * nr, -1)
        geom = self.geom.to(fx.dtype)
        geom = geom.expand(b, s_dom, -1, -1) if geom.shape[1] == 1 else geom.expand(b, -1, -1, -1)
        fx = torch.cat([geom, fx], dim=-1)
        fx = self.lift(fx)

        sht = ((self.sht_Y, self.sht_A, self.sht_Yr, self.sht_Ar)
               if self.spherical else None)
        for block in self.blocks:
            fx = block(fx, self.shape, sht, self.node_q, self.gno_idx)

        fx = self.ln_out(fx)
        fx = (self.head(fx) if self.head is not None
              else torch.cat([self.head_u(fx), self.head_p(fx)], dim=-1))
        fx = fx.reshape(b, s_dom, nx, ny, nr, -1)

        if px:
            fx = fx[:, :, :-px]
        if py:
            fx = fx[:, :, :, :-py]
        if pr:
            fx = fx[:, :, :, :, :-pr]
        return fx.squeeze(0) if squeeze else fx


class LinearOperator(nn.Module):
    """A learned preconditioner that is exactly LINEAR in its input field.

    Krylov methods assume the preconditioner behaves like a fixed linear
    operator: superposition, homogeneity ``M(a r) = a M(r)``, consistency
    across the orthogonalised directions the iteration produces. A nonlinear
    network has none of these -- the rough content of a residual shifts every
    activation, so it corrupts the answer for the smooth part riding
    underneath. Here every path from input to output is linear and bias-free,
    so the properties hold to machine precision. The practical payoff: a
    linear operator trained on ANY input distribution that spans the space is
    determined on the WHOLE space, so the training-distribution question
    dissolves.

    Geometry enters only multiplicatively -- FiLM gates computed from the
    fixed node coordinates -- which preserves linearity in the field (the
    gates are constants of the mesh, not functions of the input).

    Structure, all residual sums (linear too):

      bias-free lift, geometry-gated
        -> per-degree spectral GREEN operator: one full (channel-block x
           radial Chebyshev mode) matrix per SH degree l. This is the matrix
           form of a rotationally invariant solution operator -- exactly the
           class the eta = 1 Stokes inverse lives in -- and it mixes radial
           modes, which the diagonal per-degree branch never does.
        -> linear 3^3 stencils per subdomain: the local complement for the
           above-truncation content the spectral basis cannot represent.
        -> bias-free head, geometry-gated.
    """

    mesh_buffer_names = ("geom", "sht_Y", "sht_A", "sht_Yr", "sht_Ar", "deg", "ordf",
                         "seam_idx", "seam_cnt")

    def __init__(self, in_channels, out_channels, shape, coords,
                 n_hidden=128, n_blocks=8, lmax=16, kmax=8, n_conv=2,
                 kernel=3, depth_gates=False, radial_dense=False, pyramid=0,
                 green_mlp=False, eta_gates=False, eta_green=False,
                 nonlin=False, dilated=False, level_cond=False,
                 level_pyramid=False, stencil_scale=False, eta_lateral=0,
                 eta_stencils=0, multi_dilation=False, mode_attn=0,
                 bank_bottleneck=0, sep_stencils=False, channels_last=False,
                 seam_average=False, eta_embed_dim=16, eta_quant=0, grad_checkpoint=False):
        super().__init__()
        self.nonlin = nonlin
        # dilated: the local stencils keep a FIXED PHYSICAL footprint across
        # levels (dilation 2^(L-3), reference L3 = 9 nodes per side), instead
        # of the index-space (h-relative) footprint of a smoother.
        # level_cond: log2 of the refinement relative to L3 is appended to the
        # eta-profile embedding, so the generated Green spectra can depend on h.
        self.dilated = dilated
        self.level_cond = level_cond
        # level_pyramid: use 1 + log2(refinement) of the ``pyramid`` coarse
        # stencil levels, so the coarsest pyramid level always sits at the
        # L3 footprint (5^3 per subdomain) whatever the mesh level.
        self.level_pyramid = level_pyramid
        self.active_pyr = pyramid
        # stencil_scale: each stencil layer's output is multiplied by
        # (h_L3 / h)^q with a LEARNED exponent q per layer (init 0). A
        # differential-type local kernel wants q ~ 2, an integral-type one
        # q ~ -3; letting the data choose is the cheap version of DISCO's
        # resolution-consistent local kernels.
        self.stencil_scale = stencil_scale
        # eta_lateral = L > 0: the spherical-harmonic coefficients (degree <= L)
        # of the standardised log-eta on each shell, resampled to 16 radial
        # points, are fed to the Green hypernetwork next to the radial
        # mean/std profile -- the per-sample spectra can then react to the
        # LATERAL viscosity structure (high-contrast samples), not only to
        # the radial one.
        self.eta_lateral = int(eta_lateral)
        # eta_stencils = K > 0: VISCOSITY-DEPENDENT local stencils. Each stencil
        # layer owns K weight banks; a per-node MLP of the local viscosity
        # (standardised log-eta and its 5^3 neighbourhood mean/std) mixes them
        # (softmax), so the effective kernel changes across viscosity jumps --
        # where the shared-kernel model fails (error grows with contrast).
        # The mixing depends on eta only: still exactly linear in f.
        self.eta_stencils = int(eta_stencils)
        # multi_dilation: DCNO-style local branch -- each stencil layer adds
        # 3^3 kernels at dilations 2 and 4 (on top of the dense 5^3 kernel),
        # so the local path spans 5, 9 and 17 nodes at no extra depth: the
        # multiscale structure that high-contrast viscosity imposes on u.
        self.multi_dilation = multi_dilation
        # mode_attn = d > 0: ETA-KEYED CROSS-DEGREE ATTENTION in the spectral
        # core. Queries/keys come from the viscosity's own spectral
        # coefficients (radial profile per (l, m) mode, resampled to 16 pts,
        # plus degree/order position), values are the operator output per
        # mode: A = softmax(q k^T / sqrt(d)) is a (modes x modes) mixing that
        # depends on eta only, so the map stays exactly linear in f while
        # lateral viscosity structure can scatter energy between degrees --
        # the coupling a per-degree (block-diagonal) Green core cannot express.
        # The residual gate starts at 0: warm starts are function-preserving.
        self.mode_attn = int(mode_attn)
        if self.mode_attn > 0:
            self.attn_q = nn.Linear(18, self.mode_attn)
            self.attn_k = nn.Linear(18, self.mode_attn)
            self.attn_gate = nn.Parameter(torch.zeros(n_blocks))
        self.conv_scale_pow = nn.Parameter(torch.zeros(n_conv)) if stencil_scale else None
        self.stencil_dilation = max(1, (int(shape[1]) - 1) // 8) if dilated else 1
        self.level_scale = float(np.log2(max(1, (int(shape[1]) - 1) // 8)))
        if eta_green:
            if not eta_gates:
                raise ValueError("eta_green needs eta_gates (viscosity channel)")
            green_mlp = True
        if coords is None:
            raise ValueError("LinearOperator needs the mesh coordinates")
        if n_hidden % n_blocks:
            raise ValueError(f"hidden {n_hidden} must divide into {n_blocks} blocks")
        self.shape_in = tuple(shape)
        self.lmax, self.kmax, self.n_blocks = lmax, kmax, n_blocks

        h = n_hidden
        # eta_gates: the viscosity is moved OUT of the linear input channels
        # and INTO the multiplicative gates. The true Stokes inverse depends on
        # eta multiplicatively, which a channel-linear model cannot express;
        # gates make the operator NONLINEAR IN ETA while staying exactly
        # linear in the residual -- the property iteration needs. Input layout
        # is unchanged ([f_u, f_p, log-eta-std, (z...)]); forward routes the
        # eta channel to the gates instead of the lift.
        self.eta_gates = eta_gates
        gdim = 5 if eta_gates else 4
        lift_in = in_channels - 1 if eta_gates else in_channels
        self.lift = nn.Linear(lift_in, h, bias=False)
        self.gate_in = nn.Sequential(nn.Linear(gdim, h), nn.GELU(), nn.Linear(h, h))
        self.gate_out = nn.Sequential(nn.Linear(gdim, h), nn.GELU(), nn.Linear(h, h))
        bs = h // n_blocks
        self.bs = bs
        tok = bs * (kmax + 1)
        # Continuum-indexed Green function: a table indexed by (l, k, k') is
        # tied to one truncation, but generating the same matrices from an MLP
        # over the NORMALISED indices (l/16, k/8, k'/8) defines the operator
        # for every degree and radial mode -- at a finer mesh the transforms
        # are simply built with a higher truncation and the generator is
        # queried further out. This is what makes ONE set of weights
        # meaningful on every level.
        self.green_mlp = green_mlp
        # eta_green: the Green generator is additionally conditioned on an
        # embedding of the sample's viscosity -- lateral mean+std of the
        # standardised log-eta per radial shell, resampled to 16 fixed radial
        # points (level-independent). Each sample then gets its OWN per-degree
        # spectral matrices: the operator family {K_eta}^-1 needs an
        # eta-dependent spectrum, which channel gates alone cannot express.
        # Conditioning is eta-only, so exact linearity in r is untouched.
        self.eta_green = eta_green
        # The Green generator's whole knowledge of the viscosity is this embedding.
        # A 32-number radial summary -> 16 dims is a very thin description of a field
        # spanning four decades; eta_embed_dim widens it and eta_quant adds Q global
        # log-eta quantiles (contrast structure the radial mean/std cannot express).
        self.eta_embed_dim = int(eta_embed_dim)
        self.eta_quant = int(eta_quant)
        if eta_green:
            n_lat = ((self.eta_lateral + 1) ** 2) * 16 if self.eta_lateral > 0 else 0
            n_in = 32 + (1 if level_cond else 0) + n_lat + self.eta_quant
            hid = max(64, 2 * self.eta_embed_dim)
            self.eta_embed = nn.Sequential(nn.Linear(n_in, hid), nn.GELU(),
                                           nn.Linear(hid, self.eta_embed_dim))
            with torch.no_grad():
                self.eta_embed[-1].weight.mul_(0.1)
                self.eta_embed[-1].bias.zero_()
        if green_mlp:
            self.ggen = nn.Sequential(nn.Linear(3 + (self.eta_embed_dim if eta_green else 0), 64),
                                      nn.GELU(),
                                      nn.Linear(64, 64), nn.GELU(),
                                      nn.Linear(64, n_blocks * bs * bs))
            with torch.no_grad():
                self.ggen[-1].weight.mul_(0.1)
                self.ggen[-1].bias.normal_(0.0, 0.02)
            self.green = None
        else:
            self.green = nn.Parameter(
                0.02 * torch.randn(lmax + 1, n_blocks, tok, tok))
        self.convs = nn.ModuleList(
            nn.Conv3d(h, h, kernel, padding=kernel // 2, bias=False)
            for _ in range(n_conv))
        if multi_dilation:
            self.convs_d2 = nn.ModuleList(nn.Conv3d(h, h, 3, padding=2, dilation=2, bias=False) for _ in range(n_conv))
            self.convs_d4 = nn.ModuleList(nn.Conv3d(h, h, 3, padding=4, dilation=4, bias=False) for _ in range(n_conv))
            with torch.no_grad():
                for m in list(self.convs_d2) + list(self.convs_d4):
                    m.weight.mul_(0.1)
        else:
            self.convs_d2 = self.convs_d4 = None
        # the K bank responses are a dense h x h 5^3 convolution each and dominate the
        # cost at fine levels (profiled: 87% of an L5 training step). bank_bottleneck
        # runs them on a C-channel projection instead: cost K*C^2*k^3 + 2*h*C rather
        # than K*h^2*k^3.
        self.bank_c = int(bank_bottleneck) if bank_bottleneck else n_hidden
        if self.eta_stencils > 0:
            K, C = self.eta_stencils, self.bank_c
            self.bank_in = nn.Conv3d(h, C, 1, bias=False) if C != h else None
            self.bank_out = nn.Conv3d(C, h, 1, bias=False) if C != h else None
            self.conv_banks = nn.ParameterList(
                nn.Parameter(0.1 * torch.randn(K, C, C, kernel, kernel, kernel) / (C * kernel ** 3) ** 0.5)
                for _ in range(n_conv))
            self.bank_mix = nn.ModuleList(
                nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, K)) for _ in range(n_conv))
        else:
            self.conv_banks = self.bank_in = self.bank_out = None
        # sep_stencils: depthwise (k^3, per channel) + pointwise (1^3) instead of the
        # dense h x h k^3 stencil -- h*k^3 + h^2 instead of h^2*k^3 multiply-adds.
        self.sep_stencils = sep_stencils
        # channels_last_3d: 3D convolutions on the PVC run through oneDNN, which
        # prefers the NDHWC layout; converting once around the whole local branch
        # avoids a reorder per layer.
        self.channels_last = channels_last
        # Diamond seams: a node on the boundary between two diamonds is stored once
        # per diamond, and a finite-element field must hold the SAME value in every
        # copy. Convolutions computed per diamond do not, and an inconsistent guess
        # is NOT in the space a Krylov solver searches -- it stalls the solve on the
        # inconsistent part (measured: 5-7% rms, which was the entire warm-start
        # plateau). Averaging the copies is the orthogonal projection onto the FE
        # space; it is linear, so the operator stays exactly linear in f.
        # At 65^3 per subdomain one stencil layer's activations are ~1.4 GB, so a deep
        # local branch cannot keep them all for the backward pass. Recomputing each layer
        # instead costs one extra forward and makes depth affordable at every level.
        self.grad_checkpoint = grad_checkpoint
        self.seam_average = seam_average
        self._set_seams(coords, dev=None)
        if sep_stencils:
            self.convs_pw = nn.ModuleList(nn.Conv3d(h, h, 1, bias=False) for _ in range(n_conv))
            with torch.no_grad():
                for m in self.convs_pw:
                    m.weight.mul_(0.1)
        with torch.no_grad():
            for m in self.convs:
                m.weight.mul_(0.1)   # keep the residual stream near-identity at init
        # Geometry gates per stencil layer: the floor residual concentrates in
        # the shell-adjacent layers, and depth-shared stencils cannot treat
        # those rows differently. Gates are functions of the fixed coords, so
        # linearity in the field is untouched.
        self.conv_gates = (nn.ModuleList(
            nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h))
            for _ in range(n_conv)) if depth_gates else None)
        # Full radial-column mixing: one linear layer coupling ALL radial nodes
        # per lateral position -- boundary layers in a single hop, which the
        # small stencils reach only after many compositions.
        self.rad_dense = (nn.Conv3d(h, h, (1, 1, shape[2]),
                                    padding=(0, 0, shape[2] // 2), bias=False)
                          if radial_dense else None)
        # Linear conv pyramid: coarse-level stencils + trilinear prolongation --
        # a learned linear V-cycle inside the stage, giving the local path a
        # large effective support at small cost.
        self.pyr = (nn.ModuleList(
            nn.Conv3d(h, h, 3, padding=1, bias=False) for _ in range(pyramid))
            if pyramid else None)
        with torch.no_grad():
            if self.rad_dense is not None:
                self.rad_dense.weight.mul_(0.1)
            if self.pyr is not None:
                for m in self.pyr:
                    m.weight.mul_(0.1)
        self.head = nn.Linear(h, out_channels, bias=False)

        self._set_basis(coords, lmax, kmax, dev=None)

    def _set_basis(self, coords, lmax, kmax, dev):
        from .spherical import build_transform

        self.lmax, self.kmax = lmax, kmax
        geom = Model._physical(coords, (0, 0, 0))
        t = build_transform(np.asarray(coords, dtype=np.float64), lmax, kmax)
        deg = torch.cat([torch.full((2 * l + 1,), l, dtype=torch.long)
                         for l in range(lmax + 1)])
        ordf = torch.cat([torch.arange(2 * l + 1, dtype=torch.float32) / max(1, 2 * l)
                          for l in range(lmax + 1)])
        self.register_buffer("ordf", ordf.to(dev) if dev is not None else ordf, persistent=False)
        for nm, v in zip(("geom", "sht_Y", "sht_A", "sht_Yr", "sht_Ar", "deg"),
                         (geom, *t, deg)):
            self.register_buffer(nm, v.to(dev) if dev is not None else v,
                                 persistent=False)

    def set_mesh(self, shape, coords, lmax=None, kmax=None):
        """Point the operator at another mesh (and optionally truncation).

        Only buffers move: the stencils are index-space (h-relative, like a
        smoother), the gates are functions of the coordinates, and with
        ``green_mlp`` the spectral weights are generated for whatever (l, k)
        range the new transforms carry.
        """
        if not self.green_mlp and ((lmax or self.lmax) != self.lmax
                                   or (kmax or self.kmax) != self.kmax):
            raise ValueError("changing the truncation needs green_mlp=True")
        dev = next(self.parameters()).device
        self.shape_in = tuple(shape)
        nx_ = int(shape[1]) if len(shape) == 4 else int(shape[0])
        self.stencil_dilation = max(1, (nx_ - 1) // 8) if self.dilated else 1
        self.level_scale = float(np.log2(max(1, (nx_ - 1) // 8)))
        if self.level_pyramid and self.pyr is not None:
            self.active_pyr = min(len(self.pyr), 1 + int(round(self.level_scale)))
        self._set_basis(coords, lmax or self.lmax, kmax or self.kmax, dev)
        self._set_seams(coords, dev)
        return self

    def _set_seams(self, coords, dev):
        """Storage-slot -> physical-node map of THIS mesh (rebuilt on every set_mesh)."""
        if not self.seam_average:
            return
        key = np.round(np.asarray(coords, dtype=np.float64).reshape(-1, 3), 9)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv_t = torch.as_tensor(inv, dtype=torch.long)
        cnt_t = torch.bincount(inv_t).to(torch.float32)
        if dev is not None:
            inv_t, cnt_t = inv_t.to(dev), cnt_t.to(dev)
        self.register_buffer("seam_idx", inv_t, persistent=False)
        self.register_buffer("seam_cnt", cnt_t, persistent=False)

    def _stencil(self, conv, vol, i=0, eta_feat=None):
        d = self.stencil_dilation
        k = conv.kernel_size[0]
        if self.sep_stencils:
            h_ = vol.shape[1]
            w = conv.weight[:, :1]                                        # depthwise slice
            out = self.convs_pw[i](F.conv3d(vol, w, None, padding=(k // 2) * d,
                                            dilation=d, groups=h_))
        elif d == 1:
            out = conv(vol)
        else:
            out = F.conv3d(vol, conv.weight, None, padding=(k // 2) * d, dilation=d)
        if self.convs_d2 is not None:
            out = out + F.conv3d(vol, self.convs_d2[i].weight, None, padding=2 * d, dilation=2 * d) \
                      + F.conv3d(vol, self.convs_d4[i].weight, None, padding=4 * d, dilation=4 * d)
        if self.conv_banks is not None:
            # K bank responses, mixed per node by the viscosity features
            bank = self.conv_banks[i].to(vol.dtype)                      # (K, C, C, k, k, k)
            vb = vol if self.bank_in is None else self.bank_in(vol)
            Kb, h_ = bank.shape[0], bank.shape[1]
            resp = F.conv3d(vb, bank.reshape(Kb * h_, h_, k, k, k), None,
                            padding=(k // 2) * d, dilation=d)              # (B, K*C, nx, ny, nr)
            B_ = vb.shape[0]
            resp = resp.reshape(B_, Kb, h_, *vb.shape[2:])
            mix = torch.softmax(self.bank_mix[i](eta_feat.to(vol.dtype)), -1)   # (B, nx, ny, nr, K)
            mix = mix.permute(0, 4, 1, 2, 3)[:, :, None]                  # (B, K, 1, nx, ny, nr)
            mixed = (resp * mix).sum(1)
            out = out + (mixed if self.bank_out is None else self.bank_out(mixed))
        if self.conv_scale_pow is not None and self.level_scale != 0.0:
            out = out * torch.exp(self.conv_scale_pow[i] * (self.level_scale * float(np.log(2.0)))).to(out.dtype)
        return out

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        squeeze = fx.ndim == 5
        if squeeze:
            fx = fx.unsqueeze(0)
        b, s_dom = fx.shape[0], fx.shape[1]
        nx, ny, nr = fx.shape[2:5]

        x = fx.reshape(b, s_dom, nx * ny * nr, -1)
        if self.eta_gates:
            # channel 4 is the standardised log-viscosity: route it to the
            # gates (multiplicative, per-sample) and keep the lift linear in
            # the remaining residual channels
            eta_ch = x[..., 4:5]
            x = torch.cat([x[..., :4], x[..., 5:]], -1)
            gfeat = torch.cat([self.geom.to(x.dtype).expand(b, -1, -1, -1),
                               eta_ch], -1)
            gi = self.gate_in(gfeat)
            go = self.gate_out(gfeat)
        else:
            gi = self.gate_in(self.geom.to(x.dtype))
            go = self.gate_out(self.geom.to(x.dtype))
        h = self.lift(x) * gi

        # spectral Green operator: SH x Chebyshev analysis, one dense
        # (block-channel x radial-mode) matrix per degree, synthesis back.
        # Runs in fp32 even under bf16 autocast: the dense per-degree
        # matrices are the numerically sensitive path of the operator.
        c = h.shape[-1]
        with torch.autocast(device_type=h.device.type, enabled=False):
            hf = h.float() if h.dtype in (torch.bfloat16, torch.float16) else h
            lat = hf.reshape(b, s_dom, nx, ny, nr, c).permute(0, 5, 1, 2, 3, 4)
            lat = lat.reshape(b, c, s_dom * nx * ny, nr)
            fm = torch.einsum("mn,bcnr->bcmr", self.sht_A.to(hf.dtype), lat)
            fk = torch.einsum("bcmr,rk->bcmk", fm, self.sht_Ar.to(hf.dtype).T)
            nb, bs = self.n_blocks, c // self.n_blocks
            m = fk.shape[2]
            t = fk.reshape(b, nb, bs, m, -1).permute(0, 1, 3, 2, 4).reshape(b, nb, m, -1)
            if self.green_mlp:
                k1 = fk.shape[-1]
                ls = torch.arange(self.lmax + 1, device=hf.device,
                                  dtype=hf.dtype) / 16.0
                ks = torch.arange(k1, device=hf.device, dtype=hf.dtype) / 8.0
                gi = torch.stack([
                    ls[:, None, None].expand(-1, k1, k1),
                    ks[None, :, None].expand(self.lmax + 1, -1, k1),
                    ks[None, None, :].expand(self.lmax + 1, k1, -1)], -1)
                if self.eta_green:
                    prof = eta_ch.reshape(b, s_dom, nx, ny, nr).float()
                    pf = torch.stack([prof.mean(dim=(1, 2, 3)),
                                      prof.std(dim=(1, 2, 3))], 1)  # (b,2,nr)
                    pf = F.interpolate(pf, size=16, mode="linear",
                                       align_corners=True)
                    pf = pf.reshape(b, 32)
                    if self.eta_lateral > 0:
                        le = eta_ch.reshape(b, 1, s_dom * nx * ny, nr).float()
                        fme = torch.einsum("mn,bcnr->bcmr", self.sht_A.float(), le)[:, 0]
                        sel = self.deg <= self.eta_lateral
                        fme = fme[:, sel, :]                                   # (b, (L+1)^2, nr)
                        fme = F.interpolate(fme, size=16, mode="linear", align_corners=True)
                        pf = torch.cat([pf, fme.reshape(b, -1)], 1)
                    if self.eta_quant > 0:
                        flat_le = eta_ch.reshape(b, -1).float()
                        qs = torch.linspace(0.0, 1.0, self.eta_quant, device=flat_le.device,
                                            dtype=flat_le.dtype)
                        pf = torch.cat([pf, torch.quantile(flat_le, qs, dim=1).T.to(pf.dtype)], 1)
                    if self.level_cond:
                        pf = torch.cat([pf, pf.new_full((b, 1), self.level_scale)], 1)
                    emb = self.eta_embed(pf.to(hf.dtype))
                    giB = torch.cat(
                        [gi[None].expand(b, -1, -1, -1, -1),
                         emb[:, None, None, None, :].expand(
                             b, self.lmax + 1, k1, k1, -1)], -1)
                    g = self.ggen(giB).reshape(b, self.lmax + 1, k1, k1,
                                               nb, bs, bs)
                    wl = g.permute(0, 1, 4, 5, 2, 6, 3).reshape(
                        b, self.lmax + 1, nb, bs * k1, bs * k1)
                    # per-degree loop instead of wl[:, deg]: avoids
                    # materialising the (b, M, nb, tok, tok) tensor
                    o = torch.empty_like(t)
                    for l in range(self.lmax + 1):
                        sel = self.deg == l
                        o[:, :, sel] = torch.einsum(
                            "bqmt,bqts->bqms", t[:, :, sel], wl[:, l])
                else:
                    g = self.ggen(gi).reshape(self.lmax + 1, k1, k1, nb, bs, bs)
                    # rows (i, k), cols (j, k') -- bs-major, matching t's layout
                    wl = g.permute(0, 3, 4, 1, 5, 2).reshape(
                        self.lmax + 1, nb, bs * k1, bs * k1)
                    o = torch.einsum("bqmt,mqts->bqms", t,
                                     wl[self.deg])         # (M, nb, tok, tok)
            else:
                o = torch.einsum("bqmt,mqts->bqms", t, self.green[self.deg])
            if self.mode_attn > 0:
                le_a = eta_ch.reshape(b, 1, s_dom * nx * ny, nr).float()
                fa = torch.einsum("mn,bcnr->bcmr", self.sht_A.float(), le_a)[:, 0]   # (b, m, nr)
                fa = F.interpolate(fa, size=16, mode="linear", align_corners=True)
                pos = torch.stack([self.deg.float() / max(1, self.lmax), self.ordf], -1)  # (m, 2)
                feat = torch.cat([fa, pos[None].expand(b, -1, -1)], -1)                  # (b, m, 18)
                q = self.attn_q(feat); kk = self.attn_k(feat)
                A = torch.softmax(q @ kk.transpose(1, 2) / float(self.mode_attn) ** 0.5, -1)  # (b, m, m)
                mixed = torch.einsum("bmn,bqnt->bqmt", A.to(o.dtype), o)
                o = o + self.attn_gate.to(o.dtype)[None, :, None, None] * mixed
            o = o.reshape(b, nb, m, bs, -1).permute(0, 1, 3, 2, 4).reshape(b, c, m, -1)
            o = torch.einsum("bcmk,rk->bcmr", o, self.sht_Yr.to(hf.dtype))
            o = torch.einsum("nm,bcmr->bcnr", self.sht_Y.to(hf.dtype), o)
            o = o.reshape(b, c, s_dom, nx, ny, nr).permute(0, 2, 3, 4, 5, 1)
            ospec = o.reshape(b, s_dom, -1, c)
            h = hf + (F.gelu(ospec) if self.nonlin else ospec)

        vol = h.reshape(b * s_dom, nx, ny, nr, c).permute(0, 4, 1, 2, 3)
        if self.channels_last:
            vol = vol.contiguous(memory_format=torch.channels_last_3d)
        if self.conv_banks is not None:
            le_ = eta_ch.reshape(b * s_dom, 1, nx, ny, nr).float()
            mu_ = F.avg_pool3d(le_, 5, stride=1, padding=2, count_include_pad=False)
            sd_ = (F.avg_pool3d(le_ ** 2, 5, stride=1, padding=2, count_include_pad=False) - mu_ ** 2).clamp_min(0).sqrt()
            eta_feat = torch.cat([le_, mu_, sd_], 1).permute(0, 2, 3, 4, 1)        # (B, nx, ny, nr, 3)
        else:
            eta_feat = None
        def _layer(vol_, ef, i, conv):
            if self.conv_gates is not None:
                g = self.conv_gates[i](self.geom.to(x.dtype))       # (1, S, N, C)
                g = g.reshape(s_dom, nx, ny, nr, c).permute(0, 4, 1, 2, 3)
                g = g.repeat(b, 1, 1, 1, 1)
                cv = self._stencil(conv, vol_ * g, i, ef)
            else:
                cv = self._stencil(conv, vol_, i, ef)
            return vol_ + (F.gelu(cv) if self.nonlin else cv)

        for i, conv in enumerate(self.convs):
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                vol = torch.utils.checkpoint.checkpoint(
                    _layer, vol, eta_feat, i, conv, use_reentrant=False)
            else:
                vol = _layer(vol, eta_feat, i, conv)
        if self.rad_dense is not None:
            vol = vol + self.rad_dense(vol)
        if self.pyr is not None:
            cur, sizes = vol, []
            for conv in list(self.pyr)[:self.active_pyr]:
                sizes.append(cur.shape[2:])
                cur = F.avg_pool3d(cur, 2, ceil_mode=True)
                cur = cur + conv(cur)
            for size in reversed(sizes):
                cur = F.interpolate(cur, size=size, mode="trilinear",
                                    align_corners=True)
            vol = vol + cur
        h = vol.permute(0, 2, 3, 4, 1).reshape(b, s_dom, -1, c)

        y = self.head(h * go).reshape(b, s_dom, nx, ny, nr, -1)
        if self.seam_average:
            # project onto the FE space: every copy of a shared node gets the mean
            b_, c_ = y.shape[0], y.shape[-1]
            flat = y.reshape(b_, -1, c_)
            idx = self.seam_idx[None, :, None].expand(b_, -1, c_)
            sums = torch.zeros(b_, int(self.seam_cnt.numel()), c_,
                               dtype=flat.dtype, device=flat.device)
            sums.scatter_add_(1, idx, flat)
            avg = sums / self.seam_cnt.to(flat.dtype)[None, :, None]
            y = torch.gather(avg, 1, idx).reshape(y.shape)
        return y.squeeze(0) if squeeze else y


class OctaveOperator(nn.Module):
    """Exactly-linear V-cycle operator: shared stencils per octave, spectral
    Green core permanently at a fixed reference mesh.

    Discretisation independence by construction, on multigrid's own argument:
    index-space stencils are the h-invariant objects PER OCTAVE (the reason
    classical smoothers transfer across levels), so ONE shared stencil set
    covers every octave between the deployment mesh and the reference; the
    per-degree Green core always sees the reference discretisation (fixed
    transforms, fixed truncation, fixed gain -- nothing about it moves with
    the level). ``set_mesh`` changes only the gate geometry and the recursion
    depth. Every path is linear and bias-free; geometry enters only as
    multiplicative gates.
    """

    mesh_buffer_names = ("geom",)

    def __init__(self, in_channels, out_channels, ref_shape, ref_coords,
                 n_hidden=128, n_blocks=8, lmax=16, kmax=8, n_conv=2,
                 kernel=5, coords=None, shape=None):
        super().__init__()
        if n_hidden % n_blocks:
            raise ValueError("hidden must divide into blocks")
        self.ref_n = ref_shape[0]
        self.lmax, self.kmax, self.n_blocks = lmax, kmax, n_blocks
        h = n_hidden
        self.lift = nn.Linear(in_channels, h, bias=False)
        self.gate_in = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h))
        self.gate_out = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h))
        self.oct = nn.ModuleList(
            nn.Conv3d(h, h, kernel, padding=kernel // 2, bias=False)
            for _ in range(n_conv))
        bs = h // n_blocks
        tok = bs * (kmax + 1)
        self.green = nn.Parameter(0.02 * torch.randn(lmax + 1, n_blocks, tok, tok))
        self.head = nn.Linear(h, out_channels, bias=False)
        with torch.no_grad():
            for m in self.oct:
                m.weight.mul_(0.1)

        from .spherical import build_transform

        t = build_transform(np.asarray(ref_coords, dtype=np.float64), lmax, kmax)
        for nm, v in zip(("sht_Y", "sht_A", "sht_Yr", "sht_Ar"), t):
            self.register_buffer(nm, v, persistent=False)
        deg = torch.cat([torch.full((2 * l + 1,), l, dtype=torch.long)
                         for l in range(lmax + 1)])
        self.register_buffer("deg", deg, persistent=False)
        self.set_mesh(shape if shape is not None else ref_shape,
                      coords if coords is not None else ref_coords)

    def set_mesh(self, shape, coords, lmax=None, kmax=None):
        dev = next(self.parameters()).device if any(
            True for _ in self.parameters()) else None
        self.shape_in = tuple(shape)
        self.depth = int(round(np.log2((shape[0] - 1) / (self.ref_n - 1))))
        geom = Model._physical(coords, (0, 0, 0))
        self.register_buffer("geom", geom.to(dev) if dev is not None else geom,
                             persistent=False)
        return self

    @staticmethod
    def _restrict(vol):
        w1 = torch.tensor([0.25, 0.5, 0.25], dtype=vol.dtype, device=vol.device)
        ker = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
               ).reshape(1, 1, 3, 3, 3).repeat(vol.shape[1], 1, 1, 1, 1)
        return F.conv3d(vol, ker, stride=2, padding=1, groups=vol.shape[1])

    def _core(self, vol, b, s_dom):
        c = vol.shape[1]
        n = self.ref_n
        lat = vol.reshape(b, s_dom, c, n, n, n).permute(0, 2, 1, 3, 4, 5)
        lat = lat.reshape(b, c, s_dom * n * n, n)
        Fm = torch.einsum("mn,bcnr->bcmr", self.sht_A.to(vol.dtype), lat)
        Fk = torch.einsum("bcmr,rk->bcmk", Fm, self.sht_Ar.to(vol.dtype).T)
        nb, bs = self.n_blocks, c // self.n_blocks
        m = Fk.shape[2]
        t = Fk.reshape(b, nb, bs, m, -1).permute(0, 1, 3, 2, 4).reshape(b, nb, m, -1)
        o = torch.einsum("bqmt,mqts->bqms", t, self.green[self.deg])
        o = o.reshape(b, nb, m, bs, -1).permute(0, 1, 3, 2, 4).reshape(b, c, m, -1)
        o = torch.einsum("bcmk,rk->bcmr", o, self.sht_Yr.to(vol.dtype))
        o = torch.einsum("nm,bcmr->bcnr", self.sht_Y.to(vol.dtype), o)
        o = o.reshape(b, c, s_dom, n, n, n).permute(0, 2, 1, 3, 4, 5)
        return vol + o.reshape(b * s_dom, c, n, n, n)

    def forward(self, fx):
        squeeze = fx.ndim == 5
        if squeeze:
            fx = fx.unsqueeze(0)
        b, s_dom = fx.shape[0], fx.shape[1]
        nx, ny, nr = fx.shape[2:5]
        x = fx.reshape(b, s_dom, nx * ny * nr, -1)
        gi = self.gate_in(self.geom.to(x.dtype))
        go = self.gate_out(self.geom.to(x.dtype))
        h = self.lift(x) * gi
        c = h.shape[-1]
        vol = h.reshape(b * s_dom, nx, ny, nr, c).permute(0, 4, 1, 2, 3)

        def oct_corr(v):
            o = v
            for conv in self.oct:
                o = o + conv(o)
            return o - v

        stack = []
        cur = vol
        for _ in range(self.depth):
            stack.append(oct_corr(cur))       # the octave CORRECTION at this scale
            cur = self._restrict(cur)
        # reference level: the same stencils handle its top octave, the Green
        # core the spectral band -- the composition every deeper level reuses
        cur = self._core(cur, b, s_dom) + oct_corr(cur)
        for corr in reversed(stack):
            cur = F.interpolate(cur, size=corr.shape[2:], mode="trilinear",
                                align_corners=True) + corr
        out = cur.permute(0, 2, 3, 4, 1).reshape(b, s_dom, -1, c)
        y = self.head(out * go).reshape(b, s_dom, nx, ny, nr, -1)
        return y.squeeze(0) if squeeze else y


# ------------------------------------------------------------------------ terra_infer glue

_MODELS: dict[tuple, Model] = {}


def load_state(model, state):
    """Loads weights, tolerating keys the model no longer has but never missing ones.

    Checkpoints written before the wavelet modules stopped being allocated for
    spherical-only models still carry those (never-trained) tensors. Extra keys are
    harmless; a MISSING key would silently leave part of the model at initialisation,
    so that stays an error.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} weights: {missing[:4]}")
    return unexpected


def _load_mesh_coords(shape, n_sd):
    """Reads node coordinates from $TERRA_MESH_COORDS, if it matches this field."""
    path = os.environ.get("TERRA_MESH_COORDS")
    if not path or not os.path.exists(path):
        return None
    c = np.fromfile(path, dtype=np.float64)
    want = n_sd * shape[0] * shape[1] * shape[2] * 3
    if c.size != want:
        print(f"terra_infer/wavelet3d: {path} has {c.size} values, this field wants {want}"
              " -- falling back to index coordinates", file=sys.stderr)
        return None
    return c.reshape(n_sd, *shape, 3)


def _build(name, shape, in_ch, out_ch, n_sd=10, coords=None):
    device = os.environ.get("TERRA_NEURAL_DEVICE", "cuda")
    net = Model(
        in_channels=in_ch,
        out_channels=out_ch,
        shape=shape,
        coords=coords if coords is not None else _load_mesh_coords(shape, n_sd),
        n_hidden=int(os.environ.get("TERRA_W3D_HIDDEN", 64)),
        n_layers=int(os.environ.get("TERRA_W3D_LAYERS", 4)),
        n_heads=int(os.environ.get("TERRA_W3D_HEADS", 8)),
    ).to(device)

    checkpoint = os.environ.get("TERRA_W3D_CHECKPOINT")
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        net.load_state_dict(state.get("model", state))
        provenance = f"weights from {checkpoint}"
    else:
        provenance = "RANDOMLY INITIALISED -- output is not physical"

    net.eval()
    print(
        f"terra_infer/wavelet3d: {name} {shape} {in_ch}->{out_ch} ch, "
        f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params on {device} ({provenance})",
        file=sys.stderr,
        flush=True,
    )
    return net


def apply(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Runs the model on the whole shell at once -- the solver already hands us all
    subdomains, and attention spans them."""
    device = os.environ.get("TERRA_NEURAL_DEVICE", "cuda")
    out = {}

    for name, array in fields.items():
        n_sd, nx, ny, nr, n_comp = array.shape
        x = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)

        key = (name, n_sd, nx, ny, nr, n_comp)
        if key not in _MODELS:
            _MODELS[key] = _build(name, (nx, ny, nr), n_comp, n_comp, n_sd)

        with torch.no_grad():
            y = _MODELS[key](x)
        out[name] = y.cpu().numpy().astype(np.float32)

    return out


def self_test(device="cpu", verbose=True):
    """Checks the properties the module claims: invertibility, shapes, determinism."""
    torch.manual_seed(0)
    results = []

    # 1. The Haar pair must be the identity, or "lossless downsampling" is a lie.
    dwt, idwt = DWT3D().to(device), IDWT3D().to(device)
    z = torch.randn(2, 8, 16, 12, 10, device=device)
    sub = dwt(z)
    err = (idwt(sub) - z).abs().max().item()
    results.append(("DWT3D -> IDWT3D is the identity", err < 1e-5, f"max err {err:.2e}"))
    results.append(
        ("8 subbands, half extent per axis", tuple(sub.shape) == (2, 64, 8, 6, 5), str(tuple(sub.shape)))
    )

    # 2. Element count preserved -> the transform discards nothing.
    results.append(("element count preserved", sub.numel() == z.numel(), f"{sub.numel()} vs {z.numel()}"))

    # 3. Odd extents (TERRA's are 2^L+1) survive the pad/crop round trip.
    for shape, cin, cout in (((9, 9, 9), 3, 3), ((5, 5, 5), 1, 1), ((17, 17, 9), 3, 3)):
        net = Model(cin, cout, shape, n_hidden=32, n_layers=2, n_heads=4).to(device).eval()
        x = torch.randn(2, 10, *shape, cin, device=device)      # batch 2, ten diamonds
        with torch.no_grad():
            y = net(x)
        ok = tuple(y.shape) == (2, 10, *shape, cout)
        results.append((f"shape {shape} {cin}->{cout}", ok, str(tuple(y.shape))))

    # A field with no batch axis, exactly as the solver sends it.
    net = Model(3, 3, (9, 9, 9), n_hidden=32, n_layers=2, n_heads=4).to(device).eval()
    with torch.no_grad():
        y = net(torch.randn(10, 9, 9, 9, 3, device=device))
    results.append(("solver shape (10,9,9,9,3) round-trips", tuple(y.shape) == (10, 9, 9, 9, 3),
                    str(tuple(y.shape))))

    # The point of the change: a perturbation in one subdomain must reach the others.
    net = Model(1, 1, (9, 9, 9), n_hidden=32, n_layers=2, n_heads=4).to(device).eval()
    a = torch.zeros(10, 9, 9, 9, 1, device=device)
    bpt = a.clone(); bpt[0, 4, 4, 4, 0] = 1.0
    with torch.no_grad():
        d = (net(bpt) - net(a)).abs()
    cross = float(d[1:].max())
    results.append((f"perturbation crosses subdomains (max {cross:.2e})", cross > 1e-8, ""))

    # 4. Deterministic in eval mode (no BatchNorm running-stat drift).
    z = torch.randn(2, 10, 9, 9, 9, 1, device=device)
    with torch.no_grad():
        results.append(("eval is deterministic", torch.equal(net(z), net(z)), ""))

    if verbose:
        for label, ok, detail in results:
            print(f"  {'ok  ' if ok else 'FAIL'} : {label}" + (f"   [{detail}]" if detail else ""))
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
