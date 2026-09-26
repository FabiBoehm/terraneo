"""Linear neural solution operator for variable-viscosity Stokes on the spherical shell.

``LinearOperator`` maps (f_u, f_p, log-eta) -> (u, p) on the ten-diamond mesh and is
*exactly linear in the forcing*: a bias-free lift, multiplicative gates driven by node
geometry and viscosity, a spectral Green core (spherical-harmonic x Chebyshev analysis,
one hypernetwork-generated dense matrix per harmonic degree, synthesis back), a local
branch of residual 5^3 stencils with viscosity-mixed kernel banks, a bias-free head, and
a final projection that averages the copies of every node shared between diamonds so the
output lies in the finite-element space. Viscosity enters only through gates, the Green
generator and the bank mixing, never as a linear input channel.

One weight set serves every refinement level: the transforms are rebuilt per mesh by
``set_mesh`` and the Green generator is queried at whatever truncation the mesh needs.
Training lives in ``train_linear_mr``; ``load_state`` tolerates checkpoints that still
carry tensors from since-removed modules.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spherical import build_transform, node_quadrature

# --------------------------------------------------------------------------- Haar 3-D


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



class ViscosityPatchAttention(nn.Module):
    """Attention between physics patches, in the spirit of Transolver's
    Physics-Attention, with the patches defined by the viscosity field.

    Every node is softly assigned to one of ``n_patch`` patches by an MLP of
    viscosity features (log eta, its mean and standard deviation over the 5^3
    window around the node, and geometry). Nodes in the same physical state --
    the interior of a stiff slab, the weak channel beside it -- land in the same
    patch wherever they are on the shell. The features are averaged into one
    token per patch, the tokens attend to each other, and the result is scattered
    back with the same weights.

    The difference from Transolver: there the slice weights are computed from the
    full feature vector, which carries the forcing. Here the assignment AND the
    attention are functions of eta alone, so the module is a linear map of the
    features for a fixed viscosity -- the operator stays exactly linear in f.

    Cost is O(P n_patch C) for a mesh of P nodes, against O(P^2) for attention
    between nodes: 44 GFLOP at level 6 with 32 patches, i.e. 0.1% of the local
    branch. It supplies the long-range, viscosity-aware coupling that the
    spherical-harmonic truncation cannot represent.
    """

    def __init__(self, ch, n_patch, feat_dim, d_attn=32, hidden=64, mode="mlp"):
        super().__init__()
        self.n_patch = int(n_patch)
        self.d_attn = int(d_attn)
        self.mode = mode
        # mlp: learned assignment. quantile: centres at the per-sample quantiles of
        # log eta, no parameters. depthclass: 8 depth bands x 4 eta classes (n_patch=32).
        self.to_w = (nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, self.n_patch)) if mode == "mlp" else None)
        self.q = nn.Linear(feat_dim, self.d_attn)
        self.k = nn.Linear(feat_dim, self.d_attn)
        # zero-init: the module is the identity at initialisation, so a warm
        # start from a checkpoint without it reproduces that model exactly.
        self.gate = nn.Parameter(torch.zeros(ch))

    def logits(self, ef):
        """Patch scores from viscosity features only. ef[..., 0] is standardised
        log eta, ef[..., 6] the normalised depth."""
        if self.mode == "mlp":
            return self.to_w(ef)
        le = ef[..., 0]                                            # (b, P)
        M = self.n_patch
        if self.mode == "quantile":
            qs = torch.linspace(0.5 / M, 1 - 0.5 / M, M, device=le.device, dtype=torch.float32)
            q = torch.quantile(le.float(), qs, dim=1).T.to(le.dtype)          # (b, M)
            tau = ((le.amax(1) - le.amin(1)) / M).clamp_min(1e-3)[:, None, None]
            return -((le[..., None] - q[:, None, :]) / tau) ** 2
        if self.mode == "depthclass":
            nd, nq = 8, M // 8
            depth = ef[..., 6]
            cd = torch.linspace(0.5 / nd, 1 - 0.5 / nd, nd, device=le.device, dtype=le.dtype)
            qs = torch.linspace(0.5 / nq, 1 - 0.5 / nq, nq, device=le.device, dtype=torch.float32)
            q = torch.quantile(le.float(), qs, dim=1).T.to(le.dtype)          # (b, nq)
            tq = ((le.amax(1) - le.amin(1)) / nq).clamp_min(1e-3)[:, None, None]
            ld = -((depth[..., None] - cd[None, None, :]) * nd) ** 2           # (b, P, nd)
            lq = -((le[..., None] - q[:, None, :]) / tq) ** 2                  # (b, P, nq)
            return (ld[..., :, None] + lq[..., None, :]).reshape(le.shape[0], le.shape[1], M)
        raise ValueError(self.mode)

    def forward(self, v, ef):
        """v: (b, P, C) features, linear in f.  ef: (b, P, F) viscosity features."""
        w = torch.softmax(self.logits(ef), dim=-1)                # (b, P, M)
        denom = w.sum(1).clamp_min(1e-6)                          # (b, M)
        tok = torch.einsum("bpm,bpc->bmc", w, v) / denom[..., None]
        desc = torch.einsum("bpm,bpf->bmf", w, ef) / denom[..., None]
        att = torch.softmax(self.q(desc) @ self.k(desc).transpose(1, 2)
                            / float(self.d_attn) ** 0.5, -1)      # (b, M, M)
        mixed = torch.einsum("bmn,bnc->bmc", att, tok)
        return v + self.gate.to(v.dtype) * torch.einsum("bpm,bmc->bpc", w, mixed)

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
                 seam_average=False, eta_embed_dim=16, eta_quant=0, grad_checkpoint=False,
                 phys_attn=0, phys_attn_dim=32, spectral=True, phys_attn_layers=1,
                 phys_window_physical=False, phys_grad_feat=False, phys_mode="mlp",
                 green_patch_cond=False, spectral_layers=1):
        super().__init__()
        self.nonlin = nonlin
        # spectral=False removes the SH x Chebyshev term: lift -> stencils -> head.
        self.spectral = bool(spectral)
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
        # viscosity-patch attention: patches defined by eta, attention between
        # their tokens. feat_dim = (log eta, 5^3 mean, 5^3 std) + geometry(4).
        self.phys_attn = int(phys_attn)
        self.phys_window_physical = bool(phys_window_physical)
        self.phys_grad_feat = bool(phys_grad_feat)
        self.phys_mode = phys_mode
        self.green_patch_cond = bool(green_patch_cond) and self.phys_attn > 0
        self.phys_feat_dim = 7 + (1 if self.phys_grad_feat else 0)
        # several layers in sequence: each has its own patches and its own attention,
        # so later layers can re-partition the shell after the earlier mixing.
        self.phys = (nn.ModuleList(ViscosityPatchAttention(n_hidden, self.phys_attn, self.phys_feat_dim,
                                                           d_attn=int(phys_attn_dim), mode=phys_mode)
                                   for _ in range(max(1, int(phys_attn_layers))))
                     if self.phys_attn > 0 else None)
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
            n_in += (self.phys_attn * self.phys_feat_dim) if self.green_patch_cond else 0
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
        # spectral_layers > 1: further eta-generated spectral cores in sequence, each
        # with its own generator, separated by a multiplicative gate of viscosity and
        # geometry (a function of eta only, so the operator stays linear in f). This is
        # the FNO's layer stack with the pointwise nonlinearity replaced by the gate.
        self.spectral_layers = max(1, int(spectral_layers))
        if green_mlp and self.spectral_layers > 1:
            self.ggen_extra = nn.ModuleList()
            for _ in range(self.spectral_layers - 1):
                gg = nn.Sequential(nn.Linear(3 + (self.eta_embed_dim if eta_green else 0), 64),
                                   nn.GELU(), nn.Linear(64, 64), nn.GELU(),
                                   nn.Linear(64, n_blocks * bs * bs))
                with torch.no_grad():
                    gg[-1].weight.mul_(0.1)
                    gg[-1].bias.normal_(0.0, 0.02)
                self.ggen_extra.append(gg)
            self.spec_gates = nn.ModuleList(
                nn.Sequential(nn.Linear(gdim, h), nn.GELU(), nn.Linear(h, h))
                for _ in range(self.spectral_layers - 1))
        else:
            self.ggen_extra = self.spec_gates = None
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
        geom = _physical(coords, (0, 0, 0))
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

    def _patch_feats(self, eta_ch, b, s_dom, nx, ny, nr, dtype):
        """(b, P, F) viscosity features for the patch attention: standardised log eta,
        its mean and spread over a 5^3 window (in nodes, or in a fixed physical size when
        phys_window_physical), geometry, and optionally |grad log eta|."""
        lev = eta_ch.reshape(b * s_dom, 1, nx, ny, nr).float()
        d = max(1, (int(nx) - 1) // 8) if self.phys_window_physical else 1
        k = torch.ones(1, 1, 5, 5, 5, device=lev.device, dtype=lev.dtype)
        cnt = F.conv3d(torch.ones_like(lev), k, padding=2 * d, dilation=d)
        mup = F.conv3d(lev, k, padding=2 * d, dilation=d) / cnt
        sdp = (F.conv3d(lev ** 2, k, padding=2 * d, dilation=d) / cnt - mup ** 2).clamp_min(0).sqrt()
        feats = [lev, mup, sdp]
        if self.phys_grad_feat:
            # central difference over +-d nodes, divided by the PHYSICAL distance 2 d h
            # (h = 1/(n-1) in units of the shell thickness) so the feature reads the same
            # at every level. The earlier form divided a 2-node difference by 2d, which
            # left a factor h and made the feature 4x smaller at L5 than at L3.
            hh = 1.0 / max(1, int(nx) - 1)
            gx = (lev[:, :, 2 * d:] - lev[:, :, :-2 * d]); gx = F.pad(gx, (0, 0, 0, 0, d, d))
            gy = (lev[:, :, :, 2 * d:] - lev[:, :, :, :-2 * d]); gy = F.pad(gy, (0, 0, d, d))
            gz = (lev[..., 2 * d:] - lev[..., :-2 * d]); gz = F.pad(gz, (d, d))
            feats.append((gx ** 2 + gy ** 2 + gz ** 2).sqrt() / (2.0 * d * hh))
        ef = torch.cat(feats, 1)
        nf = ef.shape[1]
        ef = ef.reshape(b, s_dom, nf, -1).permute(0, 1, 3, 2).reshape(b, -1, nf).to(dtype)
        gm = self.geom.to(dtype).expand(b, -1, -1, -1).reshape(b, -1, 4)
        # geometry goes in the middle so that index 6 is always the normalised depth
        return torch.cat([ef[..., :3], gm, ef[..., 3:]], -1)

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
        ef = None
        if self.phys is not None and self.eta_gates:
            ef = self._patch_feats(eta_ch, b, s_dom, nx, ny, nr, h.dtype)
            if self.green_patch_cond:
                w0 = torch.softmax(self.phys[0].logits(ef), dim=-1)          # (b, P, M)
                den0 = w0.sum(1).clamp_min(1e-6)
                patch_desc = (torch.einsum("bpm,bpf->bmf", w0, ef) / den0[..., None]).reshape(b, -1)

        # spectral Green operator: SH x Chebyshev analysis, one dense
        # (block-channel x radial-mode) matrix per degree, synthesis back.
        # Runs in fp32 even under bf16 autocast: the dense per-degree
        # matrices are the numerically sensitive path of the operator.
        c = h.shape[-1]
        if self.spectral:
            for si in range(self.spectral_layers):
                ggen = self.ggen if si == 0 else self.ggen_extra[si - 1]
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
                            if self.green_patch_cond:
                                pf = torch.cat([pf, patch_desc.float()], 1)
                            emb = self.eta_embed(pf.to(hf.dtype))
                            giB = torch.cat(
                                [gi[None].expand(b, -1, -1, -1, -1),
                                 emb[:, None, None, None, :].expand(
                                     b, self.lmax + 1, k1, k1, -1)], -1)
                            g = ggen(giB).reshape(b, self.lmax + 1, k1, k1,
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
                            g = ggen(gi).reshape(self.lmax + 1, k1, k1, nb, bs, bs)
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
                if si + 1 < self.spectral_layers:
                    gf = gfeat if self.eta_gates else self.geom.to(h.dtype).expand(b, -1, -1, -1)
                    h = h * self.spec_gates[si](gf.to(h.dtype))

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

        if self.phys is not None and self.eta_gates:
            hflat = h.reshape(b, -1, c)
            for layer in self.phys:
                hflat = layer(hflat, ef)
            h = hflat.reshape(b, s_dom, -1, c)

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


def load_state(model, state):
    """Loads weights, tolerating keys the model no longer has but never missing ones.

    Checkpoints written before the wavelet modules stopped being allocated for
    spherical-only models still carry those (never-trained) tensors. Extra keys are
    harmless; a MISSING key would silently leave part of the model at initialisation,
    so that stays an error.
    """
    # patch attention became a ModuleList: map 'phys.<p>' from older checkpoints
    # onto 'phys.0.<p>' when the model expects the list form.
    if any(k.startswith("phys.0.") for k in model.state_dict()) and \
            any(k.startswith("phys.") and not k.startswith("phys.0.") for k in state):
        state = {("phys.0." + k[len("phys."):] if k.startswith("phys.") and not k[5:6].isdigit()
                  else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} weights: {missing[:4]}")
    return unexpected
