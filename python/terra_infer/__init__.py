"""Python side of terra::ml::NeuralSolver (embedded CPython).

TERRA-NG imports this module inside its own process and calls :func:`call` once
per solve. Fields arrive as ``memoryview``s over the solver's own host buffers,
so ``np.frombuffer`` wraps them without copying; whatever you return is copied
straight back into those buffers.

A field is shaped ``(n_subdomains, nx, ny, nr, n_components)`` -- the Kokkos view
shape on the C++ side -- with the components interleaved last. The Stokes system
sends two: ``u`` (3 components, velocity level) and ``p`` (1 component, one
refinement level coarser). Subdomains are independent blocks: nodes shared
between them are duplicated, and the C++ side repairs the disagreement after
unpacking, so a model may treat each subdomain on its own.

To plug in a model, register a function taking and returning
``dict[str, np.ndarray]``::

    @register("my_model")
    def my_model(fields):
        u = fields["u"]              # (n_sd, nx, ny, nr, 3), float32, read-only
        return {"u": ..., "p": ...}  # same shapes, float32

Returned arrays must be float32 and the same shape as the input; the solver
checks and raises if not.
"""

from __future__ import annotations

import os

import numpy as np

MODELS: dict[str, callable] = {}


def register(name):
    """Decorator: makes a function reachable as the solver's ``--neural-solver <name>``."""

    def wrap(fn):
        MODELS[name] = fn
        return fn

    return wrap


def call(model, buffers, shapes):
    """Entry point invoked by terra::ml::NeuralSolver.

    :param model: registered model name.
    :param buffers: ``{name: memoryview}`` over the solver's host buffers.
    :param shapes: ``{name: (n_subdomains, nx, ny, nr, n_components)}``.
    :returns: ``{name: np.ndarray}``, float32, matching shapes.
    """
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; registered: {sorted(MODELS)}")

    fields = {
        name: np.frombuffer(buf, dtype=np.float32).reshape(shapes[name])
        for name, buf in buffers.items()
    }

    out = MODELS[model](fields)

    if not isinstance(out, dict):
        raise TypeError(f"model {model!r} returned {type(out).__name__}, expected dict")

    # The solver reads these through the buffer protocol, so they have to be
    # contiguous float32 before they go back.
    return {
        name: np.ascontiguousarray(array, dtype=np.float32)
        for name, array in out.items()
    }


# ---------------------------------------------------------------- built-in models


@register("zero")
def zero_model(fields):
    """Zeros. The neutral answer, and the one that proves the plumbing is honest."""
    return {name: np.zeros_like(array) for name, array in fields.items()}


@register("echo")
def echo_model(fields):
    """The right-hand side, unchanged. Useless as a solver, ideal as a round-trip test."""
    return {name: array.copy() for name, array in fields.items()}


@register("scale")
def scale_model(fields):
    """0.5 * rhs. Deterministic and non-trivial: proves the values coming back are
    the ones Python computed, not a copy that never left."""
    return {name: 0.5 * array for name, array in fields.items()}


@register("torch")
def torch_model(fields):
    """Runs the TorchScript module named by ``$TERRA_NEURAL_CHECKPOINT``.

    The module receives ``(n_subdomains, n_components, nx, ny, nr)`` -- channels
    first, transposed from the wire layout -- with subdomains as the batch
    dimension, which is what makes a per-subdomain model parallel for free.
    """
    import torch

    global _MODULE, _DEVICE
    if _MODULE is None:
        path = os.environ.get("TERRA_NEURAL_CHECKPOINT")
        if not path:
            raise RuntimeError("model 'torch' needs $TERRA_NEURAL_CHECKPOINT")
        _DEVICE = os.environ.get("TERRA_NEURAL_DEVICE", "cuda")
        _MODULE = torch.jit.load(path, map_location=_DEVICE).eval()

    device = _DEVICE
    out = {}
    for name, array in fields.items():
        x = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        with torch.no_grad():
            y = _MODULE(x)
        out[name] = y.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
    return out


_MODULE = None
_DEVICE = "cuda"


@register("cband")
def cband_preconditioner(fields):
    """The trained spectral operator as a Stokes preconditioner.

    Applies the checkpoint named by ``$TERRA_NEURAL_CHECKPOINT`` (architecture
    switches are read from the checkpoint itself) to the incoming right-hand
    side, reproducing the training-time input pipeline: per-call rms
    normalisation of f_u, the standardised log-viscosity channel, and the
    derived grad-log-eta / div-f_u channels. Mesh coordinates come from
    ``$TERRA_MESH_COORDS`` (the raw float64 ``coords_velocity.bin`` of this
    mesh, as written by ``stokes_dataset_tool --dump-coords``).

    Viscosity: the solver does not ship eta, so this assumes the CONSTANT
    viscosity the demo runs use (log eta = 0 everywhere). Variable-viscosity
    runs need eta packed into the solver's field list first.
    """
    import torch

    from . import stokes_residual
    from .operator import load_state

    # Fixed-reference core ($TERRA_CBAND_REF_STRIDE = 2^(level - ref_level)):
    # restrict the incoming residual to the trained reference mesh, run the
    # model in its home habitat (home gain, distribution, basis), prolongate
    # the correction at the end. The trained operator never sees another
    # discretisation -- every level-dependence is projected away. Must run
    # BEFORE the build block ($TERRA_MESH_COORDS then names the REFERENCE
    # coords). Load vectors are duals: restriction is summation-weighted
    # (approximately P^T), realised as full-weighting x stride^3.
    ref_stride = int(os.environ.get("TERRA_CBAND_REF_STRIDE", "1"))
    fields_full = None
    if ref_stride > 1:
        fields_full = {k: np.array(v, copy=True) for k, v in fields.items()}
        t = torch.from_numpy(np.ascontiguousarray(fields["u"])).double()
        s_, nx_, ny_, nr_, _ = t.shape
        v = t.permute(0, 4, 1, 2, 3).reshape(s_ * 3, 1, nx_, ny_, nr_)
        w1 = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
        ker = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
               ).reshape(1, 1, 3, 3, 3)
        num = torch.nn.functional.conv3d(v, ker, stride=ref_stride, padding=1)
        den = torch.nn.functional.conv3d(torch.ones_like(v), ker,
                                         stride=ref_stride, padding=1)
        r_u = ((num / den) * float(ref_stride ** 3)).squeeze(1)
        r_u = r_u.reshape(s_, 3, *r_u.shape[1:]).permute(0, 2, 3, 4, 1)
        p_t = torch.from_numpy(np.ascontiguousarray(fields["p"])).double()[..., 0]
        r_p = (p_t[:, ::ref_stride, ::ref_stride, ::ref_stride]
               * float(ref_stride ** 3))
        fields = {"u": np.ascontiguousarray(r_u.numpy(), dtype=np.float32),
                  "p": np.ascontiguousarray(r_p.numpy(),
                                            dtype=np.float32)[..., None]}

    st = _CBAND
    if st.get("net") is None:
        path = os.environ.get("TERRA_NEURAL_CHECKPOINT")
        if not path:
            raise RuntimeError("model 'cband' needs $TERRA_NEURAL_CHECKPOINT")
        dev = os.environ.get("TERRA_NEURAL_DEVICE", "xpu")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        shape = tuple(fields["u"].shape)      # (S, nx, ny, nr, 3)
        cpath = os.environ.get("TERRA_MESH_COORDS")
        if not cpath:
            raise RuntimeError("model 'cband' needs $TERRA_MESH_COORDS")
        coords = np.fromfile(cpath, dtype=np.float64)
        coords = coords.reshape(shape[0], shape[1], shape[2], shape[3], 3)
        def _build_linear(ck_):
            from .operator import LinearOperator

            # with a hypernetwork Green the deployment truncation is free --
            # $TERRA_CBAND_LMAX / $TERRA_CBAND_KMAX scale it to the mesh
            lm_ = int(os.environ.get("TERRA_CBAND_LMAX", "0")) \
                or (ck_.get("spherical", 16) or 16)
            km_ = int(os.environ.get("TERRA_CBAND_KMAX", "0")) \
                or (ck_.get("radial_modes", 8) or 8)
            n = LinearOperator(9 if ck_.get("defect_z") else 5, 4,
                               shape[1:4], coords,
                               n_hidden=ck_["hidden"], n_blocks=ck_["heads"],
                               lmax=lm_, kmax=km_,
                               eta_gates=ck_.get("linear_eta_gates", False),
                               eta_green=ck_.get("linear_eta_green", False),
                               nonlin=ck_.get("linear_nonlin", False),
                               green_mlp=ck_.get("linear_green_mlp", False),
                               n_conv=ck_.get("linear_convs", 2),
                               kernel=ck_.get("linear_kernel", 3),
                               depth_gates=ck_.get("linear_depth_gates", False),
                               radial_dense=ck_.get("linear_radial_dense", False),
                               pyramid=ck_.get("linear_pyramid", 0)
                               ).to(dev).eval()
            load_state(n, ck_["model"])
            return n

        if not ck.get("linear"):
            raise RuntimeError("model 'cband' only loads LinearOperator checkpoints "
                               "(the wavelet/octave models were retired)")
        net = _build_linear(ck)
        load_state(net, ck["model"])
        from .spherical import node_quadrature

        # Optional subspace split ($TERRA_CBAND_FILTER=1): feed the net only the
        # part of the residual its SH x Chebyshev basis can represent -- its
        # input then has training-like statistics -- and pass the rough
        # complement through scaled by $TERRA_CBAND_ROUGH_OMEGA (a Richardson
        # step on the subspace the net cannot see; 0 discards it).
        filt = None
        if os.environ.get("TERRA_CBAND_FILTER"):
            from .spherical import build_transform

            filt = build_transform(coords, ck.get("spherical", 16) or 16,
                                   ck.get("radial_modes", 8) or 8)

        # The solver hands over LOAD vectors (M f), the model was trained on the
        # pointwise strong-form f. The lumped velocity mass IS the physical
        # per-node volume (normalized=False) -- the mean-1 convention would
        # absorb the ~5e-5 volume constant into the conversion and under-scale
        # the returned solution by that same factor (measured: the champion's
        # "inert" initial guess and IR steps were exactly this).
        wv = torch.from_numpy(node_quadrature(coords, normalized=False)).float()[..., None]
        wp = torch.nn.functional.avg_pool3d(
            wv[..., 0].unsqueeze(1), kernel_size=1, stride=2).squeeze(1)[..., None]             if False else wv[:, ::2, ::2, ::2] * 8.0  # coarse cell is 8 fine cells
        # Two-stage composition: $TERRA_CBAND_STAGE2 names a second (linear)
        # checkpoint trained on the FIRST stage's leftover errors; calls
        # alternate between the stages -- a learned two-grid cycle. Flexible
        # FGMRES and IR both accept a per-call-varying preconditioner.
        net2 = None
        dz_flags = [False]
        p2 = os.environ.get("TERRA_CBAND_STAGES") or os.environ.get("TERRA_CBAND_STAGE2")
        if p2:
            net2 = []
            for pth in p2.split(","):
                ck2 = torch.load(pth, map_location="cpu", weights_only=False)
                if not (ck.get("linear") and ck2.get("linear")):
                    raise RuntimeError("stage chains need linear checkpoints")
                net2.append(_build_linear(ck2))
                dz_flags.append(bool(ck2.get("defect_z")))
            print(f"terra_infer/cband: {1 + len(net2)}-stage cycle ({p2})", flush=True)
        # Per-degree gain compensation ($TERRA_CBAND_DEGREE_SCALE = npz with
        # degree_scale): the measured per-degree response of M K on the floor
        # subspace is equalised by rescaling the representable part of the
        # velocity output degree by degree.
        dsc = None
        pdsc = os.environ.get("TERRA_CBAND_DEGREE_SCALE")
        if pdsc:
            z_ = np.load(pdsc)
            lm_ = int(z_["lmax"])
            from .spherical import build_transform as _bt

            Yd, Ad, Yrd, Ard = _bt(coords, lm_, 8)
            degd = torch.cat([torch.full((2 * l + 1,), l) for l in range(lm_ + 1)])
            sc = torch.from_numpy(z_["degree_scale"]).float()[degd.long()]
            dsc = (Yd, Ad, Yrd, Ard, sc)
            print(f"terra_infer/cband: degree-scale from {pdsc}", flush=True)
        st.update(net2=net2, calls=0, dsc=dsc, dz_flags=dz_flags, dz_cache=None)
        st.update(net=net, dev=dev, linear=bool(ck.get("linear")), filt=filt,
                  le_mean=float(ck["log_eta_mean"]), le_std=float(ck["log_eta_std"]),
                  iJ=stokes_residual.inverse_jacobian(coords), wv=wv, wp=wp,
                  curl=bool(ck.get("curl_output")))
        print(f"terra_infer/cband: {path} on {dev}, "
              f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)

    dev = st["dev"]
    # Bootstrap harvesting: record the raw vectors this preconditioner receives
    # (same v_*.npz format the probe model writes, so --harvest consumes them).
    # This is how the SECOND-round distribution -- Krylov vectors of the
    # NEURAL-preconditioned iteration -- is captured with the model in the loop.
    probe_dir = os.environ.get("TERRA_CBAND_PROBE_DIR")
    if probe_dir:
        os.makedirs(probe_dir, exist_ok=True)
        i = len([f for f in os.listdir(probe_dir) if f.startswith("v_")])
        np.savez(os.path.join(probe_dir, f"v_{i:04d}.npz"),
                 **{k: np.array(v, copy=True) for k, v in fields.items()})
    # The champion was trained on strong-form f, so incoming LOAD vectors get the
    # lumped mass divided out. The residual-pair models were trained on raw load
    # vectors -- $TERRA_CBAND_LOAD_SPACE=1 skips the division for those.
    load_space = bool(os.environ.get("TERRA_CBAND_LOAD_SPACE"))
    f_u = torch.from_numpy(np.ascontiguousarray(fields["u"]))
    if not load_space:
        f_u = f_u / st["wv"]
    S, nx, ny, nr, _ = f_u.shape

    # rhs of the continuity equation, prolongated from the (nested) pressure grid
    fp_c = torch.from_numpy(np.ascontiguousarray(fields["p"]))[..., 0]
    if not load_space:
        fp_c = fp_c / st["wp"][..., 0]
    fp_v = torch.nn.functional.interpolate(
        fp_c.unsqueeze(1), size=(nx, ny, nr), mode="trilinear",
        align_corners=True).squeeze(1).unsqueeze(-1)

    rms = float(torch.sqrt((f_u.double() ** 2).mean()))
    scale = 1.0 / rms if rms > 0 else 1.0
    fu_n = (f_u * scale).float()
    fp_n = (fp_v * scale).float()

    comp_u = None
    if st.get("filt") is not None:
        Yt, At, Yrt, Art = st["filt"]

        def lowpass(f):   # (S, nx, ny, nr, C) -> its SH x Chebyshev projection
            s_, a_, b_, r_, c_ = f.shape
            lat = f.permute(4, 0, 1, 2, 3).reshape(c_, s_ * a_ * b_, r_)
            fm = torch.einsum("mn,cnr->cmr", At, lat)
            fk = torch.einsum("cmr,rk->cmk", fm, Art.T)
            o = torch.einsum("cmk,rk->cmr", fk, Yrt)
            o = torch.einsum("nm,cmr->cnr", Yt, o)
            return o.reshape(c_, s_, a_, b_, r_).permute(1, 2, 3, 4, 0)

        fu_low = lowpass(fu_n)
        comp_u = fu_n - fu_low
        fu_n = fu_low
        fp_n = lowpass(fp_n)

    # log-viscosity channel: the real field when the solver ships "eta"
    # (variable-viscosity runs), else the constant-viscosity zero channel
    if "eta" in fields:
        eta_t = torch.from_numpy(np.ascontiguousarray(fields["eta"])).float()
        eta_t = eta_t.reshape(S, nx, ny, nr, 1)
        eta_t = eta_t / eta_t.min().clamp_min(1e-30)   # d8 convention: min = 1
        le = torch.log(eta_t.clamp_min(1e-30))
        st["c_u"] = float(torch.exp(le.mean()))
    else:
        le = torch.zeros_like(fp_n)
        st["c_u"] = 1.0
    x = torch.cat([fu_n, fp_n, (le - st["le_mean"]) / st["le_std"]], -1)
    if not st.get("linear"):
        g = torch.zeros(S, nx, ny, nr, 3)
        dv = stokes_residual.gradient(fu_n.unsqueeze(0), st["iJ"]).diagonal(
            dim1=-2, dim2=-1).sum(-1)[..., None].squeeze(0)
        x = torch.cat([x, g, (dv / (dv.std() + 1e-8)).float()], -1)

    cycle = [st["net"]] + (st.get("net2") or [])
    pat = os.environ.get("TERRA_CBAND_PATTERN")
    if pat and st.get("net2"):
        idxs = [int(t) for t in pat.split(",")]
        k_idx = idxs[st.get("calls", 0) % len(idxs)]
    else:
        k_idx = st.get("calls", 0) % len(cycle)
    model_k = cycle[k_idx]
    st["calls"] = st.get("calls", 0) + 1
    if st.get("dz_flags", [False])[k_idx]:
        # defect conditioning: the previous stage's correction, in the same
        # per-call scaling as the residual channels
        zc = st.get("dz_cache")
        zx = (torch.zeros(S, nx, ny, nr, 4) if zc is None
              else torch.cat([zc[0], zc[1][..., None]], -1) * scale)
        x = torch.cat([x, zx.float()], -1)
    with torch.no_grad():
        out = net_out = model_k(x.to(dev)).cpu()
        if st.get("curl"):
            # curl-output model: velocity channels are a vector potential A,
            # u = curl(A) (finite differences on the block grids, div-free)
            g = stokes_residual.gradient(out[..., :3].unsqueeze(0).float(),
                                         st["iJ"].float())[0]
            u_c = torch.stack([g[..., 2, 1] - g[..., 1, 2],
                               g[..., 0, 2] - g[..., 2, 0],
                               g[..., 1, 0] - g[..., 0, 1]], dim=-1)
            out = net_out = torch.cat([u_c, out[..., 3:]], -1)

    if comp_u is not None:
        omega = float(os.environ.get("TERRA_CBAND_ROUGH_OMEGA", "0"))
        if omega:
            out = torch.cat([out[..., :3] + omega * comp_u, out[..., 3:]], -1)

    if st.get("dsc") is not None:
        Yd, Ad, Yrd, Ard, sc = st["dsc"]
        ou = out[..., :3]
        s_, nx_, ny_, nr_, _ = ou.shape
        lat = ou.permute(4, 0, 1, 2, 3).reshape(3, s_ * nx_ * ny_, nr_)
        Fm = torch.einsum("mn,cnr->cmr", Ad, lat)
        Fk = torch.einsum("cmr,rk->cmk", Fm, Ard.T)
        dFk = Fk * (sc - 1.0)[None, :, None]
        d = torch.einsum("cmk,rk->cmr", dFk, Yrd)
        d = torch.einsum("nm,cmr->cnr", Yd, d)
        d = d.reshape(3, s_, nx_, ny_, nr_).permute(1, 2, 3, 4, 0)
        out = torch.cat([ou + d, out[..., 3:]], -1)

    # Level amplitude: the operator gain rms(Ke)/rms(e) scales with h, so a
    # model trained at gain-normalized amplitude needs the per-level factor
    # back at deployment ($TERRA_CBAND_OUT_SCALE = g_ref/g_level, measured).
    oscl = float(os.environ.get("TERRA_CBAND_OUT_SCALE", "1"))
    if oscl != 1.0:
        out = out * oscl
    c_u = st.get("c_u", 1.0)   # exp(mean(log eta)); 1 for constant viscosity
    u_t = out[..., :3] / (scale * c_u)
    u = u_t.numpy().astype(np.float32)
    psign = float(os.environ.get("TERRA_CBAND_P_SIGN", "1"))
    p_fine = psign * out[..., 3] / scale
    if any(st.get("dz_flags", [False])) and k_idx == 0:
        zu = u_t.clone()
        zu[:, :, :, 0] = 0.0     # match training (hard-bc-masked corrections)
        zu[:, :, :, -1] = 0.0
        st["dz_cache"] = (zu, p_fine.clone())
    if os.environ.get("TERRA_CBAND_P_RESTRICT"):
        # full-weighting restriction instead of injection: averages out the
        # fine-scale prediction noise the ::2 subsample aliases onto the
        # coarse pressure grid
        w1 = torch.tensor([0.25, 0.5, 0.25], dtype=p_fine.dtype)
        ker = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
               ).reshape(1, 1, 3, 3, 3)
        pf = p_fine.unsqueeze(1)
        num = torch.nn.functional.conv3d(pf, ker, stride=2, padding=1)
        den = torch.nn.functional.conv3d(torch.ones_like(pf), ker, stride=2, padding=1)
        p = (num / den).squeeze(1).unsqueeze(-1).numpy().astype(np.float32)
    else:
        p = p_fine[:, ::2, ::2, ::2].unsqueeze(-1).numpy().astype(np.float32)
    if os.environ.get("TERRA_CBAND_ZERO_P"):
        p[:] = 0

    if fields_full is not None:
        # prolongate the reference-level correction back to the fine mesh
        # (nested grids: trilinear interpolation is exact on shared nodes)
        fv = fields_full["u"].shape[1:4]
        fp = fields_full["p"].shape[1:4]
        ut = torch.from_numpy(u).permute(0, 4, 1, 2, 3)
        uf = torch.nn.functional.interpolate(ut, size=fv, mode="trilinear",
                                             align_corners=True)
        u = uf.permute(0, 2, 3, 4, 1).contiguous().numpy().astype(np.float32)
        pt = p_fine.unsqueeze(1)     # reference velocity grid = densest p field
        pfl = torch.nn.functional.interpolate(pt, size=fp, mode="trilinear",
                                              align_corners=True)
        p = pfl.squeeze(1).unsqueeze(-1).contiguous().numpy().astype(np.float32)
        if os.environ.get("TERRA_CBAND_ZERO_P"):
            p[:] = 0
    return {"u": u, "p": p}


_CBAND: dict = {}


@register("probe")
def probe_model(fields):
    """Identity preconditioner that records its inputs.

    Saves every incoming Krylov-direction vector to ``$TERRA_PROBE_DIR`` and
    returns the echo, so the FGMRES runs (identity-preconditioned) while its
    true residual-direction distribution is harvested -- the reference for
    fitting the error-field spectra of the residual-pair training data.
    """
    d = os.environ.get("TERRA_PROBE_DIR", "probe_out")
    os.makedirs(d, exist_ok=True)
    i = len([f for f in os.listdir(d) if f.startswith("v_")])
    np.savez(os.path.join(d, f"v_{i:04d}.npz"),
             **{k: np.array(v, copy=True) for k, v in fields.items()})
    return {name: np.array(v, copy=True) for name, v in fields.items()}
