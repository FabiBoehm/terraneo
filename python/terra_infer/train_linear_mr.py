"""Multiresolution trainer for the LinearOperator with the hypernetwork Green.

ONE model, batches from residual-pair datasets at several levels: epochs
alternate between the datasets, and before each the operator's mesh state
(coordinates, transforms at that level's truncation) is swapped -- the
learnable state (Green generator, stencils, gates) carries no mesh index, so
this is what makes the trained operator meaningful on every level at once.

Assumes eta = 1 residual-pair data (5 input channels with a zero viscosity
channel), mean-free pressure target, hard no-slip masking.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from . import stokes_residual
from .operator import LinearOperator
from .data import load_coords, load_split, mean_free, relative_l2


def rel_l2_per_sample(a, b, eps=1e-12):
    d = tuple(range(1, a.ndim))
    return torch.sqrt(((a - b) ** 2).sum(d) / (b ** 2).sum(d).clamp_min(eps))



class _MMapTensors:
    """A memory-mapped array that behaves like the training tensors at the one place
    they are used: ``t[idx].to(dev)``. Indexing pages in only those samples, so a level
    costs its batch rather than its whole split in RAM."""

    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a

    def __len__(self):
        return len(self.a)

    @property
    def shape(self):
        return self.a.shape

    def __getitem__(self, idx):
        # A one-element torch index satisfies __index__, so NumPy can treat it as a scalar
        # and silently drop the batch dimension -- which only shows up at batch size 1.
        # Normalise to a numpy array so the batch axis always survives.
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        idx = np.atleast_1d(np.asarray(idx)) if not isinstance(idx, slice) else idx
        return torch.from_numpy(np.ascontiguousarray(self.a[idx]))


def _assembled_cache(root, max_train, max_test, eta_power, build):
    """Cache the ASSEMBLED tensors (x, y, sc, eta, ...) as .npy next to the data.

    The raw-field cache still has to be concatenated and rescaled on every run --
    at L6 that is ~120 MB/sample read plus as much again allocated. Storing the
    finished arrays removes both, and memory-mapping them removes the RAM ceiling
    on how much fine-level data a run can use.
    """
    d = os.path.join(root, "_asm_%s_%g" % (max_train if max_train is not None else "all",
                                            eta_power))
    names = ("x_tr", "y_tr", "x_te", "y_te", "sc_tr", "sc_te", "eta_tr", "eta_te",
             "c_tr", "contrast_tr")
    ok = os.path.isdir(d) and all(os.path.exists(os.path.join(d, n + ".npy")) for n in names) \
        and os.path.exists(os.path.join(d, "stats.npy"))
    if not ok:
        out, stats = build()
        tmp = d + ".tmp%d" % os.getpid()
        os.makedirs(tmp, exist_ok=True)
        for n in names:
            v = out[n]
            np.save(os.path.join(tmp, n + ".npy"),
                    v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
        np.save(os.path.join(tmp, "stats.npy"), np.asarray(stats, dtype=np.float64))
        try:
            os.replace(tmp, d)
        except OSError:
            # Another rank or job built the same cache first. os.replace cannot
            # overwrite a non-empty directory, and theirs is complete (they also
            # built into a temp dir and renamed), so discard ours and use it.
            if os.path.isdir(d):
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                raise
        print("  assembled cache -> %s" % os.path.basename(d), flush=True)
    st = np.load(os.path.join(d, "stats.npy"))
    # Staging. Batches are drawn by fancy-indexing random sample ids, so a memory-mapped
    # array on the parallel filesystem costs a scattered page fault per sample: the GPU
    # then idles while the process blocks in memmove. Pulling the arrays into RAM when
    # they fit turns that into a local copy. TERRA_CACHE_RAM_GB caps what we will hold
    # (0 disables staging); the mmap path stays for splits too large to fit.
    budget = float(os.environ.get("TERRA_CACHE_RAM_GB", "96"))
    paths = {n: os.path.join(d, n + ".npy") for n in names}
    total = sum(os.path.getsize(v) for v in paths.values()) / 2**30
    stage = 0.0 < total <= budget
    if stage:
        try:
            import psutil
            free = psutil.virtual_memory().available / 2**30
            if free < total * 1.3:
                stage = False
                print(f"  cache {total:.1f} GiB but only {free:.1f} GiB free: staying memory-mapped",
                      flush=True)
        except Exception:
            pass
    arrs = {}
    for n in names:
        if stage:
            a = np.load(paths[n])                 # read once, sequentially, into RAM
        else:
            a = np.load(paths[n], mmap_mode="r")
        if n.endswith("_te") and max_test is not None and len(a) > max_test:
            a = a[:max_test]
        arrs[n] = _MMapTensors(a)
    if stage:
        print(f"  staged {total:.1f} GiB of {os.path.basename(d)} into RAM", flush=True)
    elif total > budget:
        print(f"  cache {total:.1f} GiB exceeds TERRA_CACHE_RAM_GB={budget:g}: memory-mapped",
              flush=True)
    return arrs, float(st[0]), float(st[1])


def build_level(root, lmax, kmax, max_train=None, max_test=None, eta_power=0.0):
    coords = load_coords(root)
    state = {}

    def tensors(sp):
        le_mean, le_std, var_eta = state["le_mean"], state["le_std"], state["var_eta"]
        if var_eta:
            ch3 = (sp["log_eta"] - le_mean) / le_std
            c = np.exp(sp["log_eta"].mean(axis=(1, 2, 3, 4, 5),
                                          keepdims=True))
        else:
            ch3 = np.zeros_like(sp["f_p_v"])
            c = np.ones((len(sp["f_u"]), 1, 1, 1, 1, 1), np.float32)
        extra["eta"] = torch.from_numpy(np.exp(sp["log_eta"][..., 0]).astype(np.float32))
        # target scaling: u * eta^a * mean(eta)^(1-a); a = 0 is the mean-eta
        # rescaling, a = 1 makes the target eta*u ~ f L^2 (locally contrast-free)
        cc = np.asarray(c, np.float32).reshape(-1, 1, 1, 1, 1, 1)
        sc = (np.exp(sp["log_eta"]) ** eta_power * cc ** (1.0 - eta_power)).astype(np.float32)
        extra["sc"] = torch.from_numpy(sc)
        extra["c"] = torch.from_numpy(c.reshape(-1).astype(np.float32))
        extra["contrast"] = torch.from_numpy(
            (sp["log_eta"].max(axis=(1, 2, 3, 4, 5)) - sp["log_eta"].min(axis=(1, 2, 3, 4, 5))).astype(np.float32))
        x = np.concatenate([sp["f_u"], sp["f_p_v"],
                            ch3.astype(np.float32)], axis=-1)
        p = sp["p"] - sp["p"].mean(axis=(1, 2, 3, 4, 5), keepdims=True)
        y = np.concatenate([(sp["u"] * sc).astype(np.float32), p], axis=-1)
        return torch.from_numpy(x), torch.from_numpy(y)

    extra = {}

    def _build_all():
        """Only runs when the assembled cache is missing; this is the slow path."""
        tr = load_split(root, "train", limit=max_train)
        te = load_split(root, "test", limit=None)      # cache every test sample
        sd = float(tr["log_eta"].std())
        state["var_eta"] = sd > 1e-6
        state["le_mean"] = float(tr["log_eta"].mean()) if state["var_eta"] else 0.0
        state["le_std"] = sd if state["var_eta"] else 1.0
        xt, yt = tensors(tr)
        out = dict(x_tr=xt, y_tr=yt, eta_tr=extra["eta"], c_tr=extra["c"],
                   contrast_tr=extra["contrast"], sc_tr=extra["sc"])
        del tr
        xe, ye = tensors(te)
        out.update(x_te=xe, y_te=ye, sc_te=extra["sc"], eta_te=extra["eta"])
        del te
        return out, (state["le_mean"], state["le_std"])

    _a, le_mean, le_std = _assembled_cache(root, max_train, max_test, eta_power, _build_all)
    x_tr, y_tr, x_te, y_te = _a["x_tr"], _a["y_tr"], _a["x_te"], _a["y_te"]
    sc_tr, sc_te = _a["sc_tr"], _a["sc_te"]
    eta_tr, eta_te = _a["eta_tr"], _a["eta_te"]
    c_tr, contrast_tr = _a["c_tr"][:], _a["contrast_tr"][:]
    return dict(root=root, coords=coords, lmax=lmax, kmax=kmax,
                shape=tuple(x_tr.shape[2:5]), le_mean=le_mean, le_std=le_std,
                x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te,
                eta_tr=eta_tr, c_tr=c_tr, contrast_tr=contrast_tr, sc_tr=sc_tr, sc_te=sc_te,
                eta_te=eta_te)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--data2", default=None)
    ap.add_argument("--lmax", type=int, default=16)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--lmax2", type=int, default=24)
    ap.add_argument("--kmax2", type=int, default=16)
    ap.add_argument("--data3", default=None, help="optional third level")
    ap.add_argument("--lmax3", type=int, default=32)
    ap.add_argument("--kmax3", type=int, default=16)
    ap.add_argument("--batch-size3", type=int, default=2)
    ap.add_argument("--max-train3", type=int, default=None,
                    help="sample cap for the third level (default: --max-train)")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batch-size2", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--linear-convs", type=int, default=4)
    ap.add_argument("--linear-kernel", type=int, default=5)
    ap.add_argument("--linear-depth-gates", action="store_true")
    ap.add_argument("--mean-p-weight", type=float, default=1.0)
    ap.add_argument("--batch-mix", action="store_true",
                    help="interleave the levels PER BATCH (shuffled) instead of "
                         "per epoch: removes the level seesaw the alternating "
                         "schedule produces. Transforms are cached per level, "
                         "so the swap is a buffer assignment, not a pinv.")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from this checkpoint (adversarial "
                         "fine-tuning rounds)")
    ap.add_argument("--nonlin", action="store_true", help="GELU after the spectral core / stencils")
    ap.add_argument("--pyramid", type=int, default=0, help="coarse-stencil pyramid depth (max)")
    ap.add_argument("--invvisc-weight", type=float, default=0.0,
                    help="add w * ||e||_{1/eta} / ||u||_{1/eta}: the NEAR-NULL directions of the "
                         "variable-viscosity Stokes operator are the low-viscosity regions, and "
                         "they are exactly what a residual-driven solver cannot clean up after a "
                         "warm start -- weighting the loss by 1/eta pushes the prediction error "
                         "out of them (the opposite tilt to the energy norm)")
    ap.add_argument("--energy-weight", type=float, default=0.0,
                    help="add w * <e, A e> / <u, A u> with A the viscosity-weighted viscous "
                         "operator (the norm in which the multigrid-preconditioned solver "
                         "converges): pushes the prediction error OUT of the near-null modes "
                         "that a residual-driven Krylov method cannot see (spectrally-safe "
                         "warm start, Jolicoeur-Martineau et al. style energy fine-tune)")
    ap.add_argument("--h1-weight", type=float, default=0.0,
                    help="add w * ||grad(u_pred - u)|| / ||grad u|| (index-space gradient via the "
                         "mesh inverse Jacobian): penalises fine-scale error the L2 term under-weights")
    ap.add_argument("--target-eta-power", type=float, default=0.0,
                    help="train on u * eta^a * mean(eta)^(1-a) (a=1: locally contrast-free target)")
    ap.add_argument("--seed", type=int, default=0,
                    help="offsets weight initialisation and data order (replicates)")
    ap.add_argument("--no-spectral", action="store_true",
                    help="remove the spectral term entirely (ablation)")
    ap.add_argument("--phys-attn", type=int, default=0,
                    help="patches for viscosity-patch attention (0 = off)")
    ap.add_argument("--phys-window-physical", action="store_true",
                    help="patch statistics over a fixed physical window (dilated)")
    ap.add_argument("--phys-grad-feat", action="store_true",
                    help="add |grad log eta| to the patch features")
    ap.add_argument("--phys-mode", default="mlp", choices=["mlp", "quantile", "depthclass"],
                    help="how nodes are assigned to patches")
    ap.add_argument("--green-patch-cond", action="store_true",
                    help="condition the kernel generator on the patch descriptors")
    ap.add_argument("--spectral-layers", type=int, default=1,
                    help="number of eta-generated spectral cores in sequence")
    ap.add_argument("--phys-attn-layers", type=int, default=1,
                    help="number of patch-attention layers in sequence")
    ap.add_argument("--phys-attn-dim", type=int, default=32,
                    help="query/key width of the patch attention")
    ap.add_argument("--mode-attn", type=int, default=0,
                    help="d > 0: eta-keyed cross-degree attention (dim d) in the spectral core")
    ap.add_argument("--bank-bottleneck", type=int, default=0,
                    help="run the eta-conditioned stencil banks on a C-channel projection "
                         "(the banks are 87%% of an L5 training step at full width)")
    ap.add_argument("--refine-steps", type=int, default=0,
                    help="unrolled defect correction: predict, apply the discrete Stokes "
                         "operator to that prediction, feed the residual back through the SAME "
                         "network and add the correction. Exactly linear in f, and it hands the "
                         "model the operator instead of making it learn the whole inverse at once")
    ap.add_argument("--refine-gate", type=float, default=0.25,
                    help="initial weight of the correction (small = function-preserving start)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute the local stencil layers in the backward pass; makes a deep "
                         "local branch fit at L6, where one layer's activations are ~1.4 GB")
    ap.add_argument("--eta-embed-dim", type=int, default=16,
                    help="width of the viscosity embedding the Green hypernetwork sees")
    ap.add_argument("--eta-quant", type=int, default=0,
                    help="append Q global log-eta quantiles to that embedding")
    ap.add_argument("--seam-average", action="store_true",
                    help="project the output onto the FE space by averaging the copies of each "
                         "shared diamond-seam node. A guess that disagrees with itself there is "
                         "outside the space a Krylov solver searches, which was the entire "
                         "warm-start plateau; it also costs 5-9%% of the raw error")
    ap.add_argument("--channels-last", action="store_true",
                    help="run the local branch in channels_last_3d (oneDNN's preferred layout)")
    ap.add_argument("--sep-stencils", action="store_true",
                    help="depthwise+pointwise local stencils instead of dense h x h k^3")
    ap.add_argument("--multi-dilation", action="store_true",
                    help="DCNO-style: extra 3^3 stencils at dilations 2 and 4 per layer")
    ap.add_argument("--eta-stencils", type=int, default=0,
                    help="K viscosity-mixed stencil banks per layer (dynamic, eta-conditioned kernels)")
    ap.add_argument("--eta-lateral", type=int, default=0,
                    help="feed SH coefficients (degree <= L) of log-eta per shell to the Green hypernetwork")
    ap.add_argument("--stencil-scale", action="store_true",
                    help="learned per-layer resolution scaling (h_L3/h)^q of the stencil outputs")
    ap.add_argument("--level-pyramid", action="store_true",
                    help="pyramid depth 1 + log2(refinement): coarsest level fixed at the L3 footprint")
    ap.add_argument("--level-weights", default=None,
                    help="comma-separated loss weights per level (default: equal per batch)")
    ap.add_argument("--dilated-stencils", action="store_true",
                    help="stencil dilation 2^(L-3): fixed PHYSICAL footprint across levels")
    ap.add_argument("--level-cond", action="store_true",
                    help="feed log2(h_L3/h) to the eta-Green embedding")
    ap.add_argument("--contrast-weight-power", type=float, default=0.0,
                    help="per-sample loss weight (1 + log10 contrast)^P, mean-normalised")
    ap.add_argument("--physics-weight", type=float, default=0.0,
                    help="strong-form Stokes residual (surrogate) of the prediction per level")
    ap.add_argument("--extra-data", action="append", default=[],
                    help="additional training set 'path:lmax:kmax:batch[:max_train]' (repeatable)")
    ap.add_argument("--eta-gates", action="store_true",
                    help="route log-eta into the FiLM gates (as train_operator)")
    ap.add_argument("--eta-green", action="store_true",
                    help="condition the Green hypernetwork on the sample's radial "
                         "log-eta profile (level-free)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--epoch-frac", default=None,
                    help="comma list (per level) of the FRACTION of that level's training set to "
                         "visit per epoch, e.g. 1,0.5,0.25: the fine levels dominate the epoch "
                         "cost, so sampling a fresh random subset of them each epoch keeps the "
                         "data coverage over the run while making epochs several times cheaper")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast on lift/convs/head; the spectral Green "
                         "core stays fp32 (guarded inside the operator)")
    ap.add_argument("--cache-device", action="store_true",
                    help="keep every level's tensors resident on the device")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    # ---- optional data parallelism ---------------------------------------------------
    # No XPU collective backend is installed (no oneCCL, no internet to fetch it), so the
    # gradients are averaged through gloo on the host. The models here are 0.2-8.4 M
    # parameters, i.e. <35 MiB per all-reduce against a 260 s epoch, so the host round
    # trip costs far less than the 8x it buys. Each rank is pinned to one tile by
    # ZE_AFFINITY_MASK in the launcher, so every rank sees its own device as xpu:0.
    WORLD = int(os.environ.get("WORLD_SIZE", "1"))
    RANK = int(os.environ.get("RANK", "0"))
    DIST = WORLD > 1
    if DIST:
        import torch.distributed as dist
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        dist.init_process_group("gloo", rank=RANK, world_size=WORLD)
        torch.manual_seed(args.seed)  # identical init on every rank
    def _is_main():
        return RANK == 0
    def _sync_grads(mod):
        """Average gradients across ranks. Flattened into one buffer so the collective
        is a single message rather than one per parameter."""
        gs = [q.grad for q in mod.parameters() if q.grad is not None]
        if not gs:
            return
        flat = torch.cat([g.reshape(-1) for g in gs]).to("cpu")
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= WORLD
        flat = flat.to(gs[0].device)
        o = 0
        for g in gs:
            n = g.numel()
            g.copy_(flat[o:o + n].view_as(g)); o += n
    print("loading level 1:", args.data)
    levels = [build_level(args.data, args.lmax, args.kmax, args.max_train, args.max_test, args.target_eta_power)]
    if args.data2:
        print("loading level 2:", args.data2)
        levels.append(build_level(args.data2, args.lmax2, args.kmax2,
                                  args.max_train, args.max_test, args.target_eta_power))
    if args.data3:
        print("loading level 3:", args.data3)
        levels.append(build_level(args.data3, args.lmax3, args.kmax3,
                                  args.max_train3 or args.max_train, args.max_test, args.target_eta_power))
    bsz = [args.batch_size, args.batch_size2, args.batch_size3]
    for spec in args.extra_data:
        parts = spec.split(":")
        print("loading extra set:", parts[0])
        levels.append(build_level(parts[0], int(parts[1]), int(parts[2]),
                                  int(parts[4]) if len(parts) > 4 else args.max_train, args.max_test, args.target_eta_power))
        bsz.append(int(parts[3]))
    efrac = [float(v) for v in args.epoch_frac.split(",")] if args.epoch_frac else None
    if efrac:
        print("  per-epoch data fraction per level: "
              + ", ".join(f"{efrac[min(j, len(efrac)-1)]:.2f}" for j in range(len(levels))))
    lvw = ([float(v) for v in args.level_weights.split(",")] if args.level_weights
           else [1.0] * len(levels))
    for j, lv in enumerate(levels):
        lv["level_w"] = lvw[j] if j < len(lvw) else 1.0
        w = (1.0 + lv["contrast_tr"] / np.log(10.0)) ** args.contrast_weight_power
        lv["w_tr"] = (w / w.mean()).float()
        if (args.physics_weight > 0 or args.h1_weight > 0 or args.energy_weight > 0
                or args.invvisc_weight > 0 or args.refine_steps > 0):
            lv["inv_J"] = stokes_residual.inverse_jacobian(lv["coords"]).to(dev)
            lv["mask"] = stokes_residual.interior_mask(lv["shape"], passes=2, device=dev)
            lv["mask_c"] = stokes_residual.interior_mask(lv["shape"], passes=1, device=dev)
    if args.cache_device:
        for lv in levels:
            for k in ("x_tr", "y_tr", "x_te", "y_te"):
                lv[k] = lv[k].to(dev)
        gb = sum(lv[k].numel() * 4 for lv in levels
                 for k in ("x_tr", "y_tr", "x_te", "y_te")) / 1e9
        print(f"  level tensors cached on {args.device} ({gb:.1f} GB)")

    lv0 = levels[0]
    net = LinearOperator(5, 4, lv0["shape"], lv0["coords"],
                     n_hidden=args.hidden, n_blocks=args.heads,
                     lmax=lv0["lmax"], kmax=lv0["kmax"],
                     phys_attn=args.phys_attn, phys_attn_dim=args.phys_attn_dim,
                     phys_attn_layers=args.phys_attn_layers,
                     spectral_layers=args.spectral_layers,
                     phys_window_physical=args.phys_window_physical, phys_grad_feat=args.phys_grad_feat,
                     phys_mode=args.phys_mode, green_patch_cond=args.green_patch_cond,
                     spectral=not args.no_spectral,
                     n_conv=args.linear_convs, kernel=args.linear_kernel,
                     depth_gates=args.linear_depth_gates,
                     eta_gates=args.eta_gates, eta_green=args.eta_green,
                     nonlin=args.nonlin, dilated=args.dilated_stencils,
                     level_cond=args.level_cond, pyramid=args.pyramid,
                     level_pyramid=args.level_pyramid,
                     stencil_scale=args.stencil_scale, eta_lateral=args.eta_lateral,
                     eta_stencils=args.eta_stencils, multi_dilation=args.multi_dilation,
                     mode_attn=args.mode_attn, bank_bottleneck=args.bank_bottleneck,
                     sep_stencils=args.sep_stencils, channels_last=args.channels_last,
                     seam_average=args.seam_average, eta_embed_dim=args.eta_embed_dim,
                     eta_quant=args.eta_quant, grad_checkpoint=args.grad_checkpoint,
                     green_mlp=True).to(dev)
    print(f"  model {sum(p.numel() for p in net.parameters())/1e6:.2f}M params on {dev}")
    if args.init_from:
        ck0 = torch.load(os.path.expanduser(args.init_from), map_location=dev,
                         weights_only=False)
        sd0 = ck0["model"]; own = net.state_dict(); widened = 0
        for k, v in list(sd0.items()):
            if k in own and own[k].shape != v.shape and own[k].ndim == v.ndim:
                # width expansion (net2net-style) or shrink: the overlapping
                # slice is copied, new channels keep their (small) random init
                t = own[k].clone()
                sl = tuple(slice(0, min(a, b)) for a, b in zip(own[k].shape, v.shape))
                t[sl] = v[sl].to(t.dtype)
                sd0[k] = t; widened += 1
        missing, unexpected = net.load_state_dict(sd0, strict=False)
        print(f"  warm-started from {args.init_from}"
              + (f" ({len(missing)} new parameter tensors left at init)" if missing else "")
              + (f" ({widened} tensors width-expanded)" if widened else ""))
        if unexpected:
            print(f"  --init-from: {len(unexpected)} checkpoint tensors have no counterpart here "
                  f"and were dropped (e.g. {unexpected[:2]})")

    # Per-level buffer cache: swapping meshes must not re-run the pinv.
    keys = net.mesh_buffer_names
    for lv in levels:
        net.set_mesh(lv["shape"],
                     lv["coords"], lv["lmax"], lv["kmax"])
        lv["bufs"] = {k: getattr(net, k).clone() for k in keys if hasattr(net, k)}

    def use(lv):
        net.shape_in = tuple(lv["shape"])
        net.lmax, net.kmax = lv["lmax"], lv["kmax"]
        for k, v in lv["bufs"].items():
            net.register_buffer(k, v, persistent=False)

    use(levels[0])
    refine_gate = (torch.nn.Parameter(torch.tensor(float(args.refine_gate), device=dev))
                   if args.refine_steps > 0 else None)

    _params = list(net.parameters()) + ([refine_gate] if refine_gate is not None else [])
    opt = torch.optim.AdamW(_params, lr=args.lr, weight_decay=1e-4)
    if args.batch_mix:
        per_ep = sum((int(round((efrac[min(j, len(efrac) - 1)] if efrac else 1.0) * len(lv["x_tr"])))
                      + bsz[j] - 1) // bsz[j]
                     for j, lv in enumerate(levels))
        steps = per_ep * args.epochs
    else:
        steps = sum((len(levels[e % len(levels)]["x_tr"])
                     + bsz[e % len(levels)] - 1) // bsz[e % len(levels)]
                    for e in range(args.epochs))
    # Under data parallelism each rank walks only its own shard (batches[RANK::WORLD]),
    # so it calls sched.step() WORLD times less often than the full-dataset count above.
    # Without this the run traverses just 1/WORLD of the one-cycle curve and never
    # reaches the decay phase -- the loss climbs for the whole job.
    if WORLD > 1:
        steps = max(1, (steps + WORLD - 1) // WORLD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps)

    def fwd(xb):
        if args.amp:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                out = net(xb)
            return out.float()
        return net(xb)

    def masked(out, nr):
        m = torch.ones(1, 1, 1, 1, nr, 1, device=out.device, dtype=out.dtype)
        m[..., 0, :] = 0.0
        m[..., -1, :] = 0.0
        return torch.cat([out[..., :3] * m, out[..., 3:]], -1)

    def refined(xb, lv, eb, scb):
        """Defect correction through the same network. u,p carry the target scaling, so the
        residual is formed in physical units and the correction is scaled back."""
        out = masked(fwd(xb), xb.shape[4])
        m = lv["mask"][None, ..., None]
        for _ in range(args.refine_steps):
            u_ph = out[..., :3] / scb
            p_ph = (out[..., 3:4] / scb)[..., 0]
            r_u = xb[..., :3] - (stokes_residual.viscous_operator(u_ph, eb, lv["inv_J"])
                                 + stokes_residual.gradient(p_ph[..., None], lv["inv_J"])[..., 0, :])
            g = stokes_residual.gradient(u_ph, lv["inv_J"])
            r_p = xb[..., 3] - (g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2])
            xr = torch.cat([r_u * m, r_p[..., None] * m, xb[..., 4:]], -1)
            # N maps forcing -> scaled velocity, and r is already in forcing units, so the
            # correction is in the same units as `out`; no extra scaling belongs here.
            out = out + refine_gate * masked(fwd(xr), xb.shape[4])
        return out

    def evaluate():
        net.eval()
        res = []
        with torch.no_grad():
            for lvi, lv in enumerate(levels):
                use(lv)
                eu = ep = 0.0
                n = len(lv["x_te"])
                eb_ = min(8, bsz[lvi])               # L5/L6 test batches at the training batch size (memory)
                for i in range(0, n, eb_):
                    xb = lv["x_te"][i:i + eb_].to(dev)
                    yb = lv["y_te"][i:i + eb_].to(dev)
                    scb = lv["sc_te"][i:i + eb_].to(dev)
                    pb = (refined(xb, lv, lv["eta_te"][i:i + eb_].to(dev), scb)
                          if args.refine_steps > 0 else masked(fwd(xb), xb.shape[4]))
                    eu += relative_l2(pb[..., :3] / scb, yb[..., :3] / scb).item() * len(xb)
                    ep += relative_l2(mean_free(pb[..., 3:]),
                                      yb[..., 3:]).item() * len(xb)
                res += [eu / max(n, 1), ep / max(n, 1)]   # a set without a test split scores 0
        net.train()
        return res

    hdr = " ".join(f"{'teu' + str(j + 1):>8} {'tep' + str(j + 1):>8}"
                   for j in range(len(levels)))
    if _is_main():
        print(f"\n{'epoch':>6} {'tr':>10} {hdr} {'lr':>9} {'s':>6}")
    best = float("inf")
    start_ep = 0
    resume_path = os.path.expanduser(args.out) + ".resume"
    if os.path.exists(resume_path):
        rs = torch.load(resume_path, map_location=dev, weights_only=False)
        net.load_state_dict(rs["model"])
        opt.load_state_dict(rs["opt"])
        sched.load_state_dict(rs["sched"])
        best, start_ep = rs["best"], rs["epoch"] + 1
        print(f"  resuming from {resume_path} at epoch {start_ep} (best {best:.4f})")
    for epoch in range(start_ep, args.epochs):
        t0 = time.time()
        gen = torch.Generator().manual_seed(1234 + epoch + 100000 * args.seed)
        if args.batch_mix:
            batches = []
            for j, lv_ in enumerate(levels):
                perm_ = torch.randperm(len(lv_["x_tr"]), generator=gen)
                if efrac:
                    perm_ = perm_[: max(bsz[j], int(round(efrac[min(j, len(efrac) - 1)] * len(perm_))))]
                batches += [(j, perm_[i:i + bsz[j]])
                            for i in range(0, len(perm_), bsz[j])]
            order = torch.randperm(len(batches), generator=gen)
            batches = [batches[k] for k in order]
            if DIST:
                # every rank must run the same number of steps or the all-reduce hangs
                n_keep = (len(batches) // WORLD) * WORLD
                batches = batches[RANK:n_keep:WORLD]
        else:
            lv = levels[epoch % len(levels)]
            bs = bsz[epoch % len(levels)]
            perm = torch.randperm(len(lv["x_tr"]), generator=gen)
            batches = [(epoch % len(levels), perm[i:i + bs])
                       for i in range(0, len(perm), bs)]
        running, seen = 0.0, 0
        for j, idx in batches:
            lv = levels[j]
            use(lv)
            xb = lv["x_tr"][idx].to(dev, non_blocking=True)
            yb = lv["y_tr"][idx].to(dev, non_blocking=True)
            if args.refine_steps > 0:
                pb = refined(xb, lv, lv["eta_tr"][idx].to(dev),
                             lv["sc_tr"][idx].to(dev))
            else:
                pb = masked(fwd(xb), xb.shape[4])
            pp = mean_free(pb[..., 3:])
            if args.contrast_weight_power > 0:
                wb = lv["w_tr"][idx].to(dev)
                loss = (wb * (rel_l2_per_sample(pb[..., :3], yb[..., :3])
                              + rel_l2_per_sample(pp, yb[..., 3:]))).mean()
            else:
                loss = relative_l2(pb[..., :3], yb[..., :3]) + relative_l2(pp, yb[..., 3:])
            if args.physics_weight > 0:
                if args.target_eta_power != 0.0:
                    raise SystemExit("--physics-weight is not defined with --target-eta-power")
                cu = lv["c_tr"][idx].to(dev).view(-1, 1, 1, 1, 1)
                eb = lv["eta_tr"][idx].to(dev)
                phys = (stokes_residual.momentum_residual(
                            pb[..., :3], pb[..., 3] * cu, xb[..., :3] * cu, eb, lv["inv_J"], lv["mask"])
                        + stokes_residual.continuity_residual(
                            pb[..., :3], xb[..., 3:4] * cu, lv["inv_J"], lv["mask_c"], subsample=False))
                loss = loss + args.physics_weight * phys
            # every derivative-based term below must be restricted to the nodes where the
            # finite-difference stencil is the real operator: at the index-space faces of a
            # diamond `gradient` falls back to one-sided rows, which are not the discrete
            # operator and, for the energy form, are not even symmetric. The physics residual
            # has always masked; these terms did not, so they summed garbage over every
            # diamond face -- including all the seams.
            msk = lv["mask"][None, ..., None]                     # (1, S, nx, ny, nr, 1)
            if args.invvisc_weight > 0:
                eb_ = lv["eta_tr"][idx].to(dev)[..., None]
                scb_ = lv["sc_tr"][idx].to(dev)
                e_u = (pb[..., :3] - yb[..., :3]) / scb_ * msk
                u_t = yb[..., :3] / scb_ * msk
                d = tuple(range(1, e_u.ndim))
                num = ((e_u ** 2) / eb_).sum(d)
                den = ((u_t ** 2) / eb_).sum(d).clamp_min(1e-30)
                loss = loss + args.invvisc_weight * (num / den).sqrt().mean()
            if args.energy_weight > 0:
                eb_ = lv["eta_tr"][idx].to(dev)
                scb_ = lv["sc_tr"][idx].to(dev)
                e_u = (pb[..., :3] - yb[..., :3]) / scb_          # error in physical units
                u_t = yb[..., :3] / scb_
                d = tuple(range(1, e_u.ndim))
                num = (e_u * stokes_residual.viscous_operator(e_u, eb_, lv["inv_J"]) * msk).sum(d)
                den = (u_t * stokes_residual.viscous_operator(u_t, eb_, lv["inv_J"]) * msk).sum(d)
                loss = loss + args.energy_weight * (num.clamp_min(0) / den.clamp_min(1e-30)).sqrt().mean()
            if args.h1_weight > 0:
                gp = stokes_residual.gradient(pb[..., :3], lv["inv_J"]) * msk[..., None]
                gt = stokes_residual.gradient(yb[..., :3], lv["inv_J"]) * msk[..., None]
                d = tuple(range(1, gp.ndim))
                h1 = torch.sqrt(((gp - gt) ** 2).sum(d) / (gt ** 2).sum(d).clamp_min(1e-12)).mean()
                loss = loss + args.h1_weight * h1
            if args.mean_p_weight > 0:
                dims = tuple(range(1, pp.ndim))
                mref = torch.sqrt((yb[..., 3:] ** 2).mean(dim=dims)) + 1e-12
                loss = loss + args.mean_p_weight * (
                    pb[..., 3:].mean(dim=dims).abs() / mref).mean()
            opt.zero_grad(set_to_none=True)
            loss = loss * lv["level_w"]
            loss.backward()
            if DIST:
                _sync_grads(net)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(idx)
            seen += len(idx)

        vals = evaluate()
        # mean velocity error over the levels that HAVE a test split: an extra training
        # set without one scores 0 (see evaluate), and including it would deflate the
        # score and make runs with different level counts incomparable.
        _scored = [j for j in range(len(levels)) if len(levels[j]["x_te"]) > 0]
        score = sum(vals[2 * j] for j in _scored) / max(len(_scored), 1)
        if not _is_main():
            continue
        print(f"{epoch:>6} {running/seen:>10.4f} "
              + " ".join(f"{v:>8.4f}" for v in vals)
              + f" {sched.get_last_lr()[0]:>9.2e} {time.time()-t0:>6.1f}",
              flush=True)
        if score < best and _is_main():
            best = score
            torch.save({"model": net.state_dict(),
                        "log_eta_mean": levels[0]["le_mean"],
                        "log_eta_std": levels[0]["le_std"],
                        "hidden": args.hidden, "heads": args.heads,
                        "spherical": args.lmax, "radial_modes": args.kmax,
                        "linear": True, "linear_convs": args.linear_convs,
                        "linear_kernel": args.linear_kernel,
                        "linear_depth_gates": args.linear_depth_gates,
                        "linear_radial_dense": False,
                        "linear_green_mlp": True,
                        "linear_eta_gates": args.eta_gates,
                        "linear_eta_green": args.eta_green,
                        "linear_nonlin": args.nonlin,
                        "linear_pyramid": args.pyramid,
                        "linear_stencil_scale": args.stencil_scale,
                        "linear_eta_lateral": args.eta_lateral,
                        "linear_eta_stencils": args.eta_stencils,
                        "linear_multi_dilation": args.multi_dilation,
                        "linear_mode_attn": args.mode_attn,
                        "linear_spectral": not args.no_spectral,
                        "seed": args.seed,
                        "linear_phys_attn": args.phys_attn,
                        "linear_phys_attn_dim": args.phys_attn_dim,
                        "linear_phys_attn_layers": args.phys_attn_layers,
                        "linear_spectral_layers": args.spectral_layers,
                        "linear_phys_window_physical": args.phys_window_physical,
                        "linear_phys_grad_feat": args.phys_grad_feat,
                        "linear_phys_mode": args.phys_mode,
                        "linear_green_patch_cond": args.green_patch_cond,
                        "linear_bank_bottleneck": args.bank_bottleneck,
                        "linear_sep_stencils": args.sep_stencils,
                        "linear_channels_last": args.channels_last,
                        "linear_seam_average": args.seam_average,
                        "linear_eta_embed_dim": args.eta_embed_dim,
                        "linear_eta_quant": args.eta_quant,
                        "refine_steps": args.refine_steps,
                        "refine_gate": (float(refine_gate.detach()) if refine_gate is not None else 0.0),
                        "energy_weight": args.energy_weight,
                        "invvisc_weight": args.invvisc_weight,
                        "target_eta_power": args.target_eta_power,
                        "linear_level_pyramid": args.level_pyramid,
                        "linear_dilated": args.dilated_stencils,
                        "linear_level_cond": args.level_cond,
                        "test_rel_l2": score, "out_channels": 4}, args.out)
        if _is_main():
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "best": best},
                       resume_path)
    print(f"\nbest mean velocity error {best:.4f}, saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
