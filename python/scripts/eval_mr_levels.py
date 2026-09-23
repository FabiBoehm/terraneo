"""d8-multiresolution stage-1 on the degree-8 TEST sets at L3, L4, L5.

Reproduces the trainer's evaluation exactly (standardised log-eta channel with
the checkpoint's level-1 stats, mean-eta target scaling, shell-masked u,
mean-free p); L5 is zero-shot with the hypernetwork Green queried beyond the
trained truncations.
"""
import glob, sys
import json

import numpy as np
import torch

from terra_infer.operator import LinearOperator, load_state
from terra_infer import symmetry
import os
TTA = os.environ.get("EVAL_TTA", "0") == "1"
SMOOTH = os.environ.get("EVAL_SMOOTH", "0") == "1"
BINS = os.environ.get("EVAL_BINS", "0") == "1"
EVAL_DEV = torch.device(os.environ.get("EVAL_DEVICE", "cpu"))
MOE_T = [float(t) for t in os.environ.get("EVAL_MOE_THRESH", "0").split(",") if float(t) > 0]   # thresholds: ck[k] serves band k

M = os.environ.get("TERRA_ML_DIR", "/hppfs/scratch/0E/di35guv2/ml")   # datasets and checkpoints
CK = sys.argv[1] if len(sys.argv) > 1 else M + "/w3d_linear_d8mr.pt"
CKS = CK.split(",")
cks = [torch.load(c, map_location="cpu", weights_only=False) for c in CKS]
ck = cks[0]
CUR_CONTRAST = 1.0
print("checkpoint", CK, "TTA", TTA, "MOE_T", MOE_T, "SMOOTH", SMOOTH, "eta_gates", ck.get("linear_eta_gates"), "eta_green", ck.get("linear_eta_green"))
le_mean, le_std = float(ck["log_eta_mean"]), float(ck["log_eta_std"])
print(f"le stats: mean {le_mean:.3f} std {le_std:.3f}")

CASES = [("L3", "stokes_L3_d8", ck["spherical"], ck["radial_modes"]), ("L4", "stokes_L4_d8", 24, 16),
         ("L5 zero-shot", "stokes_L5_d8", 32, 16),
         ("L6 zero-shot", "stokes_L6_d8", 32, 16)]
import os
SUF = os.environ.get("EVAL_SUFFIX", "")          # e.g. "_hc" -> stokes_L{n}_hc test sets
if SUF:
    CASES = [(n_, r_.replace("_d8", SUF), a_, b_) for (n_, r_, a_, b_) in CASES]
CASES = [c for c in CASES if os.path.isdir(f"{M}/{c[1]}/test")]
# optional per-level truncation override, e.g. EVAL_LM_L6=48 EVAL_KM_L6=24
def _ov(nm, lm, km):
    tag = nm.split()[0]
    return (int(os.environ.get(f"EVAL_LM_{tag}", lm)),
            int(os.environ.get(f"EVAL_KM_{tag}", km)))
CASES = [(n_, r_) + _ov(n_, a_, b_) for (n_, r_, a_, b_) in CASES]
CASES = [c for c in CASES if not os.environ.get("EVAL_ONLY") or c[0].startswith(os.environ["EVAL_ONLY"])]
NTEST = int(os.environ.get("EVAL_NTEST", "16"))
NTEST_L6 = int(os.environ.get("EVAL_NTEST_L6", "8"))
TTA_MAXLEVEL = int(os.environ.get("EVAL_TTA_MAXLEVEL", "5"))  # TTA = 10 forward passes: too costly at L6 on CPU
for name, root, LM, KM in CASES:
    mesh = json.load(open(f"{M}/{root}/mesh.json"))
    Sv, Sp = tuple(mesh["velocity_shape"]), tuple(mesh["pressure_shape"])
    nv, npp = int(np.prod(Sv)), int(np.prod(Sp))
    coords = np.fromfile(f"{M}/{root}/coords_velocity.bin",
                         dtype=np.float64).reshape(*Sv, 3)
    net = LinearOperator(5, 4, Sv[1:4], coords, n_hidden=ck["hidden"],
                         n_blocks=ck["heads"], lmax=LM, kmax=KM,
                         n_conv=ck.get("linear_convs", 2),
                         kernel=ck.get("linear_kernel", 5),
                         depth_gates=ck.get("linear_depth_gates", False),
                         eta_gates=ck.get("linear_eta_gates", False),
                         eta_green=ck.get("linear_eta_green", False),
                         nonlin=ck.get("linear_nonlin", False),
                         dilated=ck.get("linear_dilated", False),
                         level_cond=ck.get("linear_level_cond", False),
                         pyramid=ck.get("linear_pyramid", 0),
                         level_pyramid=ck.get("linear_level_pyramid", False),
                         stencil_scale=ck.get("linear_stencil_scale", False),
                         eta_lateral=ck.get("linear_eta_lateral", 0),
                         eta_stencils=ck.get("linear_eta_stencils", 0), multi_dilation=ck.get("linear_multi_dilation", False), mode_attn=ck.get("linear_mode_attn", 0), eta_embed_dim=ck.get("linear_eta_embed_dim", 16), eta_quant=ck.get("linear_eta_quant", 0), seam_average=True, bank_bottleneck=ck.get("linear_bank_bottleneck", 0), sep_stencils=ck.get("linear_sep_stencils", False),
                         green_mlp=True).eval()
    lev = int(mesh["level"])
    g = symmetry.SymmetryGroup(coords) if (TTA and lev <= TTA_MAXLEVEL) else None
    if TTA and lev > TTA_MAXLEVEL: print(f"  ({name}: TTA skipped above level {TTA_MAXLEVEL})", flush=True)
    X_SPEC = [(3, True, True), (1, False, True), (1, False, False)]
    Y_SPEC = [(3, True, True), (1, False, True)]
    def by_spec(t, spec, k):
        out, o = [], 0
        for w, vec, odd in spec:
            out.append(g.transform(t[..., o:o + w], k, vec, odd)); o += w
        return torch.cat(out, dim=-1)
    def predict(xt):
        if g is None:
            return net(xt)
        acc = net(xt)
        for k in range(1, len(g)):
            acc = acc + by_spec(net(by_spec(xt, X_SPEC, k)), Y_SPEC, g.inverse_index(k))
        return acc / len(g)
    nets = []
    for ck_i in cks:
        net_i = LinearOperator(5, 4, Sv[1:4], coords, n_hidden=ck_i["hidden"], n_blocks=ck_i["heads"], lmax=LM, kmax=KM,
                               n_conv=ck_i.get("linear_convs", 2), kernel=ck_i.get("linear_kernel", 5),
                               depth_gates=ck_i.get("linear_depth_gates", False), eta_gates=ck_i.get("linear_eta_gates", False),
                               eta_green=ck_i.get("linear_eta_green", False), nonlin=ck_i.get("linear_nonlin", False),
                               dilated=ck_i.get("linear_dilated", False), level_cond=ck_i.get("linear_level_cond", False),
                               pyramid=ck_i.get("linear_pyramid", 0), level_pyramid=ck_i.get("linear_level_pyramid", False),
                               stencil_scale=ck_i.get("linear_stencil_scale", False),
                               eta_lateral=ck_i.get("linear_eta_lateral", 0),
                               eta_stencils=ck_i.get("linear_eta_stencils", 0), multi_dilation=ck_i.get("linear_multi_dilation", False), mode_attn=ck_i.get("linear_mode_attn", 0), eta_embed_dim=ck_i.get("linear_eta_embed_dim", 16), eta_quant=ck_i.get("linear_eta_quant", 0), seam_average=True, bank_bottleneck=ck_i.get("linear_bank_bottleneck", 0), sep_stencils=ck_i.get("linear_sep_stencils", False),
                               green_mlp=True).eval()
        load_state(net_i, ck_i["model"])
        nets.append(net_i.to(EVAL_DEV))
    # every checkpoint standardises log-eta with ITS OWN training statistics;
    # the input is built with ck[0]'s, so re-map the eta channel per model
    STATS = [(float(c_["log_eta_mean"]), float(c_["log_eta_std"])) for c_ in cks]
    def renorm(xt, i):
        if i == 0:
            return xt
        m0, s0 = STATS[0]; mi, si = STATS[i]
        le_ = xt[..., 4:5] * s0 + m0
        return torch.cat([xt[..., :4], (le_ - mi) / si], -1)
    def net(xt):
        if MOE_T and len(nets) >= 2:
            i = min(sum(CUR_CONTRAST >= t for t in MOE_T), len(nets) - 1)
            return nets[i](renorm(xt, i))
        return sum(n_(renorm(xt, i)) for i, n_ in enumerate(nets)) / len(nets)
    sizes = [3 * nv, npp, nv, 3 * nv, npp, nv, nv]
    eu, ep, ctr = [], [], []
    _files = sorted(glob.glob(f"{M}/{root}/test/sample_*.bin"))[:(NTEST_L6 if lev >= 6 else NTEST)]
    # the samples are 97 MiB each at L6 and live on the parallel filesystem, so read the
    # next one in a background thread while the current one is being evaluated
    from concurrent.futures import ThreadPoolExecutor
    _pool = ThreadPoolExecutor(2)
    _pending = {i: _pool.submit(np.fromfile, fn, dtype=np.float32)
                for i, fn in enumerate(_files[:2])}
    for _i, f in enumerate(_files):
        raw = _pending.pop(_i).result()
        if _i + 2 < len(_files):
            _pending[_i + 2] = _pool.submit(np.fromfile, _files[_i + 2], dtype=np.float32)
        off, fld = 0, []
        for s in sizes:
            fld.append(raw[off:off + s]); off += s
        u = fld[0].reshape(*Sv, 3).astype(np.float64)
        pfin = fld[5].reshape(*Sv).astype(np.float64)
        eta = fld[2].reshape(*Sv).astype(np.float64)
        le = np.log(np.maximum(eta, 1e-12))
        c = np.exp(le.mean())
        CUR_CONTRAST = float(np.exp(le.max() - le.min()))
        x = np.concatenate([fld[3].reshape(*Sv, 3),
                            fld[6].reshape(*Sv)[..., None],
                            ((le - le_mean) / le_std)[..., None]],
                           -1).astype(np.float32)
        with torch.no_grad():
            out = predict(torch.from_numpy(x)[None].to(EVAL_DEV))[0]
            out = out.float().cpu().numpy().astype(np.float64)
        zu = out[..., :3]
        if SMOOTH:
            kk = max(1, (Sv[1] - 1) // 8)
            if kk > 1:
                t = torch.from_numpy(zu).permute(0, 4, 1, 2, 3)
                t = torch.nn.functional.avg_pool3d(t, kernel_size=2 * kk - 1, stride=1, padding=kk - 1, count_include_pad=False)
                zu = t.permute(0, 2, 3, 4, 1).numpy()
        zu[:, :, :, 0] = 0.0
        zu[:, :, :, -1] = 0.0
        A_ = float(ck.get("target_eta_power", 0.0))
        sc = (eta ** A_) * (c ** (1.0 - A_))
        zu = zu / sc[..., None]
        ut = u
        eu.append(np.linalg.norm(zu - ut) / np.linalg.norm(ut))
        ctr.append(float(np.exp(le.max() - le.min())))
        pp = out[..., 3] - out[..., 3].mean()
        pt = pfin - pfin.mean()
        ep.append(np.linalg.norm(pp - pt) / np.linalg.norm(pt))
    if BINS:
        eu_a, c_a = np.array(eu), np.array(ctr)
        parts = []
        for lo, hi in [(1, 10), (10, 100), (100, 1000), (1000, 1e6)]:
            mk = (c_a >= lo) & (c_a < hi)
            if mk.any(): parts.append(f"[{lo:g},{hi:g}) n={mk.sum()} u={eu_a[mk].mean():.3f}")
        print(f"{name:>14} by contrast: " + "  ".join(parts))

    print(f"{name:>14} (lmax {LM}, kmax {KM}): "
          f"u {np.mean(eu):.4f} (best {np.min(eu):.3f} med {np.median(eu):.3f} "
          f"max {np.max(eu):.3f})  p {np.mean(ep):.4f}")
print("EVAL_DONE")
