"""Write manufactured test problems in the mc app's raw f64 layout, plus the
router's one-shot prediction as the warm-start guess.

prob_NNNN_eta.bin    : eta,           [S][i][j][k]
prob_NNNN_xstar.bin  : exact solution, velocity [3][S][i][j][k] then pressure [S][i][j][k]
prob_NNNN_x0.bin     : one-shot prediction, same layout
"""
import argparse, glob, json, os
import numpy as np
import torch
from terra_infer.operator import LinearOperator, load_state

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cks", required=True); ap.add_argument("--thresh", default="")
ap.add_argument("--lmax", type=int, default=12); ap.add_argument("--kmax", type=int, default=8)
ap.add_argument("--no-ctl", action="store_true")
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--raw-seams", action="store_true",
                help="do NOT average duplicated seam nodes (the guess then lies outside the "
                     "FE space and the solver cannot remove its inconsistent part)")
args = ap.parse_args()

def _seam_averager(coords):
    """Nodes on a diamond seam are stored once per diamond. A finite-element vector
    must hold the SAME value in every copy; a field computed independently per
    diamond does not, and a Krylov method can never remove the inconsistent part of
    an initial guess, because every direction it searches is consistent. Returns a
    function that averages the copies of each physical node."""
    key = np.round(coords.reshape(-1, coords.shape[-1]), 9)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    counts = np.bincount(inv)

    def make_consistent(f):
        sh = f.shape
        flat = f.reshape(len(inv), -1)
        out = np.empty_like(flat)
        for c in range(flat.shape[1]):
            sums = np.bincount(inv, weights=flat[:, c])
            out[:, c] = (sums / counts)[inv]
        return out.reshape(sh)

    return make_consistent


mesh = json.load(open(os.path.join(args.data, "mesh.json")))
Sv, Sp = tuple(mesh["velocity_shape"]), tuple(mesh["pressure_shape"])
nv, npp = int(np.prod(Sv)), int(np.prod(Sp))
coords = np.fromfile(os.path.join(args.data, "coords_velocity.bin"),
                     dtype=np.float64).reshape(*Sv, 3)
coords_p = np.fromfile(os.path.join(args.data, "coords_pressure.bin"),
                       dtype=np.float64).reshape(*Sp, 3)
mk_cons_v = _seam_averager(coords)
mk_cons_p = _seam_averager(coords_p)
os.makedirs(args.out, exist_ok=True)

cks = [torch.load(c, map_location="cpu", weights_only=False) for c in args.cks.split(",")]
T = [float(t) for t in args.thresh.split(",") if t.strip()]
nets = []



for ck in cks:
    n = LinearOperator(5, 4, Sv[1:4], coords, n_hidden=ck["hidden"], n_blocks=ck["heads"],
                       lmax=args.lmax, kmax=args.kmax,
                       n_conv=ck.get("linear_convs", 2), kernel=ck.get("linear_kernel", 5),
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
                       eta_stencils=ck.get("linear_eta_stencils", 0),
                       multi_dilation=ck.get("linear_multi_dilation", False),
                       mode_attn=ck.get("linear_mode_attn", 0), seam_average=True, eta_embed_dim=ck.get("linear_eta_embed_dim", 16), eta_quant=ck.get("linear_eta_quant", 0), bank_bottleneck=ck.get("linear_bank_bottleneck", 0), sep_stencils=ck.get("linear_sep_stencils", False),
                       green_mlp=True).eval()
    load_state(n, ck["model"]); nets.append(n)

sizes = [3 * nv, npp, nv, 3 * nv, npp, nv, nv]
files = sorted(glob.glob(os.path.join(args.data, "test", "sample_*.bin")))[: args.samples]
for i, f in enumerate(files):
    raw = np.fromfile(f, dtype=np.float32)
    off, fld = 0, []
    for s in sizes:
        fld.append(raw[off:off + s]); off += s
    u = fld[0].reshape(*Sv, 3).astype(np.float64)
    pc = fld[1].reshape(*Sp).astype(np.float64)
    eta = fld[2].reshape(*Sv).astype(np.float64)
    f_u = fld[3].reshape(*Sv, 3); fp_f = fld[6].reshape(*Sv)
    le = np.log(np.maximum(eta, 1e-12))
    contrast = float(np.exp(le.max() - le.min()))
    k = min(sum(contrast >= t for t in T), len(nets) - 1)
    lm, ls = float(cks[k]["log_eta_mean"]), float(cks[k]["log_eta_std"])
    xin = np.concatenate([f_u, fp_f[..., None], ((le - lm) / ls)[..., None]], -1).astype(np.float32)
    with torch.no_grad():
        out = nets[k](torch.from_numpy(xin)[None])[0].numpy().astype(np.float64)
    uu = out[..., :3] / np.exp(le.mean())
    uu[:, :, :, 0] = 0.0; uu[:, :, :, -1] = 0.0
    pp = out[..., 3][:, ::2, ::2, ::2]
    if not args.raw_seams:
        uu = mk_cons_v(uu)                      # project the guess into the FE space
        pp = mk_cons_p(pp[..., None])[..., 0]
        uu[:, :, :, 0] = 0.0; uu[:, :, :, -1] = 0.0   # averaging must not revive the BC

    def wr(path, uu_, pp_):                      # (S,nx,ny,nr,3) -> [3][S][i][j][k]
        with open(path, "wb") as fh:
            fh.write(np.ascontiguousarray(np.moveaxis(uu_, -1, 0), np.float64).tobytes())
            fh.write(np.ascontiguousarray(pp_, np.float64).tobytes())
    # control guess: exact solution + a SMOOTH RANDOM error of the same relative
    # size as the network's. If this one converges while the network's stalls,
    # the network's error is structurally special (near-null); if both stall,
    # the stall is a property of warm starts in this solver.
    rng = np.random.default_rng(1234 + i)
    def smooth(a, n=6):
        for _ in range(n):
            b = a.copy()
            for ax in (1, 2, 3):
                b = b + np.roll(a, 1, ax) + np.roll(a, -1, ax)
            a = b / 7.0
        return a
    eu = smooth(rng.standard_normal(u.shape))
    eu[:, :, :, 0] = 0.0; eu[:, :, :, -1] = 0.0
    eu *= np.linalg.norm(uu - u) / max(np.linalg.norm(eu), 1e-30)
    ep_ = smooth(rng.standard_normal(pc.shape))
    ep_ *= np.linalg.norm(pp - pc) / max(np.linalg.norm(ep_), 1e-30)

    tag = f"{i:04d}"
    eta.astype(np.float64).tofile(os.path.join(args.out, f"prob_{tag}_eta.bin"))
    wr(os.path.join(args.out, f"prob_{tag}_xstar.bin"), u, pc)
    wr(os.path.join(args.out, f"prob_{tag}_x0.bin"), uu, pp)
    if not args.no_ctl:
        wr(os.path.join(args.out, f"prob_{tag}_x0r.bin"), u + eu, pc + ep_)
    rel = np.linalg.norm(uu - u) / np.linalg.norm(u)
    print(f"  prob {tag}: contrast {contrast:9.1f}  expert {k}  one-shot u-err {rel:.3f}", flush=True)
print(f"{len(files)} problems written to {args.out}")
print("BENCHPROB_DONE")
