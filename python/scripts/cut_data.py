"""Compute great-circle cuts of the best and worst held-out predictions and save them as
.npz (no matplotlib on the cluster). Plotting happens on the client."""
import json, glob, sys, os
import numpy as np, torch
from scipy.spatial import cKDTree
from terra_infer.operator import LinearOperator, load_state

M = os.environ.get("TERRA_ML_DIR", "/hppfs/scratch/0E/di35guv2/ml")
ROOT, LM, KM, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
OUT = sys.argv[5]
CKS = os.environ.get("PLOT_CKS", "gen_s3pa128").split(",")
DEV = torch.device(os.environ.get("PLOT_DEVICE", "cpu"))
THRESH = [10.0, 100.0, 1000.0]

mesh = json.load(open(f"{M}/{ROOT}/mesh.json"))
Sv, Sp = tuple(mesh["velocity_shape"]), tuple(mesh["pressure_shape"])
nv, npp = int(np.prod(Sv)), int(np.prod(Sp))
co = np.fromfile(f"{M}/{ROOT}/coords_velocity.bin", dtype=np.float64).reshape(*Sv, 3)

nets, stats = [], []
for name in CKS:
    ck = torch.load(f"{M}/w3d_linear_{name}.pt", map_location="cpu", weights_only=False)
    n = LinearOperator(5, 4, Sv[1:4], co, n_hidden=ck["hidden"], n_blocks=ck["heads"],
                       lmax=LM, kmax=KM, n_conv=ck.get("linear_convs", 4),
                       kernel=ck.get("linear_kernel", 5),
                       depth_gates=ck.get("linear_depth_gates", False), eta_gates=True,
                       eta_green=True, green_mlp=True,
                       eta_stencils=ck.get("linear_eta_stencils", 0),
                       bank_bottleneck=ck.get("linear_bank_bottleneck", 0),
                       mode_attn=ck.get("linear_mode_attn", 0),
                       eta_embed_dim=ck.get("linear_eta_embed_dim", 16),
                       eta_quant=ck.get("linear_eta_quant", 0), seam_average=True,
                       phys_attn=ck.get("linear_phys_attn", 0),
                       phys_attn_dim=ck.get("linear_phys_attn_dim", 32),
                       phys_attn_layers=ck.get("linear_phys_attn_layers", 1),
                       spectral=ck.get("linear_spectral", True),
                       spectral_layers=ck.get("linear_spectral_layers", 1),
                       phys_window_physical=ck.get("linear_phys_window_physical", False),
                       phys_grad_feat=ck.get("linear_phys_grad_feat", False),
                       phys_mode=ck.get("linear_phys_mode", "mlp"),
                       green_patch_cond=ck.get("linear_green_patch_cond", False),
                       ).eval().to(DEV)
    load_state(n, ck["model"])
    nets.append(n); stats.append((float(ck["log_eta_mean"]), float(ck["log_eta_std"])))
    print("built", name, "spectral_layers", ck.get("linear_spectral_layers", 1),
          "phys_attn", ck.get("linear_phys_attn", 0), flush=True)

NT, NR = 720, 4 * Sv[1]
th = np.linspace(0.0, 2.0 * np.pi, NT)
rr = np.linspace(0.5, 1.0, NR)
TH, RR = np.meshgrid(th, rr)
PX, PY = RR * np.cos(TH), RR * np.sin(TH)
pts = np.stack([RR * np.cos(TH), np.zeros_like(TH), RR * np.sin(TH)], -1).reshape(-1, 3)
_, idx = cKDTree(co.reshape(-1, 3)).query(pts, k=1)
def cut(field):
    return field.reshape(nv, -1)[idx].reshape(NR, NT, -1).squeeze(-1)

rows = []
for f0 in sorted(glob.glob(f"{M}/{ROOT}/test/sample_*.bin"))[:N]:
    raw = np.fromfile(f0, dtype=np.float32); o = 0
    u = raw[o:o+3*nv].reshape(*Sv, 3).astype(np.float32); o += 3*nv + npp
    eta = raw[o:o+nv].reshape(*Sv).astype(np.float32); o += nv
    f_u = raw[o:o+3*nv].reshape(*Sv, 3).astype(np.float32); o += 3*nv + npp + nv
    f_p = raw[o:o+nv].reshape(*Sv).astype(np.float32)
    le = np.log(np.maximum(eta, 1e-12))
    contrast = float(np.exp(le.max() - le.min()))
    k = min(sum(contrast >= t for t in THRESH), len(nets) - 1)
    lm_, ls_ = stats[k]
    x = np.concatenate([f_u, f_p[..., None], ((le - lm_) / ls_)[..., None]], -1)
    with torch.no_grad():
        y = nets[k](torch.from_numpy(x)[None].to(DEV))[0].float().cpu().numpy()
    up = y[..., :3] / float(np.exp(le.mean()))
    up[:, :, :, 0] = 0.0; up[:, :, :, -1] = 0.0
    err = float(np.linalg.norm(up - u) / np.linalg.norm(u))
    rows.append((err, contrast, u, up, eta, os.path.basename(f0)))
rows.sort(key=lambda r: r[0])
errs = np.array([r[0] for r in rows])
print(f"{ROOT}: {len(rows)} samples, best {errs[0]:.3f} worst {errs[-1]:.3f} "
      f"median {np.median(errs):.3f} mean {errs.mean():.3f}", flush=True)

out = dict(PX=PX, PY=PY, errs=errs, root=ROOT)
for tag, r in (("best", rows[0]), ("worst", rows[-1])):
    err, contrast, u, up, eta, nm = r
    out[f"{tag}_eta"] = cut(eta).astype(np.float32)
    out[f"{tag}_exact"] = cut(np.linalg.norm(u, axis=-1)).astype(np.float32)
    out[f"{tag}_pred"] = cut(np.linalg.norm(up, axis=-1)).astype(np.float32)
    out[f"{tag}_err"] = err; out[f"{tag}_contrast"] = contrast; out[f"{tag}_name"] = nm
np.savez_compressed(OUT, **out)
print("wrote", OUT, flush=True)
