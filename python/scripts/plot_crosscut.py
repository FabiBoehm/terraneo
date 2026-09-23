"""Cross-sections of the best and worst predictions on a held-out set.
For each test sample the router picks an expert, the prediction is compared with the exact
solution, and the samples with the smallest and largest relative error are plotted as a
full great-circle cut through the spherical shell: viscosity, exact, predicted, difference.
"""
import json, glob, sys, os
import numpy as np, torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from terra_infer.operator import LinearOperator, load_state
M = os.environ.get("TERRA_ML_DIR", "/hppfs/scratch/0E/di35guv2/ml")   # datasets and checkpoints
ROOT, LM, KM, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
OUT = sys.argv[5]
CKS = os.environ.get("PLOT_CKS", "egmr6g,egmrmc4,egmrxcS,c8ckpt").split(",")
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
                       kernel=ck.get("linear_kernel", 5), depth_gates=True, eta_gates=True,
                       eta_green=True, green_mlp=True,
                       eta_stencils=ck.get("linear_eta_stencils", 0),
                       bank_bottleneck=ck.get("linear_bank_bottleneck", 0),
                       mode_attn=ck.get("linear_mode_attn", 0),
                       eta_embed_dim=ck.get("linear_eta_embed_dim", 16),
                       eta_quant=ck.get("linear_eta_quant", 0), seam_average=True).eval()
    load_state(n, ck["model"])
    nets.append(n); stats.append((float(ck["log_eta_mean"]), float(ck["log_eta_std"])))

# --- the cut: a full great circle in the y = 0 plane, sampled on a polar grid -------------
NT, NR = 720, 4 * Sv[1]
th = np.linspace(0.0, 2.0 * np.pi, NT)
rr = np.linspace(0.5, 1.0, NR)
TH, RR = np.meshgrid(th, rr)
PX, PY = RR * np.cos(TH), RR * np.sin(TH)          # plot coordinates, a true annulus
pts = np.stack([RR * np.cos(TH), np.zeros_like(TH), RR * np.sin(TH)], -1).reshape(-1, 3)
tree = cKDTree(co.reshape(-1, 3))
_, idx = tree.query(pts, k=1)                       # nearest mesh node for every grid point
def cut(field):                                     # field: (10, n, n, n[, c]) -> (NR, NT)
    flat = field.reshape(nv, -1)
    return flat[idx].reshape(NR, NT, -1).squeeze(-1)

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
        y = nets[k](torch.from_numpy(x)[None])[0].numpy()
    up = y[..., :3] / float(np.exp(le.mean()))
    up[:, :, :, 0] = 0.0; up[:, :, :, -1] = 0.0
    err = float(np.linalg.norm(up - u) / np.linalg.norm(u))
    rows.append((err, contrast, k, u, up, eta, os.path.basename(f0)))
rows.sort(key=lambda r: r[0])
best, worst = rows[0], rows[-1]
print(f"{ROOT}: {len(rows)} samples, error {rows[0][0]:.3f} (best) .. {rows[-1][0]:.3f} (worst), "
      f"median {np.median([r[0] for r in rows]):.3f}")

fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.4))
for r, (err, contrast, k, u, up, eta, nm) in enumerate((best, worst)):
    mag_t = cut(np.linalg.norm(u, axis=-1))
    mag_p = cut(np.linalg.norm(up, axis=-1))
    dif = mag_p - mag_t
    et = cut(eta)
    lim = max(mag_t.max(), mag_p.max())
    ims = [
        (axes[r, 0], et, "viscosity $\\eta$", "viridis",
         dict(norm=(matplotlib.colors.LogNorm(vmin=max(et.min(), 1e-12), vmax=et.max())
                    if et.max() / max(et.min(), 1e-12) >= 10.0 else
                    matplotlib.colors.Normalize(vmin=et.min(), vmax=et.max())))),
        (axes[r, 1], mag_t, "$|u|$ exact", "magma", dict(vmin=0, vmax=lim)),
        (axes[r, 2], mag_p, "$|u|$ predicted", "magma", dict(vmin=0, vmax=lim)),
        (axes[r, 3], dif, "difference", "coolwarm",
         dict(vmin=-abs(dif).max(), vmax=abs(dif).max())),
    ]
    for ax, arr, title, cmap, kw in ims:
        im = ax.pcolormesh(PX, PY, arr, cmap=cmap, shading="auto", rasterized=True, **kw)
        ax.set_aspect("equal"); ax.axis("off")
        if r == 0:
            ax.set_title(title, fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        if not isinstance(kw.get("norm"), matplotlib.colors.LogNorm):
            cb.locator = matplotlib.ticker.MaxNLocator(4); cb.update_ticks()
        else:
            cb.locator = matplotlib.ticker.LogLocator(numticks=5); cb.update_ticks()
    axes[r, 0].text(-1.25, 0.0, ("best" if r == 0 else "worst") +
                    f"\nerr {err:.3f}\ncontrast {contrast:.0f}", fontsize=9,
                    rotation=90, va="center", ha="center")
fig.suptitle(f"Great-circle cut through the shell, "
             f"{ROOT.replace('stokes_','').replace('_d8','')}: best and worst of "
             f"{len(rows)} held-out problems", fontsize=11)
fig.subplots_adjust(hspace=0.02, wspace=0.10, top=0.92, bottom=0.01)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
