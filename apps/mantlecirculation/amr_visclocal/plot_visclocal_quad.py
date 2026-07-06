#!/usr/bin/env python3
"""2x2 SVG figure for the localized-feature AMR study:
  [uniform grid]   [adaptive grid]
  [|u| solution]   [error-vs-DoFs convergence]
Mesh/solution panels are the high-res ParaView renders (embedded raster); the convergence panel is true
vector (zoomable). Saved as SVG (+ PNG). Usage: python3 plot_visclocal_quad.py [logfile]"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = os.path.expanduser(os.environ.get("VISCLOCAL_OUT", "~/visclocal_out"))
PV  = os.path.join(OUT, "pv")
log = sys.argv[1] if len(sys.argv) > 1 else max(
    glob.glob("/hnvme/workspace/iwia054h-mantle/slurm_logs/amr_visclocal.*.out"), key=os.path.getmtime)

pat = re.compile(r"\[(uni_L|ada_R)(\d+)\s*\]\s+dofs\s+(\d+)\s+FGMRES\s+(\d+).*?\|\|u-u\*\|\|\s+([\d.eE+-]+)")
uni, ada = [], []
for kind, k, dofs, it, l2 in pat.findall(open(log).read()):
    (uni if kind == "uni_L" else ada).append((int(dofs), int(it), float(l2)))
uni, ada = np.array(sorted(uni)), np.array(sorted(ada))

fig, ax = plt.subplots(2, 2, figsize=(13, 12))

def show(a, path, title):
    if os.path.exists(path):
        a.imshow(mpimg.imread(path))
    else:
        a.text(0.5, 0.5, "missing\n" + os.path.basename(path), ha="center", va="center")
    a.set_title(title, fontsize=13); a.axis("off")

show(ax[0, 0], os.path.join(PV, "grid_uniform.png"),  f"uniform grid  ({int(uni[-1,0]):,} DoFs, refined everywhere)")
show(ax[0, 1], os.path.join(PV, "grid_adaptive.png"), f"adaptive grid  ({int(ada[-1,0]):,} DoFs, uniform in the coeff-region, coarse outside)")
show(ax[1, 0], os.path.join(PV, "sol_adaptive.png"),  r"solution  $|u|$  (localized velocity bump at the blob)")

b = ax[1, 1]
b.loglog(uni[:, 0], uni[:, 2], "o-", color="#c0392b", lw=2, ms=8, label="uniform")
b.loglog(ada[:, 0], ada[:, 2], "s-", color="#2471a3", lw=2.2, ms=8, label="adaptive (ball: refine coeff-region, coarsen far field)")
d0, e0 = uni[0, 0], uni[0, 2]; dd = np.array([d0, uni[-1, 0]])
b.loglog(dd, e0 * (dd / d0) ** (-2 / 3), "k:", lw=1, alpha=0.5, label="2nd order (3D)")
for arr, c, dy in ((uni, "#c0392b", 6), (ada, "#2471a3", -13)):
    for d, it, l2 in arr:
        b.annotate(f"{int(it)}", (d, l2), textcoords="offset points", xytext=(5, dy), fontsize=8, color=c)
b.set_xlabel("degrees of freedom"); b.set_ylabel(r"$\|u_h - u^*\|$  (mass-weighted $L^2$)")
b.set_title("error vs DoFs  (labels = FGMRES iters)", fontsize=13)
b.grid(True, which="both", ls=":", alpha=0.5); b.legend(fontsize=10)

fig.suptitle("Localized viscosity blob + co-located feature: uniform vs interface-preserving AMR (geometric ball)", fontsize=15, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
svg = os.path.join(OUT, "visclocal_quad.svg")
fig.savefig(svg, dpi=300)          # dpi controls the embedded-raster resolution of the mesh panels in the SVG
fig.savefig(os.path.join(OUT, "visclocal_quad.png"), dpi=200)
print("wrote", svg)
