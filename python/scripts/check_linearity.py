"""Verifies the structural guarantees of LinearOperator numerically.

For a fixed viscosity field the operator must be exactly linear in the forcing:
N(a f1 + b f2) = a N(f1) + b N(f2), and N(0) = 0. Both are checked on a randomly
initialised model with every eta-dependent mechanism switched on (gates, generated
Green core, stencil banks, degree attention, viscosity-patch attention), on the
level-3 mesh. Deviations are floating-point rounding, ~1e-6 relative in fp32.

    python scripts/check_linearity.py [dataset_root]
"""
import json, os, sys
import numpy as np
import torch

from terra_infer.operator import LinearOperator

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("TERRA_ML_DIR", "/hppfs/scratch/0E/di35guv2/ml"), "stokes_L3_hc")
mesh = json.load(open(os.path.join(root, "mesh.json")))
shape = tuple(mesh["velocity_shape"])
coords = np.fromfile(os.path.join(root, "coords_velocity.bin"),
                     dtype=np.float64).reshape(*shape, 3)
torch.manual_seed(0)

net = LinearOperator(5, 4, shape, coords, n_hidden=64, n_blocks=4, lmax=8, kmax=6,
                     n_conv=2, kernel=5, depth_gates=True, eta_gates=True,
                     eta_green=True, green_mlp=True, eta_stencils=4,
                     bank_bottleneck=16, mode_attn=16, phys_attn=8,
                     seam_average=True).eval()
# the zero-initialised gates would hide those paths: switch them on
with torch.no_grad():
    if getattr(net, "attn_gate", None) is not None:
        net.attn_gate.fill_(0.5)
    if getattr(net, "phys", None) is not None:
        net.phys.gate.fill_(0.5)

S, nx, ny, nr = shape
f1 = torch.randn(1, S, nx, ny, nr, 5) * 0.1
f2 = torch.randn(1, S, nx, ny, nr, 5) * 0.1
eta = torch.randn(1, S, nx, ny, nr)
f1[..., 4] = eta; f2[..., 4] = eta          # same viscosity, different forcing
a, b = 2.5, -0.75
comb = f1.clone(); comb[..., :4] = a * f1[..., :4] + b * f2[..., :4]
zero = f1.clone(); zero[..., :4] = 0.0

with torch.no_grad():
    y1, y2, yc, y0 = net(f1), net(f2), net(comb), net(zero)
lin = (yc - (a * y1 + b * y2)).abs().max() / (a * y1 + b * y2).abs().max()
print("linearity   |N(a f1 + b f2) - a N(f1) - b N(f2)| / |.| = %.2e" % lin)
print("zero input  |N(0)|                                   = %.2e" % y0.abs().max())
ok = lin < 1e-5 and y0.abs().max() == 0.0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
