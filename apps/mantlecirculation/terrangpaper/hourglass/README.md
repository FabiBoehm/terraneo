# Hourglass control

Produces `hourglass_stab_convergence.pdf`: stand-alone multigrid residual per
V-cycle on MT256 (left) and the L2 velocity discretization error over MT32 to
MT512 (right).

The driver is `tests/test_epsilon_divdiv_quadrature_matrix.cpp`, built as
`test_epsilon_divdiv_quadrature_matrix`. It solves the viscous block on the
shell for a smooth manufactured solution with constant viscosity, once per
quadrature variant:

| series | environment | operator |
|---|---|---|
| `1pt`  | -                                  | one quadrature point |
| `1ptK` | `STAB_C_KERNGEN=0.3`               | one point plus hourglass control, C = 0.3 |
| `2ptK` | `STAB_C_KERNGEN=0.3` `RUN_QP=2`    | two radial quadrature points |

`RUN_SCALE=<level>` selects the fine level and restricts the run to the
multigrid study. Each run prints `ITER,<sol>,<series>,<level>,<cycle>,<relative
residual>,<L2 error>` and one closing `DISC,<sol>,<series>,<level>,<h>,<L2
error>,<dofs>` row, which is what the plot script reads.

## Running

`run_hourglass.sh` takes `TERRANG_BIN` (required), `TERRANG_MT` (32 ... 512)
and the optional `TERRANG_OUT` and `TERRANG_ACCOUNT`. It runs all three series
for one model size and concatenates them into `iter_MT<MT>.csv`; concatenate
those files over MT to get the full data set.

```
TERRANG_BIN=<build>/tests/test_epsilon_divdiv_quadrature_matrix \
TERRANG_MT=256 TERRANG_ACCOUNT=<project> \
  sbatch --nodes=1 --ntasks-per-node=1 run_hourglass.sh
```

The hierarchy starts at level 1, which caps the subdomain refinement at 1 and
therefore the rank count at 80. MT32 to MT256 fit on a single device; MT512
needs 80.
