# Strong-scaling sweep

Produces Fig. 5 of the preprint: time per timestep against device
count, one line per MT resolution.

`config_scal_A3.toml` is the case every point runs: A3 physics, incompressible,
10 timesteps, entropy viscosity for the energy equation.

| directory | contents |
|---|---|
| `sng2_reproduction/` | 34 scripts, SuperMUC-NG Phase 2 (Intel PVC) |
| `lumi_reproduction/` | 37 scripts, LUMI-G (AMD MI250X) |
| `submit/` | sweep generators for other machines, and the result collector |

## How a point is measured

Each point runs 10 timesteps (indices 0..9) with `--output-frequency 9`, writing
exactly one `timer_trees/timer_tree_9.json`. Per-step wall time is that file's
`timestep` node, `root_time / count`, where count is 9. `sum_time` is the sum
across ranks and `avg_time` the rank mean; neither should be divided by `count`.

Solver settings are fixed so every point does identical work: Stokes 10 FGMRES
iterations with restart 10, energy 50, all tolerances pinned to 0, two pre/post
smoothing steps.

## Running

One script per point, one `sbatch` per script. Every script takes `TERRANG_BIN`
(required) and `TERRANG_ACCOUNT`, and optionally `TERRANG_CFG` (defaults to
`config_scal_A3.toml` next to this file), `TERRANG_OUT` (defaults to a directory
named after the point) and `TERRANG_LOGDIR` (where the Slurm logs go).

SuperMUC-NG Phase 2:

```
cd sng2_reproduction
TERRANG_BIN=<build>/apps/mantlecirculation/mantlecirculation \
TERRANG_ACCOUNT=<project> \
  sbatch run_MT256_g64_A3_oe.sh
```

LUMI-G:

```
cd lumi_reproduction
TERRANG_BIN=<build>/apps/mantlecirculation/mantlecirculation \
TERRANG_ACCOUNT=<project> \
  sbatch std_MT256_g64_std.sh
```

The whole sweep on either machine:

```
for s in run_MT*_A3_oe.sh; do sbatch "$s"; done      # sng2_reproduction
for s in std_MT*_std.sh;   do sbatch "$s"; done      # lumi_reproduction
```

Then collect the per-step times, and compare against the published numbers if
a CSV is given:

```
python3 lumi_reproduction/collect_lumi.py <outroot> [published.csv]
```
