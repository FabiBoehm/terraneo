# Strong-scaling sweep

**Figure produced:** `cross_vendor_strong_scaling.png` — time per timestep
against device count, one line per MT resolution.

`config_scal_A3.toml` is the case every point runs: A3 physics, incompressible,
10 timesteps, entropy viscosity for the energy equation. It replaces the
original `config_fscmb_nsurf_lvl6_10steps.toml`, which is a hard parse failure
on the current app because the viscosity law it names was removed from the law
table.

Subdirectories:

| directory | contents |
|---|---|
| `sng2_reproduction/` | 34 scripts, SuperMUC-NG Phase 2 (Intel PVC) |
| `lumi_reproduction/` | 37 scripts + generator + feeder, LUMI-G (AMD MI250X) |
| `submit/` | sweep generators for other machines, and the result collector |

## How a point is measured

Each point runs **10 timesteps** (indices 0..9) with `--output-frequency 9`,
which writes exactly one `timer_trees/timer_tree_9.json`. Per-step wall time is
that file's `timestep` node, `root_time / count`, where count is 9.

`sum_time` in the tree is the sum **across ranks** and `avg_time` the rank mean.
Neither should be divided by `count` to get a per-step time.

Solver settings are fixed so every point does identical work: Stokes 10 FGMRES
iterations with restart 10, energy 50, all tolerances pinned to 0 so the
iteration count is never cut short, two pre/post smoothing steps.

## Two things that decide whether the numbers come out right

**Subdomain decomposition.** The original campaigns set the lateral and radial
subdomain levels independently (`--lat-sdr` / `--rad-sdr`) and they are usually
unequal: (0,1) at 4 devices, (1,0) at 8, (2,0) at 32, (3,1) at 256. Passing a
single `--refinement-level-subdomains N` applies N to both axes and changes the
decomposition. Over 31 comparable points every matching decomposition
reproduced the published value to within -4 % / +8 %, while every mismatched one
was slow — median +56 %, up to +317 %. Not one matching point was slow and not
one mismatched point was fast.

**Environment.** On SuperMUC the sweep environment sets `PSM3_GPUDIRECT=0` and
sets *neither* `I_MPI_OFFLOAD_IPC` nor the MR-cache variables. Using the
production mantle-circulation environment instead costs 13–79 % per timestep,
growing with ranks per node, and single-rank points are unaffected — the
signature of intra-node GPU peer-to-peer being disabled.

## Running

Every script takes `TERRANG_BIN` (required), and optionally `TERRANG_CFG`
(defaults to the config next to this file), `TERRANG_OUT`, `TERRANG_LOGDIR` and
`TERRANG_ACCOUNT`. Set the account or edit the `CHANGEME` placeholder before
submitting. See each subdirectory's README.
