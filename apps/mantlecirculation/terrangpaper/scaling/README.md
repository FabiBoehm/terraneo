# Strong-scaling sweep

Produces `cross_vendor_strong_scaling.png`: time per timestep against device
count, one line per MT resolution.

`config_scal_A3.toml` is the case every point runs: A3 physics, incompressible,
10 timesteps, entropy viscosity for the energy equation.

| directory | contents |
|---|---|
| `sng2_reproduction/` | 34 scripts, SuperMUC-NG Phase 2 (Intel PVC) |
| `lumi_reproduction/` | 37 scripts + generator + feeder, LUMI-G (AMD MI250X) |
| `submit/` | sweep generators for other machines, and the result collector |

## How a point is measured

Each point runs 10 timesteps (indices 0..9) with `--output-frequency 9`, writing
exactly one `timer_trees/timer_tree_9.json`. Per-step wall time is that file's
`timestep` node, `root_time / count`, where count is 9. `sum_time` is the sum
across ranks and `avg_time` the rank mean; neither should be divided by `count`.

Solver settings are fixed so every point does identical work: Stokes 10 FGMRES
iterations with restart 10, energy 50, all tolerances pinned to 0, two pre/post
smoothing steps.

## Two things that decide whether the numbers come out right

Pass the subdomain levels **per axis**, `--lat-sdr` and `--rad-sdr`. They are
usually unequal: (0,1) at 4 devices, (1,0) at 8, (2,0) at 32, (3,1) at 256. A
single `--refinement-level-subdomains N` applies N to both axes and inflates the
step time by up to a factor of four.

Keep each script's environment exactly as it is. On SuperMUC the sweep sets
`PSM3_GPUDIRECT=0` and deliberately sets neither `I_MPI_OFFLOAD_IPC` nor the
MR-cache variables; substituting the production environment costs 13-79 % per
timestep.

## Running

Every script takes `TERRANG_BIN` (required), and optionally `TERRANG_CFG`
(defaults to the config next to this file), `TERRANG_OUT`, `TERRANG_LOGDIR` and
`TERRANG_ACCOUNT`. See each subdirectory's README.
