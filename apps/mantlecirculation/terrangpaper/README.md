# TERRA-NG GPU paper: benchmark and scaling inputs

Everything the paper's mantle-circulation results were produced from, in one place.
Previously these files were split between `apps/mantlecirculation/parameterfiles/`
and `apps/mantlecirculation/bench_mt/`, and the sbatch launch scripts were not in
the repository at all.

```
benchmarks/           verification benchmark configs (A3, C1, C3, C4, C5), MMOC
production/           the MT1024 MMOC production run
scaling/              strong-scaling sweep
  config_scal_A3.toml                    the sweep case, current app
  sng2/                                  SuperMUC-NG Phase 2 launch scripts
  lumi/                                  LUMI-G launch scripts
  sng2_reproduction/                     re-run of the sng2 sweep, Sept 2026
  submit/                                sweep generators and collector
```

## Benchmarks

One config per case, all using MMOC for the energy equation at Courant number
2.7, which is what the paper's verification figure reports. The earlier EV and
SUPG variants were removed; the solver is a command-line override if you want to
compare against them.

| config | Rayleigh | viscosity | initial condition |
|---|---|---|---|
| `config_A3_mmoc.toml` | 7e3 | FK rmu 20 | Y_3^2 |
| `config_C1_mmoc.toml` | 1e5 | isoviscous | Y_4^0 + 5/7 Y_4^4 |
| `config_C3_mmoc.toml` | 1e5 | FK rmu 30 | Y_4^0 + 5/7 Y_4^4 |
| `config_C4_mmoc.toml` | 1e6 | FK rmu 30 | Y_4^0 + 5/7 Y_4^4 |
| `config_C5_mmoc.toml` | 1e7 | FK rmu 30 | Y_4^0 + 5/7 Y_4^4 |

All five take dimensional inputs, since the nondimensional keys no longer bind,
and all five parse clean against the current app with no unbound keys. The
Rayleigh number is set through `reference-viscosity`; the shell 1.22..2.22 comes
from the two radii. Pass mesh and subdomain levels on the command line.

## Production

One config, `config_MT1024_C5dim_Raeff1e7_mmoc.toml`, the last MT1024 run:
compressible TALA with MMOC, Frank-Kamenetskii rmu 100 on the Lin et al. 2022
radial profile, free slip both boundaries, broadband initial condition over
degrees 8 to 96. The four earlier MT256 and MT512 configs were removed.

It is a port, not the original file. Six keys of the pre-merge config no longer
bind and were translated, and the viscosity profile is now read from the
dimensional CSV columns rather than the pre-normalised ones. A missing column
yields a profile of zeros with no error, so do not change those two key names.
The config parses clean against the current app with no unbound keys, but the
effective Rayleigh number has not been re-verified by a run.

## Scaling sweep

Each point runs **10 timesteps** (indices 0..9) with `--output-frequency 9`, which
writes exactly one `timer_trees/timer_tree_9.json`. Per-step wall time is that
file's `timestep` node, `root_time / count`. Note `sum_time` is the sum **across
ranks** and `avg_time` the rank mean, so neither should be divided by `count`
to get a per-step time for a single rank.

Solver settings are fixed so every point does identical work: Stokes 10 FGMRES
iterations with restart 10, energy 50 FGMRES iterations, both with relative and
absolute tolerances pinned to 0 so the iteration count is never cut short.
Low-memory variants set both restarts to 5 and hold the Krylov basis in single
precision.

`scaling/sng2/` and `scaling/lumi/` are the scripts as they ran for the paper.
The `_menv` sng2 variants use a different MPI/offload environment; see below.
Those scripts name `config_fscmb_nsurf_lvl6_10steps.toml`, which is no longer
here: it is a hard parse failure on the current app, because the viscosity law
it names was removed from the law table. `config_scal_A3.toml` replaces it and
describes the same case.

### Environment matters more than expected

The sng2 scripts set `PSM3_GPUDIRECT=0` and deliberately set **neither**
`I_MPI_OFFLOAD_IPC` nor `FI_MR_CACHE_MAX_COUNT` / `PSM3_MR_CACHE_SIZE`. Using the
environment from the production mantle-circulation run scripts instead costs
13-79 % per timestep. Single-rank points are unaffected and the penalty grows
with ranks per node, which points at intra-node GPU peer-to-peer transfers.
Keep the environment in these scripts when extending the sweep.

### Reproduction on the current app

`scaling/sng2_reproduction/` re-runs every sng2 point on the merged
MMOC/compressible app. Three keys of the original config no longer bind and were
translated in `config_scal_A3.toml`, which is the only sweep config kept here
because it is the only one the current app can run:

| original                          | now                                        |
|-----------------------------------|--------------------------------------------|
| `viscosity-law="frank-kamenetskii"` | `"fk-benchmark"` (same law)              |
| `initial-temperature-sph-epsilon` | `perturbation-amplitude`                   |
| `radius-min/max`, `diffusivity`, `rayleigh-number` | unbound; supplied dimensionally on the CLI |

Two traps worth knowing. `t-end` is interpreted in **Ma** by the current app but
was nondimensional time before, so the original `t-end=1.0` stops the run after
three steps and writes no timer tree; let `--max-timesteps` govern instead. And
the A3 reference viscosity sits above the default viscosity clamp, so pass
`--viscosity-min 1e18 --viscosity-max 1e28` or the law is silently cut.

The 34 scripts there are the sweep as run on 2026-09-22: every published sng2
point, 50 energy iterations, and the original environment restored. They take
their config from `../config_scal_A3.toml` by default, and `TERRANG_BIN`,
`TERRANG_CFG` and `TERRANG_OUT` override the binary, config and output paths so
they run from a checkout rather than only from the original scratch tree.

Agreement with the published sng2 numbers over 32 of 34 points: median +2.3 %,
and within about 1 % for every point costing 3 s/step or more. Points below
3 s/step run ~5 % heavy, which is fixed per-step overhead that does not scale down.

## Degrees of freedom

The app reports `(T, u, p)` separately. The paper's per-level DoF totals are
`T + u + p`, not the velocity-pressure Stokes system alone.

## Two caveats when comparing machines

**The LUMI scripts do not write a usable timer tree.** 114 of the 123 pass
`--output-frequency 11` alongside `--max-timesteps 10`, and since output is
written when `timestep % frequency == 0`, nothing fires after step 0. Only 9 use
frequency 9 or 4. So LUMI per-step times come from the `### Timestep` log
timestamps, while sng2 times come from `timer_tree_9.json`. The two are close
but not the same measurement, and the discrepancy is worth keeping in mind when
reading a cross-vendor plot.

**The launch scripts keep their original absolute paths.** They are the record of
what actually ran, so the LUMI ones still point into `/pfs/lustrep3/...` and the
sng2 ones into the scratch tree. Adapt paths before reusing rather than expecting
them to run as-is from a checkout.
