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
  sng2_reproduction/                     SuperMUC-NG Phase 2, current app
  lumi/                                  LUMI-G launch scripts, as they ran
  lumi_reproduction/                     LUMI-G, current app (generate.sh + feed_lumi.sh)
  submit/                                sweep generators and collector
```

## Benchmarks

The ten cases of the paper's Nusselt table, from the Zhong et al. (2008)
spherical-shell suite. One config each, all MMOC at Courant number 2.7, MT256,
free slip at both boundaries, isothermal Dirichlet temperatures. The earlier
EV and SUPG variants were removed; the solver is a command-line override if you
want to compare against them.

| config | Rayleigh | contrast | perturbation |
|---|---|---|---|
| `config_A1_mmoc.toml` | 7e3 | 1 | Y_3^2 |
| `config_A3_mmoc.toml` | 7e3 | 20 | Y_3^2 |
| `config_A4_mmoc.toml` | 7e3 | 100 | Y_3^2 |
| `config_A5_mmoc.toml` | 7e3 | 1000 | Y_3^2 |
| `config_A6_mmoc.toml` | 7e3 | 1e4 | Y_3^2 |
| `config_A7_mmoc.toml` | 7e3 | 1e5 | Y_3^2 |
| `config_C1_mmoc.toml` | 1e5 | 1 | Y_4^0 + 5/7 Y_4^4 |
| `config_C3_mmoc.toml` | 1e5 | 30 | Y_4^0 + 5/7 Y_4^4 |
| `config_C4_mmoc.toml` | 1e5 | 100 | Y_4^0 + 5/7 Y_4^4 |
| `config_C1star_mmoc.toml` | 1e7 | 1 | Y_4^0 + 5/7 Y_4^4 |

`viscosity-rmu` is the top-to-bottom contrast directly, since the law is
eta(T) = rmu^(1/2 - T). The Rayleigh number is set through `reference-viscosity`,
which is 1.721988e24 at Ra = 1e5 and scales inversely with Ra; the shell
1.22..2.22 comes from the two radii. Picard iterations are 1 for the isoviscous
cases and 2 where viscosity depends on temperature. All ten parse clean against
the current app with no unbound keys.

**One caveat.** A3, C1 and C3 have contrasts confirmed by runs that reproduced
the published Nusselt numbers, and C4 follows by elimination from the paper's
C-series set {1, 30, 100}. The contrasts for A4, A5, A6 and A7 are inferred: the
paper gives the A-series set {1, 20, 100, 1e3, 1e4, 1e5} but not the per-case
assignment, and no configs or logs survive for those runs. The order is
supported by the archived radial profiles, whose interior mean temperature rises
monotonically 0.19, 0.27, 0.35, 0.41, 0.65 across A1, A4, A5, A6, A7, which is
the stagnant-lid signature of increasing contrast. Confirm against the original
benchmark paper before publishing.

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

`scaling/lumi/` holds the LUMI-G scripts as they ran for the paper. They name
`config_fscmb_nsurf_lvl6_10steps.toml`, which is not in this tree: it is a hard
parse failure on the current app, because the viscosity law it names was removed
from the law table. They also carry LUMI absolute paths. Treat them as a record
of what ran, not as something to launch.

The original sng2 launch scripts were removed for the same reason.
`scaling/sng2_reproduction/` supersedes them: same points, same environment,
running against the current app with `config_scal_A3.toml`.

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

**The archived LUMI scripts would not write a usable timer tree on the current
app.** 114 of the 123 pass `--output-frequency 11` alongside `--max-timesteps 10`,
and output fires only when the step index divides evenly, so nothing lands after
step 0. Yet the published LUMI table marks 82 of its 88 points as tree-derived
(6 as log-derived), and a `.pre-trees` version of the same table exists, so the
trees were obtained after the archived scripts, by a run or an app version not
in this tree. The reproduction in `scaling/lumi_reproduction/` sidesteps the
question by using `--output-frequency 9`, which writes exactly one
`timer_tree_9.json` per point, the same convention as the sng2 reproduction.

**The launch scripts keep their original absolute paths.** They are the record of
what actually ran, so the LUMI ones still point into `/pfs/lustrep3/...` and the
sng2 ones into the scratch tree. Adapt paths before reusing rather than expecting
them to run as-is from a checkout.

## Building on LUMI-G

The one non-obvious ingredient is Cray's GPU transport layer. Without it the
binary links fine but MPI aborts at startup with "GPU_SUPPORT_ENABLED is
requested, but GTL library is not linked" as soon as `MPICH_GPU_SUPPORT_ENABLED=1`
is set, which every job script here sets. With the default LUMI/25.03,
PrgEnv-amd, rocm/6.3.4 and craype-accel-amd-gfx90a modules:

```
cmake <source> -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.4/bin/hipcc \
  -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ROCTHRUST=ON -DKokkos_ENABLE_HWLOC=OFF \
  -DMPI_CXX_LINK_FLAGS="-Wl,--whole-archive,-lhugetlbfs,--no-whole-archive" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cray/pe/mpich/8.1.32/gtl/lib -lmpi_gtl_hsa -Wl,-rpath,/opt/cray/pe/mpich/8.1.32/gtl/lib"
make -j16 mantlecirculation
```

Check with `ldd mantlecirculation | grep gtl` before submitting anything.

## Convergence studies (hourglass, viscosity, precision)

These three figures do not come from the mantle-circulation app but from test
drivers, and none of the three was tracked in this repository. Status as of
2026-09-23:

**Hourglass control — reproduced exactly.** Driver
`tests/test_epsilon_divdiv_quadrature_matrix.cpp`, which was never committed to
any branch; it survives only as an untracked file and inside a git stash on the
Helma clone. It also needs `matrix_rhs_s{1,2,3}.inc`, the operator headers
`epsilon_divdiv_simple_2pt.hpp` and `epsilon_divdiv_simple_stab.hpp`, and a
version of `epsilon_divdiv_kerngen.hpp` that still carries a fourth template
parameter (radial quadrature points) and reads the stabilisation strength from
the environment. The current kerngen has neither.

Invocation is by environment variable, one level per run:

| series | command |
|---|---|
| `1pt`  | `RUN_SCALE=<level> ./test_epsilon_divdiv_quadrature_matrix` |
| `1ptK` | `RUN_SCALE=<level> STAB_C_KERNGEN=0.3 ...` |
| `2ptK` | `RUN_SCALE=<level> RUN_QP=2 ...` |

**`STAB_C_KERNGEN=0.3` is the published stabilisation constant.** It is recorded
nowhere else, and it matters: 0.1 gives errors 19 % above the published values
and 1.0 gives them 30 % below. At 0.3 the rerun reproduces all four levels to
eleven significant figures (level 5 2.7708062226e-04, level 6 6.9729667964e-05,
level 7 1.7505508443e-05, level 8 4.3880244792e-06). The `1pt` and `2ptK` series
matched immediately without any tuning. Level 9 is in the published figure and
has not been rerun; it needs more than one node.

**Viscosity convergence — driver not working.** `tests/qp_comparison_test.cpp`
on the `amr` branch of the Helma clone calls `set_single_quadpoint` and a
`HexPortable` kernel path that neither kerngen here possesses, so it targets a
third kerngen version. Its name also suggests quadrature rather than viscosity
profiles, so it may be the wrong file; the stronger candidate is the manufactured
solution referenced by `tests/mms_visclocal_gen.py`, not yet located.

**Precision study — producer not found.** The data
(`data/precision_study_complete.csv`, MT8 to MT4096, double vs single L2 errors)
exists only in the paper repository. No source anywhere in the local clones or in
1020 commits of the Helma clone emits those column names.
