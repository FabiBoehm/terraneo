# Stokes convergence on published viscosity profiles

Produces the two tables printed beside `visc_profiles.png`: preconditioned
FGMRES iterations to a relative residual of 1e-6, and the relative residual
reached after a fixed budget of 10 iterations.

The driver is `tests/test_epsilon_divdiv_ablock_mg_gca.cpp`, built as
`test_epsilon_divdiv_ablock_mg_gca`. It solves the Stokes saddle point on the
shell with the same preconditioner as the app: a matrix-free geometric
multigrid V-cycle with Chebyshev smoothing on the velocity block, and the
inverse lumped diagonal of the 1/eta-weighted pressure mass matrix as the Schur
approximation.

| profile | flag | viscosity range |
|---|---|---|
| Lin et al. (2022)   | `--visc-profile 3` | 4.7e20 - 5e23 Pa s |
| Stotz et al. (2017) | `--visc-profile 2` | 5.8e19 - 7e23 Pa s |

Both are read from `data/radialprofiles/ViscosityProfile_*.csv` in this
repository, columns `radius_normalized_1p22_2p22` and
`viscosity_scaled_by_min`. The driver looks for them relative to the working
directory; `TERRANG_PROFILE_DIR` overrides the location.

## Running

`run_convergence.sh` takes `TERRANG_BIN` (required), `TERRANG_MT`
(32 ... 2048) and `TERRANG_PROFILE` (`lin` or `stotz`), plus the optional
`TERRANG_MAX_CYCLES`, `TERRANG_OUT`, `TERRANG_PROFILE_DIR` and
`TERRANG_ACCOUNT`. `TERRANG_MT` selects the fine level (MT32 is level 5,
MT2048 is level 11); the subdomain refinement follows from the rank count, so
node and task counts go on the `sbatch` line:

```
TERRANG_BIN=<build>/tests/test_epsilon_divdiv_ablock_mg_gca \
TERRANG_MT=256 TERRANG_PROFILE=lin TERRANG_ACCOUNT=<project> \
  sbatch --nodes=1 --ntasks-per-node=8 run_convergence.sh
```

`TERRANG_MAX_CYCLES=100` (the default) gives the left table, the iteration
count in the `cycles` column of the final summary. `TERRANG_MAX_CYCLES=10`
gives the right table, `final_rel_res` in the same summary.

| MT | level | ranks |
|---|---|---|
| MT32   | 5  | 1 |
| MT64   | 6  | 1 |
| MT128  | 7  | 10 |
| MT256  | 8  | 10 |
| MT512  | 9  | 40 |
| MT1024 | 10 | 160 |
| MT2048 | 11 | 160 |

The rank counts are memory-driven: the outer FGMRES keeps up to 50 Krylov
vectors of the full velocity-pressure system. Only rank counts that divide the
subdomain count are legal, since the grid holds 10 * 4^lat_sdr * 2^rad_sdr
subdomains; the script rejects anything else.
