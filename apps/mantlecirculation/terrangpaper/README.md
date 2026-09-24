# TERRA-NG model description paper: inputs

Configurations and run scripts for the figures and tables of the TERRA-NG
model description paper. Each subdirectory has its own README naming the
figure it produces and the command that runs it.

| directory | produces |
|---|---|
| `benchmarks/` | `mc_validation_profiles.pdf` and the steady-state Nusselt table: ten cases of the Zhong et al. (2008) spherical-shell suite |
| `convergence/` | the two FGMRES iteration and residual tables for the Lin (2022) and Stotz (2017) viscosity profiles |
| `hourglass/` | `hourglass_stab_convergence.pdf`: multigrid residual and discretization error with and without hourglass control |
| `production/` | `e2e_diag2x2.pdf`, `e2e_crosscut.png` and `e2e_iso_tdev400.png`: the MT1024 end-to-end run |
| `scaling/` | `cross_vendor_strong_scaling.png`: the strong-scaling sweep on SuperMUC-NG Phase 2 and LUMI-G |

`benchmarks/`, `production/` and `scaling/` drive the `mantlecirculation` app
and are configured through TOML files plus command-line refinement levels.
`convergence/` and `hourglass/` drive stand-alone solver tests from `tests/`.

All scripts take their paths from the environment: `TERRANG_BIN` points at the
binary, `TERRANG_ACCOUNT` at the batch account, and `TERRANG_OUT` at the run
directory. Nothing is hard-coded to a machine.
