# Production run

Produces `e2e_diag2x2.pdf`, `e2e_crosscut.png` and `e2e_iso_tdev400.png`.

`config_MT1024_C5dim_Raeff1e7_mmoc.toml` is the MT1024 end-to-end
mantle-circulation model: TALA-compressible with MMOC, Frank-Kamenetskii
contrast 100 on the Lin et al. (2022) radial viscosity profile, free slip at
both boundaries, 4200 K at the core-mantle boundary and 300 K at the surface,
broadband initial condition over spherical-harmonic degrees 8 to 96.

## Running

64 nodes, 512 ranks:

```
srun mantlecirculation --config config_MT1024_C5dim_Raeff1e7_mmoc.toml \
  --extended-parameters \
  --refinement-level-mesh-min 4 --refinement-level-mesh-max 10 \
  --radial-extra-levels -1 --lat-sdr 4 --rad-sdr 0 \
  --outdir <outdir> --outdir-overwrite
```

To continue from a checkpoint add `--load-checkpoint 1 --checkpoint-dir
<outdir>/xdmf --checkpoint-step <step>`.

The viscosity profile is read from the dimensional CSV columns, `Radius (m)`
and `Viscosity (Pa s)`. A missing column yields a profile of zeros with no
error, so leave those two key names alone.
