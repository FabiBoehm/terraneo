# Production run

**Figures produced:** the end-to-end simulation figures — `e2e_diag2x2.pdf`,
`e2e_crosscut.png`, `e2e_iso_tdev400.png`.

One config, `config_MT1024_C5dim_Raeff1e7_mmoc.toml`: the MT1024 end-to-end
mantle-circulation model. TALA-compressible with MMOC, Frank-Kamenetskii
temperature dependence of contrast 100 applied to the Lin et al. (2022) radial
viscosity profile, free slip at both boundaries, 4200 K at the core-mantle
boundary and 300 K at the surface, broadband initial condition over spherical
harmonic degrees 8 to 96.

## Running

64 nodes, 512 ranks. Mesh and subdomain levels come from the command line:

```
srun mantlecirculation --config config_MT1024_C5dim_Raeff1e7_mmoc.toml \
  --extended-parameters \
  --refinement-level-mesh-min 4 --refinement-level-mesh-max 10 \
  --radial-extra-levels -1 --lat-sdr 4 --rad-sdr 0 \
  --outdir <outdir> --outdir-overwrite
```

To continue from a checkpoint, add `--load-checkpoint 1 --checkpoint-dir
<outdir>/xdmf --checkpoint-step <step>`. Note that velocity is re-solved from a
cold start on restart while temperature is restored exactly, so V_rms jumps on
the first interval and relaxes back over a few more; Nu is unaffected.

## Caveats

This is a port of the pre-merge config, not the original file. Six keys no
longer bind and were translated: the nondimensional-numbers switch and the
density-weighted TALA flag are gone, `diffusivity` is unbound, the viscosity law
name changed, the perturbation key was renamed, and the profile loader keys were
restructured. The config parses clean against the current app with no unbound
keys, but **the effective Rayleigh number has not been re-verified by a run.**

The viscosity profile is now read from the **dimensional** CSV columns
(`Radius (m)` / `Viscosity (Pa s)`), not the pre-normalised ones the old config
asked for. A missing column yields a profile of zeros with no error, so do not
change those two key names. Because of this the relationship between the
reference viscosity and the effective Rayleigh number may differ from the
original run, which printed a nominal Ra of 4.49e8 (the "Raeff1e7" in the run
name is the effective value after the profile raises the mean viscosity). Check
the printed Rayleigh number on the first run before treating this as a
continuation of the old one.
