# Verification benchmarks

**Figure produced:** `mc_validation_profiles.pdf` — steady-state radial
temperature profiles and `T = 0.5` isosurfaces — and the steady-state Nusselt
numbers in the verification table.

The ten cases of the Zhong et al. (2008) spherical-shell suite, as compiled by
the Euen et al. (2023) intercomparison. One config per case, all using MMOC for
the energy equation at Courant number 2.7, MT256, free slip at both boundaries,
isothermal Dirichlet temperatures.

| config | Rayleigh | viscosity contrast | perturbation |
|---|---|---|---|
| `config_A1_mmoc.toml`     | 7e3 | 1 (isoviscous) | Y_3^2 |
| `config_A3_mmoc.toml`     | 7e3 | 20             | Y_3^2 |
| `config_A4_mmoc.toml`     | 7e3 | 100            | Y_3^2 |
| `config_A5_mmoc.toml`     | 7e3 | 1000           | Y_3^2 |
| `config_A6_mmoc.toml`     | 7e3 | 1e4            | Y_3^2 |
| `config_A7_mmoc.toml`     | 7e3 | 1e5            | Y_3^2 |
| `config_C1_mmoc.toml`     | 1e5 | 1 (isoviscous) | Y_4^0 + 5/7 Y_4^4 |
| `config_C3_mmoc.toml`     | 1e5 | 30             | Y_4^0 + 5/7 Y_4^4 |
| `config_C4_mmoc.toml`     | 1e5 | 100            | Y_4^0 + 5/7 Y_4^4 |
| `config_C1star_mmoc.toml` | 1e7 | 1 (isoviscous) | Y_4^0 + 5/7 Y_4^4 |

## Running

Mesh and subdomain levels are not in the config; pass them on the command line.
MT256 is mesh level 8. Example, 8 nodes x 8 ranks:

```
srun mantlecirculation --config config_C3_mmoc.toml --extended-parameters \
  --refinement-level-mesh-min 3 --refinement-level-mesh-max 8 \
  --lat-sdr 3 --rad-sdr 0 --radial-extra-levels -1 \
  --outdir <outdir> --outdir-overwrite
```

Each case runs until the radially averaged temperature profile and the surface
Nusselt number stop changing; the configs set no step limit. Progress is written
to `nu.csv` (timestep, simulated time, Nu_top, V_rms) and radial profiles to
`radial_profiles/`. Treat a case as settled only when Nu is flat to two decimals
over several consecutive output intervals — these runs plateau, resume drifting,
and reverse, so a single flat interval means nothing.

Validated: C3 reproduces the reference run to every printed digit and settles at
Nu 6.698, inside the published interval 6.500–6.790.

## Caveats

`viscosity-rmu` is the top-to-bottom contrast directly, since the law is
`eta(T) = rmu^(1/2 - T)`. The Rayleigh number is set through
`reference-viscosity`, 1.721988e24 at Ra = 1e5, scaling inversely with Ra; the
shell 1.22–2.22 comes from the two radii. Dimensional inputs are used because
the nondimensional keys (`radius-min/max`, `diffusivity`, `rayleigh-number`) no
longer bind.

Contrasts for A3, C1 and C3 are confirmed by runs reproducing the published
Nusselt numbers, and C4 follows by elimination. **The contrasts for A4–A7 are
inferred**: the paper gives the A-series set {1, 20, 100, 1e3, 1e4, 1e5} but not
the per-case assignment, and no configs or logs survive for those runs. The
ordering is supported by the archived radial profiles, whose interior mean
temperature rises monotonically 0.19, 0.27, 0.35, 0.41, 0.65 across A1, A4, A5,
A6, A7 — the stagnant-lid signature of increasing contrast. Confirm against the
original benchmark paper before publishing.

A7 is very slow to converge: at contrast 1e5 the interior is nearly rigid. A
24 h run on 8 nodes reached step 1500 of the ~20000 needed, with Nu still
accelerating upward from the conductive initial condition. Expect to chain
checkpoint continuations, or start from a better initial state.
