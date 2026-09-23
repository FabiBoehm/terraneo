# terra-ml

*Draft: a learned initial guess for the TERRA-NG Stokes solver. First results are
promising; this has not been through a review and is ongoing work.*

A neural solution operator for variable-viscosity Stokes flow on the TERRA-NG
spherical shell,

    N_theta : (f_u, f_p, log eta)  ->  (u, p)     approximating  K_eta^-1 ,

together with the manufactured-solution data it trains on, the app-side hook that
calls it, and a bench mode in the production solver that measures what it is worth.

- `terra_data` — manufactured Stokes problems: random polynomial solutions with the
  right-hand sides derived symbolically against *the same* deviatoric operator TERRA
  implements, verified to reproduce the code's own analytic test cases.
- `terra_infer` — the operator (`operator.py`), its trainer (`train_linear_mr.py`),
  the spherical/Chebyshev transforms, the finite-difference Stokes residual, the mesh
  symmetry group, and the glue (`__init__.py`) the embedded interpreter calls.
- `scripts` — held-out evaluation, solver-bench problem generation, cross-section plots.

## Install

    pip install -e python/

On SNG-2 the working stack is `module load python/3.10.12-extended` plus the
`torch 2.7.1+xpu` virtualenv on scratch; the newer XPU wheels see no devices there.

## What we are trying to do

Every time step of a mantle convection run solves the Stokes system

    K_eta x = f,        x = (u, p),   f = (f_u, f_p),

for the velocity and pressure that balance the buoyancy `f_u` under the current
viscosity field `eta`. It is by far the most expensive part of a step: the operator
changes every step because `eta` does, and at high viscosity contrast the iterative
solver needs tens of iterations of 100–200 ms each. The solver starts from zero and is
told nothing about the previous solutions, the structure of `eta`, or what Stokes
solutions on a shell look like.

A **neural operator** is a network that learns a map between *functions* rather than
between vectors of fixed length: here the map from the pair (forcing, viscosity field)
to the solution field. It is trained once on many solved problems and then, given a
new `f` and `eta`, produces an approximate solution in a single forward pass — no
iteration. The important distinction from an ordinary image-to-image network is that
the learned object should be the *operator* `K_eta^-1`, not a lookup on one grid:
the same weights should be usable at any mesh resolution, and the output should
respond to `f` and `eta` the way the true solution operator does. Our target is a
network that is (i) exactly linear in the forcing, like the true inverse, (ii) as
mesh-independent as we can make it, and (iii) accurate enough that the solver, when it
starts from the network's output instead of from zero, needs far fewer iterations.

What we do **not** try to do is replace the solver. The prediction is used as the
initial guess of the production multigrid-preconditioned FGMRES solver; the solver
still produces and certifies the answer. A poor prediction costs iterations, never
correctness, and the required accuracy of the network is whatever saves the most
solver time (10% turns out to be very valuable).

## The operator, in simple terms

The network takes five fields on the shell mesh — the three components of `f_u`,
`f_p`, and `log eta` — and returns four — `u` and `p`. It is built from three ideas.

**1. Keep the linearity of the true inverse.** For fixed viscosity the discrete
system is linear in `f`, and all the difficulty is in how `K_eta^-1` depends on `eta`.
The network has the same structure: the forcing passes only through bias-free linear
layers, and the viscosity acts on them from the side — it scales features, it selects
which stencil is used at each node, and it decides the entries of the spectral
matrices — but is never added as a channel. Doubling `f` exactly doubles the output,
and `f = 0` gives exactly zero, at any resolution. This removes a whole family of
things the network would otherwise have to learn and lets the same weights transfer
across meshes, because linear maps are what the transforms below preserve.

**2. A global part in spectral space.** The inverse of Stokes is a *global* operator:
a load anywhere moves fluid everywhere. Writing the fields in spherical harmonics on
the sphere and Chebyshev polynomials in radius turns the shell into a small set of
modes (a few hundred to a thousand), and for radially layered viscosity the true
inverse is exactly a matrix per harmonic degree acting on the radial modes — a
Green's function. The network has such a matrix for every degree, but instead of
storing them it *generates* them with a small network fed with the degree, the radial
mode indices and a short summary of the viscosity profile. This is what makes the
global part mesh-independent: a finer mesh only means more modes are asked of the
generator, and what it learned about the operator carries over. It is also what
makes it a neural operator in the FNO sense, with the Fourier basis replaced by the
natural basis of the shell.

**3. A local part on the mesh.** Sharp viscosity contrasts are local — a slab, a
channel, a plume boundary — and the spectral truncation cannot resolve them. Four
convolutional layers on the mesh's `n x n x n` diamond grids handle this, with one
twist: the stencil used at a node is a viscosity-dependent mixture of a few learned
stencils, chosen from `eta` at that node and its neighbourhood. This branch holds most
of the weights and almost all of the compute, and it is the reason the network is
not yet fully mesh-independent (next section).

Around this: a projection of the output onto the finite-element space (nodes shared
between diamonds get the mean of their copies — essential, because the solver cannot
correct any error that lies outside its own space); a learned *defect correction*
step, which applies the network a second time to the residual of its own answer;
and a **router** that picks one of four expert copies by the viscosity contrast of the
problem, since problems with contrast 2 and contrast 10,000 want different stencils.

### The same in shapes

`B` batch, `S = 10` diamonds, `N = n^3` nodes per diamond, `C = 128` channels,
`M = (l_max+1)^2` harmonics, `k_1 = k_max+1` radial modes, `n_b = 8` blocks of
`b_s = 16` channels, `tok = b_s k_1`.

| level | n | nodes | l_max | M | k_1 | tok |
|---|---|---|---|---|---|---|
| L3 | 9 | 7,290 | 12 | 169 | 9 | 144 |
| L4 | 17 | 49,130 | 24 | 625 | 17 | 272 |
| L5 | 33 | 359,370 | 32 | 1089 | 17 | 272 |
| L6 | 65 | 2,746,250 | 32 | 1089 | 17 | 272 |

- *Lift and gate.* `(f_u, f_p)` → bias-free `Linear(4 -> 128)`; multiplied node by node
  by `gate_in(geometry ⊕ log eta)`, a `5 -> 128 -> 128` MLP.
- *Spectral Green core* (fp32). Reshape to `(B, C, S n^2, n)`; harmonic analysis
  `A (M x S n^2)` (pseudo-inverse of the basis, so seam nodes are not double-counted),
  Chebyshev analysis `A_r^T (k_1 x n)`; regroup to `(B, n_b, M, tok)`. Per degree `l`
  and block one `tok x tok` matrix, shared across the `2l+1` orders (rotational
  symmetry), generated by `ggen: (l, k, k') ⊕ eta-embedding(16) = 19 -> 64 -> 64 -> 2048`.
  The two high-contrast experts add a cross-degree attention keyed on `eta`. Synthesis
  with `Y_r`, `Y`; residual add.
- *Local branch.* Reshape to `(S B, C, n, n, n)`; four residual layers (eight in the
  top expert) of a bias-free
  `5^3` conv `128 -> 128` plus a bank: `1^3` `128 -> 32`, `K = 4` kernels
  `(32, 32, 5, 5, 5)` mixed by a `3 -> 32 -> K` softmax of `(log eta, local mean,
  local std)`, `1^3` back to `128`; GELU on the residual. 98% of the parameters,
  45 TFLOP per forward at L6 (the spectral core: 0.001).
- *Head.* `gate_out` (as `gate_in`) then `Linear(128 -> 4)`; seam-mean projection;
  the caller zeroes velocity on the Dirichlet shells.
- *Defect correction.* `x = N(f) + gamma N(f - K_eta N(f))`, one learned scalar,
  still exactly linear in `f`; doubles the cost, ~10% less error.
- *Experts.* Routed on `max eta / min eta` at thresholds `10, 10^2, 10^3`:

| expert | contrast band | stencil banks | bank width | attention | params |
|---|---|---|---|---|---|
| generalist | < 10 | – | – | no | 8.44 M |
| mid | 10 – 10^2 | 4 | 128 | no | 41.21 M |
| high | 10^2 – 10^3 | 4 | 32 | yes | 10.50 M |
| top | > 10^3 | 4 | 32 | yes | 20.81 M |

The top expert is the exception to "four convolutions": it has **eight** local layers
(`--linear-convs 8`, hence its name `c8ckpt`), which doubles its stencil reach to 16
nodes. 80.96 M parameters in the four experts together.

### What is and is not discretisation invariant

The spectral core is mesh-independent by construction: its weights are indexed by
channel, degree and radial mode, its matrices come from a generator over normalised
indices, and refining the mesh only rebuilds the fixed transform matrices
(`set_mesh`). The two experts that serve contrasts below `10^2` were trained on L3–L5
only and have never seen an L6 field; two thirds of the L6 test set is handled by them,
so the L6 column below is predominantly a zero-shot result.

The **local branch is not** invariant. Its stencils live in index space: four layers of
`5^3` reach eight nodes (16 in the eight-layer top expert), which is the whole shell at
L3 and an eighth of it at L6. And
the truncation is held at `l_max = 32` above L4 while the mesh grows, so the harmonics
represent 21% of the lateral degrees of freedom at L3–L4, 10% at L5 and 2.6% at L6. Both
point the same way as the measured error growth with level. The trainer exposes
`--dilated-stencils` (dilation `2^(L-3)`, fixed physical footprint, no new weights) and
`--sep-stencils` (depthwise + `1^3`, ~9x cheaper stencil); their effect is under test and
is *not* in the deployed experts.

## Pipeline

### 1. Generate a dataset

Dump the mesh, then build the samples. Both are per level; generation shards trivially.

    mpirun -np 1 ./stokes_dataset_tool --max-level 5 --min-level 2 --outdir $DIR
    python -m terra_data.generate --dir $DIR --num-train 1000 --num-test 200 \
        --max-degree 8 --contrast-min 1 --contrast-max 1e4 --seed 42 \
        --shard $i --num-shards 16

Each sample draws a degree `d <= 8`; the velocity is a random degree-`d` polynomial
times the radial bubble `(r^2 - r_min^2)(r_max^2 - r^2)`, so the Dirichlet conditions
hold exactly and everything stays polynomial. The viscosity is
`eta = 1 + (c - 1) (q / max|q|)^2` with `q` a random polynomial and `c` log-uniform in
`[1, 1e4]`; the right-hand sides follow symbolically. Pressure is rescaled per sample so
that `||grad p|| / ||A_eta u||` is log-uniform in `[0.5, 2]` — drawn independently the
two momentum terms differ by tens, the pressure sinks below the discretisation floor of
`f_u`, and a network correctly learns to predict zero pressure.

Samples are seeded from `(seed, split, index)`, so a training and a test sample can never
be the same draw even across datasets sharing a seed; this was verified by hashing every
file (zero overlap between the held-out sets and ~96k training files). The held-out split
also selects the best checkpoint, so it is validation data in the strict sense.

Datasets are large (an L6 sample is 97 MB) and regenerable — put them on scratch. Give
each expert an additional set restricted to its contrast band (`--contrast-min/max`).

To check the generator against TERRA itself:

    python -c "from terra_data.stokes_symbolic import validate_against_terra_testcase as v; v()"
    python -c "from terra_infer.stokes_residual import validate; validate()"

### 2. Train

The recipe behind the deployed top-band expert (warm-started from its four-layer
predecessor `egmrhcS`, with the local branch deepened to eight layers):

    python -m terra_infer.train_linear_mr \
        --data $ML/stokes_L3_hc --data2 $ML/stokes_L4_hc --data3 $ML/stokes_L5_hcall \
        --extra-data $ML/stokes_L5_hc3:32:16:2:2000 \
        --extra-data $ML/stokes_L6_hc:32:16:1:60 \
        --extra-data $ML/stokes_L6_hc2:32:16:1:200 \
        --lmax 12 --kmax 8 --lmax2 24 --kmax2 16 --lmax3 32 --kmax3 16 \
        --batch-size 8 --batch-size2 4 --batch-size3 2 --batch-mix \
        --max-train 8000 --max-train3 1600 --max-test 32 \
        --epochs 14 --lr 2e-4 --amp --grad-checkpoint \
        --hidden 128 --heads 8 --linear-convs 8 --linear-kernel 5 --linear-depth-gates \
        --eta-gates --eta-green --eta-stencils 4 --bank-bottleneck 32 --mode-attn 32 \
        --h1-weight 0.5 --seam-average --epoch-frac 1,1,0.5,0.5,1,0.5 \
        --init-from $ML/w3d_linear_egmrhcS.pt --device xpu --out $ML/w3d_linear_new.pt

`--extra-data path:lmax:kmax:batch[:max_train]` adds a dataset; `--epoch-frac` gives the
fraction of *each* dataset visited per epoch (one value per dataset, in order).
`--batch-mix` interleaves shuffled batches from all datasets so the weights never see a
per-epoch level seesaw. The loss is relative L2 on velocity and mean-free pressure plus an
`H^1` gradient term (`--h1-weight`), restricted to nodes where the finite-difference
stencil is the true operator; velocity targets are scaled by the sample's geometric-mean
viscosity. `--refine-steps 1 --refine-gate 0.4` adds the defect correction. Augmentation
is over the mesh's exact symmetry group.

Practicalities that matter on one PVC tile:

- The assembled cache (`_asm_*` next to each dataset) is read into RAM when it fits
  (`TERRA_CACHE_RAM_GB`, default 96); memory-mapped from the parallel filesystem the
  per-batch gather stalls on page faults and an epoch that takes 260 s staged does not
  finish in 20 min.
- Epoch cost is dominated by the local branch: ~39 min for the recipe above, ~5 min for
  the same recipe with `--linear-convs 0`.
- `WORLD_SIZE=8 RANK=<r>` (one process per tile, `ZE_AFFINITY_MASK=<r>`) averages
  gradients through gloo on the host: 4.2x faster epochs, but the effective batch is 8x
  larger and the learning rate must be rescaled.
- Learning rate: the spectral-only model has its optimum near `6.4e-3` (16x the
  historical `4e-4`); the full model with convolutions is **unstable** at `3.2e-3`
  (loss spikes to 300). Keep `2e-4` for anything with a local branch unless a fresh
  ladder says otherwise.
- The trainer resumes from `<out>.resume` if it exists — use a fresh `--out` per run.
- The SNG-2 `test` partition shares GPUs between jobs; use `general` for any GPU work.

Every architecture switch is recorded in the checkpoint (`hidden`, `heads`,
`spherical`, `radial_modes`, `linear_convs`, `linear_kernel`, `linear_eta_gates`,
`linear_eta_green`, `linear_eta_stencils`, `linear_bank_bottleneck`,
`linear_mode_attn`, `linear_eta_embed_dim`, `linear_eta_quant`, `linear_dilated`,
`linear_sep_stencils`, `log_eta_mean`, `log_eta_std`, `test_rel_l2`), and
`load_state` tolerates keys the model no longer has.

### 3. Evaluate

Held-out error per level for the router over four experts (comma-separated
checkpoints; `EVAL_MOE_THRESH` gives the band thresholds):

    EVAL_DEVICE=xpu EVAL_MOE_THRESH=10,100,1000 EVAL_NTEST=32 EVAL_NTEST_L6=27 \
      python scripts/eval_mr_levels.py $ML/w3d_linear_egmr6g.pt,$ML/w3d_linear_egmrmc4.pt,$ML/w3d_linear_egmrxcS.pt,$ML/w3d_linear_c8ckpt.pt

reports mean / best / median / worst relative L2 for velocity and the pressure error at
each level. On the GPU with sample prefetching this takes about a minute; on CPU
25 minutes. `EVAL_LM_L6=48 EVAL_KM_L6=24` overrides the truncation at a level;
`EVAL_ONLY=L6` restricts to one.

    python scripts/plot_crosscut.py stokes_L5_d8 32 16 24 crosscut_L5.png

draws the best and worst held-out predictions as a great-circle cut through the shell
(viscosity, exact speed, predicted speed, difference).

### 4. Use it as the solver's initial guess

`scripts/make_bench_problem.py` writes held-out problems in the app's raw layout
(exact solution, one-shot prediction, and a control guess = exact solution plus a smooth
random error of the network's size). The app's bench mode then solves each from zero,
from the prediction and from the control, at every iteration budget:

    mantlecirculation --config config.toml --set-nondimensional-numbers \
        --refinement-level-mesh-min 2 --refinement-level-mesh-max 5 \
        --stokes-krylov-restart 120 --stokes-krylov-max-iterations 120 \
        --stokes-krylov-relative-tolerance 1e-14 \
        --stokes-bench-problem $W --stokes-bench-count 8 \
        --stokes-bench-its 1,2,3,4,6,8,12,16,24,32,48,64,96,120

Each `BENCH`/`BENCHEND` log line carries the arm, the budget, the residual and the
velocity error against the exact solution. For time-stepping runs, the solver's
`--stokes-guess-proj 3` (residual-optimal projection on the last three solutions) is the
production warm start; the network guess is for a cold problem.

### 5. Call it from the solver

Build with `-DTERRA_ENABLE_PYTHON=ON`; `terra::ml::NeuralSolver` (`src/terra/ml`)
ships fields zero-copy to the `cband` entry point in `terra_infer/__init__.py`
(`TERRA_NEURAL_CHECKPOINT`, `TERRA_MESH_COORDS`, `TERRA_NEURAL_DEVICE=cpu` — a second
SYCL stack inside the app process is not safe). The glue currently passes through the
`eta_green` and `linear_convs` switches only; the deployed experts also use stencil banks,
the bottleneck and attention, which the glue does not yet construct — extending
`_build_linear` is required before the router can run inside the app.

## Results

Held-out relative L2 velocity error, 27–32 samples per level, four-expert router:

| level | best | median | **mean** | worst | pressure |
|---|---|---|---|---|---|
| L3 (9^3) | 0.030 | 0.069 | **0.084** | 0.316 | 0.040 |
| L4 (17^3) | 0.027 | 0.097 | **0.112** | 0.349 | 0.030 |
| L5 (33^3) | 0.035 | 0.120 | **0.135** | 0.389 | 0.034 |
| L6 (65^3) | 0.075 | 0.156 | **0.189** | 0.564 | 0.050 |

A single un-routed model scores 0.203 / 0.245 / 0.267 / 0.312. The distribution is
skewed: medians sit well below means and a few high-contrast problems set the worst
case (the L5 worst case is a narrow low-viscosity channel cutting across the shell).

As a solver initial guess (8 held-out problems per level, FGMRES + MG/Schur, median over problems of the
exact iteration at which the error first crosses the target; in brackets the share of the
eight problems that reach it within the 120-iteration budget, the median being over those):

| target | L3 cold / warm | L4 | L5 | L6 |
|---|---|---|---|---|
| 0.2 | 18.5 (100%) / **0** | 14 (75%) / **0** | 14.5 (75%) / **0** | 12.5 (75%) / **0** |
| 0.1 | 14 (87%) / **0** | 16.5 (75%) / **6** | 16.5 (75%) / **7** | 14 (75%) / **3.5** |
| 0.05 | 15.5 (75%) / **5.5** | 18 (62%) / **10.5** | 18 (62%) / **11.5** | 15.5 (75%) / **6** |
| 0.02 | 18 (62%) / **11** | 20 (62%) / **11** (87%) | 20 (62%) / **13** (87%) | 17.5 (75%) / **9** (87%) |
| 0.01 | 20 (62%) / **15** | 22 (62%) / **13** (75%) | 22 (62%) / **14** (75%) | 20 (75%) / **13** (87%) |

Coverage is 100% where not given. The prediction is below 20% error before the solver starts, on every problem at every
level; 10% costs at most seven iterations against 14–16.5 cold; below that the saving is
about a factor of two, and the warm start converges on problems where the cold solve
does not (five of the 24 problems at L4–L6 still stand above 0.3 after the full cold
budget and end below 0.05 from the prediction). The control arm — exact solution plus a
smooth random error of exactly the network's size — reaches 1% on none of the 32
problems: the gain is not the size of the starting error but that the prediction's error
lies where the Krylov space can reach it, which is what the seam projection buys.

Two things easy to misread: FGMRES minimises the residual, so the *error* rises by an
order of magnitude over the first iterations from a cold start before it falls, and the
solver does not beat the raw prediction until about eight iterations — at 10–20%
tolerance, zero iterations is the right choice. And the prediction's starting
*residual* is not small (0.066 / 0.12 / 0.36 / 1.15 at L3–L6, above the cold start's 1
at L6): applying `K_eta` amplifies grid-scale error by `h^-2`. Tolerances must be
relative to `||b||`, never to `||r_0||`, and a residual term in the loss is the obvious
untried lever.

A forward pass costs 11 / 40 / 191 / 2000 ms at L3–L6 on one PVC tile, against
66 / 83 / 110 / 162 ms per solver iteration.

## Known limits

- Error grows with refinement for the reasons in *What is and is not discretisation
  invariant*; the L6 data is thin (3.7k training samples against 26k at L5, and 27 test
  samples).
- As a **preconditioner** inside FGMRES, every learned operator tried — five
  architectures, seven training objectives including unrolled exact-operator rollouts
  and direct spectral-radius minimisation, a dozen deployment schemes — plateaus at
  variable viscosity (residual 0.06–0.8 with error above 1). Root cause: the near-null
  modes of high-contrast Stokes have negligible residual signature, which is exactly the
  signal a preconditioner is fed. At constant viscosity a learned two-stage cycle
  reached 8.6e-7 in 53 iterations. The initial-guess role above is what works.
- Enlarging the spectral core (fewer, wider blocks; more radial modes; more degrees)
  does not help: the spectral branch is not capacity-limited. Without the local branch
  the error is ~0.8 at every level.
- Measured and rejected as speedups: `channels_last` (no effect), bf16 transforms
  (slower end to end, 22% worse at L4), explicit im2col matmul (the `5^3` conv already
  runs at 47 TFLOP/s, faster per flop than the GEMM).
