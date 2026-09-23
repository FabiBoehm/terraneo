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

The map is used as the **initial guess** of the production multigrid-preconditioned
FGMRES solver. The solver still produces and certifies the answer; the network only
decides where the iteration starts, so a wrong prediction costs iterations, never
correctness.

## Install

    pip install -e python/

On SNG-2 the working stack is `module load python/3.10.12-extended` plus the
`torch 2.7.1+xpu` virtualenv on scratch; the newer XPU wheels see no devices there.

## The operator

The discrete system `K_eta x = f` is linear in `f` for fixed `eta`; all the difficulty
is in how `K_eta^-1` depends on `eta`. The model has the same structure: it is
**exactly linear in `(f_u, f_p)` and nonlinear in `eta`**. Every path the forcing takes
is bias-free and linear; viscosity enters only through multiplicative gates, through
the generator of the spectral matrices, and through the mixing weights of the local
stencils — never as a linear input channel. One weight set serves all four
refinement levels (L3–L6, `n = 9, 17, 33, 65` nodes per diamond edge).

Shapes below: `B` batch, `S = 10` diamonds, `N = n^3` nodes per diamond, `C = 128`
channels, `M = (l_max+1)^2` harmonics, `k_1 = k_max+1` radial modes, `n_b = 8` blocks,
`b_s = 16`, `tok = b_s k_1`.

| level | n | nodes | l_max | M | k_1 | tok |
|---|---|---|---|---|---|---|
| L3 | 9 | 7,290 | 12 | 169 | 9 | 144 |
| L4 | 17 | 49,130 | 24 | 625 | 17 | 272 |
| L5 | 33 | 359,370 | 32 | 1089 | 17 | 272 |
| L6 | 65 | 2,746,250 | 32 | 1089 | 17 | 272 |

**Input split.** Channels 0–3 (`f_u`, `f_p`) go to a bias-free `Linear(4 -> 128)`.
Channel 4 (standardised `log eta`) never enters that lift: concatenated with four
geometry features (Cartesian position, normalised depth) it drives `gate_in`, a
`5 -> 128 -> 128` MLP whose output multiplies the lifted features node by node.

**Spectral Green core** (runs in fp32 under bf16 autocast). Reshape to
`(B, C, S n^2, n)`; spherical-harmonic analysis `A (M x S n^2)` (the pseudo-inverse of
the real-harmonic basis — nodes shared between diamonds are stored once per diamond, so a
quadrature would double-count the seams); Chebyshev analysis `A_r^T`
(`A_r = Y_r^+`, `k_1 x n`); regroup `(B, C, M, k_1) -> (B, n_b, M, tok)`. For each degree
`l` one dense `tok x tok` matrix per block acts on all orders of that degree: sharing
across orders is the rotational symmetry of the radially-varying-viscosity operator, not
an approximation. The matrices are **generated, not stored**: `ggen` maps the normalised
indices `(l, k, k')` concatenated with a 16-dimensional embedding of the sample's
viscosity (radial mean and std of `log eta`, resampled to 16 points) through
`19 -> 64 -> 64 -> n_b b_s^2 = 2048`. So every sample gets its own Green operator, and a
finer mesh simply queries the generator at a larger truncation. The two high-contrast
experts add a cross-degree attention (`(B, M, 18) -> q, k (18 -> 32) -> A (B, M, M)`,
keyed on `eta` only, gated residual). Synthesis with `Y_r (n x k_1)` then
`Y (S n^2 x M)`; added to the features as a residual.

**Local branch.** Reshape to `(S B, C, n, n, n)`; four residual layers, each a bias-free
`5^3` convolution (`128 -> 128`) plus a viscosity-mixed kernel bank: a `1^3` projection
`128 -> 32`, `K = 4` kernels `(32, 32, 5, 5, 5)`, a `3 -> 32 -> K` softmax computed at
every node from `(log eta, its 5^3 mean, its 5^3 std)`, and `1^3` back to `128`. The
effective stencil therefore varies node by node with the local viscosity. GELU on the
residual. This branch is 98% of the parameters and ~95% of the compute (45 TFLOP per
forward at L6; the whole spectral core is 0.001).

**Head, projection, boundary.** `gate_out` (same inputs as `gate_in`) multiplies the
features, `Linear(128 -> 4)` yields `(u, p)`. The output is then projected onto the
finite-element space: every copy of a node shared between diamonds is replaced by the
mean of its copies (`--seam-average`). This is not cosmetic — a Krylov method cannot
remove any component of a guess that lies outside that space, and before the projection
the solver plateaued at 6% error regardless of budget. The caller zeroes velocity on the
two Dirichlet shells.

**Defect correction.** `x = N(f) + gamma N(f - K_eta N(f))` with one learned scalar
`gamma`: classical iterative refinement through the same weights, still exactly linear
in `f`. Doubles the forward cost, adds no parameters, worth about 10% error.

**Mixture of experts.** Four copies, routed deterministically on `max eta / min eta`
against thresholds `10, 10^2, 10^3`. They are not identical:

| expert | contrast band | stencil banks | bank width | attention | params |
|---|---|---|---|---|---|
| generalist | < 10 | – | – | no | 8.44 M |
| mid | 10 – 10^2 | 4 | 128 | no | 41.21 M |
| high | 10^2 – 10^3 | 4 | 32 | yes | 10.50 M |
| top | > 10^3 | 4 | 32 | yes | 10.50 M |

### What is and is not discretisation invariant

The spectral core is mesh-independent by construction: its weights are indexed by
channel, degree and radial mode, its matrices come from a generator over normalised
indices, and refining the mesh only rebuilds the fixed transform matrices
(`set_mesh`). The two experts that serve contrasts below `10^2` were trained on L3–L5
only and have never seen an L6 field; two thirds of the L6 test set is handled by them,
so the L6 column below is predominantly a zero-shot result.

The **local branch is not** invariant. Its stencils live in index space: four layers of
`5^3` reach eight nodes, which is the whole shell at L3 and an eighth of it at L6. And
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

The recipe behind the deployed top-band expert (warm-started from its predecessor):

    python -m terra_infer.train_linear_mr \
        --data $ML/stokes_L3_hc --data2 $ML/stokes_L4_hc --data3 $ML/stokes_L5_hcall \
        --extra-data $ML/stokes_L5_hc3:32:16:2:2000 \
        --extra-data $ML/stokes_L6_hc:32:16:1:60 \
        --extra-data $ML/stokes_L6_hc2:32:16:1:200 \
        --lmax 12 --kmax 8 --lmax2 24 --kmax2 16 --lmax3 32 --kmax3 16 \
        --batch-size 8 --batch-size2 4 --batch-size3 2 --batch-mix \
        --max-train 8000 --max-train3 1600 --max-test 32 \
        --epochs 14 --lr 2e-4 --amp --grad-checkpoint \
        --hidden 128 --heads 8 --linear-convs 4 --linear-kernel 5 --linear-depth-gates \
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
