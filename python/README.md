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

A **neural operator** (Kovachki et al., *Neural Operator: Learning Maps Between Function
Spaces*, arXiv:2108.08481) learns a map between *function spaces* rather than between
vectors of fixed length: here, from the pair (forcing, viscosity field) to the solution
field. It is trained once on many solved problems and then, given a new $f$ and $\eta$,
produces an approximate solution in one forward pass, with no iteration.

The distinction from an ordinary image-to-image network is that the learned object
should be the *operator* $K_\eta^{-1}$ and not a lookup on one grid. Concretely, one set
of weights should act on any discretisation of the input and, as the mesh is refined,
converge to a single continuum operator — what that paper calls **discretisation
convergence**. Our three targets, then: exact linearity in the forcing, discretisation
convergence as far as we can achieve it, and enough accuracy that the solver started
from the prediction needs far fewer iterations than from zero.

What we do **not** try to do is replace the solver. The prediction is used as the
initial guess of the production multigrid-preconditioned FGMRES solver; the solver
still produces and certifies the answer. A poor prediction costs iterations, never
correctness, and the required accuracy of the network is whatever saves the most
solver time (10% turns out to be very valuable).

## The operator

We learn the solution operator of variable-viscosity Stokes flow on the spherical shell
$\Omega=\{x\in\mathbb R^3:\ r_{\min}\le|x|\le r_{\max}\}$,

$$-\nabla\cdot\bigl(2\eta\,\varepsilon(u)\bigr)+\nabla p=f_u,\qquad \nabla\cdot u=f_p,
\qquad \varepsilon(u)=\tfrac12\bigl(\nabla u+\nabla u^{\top}\bigr),$$

with $u=0$ on both bounding spheres. Discretised on the mesh this is a square linear
system $K_\eta\,x=f$ with $x=(u,p)$ and $f=(f_u,f_p)$, and the object we approximate is

$$\mathcal G^{\dagger}:\ (f,\eta)\ \longmapsto\ x \;=\; K_\eta^{-1}f .$$

Two structural facts drive every decision below.

1. **$\mathcal G^{\dagger}$ is linear in $f$** and nonlinear only in $\eta$. For a fixed
   viscosity field, doubling the load doubles the flow.
2. **$\mathcal G^{\dagger}$ is non-local.** $K_\eta^{-1}$ has a dense Green's function:
   a load anywhere moves fluid everywhere, with an influence that decays slowly.

### The template

A neural operator in the sense of Kovachki et al. is a lifting, a stack of layers each
combining a non-local integral term with a local term, and a projection:

$$\mathcal G_\theta \;=\; \mathcal Q\circ\bigl(\sigma\circ(\mathcal W_{T-1}+\mathcal K_{T-1}+b_{T-1})\bigr)\circ\cdots\circ\bigl(\sigma\circ(\mathcal W_{0}+\mathcal K_{0}+b_{0})\bigr)\circ\mathcal P,$$

$$(\mathcal K_t v)(x)\;=\;\int_{\Omega}\kappa_t\bigl(x,y,a(x),a(y)\bigr)\,v(y)\,\mathrm dy ,$$

where $a$ is the PDE's parameter function, $\mathcal K_t$ carries the non-locality,
$\mathcal W_t$ is a local term (pointwise in FNO), $b_t$ is a bias and $\sigma$ a
nonlinearity. Here $a=\log\eta$, the operator input is $f$, and our model is this
template with four deliberate specialisations:

$$v_0=\mathcal Pf=g_{\mathrm{in}}(\eta)\odot A f,\qquad
v_{t+1}=v_t+\mathcal K(\eta)\,v_t+\mathcal W(\eta)\,v_t,\qquad
x=\Pi\bigl[g_{\mathrm{out}}(\eta)\odot B\,v_T\bigr],$$

with $A\in\mathbb R^{C\times4}$ and $B\in\mathbb R^{4\times C}$ bias-free, $C=128$
channels, $\odot$ a channel-wise product, and $\Pi$ the projection onto the
finite-element space. Note what is absent: no bias, and no $\sigma$ on the path the
forcing takes.

### 1. The integral term acts in the shell's own basis

FNO evaluates $\mathcal K$ by assuming a translation-invariant kernel
$\kappa(x,y)=\kappa(x-y)$ on a torus, so that the convolution theorem turns the
integral into a diagonal multiplication in Fourier space, with one learned weight
matrix per mode up to a truncation. A spherical shell is not a torus, so we use its
natural basis: expand every channel in real spherical harmonics laterally and
Chebyshev polynomials radially,

$$v(x)\;=\;\sum_{\ell=0}^{\ell_{\max}}\sum_{m=-\ell}^{\ell}\sum_{k=0}^{k_{\max}}
\hat v_{\ell m k}\;Y_{\ell}^{m}(\theta,\varphi)\,T_k(r).$$

If the kernel is invariant under rotation — which $K_\eta^{-1}$ is whenever the
viscosity is radially layered — then it cannot couple different $(\ell,m)$, and its
action reduces to one dense matrix per degree, shared across that degree's $2\ell+1$
orders:

$$\widehat{(\mathcal K v)}_{\ell m k}\;=\;\sum_{k'}\;\bigl[G_{\ell}\bigr]_{kk'}\;\hat v_{\ell m k'} .$$

These blocks $G_\ell$ are the exact analogue of FNO's per-mode weights, and replacing
the FFT with the spherical harmonic transform is the same substitution the spherical
FNO makes for weather models. Sharing across $m$ is a symmetry of the operator, not an
approximation. In our implementation the $C$ channels are split into $n_b=8$ groups of
$b_s=16$, and $G_\ell$ acts on the stacked (channel, radial-mode) coefficients of each
group, so one block has size $b_s(k_{\max}+1)\times b_s(k_{\max}+1)$.

### 2. The kernel is generated from the viscosity, not stored

The template permits the kernel to depend on the parameter, $\kappa(x,y,a(x),a(y))$.
Feeding $\eta$ pointwise would tie the weights to grid points, so instead we summarise
it and *generate* the blocks. Let $e(\eta)\in\mathbb R^{16}$ be an embedding of the
radial mean and standard deviation of $\log\eta$ resampled to 16 radii. A small network
$\Gamma$ then writes every block entry from the normalised indices,

$$\bigl[G_{\ell}\bigr]^{(b)}_{(c,k),(c',k')}\;=\;\Gamma\Bigl(\tfrac{\ell}{\ell_{\max}},\tfrac{k}{k_{\max}},\tfrac{k'}{k_{\max}};\,e(\eta)\Bigr)^{(b)}_{cc'} .$$

Two things follow. Every sample gets its own Green's function, conditioned on its own
viscosity profile. And the learned object is a function of the *continuous* indices
$(\ell,k,k')$ rather than a table indexed by grid points, so a finer mesh simply
evaluates $\Gamma$ at more modes — which is why this term is discretisation convergent.
The two high-contrast experts add a learned mixing across degrees, keyed on $\eta$
alone, for the cases where strong lateral contrast breaks the rotational symmetry that
makes $G_\ell$ block-diagonal in the first place.

### 3. The local term is a short-range kernel, not a pointwise one

In FNO, $\mathcal W_t$ is pointwise and all non-locality sits in $\mathcal K_t$. That is
not sufficient here. Truncating at $\ell_{\max}=32$ cannot represent the edge of a stiff
slab, and at four orders of magnitude of viscosity contrast that edge is exactly where
the solution lives. So we widen $\mathcal W_t$ into a short-range kernel — a stencil of
half-width 2 in each index direction — whose weights vary with the local viscosity:

$$(\mathcal W v)(x_i)\;=\;\sum_{\|\delta\|_\infty\le2}\Bigl[\,W_\delta\;+\;\sum_{j=1}^{K}c_j\bigl(\eta;x_i\bigr)\,B^{(j)}_{\delta}\Bigr]\,v\bigl(x_{i+d\,\delta}\bigr),$$

where $\delta$ runs over the $5^3$ neighbour offsets, $W_\delta\in\mathbb R^{C\times C}$
are dense weights, $B^{(j)}$ are $K=4$ learned kernels acting through a 32-channel
bottleneck, the mixing weights $c_j$ are a softmax of $\log\eta$ at $x_i$ together with
its mean and standard deviation over the same $5^3$ window, and $d$ is a dilation. The
effective stencil therefore differs at every node according to the viscosity around it.

This term is what makes the model work — without it the error is roughly $0.8$ at every
level — and it is also the one component defined on the grid rather than on the
continuum. With $T$ layers its reach is $2Td$ nodes, so its *physical* footprint is
$2Td\,h$ where $h$ is the node spacing. Holding $d=1$ while refining the mesh shrinks
that footprint in proportion to $h$; choosing $d\propto1/h$ holds it fixed. Section
*Discretisation convergence* below returns to this with measurements.

### 4. No activation and no bias on the forcing path

A generic neural operator applies $\sigma$ after every layer, which is what buys
universal approximation. We omit it, together with every bias, wherever the forcing
flows. The viscosity enters only multiplicatively — through the gates
$g_{\mathrm{in}},g_{\mathrm{out}}$, through the generator $\Gamma$, and through the
mixing weights $c_j$ — so the composed map satisfies, exactly and at any resolution,

$$\mathcal G_\theta(\alpha f_1+\beta f_2,\ \eta)\;=\;\alpha\,\mathcal G_\theta(f_1,\eta)+\beta\,\mathcal G_\theta(f_2,\eta).$$

In particular $\mathcal G_\theta(0,\eta)=0$. This is a strictly smaller hypothesis class
than the universal one, chosen because it is the structure $K_\eta^{-1}$ actually has:
the network cannot waste capacity learning that small loads behave like large ones, and
it cannot drift out of that class during training. All the nonlinear capacity is spent
where the true difficulty is, on how $\eta$ shapes the operator.

### Projection, defect correction, experts

**Projection onto the finite-element space.** The mesh stores nodes shared between
diamonds once per diamond; $\Pi$ replaces every copy of a shared node by the mean of its
copies. This is not cosmetic. A Krylov method can only remove error components that lie
in its own space, so any part of the prediction outside the finite-element space is
error the solver cannot touch: before adding $\Pi$ the solver plateaued at 6% error
regardless of budget. The caller then zeroes velocity on the two Dirichlet shells.

**Defect correction.** One forward pass leaves a residual $r=f-K_\eta\mathcal G_\theta f$.
Applying the same operator to it and adding the result back,

$$x\;=\;\mathcal G_\theta f+\gamma\,\mathcal G_\theta\bigl(f-K_\eta\,\mathcal G_\theta f\bigr)
\;=\;\bigl[(1+\gamma)\,\mathcal G_\theta-\gamma\,\mathcal G_\theta K_\eta\mathcal G_\theta\bigr]f,$$

is classical iterative refinement through the same weights, with one learned scalar
$\gamma$. The right-hand form shows it stays exactly linear in $f$. It doubles the
forward cost, adds no parameters, and is worth about 10% of the error.

**Experts.** Four copies of the operator, selected deterministically by the contrast
$c=\max\eta/\min\eta$ against thresholds $10,10^2,10^3$. Since $c$ depends on $\eta$
only, routing preserves linearity in $f$. Problems at $c=2$ and $c=10^4$ want genuinely
different stencils, and one shared weight set is measurably worse than four specialised
ones (mean error 0.20–0.31 against 0.084–0.189).

### In shapes, as implemented

`B` batch, `S = 10` diamonds, `N = n^3` nodes per diamond, `C = 128` channels,
`M = (l_max+1)^2` harmonics, `k_1 = k_max+1` radial modes, `n_b = 8` blocks of
`b_s = 16` channels, `tok = b_s k_1`.

| level | n | nodes | l_max | M | k_1 | tok |
|---|---|---|---|---|---|---|
| L3 | 9 | 7,290 | 12 | 169 | 9 | 144 |
| L4 | 17 | 49,130 | 24 | 625 | 17 | 272 |
| L5 | 33 | 359,370 | 32 | 1089 | 17 | 272 |
| L6 | 65 | 2,746,250 | 32 | 1089 | 17 | 272 |

Reading the pipeline of the previous section off the tensors:

- **Lift and gate.** $(f_u,f_p)$ → bias-free `Linear(4 -> 128)`, multiplied node by node
  by `gate_in`, a `5 -> 128 -> 128` MLP of (Cartesian position, normalised depth,
  $\log\eta$).
- **Integral term** (kept in fp32 under bf16 autocast). Reshape to `(B, C, S n^2, n)`;
  harmonic analysis by $A$ of shape `(M, S n^2)` — the *pseudo-inverse* of the basis
  rather than a quadrature, because nodes shared between diamonds are stored once per
  diamond and a quadrature would double-count the seams; Chebyshev analysis by
  $A_r^{\top}$, shape `(k_1, n)`; regroup to `(B, n_b, M, tok)`; apply one
  `tok x tok` matrix per degree and block, emitted by
  `ggen: 19 -> 64 -> 64 -> 2048` from the three normalised indices and the 16-dimensional
  $\eta$ embedding; synthesise with $Y_r$ then $Y$; add as a residual. The optional
  cross-degree mixing is an attention with `18 -> 32` queries and keys, giving a dense
  `(B, M, M)` map, gated and keyed on $\eta$ alone.
- **Local term.** Reshape to `(S B, C, n, n, n)`; four residual layers (eight in the top
  expert), each a bias-free `5^3` convolution `128 -> 128` plus the bank: `1^3` down to
  32 channels, `K = 4` kernels of shape `(32, 32, 5, 5, 5)`, softmax weights from a
  `3 -> 32 -> K` MLP, `1^3` back to 128. Added as a residual with **no** activation —
  the `--nonlin` flag would insert a GELU and is off in every deployed expert, since it
  would destroy linearity in $f$. This term holds 98% of the parameters and costs
  45 TFLOP per forward at L6, against 0.001 for the integral term.
- **Head.** `gate_out` (same inputs as `gate_in`) then bias-free `Linear(128 -> 4)`,
  followed by the seam-mean projection $\Pi$; the caller zeroes velocity on the two
  Dirichlet shells.

The four deployed experts:

| expert | contrast band | stencil banks | bank width | attention | params |
|---|---|---|---|---|---|
| generalist | < 10 | – | – | no | 8.44 M |
| mid | 10 – 10^2 | 4 | 128 | no | 41.21 M |
| high | 10^2 – 10^3 | 4 | 32 | yes | 10.50 M |
| top | > 10^3 | 4 | 32 | yes | 20.81 M |

The top expert is the exception to "four convolutions": it has **eight** local layers
(`--linear-convs 8`, hence its name `c8ckpt`), which doubles its stencil reach to 16
nodes. 80.96 M parameters in the four experts together.

### Discretisation convergence: what holds and what does not

A model is discretisation convergent when one set of weights acts on any discretisation
and the outputs converge to a single continuum operator as the mesh is refined. Our two
terms behave oppositely, and the split is the central caveat of this work.

**The integral term converges.** Its weights are indexed by channel, degree and radial
mode, and they are produced by $\Gamma$ from normalised indices, so they are values of a
function on the continuum rather than a table tied to nodes. Refining the mesh rebuilds
only the fixed transform matrices $Y,Y_r$ and queries $\Gamma$ at more modes.

**The local term does not.** Its offsets $\delta$ are counted in nodes, so with $T$
layers of half-width 2 and dilation $d$ its physical reach is

$$\rho \;=\; 2\,T\,d\,h,\qquad h \;\sim\; \frac{L}{n},$$

and holding $d=1$ while $n$ grows sends $\rho\to0$: the term degenerates towards the
pointwise $\mathcal W_t$ of the generic template, which is a *different* operator from
the one trained at the coarse level. In numbers, four layers reach 8 nodes — the whole
shell at L3, an eighth of it at L6 (the eight-layer top expert reaches 16, so a quarter).
Choosing $d\propto1/h$ — the `--dilated-stencils` option, $d=2^{L-3}$ — holds $\rho$
fixed and costs no extra weights.

A second, milder violation: the truncation is held at $\ell_{\max}=32$ above L4 while
the mesh keeps growing, so the harmonics span 21% of the lateral degrees of freedom at
L3–L4, 10% at L5 and 2.6% at L6. Both effects point the same way as the measured error
growth with level.

Measurements confirm the diagnosis. Against index-space stencils, dilated ones cut the
held-out error ratio between L6 and L3 from **1.9 to 1.0** — error that no longer grows
with refinement — with the two curves crossing at L6 (see *Results*). Separability
(`--sep-stencils`, depthwise plus `1^3`, about 9x cheaper) is orthogonal: it saves
compute, costs accuracy, and does nothing for convergence. Neither option is in the
deployed experts yet.

Finally, an accident of training history that the results below inherit: the two experts
serving contrasts below $10^2$ were trained on L3–L5 only and have never seen an L6
field, and they handle two thirds of the L6 test set — so the L6 column is largely a
zero-shot result.

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
