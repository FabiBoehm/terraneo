# terra-ml

*Draft: a learned initial guess for the TERRA-NG Stokes solver. First results are
promising; this has not been through a review and is ongoing work.*

A neural solution operator for variable-viscosity Stokes flow on the TERRA-NG
spherical shell — a network that takes the forcing and the viscosity field and returns
velocity and pressure in one forward pass — together with the synthetic data it trains
on, the code path by which the simulation calls it, and a benchmark mode in the
production solver that measures how many iterations it saves.

- `terra_data` — training data by the method of manufactured solutions: pick a random
  polynomial velocity, pressure and viscosity field, and compute symbolically the forcing
  that makes them an exact Stokes solution, using the same form of the viscous operator
  TERRA implements. Verified against the code's own analytic test cases.
- `terra_infer` — the operator (`operator.py`), its trainer (`train_linear_mr.py`),
  the spherical-harmonic and Chebyshev transforms, a finite-difference form of the
  Stokes operator used in the loss and in defect correction, the mesh's symmetry group
  used for augmentation, and the entry point (`__init__.py`) the simulation's embedded
  Python interpreter calls.
- `scripts` — held-out evaluation, solver-bench problem generation, cross-section plots.

## Install

    pip install -e python/

On SNG-2 the working stack is `module load python/3.10.12-extended` plus the
`torch 2.7.1+xpu` virtualenv on scratch; the newer XPU wheels see no devices there.

## The problem

TERRA-NG simulates mantle convection on a spherical shell. Every time step it must find
the flow that balances the buoyancy under the current viscosity field, by solving the
Stokes equations

$$-\nabla\cdot\left(2\eta\,\varepsilon(u)\right)+\nabla p=f_u,\qquad \nabla\cdot u=f_p,
\qquad \varepsilon(u)=\tfrac12\left(\nabla u+\nabla u^{\top}\right),$$

on $\Omega=\{x\in\mathbb R^3:\ r_{\min}\le|x|\le r_{\max}\}$ with $u=0$ on both bounding
spheres. Here $u$ is velocity, $p$ pressure, $\eta(x)$ the viscosity, $f_u$ the buoyancy
force and $f_p$ a mass source (zero in production, kept for generality). Discretised on
the mesh described below, this is one square linear system

$$K_\eta\,x=f,\qquad x=(u,p),\qquad f=(f_u,f_p),$$

whose matrix $K_\eta$ depends on the viscosity. It is the dominant cost of a time step
for two reasons: the viscosity changes every step, so nothing from the previous solve
is reused, and at high viscosity contrast — four orders of magnitude between a cold
slab and the material around it — the iterative solver needs tens of iterations, each
costing 66 ms at level 3 and 162 ms at level 6.

**What we build.** A network that predicts the solution directly from the data of the
problem,

$$\mathcal G_\theta:\ (f,\ \eta)\ \longmapsto\ x\ \approx\ K_\eta^{-1}f,$$

and is used as the **initial guess** of the existing solver. The solver still produces
and certifies the answer by driving the residual to tolerance; the network only decides
where the iteration starts. A wrong prediction costs iterations, never correctness,
which is what makes this safe to deploy. We do not try to replace the solver, and we do
not use the network as a preconditioner (that was tried and does not work; see *Known
limits*).

**The mesh.** TERRA-NG covers the sphere with ten curvilinear *diamonds*, each an
$n\times n$ grid of nodes laterally and $n$ nodes radially, so a diamond holds $n^3$
nodes and the shell holds $10n^3$. Refinement level $L$ has $n=2^L+1$: 9, 17, 33, 65 at
levels 3–6, i.e. 7 290 to 2.75 million nodes. Diamonds share their edge nodes; the data
layout stores each shared node once *per diamond*, so a node on a seam between two
diamonds appears twice in memory and a corner node up to five times. That detail matters
twice below.

**Why a neural operator.** A network that maps a picture of $(f,\eta)$ on one grid to a
picture of $x$ on the same grid learns nothing that transfers: the weights are tied to
the pixel layout, and a finer mesh is a different problem. Kovachki et al. (*Neural
Operator: Learning Maps Between Function Spaces*, arXiv:2108.08481) formalise the
alternative: parameterise a map between *functions*, so that one set of weights acts
on any discretisation of the input and, as the mesh is refined, the outputs converge to
a single continuum operator. They call this **discretisation convergence**, and it is
the property that lets a model trained mostly on cheap coarse meshes be used on the
expensive fine one. Our goals, in order of how strictly we achieve them:

1. **Exact linearity in $f$.** For fixed $\eta$ the true map $f\mapsto K_\eta^{-1}f$ is
   linear. We build that in rather than hoping to learn it.
2. **Discretisation convergence** for as much of the model as possible. One component
   violates it, deliberately, and the violation is measured and documented.
3. **Enough accuracy to save solver time.** Around 10% relative error turns out to be
   worth a factor of two in iterations; below that the returns diminish.


## The operator

### Two facts that shape the design

The true solution operator $\mathcal G^\dagger(f,\eta)=K_\eta^{-1}f$ has two properties
that every decision below traces back to.

1. **It is linear in $f$ and nonlinear only in $\eta$.** Doubling the load doubles the
   flow; changing the viscosity changes the operator itself.
2. **It is non-local.** $K_\eta^{-1}$ has a dense Green's function: a load anywhere
   moves fluid everywhere, with an influence that decays only algebraically with
   distance. No stencil of fixed width can represent it.

### Notation

| symbol | meaning | value |
|---|---|---|
| $L$ | refinement level | 3–6 |
| $n$ | nodes along each edge of a diamond, $n=2^L+1$ | 9, 17, 33, 65 |
| $h$ | node spacing; radially $(r_{\max}-r_{\min})/(n-1)$, laterally similar | — |
| $P$ | nodes in the shell as stored, $10n^3$ | 7 290 … 2 746 250 |
| $d_v$ | feature channels the network carries at every node | 128 |
| $\ell, m$ | spherical-harmonic degree and order, $0\le\ell\le\ell_{\max}$, $-\ell\le m\le\ell$ | $\ell_{\max}=12,24,32,32$ |
| $k$ | Chebyshev radial index, $0\le k\le k_{\max}$; $k_1=k_{\max}+1$ | $k_{\max}=8,16,16,16$ |
| $Y_\ell^m$, $T_k$ | real spherical harmonic, Chebyshev polynomial | — |
| $n_b$, $b_s$ | channel groups and channels per group, $d_v=n_b b_s$ | 8, 16 |
| $n_{loc}$ | number of local layers | 4 (8 in the top expert) |
| $J$ | learned stencil kernels per local layer | 4 |
| $d$ | stencil dilation, in nodes | 1 (or $(n-1)/8$ with `--dilated-stencils`) |
| $\chi$ | viscosity contrast $\max\eta/\min\eta$ of a sample | 1 – $10^4$ |
| $\Pi$ | projection onto the finite-element space, defined in *Projection* below | — |
| $\odot$ | product channel by channel at each node | — |

Two typographic conventions. $K_\eta$ (roman) is always the discrete Stokes matrix and
$\mathcal K$ (script) always the network's integral term. Boldface is not used; vectors
and fields are clear from context.

**Inputs and outputs, concretely.** At every node the network receives five numbers:
the three components of $f_u$, the value of $f_p$, and the *standardised*
log-viscosity $(\log\eta-\mu)/s$, where $\mu$ and $s$ are the mean and standard
deviation of $\log\eta$ over the training set (stored in the checkpoint as
`log_eta_mean`, `log_eta_std`). It returns four numbers: the three components of $u$
and $p$. The velocity targets are scaled by each sample's geometric-mean viscosity
$\exp\langle\log\eta\rangle$ before training, since flow speed is inversely
proportional to viscosity and this removes a factor of $10^4$ of dynamic range from the
regression; the scaling is undone at inference. Four *geometry* numbers per node are
also available to the model but are fixed by the mesh, not part of the data: the
Cartesian position $(x_1,x_2,x_3)$ and the normalised depth
$(r-r_{\min})/(r_{\max}-r_{\min})$.

### The neural-operator template

Kovachki et al. write a neural operator as a lifting $\mathcal P$, a stack of $T$
layers, and a projection $\mathcal Q$,

$$\mathcal G=\mathcal Q\circ\mathcal L_{T-1}\circ\cdots\circ\mathcal L_0\circ\mathcal P,$$

where $\mathcal P$ and $\mathcal Q$ act node by node (they change the number of channels,
not the function's support), and each layer combines a local term $\mathcal W_t$, a
non-local term $\mathcal K_t$, a bias $b_t$ and a nonlinearity $\sigma$:

$$\mathcal L_t(v)=\sigma\left(\mathcal W_t v+\mathcal K_t v+b_t\right).$$

The non-local term is an integral against a learned kernel, which may depend on the
PDE's parameter function $a$:

$$(\mathcal K_t v)(x)=\int_\Omega\kappa_t\left(x,y,a(x),a(y)\right)v(y)\,dy.$$

In the Fourier neural operator (FNO) $\mathcal W_t$ is pointwise, $\kappa_t$ is a
function of $x-y$ only, and the integral is evaluated by FFT. Here $a=\log\eta$, and the
"input" the operator acts on is $f$.

### Our instance, in four steps

Write $v$ for the feature field: a function on the shell with $d_v=128$ values at each
node, initialised from $f$ and transformed in place.

**Step 1 — lift.** Map the 4 forcing values at each node to $d_v$ features, then rescale
every feature by a factor that depends on the viscosity and position at that node:

$$v\ \leftarrow\ g_{in}(\eta)\odot\left(W_P\,f\right).$$

$W_P\in\mathbb R^{d_v\times4}$ is a learned matrix with no bias. $g_{in}$ is a learned
two-layer MLP, $\mathrm{Linear}(5\to128)$, GELU, $\mathrm{Linear}(128\to128)$, evaluated
at each node on the four geometry numbers and the standardised $\log\eta$ there; its
output is the vector of $d_v$ scale factors.

**Step 2 — the integral term, applied once.** The global coupling; sections 1 and 2
below define $\mathcal K(\eta)$:

$$v\ \leftarrow\ v+\mathcal K(\eta)\,v.$$

**Step 3 — local layers, $n_{loc}$ of them.** Each layer $t$ has its own stencil
operator $\mathcal W_t(\eta)$ — a $5^3$ stencil whose weights depend on the local
viscosity, defined in full in section 3 — and its own gate $g_t$, a learned MLP
$\mathrm{Linear}(4\to128)$, GELU, $\mathrm{Linear}(128\to128)$ of the four geometry
numbers only (position and depth, not viscosity):

$$v\ \leftarrow\ v+\mathcal W_t(\eta)\left(g_t\odot v\right),\qquad t=1,\dots,n_{loc}.$$

**Step 4 — project.** Rescale by an output gate $g_{out}$ of exactly the same form and
inputs as $g_{in}$ (separate weights), map the $d_v$ features back to the four
components of $x=(u,p)$ with a learned bias-free $W_Q\in\mathbb R^{4\times d_v}$, and
project onto the finite-element space:

$$x\ \leftarrow\ \Pi\left(W_Q\left(g_{out}(\eta)\odot v\right)\right).$$

Compared with the template, three things are different, all on purpose.

- **$\mathcal K$ and $\mathcal W$ are not interleaved.** The template alternates them in
  every layer. We apply $\mathcal K$ once and then $n_{loc}$ local layers, because one
  application of $\mathcal K$ already couples every radial mode with every other at each
  harmonic degree — a second gains little — whereas each local layer extends the local
  term's reach by two nodes, and reach is what the model lacks.
- **No bias.** Every $b_t=0$, and $W_P$, $W_Q$ have none.
- **No nonlinearity on the path $f$ takes.** The only nonlinear functions in the model
  are the gates $g$, the kernel generator $\Gamma$ (section 2) and the stencil mixing
  weights $\lambda$ (section 3) — and all of them are functions of $\eta$ and geometry,
  never of $f$. Section 4 states what this buys.

### 1. The integral term acts in the shell's own basis

FNO makes the integral cheap by assuming a translation-invariant kernel
$\kappa(x,y)=\kappa(x-y)$ on a periodic box: the integral is then a convolution, the
FFT diagonalises it, and the model learns one weight matrix per Fourier mode up to a
truncation. A spherical shell has no translations, but it has rotations about its
centre, and the basis adapted to those is spherical harmonics laterally and Chebyshev
polynomials radially. This subsection defines that expansion; the next two say what the
model does with it.

*Coordinates.* Write a point as $x=(r,\theta,\varphi)$: radius $r\in[r_{\min},r_{\max}]$,
colatitude $\theta\in[0,\pi]$, longitude $\varphi\in[0,2\pi)$. The radius is mapped to
the Chebyshev interval by

$$\hat r=\frac{2r-r_{\min}-r_{\max}}{r_{\max}-r_{\min}}\in[-1,1].$$

*The basis functions.* $Y_\ell^m(\theta,\varphi)$ are the real spherical harmonics: the
eigenfunctions of the Laplacian on the unit sphere, indexed by degree $\ell=0,1,2,\dots$
and order $m=-\ell,\dots,\ell$. Degree sets the lateral scale — $Y_\ell^m$ oscillates
roughly $\ell$ times around a great circle, so $\ell_{\max}=32$ resolves nothing finer
than about $1/32$ of the circumference — and the $2\ell+1$ orders of one degree are
rotated copies of each other, which is the property section 2 exploits. Radially,
$T_k(\hat r)=\cos(k\arccos\hat r)$ is the Chebyshev polynomial of degree $k$; $T_0=1$,
$T_1=\hat r$, $T_2=2\hat r^2-1$, and $T_k$ oscillates $k$ times across the shell's
thickness.

*The expansion.* The feature field $v$ has $d_v$ channels; each channel is a scalar
field on the shell and is expanded on its own. For one channel,

$$v(r,\theta,\varphi)\ \approx\ \sum_{\ell=0}^{\ell_{\max}}\ \sum_{m=-\ell}^{\ell}\ \sum_{k=0}^{k_{\max}}
\hat v_{\ell mk}\ \,Y_\ell^m(\theta,\varphi)\ T_k(\hat r),$$

a sum of $M\cdot k_1$ terms, with $M=(\ell_{\max}+1)^2$ lateral functions (all pairs
$(\ell,m)$ up to $\ell_{\max}$) and $k_1=k_{\max}+1$ radial ones. The coefficients
$\hat v_{\ell mk}$ are the field's representation in this basis; there is one such set
of $M k_1$ numbers per channel. Because $Mk_1$ is far smaller than the number of nodes
— $1089\times17\approx18{,}500$ against $2.75$ million at level 6 — the expansion is a
*truncation*: it can represent exactly only fields that are smooth at the scale of
$\ell_{\max}$ and $k_{\max}$, and for any other field the coefficients are chosen as the
best fit in the least-squares sense. That is why the symbol is $\approx$, and it is the
reason section 3 exists.

*How the coefficients are computed.* The mesh is a product: every diamond has the same
$n$ radial layers, so the $10n^3$ stored nodes are $10n^2$ lateral positions
$(\theta_a,\varphi_a)$ times $n$ radii $\hat r_b$. That lets the transform separate into
a lateral and a radial factor. Define two fixed matrices from the node positions,

$$Y_{a,(\ell m)}=Y_\ell^m(\theta_a,\varphi_a)\quad(10n^2\times M),\qquad
(Y_r)_{b,k}=T_k(\hat r_b)\quad(n\times k_1).$$

Arrange one channel of $v$ as a matrix $V$ of size $10n^2\times n$ (lateral node by
radial node) and its coefficients as $\hat V$ of size $M\times k_1$ (harmonic by radial
mode). Then

$$\text{synthesis:}\quad V=Y\,\hat V\,Y_r^{\top},\qquad\qquad
\text{analysis:}\quad \hat V=Y^{+}\,V\,(Y_r^{+})^{\top},$$

where $Y^{+}$ and $Y_r^{+}$ are the Moore–Penrose pseudo-inverses, i.e. the
least-squares fit. A quadrature rule (weighting each node by the area it represents)
would be the textbook choice for analysis, but it is wrong for this storage layout: seam
nodes appear in two or more diamonds and would be counted two or more times. The
least-squares fit is unaffected by duplicated rows. Both matrices depend only on the
mesh and are built once per level; they are not learned.

*What the operator does to the coefficients.* If a kernel is invariant under rotations
of the sphere, it cannot couple a harmonic $(\ell,m)$ to any $(\ell',m')$ with
$\ell'\ne\ell$, and it must act identically on all $2\ell+1$ orders $m$ of a given
degree. The Stokes inverse has this property whenever $\eta$ depends on $r$ only.
Under it the integral term collapses to one dense matrix per degree acting on the radial
coefficients:

$$\widehat{(\mathcal K v)}_{\ell mk}=\sum_{k'}\left[G_\ell\right]_{kk'}\,\hat v_{\ell mk'}.$$

The matrices $G_\ell$ are the exact counterpart of FNO's per-mode weights, with the
spherical-harmonic transform standing in for the FFT — the same substitution the
spherical FNO makes for weather models. Sharing across $m$ is a symmetry of the true
operator, not an approximation.

*Channels.* The channels are not treated independently. The $d_v$ channels are split
into $n_b=8$ groups of $b_s=16$, and within a group $G_\ell$ mixes channels and radial
modes together: for each degree $\ell$ and each group there is one matrix of size
$(b_s k_1)\times(b_s k_1)$ acting on the group's stacked (channel, radial-mode)
coefficients. The block size $b_s k_1$ is called `tok` in the code.

### 2. The kernel is generated from the viscosity, not stored

The template allows the kernel to depend on the parameter, $\kappa(x,y,a(x),a(y))$. If
we let every $G_\ell$ be a free matrix, the model would learn one Green's function for
the *average* viscosity of the training set, and could not respond to the viscosity of
the problem in front of it. So the matrices are *generated* from $\eta$ by a small
network.

*Summarising the viscosity.* Take the radial profiles of the mean and standard deviation
of $\log\eta$ over each spherical surface, resample each to 16 radii, and pass the 32
numbers through an MLP $32\to64\to16$. The result $e(\eta)\in\mathbb R^{16}$ is the
embedding. (Its last layer is initialised small, so at the start of training every
sample gets nearly the same kernel and the dependence on $\eta$ is learned gradually.)

*Generating the entries.* A learned network $\Gamma$ — $\mathrm{Linear}(19\to64)$, GELU,
$\mathrm{Linear}(64\to64)$, GELU, $\mathrm{Linear}(64\to2048)$, with $3+16=19$ inputs and
$n_b b_s^2=2048$ outputs — is evaluated once per index triple $(\ell,k,k')$:

$$\left[G_\ell\right]^{(q)}_{(c,k),(c',k')}
=\Gamma\!\left(\frac{\ell}{\ell_{\max}},\ \frac{k}{k_{\max}},\ \frac{k'}{k_{\max}}\ ;\ e(\eta)\right)^{(q)}_{cc'}.$$

The three arguments before the semicolon are the normalised indices; the argument after
it is the conditioning. $q=1,\dots,n_b$ selects the channel group and $c,c'$ are channels
within it, so the 2048 outputs are one $b_s\times b_s$ channel-mixing matrix for each of
the $n_b$ groups. Evaluating $\Gamma$ on the whole index grid fills every $G_\ell$.

Two consequences follow. Every sample gets its own Green's function, conditioned on its
own viscosity profile. And the learned object is a smooth function of *continuous*
indices rather than a table with one entry per grid point, so a finer mesh — which means
a larger $\ell_{\max}$ and $k_{\max}$ — simply evaluates $\Gamma$ at more points. This is
exactly what makes the integral term discretisation convergent.

*Lateral contrast.* The block-diagonal structure was derived for $\eta=\eta(r)$. When
the viscosity varies strongly laterally, harmonics of different degree do couple. The
two experts that serve the highest contrasts add a learned coupling across degrees: an
attention over the $M$ harmonics. For each $(\ell,m)$ an 18-vector is formed from the
radial profile of the $(\ell,m)$-component of $\log\eta$ (16 values) and the normalised
$\ell$ and $m$ (2 values); learned matrices $Q_a,K_a\in\mathbb R^{32\times18}$ (with
biases) map it to a query and a key, and $A=\mathrm{softmax}(QK^{\top}/\sqrt{32})$ is an
$M\times M$ mixing matrix applied to the coefficients of every channel group. It is added
through a learned gate $\in\mathbb R^{n_b}$, one scalar per channel group, initialised
at zero. Because it is built from $\eta$ alone, it is a fixed
linear map of the coefficients for a given problem.

### 3. The local term is a short-range kernel, not a pointwise one

In FNO, $\mathcal W_t$ is pointwise and all spatial coupling is in $\mathcal K_t$. That is
not enough here. A harmonic truncation at $\ell_{\max}=32$ resolves lateral features no
finer than about $1/32$ of the circumference, and at level 6 that is 2.6% of the mesh's
lateral degrees of freedom; the edge of a stiff slab, where velocity gradients are
steepest, lies far below that scale. So $\mathcal W_t$ is widened from a point to a
$5\times5\times5$ stencil in the diamond's index space, whose weights vary with the
viscosity around the node:

$$(\mathcal W_t v)(x_i)=\sum_{\|\delta\|_\infty\le2}
\left[\,W_{t,\delta}+\sum_{j=1}^{J}\lambda_j(\eta;x_i)\,\Psi^{(j)}_{t,\delta}\right]
v\!\left(x_{i+d\delta}\right).$$

Reading it term by term:

- $x_i$ is a node and $\delta\in\{-2,\dots,2\}^3$ ranges over its $5^3$ index offsets, so
  $x_{i+d\delta}$ is the node $d\delta$ steps away along the diamond's grid directions,
  with $d$ the dilation.
- $W_{t,\delta}\in\mathbb R^{d_v\times d_v}$ is a dense weight for each offset: a standard
  $5^3$ convolution with $128\to128$ channels, the same at every node.
- $\Psi^{(j)}_{t,\delta}$ are $J=4$ further kernels. To keep them affordable they act
  through a *bottleneck*: the features are projected $128\to32$ by a learned matrix
  $B_{in}\in\mathbb R^{32\times128}$ (a $1\times1\times1$ convolution, no bias), each
  $\Psi^{(j)}_t$ is a $5^3$ kernel with $32\to32$ channels, and the mixed result is
  projected back by $B_{out}\in\mathbb R^{128\times32}$. $B_{in}$ and $B_{out}$ are
  shared by all local layers; the kernels are per layer. In the formula above,
  $\Psi^{(j)}_{t,\delta}$ stands for the composite $B_{out}\Psi^{(j)}_{t,\delta}B_{in}$.
  The bottleneck width 32 is what the expert table calls "bank width".
- $\lambda_j(\eta;x_i)$ are the mixing weights: a softmax over $j$ output by a learned
  two-layer MLP, $\mathrm{Linear}(3\to32)$, GELU, $\mathrm{Linear}(32\to J)$, one per
  local layer, whose three inputs are $\log\eta$ at $x_i$ and the mean and standard
  deviation of $\log\eta$ over the same $5^3$ window. The mean says how stiff the neighbourhood is,
  the deviation says whether an interface runs through it.

So every node has its own effective stencil, chosen by the viscosity around it. This term
holds 98% of the parameters and almost all of the compute — 45 TFLOP per forward pass at
level 6, against about 1.5 TFLOP for the integral term's two transforms, by operation
count — and it is what makes the model work:
without it the error is about 0.8 at every level.

It is also the one component defined on the grid rather than on the continuum. With
$n_{loc}$ layers the term reaches $2n_{loc}d$ nodes from any node, a *physical* distance

$$\rho=2\,n_{loc}\,d\,h.$$

With $d=1$, refining the mesh (smaller $h$) shrinks $\rho$; with $d\propto1/h$ it stays
fixed. *Discretisation convergence* below measures what this costs.

### 4. No nonlinearity on the forcing path, and what it guarantees

A generic neural operator applies $\sigma$ after every layer; that is what gives it
universal approximation. We omit it, together with every bias, wherever $f$ flows. All
the nonlinearity lives in functions of $\eta$ and geometry — $g_{in},g_t,g_{out}$,
$\Gamma$, the degree-mixing attention, and $\lambda$ — which enter only by multiplying
$v$ or by setting the weights that multiply $v$. Consequently, for every viscosity field
and at every resolution, the network satisfies exactly

$$\mathcal G_\theta(\alpha f_1+\beta f_2,\ \eta)=\alpha\,\mathcal G_\theta(f_1,\eta)+\beta\,\mathcal G_\theta(f_2,\eta),
\qquad\text{and in particular}\qquad\mathcal G_\theta(0,\eta)=0.$$

`scripts/check_linearity.py` verifies this numerically on a random model and mesh; the
relative deviation is $10^{-6}$, i.e. floating-point rounding. It is a strictly smaller hypothesis class than the universal one, and that is the
point: it is the class the true operator belongs to. The network cannot waste capacity
learning that small loads behave like large ones, cannot produce spurious flow from zero
forcing, and cannot drift out of the class during training. The `--nonlin` flag inserts
a GELU after the integral and local terms and gives all of this up; it is off in every
deployed expert.

### Every learned parameter

The table lists every learned object in the model, where it acts, its exact form, and
its parameter count. "Bias" means the layer has an additive bias; every layer on the
path the forcing takes has none, and every layer with a bias acts only on viscosity and
geometry — that split is what section 4 relies on.

| symbol | acts in | exact form | parameters |
|---|---|---|---|
| $W_P$ | step 1 | $\mathrm{Linear}(4\to128)$, no bias | 512 |
| $g_{in}$ | step 1 | $\mathrm{Linear}(5\to128)$+bias, GELU, $\mathrm{Linear}(128\to128)$+bias | 17 280 |
| $e(\eta)$ | section 2 | $\mathrm{Linear}(32\to64)$+bias, GELU, $\mathrm{Linear}(64\to16)$+bias | 3 152 |
| $\Gamma$ | section 2 | $\mathrm{Linear}(19\to64)$+bias, GELU, $\mathrm{Linear}(64\to64)$+bias, GELU, $\mathrm{Linear}(64\to2048)$+bias | 138 560 |
| $Q_a$, $K_a$, gate | section 2, high/top experts only | $\mathrm{Linear}(18\to32)$+bias, twice; gate $\in\mathbb R^{8}$ | 608 + 608 + 8 |
| $g_t$ | step 3, per local layer | $\mathrm{Linear}(4\to128)$+bias, GELU, $\mathrm{Linear}(128\to128)$+bias | 17 152 per layer |
| $W_{t,\delta}$ | section 3, per local layer | $5^3$ convolution, $128\to128$ channels, no bias: $128\cdot128\cdot125$ | 2 048 000 per layer |
| $B_{in}$, $B_{out}$ | section 3, shared by all local layers | $1^3$ convolutions $128\to32$ and $32\to128$, no bias | 4 096 each |
| $\Psi^{(j)}_t$ | section 3, per local layer | $J=4$ kernels, $5^3$, $32\to32$, no bias: $4\cdot32\cdot32\cdot125$ | 512 000 per layer |
| $\lambda$-MLP | section 3, per local layer | $\mathrm{Linear}(3\to32)$+bias, GELU, $\mathrm{Linear}(32\to4)$+bias | 260 per layer |
| $g_{out}$ | step 4 | as $g_{in}$, separate weights | 17 280 |
| $W_Q$ | step 4 | $\mathrm{Linear}(128\to4)$, no bias | 512 |
| $\gamma$ | defect correction, optional | one scalar, initial value 0.25 | 1 |

The transform matrices $Y$, $Y_r$ and their pseudo-inverses are *not* learned: they are
computed from the node positions of each level and stored as buffers. Nor are the
geometry features.

The four deployed experts are exactly these pieces in different combinations, and the
counts add up to the checkpoint sizes:

| | generalist | mid | high | top |
|---|---|---|---|---|
| $n_{loc}$ | 4 | 4 | 4 | 8 |
| $W_{t,\delta}$, all layers | 8 192 000 | 8 192 000 | 8 192 000 | 16 384 000 |
| $\Psi$ banks, all layers | – | 32 768 000 (full width, $128\to128$) | 2 048 000 | 4 096 000 |
| $B_{in}+B_{out}$ | – | – (no bottleneck) | 8 192 | 8 192 |
| $\lambda$-MLPs | – | 1 040 | 1 040 | 2 080 |
| $g_t$, all layers | 68 608 | 68 608 | 68 608 | 137 216 |
| $\Gamma$ + $e(\eta)$ | 141 712 | 141 712 | 141 712 | 141 712 |
| degree attention | – | – | 1 224 | 1 224 |
| $W_P$, $W_Q$, $g_{in}$, $g_{out}$ | 35 584 | 35 584 | 35 584 | 35 584 |
| **total** | **8 437 904** | **41 206 944** | **10 496 360** | **20 806 008** |

Two readings of this table. The integral term — everything that makes the model a
neural operator in the FNO sense — is 0.14 M parameters, under 2% of the smallest
expert; the local stencils are the rest. And the mid expert is four times the size of
the others only because its banks act at full width, which the bottleneck later made
unnecessary.

### Experimental: attention between viscosity patches

Not part of any deployed expert; under test (`--phys-attn 32`). It adds long-range
coupling that neither the truncated harmonics nor the stencils can express, in the
spirit of Transolver's physics attention (Wu et al., ICML 2024) but keyed on the
viscosity so that linearity in $f$ survives. Every node $i$ is assigned softly to
$M_p=32$ patches by weights $w_{im}$, a softmax over $m$ of a learned MLP
$\mathrm{Linear}(7\to64)$, GELU, $\mathrm{Linear}(64\to32)$ of seven viscosity and
geometry numbers ($\log\eta$, its $5^3$ mean and spread, and the four geometry
features). The features are averaged into one token per patch,
$s_m=\sum_i w_{im}v_i/\sum_i w_{im}$, and the same weights average the seven inputs into
a descriptor $c_m$ of what each patch physically is. Learned
$Q_p,K_p\in\mathbb R^{32\times7}$ (with biases) turn descriptors into queries and keys,
$A=\mathrm{softmax}(QK^{\top}/\sqrt{32})$ lets the patches attend to each other, and the
result is scattered back through a gate $\in\mathbb R^{128}$ initialised at zero:
$v_i\leftarrow v_i+\mathrm{gate}\odot\sum_m w_{im}(As)_m$. Since $w$, $A$ and the gate
depend on $\eta$ and geometry only, the module is a linear map of $v$; 3 232 parameters;
cost $O(P\,M_p\,d_v)$, about 0.1% of the local branch at level 6. It sits between the
last local layer and step 4.

### Projection, defect correction, experts

**Projection onto the finite-element space.** Recall that seam nodes are stored once
per diamond. The network processes each diamond's copy independently, so its raw output
generally assigns *different* values to the copies of one physical node — a function
that does not exist in the solver's finite-element space. $\Pi$ replaces every copy by
the mean over the copies of that node. This is essential, not cosmetic: a Krylov solver
can only remove error components that lie in its own space, and any part of the guess
outside it is error the solver cannot touch. Before $\Pi$ was added the solver plateaued
at 6% error no matter how many iterations it was given. The solver-side glue then sets
$u=0$ on the two Dirichlet shells.

**Defect correction.** One pass leaves a residual $r=f-K_\eta\,\mathcal G_\theta f$.
Feeding the residual through the same network and adding the correction back,

$$x=\mathcal G_\theta f+\gamma\,\mathcal G_\theta\!\left(f-K_\eta\,\mathcal G_\theta f\right)
=\left[(1+\gamma)\,\mathcal G_\theta-\gamma\,\mathcal G_\theta K_\eta\,\mathcal G_\theta\right]f,$$

is classical iterative refinement with one learned scalar $\gamma$ (initialised at
0.25). The right-hand form shows it is still linear in $f$. In training, $K_\eta$ is
evaluated by a finite-difference form of the Stokes operator on the diamond grid
(`terra_infer/stokes_residual.py`), which coincides with the true discrete operator at
interior nodes; the residual is masked to those nodes. It doubles the forward cost, adds
one parameter, and reduces the error by about 10% relative. It is enabled with
`--refine-steps 1`.

**Experts.** Four copies of the operator are trained, and one is selected per problem by
its contrast $\chi$ against the thresholds $10$, $10^2$, $10^3$. Since $\chi$ is a
function of $\eta$ alone, routing keeps the map linear in $f$. The motivation is
empirical: a problem at $\chi=2$ is nearly constant-viscosity Stokes and a problem at
$\chi=10^4$ is dominated by interfaces, they want different stencils, and one shared
weight set is measurably worse than four specialised ones — mean error 0.20 / 0.25 /
0.27 / 0.31 at levels 3–6 for a single model against 0.084 / 0.112 / 0.135 / 0.189 for
the router (see *Results*).

### As implemented: shapes and sizes

| level | $n$ | $P$ | $\ell_{\max}$ | $M$ | $k_1$ | block size $b_sk_1$ |
|---|---|---|---|---|---|---|
| 3 | 9 | 7 290 | 12 | 169 | 9 | 144 |
| 4 | 17 | 49 130 | 24 | 625 | 17 | 272 |
| 5 | 33 | 359 370 | 32 | 1 089 | 17 | 272 |
| 6 | 65 | 2 746 250 | 32 | 1 089 | 17 | 272 |

The tensor pipeline, with `B` the batch size and `S = 10` diamonds:

- **Lift and gate.** Input `(B, S, n, n, n, 5)`; the log-viscosity channel is split off
  and the other four go through `Linear(4 -> 128)`; multiplied by `gate_in`, a
  `5 -> 128 -> 128` MLP of geometry and log-viscosity.
- **Integral term** (kept in fp32 even under bf16 autocast, since the transforms are
  ill-conditioned in half precision). Reshape to `(B, 128, S n², n)`; apply $Y^{+}$ and
  $Y_r^{+}$ to get `(B, 128, M, k_1)`; regroup channels into `(B, 8, M, 16 k_1)`; for
  each degree $\ell$ multiply the group's block by $G_\ell$, which `ggen`
  (`19 -> 64 -> 64 -> 2048`) has produced from the normalised indices and the
  16-dimensional embedding; apply $Y_r$ then $Y$; add to the features. The degree-mixing
  attention, where present, forms `18 -> 32` queries and keys per harmonic and applies
  the resulting `(B, M, M)` matrix to the coefficients before synthesis.
- **Local layers.** Reshape to `(S B, 128, n, n, n)`. Each layer: multiply by a geometry
  gate, apply the dense bias-free `5³` convolution `128 -> 128`, add the bank (`1³`
  `128 -> 32`, four `5³` kernels `32 -> 32`, softmax mix from a `3 -> 32 -> 4` MLP, `1³`
  `32 -> 128`), add to the features. No activation.
- **Head.** Multiply by `gate_out`, `Linear(128 -> 4)`, then the seam-mean projection
  $\Pi$.

The four deployed experts:

| expert | contrast band $\chi$ | $n_{loc}$ | stencil banks $J$ | bank width | degree attention | parameters |
|---|---|---|---|---|---|---|
| generalist | $<10$ | 4 | none | – | no | 8.44 M |
| mid | $10$ – $10^2$ | 4 | 4 | 128 (no bottleneck) | no | 41.21 M |
| high | $10^2$ – $10^3$ | 4 | 4 | 32 | yes | 10.50 M |
| top | $>10^3$ | **8** | 4 | 32 | yes | 20.81 M |

80.96 M parameters in total. The generalist's local layers are the dense $W_{t,\delta}$
alone. The mid expert's banks act at full width, which is why it is four times larger;
it predates the bottleneck and was not retrained. The top
expert has twice the local depth (`--linear-convs 8`, hence its checkpoint name
`c8ckpt`) and therefore reaches 16 nodes instead of 8.

### Discretisation convergence: what holds and what does not

A model is discretisation convergent when one set of weights acts on any discretisation
and the outputs converge to a single continuum operator as the mesh is refined. The two
terms of this model behave oppositely, and the split is the central caveat of this work.

**The integral term converges.** Its weights are indexed by channel, degree and radial
mode, and they are the values of the smooth function $\Gamma$ at normalised indices —
not a table tied to nodes. Refining the mesh rebuilds only the transform matrices
$Y,Y_r$ (which depend on node positions) and evaluates $\Gamma$ at more modes. Nothing
learned is resolution-specific.

**The local term does not.** Its offsets are counted in nodes, so its physical reach is
$\rho=2n_{loc}dh$, and $h$ halves with every level. With $d=1$:

| level | $h$ in units of the shell thickness, $1/(n-1)$ | reach of 4 layers, $8h$ | fraction of the thickness |
|---|---|---|---|
| 3 | 0.125 | 8 nodes = 1.0 | all of it |
| 4 | 0.0625 | 0.5 | half |
| 5 | 0.031 | 0.25 | a quarter |
| 6 | 0.016 | 0.125 | an eighth |

In the limit $h\to0$ the term degenerates to a pointwise map — the generic template's
$\mathcal W_t$ — which is a *different* operator from the one that was trained at level
3. The model does not converge to anything as the mesh is refined; it changes character.
The option `--dilated-stencils` sets $d=(n-1)/8$, i.e. $d=1,2,4,8$ at levels 3–6, so that
$\rho$ stays at the level-3 value. It costs no weights.

**A second, milder violation.** The truncation is held at $\ell_{\max}=32$ from level 5
upward while the mesh keeps refining, so the harmonics span 21% of the lateral degrees
of freedom at levels 3–4, 10% at level 5, and 2.6% at level 6. The integral term still
converges — to the operator truncated at $\ell_{\max}=32$ — but that operator represents
less and less of the solution.

**Measured.** Both effects predict that the error grows with level, and it does (see
*Results*). Isolating the first: models trained with dilated stencils have a held-out
error ratio between level 6 and level 3 of **1.0–1.2**, against **1.9–2.8** for
index-space stencils, consistently across three experiments with different training
histories. Dilation therefore removes the level dependence. It does not, so far, improve
the absolute error: at coarse levels dilated models are about twice as inaccurate as
index-space ones, and at level 6 the best of each are tied. Whether dilation wins at
level 7, where the index-space model would degrade further, has not been tested.
Neither option is in the deployed experts.

**A caveat that the results inherit.** The two experts that serve $\chi<10^2$ were
trained on levels 3–5 only and have never seen a level-6 field. They handle two thirds
of the level-6 test set, so the level-6 column of every table below is largely a
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

Held-out relative $L^2$ velocity error $\|u_{pred}-u\|/\|u\|$ on the general test sets
(viscosity contrast 1–$10^4$, 27–32 samples per level), four-expert router:

| level | best | median | **mean** | worst | pressure |
|---|---|---|---|---|---|
| L3 (9^3) | 0.030 | 0.069 | **0.084** | 0.316 | 0.040 |
| L4 (17^3) | 0.027 | 0.097 | **0.112** | 0.349 | 0.030 |
| L5 (33^3) | 0.035 | 0.120 | **0.135** | 0.389 | 0.034 |
| L6 (65^3) | 0.075 | 0.156 | **0.189** | 0.564 | 0.050 |

A single model without the router scores 0.203 / 0.245 / 0.267 / 0.312 at levels 3–6. The distribution is
skewed: medians sit well below means and a few high-contrast problems set the worst
case (the level-5 worst case is a narrow low-viscosity channel cutting across the shell,
exactly the structure that neither a truncated harmonic expansion nor an 8-node stencil
can represent).

As a solver initial guess. Eight held-out problems per level are solved by the production
solver (FGMRES preconditioned by geometric multigrid with a Schur-complement treatment of
the pressure) starting from zero ("cold") and from the prediction ("warm"). Each cell is
the median, over the problems that reach the target at all within 120 iterations, of the
iteration at which the velocity error first drops below the target; the bracket is the
share of the eight problems that reach it:

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
at L6): applying $K_\eta$ amplifies grid-scale error by $h^{-2}$. Tolerances must be
relative to $\|f\|$, never to the initial residual $\|r_0\|$, and a residual term in the loss is the obvious
untried lever.

A forward pass costs 11 / 40 / 191 / 2000 ms at L3–L6 on one PVC tile, against
66 / 83 / 110 / 162 ms per solver iteration.

## Known limits

- Error grows with refinement for the reasons in *Discretisation convergence*; the L6 data is thin (3.7k training samples against 26k at L5, and 27 test
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
