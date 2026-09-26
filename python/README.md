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
  Stokes operator used in the loss, the mesh's symmetry group
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
2. **Discretisation convergence.** Every learned component of the operator is defined
   on the continuum; what still limits transfer to finer meshes is measured and documented.
3. **Enough accuracy to save solver time.** Around 10% relative error turns out to be
   worth a factor of two in iterations; below that the returns diminish.


## The operator

The true solution operator $\mathcal G^\dagger(f,\eta)=K_\eta^{-1}f$ has two properties
that every decision below traces back to.

### Two facts that shape the design

1. **It is linear in $f$ and nonlinear only in $\eta$.** Doubling the load doubles the
   flow; changing the viscosity changes the operator itself.
2. **It is non-local.** $K_\eta^{-1}$ has a dense Green's function: a load anywhere
   moves fluid everywhere, with an influence that decays only algebraically with
   distance. No stencil of fixed width can represent it.

### How the design was arrived at

Every component below is in the model because removing it raises the held-out error
by at least 10 % at the levels trained on, or breaks the transfer to finer meshes;
everything that failed that test in the ablation campaign that produced this design
was removed. The model has 180 528 parameters.

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
| $M_p$ | number of viscosity patches | 32 |
| $\Pi$ | projection onto the finite-element space, defined in *Projection* below | — |
| $\odot$ | product channel by channel at each node | — |

$K_\eta$ (roman) is always the discrete Stokes matrix and $\mathcal K$ (script) always
the network's integral term.

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

**Step 3 — attention between viscosity patches, applied once.** The second coupling,
between regions of the shell in the same viscosity state; section 3 defines
$\mathcal A(\eta)$:

$$v\ \leftarrow\ v+\gamma\odot\mathcal A(\eta)\,v .$$

**Step 4 — project.** Rescale by an output gate $g_{out}$ of exactly the same form and
inputs as $g_{in}$ (separate weights), map the $d_v$ features back to the four
components of $x=(u,p)$ with a learned bias-free $W_Q\in\mathbb R^{4\times d_v}$, and
project onto the finite-element space:

$$x\ \leftarrow\ \Pi\left(W_Q\left(g_{out}(\eta)\odot v\right)\right).$$

Compared with the template, three things are different, all on purpose.

- **There is no local term at all.** The template's $\mathcal W_t$ — a stencil or a
  pointwise map on the grid — is absent. Both coupling terms are defined on the
  continuum: one in the shell's spectral basis, one over patches of nodes grouped by
  their physical state. This is what makes the operator usable on meshes it was never
  trained on (see *Discretisation convergence*).
- **No bias.** $W_P$, $W_Q$ and every layer on the forcing path have none.
- **No nonlinearity on the path $f$ takes.** The only nonlinear functions in the model
  are the gates $g$, the kernel generator $\Gamma$ (section 2) and the patch assignment
  and attention weights (section 3) — and all of them are functions of $\eta$ and
  geometry, never of $f$. Section 4 states what this buys.

### 1. The integral term acts in the shell's own basis

FNO makes the integral cheap by assuming a translation-invariant kernel
$\kappa(x,y)=\kappa(x-y)$ on a periodic box: the integral is then a convolution, the
FFT diagonalises it, and the model learns one weight matrix per Fourier mode up to a
truncation. A spherical shell has no translations, but it has rotations about its
centre, and the basis adapted to those is spherical harmonics laterally and Chebyshev
polynomials radially. This subsection defines that expansion; the next says what the
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
would be the textbook choice for analysis; it was not used because the area weights of
this mesh's stored node set, with seam nodes present in two or more diamonds, would have
to be computed and corrected for the duplicates. The least-squares fit is not immune to
the duplicates either — a repeated row counts twice — but the copies of a seam node carry
the same value, so the extra weight distorts the fit mildly and never makes it
inconsistent. Both matrices depend only on the mesh and are built once per level; they
are not learned. (Doubling the truncation was tested and cut: it costs 4x and is worse;
at $\ell_{\max}=64$ on the level-5 node count the least-squares analysis is
ill-conditioned.)

*What the operator does to the coefficients.* A kernel that is invariant under rotations
of the sphere cannot couple a harmonic $(\ell,m)$ to any $(\ell',m')$ with
$\ell'\ne\ell$, and must act identically on all $2\ell+1$ orders $m$ of one degree — this
is the spherical analogue of "a convolution is diagonal in Fourier space". The Stokes
inverse has this invariance whenever $\eta$ depends on $r$ only. For a single channel,
the integral term then reduces to one small dense matrix per degree, acting on the
$k_1$ radial coefficients of every harmonic of that degree:

$$\widehat{(\mathcal K v)}_{\ell mk}=\sum_{k'=0}^{k_{\max}}\left[G_\ell\right]_{kk'}\,\hat v_{\ell mk'},
\qquad G_\ell\in\mathbb R^{k_1\times k_1}\ \text{(single-channel form)}.$$

Note what the indices say: the output at $(\ell,m,k)$ depends only on inputs at the
same $(\ell,m)$, and $G_\ell$ carries no $m$. These matrices are the counterpart of
FNO's per-mode weights, with the spherical-harmonic transform standing in for the FFT —
the same substitution the spherical FNO makes for weather models.

**How far the derivation actually carries.** The block-diagonal form is a theorem for a
rotation-invariant operator acting on *scalar* fields. The field that enters the
transform here is not that: the channels are learned features that start as pointwise
mixtures of the *Cartesian* components of $f_u$ — which rotate as a vector, so their
scalar-harmonic expansions couple degree $\ell$ to $\ell\pm1$ — and they have been
multiplied pointwise by an $\eta$-dependent gate, which breaks rotation invariance
outright. So for the model as built, per-degree mixing is an **inductive bias**, chosen
because the layered operator has this structure and because the spherical FNO shows it
works when applied channel-wise to arbitrary features; it is not an exact symmetry of
the network, and "Green's function" below is a name for the learned blocks, not a claim
that they equal one. What is solid is the parameterisation: the blocks are indexed by
$(\ell,k,k')$ and not by nodes, which is what makes this term reusable across levels.

*Channels.* The channels are not treated independently, because the $d_v$ features are
not physical components but learned ones, and mixing them is where the operator's
expressiveness lives. The $d_v$ channels are split into $n_b=8$ groups of $b_s=16$, and
within a group the matrix mixes channels and radial modes together. Writing
$\hat v^{(q)}_{c,\ell mk}$ for the coefficient of channel $c$ of group $q$, the form
actually used is

$$\widehat{(\mathcal K v)}^{(q)}_{c,\ell mk}
=\sum_{c'=1}^{b_s}\ \sum_{k'=0}^{k_{\max}}
\left[G_\ell\right]^{(q)}_{(c,k),(c',k')}\,\hat v^{(q)}_{c',\ell mk'},
\qquad G^{(q)}_\ell\in\mathbb R^{(b_sk_1)\times(b_sk_1)},$$

so for each degree $\ell$ there are $n_b$ matrices, one per group, each acting on the
group's stacked (channel, radial-mode) vector of length $b_sk_1$ — 144 at level 3, 272
from level 4 on. That block size is called `tok` in the code. Groups do not exchange
information inside $\mathcal K$; they do in the patch attention, which mixes all 128
channels.

### 2. The kernel is generated from the viscosity, not stored

The template allows the kernel to depend on the parameter, $\kappa(x,y,a(x),a(y))$. If
every $G_\ell^{(q)}$ were a free matrix, the model would learn one Green's function for
the *average* viscosity of the training set and could not respond to the viscosity of
the problem in front of it. So the matrices are *generated* from $\eta$ by a small
network, once per sample. (A fixed table was tested: 43 % worse at the trained levels,
and it diverges — error $10^8$ — when queried on a finer mesh.)

*Summarising the viscosity.* For each of the $n$ radial layers, take the mean and the
standard deviation of $\log\eta$ over that spherical surface (all $10n^2$ lateral nodes
of the layer). That gives two profiles of length $n$; each is resampled to 16 radii by
linear interpolation, so that the summary has the same size at every level. The 32
numbers pass through a learned MLP, $\mathrm{Linear}(32\to64)$, GELU,
$\mathrm{Linear}(64\to16)$, giving the embedding $e(\eta)\in\mathbb R^{16}$. Its last
layer is initialised with small weights, so at the start of training every sample gets
nearly the same kernel and the dependence on $\eta$ is learned gradually rather than
imposed by a random initialisation.

*Generating the entries.* A learned network $\Gamma$ — $\mathrm{Linear}(19\to64)$, GELU,
$\mathrm{Linear}(64\to64)$, GELU, $\mathrm{Linear}(64\to2048)$ — takes $3+16=19$ inputs:
three scaled indices and the embedding. It is evaluated for every index triple
$(\ell,k,k')$ with $0\le\ell\le\ell_{\max}$ and $0\le k,k'\le k_{\max}$, i.e.
$(\ell_{\max}+1)\,k_1^2$ times per sample, and each evaluation returns
$n_bb_s^2=8\cdot16^2=2048$ numbers, which are read as one $b_s\times b_s$
channel-mixing block per group:

$$\left[G_\ell\right]^{(q)}_{(c,k),(c',k')}
=\Gamma\!\left(\frac{\ell}{16},\ \frac{k}{8},\ \frac{k'}{8}\ ;\ e(\eta)\right)^{(q)}_{cc'}.$$

The three arguments before the semicolon are the indices, divided by fixed constants
(not by the current truncation) so that a given mode means the same thing at every
level; the argument after it is the conditioning; the
superscript and subscripts on the right pick one of the 2048 outputs. Assembling the
outputs over all $(k,k')$ for a fixed $\ell$ and $q$ fills the $(b_sk_1)\times(b_sk_1)$
matrix $G^{(q)}_\ell$ of section 1. Everything the integral term learns is in $\Gamma$
and $e$: 138 560 + 3 152 parameters — 78 % of the whole model.

Two consequences follow. Every sample gets its own Green's function, conditioned on its
own viscosity profile. And the learned object is a smooth function of *continuous*
indices rather than a table with one entry per grid point, so a finer mesh — which
allows a larger $\ell_{\max}$ and $k_{\max}$ — simply evaluates $\Gamma$ at more points.
This is exactly what makes the integral term discretisation convergent: raising the
truncation only asks $\Gamma$ for modes it has not been trained on, at arguments beyond
the trained range ($\ell/16>2$, $k/8>2$), which is extrapolation, not a different
function. That is one reason the truncation stops growing at level 5.

*What the kernel does not see.* $e(\eta)$ carries only the radial profile of the
viscosity — nothing about *where* laterally a slab or a channel sits. Giving the
generator lateral information (the harmonic content of $\log\eta$, or the patch
descriptors of section 3) overfits; the lateral structure is handled by the next term
instead.

### What if the viscosity is not layered?

Section 1 derived the block-diagonal form for $\eta=\eta(r)$. A slab or a plume is a
lateral variation, and for $\eta=\eta(r,\theta,\varphi)$ the inverse is *not*
block-diagonal: a load in one harmonic drives flow in every harmonic, with a dense
coupling that depends on the shape of $\eta$. This is the central difficulty of the
problem, and the architecture is organised around the way the exact operator decomposes.

Split the viscosity into its radial mean and the lateral anomaly,
$\eta=\eta_0(r)+\delta\eta(x)$, so that $K_\eta=K_0+\delta K$ with $K_0$ the layered
operator and $\delta K=-\nabla\cdot\left(2\,\delta\eta\,\varepsilon(\cdot)\right)$, which
is *local*: it only differentiates and multiplies by $\delta\eta$. Then

$$K_\eta^{-1}=K_0^{-1}-K_0^{-1}\,\delta K\,K_0^{-1}+K_0^{-1}\,\delta K\,K_0^{-1}\,\delta K\,K_0^{-1}-\cdots$$

The exact inverse for laterally varying viscosity alternates the layered *global*
operator with corrections weighted by the anomaly. This is the **motivation** for
pairing the integral term with a second, anomaly-aware term — it says what kind of
operators must be composed — and not a derivation of the network: no layer of the model
is one of these terms.

- **$K_0^{-1}$ corresponds to the integral term.** $G_\ell$ is generated from the radial
  mean and spread of $\log\eta$, so it is conditioned on the layered part of the
  viscosity and, by construction, sees nothing of the lateral anomaly.
- **The anomaly correction corresponds to the patch attention** of section 3, whose
  every weight is a function of the viscosity anomaly at the nodes.

The series converges only for $\|K_0^{-1}\delta K\|<1$, roughly modest contrast. At
$\chi=10^4$ it does not; the interfaces dominate, and that regime sets the worst cases
in the results below.

### 3. The second term: attention between viscosity patches

*Why not a local term.* The template's $\mathcal W_t$, realised on a mesh, is a stencil:
a weighted sum over a node's neighbours in index space. Its reach is counted in nodes,
so its physical footprint halves with every refinement level, and the operator it
computes at level 6 is a different one from the operator trained at level 3. That is
the one component of the template that cannot be discretisation convergent, and it is
the one this model does without.

*What is used instead.* The idea is Transolver's physics attention (Wu et al., ICML 2024):
mesh points that share a *physical state* should exchange information regardless of
where they are, so instead of attention between the $P$ nodes (cost $P^2$), attend
between a few *patches* of nodes in the same state. The physical state here is the
viscosity. The one change from Transolver is that the assignment and the attention are
computed from $\eta$ alone, so the whole term stays a linear map of the features and
the operator stays exactly linear in $f$.

*Step (a) — assign nodes to patches.* For every node $x_i$ take seven numbers that
depend only on $\eta$ and geometry: the standardised $\log\eta$ at the node, its mean and
standard deviation over the $5^3$ window of nodes around it (the spread marks an
interface), and the four geometry features. A learned MLP,
$\mathrm{Linear}(7\to64)$, GELU, $\mathrm{Linear}(64\to M_p)$, gives $M_p=32$ scores,
and a softmax over patches turns them into weights

$$w_{im}\ \ge 0,\qquad \sum_{m=1}^{M_p} w_{im}=1 .$$

So node $i$ belongs *softly* to the $M_p$ patches. Nodes deep inside stiff material get
similar weight vectors wherever they are on the shell; nodes at an interface get their
own patches. The patches have whatever shape the viscosity field has — they are not
geometric blocks.

*Step (b) — pool features into one token per patch.* With $v_i\in\mathbb R^{d_v}$ the
feature vector at node $i$,

$$s_m=\frac{\sum_i w_{im}\,v_i}{\sum_i w_{im}}\qquad\text{(linear in } v\text{)},$$

and, with the same weights, a descriptor $c_m\in\mathbb R^7$ of what each patch
physically is — the weighted mean of the seven inputs.

*Step (c) — let the patches attend to each other.* Learned $Q_p,K_p\in\mathbb R^{32\times7}$
(with biases) map the descriptors to queries and keys, and

$$A_{mn}=\mathrm{softmax}_n\!\left(\frac{Q_pc_m\cdot K_pc_n}{\sqrt{32}}\right),\qquad
\tilde s_m=\sum_{n=1}^{M_p} A_{mn}\,s_n .$$

"Patch $m$ attends to patch $n$" means "material in state $m$ receives influence from
material in state $n$", and how much is learned as a function of the two states — a
weak channel next to a stiff slab, for instance. Because $A$ is built from the
descriptors and never from $v$, this step is still a linear map of $v$.

*Step (d) — scatter back.* Each node receives the updated tokens of the patches it
belongs to, through a learned channel gate $\gamma\in\mathbb R^{d_v}$ initialised at zero:

$$(\mathcal A v)_i=\sum_m w_{im}\,\tilde s_m,\qquad v_i\ \leftarrow\ v_i+\gamma\odot(\mathcal A v)_i .$$

*Properties.* Every weight in the term — $w$, $A$, $\gamma$ — is a function of $\eta$
and geometry, so for a fixed viscosity field the term is linear in $v$. The assignment
is pointwise and the descriptors are physical, so the same weights act on any mesh. The
cost is $O(P\,M_p\,d_v)$, about 44 GFLOP at level 6, and the term has 3 232
parameters. Enlarging it — 64 or 128 patches, two or three layers in sequence, a wider
attention — does not lower the error at the trained levels and slightly raises it at the
levels above them, so it stays at 32 patches and one layer.

### 4. No nonlinearity on the forcing path, and what it guarantees

A generic neural operator applies $\sigma$ after every layer; that is what gives it
universal approximation. We omit it, together with every bias, wherever $f$ flows. All
the nonlinearity lives in functions of $\eta$ and geometry — $g_{in},g_{out}$, $\Gamma$,
the patch assignment $w$ and the attention $A$ — which enter only by multiplying $v$ or
by setting the weights that multiply $v$. Consequently, for every viscosity field and at
every resolution, the network satisfies exactly

$$\mathcal G_\theta(\alpha f_1+\beta f_2,\ \eta)=\alpha\,\mathcal G_\theta(f_1,\eta)+\beta\,\mathcal G_\theta(f_2,\eta),
\qquad\text{and in particular}\qquad\mathcal G_\theta(0,\eta)=0.$$

`scripts/check_linearity.py` verifies this numerically on a random model and mesh; the
relative deviation is $10^{-6}$, i.e. floating-point rounding. It is a strictly smaller
hypothesis class than the universal one, and that is the point: it is the class the
true operator belongs to. The network cannot waste capacity learning that small loads
behave like large ones, cannot produce spurious flow from zero forcing, and cannot drift
out of the class during training. The `--nonlin` flag inserts a GELU after the coupling
terms and gives all of this up; it is off.

### Every learned parameter

| symbol | acts in | exact form | parameters |
|---|---|---|---|
| $W_P$ | step 1 | $\mathrm{Linear}(4\to128)$, no bias | 512 |
| $g_{in}$ | step 1 | $\mathrm{Linear}(5\to128)$+bias, GELU, $\mathrm{Linear}(128\to128)$+bias | 17 280 |
| $e(\eta)$ | section 2 | $\mathrm{Linear}(32\to64)$+bias, GELU, $\mathrm{Linear}(64\to16)$+bias | 3 152 |
| $\Gamma$ | section 2 | $\mathrm{Linear}(19\to64)$+bias, GELU, $\mathrm{Linear}(64\to64)$+bias, GELU, $\mathrm{Linear}(64\to2048)$+bias | 138 560 |
| patch assignment | section 3 (a) | $\mathrm{Linear}(7\to64)$+bias, GELU, $\mathrm{Linear}(64\to32)$+bias | 2 592 |
| $Q_p$, $K_p$ | section 3 (c) | $\mathrm{Linear}(7\to32)$+bias, twice | 512 |
| $\gamma$ | section 3 (d) | $\in\mathbb R^{128}$, initialised at zero | 128 |
| $g_{out}$ | step 4 | as $g_{in}$, separate weights | 17 280 |
| $W_Q$ | step 4 | $\mathrm{Linear}(128\to4)$, no bias | 512 |
| **total** | | | **180 528** |

"Bias" means the layer has an additive bias; every layer on the path the forcing takes
has none, and every layer with a bias acts only on viscosity and geometry — that split
is what section 4 relies on. The transform matrices $Y$, $Y_r$ and their pseudo-inverses
are *not* learned: they are computed from the node positions of each level and stored
as buffers. Nor are the geometry features. The integral term (the last two rows of
section 2) is 78 % of the model; the patch attention 1.8 %.

### Projection

Recall that seam nodes are stored once per diamond. The network processes each
diamond's copy independently, so its raw output generally assigns *different* values to
the copies of one physical node — a function that does not exist in the solver's
finite-element space. $\Pi$ replaces every copy by the mean over the copies of that
node. This is essential, not cosmetic: a Krylov solver can only remove error components
that lie in its own space, and any part of the guess outside it is error the solver
cannot touch. Without $\Pi$ the solver plateaus at 6 % error no matter how many
iterations it is given. The solver-side glue then sets $u=0$ on the two Dirichlet
shells.

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
  16-dimensional embedding; apply $Y_r$ then $Y$; add to the features.
- **Patch attention.** Flatten to `(B, P, 128)`; viscosity features `(B, P, 7)`;
  assignment `(B, P, 32)` by softmax; tokens and descriptors `(B, 32, 128)` and
  `(B, 32, 7)` by weighted means; attention `(B, 32, 32)`; scatter back and add through
  the channel gate. No activation.
- **Head.** Multiply by `gate_out`, `Linear(128 -> 4)`, then the seam-mean projection
  $\Pi$.

Cost per forward pass at level 6, by operation count: ~1.5 TFLOP for the two spectral
transforms, ~0.04 TFLOP for the patch attention.

### Discretisation convergence: what holds and what does not

A model is discretisation convergent when one set of weights acts on any discretisation
and the outputs converge to a single continuum operator as the mesh is refined.

**Both terms of this operator are defined on the continuum.** The integral term's
weights are values of $\Gamma$ at normalised indices, not a table tied to nodes;
refining the mesh rebuilds only the transform matrices $Y,Y_r$ and evaluates $\Gamma$ at
more modes. The patch term assigns each node by its own viscosity state and geometry,
and its learned weights act on patch tokens, so nothing in it depends on the node count.

**Measured.** Trained on levels 3–5 with level 6 never seen, the model's error is
0.172 / 0.239 / 0.231 / 0.211: it does not grow beyond the training data at all, and
level 6 scores below level 4. Trained on levels 3–4 only, it is
0.215 / 0.321 / 0.384 / 0.434, so two refinement levels beyond the training data cost a
factor of 1.35 over the finest trained level. Training through level 5 removes that.

**What still limits it.** The truncation is held at $\ell_{\max}=32$ from level 5
upward while the mesh keeps refining, so the harmonics span 21 % of the lateral degrees
of freedom at levels 3–4, 10 % at level 5, and 2.6 % at level 6; the term converges to
the operator truncated at 32, which represents less of the solution as the mesh grows.
Raising the truncation is not the answer (section 1); the patch term is what carries the
sub-truncation structure. And the patch statistics use a $5^3$ window in nodes rather
than a fixed physical size; a physical-size window was tested and is no better.

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

Datasets are large (an L6 sample is 97 MB) and regenerable — put them on scratch. The
high-contrast sets used in the training recipe are generated with `--contrast-min 100`.

To check the generator against TERRA itself:

    python -c "from terra_data.stokes_symbolic import validate_against_terra_testcase as v; v()"
    python -c "from terra_infer.stokes_residual import validate; validate()"

### 2. Train

The recipe behind the current checkpoint (`w3d_linear_lad_xgen.pt`): levels 3–5,
general-contrast sets only, level 6 held out. One process per GPU tile, four tiles,
gradients averaged through gloo on the host:

    for r in 0 1 2 3; do
      WORLD_SIZE=4 RANK=$r ZE_AFFINITY_MASK=$r MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
      python -m terra_infer.train_linear_mr \
        --hidden 128 --heads 8 --linear-convs 0 --eta-gates --eta-green --phys-attn 32 \
        --data $ML/stokes_L3_d8 --data2 $ML/stokes_L4_d8 --data3 $ML/stokes_L5_d8ok \
        --lmax 12 --kmax 8 --lmax2 24 --kmax2 16 --lmax3 32 --kmax3 16 \
        --batch-size 32 --batch-size2 16 --batch-size3 4 --batch-mix \
        --epochs 120 --lr 6e-3 --h1-weight 0.5 --seam-average \
        --amp --grad-checkpoint --max-test 32 --device xpu \
        --out $ML/w3d_linear_xgen.pt &
    done; wait

Training on the general-contrast sets alone beats adding the high-contrast sets at
every level, although the general sets are a third of the data: the high-contrast sets
double the error below contrast 10 and buy ground only above contrast 1000. The
`--extra-data path:lmax:kmax:batch[:max_train]` option adds a dataset; `--batch-mix`
interleaves shuffled batches from all datasets so the weights never see a per-epoch
level seesaw. The learning rate follows a one-cycle schedule over the whole run. The
loss is relative L2 on velocity and mean-free pressure plus an `H^1` gradient term
(`--h1-weight`), restricted to nodes where the finite-difference stencil is the true
operator; velocity targets are scaled by the sample's geometric-mean viscosity.
Augmentation is over the mesh's exact symmetry group. `--spectral-layers N` stacks
several spectral cores (section 2) with viscosity gates between them; `--seed` fixes
the initialisation.

Practicalities that matter on PVC:

- The assembled cache (`_asm_*` next to each dataset) is read into RAM when it fits
  (`TERRA_CACHE_RAM_GB`, default 96); memory-mapped from the parallel filesystem the
  per-batch gather stalls on page faults. Each rank stages its own copy, so the recipe
  above needs ~95 GiB per rank and four ranks per 512 GB node. Build the caches once
  in a single process before starting several jobs on the same datasets.
- An epoch of the recipe above takes ~65 s on four tiles. 120 epochs is not the
  optimum: the held-out error was still falling when the schedule ended, and 60 epochs
  is far short of it (0.26 against 0.19 at level 3).
- The host all-reduce costs ~1.5 s per step, so the batch has to grow with the rank
  count for the ranks to pay off.
- Learning rate: the optimum is near `6e-3`; `2e-3` is 5–12 % worse at every level.
- The trainer resumes from `<out>.resume` if it exists — use a fresh `--out` per run.
- The SNG-2 `test` partition shares GPUs between jobs; use `general` for any GPU work.

Every architecture switch is recorded in the checkpoint (`hidden`, `heads`,
`spherical`, `radial_modes`, `linear_convs`, `linear_eta_gates`, `linear_eta_green`,
`linear_phys_attn`, `linear_phys_attn_dim`, `linear_phys_attn_layers`,
`linear_spectral_layers`, `log_eta_mean`, `log_eta_std`, `test_rel_l2`), and
`load_state` tolerates keys the model no longer has.

### 3. Evaluate

Held-out error per level on the general-contrast test sets:

    EVAL_DEVICE=xpu EVAL_NTEST=32 EVAL_NTEST_L6=27 \
      python scripts/eval_mr_levels.py $ML/w3d_linear_xgen.pt

reports mean / best / median / worst relative L2 for velocity and the pressure error at
each level (a comma-separated list of checkpoints is scored as a set routed by viscosity
contrast, with `EVAL_MOE_THRESH` giving the band thresholds; without it a list of
checkpoints is averaged instead). `EVAL_BINS=1` adds a breakdown by viscosity contrast.
On the GPU with sample prefetching this takes about a minute; on CPU 25 minutes. `EVAL_LM_L6=48 EVAL_KM_L6=24` overrides the truncation at a level;
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
SYCL stack inside the app process is not safe). The glue builds the operator from the
switches recorded in the checkpoint, including the patch attention and the number of
spectral layers; this operator has not yet been exercised inside the app.

## Results

Held-out relative $L^2$ velocity error $\|u_{pred}-u\|/\|u\|$ on the general test sets
(viscosity contrast 1–$10^4$, 27–32 samples per level), one model, trained on levels
3–5, level 6 never seen:

| level | best | median | **mean** | worst | pressure |
|---|---|---|---|---|---|
| L3 (9^3) | 0.025 | 0.157 | **0.172** | 0.587 | 0.049 |
| L4 (17^3) | 0.032 | 0.181 | **0.239** | 1.007 | 0.035 |
| L5 (33^3) | 0.034 | 0.179 | **0.231** | 0.966 | 0.035 |
| L6 (65^3, unseen) | 0.038 | 0.172 | **0.211** | 0.955 | 0.032 |

The error does not grow with refinement above level 4: the two levels the model was
never trained on score at or below the finest trained level.

The mean is set almost entirely by high viscosity contrast. Splitting the same test
sets by $\chi=\max\eta/\min\eta$:

| contrast band | samples | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|
| 1 – 10 | 12 | 0.053 | 0.065 | 0.067 | 0.076 |
| 10 – 100 | 8 | 0.169 | 0.205 | 0.198 | 0.201 |
| 100 – 1000 | 3 | 0.251 | 0.333 | 0.330 | 0.292 |
| above 1000 | 9 | 0.308 | 0.470 | 0.445 | 0.408 |

At low contrast the operator is already well inside the useful range. What sets the
mean is the top two bands, where the solution is controlled by narrow weak channels and
sharp interfaces — structure that the harmonic truncation cannot represent and that the
patch term alone has to carry. That is where the remaining work is.

The solver benchmark (pipeline step 4: cold start against the prediction as initial
guess, iterations to reach a given velocity error) has not yet been run with this
operator.

## Known limits

- The mean error is 0.17–0.24 and is set by the two highest contrast bands, where it
  reaches 0.31–0.47. The target for a useful initial guess is 0.1 at every level, which
  needs roughly a factor of two in those bands. The held-out error was still falling at
  the end of the 120-epoch schedule, the model has a single spectral layer, and the
  lateral structure of the viscosity reaches the operator only through the 3 232
  parameters of the patch term, so longer schedules, stacked spectral cores
  (`--spectral-layers`), more patches and coupling across harmonic degrees are the open
  levers.
- Training data is thin above level 3: 10 000 general-contrast samples at level 3
  against 1 000 at level 4 and 737 at level 5, which is the likeliest reason level 3 is
  the most accurate level at the same relative truncation.
- Contrast-routed experts, which gave the previous design a factor of 2.4, do not work
  for this operator: the kernel generator is conditioned on viscosity statistics, so an
  expert trained on a narrow contrast band extrapolates badly outside it and is worse
  than a general model even inside it.
- Error grows with refinement for the reasons in *Discretisation convergence*: the
  truncation stops at $\ell_{\max}=32$ and the patch statistics use a window in nodes.
- As a **preconditioner** inside FGMRES, every learned operator tried — several
  architectures and training objectives, including unrolled exact-operator rollouts and
  direct spectral-radius minimisation — plateaus at variable viscosity (residual
  0.06–0.8 with error above 1). Root cause: the near-null modes of high-contrast Stokes
  have negligible residual signature, which is exactly the signal a preconditioner is
  fed. The initial-guess role is what works.
- The operator has not yet been run inside the simulation; the solver-side glue is
  updated for it but untested.
