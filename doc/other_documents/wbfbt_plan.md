# w-BFBT Schur Complement Preconditioner — Implementation Plan for terraneo

Reference: Rudi, Stadler, Ghattas, "Weighted BFBT Preconditioner for Stokes Flow Problems
with Highly Heterogeneous Viscosity", SIAM J. Sci. Comput. 39(5), 2017
(arXiv:1607.03936).

## Motivation

A7 (FK rmu=1e5) on MT256 with our current solver stack fails to converge: relative
residual pinned at 1.000 through all 10 outer FGMRES iterations, at all timesteps
tested (TS 1 through TS 229). The bottleneck is the Schur complement approximation
S^-1 ≈ M_p(1/μ) (inverse-viscosity-weighted lumped pressure mass matrix), which the
Rudi paper documents as the canonical failure mode at high viscosity contrast.

w-BFBT replaces M_p(1/μ) with a viscosity-weighted BFBT approximation that the paper
demonstrates is robust up to viscosity contrasts of 1e10. At DR=1e8 with 28 sinkers,
M_p(1/μ) needs ~600 GMRES iterations where w-BFBT needs ~100 (both for 1e-6 residual
reduction).

## What the paper actually says

w-BFBT formula (eq. 7):

    S^-1 ≈ (B Cwl^-1 B^T)^-1 . (B Cwl^-1 A Dwr^-1 B^T) . (B Dwr^-1 B^T)^-1

with weights

    Cwl = Dwr = M_u_lumped(sqrt(μ))

i.e., LUMPED velocity mass matrices weighted by sqrt(μ). When w_l = w_r the two
pressure-Poisson-like operators are identical, so we only build one. The paper
provides spectral-equivalence eigenvalue bounds in inf-dim and demonstrates near-
mesh-independent convergence in numerical experiments.

Action of S^-1 on a pressure vector y:

    1. Solve K_w . z1 = y                  // pressure-space Poisson V-cycle
    2. z2 = B Cwl^-1 A Dwr^-1 B^T . z1     // matvec only
    3. Solve K_w . z3 = z2                 // same V-cycle
    4. return z3

where K_w = B Cwl^-1 B^T.

The paper reinterprets K_w as a variable-coefficient Poisson op with coefficient 1/√μ
and re-discretizes it on a continuous space, because their pressure is discontinuous
P^disc. Our pressure is continuous Q1, so we can use DivKGrad directly with
k = 1/sqrt(μ).

K_w V-cycle: 3 pre + 3 post Chebyshev-accelerated point-Jacobi (same as our viscous).

Boundary modification (§6): a damping factor near Dirichlet boundaries. We have
free-slip everywhere, so skip in v1.

## Mapping to terraneo

| paper                     | our code                                                |
| ---                       | ---                                                     |
| A (viscous block)         | K_->block_11() (Viscous)                                |
| B^T (gradient)            | K_->block_12() (Gradient)                               |
| B (divergence)            | K_->block_21() (Divergence)                             |
| M_u (velocity mass)       | M_ (ViscousMass), already built                         |
| M_u_lumped(sqrt(μ))       | NEW: weighted lumped velocity mass, weights sqrt(eta_)  |
| K_w = B Cw^-1 B^T         | NEW: pressure-space operator, abstracted behind a       |
|                           | solver interface (see Phase 1)                          |
| K_w^-1                    | NEW: any solver satisfying the interface; v1 will use   |
|                           | DivKGrad(k = 1/sqrt(eta)) + scalar Q1 MG V-cycle        |
| S^-1 (Schur prec.)        | NEW: WBFBTSchurPreconditioner replacing                 |
|                           | current DiagonalSolver<PressureMass>                    |

## Architecture

To keep the assembly/solver of K_w independent from the w-BFBT orchestration, K_w
lives behind a thin interface:

    // Pressure-space operator approximating K_w = B C_w^-1 B^T together with a
    // way to invert it.  Different implementations choose how to assemble and
    // invert this; the WBFBTSchurPreconditioner does not care.
    class WBFBTPressurePoissonSolver {
    public:
      // Apply K_w^-1 to rhs, write into sol.  Should be cheap enough to call
      // twice per outer-FGMRES iteration (it is the dominant cost of w-BFBT).
      virtual void solve(const VectorQ1Scalar<T>& rhs,
                               VectorQ1Scalar<T>& sol) = 0;

      // Refresh internal state when fine-level viscosity has changed (called
      // from stokes.update_viscosity).  May rebuild coarse coefficients,
      // re-compute eigenvalue bounds for Chebyshev smoothers, etc.
      virtual void refresh(const VectorQ1Scalar<T>& sqrt_eta_fine) = 0;
    };

The first concrete implementation is `DivKGradMGPressurePoissonSolver`:

  - Holds a re-discretized DivKGrad hierarchy on continuous Q1 pressure space
    with coefficient k = 1/sqrt(eta).
  - Owns a scalar-Q1 MG V-cycle mirroring the viscous one (Chebyshev smoothers
    order 2, 3 pre/post, scalar Prolongation/Restriction, coarse PCG).
  - `refresh()` re-projects sqrt_eta from velocity-level down to each coarser
    pressure level (via Restriction) and rebuilds the coarse-level k = 1/sqrt
    coefficient grids.  Inverse-diagonal and Chebyshev eigenvalue bounds may
    need recomputation.

Future swap-ins (deferred): an AMG-based variant, a Krylov-only solver, a
single-V-cycle preconditioned PCG, an Operator-only re-discretization that
inverts `B C_w^-1 B^T` literally rather than via a re-discretized DivKGrad, etc.

The orchestrator class `WBFBTSchurPreconditioner` holds:

  - Refs to A (K_->block_11()), B^T (K_->block_12()), B (K_->block_21())
  - C_w_inv_diag (pressure-projected inverse-lumped weighted velocity mass)
  - std::unique_ptr<WBFBTPressurePoissonSolver> for K_w^-1
  - Pressure and velocity temporaries

Its apply(y_p -> z_p) executes the 4-step action:

  1. poisson_solver_->solve(y, z1)
  2. apply B^T to z1                  → v1 (velocity)
  3. scale by D_w^-1 elementwise      → v1
  4. apply A to v1                    → v2
  5. scale by C_w^-1 elementwise      → v2
  6. apply B to v2                    → z2 (pressure)
  7. poisson_solver_->solve(z2, z3)
  8. write z3 to z_p

## Implementation phases

### Phase 1 — `WBFBTPressurePoissonSolver` interface + first impl

Step 1a: define the interface in a new header
`apps/mantlecirculation/src/wbfbt_pressure_poisson.hpp`:

  - Pure-virtual (or concept-based) `WBFBTPressurePoissonSolver` with
    `solve(rhs, sol)` and `refresh(sqrt_eta_fine)`.
  - Document the contract: solve must be reasonably accurate but cheap; refresh
    is called from `stokes.update_viscosity` whenever eta changes.

Step 1b: first concrete implementation `DivKGradMGPressurePoissonSolver`:

  - Build `sqrt_eta_` per level (`VectorQ1Scalar`), seeded from eta at ctor time.
  - Build `k_inv_sqrt_eta_` per pressure-MG level (= 1/sqrt(eta) projected).
  - Build a `DivKGrad<ScalarType>` per pressure-MG level using k_inv_sqrt_eta_;
    re-discretized mode only (no stored matrices, no GCA), so refresh is free
    when the underlying coefficient grid is updated in place.
  - Build a scalar-Q1 MG stack on the pressure domains:
    `ProlongationConstant`/`Restriction`, Chebyshev smoothers (order 2, 3 pre/3
    post), coarse PCG.
  - `solve()` calls `linalg::solvers::solve(multigrid, A, sol, rhs)`.
  - `refresh()` writes new sqrt_eta_fine into the velocity-level grid, then
    restricts down to each coarser pressure level and updates 1/sqrt(eta);
    re-computes inverse diagonals and Chebyshev max-eigenvalue estimates.

  Reference implementations to model after:
   - `tests/test_div_k_grad_multigrid_gca.cpp` for the DivKGrad MG hierarchy.
   - `tests/test_laplace_multigrid.cpp` for the scalar Q1 prolongation/restriction.
   - `apps/mantlecirculation/src/stokes_solver.hpp` for how Chebyshev smoothers,
     inverse diagonals, and the Multigrid object are wired.

Step 1c: micro-test in `tests/test_wbfbt_pressure_poisson.cpp`: build the solver
in isolation with a known sqrt_eta field and a synthetic Poisson RHS, check
that the solve converges (residual reduction by ~1e-6 in <30 V-cycles).

### Phase 2 — Weighted-lumped velocity mass C_w

- sqrt_eta_ on velocity level (built in Phase 1).
- Construct M_u_w = ViscousMass with sqrt(eta) integrated into the form. Two options:
  - (a) Extend `ViscousMass` to accept a nodal weight field (cleanest).
  - (b) Wrap: scale by sqrt_eta before/after applying the unweighted M_u, then lump.
- Set lumped-diagonal mode, mirror the pressure-mass setup at stokes_solver.hpp:560.
- Store C_w_inv_diag as a VectorQ1Vec, refreshed from `update_viscosity`.

### Phase 3 — New Schur preconditioner type

- New class `WBFBTSchurPreconditioner<...>` implementing `OperatorLike`. Members:
  - refs to: B (Gradient block_12) and its adjoint, A (viscous block_11),
    C_w_inv_diag (pressure-space),
  - the K_w hierarchy + its Multigrid<...> solver.
- `apply(y_p -> z_p)` executes the 4-step action above.
- Two temporary pressure vectors and two temporary velocity vectors as members
  (sized once at ctor).
- Wire into `BlockTriangularPreconditioner2x2` in place of the existing
  `DiagonalSolver<PressureMass>` at stokes_solver.hpp:572.

### Phase 4 — Parameters + CLI

- `--stokes-schur-precond=mass|wbfbt` (default `mass` to preserve current behavior).
- `--stokes-wbfbt-poisson-num-vcycles` (default 1)
- `--stokes-wbfbt-poisson-cheby-order` (default 2)
- `--stokes-wbfbt-poisson-num-smoothing-steps` (default 3)
- `--stokes-wbfbt-poisson-rel-tol` (default 1e-6 for the V-cycle wrapper)
- Refresh hook: extend `update_viscosity` to also refresh sqrt_eta and re-prepare K_w.
  Use re-discretized DivKGrad (no stored matrices, no Galerkin coarse) so coefficient
  changes are free — this avoids the GCA-reassembly cost problem we hit on the Stokes
  viscous side.

### Phase 5 — Validation

- Unit test: w-BFBT spectral check on fixed problems. Compare GMRES iter counts
  against M_p(1/μ) baseline on:
  - isoviscous (should be similar)
  - DR=1e3 (small but positive improvement expected)
  - DR=1e5 (large improvement; A7 target)
- Integration test: rerun A7 with --stokes-schur-precond=wbfbt; check that
  stokes_fgmres reduces rel.residual to 1e-4 within ~50-100 iters/step (paper reports
  ~100 for 1e-6 at DR=1e8 with sinkers).

## Scope / risks

- Biggest unknown: which discretization of K_w on continuous Q1 pressure matches the
  abstract B Cw^-1 B^T action. The paper says the underlying inf-dim operator is a
  variable-coefficient Poisson with coefficient derived from C_w^-1, and uses 1/sqrt(μ)
  as the coefficient. For continuous Q1 this should be a straightforward DivKGrad
  call. Spectral equivalence in the paper is proven in inf-dim, not for a specific
  re-discretization, so we have some discretization freedom.
- Boundary damping (§6) may bite at the surface/CMB even with free-slip. Implement
  only if needed.
- Refresh frequency: every time eta changes, K_w (via 1/sqrt(eta) coefficient) and
  C_w_inv_diag (via sqrt(eta) weights) change. Both are re-discretized (no stored
  Galerkin coarse), so refresh is O(N). No GCA-reassembly cost problem.

## Effort estimate

- Phase 1 (pressure MG for K_w): 1-2 days. Most code by analogy to viscous MG.
- Phase 2 (weighted lumped mass): half day.
- Phase 3 (new Schur preconditioner): 1 day.
- Phase 4 (CLI plumbing): half day.
- Phase 5 (tests): 1 day, more if boundary damping needed.
- Total: ~4-5 days. Phase 1 is the biggest chunk.

## Cheapest experiment before committing

Run A7 with current solver but raise outer iter caps:

    stokes-krylov-max-iterations=300
    stokes-krylov-restart=200
    stokes-krylov-relative-tolerance=1e-4

If FGMRES eventually converges in ~200 iters/step, the baseline is just slow and
w-BFBT would roughly halve the iteration count. If it still stalls at residual ~0.99
with restart=300, that's strong evidence the Schur approximation itself is broken and
w-BFBT is the right investment.
