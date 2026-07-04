# AMR adaptive-multigrid iteration penalty at a viscosity kink — investigation notes

**Status:** open (root cause characterized as a *numerical* property, not a code bug; two
confirming experiments queued). Branch `amr` / `mc-amr`, single-GPU (np=1) on helma/NHR@FAU H100.

---

## 1. Background

We are building **block-octree AMR ("approach B")** for TerraNeo. The solver of interest is an
**adaptive LDR-multigrid** used as an FGMRES preconditioner for the Stokes viscous block
(`EpsilonDivDivKerngen`, i.e. the `∇·(2μ ε)` operator).

- The forest is **fixed**; each MG level is the *same* forest at a coarser intra-block LDR
  (lateral-diamond-refinement) resolution. Radii also coarsen 2:1 per level.
- Level operator = `AdaptiveDistributedConstrainedOperator<EpsilonDivDivKerngen>` = **CᵀAC** with
  Dirichlet elimination, where **C** = the 2:1 hanging-node constraint (`hanging = Σ w·parents`,
  geometric weights 0.5/0.25).
- Transfer = `AdaptiveProlongation = C_fine∘S∘C_coarse`, `AdaptiveRestriction` = its exact
  transpose (`S` = `ProlongationVecConstant`, a *linear/Q1* per-block transfer).
- Because the MG coarse solver is an inner Krylov, the preconditioner is **non-linear** → the outer
  solver **must be FGMRES**, not PCG.

### The two test cases

| | `test_adaptive_mg_gpu` (**benchmark**) | `test_adaptive_visckink_gpu` (**visckink**) |
|---|---|---|
| viscosity | mild `k = 2 + sin z` (~1.5×) | **contrast-10** radial `tanh` kink at r=0.75 |
| solution | two-bubble (`mms_plume_rhs.inc`) | co-located `tanh` velocity kink (`mms_visckink_rhs.inc`) |
| S_rad | **1** (no radial subdomains) | **2** (radial subdomains) |
| MG levels | 4 (LDR 2→5) | 3 (LDR 2→4) |
| coarse solver | **unpreconditioned** PCG | **diagonal-preconditioned** PCG (fold diagonal) |

---

## 2. The symptom

- **Benchmark** (refinement sweep, tuned 4-level MG): FGMRES **7/7/8/7/8** across 0→42048 hanging
  nodes. Adaptive ≈ uniform — *no penalty*.
- **Visckink** (3-level MG): **uniform L2 = 9** (mesh- AND coefficient-independent), but
  **adaptive R2 = 21** (broad kink S=8) / **15** (sharp S=16, 32). A ~2.3× penalty, only on the
  adaptive mesh, and **coefficient-sensitive**.
- Discretization accuracy is fine: at matched dofs the adaptive L2 error ties or beats uniform
  (S32: adaptive 1.303e-3 @2.93M vs uniform 1.305e-3 @5.28M).

So: the penalty is **hanging-specific, coefficient-amplified, and interface-local**.

---

## 3. What is RULED OUT (with evidence)

1. **Pure geometry (hanging alone).** The benchmark has hanging + *mild* k and shows no penalty
   (7≈8). So hanging nodes alone are fine.
2. **Coefficient alone.** Uniform L2 with the full contrast-10 kink = 9, same as mild. So the sharp
   coefficient on a conforming mesh is fine.
3. **Geometric interpolation across the jump.** Uniform *also* interpolates its correction across
   the same kink (the kink is in the uniform mesh too) and stays at 9. So it is specifically the
   jump landing **on a 2:1 interface** that hurts, not interpolation across a jump per se.
4. **Smoother diagonal as the *sole* cause.** The Chebyshev smoother is fed an approximate diagonal
   (`assemble_distributed` folds each hanging DoF's block diagonal into its parents with weight `w`,
   over-counting the self-term and omitting cross terms; every wrong term ∝ k, so at contrast 10 the
   parent diagonal is ~10× too large). This is a **real defect** — but three different diagonal
   variants all leave adaptive pinned in a **21–25 floor**, far from 9, while a deliberately-wrong
   "colored-probe" diagonal swung *uniform* 9→15 but barely moved adaptive (21→22). A quantity that
   is *insensitive to the diagonal* is not diagonal-limited. So the diagonal is a bystander, not the
   dominant cause.
   - Failed diagonal fixes: no-fold (class-sum only) = **25**; stochastic Hutchinson N=64 = broke
     even uniform (9→**150**, EpsilonDivDiv is not diagonally dominant); colored-probe = invalid
     coloring on the variable adaptive resolution (uniform 9→**15**).
5. **AMR code bug — NONE FOUND.** Four independent deep code audits, all clean:
   - **Constraint construction** (`adaptive_exchange.hpp build_2to1_tables`): radial (RLOW/RHIGH)
     and lateral faces use one identical axis-agnostic path; weights exact even for non-uniform
     radii (`adaptive_shell_radius` places every 2:1 hanging node at `frac=0.5`).
   - **C / Cᵀ transpose** (`adaptive_2to1_kokkos.hpp`): forward constraint, transpose fold, and
     exchange fold all use the *same* `con_wt` table with a single power of `w`, atomically
     accumulated — exact transpose (no w-vs-w² bug).
   - **Transfer** (`ProlongationVecConstant`/`RestrictionVecConstant`): genuine 2:1 *linear* (Q1)
     interpolation, correct for S_rad=2 (each radial subdomain is a separate subdomain, interface =
     ordinary shared node), exact transpose. `_constant` weights assume **uniform** radial spacing;
     the visckink uses `uniform_shell_radii` so they are exact.
   - **Operator/constraint geometry consistency**: `EpsilonDivDivKerngen` reads the *same*
     `radii_(subdomain, r)` array the mesh built (array-based, never recomputed), and for a radial
     2:1 node the mesh radius is exactly `0.5·(r_p0 + r_p1)` — matching the 0.5 constraint weight.
     Consistent by construction.

---

## 4. Current best explanation (a numerical limitation, not a defect)

The ~2× penalty is the **coarse-grid correction failing at a viscosity jump that sits exactly on a
2:1 hanging interface**:

- The hanging-node constraint `C` (`apply_constraint_device`) forces the coarse correction at the
  interface to `hanging = Σ w·parents` — the **geometric average**, completely **coefficient-blind**
  (no `k` anywhere).
- A 10× viscosity jump on the interface makes the true fine solution have a **derivative kink**
  there (stress `2k·ε` continuous, `k` jumps → strain jumps). The geometric-average constraint
  **structurally cannot represent a kink**, so the prolongated coarse correction is smoothed flat
  exactly across the jump — precisely where the error lives. The smoother must then resolve the
  interface mode alone every cycle → the penalty.
- Uniform has **no** `apply_constraint_device` (no hanging nodes), so its coarse correction carries
  the kink through the interface nodes → stays at 9.

This is the standard geometric hanging-node treatment: correct for smooth coefficients (why the
benchmark and low-contrast cases are fine), deficient for a sharp jump *on* the interface. The
diagonal defect (§3.4) is a secondary contributor in the same region.

### The structural difference that lights it up

The genuinely-new-in-visckink path is **radial-FACE hanging**. With S_rad=1 (benchmark) the radial
faces (RLOW/RHIGH) are CMB/surface *domain boundaries* — never 2:1 interfaces; only lateral-face
hanging exists. With S_rad=2, the R1→R2 refinement puts a subdiv2 band around r=0.75 whose radial
neighbors are subdiv1 → the hanging interface is a **radial face**, and the viscosity kink sits
**exactly on it**. That geometry×physics coincidence exists only in the visckink. (Note: S_rad=1
still has odd-radial hanging *nodes* on lateral faces, so radial *interpolation* is shared — it is
radial *face* hanging coincident with the jump that is unique.)

---

## 5. Experiments in flight (to confirm §4)

- **537750 — naked CG** (`solve_naked`, visckink, LDR=3, unpreconditioned CG on CᵀAC, no MG):
  operator-conditioning probe. **RESULT (DONE):** uniform L0/L1/L2 = 168/306/**630**; adaptive
  R0/R1/R2 = 168/306/**648** (R2 has 20480 hanging; R0/R1 flood to uniform so match exactly).
  Adaptive R2 (648, 363618 dofs, finest=subdiv2) ≈ uniform L2 (630, 665730 dofs, same finest h) —
  **1.03×**. Naked-CG κ is set by `h_min`, and adaptive R2's finest elements have the same `h_min`
  as uniform L2, so they *should* match, and do. **⇒ the CᵀAC operator with hanging at contrast-10
  is as well-conditioned as uniform; the hanging interface adds ~nothing to κ.** Since the same mesh
  goes 9→21 only *under the MG*, the ~2× penalty is confirmed to live in the **MG coarse-grid
  correction**, not the operator. (Confirms §4.)
- **537788 — mild-k control** (`test_adaptive_mg_gpu` now with **S_rad=2** radial hanging but the
  *mild* `2+sin z` coefficient): isolates radial-face hanging *from* the jump. If it stays 7/8 →
  radial-face hanging alone is fine and the jump is required (⇒ §4). If it shows a penalty at mild k
  → radial-face hanging itself is the problem (redirect the hunt there).

Both were pending on the project-wide GPU association cap (`AssocGrpGRES`) at last check.

---

## 6. Untried levers / next steps

1. **Unpreconditioned coarse CG in the visckink** (match the benchmark). The visckink coarse solve
   is diagonal-preconditioned with the *bad fold diagonal*; both solve to 1e-8, so it only matters
   if the coarse PCG is hitting its 300-iter cap without converging. Cheap to test (swap `CoarsePrec`
   in `solve_level`).
2. **Deeper hierarchy** (4 levels like the benchmark) — better coarse-interface resolution.
3. **Coefficient-aware constraint / transfer** (the real fix for high contrast): replace the
   geometric `w` in `apply_constraint_device` with viscosity-weighted (harmonic/k-weighted)
   interpolation at hanging nodes, so the coarse correction can carry the kink. This is a localized
   change to the constraint weights, not a full GCA rebuild.

---

## 7. Key code locations

- Constraint apply / transpose / exchange: `src/terra/grid/shell/adaptive_2to1_kokkos.hpp`
  (`apply_constraint_device`, `apply_constraint_transpose_device`, `apply_exchange_device`,
  `apply_class_sum_device`, `apply_class_broadcast_device`).
- 2:1 table construction (weights, radial vs lateral faces, nx≥3 guard):
  `src/terra/grid/shell/adaptive_exchange.hpp` (`build_2to1_tables`, `face_axes`).
- On-device assembly at np=1: `src/terra/grid/shell/adaptive_distribute.hpp`
  (`AdaptiveDistributedConstrainedOperator::apply_impl`).
- Hanging-node radial geometry: `src/terra/grid/shell/adaptive_geometry.hpp` (`adaptive_shell_radius`).
- Operator radii read: `src/terra/fe/wedge/operators/shell/.../epsilon_divdiv_kerngen.hpp`.
- Tests + MG/naked-CG setup + transfer classes: `tests/test_adaptive_mg_gpu.cpp`,
  `tests/test_adaptive_visckink_gpu.cpp`.

## 8. Latent (not affecting this study)

The radial 0.5 constraint weight is exact only for **uniform** base radii. If a future run combines
**non-uniform base radii** (tanh/mapped shells) with **subdivision ≥ 2** radial grading, a parent
pair can straddle a base-shell breakpoint asymmetrically so `0.5·(r_p0+r_p1) ≠ r(c_h)` — the 0.5
weight would then be slightly non-conforming on radial interfaces. Under `uniform_shell_radii` it is
exact.
