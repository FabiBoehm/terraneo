# Two-Pass AMG Interpolation for GCA

## Problem

The current UB AMG interpolation computes weights as:
```
w_j = |a_ij| / Σ_k |a_ik|
```
using only **direct** fine-to-parent couplings. This ignores fine-to-fine (F-F)
connections entirely. At high coefficient contrast, the weights degenerate
(near-zero for low-k parents), making P rank-deficient at interfaces.

## Literature: Classical AMG Interpolation (Ruge-Stüben / Black Box MG)

The standard AMG interpolation uses the **small-residual assumption** (Ae ≈ 0)
and accounts for F-F connections by **collapsing** them through known weights:

```
w_j = -(a_ij + Σ_{m∈F_s} a_im · a_mj / Σ_{k∈C_i} a_mk) / ã_ii
```

where:
- `a_ij` = assembled coupling from fine node i to parent j
- `a_im` = coupling from i to strongly-connected fine neighbor m
- `a_mj` = coupling from fine neighbor m to parent j of i
- `ã_ii = a_ii + Σ_{m∈F_w} a_im` = diagonal + lumped weak connections
- `F_s` = strongly connected fine neighbors (2-parent nodes with known weights)
- `F_w` = weakly connected fine neighbors (lumped into diagonal)

The indirect term `Σ a_im · a_mj / Σ a_mk` redistributes the F-F coupling
to the parents proportionally, preventing the degenerate zero-weight problem.

## Two-Pass Strategy (adapted from Black Box MG)

### Node classification in the hex/wedge mesh (coarsening by 2):

| Type | Coordinates | Parents | Role |
|------|-------------|---------|------|
| Coarse | x_even, y_even, r_even | — | Identity in P |
| Edge (2-parent) | x_even, y_even, r_odd | 2 (radial) | Pass 1 |
| Face/interior (4-parent) | x or y odd | 2–4 | Pass 2 |

### Pass 1: 2-parent nodes (radial edge midpoints)

These nodes sit between two radially-aligned coarse parents. Their fine
neighbors are all 4-parent nodes (not yet computed), so F-F connections
can only be **lumped** (no collapse possible yet):

```
ã_ii = a_ii + Σ_{m∈F} a_im
w_j = -a_ij / ã_ii
```

For interior M-matrix nodes (row sum = 0), this equals the current formula
`|a_ij|/Σ|a_ik|`. So Pass 1 produces the **same weights** as the current
implementation for these nodes.

### Pass 2: 4-parent nodes (lateral midpoints, diagonal midpoints, interior)

These nodes have fine neighbors that include **2-parent nodes from Pass 1**.
Those connections are **collapsed** using the Pass 1 weights:

For each strongly-connected 2-parent neighbor m of node i:
- Look up a_mj: coupling from m to each parent j of i (from shared elements)
- Compute indirect contribution: `indirect_j += a_im · a_mj / Σ_{k∈C_i} a_mk`

Then:
```
ã_ii = a_ii + Σ_{m∈F_w} a_im          (lump remaining weak F connections)
w_j = -(a_ij + Σ indirect_j) / ã_ii   (includes collapsed F-F-C paths)
```

This is where the two-pass approach differs from the current code: the
indirect term brings in coefficient information from neighboring elements
through F-F-C paths, preventing weight degeneracy at interfaces.

## Implementation Plan

### Step 1: Add new InterpolationMode

In `gca.hpp`, add to the enum:
```cpp
enum class InterpolationMode
{
    Constant,
    Linear,
    OperatorDependent,
    UnknownBasedAMG,
    UnknownBasedAMGLateral,
    UnknownBasedAMGTwoPass,       // <-- new
};
```

### Step 2: New method `precompute_two_pass_weights()`

Add a new method to `TwoGridGCA` that implements the two-pass computation.

**Kernel structure:**

```
precompute_two_pass_weights()
├── Allocate: parent_weight_0..3_, ub_weights_ (same as current)
├── Allocate: pass1_weights_(2, num_sd, nx, ny, nr)  // store Pass 1 results
│
├── PASS 1: Kokkos::parallel_for over all (sd, x, y, r)
│   ├── Skip non-fine nodes (all even)
│   ├── Skip 4-parent nodes (x or y odd)
│   ├── For 2-parent nodes (x_even, y_even, r_odd):
│   │   ├── Assemble full row: a_ii, a_ij (to 2 parents), a_im (to all others)
│   │   ├── Compute: ã_ii = a_ii + Σ_F a_im
│   │   ├── w_j = -a_ij / ã_ii  (per component for vectorial)
│   │   └── Store in pass1_weights_(0..1, sd, x, y, r)
│   └── Kokkos::fence()
│
└── PASS 2: Kokkos::parallel_for over all (sd, x, y, r)
    ├── Skip non-fine nodes and 2-parent nodes
    ├── For 4-parent nodes (x or y odd):
    │   ├── Identify parents C_i (2 or 4 depending on r parity)
    │   ├── Assemble full row: a_ii, a_ij (to parents), a_im (to neighbors)
    │   ├── For each neighbor m:
    │   │   ├── Classify: is m a 2-parent node? (m_x even && m_y even && m_r odd)
    │   │   ├── If 2-parent (strong F): compute a_mj for each parent j of i
    │   │   │   └── Loop over shared elements of m and j, extract A(m_local, j_local)
    │   │   │   └── Compute: indirect_j += a_im · a_mj / Σ_k a_mk
    │   │   └── If not 2-parent (weak F): accumulate into a_weak
    │   ├── Compute: ã_ii = a_ii + a_weak
    │   ├── w_j = -(a_ij + indirect_j) / ã_ii  (per component for vectorial)
    │   └── Store in parent_weight_0..3_ and ub_weights_
    └── Kokkos::fence()
```

### Step 3: Full-row assembly helper (device function)

Extract the row assembly logic into a reusable device function:

```cpp
KOKKOS_INLINE_FUNCTION
void assemble_row_at_node(
    int sd, int x, int y, int r,      // fine node position
    ScalarT& a_ii,                      // diagonal (out)
    int& num_neighbors,                 // number of off-diagonal entries (out)
    int* nbr_x, int* nbr_y, int* nbr_r, // neighbor coordinates (out)
    ScalarT* a_ij_vals,                 // coupling values (out)
    int component = 0                   // for vectorial: which diagonal block
) const
```

This iterates over all surrounding hexes/wedges of node (x,y,r), extracting
the diagonal and all off-diagonal entries from the local element matrices.
Max stencil size for wedge mesh: ~14 neighbors (allocate fixed arrays of 20).

### Step 4: Cross-node coupling helper (device function)

For the indirect term, compute coupling from fine neighbor m to parent j:

```cpp
KOKKOS_INLINE_FUNCTION
ScalarT coupling_between_nodes(
    int sd,
    int mx, int my, int mr,    // fine neighbor m
    int jx, int jy, int jr,    // parent j
    int component = 0
) const
```

This iterates over hexes containing both m and j, summing A(m_local, j_local)
from their shared wedges. The shared hexes are at positions where both nodes
fall within [hx, hx+1] × [hy, hy+1] × [hr, hr+1].

### Step 5: Wire into constructor

In the TwoGridGCA constructor (line ~127):
```cpp
else if ( interpolation_mode == InterpolationMode::UnknownBasedAMGTwoPass )
{
    precompute_two_pass_weights();
}
```

And in the P-matrix construction (lines 745-754, 884-897), add the new mode
alongside the existing AMG modes — it stores weights in the same
`parent_weight_0..3_` and `ub_weights_` arrays, so no changes needed there.

### Step 6: Add to test

In `test_div_k_grad_gca_interpolation_modes.cpp`, add the new mode to the
`modes` vector so it's benchmarked against existing modes.

### Step 7: Remove dead code

Remove the unused `compute_amg_weights_at_node()` method (lines 574-647).

## File changes

| File | Changes |
|------|---------|
| `src/terra/linalg/solvers/gca/gca.hpp` | Add enum value, new method, helpers, remove dead code |
| `tests/test_div_k_grad_gca_interpolation_modes.cpp` | Add new mode to test |

## Key design decisions

1. **Fixed-size stencil arrays**: Use stack-allocated arrays of size 20 in the
   Kokkos kernels (max ~14 neighbors for wedge mesh). Avoids dynamic allocation.

2. **Signed values, not absolute**: Use actual signed matrix entries throughout,
   lumping positive off-diagonals into the diagonal (proper AMG treatment of
   non-M-matrix entries).

3. **Strength threshold**: Classify ALL 2-parent fine neighbors as strong
   (eligible for collapse). Other fine neighbors are weak (lumped). No θ
   parameter needed — the two-pass structure naturally defines strong vs weak.

4. **Per-component weights for vectorial**: Same as current UB AMG — each
   velocity component d uses the d-th diagonal block of the local matrices.

5. **pass1_weights_ storage**: A `Kokkos::View<ScalarT*****, LayoutRight>`
   of shape `(num_comp * 2, num_sd, nx, ny, nr)` — only 2 parents per
   Pass 1 node, times num_comp for vectorial operators.
