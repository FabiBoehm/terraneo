# Cell-Based Vanka Smoother — Implementation Plan (Option B)

## Overview

Implement a cell-based (hex-cell) Vanka smoother that pre-assembles and stores the
full local coupling matrix for each hex cell, then uses those during smoothing.
This generalizes the existing `BlockJacobi` (which only stores per-node diagonal blocks)
to capture inter-node couplings within each cell.

**Key dimensions**: 8 nodes/cell × BlockSize DoFs/node = `CellDim` (e.g., 24 for 3D vectors).

---

## Step 1: Add general LU-based inverse to `dense::Mat`

**File**: `src/terra/dense/mat.hpp`

**Why**: `dense::Mat::inv()` currently only supports 2×2 and 3×3. For Vanka we need
to invert 24×24 (or 8×8 for scalar) matrices on-device.

**What**:
- Add a general LU decomposition with partial pivoting to `dense::Mat::inv()` for
  arbitrary sizes (the existing 2×2 and 3×3 specializations remain as fast paths).
- Must be `KOKKOS_INLINE_FUNCTION` (no heap allocation — use stack arrays since
  `CellDim` is compile-time known and ≤24).
- The general case: in-place LU factorization with partial pivoting, then back-substitution
  to form the inverse column-by-column.

---

## Step 2: Implement `compute_cell_vanka_matrices()`

**File**: `src/terra/linalg/solvers/cell_vanka.hpp` (new file)

**Function signature** (analogous to `compute_inverse_block_diagonal`):
```cpp
template <typename OperatorT, int BlockSize>
Kokkos::View<dense::Mat<Scalar, 8*BlockSize, 8*BlockSize>****, grid::Layout>
compute_cell_vanka_matrices(const OperatorT& A, const DistributedDomain& domain);
```

**Returns**: One `CellDim × CellDim` inverse matrix per hex cell, indexed by
`(local_subdomain, x_cell, y_cell, r_cell)`.

### 2a: Local assembly (parallel loop over cells)

For each hex cell C at `(xc, yc, rc)`:

1. Initialize `CellDim × CellDim` local Vanka matrix V to zero.
2. Loop over the **3×3×3 neighborhood** of hex cells: offsets `(dx, dy, dr)` ∈ `{-1, 0, 1}³`.
   - Clamp to valid cell range: `0 ≤ xc+dx < num_cells_x`, etc.
   - For each neighbor cell `(xn, yn, rn)` and each of its 2 wedges:
     - Retrieve `local_mat = A.get_local_matrix(subdomain, xn, yn, rn, wedge)`.
     - For each pair of local nodes `(a, b)` in the wedge (a, b ∈ 0..5):
       - Map `a` → global coords `(gxa, gya, gra)` using the wedge offset arrays.
       - Map `b` → global coords `(gxb, gyb, grb)`.
       - Check if both are nodes of cell C: `gx ∈ {xc, xc+1}`, `gy ∈ {yc, yc+1}`, `gr ∈ {rc, rc+1}`.
       - If yes, compute cell-local indices:
         `ia = (gxa - xc) + 2*(gya - yc) + 4*(gra - rc)` (0..7)
         `ib = (gxb - xc) + 2*(gyb - yc) + 4*(grb - rc)` (0..7)
       - Add the `BlockSize × BlockSize` coupling block:
         `V(ia*BS + di, ib*BS + dj) += local_mat(a + di*6, b + dj*6)` for `di, dj ∈ 0..BS-1`.

3. Store V into the output view.

**Note**: This loop accesses element matrices from neighboring cells on the **same subdomain** only.
Cells near subdomain boundaries will have incomplete Vanka matrices (missing contributions from
elements on adjacent subdomains).

### 2b: Cross-subdomain communication

Extend the row-by-row additive communication pattern from `compute_inverse_block_diagonal()`.

**Approach**: For each cell near a subdomain boundary, the missing contributions affect
specific coupling blocks in the Vanka matrix. We handle this by:

1. **Decompose the Vanka matrix into per-node contributions**: For each node n of cell C,
   the row/column block of V corresponding to n (BlockSize rows/cols) contains contributions
   from all wedges touching n. The DIAGONAL sub-block `V[n,n]` is the same as the block
   diagonal entry. The OFF-DIAGONAL sub-blocks `V[n,m]` are the assembled couplings between
   nodes n and m.

2. **Additive communication of node coupling blocks**: Rather than communicating entire
   cell matrices, communicate assembled node-to-node coupling blocks stored at nodes.

   For each node, store the coupling blocks to its 8 hex-cell co-located nodes (i.e.,
   the 7 other nodes within each cell the node belongs to). These are stored as part
   of an auxiliary `Grid4DData` structure and communicated additively at subdomain boundaries
   using the existing `pack_send_and_recv/unpack_and_reduce` infrastructure.

3. **Gather from communicated node data into cell matrices**: After communication, each
   cell gathers the fully assembled coupling blocks from its 8 nodes to build the
   complete Vanka matrix.

**Simplified first version**: For the initial implementation, skip cross-subdomain
communication. The resulting Vanka matrices at boundaries are approximate but still valid
as a smoother (the missing contributions are small relative to the total). Document this
as a known limitation to be addressed in a follow-up.

### 2c: Invert all cell matrices

```cpp
Kokkos::parallel_for("CellVanka::invert", cells_policy,
    KOKKOS_LAMBDA(int s, int xc, int yc, int rc) {
        cell_matrices(s, xc, yc, rc) = cell_matrices(s, xc, yc, rc).inv();
    });
```

Uses the general LU-based `inv()` from Step 1.

---

## Step 3: Implement `CellVanka` smoother class

**File**: `src/terra/linalg/solvers/cell_vanka.hpp`

**Class**: `CellVanka<OperatorT, BlockSize>` satisfying the `SolverLike` concept.

### Interface (mirrors `BlockJacobi`):

```cpp
template <OperatorLike OperatorT, int BlockSize>
class CellVanka {
public:
    using CellDim = 8 * BlockSize;
    using CellMatrixType = dense::Mat<ScalarType, CellDim, CellDim>;
    using InverseCellMatricesType = Kokkos::View<CellMatrixType****, grid::Layout>;

    CellVanka(const InverseCellMatricesType& inv_cell_matrices,
              int iterations,
              const SolutionVectorType& tmp,
              ScalarType omega = 1.0 / 8.0);

    void solve_impl(OperatorType& A, SolutionVectorType& x, const RHSVectorType& b);
};
```

### Smoothing iteration (additive Vanka):

```
for each iteration:
    1. tmp = b - A*x                           (compute residual)
    2. correction = 0                           (zero out correction vector)
    3. parallel_for over all cells C:
        a. Gather: collect tmp at C's 8 nodes → local_r (CellDim vector)
        b. Solve:  local_c = inv_cell_matrices(C) * local_r
        c. Scatter: atomic_add local_c to correction at C's 8 nodes
    4. x = x + omega * correction
```

**Relaxation**: Use `omega` as the damping parameter. For additive Vanka with ~8 overlapping
cells per node, typical values are `omega ∈ [0.5/8, 1.0/8]` ≈ `[0.0625, 0.125]`.
The optimal value depends on the problem; expose it as a constructor parameter.

**Gather/scatter**: Reuse the existing wedge offset arrays (`w0_dx`, etc.) to map
cell-local node indices (0..7) to grid coordinates. The 8 hex nodes of cell `(xc, yc, rc)`:
```
node i → (xc + i%2, yc + (i/2)%2, rc + i/4)    for i = 0..7
```

---

## Step 4: Test

**File**: `tests/test_cell_vanka_smoother.cpp` (new, modeled after `test_block_jacobi_smoother.cpp`)

1. **Correctness**: Verify that the assembled Vanka matrix diagonal blocks match
   `compute_inverse_block_diagonal` (the diagonal of the Vanka matrix should equal
   the block diagonal).

2. **Convergence**: Compare convergence rates of:
   - Point Jacobi (existing)
   - Block Jacobi (existing)
   - Cell Vanka (new)
   on the same test problems (constant viscosity, radial profiles, lateral viscosity).

3. **Multigrid**: Use cell Vanka as a multigrid smoother and compare V-cycle convergence.

---

## Storage Cost

Per hex cell: one `CellDim × CellDim` matrix = `(8×3)² = 576` doubles = **4.5 KB/cell**.
For a level-5 grid with ~300K cells, that's ~1.3 GB. Manageable for moderate resolutions.

---

## File Summary

| File | Changes |
|------|---------|
| `src/terra/dense/mat.hpp` | Add general LU-based `inv()` for arbitrary-size matrices |
| `src/terra/linalg/solvers/cell_vanka.hpp` | New file: `compute_cell_vanka_matrices()` + `CellVanka` class |
| `tests/test_cell_vanka_smoother.cpp` | New file: convergence tests |
| `tests/CMakeLists.txt` | Register new test |

---

## Implementation Order

1. **Step 1** — General `inv()` in `dense::Mat` (prerequisite for everything)
2. **Step 2** — `compute_cell_vanka_matrices()` (local-only assembly, no communication)
3. **Step 3** — `CellVanka` smoother class
4. **Step 4** — Test (correctness + convergence)
5. **Follow-up** — Add cross-subdomain communication for boundary cells
