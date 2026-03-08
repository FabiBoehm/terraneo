# Vanka Smoother Implementation Plan

## Background & Explanation

### What is a Vanka smoother?

A **Vanka smoother** (also called "local pressure Schur complement" smoother) is a block relaxation method designed for saddle-point systems like Stokes:

```
[ A    B^T ] [ u ]   [ f ]
[ B    0   ] [ p ] = [ g ]
```

Standard point-wise smoothers (Jacobi, Gauss-Seidel) perform poorly on these indefinite systems because they cannot couple velocity and pressure unknowns. Vanka's insight is to **solve small local saddle-point systems per element/cell**, updating all velocity and pressure DOFs associated with that cell simultaneously.

For each cell/element patch:
1. Extract the local residual for all DOFs touching that cell
2. Assemble the local Stokes matrix (velocity-velocity, gradient, divergence blocks)
3. Solve the small dense local system
4. Apply the correction to the global solution (with relaxation)

### How it fits in Terraneo

Currently, the Stokes multigrid test (`test_stokes_pbicgstab_multigrid.cpp`) uses a **block-diagonal preconditioner** where the velocity block is preconditioned with a Jacobi-smoothed multigrid V-cycle, and pressure uses an identity. This is effective but the Jacobi smoother only operates on the velocity block — it never couples velocity and pressure.

A Vanka smoother would replace this setup as a **coupled smoother for the full Stokes system**, operating directly on the `Block2x2OperatorLike` system. It can be used as the smoother inside a multigrid V-cycle for the **entire** Stokes operator (not just the velocity block).

### Cell-based Vanka for the Terraneo grid

In Terraneo's spherical shell discretization:
- Each **hex cell** contains 2 wedges, with **8 vertex nodes** (corners of the hex)
- Velocity lives on a **fine grid** (Q1-iso-Q2), pressure on a **coarse grid** (Q1)
- Each coarse pressure node corresponds to a fine-grid hex cell center

**Per hex cell, the local Vanka system involves:**
- **Velocity DOFs**: 8 nodes × 3 components = **24 velocity unknowns** (on the fine grid)
- **Pressure DOFs**: 8 nodes × 1 = **8 pressure unknowns** (on the coarse grid)
  *(Note: the pressure grid is one level coarser, so a fine hex cell maps to specific coarse nodes — this mapping needs careful handling)*
- **Total local system size**: up to **32 × 32** (or smaller depending on the exact patch definition)

The local system is assembled from the two wedge element matrices (each 18×18 for velocity, plus gradient/divergence coupling), then solved via dense LU factorization.

## Implementation Plan

### Step 1: Add dense LU solver to `dense/mat.hpp`

The existing `inv()` only supports 2×2 and 3×3 matrices. For the Vanka local solve we need a general dense solver for systems up to ~32×32. Add a `KOKKOS_INLINE_FUNCTION` Gaussian elimination with partial pivoting (LU solve) that works for arbitrary `N×N` `dense::Mat`.

**File**: `src/terra/dense/mat.hpp`

**Add**:
- `lu_solve(Mat<T,N,N>& A, Vec<T,N>& b)` — in-place LU factorization and solve, GPU-compatible

### Step 2: Create `vanka.hpp` smoother class

**File**: `src/terra/linalg/solvers/vanka.hpp`

Create a `Vanka` class that satisfies `SolverLike`:

```cpp
template <Block2x2OperatorLike OperatorT>
class Vanka {
public:
    using OperatorType = OperatorT;
    // ...

    Vanka(
        // grid/domain info needed for element iteration
        const grid::shell::DistributedDomain& domain_velocity,
        const grid::shell::DistributedDomain& domain_pressure,
        const grid::Grid3DDataVec<ScalarType, 3>& grid,
        const grid::Grid2DDataScalar<ScalarType>& radii,
        int iterations,
        ScalarType omega,  // relaxation parameter
        SolutionVectorType& tmp
    );

    void solve_impl(OperatorType& A, SolutionVectorType& x, const RHSVectorType& b);
};
```

**Key design decisions:**
- The smoother iterates over **hex cells** using `Kokkos::parallel_for` with the cell-based MDRangePolicy (same as existing operators)
- Each cell kernel:
  1. Computes the local residual `r = b - Ax` for the cell's DOFs
  2. Assembles the local Stokes matrix from the two wedge contributions
  3. Solves the local dense system via LU
  4. Applies the correction with relaxation ω via `Kokkos::atomic_add`
- The approach is **additive Vanka** (all cells updated simultaneously in parallel) — this is naturally parallel but may require under-relaxation (ω < 1) for stability

**Coloring alternative**: If multiplicative (Gauss-Seidel-style) Vanka is needed for better convergence, implement a multi-color scheme where cells of the same color don't share DOFs. The hex grid naturally supports a multi-color decomposition. This can be a follow-up enhancement.

### Step 3: Implement the local element matrix assembly kernel

The core of the Vanka smoother is the per-cell kernel. This needs to:

1. **Extract local velocity coefficients** from the global vector using the existing `extract_local_wedge_vector_coefficients()` and `extract_local_wedge_scalar_coefficients()` helpers
2. **Assemble local matrices** for all 4 blocks:
   - A_local (24×24): velocity-velocity from 2 wedge matrices (each 18×18), mapped to the 8 hex nodes
   - B^T_local (24×8): gradient block
   - B_local (8×24): divergence block
   - O_local (8×8): zero block
3. **Assemble into a single local system** of size up to 32×32
4. **Solve** via LU factorization
5. **Scatter corrections** back via atomic adds

The assembly reuses the existing quadrature and shape function infrastructure from `fe/wedge/integrands.hpp` and `fe/wedge/kernel_helpers.hpp`. The local matrix assembly follows the same pattern as `EpsilonDivDiv`'s stored matrix assembly.

### Step 4: Register in CMakeLists.txt and add includes

**Files to modify:**
- `src/terra/linalg/solvers/CMakeLists.txt` — add `vanka.hpp`
- `src/terra/linalg/linalg.hpp` — add include if needed

### Step 5: Write a test

**File**: `tests/test_stokes_vanka_multigrid.cpp`

Create a test that:
1. Sets up a Stokes problem (reuse the existing test structure from `test_stokes_pbicgstab_multigrid.cpp`)
2. Uses the Vanka smoother as pre/post-smoother in a multigrid V-cycle for the full Stokes system
3. Uses FGMRES or PBiCGStab as the outer Krylov solver
4. Verifies convergence and solution accuracy

## Key Challenges & Considerations

1. **Fine-coarse grid mapping**: Velocity lives on the fine grid, pressure on the coarse grid. The Vanka patch must correctly map between the two grids. Each fine-grid hex cell's 8 corner nodes map to coarse-grid nodes — this is the Q1-iso-Q2/Q1 structure already handled by `VectorQ1IsoQ2Q1`.

2. **Local matrix size**: With 24 velocity + 8 pressure DOFs per hex cell = 32×32 local system. This is feasible for on-the-fly LU on GPU, but careful about register pressure. Could consider pre-computing and storing local matrices (using `LocalMatrixStorage` pattern) if assembly cost dominates.

3. **Boundary treatment**: Dirichlet boundary nodes need special handling — either pin those rows/columns in the local system or skip boundary DOFs.

4. **Parallel correctness**: Additive Vanka with atomic adds is straightforward but may require smaller ω. Multiplicative (colored) Vanka converges faster per iteration but is more complex to implement.

5. **Template compatibility**: The smoother must satisfy `SolverLike` and work with the `Block2x2OperatorLike` Stokes operator. The multigrid template expects uniform smoother/operator types across all levels.
