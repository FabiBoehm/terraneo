# terra-ml

Operator learning for the TERRA-NG Stokes solver.

* `terra_data` — manufactured Stokes problems on the spherical shell. Random polynomial
  solutions, right-hand sides derived analytically with sympy against *the same*
  deviatoric operator TERRA implements, validated to symbolically zero difference on the
  code's own analytic test cases.
* `terra_infer` — the neural operator (`wavelet3d`), its training driver, the
  finite-difference Stokes residual used as a physics loss, the mesh symmetry group used
  for augmentation, and the spherical-harmonic/Chebyshev spectral branch.

Install so the embedded interpreter in `terra::ml::NeuralSolver` and the batch scripts
find the packages without `PYTHONPATH` surgery:

    pip install -e python/

## Generating a dataset

    mpirun -np 1 ./stokes_dataset_tool --max-level 3 --min-level 2 --outdir $DIR
    python -m terra_data.generate --dir $DIR --num-train 1000 --num-test 200 \
        --max-degree 4 --contrast-min 1 --contrast-max 1e4 --seed 42

Datasets are large and regenerable — keep them in an hpc-workspace, not in `$HOME`.

## Training

    python -m terra_infer.train_wavelet3d --data $DIR --epochs 720 --attention softmax \
        --symmetry-aug --physics-weight 2 --grad-log-eta --div-fu --fine-continuity \
        --mean-free-target --head-mlp --mean-p-weight 1.0 --momentum-margin 2 \
        --hidden 128 --layers 8

Add `--no-wavelet --spherical 12 --radial-modes 8` for the purely spectral operator,
which is discretisation invariant and can be evaluated on any finer nested mesh.

## What this adds

A neural operator that learns the Stokes solution map on the spherical shell, plus the
data generation and the solver-side hook needed to use it from TERRA.

**Two model families**, both trained on level 3 (7,290 velocity nodes):

| model | params | level 3 | level 5 (359,370 nodes) |
|---|---|---|---|
| wavelet attention + softmax | 4.71M | **0.1169** | 0.7593 native / 0.1561 restricted |
| spectral, no wavelet branch | 0.90M | 0.1498 | **0.2634 native** |

Relative L2 on velocity over the 200-sample test set; predicting zero scores 1.0.

The wavelet model is more accurate at the resolution it was trained on. The spectral one
is **discretisation invariant**: spherical harmonics laterally times Chebyshev radially
give a coefficient tensor of fixed shape whatever the mesh, the channel mixing carries no
mode index, and every other layer is pointwise — so the same weights run natively on any
refinement, and the error rises only 1.76x over 49x the node count before flattening.
Adding the wavelet branch back to that same model collapses level 5 to 0.79, which is
what identifies the grid-tied branch as the obstacle.

For the wavelet model there is a second route: because the meshes are nested (2^L+1 per
axis, coarse coordinates identical to the fine ones), restricting the inputs by a stride,
applying the operator, and interpolating back gives 0.1561 at level 5 — lower error, but
it discards the fine data and only works while the solution is resolved on the coarse
mesh.

### Predictions

Equatorial cut of the spectral operator on the easiest and hardest test samples. Analytic
and predicted share a colour scale per component, so amplitude damping stays visible.

Easiest, sample #74 (viscosity contrast 1.6, rel L2 0.051):

![easy sample](doc/spectral_sample_easy.png)

Hardest, sample #47 (contrast 3696, rel L2 0.575):

![hard sample](doc/spectral_sample_hard.png)

Error correlates with viscosity contrast at 0.61 across the test set and concentrates in
the low-viscosity channels — within sample #47, 0.80 in the softest viscosity decile
against 0.27 in the stiffest.
