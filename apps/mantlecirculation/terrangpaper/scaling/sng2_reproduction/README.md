# SuperMUC-NG Phase 2 strong-scaling sweep

**Figure produced:** the SuperMUC-NG line of `cross_vendor_strong_scaling.png`.

34 points, standard mode only, from MT32 on 1 device to MT1024 on 512 devices,
8 ranks per node on Intel PVC. Named `run_MT<level>_g<devices>_A3_oe.sh`.

## Running

```
TERRANG_BIN=<path-to>/mantlecirculation \
TERRANG_ACCOUNT=<your-account> \
  sbatch run_MT256_g64_A3_oe.sh
```

`TERRANG_CFG` defaults to `../config_scal_A3.toml`; `TERRANG_OUT` defaults to a
directory named after the point, under the submission directory;
`TERRANG_LOGDIR` sets where the Slurm logs go.

Per-step time comes from `<out>/timer_trees/timer_tree_9.json`, node `timestep`,
`root_time / count`.

## Environment

These scripts carry the original sweep environment, which matters: they set
`PSM3_GPUDIRECT=0` and deliberately set *neither* `I_MPI_OFFLOAD_IPC` nor
`FI_MR_CACHE_MAX_COUNT` / `PSM3_MR_CACHE_SIZE`. Substituting the production
mantle-circulation environment costs 13–79 % per timestep. Do not "modernise"
these exports.

## Result

Reproduces the published numbers well: over the 32 points measured, median
deviation +2.3 %, and within about 1 % for every point costing 3 s/step or more.
Points below 3 s/step run roughly 5 % heavy, which is fixed per-step overhead
that does not scale down.

Only standard mode is covered. The published figure also contains low-memory
SuperMUC points, which were never in the archived script set.
