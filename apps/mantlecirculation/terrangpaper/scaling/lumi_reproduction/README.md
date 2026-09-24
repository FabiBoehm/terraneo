# LUMI-G strong-scaling sweep

**Figure produced:** the LUMI-G line of `cross_vendor_strong_scaling.png`.

37 points, standard mode, from MT32 on 1 GCD to MT2048 on 4096 GCDs, 8 ranks
per node on AMD MI250X. Named `std_MT<level>_g<gcds>_std.sh`.

| file | purpose |
|---|---|
| `points_std_treeera.txt` | the point table: MT, GCDs, mode, nodes, ranks/node, mesh min/max, lat_sdr, rad_sdr, radial-extra-levels, walltime |
| `generate_std.sh` | rebuild all 37 scripts from that table |
| `feed_std.sh` | submit them in ascending node count; idempotent, skips finished and queued points |
| `collect_lumi.py` | read the timer trees and compare against a published CSV |

## Running

```
TERRANG_BIN=<path-to>/mantlecirculation \
TERRANG_ACCOUNT=<your-project> \
  sbatch std_MT256_g64_std.sh
```

or the whole sweep:

```
TERRANG_BIN=... TERRANG_ACCOUNT=... MAX_NODES=1024 ./feed_std.sh
```

Then:

```
python3 collect_lumi.py <outroot> [published.csv]
```

`TERRANG_CFG` defaults to `../config_scal_A3.toml`; `TERRANG_OUTROOT` and
`TERRANG_OUT` set where results go; `TERRANG_LOGDIR` where Slurm logs go.

## Provenance

These are regenerated from the campaign that produced the published numbers,
found on LUMI under `~/terraneo/apps/mantlecirculation/bench_mt/jobs`. A second,
earlier set of LUMI scripts exists that passes `--output-frequency 11` alongside
`--max-timesteps 10` and therefore writes no timer tree at all; it is not the
source of the published values and is not reproduced here.

The decomposition is taken per-axis from that campaign (`--lat-sdr` /
`--rad-sdr`). See `../README.md` for why that matters.

## Building on LUMI

One non-obvious ingredient: Cray's GPU transport layer. Without it the binary
links but MPI aborts at startup with "GPU_SUPPORT_ENABLED is requested, but GTL
library is not linked", because every script here sets
`MPICH_GPU_SUPPORT_ENABLED=1`. With the LUMI/25.03, PrgEnv-amd, rocm/6.3.4 and
craype-accel-amd-gfx90a modules:

```
cmake <source> -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.4/bin/hipcc \
  -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ROCTHRUST=ON -DKokkos_ENABLE_HWLOC=OFF \
  -DMPI_CXX_LINK_FLAGS="-Wl,--whole-archive,-lhugetlbfs,--no-whole-archive" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cray/pe/mpich/8.1.32/gtl/lib -lmpi_gtl_hsa -Wl,-rpath,/opt/cray/pe/mpich/8.1.32/gtl/lib"
make -j16 mantlecirculation
```

Check with `ldd mantlecirculation | grep gtl` before submitting.

## Result

33 of 37 points reproduced, 30 of those within 5 % of published and none above
10 %. The four not yet obtained — MT1024 at 2048 GCDs and MT2048 at 1024, 2048
and 4096 GCDs — were lost to a fabric fault (`OFI poll failed`, "No route to
host") and simply need resubmitting.
