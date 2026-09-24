# LUMI-G strong-scaling sweep

Produces the LUMI-G line of `cross_vendor_strong_scaling.png`.

37 standard-mode points, MT32 on 1 GCD to MT2048 on 4096 GCDs, 8 ranks per node
on AMD MI250X. Named `std_MT<level>_g<gcds>_std.sh`.

| file | purpose |
|---|---|
| `points_std_treeera.txt` | point table: MT, GCDs, mode, nodes, ranks/node, mesh min/max, lat_sdr, rad_sdr, radial-extra-levels, walltime |
| `generate_std.sh` | rebuild all 37 scripts from that table |
| `feed_std.sh` | submit in ascending node count; idempotent |
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
python3 collect_lumi.py <outroot> [published.csv]
```

`TERRANG_CFG` defaults to `../config_scal_A3.toml`; `TERRANG_OUTROOT`,
`TERRANG_OUT` and `TERRANG_LOGDIR` set where results and logs go.

## Building on LUMI

Cray's GPU transport layer must be linked, or MPI aborts at startup with
"GPU_SUPPORT_ENABLED is requested, but GTL library is not linked". With the
LUMI/25.03, PrgEnv-amd, rocm/6.3.4 and craype-accel-amd-gfx90a modules:

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
