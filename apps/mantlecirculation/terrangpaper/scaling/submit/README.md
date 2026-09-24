# Sweep generators and collector

Not tied to one figure: these generate sweep job scripts for machines other than
SuperMUC-NG and LUMI, and collect the results.

| file | purpose |
|---|---|
| `submit_bench_mt_lumi.py` | generate a sweep for LUMI-G |
| `submit_bench_mt_mn5.py`  | generate a sweep for MareNostrum 5 |
| `submit_bench_mt_jb.py`   | generate a sweep for JUWELS Booster |
| `submit_mc_lumi.py`       | generate mantle-circulation production jobs for LUMI-G |
| `collect_bench_mt.py`     | read the emitted manifest and write a results CSV |

Machine-specific paths have been replaced by `TERRANG_BIN` / `TERRANG_ROOT`
environment lookups; set those before running. The account is a `{ACCOUNT}`
template field filled in at generation time.

These produce the same kind of job scripts as `../sng2_reproduction` and
`../lumi_reproduction`, which are the two sweeps actually reproduced here and
are the better starting point if your machine resembles either.
