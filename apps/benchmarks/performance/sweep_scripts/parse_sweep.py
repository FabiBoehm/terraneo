#!/usr/bin/env python3
"""Parse full sweep results: throughput from stdout + counters from rocprof CSVs."""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

CU_NUM = 110
SIMD_NUM = CU_NUM * 4
RESULT_DIR = "sweep_results"
SWEEP_LOG  = sys.argv[1] if len(sys.argv) > 1 else None

# Parse throughput from the stdout log
throughput = {}  # tag -> gdofs
if SWEEP_LOG and os.path.exists(SWEEP_LOG):
    with open(SWEEP_LOG) as f:
        current_tag = None
        for line in f:
            m = re.search(r'=== Config lat=(\d+) r=(\d+) rp=(\d+)', line)
            if m:
                current_tag = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
                continue
            # HIP is the second data row (id=1), Kokkos is id=0
            m = re.search(r'\|\s+505284102\s*\|\s+[\d.e+-]+\s*\|\s*(\d)\s*\|\s*8\s*\|\s*[\d\-: ]+\s*\|\s*([\d.e+]+)\s*\|', line)
            if m and current_tag:
                id_ = m.group(1)
                rate = float(m.group(2)) / 1e9
                if id_ == '0':
                    throughput.setdefault(current_tag, {})['kokkos'] = rate
                elif id_ == '1':
                    throughput.setdefault(current_tag, {})['hip'] = rate

# Parse rocprof CSVs per config
def parse_csv(path):
    data = defaultdict(float)
    count = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            kname = row.get('KernelName', '')
            if 'epsdivdiv_dn_matvec' not in kname:
                continue
            grd = int(row.get('grd', 0))
            if grd < 1000000:
                continue
            count += 1
            for col, val in row.items():
                try:
                    v = float(val)
                    if v > data[col]:
                        data[col] = v
                except (ValueError, TypeError):
                    pass
    return data, count

configs = sorted(glob.glob(os.path.join(RESULT_DIR, "cfg_*.csv")))

rows = []
for path in configs:
    base = os.path.basename(path)
    tag = base.replace("cfg_", "").replace(".csv", "")
    lat, r, rp = tag.split("_")
    team = 2 * int(lat) * int(lat) * int(r)

    data, count = parse_csv(path)
    if count == 0:
        rows.append((lat, r, rp, team, None, None, None, None, None, None, None, None, None, None))
        continue

    gui = data.get('GRBM_GUI_ACTIVE', 1)
    waves = data.get('SQ_WAVES', 0)
    wave_cyc = data.get('SQ_WAVE_CYCLES', 0)
    busy_cyc = data.get('SQ_BUSY_CYCLES', 0)
    mean_occ = wave_cyc / busy_cyc / CU_NUM * 4 if busy_cyc > 0 else 0

    active_valu = data.get('SQ_ACTIVE_INST_VALU', 0)
    valu_busy = 100 * active_valu * 4 / SIMD_NUM / gui if gui > 0 else 0

    wait = data.get('SQ_WAIT_INST_ANY', 0)
    wait_pct = 100 * wait / wave_cyc if wave_cyc > 0 else 0

    bank_conf = data.get('SQ_LDS_BANK_CONFLICT', 0)
    bank_pct = 100 * bank_conf / gui / CU_NUM if gui > 0 else 0

    valu_insts = data.get('SQ_INSTS_VALU', 0)
    salu_insts = data.get('SQ_INSTS_SALU', 0)
    lds_insts = data.get('SQ_INSTS_LDS', 0)
    total_insts = valu_insts + salu_insts + lds_insts
    salu_frac = 100 * salu_insts / total_insts if total_insts > 0 else 0

    tcc_rd = data.get('TCC_READ_sum', 0)
    tcc_wr = data.get('TCC_WRITE_sum', 0)
    ea_rd = data.get('TCC_EA_RDREQ_32B_sum', 0)
    ea_wr = data.get('TCC_EA_WRREQ_sum', 0)
    ea_wr64 = data.get('TCC_EA_WRREQ_64B_sum', 0)

    begin = data.get('BeginNs', 0)
    end = data.get('EndNs', 0)
    dur_s = (end - begin) * 1e-9

    l2_bytes = (tcc_rd + tcc_wr) * 64.0
    hbm_bytes = ea_rd * 32.0 + (ea_wr - ea_wr64) * 32.0 + ea_wr64 * 64.0
    l2_bw = l2_bytes / dur_s / 1e9 if dur_s > 0 else 0
    hbm_bw = hbm_bytes / dur_s / 1e9 if dur_s > 0 else 0

    vgpr = int(data.get('arch_vgpr', 0))
    lds = int(data.get('lds', 0))

    gdofs_hip = throughput.get(tag, {}).get('hip')

    rows.append((lat, r, rp, team, gdofs_hip, mean_occ, valu_busy, wait_pct, bank_pct, salu_frac, l2_bw, hbm_bw, vgpr, lds))

# Sort by throughput (best first)
rows.sort(key=lambda x: -(x[4] or 0))

# Print table
print(f"\n{'Config':<12} {'team':>5} {'Gdofs/s':>8} {'waves/CU':>9} {'VALU%':>6} {'wait%':>6} {'LDSconf%':>9} {'SALU%':>6} {'L2BW':>7} {'HBM BW':>7} {'VGPR':>5} {'LDS(B)':>6}")
print("-" * 110)
for lat, r, rp, team, gdofs, mean_occ, valu_busy, wait_pct, bank_pct, salu_frac, l2_bw, hbm_bw, vgpr, lds in rows:
    cfg = f"({lat},{r},{rp})"
    if gdofs is None:
        print(f"{cfg:<12} {team:>5} {'FAIL':>8}")
    else:
        print(f"{cfg:<12} {team:>5} {gdofs:>8.2f} {mean_occ:>9.2f} {valu_busy:>5.1f}% {wait_pct:>5.1f}% {bank_pct:>8.2f}% {salu_frac:>5.1f}% {l2_bw:>6.0f}G {hbm_bw:>6.0f}G {vgpr:>5} {lds:>6}")
