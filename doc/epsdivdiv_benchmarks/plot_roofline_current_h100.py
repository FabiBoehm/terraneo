"""
Roofline plot for the current EpsilonDivDivKerngen on H100 SXM5.
Two memory roofs (HBM3, L2) + two markers (one per memory level).
Numbers from ncu run on benchmark_operators --min-level 8 --max-level 8.
"""
import matplotlib.pyplot as plt
import numpy as np

# --- ncu measurements (current EpsilonDivDivKerngen, L8) ---
dadd = 17_408_983_040
dmul = 50_620_006_400
dfma = 67_622_666_240
flops_total = dadd + dmul + 2 * dfma  # double-counts FMA
dram_bytes = 13.83e9
l2_bytes   = 131.15e9
duration_s = 28.46e-3

gflops = flops_total / duration_s / 1e9
ai_dram = flops_total / dram_bytes
ai_l2   = flops_total / l2_bytes

# --- H100 SXM5 peaks ---
peak_fp64_gflops = 27_100     # ncu-derived: 8448 dfma inst/cycle * 1.605 GHz * 2 FLOP/FMA
peak_hbm_gbs     = 3350       # HBM3 datasheet
peak_l2_gbs      = 5430       # ncu SoL: achieved 4.60 TB/s = 84.7% -> peak 5.43 TB/s

# --- roofline curves ---
ai = np.logspace(-2, 2, 500)
hbm_roof = np.minimum(ai * peak_hbm_gbs, peak_fp64_gflops)
l2_roof  = np.minimum(ai * peak_l2_gbs,  peak_fp64_gflops)
compute_roof = np.full_like(ai, peak_fp64_gflops)

fig, ax = plt.subplots(figsize=(14, 10))

ax.loglog(ai, hbm_roof, color='C0', lw=4.5, label=f'DRAM roof ({peak_hbm_gbs} GB/s)')
ax.loglog(ai, l2_roof,  color='C1', lw=4.5, label=f'L2 roof ({peak_l2_gbs} GB/s)')
ax.loglog(ai, compute_roof, color='gray', lw=3.2, ls='--',
          label=f'FP64 peak ({peak_fp64_gflops/1000:.1f} TFLOP/s)')

# Markers
ax.plot(ai_dram, gflops, marker='o', ms=28, color='C0', mec='black', mew=2.5,
        ls='', label=f'EpsDivDiv @ DRAM AI ({ai_dram:.2f} F/B)')
ax.plot(ai_l2,   gflops, marker='s', ms=28, color='C1', mec='black', mew=2.5,
        ls='', label=f'EpsDivDiv @ L2 AI ({ai_l2:.2f} F/B)')

# % of memory roof at this AI
pct_dram = 100 * gflops / (ai_dram * peak_hbm_gbs) if ai_dram * peak_hbm_gbs < peak_fp64_gflops else 100 * gflops / peak_fp64_gflops
pct_l2   = 100 * gflops / (ai_l2   * peak_l2_gbs)  if ai_l2   * peak_l2_gbs   < peak_fp64_gflops else 100 * gflops / peak_fp64_gflops

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=26)
ax.set_ylabel('Performance (GFLOP/s)', fontsize=26)
ax.set_title(f'Roofline — H100 SXM5, level 8\n'
             f'DRAM %: {pct_dram:.1f},  L2 %: {pct_l2:.1f}',
             fontsize=26, fontweight='bold')
ax.set_xlim(0.05, 100)
ax.set_ylim(50, 1e5)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.grid(True, which='both', ls=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=20, framealpha=0.95)

plt.tight_layout()
out = 'roofline_current_h100.png'
plt.savefig(out, dpi=140)
print(f'Saved: {out}')
print(f'  Achieved:    {gflops/1000:.3f} TFLOP/s')
print(f'  DRAM AI:     {ai_dram:.3f} FLOP/B   ({pct_dram:.1f}% of HBM roof at this AI)')
print(f'  L2 AI:       {ai_l2:.3f} FLOP/B   ({pct_l2:.1f}% of L2 roof at this AI)')
