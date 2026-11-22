#!/usr/bin/env python3
"""
plot_xai_benchmark.py
Plots publication-grade comparison of Single-Pass In-Flight Autoencoder
Attribution (Phi_t) vs. Post-Hoc Explainers (TreeSHAP, KernelSHAP).
"""

import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(SCRIPT_DIR, "xai_explainer_benchmark.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "xai_explainer_benchmark.pdf")

plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.autolayout': False
})

methods = [
    "Single-Pass AE (Rust INT8)",
    "Single-Pass AE (ONNX INT8)",
    "TreeSHAP (100-Tree IF)",
    "KernelSHAP (N=25)",
    "KernelSHAP (N=50)",
    "KernelSHAP (N=100)"
]

latencies_ms = [0.115, 0.071, 0.946, 12.79, 11.79, 12.78]
throughputs_fps = [8687, 14042, 1057, 78.2, 84.8, 78.3]
evals_per_frame = [1, 1, 100, 2500, 5000, 10000]

colors = ['#1f77b4', '#3896d8', '#ff7f0e', '#d62728', '#b2182b', '#67001f']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
plt.subplots_adjust(wspace=0.32, left=0.10, right=0.94, top=0.88, bottom=0.22)

# Panel A: Latency (Log Scale)
y_pos = np.arange(len(methods))
bars1 = ax1.barh(y_pos, latencies_ms, color=colors, edgecolor='black', linewidth=0.6, alpha=0.88)
ax1.set_xscale('log')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(methods, fontsize=10, weight='semibold')
ax1.invert_yaxis()
ax1.set_xlabel('Per-Frame Explanation Latency (ms) [Log Scale]', fontsize=11, weight='semibold')
ax1.set_title(r'$\bf{(a)}$ Explanation Latency Overhead per Frame', fontsize=12, pad=10, loc='left')
ax1.grid(axis='x', linestyle='--', alpha=0.5)

# Value annotations
for bar, val in zip(bars1, latencies_ms):
    if val < 1.0:
        label = f"{val*1000:.0f} µs"
    else:
        label = f"{val:.1f} ms"
    ax1.text(val * 1.18, bar.get_y() + bar.get_height()/2, label,
             va='center', ha='left', fontsize=9.5, weight='bold', color='#222222')
ax1.set_xlim(0.01, 80)

# Panel B: Gateway Throughput (fps)
bars2 = ax2.bar(y_pos, throughputs_fps, color=colors, edgecolor='black', linewidth=0.6, alpha=0.88)
ax2.set_yscale('log')
ax2.set_xticks(y_pos)
ax2.set_xticklabels([
    "Rust AE\n(INT8)",
    "ONNX AE\n(INT8)",
    "TreeSHAP\n(IF)",
    "Kernel\n(N=25)",
    "Kernel\n(N=50)",
    "Kernel\n(N=100)"
], fontsize=9.5, weight='semibold')
ax2.set_ylabel('In-Flight Throughput (Frames / Sec) [Log Scale]', fontsize=11, weight='semibold')
ax2.set_title(r'$\bf{(b)}$ Edge Gateway Processing Throughput', fontsize=12, pad=10, loc='left')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# Target budget line (1,000 fps line for high-speed edge telemetry)
ax2.axhline(1000, color='#333333', linestyle=':', linewidth=1.2, label='1,000 fps Real-Time Edge Threshold')
ax2.legend(loc='upper right', fontsize=9.5, frameon=True)

for bar, val in zip(bars2, throughputs_fps):
    ax2.text(bar.get_x() + bar.get_width()/2, val * 1.25, f"{val:,.0f}",
             va='bottom', ha='center', fontsize=9, weight='bold', color='#222222')
ax2.set_ylim(10, 50000)

fig.suptitle(
    r"Quantitative Explainability Benchmark: In-Flight Zero-Sampling $\mathbf{\Phi}_t$ vs. Post-Hoc SHAP",
    fontsize=13.5, weight='bold', y=0.98
)

print(f"Saving XAI benchmark figure to {OUT_PNG} and {OUT_PDF}...")
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_PDF, bbox_inches='tight')
plt.close()
print("Figure generated successfully!")
