#!/usr/bin/env python3
"""
plot_dual_dataset_fingerprints.py
Generates publication-quality comparative attribution figures comparing
the zero-sampling single-pass Autoencoder fingerprints (Phi_t) across
both Edge-IIoTset and the expanded CIC-IoT-2023 benchmarks.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUST_EDGE_DIR = os.path.dirname(SCRIPT_DIR)

# Paths for Edge-IIoTset
EDGE_DECISIONS = os.path.join(RUST_EDGE_DIR, "decisions_multi_attack.jsonl")
EDGE_LABELS = os.path.join(RUST_EDGE_DIR, "multi_attack_eval.csv")
EDGE_META = os.path.join(SCRIPT_DIR, "xai_meta.json")

# Paths for Expanded CIC-IoT-2023
CICIOT_DECISIONS = os.path.join(RUST_EDGE_DIR, "decisions_ciciot_expanded.jsonl")
CICIOT_LABELS = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/data/ciciot2023/ciciot_eval_expanded_44.csv"
CICIOT_META = os.path.join(SCRIPT_DIR, "xai_meta_ciciot.json")

OUT_PNG = os.path.join(SCRIPT_DIR, "dual_dataset_attribution_comparison.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "dual_dataset_attribution_comparison.pdf")
CICIOT_PNG = os.path.join(SCRIPT_DIR, "ciciot_attribution_fingerprints.png")
CICIOT_PDF = os.path.join(SCRIPT_DIR, "ciciot_attribution_fingerprints.pdf")

def load_edge_data():
    with open(EDGE_META) as f:
        meta = json.load(f)
    base_feats = meta["feature_cols"]
    feat_names = [f"mean({c})" for c in base_feats] + [f"std({c})" for c in base_feats]
    df = pd.read_csv(EDGE_LABELS, usecols=["Attack_type"])
    phis = []
    with open(EDGE_DECISIONS) as f:
        for line in f:
            phis.append(json.loads(line)["phi"])
    return df, np.array(phis, dtype=np.float32), feat_names

def load_ciciot_data():
    with open(CICIOT_META) as f:
        meta = json.load(f)
    base_feats = meta["feature_cols"]
    feat_names = [f"mean({c})" for c in base_feats] + [f"std({c})" for c in base_feats]
    df = pd.read_csv(CICIOT_LABELS, usecols=["Attack_type"])
    phis = []
    with open(CICIOT_DECISIONS) as f:
        for line in f:
            phis.append(json.loads(line)["phi"])
    return df, np.array(phis, dtype=np.float32), feat_names

def main():
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'figure.autolayout': False
    })

    common_categories = [
        "Normal",
        "DDoS_UDP",
        "DDoS_ICMP",
        "Port_Scanning",
        "Vulnerability_scanner",
        "Password"
    ]
    common_pretty = [
        "Normal (Benign)",
        "DDoS UDP",
        "DDoS ICMP",
        "Port Scanning",
        "Vuln Scanner",
        "Password Brute"
    ]

    # 1. Load Edge-IIoTset
    edge_df, edge_phis, edge_feats = load_edge_data()
    edge_selected = [
        "mean(icmp.seq_le)",
        "mean(icmp.checksum)",
        "mean(udp.stream)",
        "mean(tcp.connection.syn)",
        "mean(tcp.dstport)",
        "mean(tcp.len)",
        "mean(http.content_length)",
        "std(http.content_length)"
    ]
    edge_idx = [edge_feats.index(f) for f in edge_selected]
    edge_labels = [
        "icmp.seq [mean]",
        "icmp.chk [mean]",
        "udp.stream [mean]",
        "tcp.syn [mean]",
        "tcp.dstport [mean]",
        "tcp.len [mean]",
        "http.len [mean]",
        "http.len [std]"
    ]

    edge_hm = np.zeros((len(common_categories), len(edge_idx)))
    for r, cat in enumerate(common_categories):
        sub = edge_df[edge_df["Attack_type"] == cat].index.values
        if len(sub) > 0:
            edge_hm[r] = np.mean(edge_phis[sub], axis=0)[edge_idx]

    # 2. Load Expanded CIC-IoT-2023
    ciciot_df, ciciot_phis, ciciot_feats = load_ciciot_data()
    
    ciciot_all_cats = [
        "Normal",
        "DDoS_UDP",
        "DDoS_ICMP",
        "SYN_Flood",
        "SlowLoris",
        "Mirai",
        "Port_Scanning",
        "Password",
        "Vulnerability_scanner"
    ]
    ciciot_all_pretty = [
        "Normal (Benign)",
        "DDoS UDP Flood",
        "DDoS ICMP Flood",
        "DoS SYN Flood",
        "DDoS SlowLoris",
        "Mirai Botnet",
        "Port Scanning",
        "Password Brute",
        "Vuln Scanner"
    ]
    
    ciciot_selected = [
        "mean(ICMP)",
        "mean(UDP)",
        "mean(Rate)",
        "mean(Number)",
        "mean(rst_count)",
        "mean(syn_count)",
        "mean(SSH)",
        "mean(Tot sum)",
        "std(HTTP)"
    ]
    ciciot_idx = [ciciot_feats.index(f) for f in ciciot_selected]
    ciciot_labels = [
        "ICMP [mean]",
        "UDP [mean]",
        "Rate [mean]",
        "Number [mean]",
        "rst_cnt [mean]",
        "syn_cnt [mean]",
        "SSH [mean]",
        "Tot sum [mean]",
        "HTTP [std]"
    ]

    ciciot_hm = np.zeros((len(ciciot_all_cats), len(ciciot_idx)))
    for r, cat in enumerate(ciciot_all_cats):
        sub = ciciot_df[ciciot_df["Attack_type"] == cat].index.values
        if len(sub) > 0:
            ciciot_hm[r] = np.mean(ciciot_phis[sub], axis=0)[ciciot_idx]

    # Subset of CIC-IoT matching common categories for side-by-side comparison
    ciciot_common_hm = np.zeros((len(common_categories), len(ciciot_idx)))
    for r, cat in enumerate(common_categories):
        sub = ciciot_df[ciciot_df["Attack_type"] == cat].index.values
        if len(sub) > 0:
            ciciot_common_hm[r] = np.mean(ciciot_phis[sub], axis=0)[ciciot_idx]

    # -------------------------------------------------------------------------
    # FIGURE 1: Dual-Dataset Cross-Testbed Comparison Figure (Side-by-Side Heatmaps)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    plt.subplots_adjust(wspace=0.28, left=0.10, right=0.92, top=0.88, bottom=0.18)

    # Edge-IIoTset Panel
    sns.heatmap(
        edge_hm,
        ax=ax1,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar_kws={'label': r'Reconstruction Error $\Phi_{t,j}$', 'shrink': 0.8},
        xticklabels=edge_labels,
        yticklabels=common_pretty,
        linewidths=0.7,
        linecolor='white',
        annot_kws={"size": 9.5, "weight": "bold"}
    )
    ax1.set_title(
        r"$\bf{(a)}$ Edge-IIoTset Benchmark (44 Features)",
        fontsize=12, pad=10, loc='left'
    )
    ax1.set_xticklabels(edge_labels, rotation=35, ha='right', fontsize=9.5)
    ax1.set_yticklabels(common_pretty, rotation=0, fontsize=10.5, weight='semibold')

    # CIC-IoT-2023 Panel
    sns.heatmap(
        ciciot_common_hm,
        ax=ax2,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar_kws={'label': r'Reconstruction Error $\Phi_{t,j}$', 'shrink': 0.8},
        xticklabels=ciciot_labels,
        yticklabels=[""] * len(common_categories),
        linewidths=0.7,
        linecolor='white',
        annot_kws={"size": 9.5, "weight": "bold"}
    )
    ax2.set_title(
        r"$\bf{(b)}$ CIC-IoT-2023 Benchmark (44 Features, Independent Lab)",
        fontsize=12, pad=10, loc='left'
    )
    ax2.set_xticklabels(ciciot_labels, rotation=35, ha='right', fontsize=9.5)

    fig.suptitle(
        r"Cross-Dataset Physical Explainability: In-Flight Feature Attribution Fingerprints ($\Phi_t$)",
        fontsize=13.5, weight='bold', y=0.97
    )

    print(f"Saving Dual-Dataset comparison figure to {OUT_PNG} and {OUT_PDF}...")
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.savefig(OUT_PDF, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 2: Dedicated Expanded CIC-IoT-2023 Publication Figure (All 9 Threat Classes)
    # -------------------------------------------------------------------------
    fig2, (ax_heat, ax_bar) = plt.subplots(
        2, 1, figsize=(13, 11.2), gridspec_kw={'height_ratios': [1.3, 1.25]}
    )
    plt.subplots_adjust(hspace=0.48, top=0.94, bottom=0.09, left=0.15, right=0.95)

    sns.heatmap(
        ciciot_hm,
        ax=ax_heat,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar_kws={'label': r'Reconstruction Attribution $\Phi_{t,j} = (x_j - \hat{x}_j)^2$', 'shrink': 0.85},
        xticklabels=ciciot_labels,
        yticklabels=ciciot_all_pretty,
        linewidths=0.8,
        linecolor='white',
        annot_kws={"size": 9.5, "weight": "bold"}
    )
    ax_heat.set_title(
        r"$\bf{(a)}$ Comprehensive Attribution Heatmap across 8 Attack Vectors (CIC-IoT-2023)",
        fontsize=12.5, pad=12, loc='left'
    )
    ax_heat.set_xticklabels(ciciot_labels, rotation=28, ha='right', fontsize=9.5)
    ax_heat.set_yticklabels(ciciot_all_pretty, rotation=0, fontsize=10, weight='semibold')

    # Panel B: Grouped Bar Chart of 5 core contrasting attack archetypes
    focus_cats = ["Normal", "DDoS_UDP", "DDoS_ICMP", "SYN_Flood", "SlowLoris", "Password"]
    focus_pretty = ["Normal", "UDP Flood", "ICMP Flood", "SYN Flood", "SlowLoris", "Password Brute"]
    attack_palette = {
        "Normal": "#9e9e9e",
        "DDoS_UDP": "#1f77b4",
        "DDoS_ICMP": "#ff7f0e",
        "SYN_Flood": "#d62728",
        "SlowLoris": "#8c564b",
        "Password": "#9467bd"
    }

    x = np.arange(len(ciciot_labels))
    bar_width = 0.135

    for i, (cat, name) in enumerate(zip(focus_cats, focus_pretty)):
        cat_idx = ciciot_all_cats.index(cat)
        vals = ciciot_hm[cat_idx]
        offset = (i - (len(focus_cats) - 1) / 2) * bar_width
        ax_bar.bar(
            x + offset,
            vals,
            width=bar_width,
            label=name,
            color=attack_palette[cat],
            alpha=0.90,
            edgecolor='white',
            linewidth=0.6
        )

    ax_bar.set_title(
        r"$\bf{(b)}$ Dominant Physical Feature Signatures for Core Threat Archetypes (CIC-IoT-2023)",
        fontsize=12.5, pad=10, loc='left'
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ciciot_labels, fontsize=10, weight='semibold')
    ax_bar.set_ylabel(r"Attribution Score $\Phi_{t,j}$", fontsize=11)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.45)
    ax_bar.set_ylim(0, 1.08)

    ax_bar.legend(
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc',
        fontsize=9.5,
        ncol=3
    )

    print(f"Saving expanded CIC-IoT-2023 dedicated figure to {CICIOT_PNG} and {CICIOT_PDF}...")
    fig2.savefig(CICIOT_PNG, dpi=300, bbox_inches='tight')
    fig2.savefig(CICIOT_PDF, bbox_inches='tight')
    plt.close()

    print("All attribution figures regenerated successfully!")

if __name__ == "__main__":
    main()
