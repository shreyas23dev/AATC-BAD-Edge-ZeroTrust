#!/usr/bin/env python3
"""
plot_fingerprints.py
Generates publication-grade figures visualising the zero-sampling single-pass
Autoencoder attribution fingerprints (Phi_t) across distinct IoT threat families.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUST_EDGE_DIR = os.path.dirname(SCRIPT_DIR)

DECISIONS_PATH = os.path.join(RUST_EDGE_DIR, "decisions_multi_attack.jsonl")
LABELS_PATH = os.path.join(RUST_EDGE_DIR, "multi_attack_eval.csv")
META_PATH = os.path.join(SCRIPT_DIR, "xai_meta.json")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "xai_attribution_fingerprints.png")
OUTPUT_PDF = os.path.join(SCRIPT_DIR, "xai_attribution_fingerprints.pdf")

def load_data():
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    
    base_features = meta["feature_cols"]
    feature_names = [f"mean({c})" for c in base_features] + [f"std({c})" for c in base_features]
    
    print("Reading ground-truth labels...")
    df_labels = pd.read_csv(LABELS_PATH, usecols=["Attack_type"])
    
    print("Reading XAI decision stream records (phi vectors)...")
    phi_records = []
    with open(DECISIONS_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            phi_records.append(rec["phi"])
            
    phi_matrix = np.array(phi_records, dtype=np.float32)
    print(f"Loaded phi matrix shape: {phi_matrix.shape}")
    
    df_labels["phi_idx"] = np.arange(len(df_labels))
    return df_labels, phi_matrix, feature_names

def main():
    df_labels, phi_matrix, feature_names = load_data()
    
    categories = [
        "Normal",
        "DDoS_UDP",
        "DDoS_ICMP",
        "Port_Scanning",
        "Vulnerability_scanner",
        "Password"
    ]
    pretty_names = {
        "Normal": "Normal (Benign)",
        "DDoS_UDP": "DDoS UDP Flood",
        "DDoS_ICMP": "DDoS ICMP Flood",
        "Port_Scanning": "Port Scanning",
        "Vulnerability_scanner": "Vulnerability Scanner",
        "Password": "Password Brute-Force"
    }
    
    # Compute mean phi for each category
    mean_phi_dict = {}
    for cat in categories:
        indices = df_labels[df_labels["Attack_type"] == cat].index.values
        if len(indices) > 0:
            mean_phi_dict[cat] = np.mean(phi_matrix[indices], axis=0)
        else:
            mean_phi_dict[cat] = np.zeros(len(feature_names))
            
    # Protocol-grouped key features for clear, interpretable visualization
    grouped_feature_list = [
        # ICMP Protocol Anomalies
        "mean(icmp.seq_le)",
        "mean(icmp.checksum)",
        "std(icmp.seq_le)",
        # UDP Protocol Anomalies
        "mean(udp.stream)",
        "std(udp.stream)",
        # TCP Connection & Reconnaissance Anomalies
        "mean(tcp.connection.syn)",
        "mean(tcp.dstport)",
        "mean(tcp.flags.ack)",
        "std(tcp.ack)",
        # TCP Payload / Sequence Anomalies
        "mean(tcp.len)",
        "std(tcp.len)",
        "std(tcp.seq)",
        # HTTP Application Layer Anomalies
        "mean(http.content_length)",
        "std(http.content_length)",
        "mean(http.response)"
    ]
    
    selected_indices = [feature_names.index(f) for f in grouped_feature_list]
    
    # Clean display labels
    clean_feature_labels = [
        f.replace("mean(", "").replace("std(", "").replace(")", "") + (" [std]" if "std(" in f else " [mean]")
        for f in grouped_feature_list
    ]
    clean_cat_labels = [pretty_names[c] for c in categories]
    
    # Build matrix for heatmap (rows = categories, cols = selected features)
    heatmap_data = np.zeros((len(categories), len(selected_indices)))
    for r, cat in enumerate(categories):
        for c, f_idx in enumerate(selected_indices):
            heatmap_data[r, c] = mean_phi_dict[cat][f_idx]
            
    # Setup Figure (2 panels)
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'figure.autolayout': False
    })
    
    fig, (ax_heat, ax_bar) = plt.subplots(
        2, 1, figsize=(13, 9.8), gridspec_kw={'height_ratios': [1.1, 1.25]}
    )
    plt.subplots_adjust(hspace=0.48, top=0.94, bottom=0.10, left=0.13, right=0.95)
    
    # ----------------------------------------------------
    # Panel A: Heatmap of Attribution Fingerprints (Phi)
    # ----------------------------------------------------
    hm = sns.heatmap(
        heatmap_data,
        ax=ax_heat,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar_kws={'label': r'Reconstruction Attribution $\Phi_{t,j} = (x_j - \hat{x}_j)^2$', 'shrink': 0.85},
        xticklabels=clean_feature_labels,
        yticklabels=clean_cat_labels,
        linewidths=0.8,
        linecolor='white',
        annot_kws={"size": 9.5, "weight": "bold"}
    )
    ax_heat.set_title(
        r"$\bf{(a)}$ Cross-Threat Attribution Heatmap: Zero-Sampling Autoencoder Fingerprints $\Phi_t$",
        fontsize=12.5, pad=12, loc='left'
    )
    ax_heat.set_xticklabels(clean_feature_labels, rotation=32, ha='right', fontsize=9.5)
    ax_heat.set_yticklabels(clean_cat_labels, rotation=0, fontsize=10.5, weight='semibold')
    
    # Protocol section divider annotations above heatmap
    protocol_sections = [
        (0, 2.95, "ICMP Floods", "#d95f02"),
        (3.05, 4.95, "UDP Floods", "#1f77b4"),
        (5.05, 8.95, "TCP Reconnaissance", "#2ca02c"),
        (9.05, 11.95, "TCP Payload", "#e7298a"),
        (12.05, 14.95, "HTTP Application", "#7570b3")
    ]
    for x_start, x_end, label, color in protocol_sections:
        ax_heat.annotate(
            label,
            xy=((x_start + x_end) / 2, -0.65),
            xycoords='data',
            ha='center', va='bottom',
            fontsize=9, weight='bold', color=color,
            annotation_clip=False
        )
    
    # ----------------------------------------------------
    # Panel B: Grouped Bar Chart of Dominant Signatures
    # ----------------------------------------------------
    # Pick 8 most contrasting features
    top_contrast_features = [
        "mean(udp.stream)",
        "mean(icmp.seq_le)",
        "mean(icmp.checksum)",
        "mean(tcp.connection.syn)",
        "mean(tcp.dstport)",
        "mean(tcp.len)",
        "mean(http.content_length)",
        "std(http.content_length)"
    ]
    top_contrast_indices = [feature_names.index(f) for f in top_contrast_features]
    top_contrast_labels = [
        f.replace("mean(", "").replace("std(", "").replace(")", "") + 
        (" (std)" if "std(" in f else " (mean)")
        for f in top_contrast_features
    ]
    
    attack_palette = {
        "Normal": "#9e9e9e",
        "DDoS_UDP": "#1f77b4",
        "DDoS_ICMP": "#ff7f0e",
        "Port_Scanning": "#2ca02c",
        "Vulnerability_scanner": "#d62728",
        "Password": "#9467bd"
    }
    
    x = np.arange(len(top_contrast_features))
    bar_width = 0.135
    
    for i, cat in enumerate(categories):
        vals = [mean_phi_dict[cat][idx] for idx in top_contrast_indices]
        offset = (i - (len(categories) - 1) / 2) * bar_width
        ax_bar.bar(
            x + offset, vals,
            width=bar_width,
            label=pretty_names[cat],
            color=attack_palette[cat],
            alpha=0.92,
            edgecolor='black',
            linewidth=0.6
        )
        
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(top_contrast_labels, rotation=18, ha='right', fontsize=10, weight='semibold')
    ax_bar.set_ylabel(r"Attribution Score $\Phi_{t,j}$", fontsize=11.5)
    ax_bar.set_title(
        r"$\bf{(b)}$ Attack Discrimination Profiles for Dominant Protocol Signatures",
        fontsize=12.5, pad=12, loc='left'
    )
    ax_bar.grid(axis='y', linestyle='--', alpha=0.55)
    ax_bar.legend(loc='upper right', frameon=True, framealpha=0.96, fontsize=9.5, ncol=2)
    ax_bar.set_ylim(0, 1.15)
    
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    print(f"Generated publication figures successfully:\n  PNG: {OUTPUT_PNG}\n  PDF: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
