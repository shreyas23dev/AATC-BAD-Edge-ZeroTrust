#!/usr/bin/env python3
"""
analyze_expanded_ciciot.py
Analyzes AATC-BAD decision breakdown and feature attribution across all 8 attack classes
on the expanded CIC-IoT-2023 benchmark suite.
"""

import os
import json
import pandas as pd
import numpy as np

DATA_DIR = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/data/ciciot2023"
RUST_EDGE_DIR = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"

EVAL_CSV = os.path.join(DATA_DIR, "ciciot_eval_expanded_44.csv")
DECISIONS_FILE = os.path.join(RUST_EDGE_DIR, "decisions_ciciot_expanded.jsonl")
META_FILE = os.path.join(RUST_EDGE_DIR, "xai/xai_meta_ciciot.json")
RESULTS_OUT = os.path.join(RUST_EDGE_DIR, "xai/ciciot_expanded_results.json")

def main():
    with open(META_FILE, "r") as f:
        meta = json.load(f)
        
    feature_cols = meta["feature_cols"]
    all_feature_names = [f"mean({c})" for c in feature_cols] + [f"std({c})" for c in feature_cols]
    
    print(f"Reading ground truth labels from {EVAL_CSV}...")
    df_eval = pd.read_csv(EVAL_CSV, usecols=["Attack_type", "src_ip"])
    
    print(f"Reading decision records from {DECISIONS_FILE}...")
    decisions = []
    phi_list = []
    with open(DECISIONS_FILE, "r") as f:
        for line in f:
            rec = json.loads(line)
            decisions.append(rec["decision"])
            if "phi" in rec and rec["phi"] is not None:
                phi_list.append(rec["phi"])
            else:
                phi_list.append([0.0] * 88)
                
    df_eval["decision"] = decisions[:len(df_eval)]
    phi_matrix = np.array(phi_list[:len(df_eval)], dtype=np.float32)
    
    categories = [
        "Normal",
        "DDoS_ICMP",
        "SYN_Flood",
        "SlowLoris",
        "Mirai",
        "DDoS_UDP",
        "Port_Scanning",
        "Password",
        "Vulnerability_scanner"
    ]
    summary_rows = []
    
    print("\n" + "=" * 90)
    print("Expanded CIC-IoT-2023 Cross-Dataset Validation Summary (AATC-BAD Single-Pass INT8)")
    print("=" * 90)
    print(f"{'Threat Category':<24} | {'Total':<6} | {'ALLOW':<6} | {'VERIFY':<6} | {'BLOCK':<6} | {'Pure BLOCK %':<12} | {'Mitigated %':<12}")
    print("-" * 90)
    
    attribution_fingerprints = {}
    
    total_atk_allow = 0
    total_atk_verify = 0
    total_atk_block = 0
    total_atk_frames = 0
    
    for cat in categories:
        sub = df_eval[df_eval["Attack_type"] == cat]
        total = len(sub)
        if total == 0:
            continue
            
        counts = sub["decision"].value_counts().to_dict()
        allow = counts.get("ALLOW", 0)
        verify = counts.get("VERIFY", 0)
        block = counts.get("BLOCK", 0)
        
        block_pct = (block / total) * 100.0
        mitigated_pct = ((verify + block) / total) * 100.0
        
        if cat != "Normal":
            total_atk_allow += allow
            total_atk_verify += verify
            total_atk_block += block
            total_atk_frames += total
            
        summary_rows.append({
            "category": cat,
            "total": total,
            "allow": allow,
            "verify": verify,
            "block": block,
            "pure_block_pct": round(block_pct, 2),
            "mitigated_pct": round(mitigated_pct, 2)
        })
        
        print(f"{cat:<24} | {total:<6} | {allow:<6} | {verify:<6} | {block:<6} | {block_pct:>10.2f}% | {mitigated_pct:>10.2f}%")
        
        if cat != "Normal":
            cat_indices = sub.index.values
            cat_phi = np.mean(phi_matrix[cat_indices], axis=0)
            top_feats_idx = np.argsort(cat_phi)[::-1][:4]
            top_feats = []
            for idx in top_feats_idx:
                top_feats.append({
                    "feature": all_feature_names[idx],
                    "error": float(cat_phi[idx])
                })
            attribution_fingerprints[cat] = top_feats
            
    print("=" * 90)
    
    atk_block_pct = (total_atk_block / total_atk_frames) * 100.0
    atk_mit_pct = ((total_atk_verify + total_atk_block) / total_atk_frames) * 100.0
    print(f"{'Aggregated Attack Stream':<24} | {total_atk_frames:<6} | {total_atk_allow:<6} | {total_atk_verify:<6} | {total_atk_block:<6} | {atk_block_pct:>10.2f}% | {atk_mit_pct:>10.2f}%")
    
    # High-impact threats (excl. web probes)
    hi_sub = df_eval[(df_eval["Attack_type"] != "Normal") & (df_eval["Attack_type"] != "Vulnerability_scanner")]
    hi_counts = hi_sub["decision"].value_counts().to_dict()
    hi_allow = hi_counts.get("ALLOW", 0)
    hi_verify = hi_counts.get("VERIFY", 0)
    hi_block = hi_counts.get("BLOCK", 0)
    hi_total = len(hi_sub)
    hi_block_pct = (hi_block / hi_total) * 100.0
    hi_mit_pct = ((hi_verify + hi_block) / hi_total) * 100.0
    print(f"{'High-Impact Threats':<24} | {hi_total:<6} | {hi_allow:<6} | {hi_verify:<6} | {hi_block:<6} | {hi_block_pct:>10.2f}% | {hi_mit_pct:>10.2f}%")
    print("=" * 90)

    print("\nTop Attribution Fingerprints (Phi_t) by Threat Category on CIC-IoT-2023:")
    for cat, feats in attribution_fingerprints.items():
        print(f"\n[{cat}]")
        for f in feats:
            print(f"  - {f['feature']:<30} : error = {f['error']:.6f}")
            
    with open(RESULTS_OUT, "w") as f:
        json.dump({
            "summary": summary_rows,
            "fingerprints": attribution_fingerprints,
            "overall_attack_mitigated_pct": round(atk_mit_pct, 2),
            "high_impact_mitigated_pct": round(hi_mit_pct, 2)
        }, f, indent=2)
    print(f"\nResults written to {RESULTS_OUT}")

if __name__ == "__main__":
    main()
