#!/usr/bin/env python3
"""
analyze_ciciot_results.py
Evaluates AATC-BAD decision breakdown by threat category and extracts domain-specific
feature attribution fingerprints (Phi_t) on the CIC-IoT-2023 benchmark.
"""

import os
import json
import pandas as pd
import numpy as np

DATA_DIR = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/data/ciciot2023"
RUST_EDGE_DIR = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"

EVAL_CSV = os.path.join(DATA_DIR, "ciciot_eval_44.csv")
DECISIONS_FILE = os.path.join(RUST_EDGE_DIR, "decisions_ciciot.jsonl")
META_FILE = os.path.join(RUST_EDGE_DIR, "xai/xai_meta_ciciot.json")
RESULTS_OUT = os.path.join(RUST_EDGE_DIR, "xai/ciciot_cross_dataset_results.json")

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
    
    categories = ["Normal", "DDoS_UDP", "DDoS_ICMP", "Port_Scanning", "Vulnerability_scanner", "Password"]
    summary_rows = []
    
    print("\n" + "=" * 80)
    print("CIC-IoT-2023 Cross-Dataset Validation Summary (AATC-BAD Single-Pass INT8 Autoencoder)")
    print("=" * 80)
    print(f"{'Threat Category':<24} | {'Total':<6} | {'ALLOW':<6} | {'VERIFY':<6} | {'BLOCK':<6} | {'Pure BLOCK %':<12} | {'Mitigated %':<12}")
    print("-" * 80)
    
    attribution_fingerprints = {}
    
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
        
        # Attribution fingerprint for attack categories
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
            
    print("=" * 80)
    
    print("\nTop Attribution Fingerprints (Phi_t) by Threat Category on CIC-IoT-2023:")
    for cat, feats in attribution_fingerprints.items():
        print(f"\n[{cat}]")
        for item in feats:
            print(f"  - {item['feature']:<30} : error = {item['error']:.6f}")
            
    results_payload = {
        "dataset": "CIC-IoT-2023",
        "summary_table": summary_rows,
        "attribution_fingerprints": attribution_fingerprints
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nResults successfully written to {RESULTS_OUT}")

if __name__ == "__main__":
    main()
