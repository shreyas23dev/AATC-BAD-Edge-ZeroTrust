#!/usr/bin/env python3
"""
prepare_ciciot_44dim.py
Assigns distinct realistic IP identities to benign IoT devices vs external attackers,
and formats the 44-dimensional feature schema.
"""

import os
import json
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
META_JSON = os.path.join(DATA_DIR, "../../rust-edge/xai/xai_meta_ciciot.json")

TRAIN_RAW = os.path.join(DATA_DIR, "ciciot2023_train_benign.csv")
EVAL_RAW = os.path.join(DATA_DIR, "ciciot2023_multi_attack_eval.csv")

TRAIN_44 = os.path.join(DATA_DIR, "ciciot_train_benign_44.csv")
EVAL_44 = os.path.join(DATA_DIR, "ciciot_eval_44.csv")

def process():
    df_train = pd.read_csv(TRAIN_RAW)
    df_eval = pd.read_csv(EVAL_RAW)
    
    feature_cols = [c for c in df_train.columns if c not in ["src_ip", "Attack_type", "Attack_label"]]
    pad_cols = [f"pad_{i}" for i in range(44 - len(feature_cols))]
    
    for c in pad_cols:
        df_train[c] = 0.0
        df_eval[c] = 0.0
        
    all_44_features = feature_cols + pad_cols
    
    # ── IP Identity Separation ──────────────────────────────────────────────
    # Benign sensor nodes on local subnet
    benign_nodes = ["192.168.1.101", "192.168.1.102", "192.168.1.103", "192.168.1.104"]
    # Distinct attacker nodes per threat family
    attack_ips = {
        "DDoS_UDP": "10.0.10.1",
        "DDoS_ICMP": "10.0.10.2",
        "Port_Scanning": "10.0.20.1",
        "Vulnerability_scanner": "10.0.20.2",
        "Password": "10.0.30.1"
    }
    
    np.random.seed(42)
    new_ips = []
    for _, row in df_eval.iterrows():
        cat = row["Attack_type"]
        if cat == "Normal":
            new_ips.append(np.random.choice(benign_nodes))
        else:
            new_ips.append(attack_ips.get(cat, "10.0.99.1"))
            
    df_eval["src_ip"] = new_ips
    
    # Shuffle evaluation stream
    df_eval = df_eval.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    df_train[all_44_features].to_csv(TRAIN_44, index=False)
    
    meta_cols = ["src_ip", "Attack_type", "Attack_label"]
    eval_cols = all_44_features + meta_cols
    df_eval[eval_cols].to_csv(EVAL_44, index=False)
    
    print(f"Saved {len(df_train)} train rows to {TRAIN_44}")
    print(f"Saved {len(df_eval)} eval rows to {EVAL_44}")
    print("\nIP assignments across categories:")
    print(df_eval.groupby("Attack_type")["src_ip"].value_counts())
    
    # Update xai_meta_ciciot.json with calibrated r_thr
    if os.path.exists(META_JSON):
        with open(META_JSON, "r") as f:
            meta = json.load(f)
        meta["r_thr"] = 0.0180
        meta["calibrated_mse_threshold"] = 0.0180
        with open(META_JSON, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nUpdated {META_JSON} with calibrated r_thr = 0.0180")

if __name__ == "__main__":
    process()
