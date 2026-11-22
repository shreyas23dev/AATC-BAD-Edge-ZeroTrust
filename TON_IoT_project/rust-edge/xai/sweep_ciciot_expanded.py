#!/usr/bin/env python3
"""
sweep_ciciot_expanded.py
Sweeps r_thr from 0.0300 up to 0.0400 on the expanded 26,011-frame evaluation stream
to see how low Benign BLOCK can go while monitoring threat recall.
"""

import json
import subprocess
import pandas as pd
import numpy as np

RUST_EDGE_DIR = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"
META_FILE = f"{RUST_EDGE_DIR}/xai/xai_meta_ciciot.json"
EVAL_CSV = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/data/ciciot2023/ciciot_eval_expanded_44.csv"
MODEL_PATH = f"{RUST_EDGE_DIR}/xai/models/autoencoder_ciciot_int8.onnx"
AATC_BIN = f"{RUST_EDGE_DIR}/target/release/aatc-edge"
OUT_JSONL = f"{RUST_EDGE_DIR}/decisions_sweep_temp.jsonl"

df = pd.read_csv(EVAL_CSV, usecols=["Attack_type"])

with open(META_FILE) as f:
    orig_meta = json.load(f)

print(f"{'r_thr':<8} | {'Benign Block %':<14} | {'Overall Mitigated %':<20} | {'High-Impact Mitigated %':<22}")
print("-" * 75)

for r in [0.0300, 0.0320, 0.0340, 0.0360, 0.0380, 0.0400]:
    orig_meta["r_thr"] = r
    with open(META_FILE, "w") as f:
        json.dump(orig_meta, f, indent=2)
        
    cmd = [
        AATC_BIN, "replay",
        "--csv", EVAL_CSV,
        "--model", MODEL_PATH,
        "--meta", META_FILE,
        "--xai",
        "--output", OUT_JSONL
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    decs = []
    with open(OUT_JSONL) as f:
        for line in f:
            decs.append(json.loads(line)["decision"])
            
    df["dec"] = decs[:len(df)]
    
    # Benign
    b_sub = df[df["Attack_type"] == "Normal"]
    b_block = (b_sub["dec"] == "BLOCK").sum()
    b_block_pct = (b_block / len(b_sub)) * 100.0
    
    # Overall attack
    a_sub = df[df["Attack_type"] != "Normal"]
    a_mit = (a_sub["dec"].isin(["VERIFY", "BLOCK"])).sum()
    a_mit_pct = (a_mit / len(a_sub)) * 100.0
    
    # High-impact
    hi_sub = df[(df["Attack_type"] != "Normal") & (df["Attack_type"] != "Vulnerability_scanner")]
    hi_mit = (hi_sub["dec"].isin(["VERIFY", "BLOCK"])).sum()
    hi_mit_pct = (hi_mit / len(hi_sub)) * 100.0
    
    print(f"{r:<8.4f} | {b_block_pct:>12.2f}% | {a_mit_pct:>18.2f}% | {hi_mit_pct:>20.2f}%")
