#!/usr/bin/env python3
"""
download_expanded_ciciot.py
Expands CIC-IoT-2023 evaluation suite with:
1. DDoS-SlowLoris (2,000 rows) - Slow-burn HTTP connection starvation
2. DoS-SYN_Flood (2,000 rows) - Protocol TCP SYN/RST exhaustion
3. Mirai-udpplain (2,000 rows) - Iconic IoT botnet payload flood
"""

import os
import io
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://huggingface.co/datasets/bencorn/CIC-IoT-2023/resolve/main"

TARGETS = {
    "SlowLoris": ("CSV/CSV/DDoS-SlowLoris/DDoS-SlowLoris.pcap.csv", 2001, "10.0.40.1"),
    "SYN_Flood": ("CSV/CSV/DoS-SYN_Flood/DoS-SYN_Flood.pcap.csv", 2001, "10.0.40.2"),
    "Mirai": ("CSV/CSV/Mirai-udpplain/Mirai-udpplain.pcap.csv", 2001, "10.0.50.1"),
}

def stream_head(rel_path, target_rows):
    url = f"{BASE_URL}/{rel_path}"
    print(f"Streaming {target_rows} rows from {rel_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    rows = []
    header = None
    with urllib.request.urlopen(req) as resp:
        buf = ""
        while len(rows) < target_rows + 1:
            chunk = resp.read(65536)
            if not chunk:
                break
            buf += chunk.decode('utf-8', errors='ignore')
            lines = buf.split('\n')
            buf = lines[-1]
            for l in lines[:-1]:
                l = l.strip()
                if l:
                    if header is None:
                        header = l.split(',')
                    else:
                        rows.append(l)
                    if len(rows) >= target_rows:
                        break
    csv_text = ",".join(header) + "\n" + "\n".join(rows)
    return pd.read_csv(io.StringIO(csv_text))

def main():
    existing_eval_path = os.path.join(DATA_DIR, "ciciot_eval_44.csv")
    df_existing = pd.read_csv(existing_eval_path)
    print(f"Existing eval set has {len(df_existing)} rows across categories: {df_existing['Attack_type'].unique()}")
    
    feature_cols = [c for c in df_existing.columns if c not in ["src_ip", "Attack_type", "Attack_label"]]
    
    new_dfs = []
    for cat_name, (rel_path, n_rows, src_ip) in TARGETS.items():
        df_new = stream_head(rel_path, n_rows)
        # Pad columns
        for p in [f"pad_{i}" for i in range(5)]:
            df_new[p] = 0.0
        df_new["src_ip"] = src_ip
        df_new["Attack_type"] = cat_name
        df_new["Attack_label"] = 1
        new_dfs.append(df_new[df_existing.columns])
        print(f"  Successfully loaded {len(df_new)} rows for {cat_name} (src_ip={src_ip})")
        
    df_expanded = pd.concat([df_existing] + new_dfs, ignore_index=True)
    
    # Shuffle to maintain realistic mixed interleaved telemetry stream
    df_expanded = df_expanded.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    expanded_path = os.path.join(DATA_DIR, "ciciot_eval_expanded_44.csv")
    df_expanded.to_csv(expanded_path, index=False)
    print(f"\nExpanded evaluation set saved to {expanded_path} ({len(df_expanded)} total rows).")
    print(df_expanded["Attack_type"].value_counts())

if __name__ == "__main__":
    main()
