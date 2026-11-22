#!/usr/bin/env python3
"""
download_ciciot_slices.py
Streams lightweight, balanced slices of CIC-IoT-2023 from Hugging Face:
- 20,000 Benign rows (10k train, 10k eval)
- 2,000 rows each across 5 distinct attack families:
    1. DDoS-UDP_Flood
    2. DDoS-ICMP_Flood
    3. Recon-PortScan
    4. VulnerabilityScan
    5. DictionaryBruteForce
"""

import os
import io
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://huggingface.co/datasets/bencorn/CIC-IoT-2023/resolve/main"

FILES = {
    "Benign": ("CSV/CSV/Benign_Final/BenignTraffic.pcap.csv", 20000),
    "DDoS_UDP": ("CSV/CSV/DDoS-UDP_Flood/DDoS-UDP_Flood.pcap.csv", 2000),
    "DDoS_ICMP": ("CSV/CSV/DDoS-ICMP_Flood/DDoS-ICMP_Flood.pcap.csv", 2000),
    "Port_Scanning": ("CSV/CSV/Recon-PortScan/Recon-PortScan.pcap.csv", 2000),
    "Vulnerability_scanner": ("CSV/CSV/VulnerabilityScan/VulnerabilityScan.pcap.csv", 2000),
    "Password": ("CSV/CSV/DictionaryBruteForce/DictionaryBruteForce.pcap.csv", 2000),
}

def stream_csv_head(rel_path, target_rows):
    url = f"{BASE_URL}/{rel_path}"
    print(f"Streaming {target_rows} rows from {rel_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    rows = []
    header = None
    
    with urllib.request.urlopen(req) as resp:
        # Stream line by line until we have target_rows + 1 (for header)
        buffer = ""
        while len(rows) < target_rows + 1:
            chunk = resp.read(65536)
            if not chunk:
                break
            buffer += chunk.decode('utf-8', errors='ignore')
            lines = buffer.split('\n')
            buffer = lines[-1] # keep remainder
            for line in lines[:-1]:
                line = line.strip()
                if line:
                    if header is None:
                        header = line.split(',')
                    else:
                        rows.append(line)
                    if len(rows) >= target_rows:
                        break
                        
    print(f"  Successfully extracted {len(rows)} data rows.")
    csv_text = ",".join(header) + "\n" + "\n".join(rows)
    df = pd.read_csv(io.StringIO(csv_text))
    return df

def main():
    dfs = {}
    for label, (rel_path, n_rows) in FILES.items():
        dfs[label] = stream_csv_head(rel_path, n_rows)
        
    print("\nDataset slices loaded successfully:")
    for k, v in dfs.items():
        print(f"  - {k}: {v.shape}")
        
    # Split Benign into 10k train and 10k eval
    df_benign_train = dfs["Benign"].iloc[:10000].copy()
    df_benign_eval = dfs["Benign"].iloc[10000:20000].copy()
    df_benign_eval["Attack_type"] = "Normal"
    df_benign_eval["Attack_label"] = 0
    
    # Save training set (clean benign)
    train_path = os.path.join(DATA_DIR, "ciciot2023_train_benign.csv")
    df_benign_train.to_csv(train_path, index=False)
    print(f"\nSaved training set to {train_path} ({len(df_benign_train)} rows)")
    
    # Assign attack labels and build balanced eval set
    eval_dfs = [df_benign_eval]
    for cat in ["DDoS_UDP", "DDoS_ICMP", "Port_Scanning", "Vulnerability_scanner", "Password"]:
        df_attack = dfs[cat].copy()
        df_attack["Attack_type"] = cat
        df_attack["Attack_label"] = 1
        eval_dfs.append(df_attack)
        
    df_eval = pd.concat(eval_dfs, ignore_index=True)
    
    # Assign synthetic IoT device IDs so AATC-BAD device tracking works
    # 5 tracked device IPs simulating compromised and healthy IoT devices
    np.random.seed(42)
    device_pool = ["192.168.1.101", "192.168.1.102", "192.168.1.103", "192.168.1.104", "192.168.1.105"]
    df_eval["src_ip"] = np.random.choice(device_pool, size=len(df_eval))
    
    # Shuffle evaluation stream
    df_eval = df_eval.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    eval_path = os.path.join(DATA_DIR, "ciciot2023_multi_attack_eval.csv")
    df_eval.to_csv(eval_path, index=False)
    print(f"Saved evaluation stream to {eval_path} ({len(df_eval)} rows)")
    print("\nEvaluation stream distribution:")
    print(df_eval["Attack_type"].value_counts())

if __name__ == "__main__":
    main()
