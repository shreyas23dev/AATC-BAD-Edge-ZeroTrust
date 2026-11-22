import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

BASE_DIR = Path = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project"
VERIFIED_CIC = os.path.join(BASE_DIR, "rust-edge/decisions_cic_verified.jsonl")
EVAL_CSV = os.path.join(BASE_DIR, "data/ciciot2023/ciciot_eval_expanded_44.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "data/ciciot2023/ciciot_train_benign_44.csv")

print("=" * 80)
print("1. Anomaly Score Distributions in decisions_cic_verified.jsonl")
print("=" * 80)
scores = defaultdict(list)
with open(VERIFIED_CIC) as f:
    for line in f:
        rec = json.loads(line)
        scores[rec['label']].append(rec['anomaly_score'])

print(f"{'Class':<25} {'Count':>6} {'Min':>7} {'Mean':>7} {'Median':>7} {'P90':>7} {'P99':>7} {'Max':>7} {'%>=0.35':>8} {'%>=0.4554':>9}")
print("-" * 95)
for k in sorted(scores.keys()):
    arr = np.array(scores[k])
    pct_border = np.mean(arr >= 0.35) * 100
    pct_overt = np.mean(arr >= 0.4554) * 100
    print(f"{k:<25} {len(arr):>6} {arr.min():>7.4f} {arr.mean():>7.4f} {np.median(arr):>7.4f} {np.percentile(arr, 90):>7.4f} {np.percentile(arr, 99):>7.4f} {arr.max():>7.4f} {pct_border:>7.2f}% {pct_overt:>8.2f}%")

print("\n" + "=" * 80)
print("2. Raw Feature Distributions in Benign Train vs DDoS_UDP & Vulnerability_scanner")
print("=" * 80)

df_train = pd.read_csv(TRAIN_CSV)
df_eval = pd.read_csv(EVAL_CSV)

feature_cols = [c for c in df_train.columns if c not in ["src_ip", "Attack_type", "Attack_label"]]

features_to_check = [
    "Rate", "Number", "Tot size", "Tot sum", "AVG", "Std", "Min", "Max",
    "Header_Length", "Protocol Type", "Time_To_Live", "UDP", "TCP", "HTTP", "HTTPS", "DNS"
]

print(f"\nFeature Percentiles in Benign Calibration Data (N={len(df_train)}):")
print(f"{'Feature':<18} {'Min':>10} {'Median':>10} {'P95':>10} {'P99':>10} {'P99.5':>10} {'Max':>10}")
print("-" * 80)
benign_thresh = {}
for feat in features_to_check:
    val = df_train[feat].values
    p95 = np.percentile(val, 95)
    p99 = np.percentile(val, 99)
    p995 = np.percentile(val, 99.5)
    benign_thresh[feat] = {"p95": p95, "p99": p99, "p99.5": p995, "max": val.max()}
    print(f"{feat:<18} {val.min():>10.2f} {np.median(val):>10.2f} {p95:>10.2f} {p99:>10.2f} {p995:>10.2f} {val.max():>10.2f}")

print("\nComparing with Attack Classes:")
for cat in ["Normal", "DDoS_UDP", "Vulnerability_scanner", "Mirai", "DDoS_ICMP", "SYN_Flood"]:
    sub = df_eval[df_eval["Attack_type"] == cat]
    if len(sub) == 0:
        continue
    print(f"\n--- Class: {cat} (N={len(sub)}) ---")
    for feat in ["Rate", "Number", "Tot size", "UDP", "HTTP", "TCP"]:
        val = sub[feat].values
        p99_benign = benign_thresh[feat]["p99"]
        pct_above = np.mean(val > p99_benign) * 100
        print(f"  {feat:<12}: Median={np.median(val):>8.2f}, Mean={val.mean():>8.2f}, P90={np.percentile(val, 90):>8.2f}, Max={val.max():>8.2f} | > Benign P99 ({p99_benign:.2f}): {pct_above:>6.2f}%")
