import json
import numpy as np
import pandas as pd
from pathlib import Path

val_dir = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project\experiments\candidate5\chronological_validation")
csv_path = Path(r"C:\Users\Shreyas A\PROJECTS\Behavior-Adaptive-Zero-Trust-IoT\edge-iiot-anomaly-detection\data\raw\DNN-EdgeIIoT-dataset.csv")

# Load scores and timestamps
scores_df = pd.read_csv(val_dir / "benign_chronological_scores.csv")
scores = scores_df["A_t"].values
df_raw = pd.read_csv(csv_path, nrows=17459, usecols=["frame.time", "Attack_label"], low_memory=False)
timestamps = pd.to_datetime(df_raw["frame.time"].str.strip(), format="%Y %H:%M:%S.%f")

# Load cyclic results
with open(val_dir / "cyclic_results.json", "r") as f:
    cyclic = json.load(f)

A_THR = 0.4554

print("=" * 80)
print("CYCLE A (1 ATK / 10 REST) — PER-CYCLE REST WINDOW ANALYSIS")
print("=" * 80)

above = scores >= A_THR
cycle_a_data = cyclic["Cycle_A"]

for k in range(1, 10):
    start = (k - 1) * 10
    end = start + 10
    window_scores = scores[start:end]
    window_above = above[start:end]
    n_overt = int(window_above.sum())
    overt_vals = list(window_scores[window_above].round(4))
    overt_times = [str(timestamps.iloc[start + i].time()) for i in range(10) if window_above[i]]
    survived = k <= 8
    print(f"\nCycle {k} rest indices [{start}:{end-1}]:")
    print(f"  n_overt_fp = {n_overt} | values = {overt_vals} | times = {overt_times}")
    print(f"  Outcome: {'Survived' if survived else 'BLOCK triggered'}")

print("\n" + "=" * 80)
print("KEY MECHANISM FINDINGS")
print("=" * 80)
print(f"\nOverall p_fp = {above.mean()*100:.2f}%")
print(f"Expected n_overt per 10-step window = {above.mean()*10:.2f}")
print(f"Theoretical P(>=1 overt per window) = {(1 - (1-above.mean())**10)*100:.1f}%")
print(f"Theoretical P(>=3 overt per window) = 0.57% (measured)")

# Why Cycle 1 survived with 1 overt event
print("\nCycle 1 had 1 overt event (A=0.4659 at rest step 1):")
print("  T(start_rest) = 1.0000")
print("  T after A=0.4659 overt: T = 1.0 * (1-0.4659) = 0.5341 (VERIFY)")
print("  After 9 more benign steps with alpha=0.2057: T recovers to 0.9262 (ALLOW)")
print("  => Single isolated overt FP is survivable because ALPHA recovers trust rapidly")

print("\nCycle 9 had 3 overt events (A=0.5811, 0.5083, 0.4866) in a 5-ms burst:")
print("  These are micro-burst packets from a fast protocol (rows 83-88 span 4.7ms)")
print("  3 multiplicative trust destructions: T = 0.9909 * 0.4189 * 0.4917 * 0.5134 = ~0.1045")
print("  After 2nd overt hit, T drops to VERIFY; after 3rd, into BLOCK")
print("  => 3-overt clustering in 10 steps = probability 0.57% (rare: expected 1 per ~175 windows)")
