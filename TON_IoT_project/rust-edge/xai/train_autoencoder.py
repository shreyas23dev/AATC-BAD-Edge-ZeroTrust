#!/usr/bin/env python3
"""
train_autoencoder.py — Phase 1: Train Autoencoder & Export Single-Pass XAI ONNX
================================================================================
Trains a symmetric deep autoencoder on benign IoT network traffic using 88-dim
flow-level windowed features (44 means + 44 stds).

Single-Pass XAI Principle:
  x_hat = Decoder(Encoder(x_scaled))
  Phi_t = (x_scaled - x_hat)^2        # shape [batch, 88] -> feature attribution
  A_t   = mean(Phi_t, dim=1)          # shape [batch, 1]  -> scalar anomaly score

Both A_t and Phi_t are emitted simultaneously in a single forward pass,
avoiding expensive sampling-based post-hoc explanations (e.g. TreeSHAP/KernelSHAP).
"""

import sys, os, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ── Paths ────────────────────────────────────────────────────────────────────
if platform.system() == "Linux" and os.path.exists("/mnt/c"):
    DATASET = Path("/mnt/c/Users/Shreyas A/PROJECTS/Behavior-Adaptive-Zero-Trust-IoT/edge-iiot-anomaly-detection/data/raw/DNN-EdgeIIoT-dataset.csv")
    EVAL_CSV = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge/eval_set.csv")
    META_PATH = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge/model_meta.json")
    OUT_DIR = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge/xai")
else:
    DATASET = Path(r"C:\Users\Shreyas A\PROJECTS\Behavior-Adaptive-Zero-Trust-IoT\edge-iiot-anomaly-detection\data\raw\DNN-EdgeIIoT-dataset.csv")
    EVAL_CSV = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project\rust-edge\eval_set.csv")
    META_PATH = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project\rust-edge\model_meta.json")
    OUT_DIR = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project\rust-edge\xai")

MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ONNX_F32_OUT = MODELS_DIR / "autoencoder_f32.onnx"
XAI_META_OUT = OUT_DIR / "xai_meta.json"

# Hyperparameters
WINDOW_SIZE = 16
TRAIN_ROWS = 10_000
TOTAL_ROWS = 17_459
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
DEVICE_COL = "ip.src_host"

print("=" * 65)
print("AATC-BAD XAI Extension — Train Autoencoder (Single-Pass XAI)")
print("=" * 65)

# ── 1. Feature Columns & Windowing ───────────────────────────────────────────
with open(META_PATH, "r") as f:
    base_meta = json.load(f)
feature_cols = base_meta["feature_cols"]
n_raw = len(feature_cols)
n_win = n_raw * 2  # 88
print(f"Loaded {n_raw} base features -> {n_win} windowed features.")

print(f"\n[1/5] Loading training rows ({TOTAL_ROWS} head from raw CSV)...")
df = pd.read_csv(DATASET, nrows=TOTAL_ROWS, low_memory=False)
X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

def compute_window_features(X, df, device_col, window_size):
    n, f = X.shape
    means = np.zeros((n, f), dtype=np.float32)
    stds  = np.zeros((n, f), dtype=np.float32)

    device_to_indices = {}
    for global_idx, device in enumerate(df[device_col].values):
        device_to_indices.setdefault(device, []).append(global_idx)

    for device, indices in device_to_indices.items():
        for j, global_idx in enumerate(indices):
            start = max(0, j - window_size + 1)
            window = X[indices[start:j+1]]
            means[global_idx] = window.mean(axis=0)
            if len(window) > 1:
                stds[global_idx] = window.std(axis=0)
    return np.concatenate([means, stds], axis=1).astype(np.float32)

X_win = compute_window_features(X_raw, df, DEVICE_COL, WINDOW_SIZE)
print(f"Windowed features computed: {X_win.shape}")

# ── 2. Scaler Calibration (Benign-only) ──────────────────────────────────────
print("\n[2/5] Fitting MinMaxScaler on benign training rows...")
benign_mask = (df["Attack_label"].values[:TRAIN_ROWS] == 0)
X_train_raw = X_win[:TRAIN_ROWS][benign_mask]
print(f"Benign training samples: {len(X_train_raw)}")

scaler = MinMaxScaler()
scaler.fit(X_train_raw)
data_min = scaler.data_min_.astype(np.float32)
data_scale = scaler.scale_.astype(np.float32)

X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)

# ── 3. Define Autoencoder with Embedded Scaling & XAI Output ─────────────────
print("\n[3/5] Constructing Autoencoder architecture...")

class SinglePassXAIAutoencoder(nn.Module):
    def __init__(self, data_min, data_scale):
        super().__init__()
        # Register min and scale as buffers (serialized into ONNX graph)
        self.register_buffer("data_min", torch.from_numpy(data_min).unsqueeze(0))
        self.register_buffer("data_scale", torch.from_numpy(data_scale).unsqueeze(0))
        
        # Encoder: 88 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(88, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Decoder: 16 -> 32 -> 64 -> 88
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 88),
            nn.Sigmoid()  # Target is scaled in [0, 1]
        )

    def forward(self, x_raw):
        # 1. Scale input: x_scaled = clamp((x_raw - min) * scale, 0, 1)
        x_scaled = torch.clamp((x_raw - self.data_min) * self.data_scale, 0.0, 1.0)
        
        # 2. Forward reconstruction
        bottleneck = self.encoder(x_scaled)
        reconstructed = self.decoder(bottleneck)
        
        # 3. Single-Pass XAI outputs
        # Phi_t: squared error per feature [batch, 88]
        phi = torch.square(x_scaled - reconstructed)
        
        # A_t: mean squared error across all 88 features [batch, 1]
        anomaly_score = torch.mean(phi, dim=-1, keepdim=True)
        
        return anomaly_score, phi

model = SinglePassXAIAutoencoder(data_min, data_scale)

# ── 4. Train Autoencoder ─────────────────────────────────────────────────────
print(f"\n[4/5] Training Autoencoder for {EPOCHS} epochs (Adam, lr={LEARNING_RATE})...")
dataset = TensorDataset(torch.from_numpy(X_train_raw))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for (batch_x,) in dataloader:
        optimizer.zero_grad()
        anomaly_score, phi = model(batch_x)
        # Loss is mean anomaly score
        loss = anomaly_score.mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_x)
    
    epoch_loss /= len(dataset)
    if epoch % 10 == 0 or epoch == 1:
        print(f"      Epoch {epoch:2d}/{EPOCHS} | Train Loss (MSE): {epoch_loss:.6f}")

# ── Evaluate Calibration & Anomaly Threshold ─────────────────────────────────
model.eval()
with torch.no_grad():
    train_scores, _ = model(torch.from_numpy(X_train_raw))
    train_scores = train_scores.numpy().flatten()

# Calibrate A_thr at 90th percentile of benign training reconstruction error
a_thr_ae = float(np.percentile(train_scores, 90))
score_min = float(train_scores.min())
score_max = float(train_scores.max())
print(f"\nTraining Calibration:")
print(f"      Reconstruction MSE min : {score_min:.6f}")
print(f"      Reconstruction MSE max : {score_max:.6f}")
print(f"      90th percentile (A_thr): {a_thr_ae:.6f}")

# ── Test on Chronological Eval Set ───────────────────────────────────────────
if EVAL_CSV.exists():
    eval_df = pd.read_csv(EVAL_CSV, low_memory=False)
    X_eval_raw = eval_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    X_eval_win = compute_window_features(X_eval_raw, eval_df, DEVICE_COL, WINDOW_SIZE)
    y_eval = eval_df["Attack_label"].values

    with torch.no_grad():
        eval_scores, _ = model(torch.from_numpy(X_eval_win))
        eval_scores = eval_scores.numpy().flatten()

    benign_scores = eval_scores[y_eval == 0]
    attack_scores = eval_scores[y_eval == 1]

    print(f"\nEval Set Validation ({len(eval_df)} samples):")
    print(f"      Benign Mean Recon Error : {benign_scores.mean():.6f} (std={benign_scores.std():.6f})")
    print(f"      Attack Mean Recon Error : {attack_scores.mean():.6f} (std={attack_scores.std():.6f})")
    
    benign_blocks = (benign_scores > a_thr_ae).sum()
    attack_blocks = (attack_scores > a_thr_ae).sum()
    print(f"      Benign Flagged (> A_thr): {benign_blocks}/{len(benign_scores)} ({benign_blocks/len(benign_scores)*100:.1f}%)")
    print(f"      Attack Flagged (> A_thr): {attack_blocks}/{len(attack_scores)} ({attack_blocks/len(attack_scores)*100:.1f}%)")

# ── 5. Export to ONNX ────────────────────────────────────────────────────────
print(f"\n[5/5] Exporting ONNX model to {ONNX_F32_OUT.name}...")
dummy_input = torch.zeros(1, 88, dtype=torch.float32)

try:
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_F32_OUT),
        input_names=["float_input"],
        output_names=["anomaly_score", "phi"],
        dynamic_axes={
            "float_input": {0: "batch_size"},
            "anomaly_score": {0: "batch_size"},
            "phi": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
except TypeError:
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_F32_OUT),
        input_names=["float_input"],
        output_names=["anomaly_score", "phi"],
        dynamic_axes={
            "float_input": {0: "batch_size"},
            "anomaly_score": {0: "batch_size"},
            "phi": {0: "batch_size"},
        },
        opset_version=17,
    )

f32_size_kb = ONNX_F32_OUT.stat().st_size / 1024
print(f"      Saved ONNX f32 -> {ONNX_F32_OUT} ({f32_size_kb:.1f} KB)")

# Save XAI metadata
xai_meta = {
    "architecture": "88-64-32-16-32-64-88",
    "n_features_raw": n_raw,
    "n_features_win": n_win,
    "feature_cols": feature_cols,
    "window_size": WINDOW_SIZE,
    "score_min": score_min,
    "score_max": score_max,
    "a_thr_ae": a_thr_ae,
    "onnx_f32": str(ONNX_F32_OUT),
    "explanation_dim": n_win,
    "notes": "Single-pass XAI: anomaly_score=[1], phi=[88]. Embedded min-max scaling."
}

with open(XAI_META_OUT, "w") as f:
    json.dump(xai_meta, f, indent=2)
print(f"      Saved XAI meta -> {XAI_META_OUT}")

print("\n" + "=" * 65)
print("Phase 1 Complete: Autoencoder trained and ONNX exported.")
print("=" * 65)
