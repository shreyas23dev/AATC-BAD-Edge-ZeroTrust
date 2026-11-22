#!/usr/bin/env python3
"""
train_ciciot_autoencoder.py
Trains symmetric Autoencoder on CIC-IoT-2023 benign traffic, exports single-pass
ONNX model with embedded MinMax normalization, and quantizes weights to INT8.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ciciot2023"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "ciciot_train_benign_44.csv"
ONNX_F32 = MODELS_DIR / "autoencoder_ciciot_f32.onnx"
ONNX_INT8 = MODELS_DIR / "autoencoder_ciciot_int8.onnx"
META_OUT = Path(__file__).resolve().parent / "xai_meta_ciciot.json"

WINDOW_SIZE = 16
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

def compute_window_features(X_raw: np.ndarray, window_size: int = 16) -> np.ndarray:
    """Sliding window mean & population std (ddof=0) matching window.rs exactly."""
    n_samples, n_feats = X_raw.shape
    X_win = np.zeros((n_samples, n_feats * 2), dtype=np.float32)
    
    for t in range(n_samples):
        start = max(0, t - window_size + 1)
        w = X_raw[start : t + 1]
        X_win[t, :n_feats] = np.mean(w, axis=0)
        if len(w) > 1:
            X_win[t, n_feats:] = np.std(w, axis=0, ddof=0)
        else:
            X_win[t, n_feats:] = 0.0
            
    return X_win

class SymmetricAutoencoder(nn.Module):
    def __init__(self, in_dim=88, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class SinglePassXAIModel(nn.Module):
    def __init__(self, ae: SymmetricAutoencoder, x_min: np.ndarray, x_max: np.ndarray):
        super().__init__()
        self.ae = ae
        scale = x_max - x_min
        scale[scale <= 1e-6] = 1.0
        
        self.register_buffer("x_min", torch.tensor(x_min, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).unsqueeze(0))

    def forward(self, x_raw):
        # 1. Embedded Robust MinMax scaling (clips extreme outliers to [0, 1])
        x_scaled = torch.clamp((x_raw - self.x_min) / self.scale, 0.0, 1.0)
        
        # 2. Forward autoencoder reconstruction
        x_hat = self.ae(x_scaled)
        
        # 3. Local Feature Attribution vector Phi_t (squared error per feature)
        phi = (x_scaled - x_hat) ** 2
        
        # 4. Composite scalar anomaly score A_t = mean(phi)
        a_t = torch.mean(phi, dim=-1, keepdim=True)
        
        return a_t, phi

def main():
    print(f"Loading training data from {TRAIN_CSV}...")
    df_train = pd.read_csv(TRAIN_CSV)
    feature_cols = list(df_train.columns)
    assert len(feature_cols) == 44, f"Expected 44 features, got {len(feature_cols)}"
    
    X_raw = df_train.values.astype(np.float32)
    print(f"Computing 16-frame sliding window on {len(X_raw)} frames...")
    X_win = compute_window_features(X_raw, WINDOW_SIZE)
    print(f"Windowed feature shape: {X_win.shape}")
    
    # Robust percentile-calibrated MinMax bounds (0.5th and 99.5th percentiles)
    # Prevents transient benign outlier spikes from crushing attack dynamic range
    p_min = np.percentile(X_win, 0.5, axis=0).astype(np.float32)
    p_max = np.percentile(X_win, 99.5, axis=0).astype(np.float32)
    p_scale = p_max - p_min
    p_scale[p_scale <= 1e-6] = 1.0
    
    X_scaled = np.clip((X_win - p_min) / p_scale, 0.0, 1.0).astype(np.float32)
    
    dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(X_scaled))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    ae = SymmetricAutoencoder(in_dim=88, latent_dim=16).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print("Beginning training loop (40 epochs)...")
    for epoch in range(1, EPOCHS + 1):
        ae.train()
        total_loss = 0.0
        for batch_x, target_x in dataloader:
            batch_x, target_x = batch_x.to(device), target_x.to(device)
            optimizer.zero_grad()
            out = ae(batch_x)
            loss = criterion(out, target_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            
        epoch_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:2d}/{EPOCHS:2d}] — Reconstruction Loss (MSE): {epoch_loss:.6f}")
            
    ae.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(X_scaled).to(device)
        reconstructed = ae(X_all).cpu().numpy()
        errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        r_thr = float(np.percentile(errors, 90))
        r_min = float(errors.min())
        r_max = float(errors.max())
        print(f"\nCalibration metrics on benign training set (Robust Scaled):")
        print(f"  MSE min: {r_min:.6f}, max: {r_max:.6f}, 90th-pct (r_thr): {r_thr:.6f}")
        
    # Wrap in single-pass model with robust bounds
    single_pass_model = SinglePassXAIModel(ae.cpu(), p_min, p_max)
    single_pass_model.eval()
    
    # Export Float32 ONNX
    dummy_input = torch.from_numpy(X_win[:1]).float()
    print(f"\nExporting Float32 ONNX to {ONNX_F32}...")
    torch.onnx.export(
        single_pass_model,
        dummy_input,
        str(ONNX_F32),
        input_names=["float_input"],
        output_names=["anomaly_score", "phi"],
        dynamic_axes={
            "float_input": {0: "batch_size"},
            "anomaly_score": {0: "batch_size"},
            "phi": {0: "batch_size"},
        },
        opset_version=14,
        dynamo=False,
    )
    f32_size_kb = os.path.getsize(ONNX_F32) / 1024
    print(f"  Float32 model exported successfully: {f32_size_kb:.2f} KB")
    
    # Quantize to INT8
    print(f"Quantizing Linear projections to INT8 -> {ONNX_INT8}...")
    quantize_dynamic(
        model_input=str(ONNX_F32),
        model_output=str(ONNX_INT8),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        reduce_range=False,
    )
    int8_size_kb = os.path.getsize(ONNX_INT8) / 1024
    print(f"  INT8 model exported successfully: {int8_size_kb:.2f} KB ({((f32_size_kb - int8_size_kb)/f32_size_kb)*100:.1f}% reduction)")
    
    # Save meta json
    meta = {
        "dataset": "CIC-IoT-2023",
        "architecture": "88-64-32-16-32-64-88",
        "n_features_raw": 44,
        "n_features_win": 88,
        "feature_cols": feature_cols,
        "window_size": WINDOW_SIZE,
        "score_min": r_min,
        "score_max": r_max,
        "r_thr": r_thr,
        "f32_size_kb": f32_size_kb,
        "int8_size_kb": int8_size_kb,
        "onnx_f32": str(ONNX_F32),
        "onnx_int8": str(ONNX_INT8)
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_OUT}")

if __name__ == "__main__":
    main()
