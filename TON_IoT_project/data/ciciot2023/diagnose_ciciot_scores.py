import os
import json
import pandas as pd
import numpy as np
import onnxruntime as ort

DATA_DIR = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/data/ciciot2023"
MODEL_PATH = "/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge/xai/models/autoencoder_ciciot_int8.onnx"
EVAL_CSV = os.path.join(DATA_DIR, "ciciot_eval_44.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "ciciot_train_benign_44.csv")

session = ort.InferenceSession(MODEL_PATH)

df_train = pd.read_csv(TRAIN_CSV)
df_eval = pd.read_csv(EVAL_CSV)

feature_cols = [c for c in df_train.columns if c not in ["src_ip", "Attack_type", "Attack_label"]]

def compute_win(X, window_size=16):
    n_samples, n_feats = X.shape
    X_win = np.zeros((n_samples, n_feats * 2), dtype=np.float32)
    for t in range(n_samples):
        start = max(0, t - window_size + 1)
        w = X[start:t+1]
        X_win[t, :n_feats] = np.mean(w, axis=0)
        if len(w) > 1:
            X_win[t, n_feats:] = np.std(w, axis=0, ddof=0)
        else:
            X_win[t, n_feats:] = 0.0
    return X_win

print("Evaluating benign training slice...")
X_train_win = compute_win(df_train[feature_cols].values.astype(np.float32))
out_train = session.run(["anomaly_score"], {"float_input": X_train_win[:1000]})
err_train = out_train[0].flatten()
print(f"Train MSE: min={err_train.min():.6f}, median={np.median(err_train):.6f}, 90th={np.percentile(err_train, 90):.6f}, max={err_train.max():.6f}")

print("\nEvaluating eval stream by class...")
for cat in ["Normal", "DDoS_UDP", "DDoS_ICMP", "Port_Scanning", "Vulnerability_scanner", "Password"]:
    sub = df_eval[df_eval["Attack_type"] == cat]
    if len(sub) == 0:
        continue
    X_sub_win = compute_win(sub[feature_cols].values.astype(np.float32))
    out = session.run(["anomaly_score"], {"float_input": X_sub_win[:1000]})
    err = out[0].flatten()
    print(f"Cat {cat:22s} ({len(sub)} frames): min={err.min():.6f}, median={np.median(err):.6f}, 90th={np.percentile(err, 90):.6f}, max={err.max():.6f}")
