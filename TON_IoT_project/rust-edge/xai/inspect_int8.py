import json, onnxruntime as ort, pandas as pd, numpy as np
from pathlib import Path

PROJ = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge")
EVAL = PROJ / "eval_set.csv"
META = PROJ / "xai" / "xai_meta.json"
MODEL = PROJ / "xai" / "models" / "autoencoder_int8.onnx"

with open(META) as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
window_size = meta["window_size"]
DEVICE_COL = "ip.src_host"

df = pd.read_csv(EVAL, low_memory=False)
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

X_win = compute_window_features(X_raw, df, DEVICE_COL, window_size)
y = df["Attack_label"].values

sess = ort.InferenceSession(str(MODEL))
input_name = sess.get_inputs()[0].name

scores = []
for i in range(len(X_win)):
    out = sess.run(None, {input_name: X_win[i:i+1]})
    scores.append(float(out[0][0][0]))

scores = np.array(scores)
benign = scores[y == 0]
attack = scores[y == 1]

print(f"Benign INT8 MSE : min={benign.min():.6f}, max={benign.max():.6f}, mean={benign.mean():.6f}, 90th={np.percentile(benign, 90):.6f}")
print(f"Attack INT8 MSE : min={attack.min():.6f}, max={attack.max():.6f}, mean={attack.mean():.6f}, 90th={np.percentile(attack, 90):.6f}")

# Finding best threshold
for thr in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]:
    fpr = (benign > thr).mean()
    tpr = (attack > thr).mean()
    print(f"Threshold {thr:.4f} -> FPR={fpr*100:5.1f}%, TPR={tpr*100:5.1f}%")
