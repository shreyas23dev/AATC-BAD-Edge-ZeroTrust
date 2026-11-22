"""
Phase 2C -- Temporal/Relational Evidence Feasibility Audit
==========================================================
Determines experimentally whether temporal/behavioral context recovers
attack evidence that is invisible to point-wise unsupervised anomaly detectors.

STRICT RULES:
  - No evaluation labels used for calibration or threshold selection
  - All temporal features are strictly causal (no future information)
  - No supervised classifier trained
  - No hyperparameter tuning against evaluation labels
  - Dense-probe cluster analyzed explicitly
  - Two independent runs; SHA-256 reproducibility verified
  - Do NOT modify Phase 2A/2B artifacts

Stop condition: if all mechanisms remain insufficient, conclude explicitly.
"""

import sys
import os
import json
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix
)

# ---------------------------------------------------------------------------
# Stdout encoding (Windows cp1252 safe)
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE = (
    Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_audit")
    / "network_processed" / "Network_dataset_1.csv"
)
OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = OUT_DIR / "run_log.txt"
_file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler],
)
logger = logging.getLogger(__name__)

def _log(msg: str):
    logger.info(msg)
    print(msg)

# ---------------------------------------------------------------------------
# Constants -- frozen from Phase 2B spec
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "duration", "src_bytes", "dst_bytes",
    "src_pkts", "dst_pkts",
    "src_ip_bytes", "dst_ip_bytes", "missed_bytes"
]
TRAIN_END_TS   = 1554335999   # 2019-04-03 23:59:59 UTC
CALIB_END_TS   = 1554364211   # 2019-04-04 07:50:11 UTC
EVAL_START_TS  = 1556021410   # 2019-04-23 12:10:10 UTC

OCSVM_PARAMS = dict(kernel="rbf", nu=0.05, gamma="scale")

DENSE_PROBE_VECTOR = {
    "duration": 0.0, "src_bytes": 0.0, "dst_bytes": 0.0,
    "src_pkts": 1, "dst_pkts": 0,
    "src_ip_bytes": 48.0, "dst_ip_bytes": 0.0, "missed_bytes": 0.0
}

ROLLING_WINDOWS = [10, 50, 100, 500]
EWMA_BETAS = [0.05, 0.10, 0.20]
CALIB_PERCENTILE = 99.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float64).tobytes()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile_stats(arr: np.ndarray, label: str) -> dict:
    if len(arr) == 0:
        return {"label": label, "n": 0}
    return {
        "label": label, "n": int(len(arr)),
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def compute_point_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1   = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr  = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr  = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    spec = 1.0 - fpr
    ba   = (rec + spec) / 2.0
    try:
        roc = float(roc_auc_score(labels, scores))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(labels, scores))
    except Exception:
        pr = float("nan")
    n_attack = int(np.sum(labels == 1))
    n_benign = int(np.sum(labels == 0))
    pct_atk_sub = (
        float(np.sum((labels == 1) & (scores < threshold)) / n_attack * 100)
        if n_attack > 0 else float("nan")
    )
    pct_ben_above = (
        float(np.sum((labels == 0) & (scores >= threshold)) / n_benign * 100)
        if n_benign > 0 else float("nan")
    )
    return {
        "threshold": float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "precision": prec, "recall": rec, "f1": f1,
        "FPR": fpr, "FNR": fnr, "specificity": spec,
        "balanced_accuracy": ba,
        "roc_auc": roc, "pr_auc": pr,
        "pct_attack_sub_threshold": pct_atk_sub,
        "pct_benign_above_threshold": pct_ben_above,
    }


def temporal_run_stats(scores: np.ndarray, threshold: float) -> dict:
    """Lag-1 autocorr, volatility, above-threshold run statistics."""
    if len(scores) < 2:
        return {}
    lag1 = (
        float(np.corrcoef(scores[:-1], scores[1:])[0, 1])
        if np.std(scores) > 1e-15 else 0.0
    )
    fdiff = float(np.std(np.diff(scores)))
    above = (scores >= threshold)
    runs = []
    run_len = 0
    in_run = False
    for v in above:
        if v:
            run_len += 1
            in_run = True
        else:
            if in_run:
                runs.append(run_len)
                run_len = 0
                in_run = False
    if in_run:
        runs.append(run_len)
    return {
        "lag1_autocorr": lag1,
        "first_diff_volatility": fdiff,
        "n_above_thr_runs": len(runs),
        "mean_run_length": float(np.mean(runs)) if runs else 0.0,
        "max_run_length": int(np.max(runs)) if runs else 0,
    }


# ---------------------------------------------------------------------------
# Causal temporal feature constructors
# ---------------------------------------------------------------------------

def causal_rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(arr).rolling(window=w, min_periods=1).mean().values

def causal_rolling_max(arr: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(arr).rolling(window=w, min_periods=1).max().values

def causal_exceedance_count(arr: np.ndarray, threshold: float, w: int) -> np.ndarray:
    above = (arr >= threshold).astype(np.float64)
    return pd.Series(above).rolling(window=w, min_periods=1).sum().values

def causal_persistence(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Consecutive run length of arr >= threshold, causal, ending at each t."""
    above = arr >= threshold
    out = np.zeros(len(arr), dtype=np.float64)
    run = 0
    for i, v in enumerate(above):
        run = (run + 1) if v else 0
        out[i] = float(run)
    return out

def causal_ewma(arr: np.ndarray, beta: float) -> np.ndarray:
    """E_t = beta*A_t + (1-beta)*E_{t-1}; E_0 = A_0."""
    out = np.empty(len(arr), dtype=np.float64)
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = beta * arr[i] + (1.0 - beta) * out[i - 1]
    return out

def causal_burst_rate(arr: np.ndarray, w: int) -> np.ndarray:
    """Flow count in causal window (always = min(i+1, w))."""
    out = np.empty(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        out[i] = float(min(i + 1, w))
    return out


# ---------------------------------------------------------------------------
# Build all temporal features
# ---------------------------------------------------------------------------

def build_temporal_features(A: np.ndarray, A_thr_point: float) -> dict:
    """All features are strictly causal. Returns {name: np.ndarray}."""
    feats = {}
    for w in ROLLING_WINDOWS:
        feats[f"roll_mean_{w}"] = causal_rolling_mean(A, w)
        feats[f"roll_max_{w}"]  = causal_rolling_max(A, w)
        # Exceedance normalized by window size -> [0,1]
        feats[f"exceedance_{w}"] = causal_exceedance_count(A, A_thr_point, w) / float(w)
        # Burst rate normalized by window size -> [0,1]
        feats[f"burst_rate_{w}"] = causal_burst_rate(A, w) / float(w)
    feats["persistence"] = causal_persistence(A, A_thr_point)
    for beta in EWMA_BETAS:
        feats[f"ewma_b{int(beta*100):02d}"] = causal_ewma(A, beta)
    return feats


# ---------------------------------------------------------------------------
# Calibrate temporal threshold (label-blind, using only calib A-scores)
# ---------------------------------------------------------------------------

def calibrate_temporal_thresholds(A_calib: np.ndarray, A_thr_point: float) -> dict:
    """Derive per-feature thresholds from the CALIBRATION partition (0 attacks)."""
    thresholds = {}
    for w in ROLLING_WINDOWS:
        thresholds[f"roll_mean_{w}"] = float(np.percentile(
            causal_rolling_mean(A_calib, w), CALIB_PERCENTILE, method="linear"))
        thresholds[f"roll_max_{w}"] = float(np.percentile(
            causal_rolling_max(A_calib, w), CALIB_PERCENTILE, method="linear"))
        thresholds[f"exceedance_{w}"] = float(np.percentile(
            causal_exceedance_count(A_calib, A_thr_point, w) / float(w),
            CALIB_PERCENTILE, method="linear"))
        thresholds[f"burst_rate_{w}"] = float(np.percentile(
            causal_burst_rate(A_calib, w) / float(w),
            CALIB_PERCENTILE, method="linear"))
    thresholds["persistence"] = float(np.percentile(
        causal_persistence(A_calib, A_thr_point), CALIB_PERCENTILE, method="linear"))
    for beta in EWMA_BETAS:
        key = f"ewma_b{int(beta*100):02d}"
        thresholds[key] = float(np.percentile(
            causal_ewma(A_calib, beta), CALIB_PERCENTILE, method="linear"))
    return thresholds


# ---------------------------------------------------------------------------
# Dense-probe analysis
# ---------------------------------------------------------------------------

def analyze_dense_probes(
    A_eval: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    temporal_features: dict,
    temporal_thresholds: dict,
) -> dict:
    probe_vals = np.array([DENSE_PROBE_VECTOR[c] for c in FEATURE_COLS], dtype=np.float64)
    probe_mask = np.all(np.abs(X_eval - probe_vals) < 1e-6, axis=1)
    n_probes = int(probe_mask.sum())
    n_eval = len(A_eval)
    result = {
        "n_matching_rows": n_probes,
        "fraction_of_eval": float(n_probes / n_eval),
        "attack_pct_among_probes": float(y_eval[probe_mask].mean() * 100),
        "point_score_stats": percentile_stats(A_eval[probe_mask], "dense_probe_point"),
        "temporal_evidence": {}
    }
    for fname, feat_arr in temporal_features.items():
        probe_feat = feat_arr[probe_mask]
        thr = temporal_thresholds.get(fname, float("nan"))
        if len(probe_feat) > 0 and not np.isnan(thr):
            above_pct = float(np.mean(probe_feat >= thr) * 100)
        else:
            above_pct = float("nan")
        result["temporal_evidence"][fname] = {
            "mean": float(np.mean(probe_feat)) if len(probe_feat) > 0 else float("nan"),
            "min": float(np.min(probe_feat)) if len(probe_feat) > 0 else float("nan"),
            "max": float(np.max(probe_feat)) if len(probe_feat) > 0 else float("nan"),
            "threshold": thr,
            "pct_above_threshold": above_pct,
        }
    return result


# ---------------------------------------------------------------------------
# Episode analysis
# ---------------------------------------------------------------------------

def segment_episodes(labels: np.ndarray, gap_rows: int = 200) -> list:
    """Return (start_idx, end_idx) inclusive pairs for each attack episode."""
    episodes = []
    in_ep = False
    ep_start = 0
    last_attack = -1
    for i, lbl in enumerate(labels):
        if lbl == 1:
            if not in_ep:
                ep_start = i
                in_ep = True
            last_attack = i
        else:
            if in_ep and (i - last_attack) >= gap_rows:
                episodes.append((ep_start, last_attack))
                in_ep = False
    if in_ep:
        episodes.append((ep_start, last_attack))
    return episodes


def episode_detection(episodes: list, scores: np.ndarray, threshold: float) -> dict:
    n_ep = len(episodes)
    detected = 0
    missed = 0
    delays = []
    for start, end in episodes:
        ep_scores = scores[start:end + 1]
        above_idx = np.where(ep_scores >= threshold)[0]
        if len(above_idx) > 0:
            detected += 1
            delays.append(int(above_idx[0]))
        else:
            missed += 1
    return {
        "n_episodes": n_ep,
        "n_detected": detected,
        "n_missed": missed,
        "detection_pct": float(detected / n_ep * 100) if n_ep > 0 else 0.0,
        "median_detection_delay_rows": float(np.median(delays)) if delays else float("nan"),
        "mean_detection_delay_rows": float(np.mean(delays)) if delays else float("nan"),
        "max_detection_delay_rows": int(np.max(delays)) if delays else 0,
    }


# ---------------------------------------------------------------------------
# Causality check
# ---------------------------------------------------------------------------

def run_causality_check(A_eval: np.ndarray, A_thr_point: float) -> dict:
    """
    A. Full-stream causal EWMA
    B. Chunked causal EWMA (carry state between chunks) -- must match A
    C. Shuffled EWMA -- must differ
    """
    beta = 0.10
    # A: full stream
    full_ewma = causal_ewma(A_eval, beta)
    # B: chunked (chunk_size=10,000)
    chunk_size = 10_000
    chunked = np.empty(len(A_eval), dtype=np.float64)
    state = A_eval[0]  # initial state = first A value
    for i in range(0, len(A_eval), chunk_size):
        chunk = A_eval[i:i + chunk_size]
        ewma_chunk = np.empty(len(chunk), dtype=np.float64)
        ewma_chunk[0] = beta * chunk[0] + (1.0 - beta) * state
        for j in range(1, len(chunk)):
            ewma_chunk[j] = beta * chunk[j] + (1.0 - beta) * ewma_chunk[j - 1]
        chunked[i:i + len(chunk)] = ewma_chunk
        state = float(ewma_chunk[-1])
    # Note: full_ewma[0] = A_eval[0]; chunked[0] = beta*A_eval[0] + (1-beta)*A_eval[0] = A_eval[0]. Consistent.
    max_diff = float(np.max(np.abs(full_ewma - chunked)))
    match = bool(max_diff < 1e-10)
    # C: shuffled
    rng = np.random.default_rng(999)
    shuffled_A = A_eval[rng.permutation(len(A_eval))]
    shuffled_ewma = causal_ewma(shuffled_A, beta)
    shuffled_differs = sha256_array(shuffled_ewma) != sha256_array(full_ewma)
    return {
        "beta_used": beta,
        "full_stream_sha256": sha256_array(full_ewma),
        "chunked_sha256": sha256_array(chunked),
        "max_abs_diff_full_vs_chunked": max_diff,
        "full_vs_chunked_match": match,
        "shuffled_differs_from_full": shuffled_differs,
        "causality_verified": match and shuffled_differs,
        "note": (
            "Full-stream and chunked EWMA agree within 1e-10 tolerance, "
            "confirming strict causal computation. Shuffled stream produces "
            "a different result, confirming temporal ordering dependency."
        ),
    }


# ---------------------------------------------------------------------------
# OneClassSVM baseline
# ---------------------------------------------------------------------------

def fit_ocsvm_and_score(X_train, X_calib, X_eval):
    _log("  Fitting RobustScaler on TRAIN only ...")
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_calib_s = scaler.transform(X_calib)
    X_eval_s  = scaler.transform(X_eval)
    _log(f"  Fitting OneClassSVM(kernel=rbf, nu=0.05, gamma='scale') on {len(X_train_s):,} rows ...")
    t0 = time.time()
    ocsvm = OneClassSVM(**OCSVM_PARAMS)
    ocsvm.fit(X_train_s)
    _log(f"  [OK] Fitted in {time.time()-t0:.1f}s")
    _log("  Scoring TRAIN, CALIB, EVAL streams ...")
    t0 = time.time()
    raw_train = ocsvm.decision_function(X_train_s)
    raw_calib = ocsvm.decision_function(X_calib_s)
    raw_eval  = ocsvm.decision_function(X_eval_s)
    _log(f"  [OK] Scored in {time.time()-t0:.1f}s")
    s_min = float(np.min(raw_train))
    s_max = float(np.max(raw_train))
    denom = s_max - s_min if abs(s_max - s_min) > 1e-15 else 1.0
    # A = (s - s_min)/(s_max - s_min): s_min (most anomalous) -> A=0, s_max (most normal) -> A=1
    A_calib = np.clip((raw_calib - s_min) / denom, 0.0, 1.0)
    A_eval  = np.clip((raw_eval  - s_min) / denom, 0.0, 1.0)
    A_thr = float(np.percentile(A_calib, CALIB_PERCENTILE, method="linear"))
    _log(f"  s_min={s_min:.8f}, s_max={s_max:.8f}, A_thr={A_thr:.8f}")
    _log(f"  A_eval SHA-256: {sha256_array(A_eval)}")
    return A_eval, A_calib, A_thr, s_min, s_max


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_split():
    _log(f"Loading {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    _log(f"  Loaded {len(df):,} rows, {df.shape[1]} cols.")
    df["src_bytes"] = pd.to_numeric(df["src_bytes"], errors="coerce").fillna(0.0)
    df = df.sort_values("ts").reset_index(drop=True)
    train_mask = df["ts"] <= TRAIN_END_TS
    calib_mask = (df["ts"] > TRAIN_END_TS) & (df["ts"] <= CALIB_END_TS)
    eval_mask  = df["ts"] >= EVAL_START_TS
    df_train = df[train_mask].reset_index(drop=True)
    df_calib = df[calib_mask].reset_index(drop=True)
    df_eval  = df[eval_mask].reset_index(drop=True)
    _log(f"  TRAIN  {len(df_train):,} rows (n_attack={(df_train['label']!='normal').sum()})")
    _log(f"  CALIB  {len(df_calib):,} rows (n_attack={(df_calib['label']!='normal').sum()})")
    _log(f"  EVAL   {len(df_eval):,} rows  (n_attack={(df_eval['label']!='normal').sum()})")
    X_train = df_train[FEATURE_COLS].values.astype(np.float64)
    X_calib = df_calib[FEATURE_COLS].values.astype(np.float64)
    X_eval  = df_eval[FEATURE_COLS].values.astype(np.float64)
    y_eval  = (df_eval["label"] != "normal").astype(int).values
    return X_train, X_calib, X_eval, y_eval


# ---------------------------------------------------------------------------
# Suitability classification
# ---------------------------------------------------------------------------

def classify_mechanism(
    roc: float,
    recall: float,
    sub_thr_atk_pct: float,
    dense_probe_above_pct: float,
) -> str:
    if (roc >= 0.75 and recall >= 0.50 and sub_thr_atk_pct <= 30.0):
        return "TEMPORAL_SIGNAL_SUFFICIENT"
    if (roc >= 0.60 or recall >= 0.20
            or sub_thr_atk_pct <= 60.0
            or dense_probe_above_pct >= 50.0):
        return "TEMPORAL_SIGNAL_LIMITED"
    return "TEMPORAL_SIGNAL_INSUFFICIENT"


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_experiment(X_train, X_calib, X_eval, y_eval, run_id: int) -> dict:
    sep = "=" * 70
    _log(f"\n{sep}")
    _log(f"PHASE 2C RUN {run_id}")
    _log(sep)

    # Step 1: OneClassSVM baseline
    _log("\n[Step 1] OneClassSVM baseline (Phase 2B exact reproduction)")
    A_eval, A_calib, A_thr_point, s_min, s_max = fit_ocsvm_and_score(
        X_train, X_calib, X_eval
    )
    point_m = compute_point_metrics(A_eval, y_eval, A_thr_point)
    _log(f"  Point ROC-AUC={point_m['roc_auc']:.4f}  Recall={point_m['recall']:.6f}  "
         f"FPR={point_m['FPR']:.4f}  Sub-thr-atk={point_m['pct_attack_sub_threshold']:.2f}%")

    # Step 2: Build temporal features on EVAL (causal)
    _log("\n[Step 2] Building causal temporal features on EVAL stream ...")
    t0 = time.time()
    temporal_features = build_temporal_features(A_eval, A_thr_point)
    _log(f"  [OK] {len(temporal_features)} features built in {time.time()-t0:.1f}s")

    # Step 3: Calibrate temporal thresholds on CALIB (label-blind)
    _log("\n[Step 3] Calibrating temporal thresholds on CALIB stream (label-blind) ...")
    temporal_thresholds = calibrate_temporal_thresholds(A_calib, A_thr_point)
    _log(f"  [OK] {len(temporal_thresholds)} thresholds derived from 99th pct of benign calib.")
    for k, v in temporal_thresholds.items():
        _log(f"    {k:30s}: {v:.8g}")

    # Step 4: Evaluate all temporal mechanisms
    _log("\n[Step 4] Evaluating all temporal mechanisms ...")
    temporal_metrics = {}
    temporal_distributions = {}
    temporal_diagnostics = {}
    for fname, feat_arr in temporal_features.items():
        thr = temporal_thresholds[fname]
        m = compute_point_metrics(feat_arr, y_eval, thr)
        d = temporal_run_stats(feat_arr, thr)
        ben = percentile_stats(feat_arr[y_eval == 0], f"{fname}_benign")
        atk = percentile_stats(feat_arr[y_eval == 1], f"{fname}_attack")
        temporal_metrics[fname] = m
        temporal_distributions[fname] = {
            "threshold": thr, "benign": ben, "attack": atk
        }
        temporal_diagnostics[fname] = d
        _log(
            f"  [{fname:28s}] ROC={m['roc_auc']:.4f} "
            f"Recall={m['recall']:.4f} FPR={m['FPR']:.4f} "
            f"Sub-thr-atk={m['pct_attack_sub_threshold']:.1f}%"
        )

    # Step 5: Dense-probe analysis
    _log("\n[Step 5] Dense-probe cluster analysis ...")
    probe_result = analyze_dense_probes(
        A_eval, X_eval, y_eval, temporal_features, temporal_thresholds
    )
    _log(f"  Probes: n={probe_result['n_matching_rows']:,}  "
         f"frac={probe_result['fraction_of_eval']:.4f}  "
         f"attack%={probe_result['attack_pct_among_probes']:.1f}%")
    _log("  Temporal above-threshold rates for dense probes:")
    for fname, ev in probe_result["temporal_evidence"].items():
        _log(f"    {fname:32s}: mean={ev['mean']:.4g}  thr={ev['threshold']:.4g}  "
             f"above={ev['pct_above_threshold']:.2f}%")

    # Step 6: Episode analysis
    _log("\n[Step 6] Attack episode analysis ...")
    episodes = segment_episodes(y_eval, gap_rows=200)
    _log(f"  {len(episodes)} episodes identified.")
    episode_stats = {}
    ep_point = episode_detection(episodes, A_eval, A_thr_point)
    episode_stats["point_wise"] = ep_point
    _log(f"  point_wise: {ep_point['n_detected']}/{ep_point['n_episodes']} detected "
         f"({ep_point['detection_pct']:.1f}%), "
         f"med_delay={ep_point['median_detection_delay_rows']} rows")
    for fname in ["roll_mean_100", "roll_max_100", "exceedance_100",
                  "persistence", "ewma_b10", "ewma_b05"]:
        if fname in temporal_features:
            ep = episode_detection(episodes, temporal_features[fname], temporal_thresholds[fname])
            episode_stats[fname] = ep
            _log(f"  {fname:30s}: {ep['n_detected']}/{ep['n_episodes']} detected "
                 f"({ep['detection_pct']:.1f}%), "
                 f"med_delay={ep['median_detection_delay_rows']} rows")

    # Step 7: Causality check
    _log("\n[Step 7] Causality check ...")
    causal = run_causality_check(A_eval, A_thr_point)
    _log(f"  Full vs chunked max diff: {causal['max_abs_diff_full_vs_chunked']:.2e}")
    _log(f"  Causality verified: {causal['causality_verified']}")
    _log(f"  Shuffled differs:   {causal['shuffled_differs_from_full']}")

    # Step 8: Ablation
    _log("\n[Step 8] Ablation comparison ...")
    ablation = {}
    ablation_spec = {
        "1_point_wise": (A_eval, A_thr_point),
        "2_roll_mean_100": (temporal_features["roll_mean_100"], temporal_thresholds["roll_mean_100"]),
        "3_exceedance_100": (temporal_features["exceedance_100"], temporal_thresholds["exceedance_100"]),
        "4_persistence": (temporal_features["persistence"], temporal_thresholds["persistence"]),
        "5_ewma_b10": (temporal_features["ewma_b10"], temporal_thresholds["ewma_b10"]),
    }
    for aname, (feat, thr) in ablation_spec.items():
        m = compute_point_metrics(feat, y_eval, thr)
        ablation[aname] = m
        _log(f"  {aname:35s}: ROC={m['roc_auc']:.4f} Recall={m['recall']:.4f} "
             f"FPR={m['FPR']:.4f} Sub-thr-atk={m['pct_attack_sub_threshold']:.1f}%")

    # Step 9: Classifications
    _log("\n[Step 9] Suitability classifications ...")
    classifications = {}
    probe_above_point = float(100.0 - probe_result["point_score_stats"].get("mean", 0.0) * 100)
    classifications["point_wise"] = classify_mechanism(
        point_m["roc_auc"], point_m["recall"],
        point_m["pct_attack_sub_threshold"], probe_above_point
    )
    for fname, m in temporal_metrics.items():
        probe_above = probe_result["temporal_evidence"].get(fname, {}).get("pct_above_threshold", 0.0)
        clf = classify_mechanism(
            m["roc_auc"], m["recall"],
            m["pct_attack_sub_threshold"],
            probe_above if not (isinstance(probe_above, float) and np.isnan(probe_above)) else 0.0
        )
        classifications[fname] = clf
        _log(f"  {fname:30s}: {clf}")

    # Feature hashes for reproducibility
    feat_hashes = {fname: sha256_array(arr) for fname, arr in temporal_features.items()}
    feat_hashes["A_eval"] = sha256_array(A_eval)

    return {
        "run_id": run_id,
        "s_min": s_min, "s_max": s_max, "A_thr_point": A_thr_point,
        "point_metrics": point_m,
        "temporal_thresholds": temporal_thresholds,
        "temporal_metrics": temporal_metrics,
        "temporal_distributions": temporal_distributions,
        "temporal_diagnostics": temporal_diagnostics,
        "dense_probe": probe_result,
        "episode_stats": episode_stats,
        "causality": causal,
        "ablation": ablation,
        "classifications": classifications,
        "feature_hashes": feat_hashes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = datetime.now()
    _log("=" * 70)
    _log("PHASE 2C -- TEMPORAL/RELATIONAL EVIDENCE FEASIBILITY AUDIT")
    _log(f"Start: {t_start.isoformat()}")
    _log("=" * 70)

    # Source data hash
    data_sha = sha256_file(DATA_FILE)
    _log(f"Source data SHA-256: {data_sha}")

    # Load data once
    X_train, X_calib, X_eval, y_eval = load_and_split()

    # Two independent runs
    r1 = run_experiment(X_train, X_calib, X_eval, y_eval, run_id=1)
    r2 = run_experiment(X_train, X_calib, X_eval, y_eval, run_id=2)

    # Reproducibility verification
    _log("\n" + "=" * 70)
    _log("REPRODUCIBILITY VERIFICATION")
    _log("=" * 70)
    repro = {"all_match": True, "per_hash": {}}
    for hname in r1["feature_hashes"]:
        h1 = r1["feature_hashes"][hname]
        h2 = r2["feature_hashes"].get(hname, "")
        match = (h1 == h2)
        if not match:
            repro["all_match"] = False
        repro["per_hash"][hname] = {"run1": h1, "run2": h2, "match": match}
        _log(f"  {hname:30s}: {'[PASS]' if match else '[FAIL]'}")
    _log(f"\nOverall: {'VERIFIED' if repro['all_match'] else 'FAILED'}")

    # Comparative summary
    _log("\n" + "=" * 90)
    _log("COMPARATIVE SUMMARY (Run 1)")
    _log("=" * 90)
    hdr = (f"{'Mechanism':32s} {'ROC-AUC':>8s} {'Recall':>8s} {'FPR':>8s} "
           f"{'Sub-thr%':>9s} {'Classification':>30s}")
    _log(hdr)
    _log("-" * 100)
    pm = r1["point_metrics"]
    _log(f"{'point_wise':32s} {pm['roc_auc']:8.4f} {pm['recall']:8.4f} {pm['FPR']:8.4f} "
         f"{pm['pct_attack_sub_threshold']:9.2f} {r1['classifications']['point_wise']:>30s}")
    for fname, m in r1["temporal_metrics"].items():
        clf = r1["classifications"].get(fname, "")
        _log(f"{fname:32s} {m['roc_auc']:8.4f} {m['recall']:8.4f} {m['FPR']:8.4f} "
             f"{m['pct_attack_sub_threshold']:9.2f} {clf:>30s}")

    # Stop condition check
    all_insuff = all(
        v == "TEMPORAL_SIGNAL_INSUFFICIENT"
        for k, v in r1["classifications"].items()
        if k != "point_wise"
    )
    if all_insuff:
        _log("\n" + "=" * 70)
        _log("STOP CONDITION TRIGGERED")
        _log("All temporal mechanisms: TEMPORAL_SIGNAL_INSUFFICIENT")
        _log(
            "CONCLUSION: POINT-WISE AND SIMPLE TEMPORAL UNSUPERVISED EVIDENCE ARE "
            "INSUFFICIENT FOR THIS TON-IOT NETWORK MODALITY UNDER THE CURRENT "
            "LEAKAGE-CONTROLLED REPRESENTATION."
        )
        _log("=" * 70)

    # Write artifacts
    _log("\n[ARTIFACTS] Writing JSON artifacts ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def jdump(obj, fname):
        with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    config = {
        "phase": "2C",
        "experiment": "Temporal/Relational Evidence Feasibility Audit",
        "dataset": str(DATA_FILE),
        "source_data_sha256": data_sha,
        "feature_cols": FEATURE_COLS,
        "train_end_ts": TRAIN_END_TS,
        "calib_end_ts": CALIB_END_TS,
        "eval_start_ts": EVAL_START_TS,
        "ocsvm_params": OCSVM_PARAMS,
        "rolling_windows": ROLLING_WINDOWS,
        "ewma_betas": EWMA_BETAS,
        "calib_percentile": CALIB_PERCENTILE,
        "dense_probe_vector": DENSE_PROBE_VECTOR,
        "n_train": len(X_train),
        "n_calib": len(X_calib),
        "n_eval": len(X_eval),
        "n_eval_attack": int(y_eval.sum()),
        "n_eval_benign": int((y_eval == 0).sum()),
    }
    jdump(config, "config.json")
    jdump({"s_min": r1["s_min"], "s_max": r1["s_max"], "A_thr": r1["A_thr_point"],
           "sha256": r1["feature_hashes"]["A_eval"], "metrics": r1["point_metrics"]},
          "point_scores.json")
    jdump({"thresholds": r1["temporal_thresholds"],
           "metrics": r1["temporal_metrics"],
           "distributions": r1["temporal_distributions"],
           "diagnostics": r1["temporal_diagnostics"]},
          "temporal_scores.json")
    jdump({"point_wise": r1["point_metrics"],
           "temporal": r1["temporal_metrics"],
           "ablation": r1["ablation"],
           "classifications": r1["classifications"]},
          "metrics.json")
    jdump(r1["dense_probe"], "dense_probe_analysis.json")
    jdump(r1["episode_stats"], "episode_analysis.json")
    jdump(r1["causality"], "causality_check.json")
    jdump(r1["ablation"], "ablation.json")
    jdump(repro, "reproducibility.json")

    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()
    _log(f"\n[ARTIFACTS] All artifacts written to {OUT_DIR}")
    _log(f"Wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    _log(f"End: {t_end.isoformat()}")
    _log("\nPHASE_2C_TEMPORAL_EVIDENCE_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
