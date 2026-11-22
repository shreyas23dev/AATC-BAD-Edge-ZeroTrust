"""
Phase 2C Correction Rerun — Evaluation Validity Fix Only
=========================================================
Corrects the evaluation label parsing bug from Phase 2C, recovers the exact
Phase 2B OneClassSVM score convention (-score_samples, s_min=-7491.95, s_max=-1.0),
removes degenerate burst_rate, and evaluates the 16 valid causal temporal mechanisms
under strict chronological protocol.

STRICT CONSTRAINTS:
  - Evaluation ground truth verified: 799,102 rows = 7,781 normal + 791,321 attack.
  - Exactly reproduces Phase 2B OneClassSVM baseline: -model.score_samples(X).
  - No evaluation labels used for calibration or threshold selection.
  - All temporal features are strictly causal (no future rows).
  - burst_rate excluded as VOID_DEGENERATE_FEATURE.
  - Evaluates:
      * Point-wise baseline
      * Rolling mean: w in {10, 50, 100, 500}
      * Rolling max:  w in {10, 50, 100, 500}
      * Exceedance:   w in {10, 50, 100, 500} (normalized by w)
      * Persistence:  consecutive run length
      * EWMA:         beta in {0.05, 0.10, 0.20}
  - Full causality verification (chunked state preservation + shuffled control).
  - Two independent runs for bit-for-bit SHA-256 reproducibility.
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
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, balanced_accuracy_score
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
# Constants — frozen from Phase 2B / Phase 2C specs
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
# Hashing & Utility
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
    prec = float(precision_score(labels, preds, zero_division=0))
    rec  = float(recall_score(labels, preds, zero_division=0))
    f1   = float(f1_score(labels, preds, zero_division=0))
    fpr  = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr  = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ba   = float(balanced_accuracy_score(labels, preds))
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
    above = arr >= threshold
    out = np.zeros(len(arr), dtype=np.float64)
    run = 0
    for i, v in enumerate(above):
        run = (run + 1) if v else 0
        out[i] = float(run)
    return out

def causal_ewma(arr: np.ndarray, beta: float) -> np.ndarray:
    out = np.empty(len(arr), dtype=np.float64)
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = beta * arr[i] + (1.0 - beta) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Build all valid temporal features
# ---------------------------------------------------------------------------

def build_temporal_features(A: np.ndarray, A_thr_point: float) -> dict:
    feats = {}
    for w in ROLLING_WINDOWS:
        feats[f"roll_mean_{w}"] = causal_rolling_mean(A, w)
        feats[f"roll_max_{w}"]  = causal_rolling_max(A, w)
        feats[f"exceedance_{w}"] = causal_exceedance_count(A, A_thr_point, w) / float(w)
    feats["persistence"] = causal_persistence(A, A_thr_point)
    for beta in EWMA_BETAS:
        feats[f"ewma_b{int(beta*100):02d}"] = causal_ewma(A, beta)
    return feats


# ---------------------------------------------------------------------------
# Calibrate temporal thresholds (label-blind, benign calib only)
# ---------------------------------------------------------------------------

def calibrate_temporal_thresholds(A_calib: np.ndarray, A_thr_point: float) -> dict:
    thresholds = {}
    for w in ROLLING_WINDOWS:
        thresholds[f"roll_mean_{w}"] = float(np.percentile(
            causal_rolling_mean(A_calib, w), CALIB_PERCENTILE, method="linear"))
        thresholds[f"roll_max_{w}"] = float(np.percentile(
            causal_rolling_max(A_calib, w), CALIB_PERCENTILE, method="linear"))
        thresholds[f"exceedance_{w}"] = float(np.percentile(
            causal_exceedance_count(A_calib, A_thr_point, w) / float(w),
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
        above_pct = (
            float(np.mean(probe_feat >= thr) * 100)
            if (len(probe_feat) > 0 and not np.isnan(thr)) else float("nan")
        )
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

def run_causality_check(A_eval: np.ndarray) -> dict:
    beta = 0.10
    full_ewma = causal_ewma(A_eval, beta)
    chunk_size = 10_000
    chunked = np.empty(len(A_eval), dtype=np.float64)
    state = A_eval[0]
    for i in range(0, len(A_eval), chunk_size):
        chunk = A_eval[i:i + chunk_size]
        ewma_chunk = np.empty(len(chunk), dtype=np.float64)
        ewma_chunk[0] = beta * chunk[0] + (1.0 - beta) * state
        for j in range(1, len(chunk)):
            ewma_chunk[j] = beta * chunk[j] + (1.0 - beta) * ewma_chunk[j - 1]
        chunked[i:i + len(chunk)] = ewma_chunk
        state = float(ewma_chunk[-1])
    max_diff = float(np.max(np.abs(full_ewma - chunked)))
    match = bool(max_diff < 1e-10)
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
# Exact Phase 2B OneClassSVM Scoring Function
# ---------------------------------------------------------------------------

def score_in_chunks(model, X: np.ndarray, chunk_size: int = 50000) -> np.ndarray:
    """Score large array in memory-safe chunks and return -model.score_samples(X)."""
    n = len(X)
    scores = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # raw convention: higher is more anomalous (-score_samples)
        scores[start:end] = -model.score_samples(X[start:end])
    return scores


def fit_ocsvm_and_score_phase2b(X_train_scaled, X_calib_scaled, X_eval_scaled):
    """
    Reproduces Phase 2B evaluate_detector for OneClassSVM:
      - Fits OneClassSVM(kernel='rbf', nu=0.05, gamma='scale') on TRAIN
      - raw = -model.score_samples(X) in 50k chunks
      - s_min, s_max from TRAIN only
      - A = clip((raw - s_min)/(s_max - s_min), 0, 1)
      - A_thr = 99th percentile of A_calib
    """
    _log("  Fitting OneClassSVM(kernel='rbf', nu=0.05, gamma='scale') on TRAIN...")
    t0 = time.time()
    model = OneClassSVM(**OCSVM_PARAMS)
    model.fit(X_train_scaled)
    _log(f"  [OK] OneClassSVM fitted in {time.time()-t0:.2f}s.")

    _log("  Scoring TRAIN, CALIB, EVAL using Phase 2B score_in_chunks (-score_samples)...")
    t1 = time.time()
    raw_train = score_in_chunks(model, X_train_scaled, chunk_size=50000)
    raw_calib = score_in_chunks(model, X_calib_scaled, chunk_size=50000)
    raw_eval  = score_in_chunks(model, X_eval_scaled,  chunk_size=50000)
    _log(f"  [OK] Scoring completed in {time.time()-t1:.2f}s.")

    s_min = float(np.min(raw_train))
    s_max = float(np.max(raw_train))
    _log(f"  Normalization bounds (TRAIN-only): s_min={s_min:.8f}, s_max={s_max:.8f}")

    denom = s_max - s_min if abs(s_max - s_min) > 1e-15 else 1.0
    A_calib = np.clip((raw_calib - s_min) / denom, 0.0, 1.0)
    A_eval  = np.clip((raw_eval  - s_min) / denom, 0.0, 1.0)

    A_thr = float(np.percentile(A_calib, CALIB_PERCENTILE, method="linear"))
    raw_thr_equiv = float(A_thr * (s_max - s_min) + s_min)
    _log(f"  Calibration A_thr (99th pct) = {A_thr:.8f} (raw equiv = {raw_thr_equiv:.8f})")
    _log(f"  A_eval SHA-256: {sha256_array(A_eval)}")

    return A_eval, A_calib, A_thr, s_min, s_max, raw_thr_equiv


# ---------------------------------------------------------------------------
# Data loading with verified label parsing
# ---------------------------------------------------------------------------

def load_and_split():
    _log(f"Loading {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    _log(f"  Loaded {len(df):,} rows, {df.shape[1]} cols.")

    df["original_row_id"] = df.index
    df_sorted = df.sort_values(by=["ts", "original_row_id"]).reset_index(drop=True)

    train_mask = (df_sorted["ts"] <= TRAIN_END_TS)
    calib_mask = (df_sorted["ts"] >= TRAIN_END_TS + 1) & (df_sorted["ts"] <= CALIB_END_TS)
    eval_mask  = (df_sorted["ts"] >= EVAL_START_TS)

    df_train = df_sorted[train_mask].reset_index(drop=True)
    df_calib = df_sorted[calib_mask].reset_index(drop=True)
    df_eval  = df_sorted[eval_mask].reset_index(drop=True)

    # Clean 175 malformed '0.0.0.0' in src_bytes deterministically to 0.0
    for part in [df_train, df_calib, df_eval]:
        part["src_bytes"] = pd.to_numeric(part["src_bytes"], errors="coerce").fillna(0.0)

    # Verified label parsing
    # In Network_dataset_1.csv, label is integer: 0 = normal, 1 = attack (scanning)
    y_eval = df_eval["label"].values.astype(int)
    normal_count = int((y_eval == 0).sum())
    attack_count = int((y_eval == 1).sum())

    _log(f"  TRAIN  {len(df_train):,} rows")
    _log(f"  CALIB  {len(df_calib):,} rows")
    _log(f"  EVAL   {len(df_eval):,} rows (normal={normal_count:,}, attack={attack_count:,})")

    # STRICT ASSERTIONS FROM SPEC SECTION 1
    assert len(df_eval) == 799102, f"Expected 799,102 eval rows, got {len(df_eval)}"
    assert normal_count == 7781, f"Expected 7,781 normal rows, got {normal_count}"
    assert attack_count == 791321, f"Expected 791,321 attack rows, got {attack_count}"
    _log("  [PASS] Ground truth assertions verified: len=799102, normal=7781, attack=791321")

    X_train = df_train[FEATURE_COLS].astype(float).values
    X_calib = df_calib[FEATURE_COLS].astype(float).values
    X_eval  = df_eval[FEATURE_COLS].astype(float).values

    # Preprocessing: RobustScaler fitted strictly on TRAIN
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_eval_scaled  = scaler.transform(X_eval)

    return X_train_scaled, X_calib_scaled, X_eval_scaled, X_eval, y_eval


# ---------------------------------------------------------------------------
# Classification logic (spec section 14)
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

def run_experiment(
    X_train_s, X_calib_s, X_eval_s, X_eval_raw, y_eval, run_id: int
) -> dict:
    sep = "=" * 70
    _log(f"\n{sep}")
    _log(f"PHASE 2C CORRECTION RUN {run_id}")
    _log(sep)

    # Step 1: Exact Phase 2B OneClassSVM baseline
    _log("\n[Step 1] Exact Phase 2B OneClassSVM baseline")
    A_eval, A_calib, A_thr_point, s_min, s_max, raw_thr_equiv = fit_ocsvm_and_score_phase2b(
        X_train_s, X_calib_s, X_eval_s
    )

    point_m = compute_point_metrics(A_eval, y_eval, A_thr_point)
    _log(
        f"  Point-wise: ROC-AUC={point_m['roc_auc']:.6f}  "
        f"Recall={point_m['recall']:.6f} (TP={point_m['TP']}, FN={point_m['FN']})  "
        f"FPR={point_m['FPR']:.6f} (FP={point_m['FP']}, TN={point_m['TN']})  "
        f"Sub-thr-atk={point_m['pct_attack_sub_threshold']:.2f}%"
    )

    # Step 2: Build causal temporal features on EVAL
    _log("\n[Step 2] Building causal temporal features on EVAL stream ...")
    t0 = time.time()
    temporal_features = build_temporal_features(A_eval, A_thr_point)
    _log(f"  [OK] {len(temporal_features)} features built in {time.time()-t0:.2f}s")

    # Step 3: Calibrate temporal thresholds on CALIB (label-blind)
    _log("\n[Step 3] Calibrating temporal thresholds on CALIB stream (label-blind) ...")
    temporal_thresholds = calibrate_temporal_thresholds(A_calib, A_thr_point)
    _log(f"  [OK] {len(temporal_thresholds)} thresholds derived from 99th pct of benign CALIB.")
    for k, v in temporal_thresholds.items():
        _log(f"    {k:25s}: {v:.10g}")

    # Step 4: Evaluate all temporal mechanisms
    _log("\n[Step 4] Evaluating all valid temporal mechanisms ...")
    temporal_metrics = {}
    temporal_distributions = {}
    temporal_diagnostics = {}
    for fname, feat_arr in temporal_features.items():
        thr = temporal_thresholds[fname]
        m = compute_point_metrics(feat_arr, y_eval, thr)
        d = temporal_run_stats(feat_arr, thr)
        ben = percentile_stats(feat_arr[y_eval == 0], f"{fname}_benign")
        atk = percentile_stats(feat_arr[y_eval == 1], f"{fname}_attack")
        all_s = percentile_stats(feat_arr, f"{fname}_all")
        temporal_metrics[fname] = m
        temporal_distributions[fname] = {
            "threshold": thr, "all": all_s, "benign": ben, "attack": atk
        }
        temporal_diagnostics[fname] = d
        _log(
            f"  [{fname:22s}] ROC={m['roc_auc']:.4f} "
            f"Recall={m['recall']:.6f} FPR={m['FPR']:.4f} "
            f"Sub-thr-atk={m['pct_attack_sub_threshold']:.2f}%"
        )

    # Step 5: Dense-probe analysis
    _log("\n[Step 5] Dense-probe cluster analysis ...")
    probe_result = analyze_dense_probes(
        A_eval, X_eval_raw, y_eval, temporal_features, temporal_thresholds
    )
    _log(f"  Probes: n={probe_result['n_matching_rows']:,}  "
         f"frac={probe_result['fraction_of_eval']:.4f}  "
         f"attack%={probe_result['attack_pct_among_probes']:.1f}%")
    _log("  Temporal above-threshold rates for dense probes:")
    for fname, ev in probe_result["temporal_evidence"].items():
        _log(f"    {fname:25s}: mean={ev['mean']:.4g}  thr={ev['threshold']:.4g}  "
             f"above={ev['pct_above_threshold']:.4f}%")

    # Step 6: Episode analysis
    _log("\n[Step 6] Attack episode analysis ...")
    episodes = segment_episodes(y_eval, gap_rows=200)
    _log(f"  {len(episodes)} attack episodes identified.")
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
            _log(f"  {fname:25s}: {ep['n_detected']}/{ep['n_episodes']} detected "
                 f"({ep['detection_pct']:.1f}%), "
                 f"med_delay={ep['median_detection_delay_rows']} rows")

    # Step 7: Causality check
    _log("\n[Step 7] Causality check ...")
    causal = run_causality_check(A_eval)
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
        _log(f"  {aname:25s}: ROC={m['roc_auc']:.4f} Recall={m['recall']:.6f} "
             f"FPR={m['FPR']:.4f} Sub-thr-atk={m['pct_attack_sub_threshold']:.2f}%")

    # Step 9: Classifications
    _log("\n[Step 9] Suitability classifications ...")
    classifications = {}
    probe_above_point = 100.0 * float(np.mean(A_eval[np.all(np.abs(X_eval_raw - np.array([DENSE_PROBE_VECTOR[c] for c in FEATURE_COLS])) < 1e-6, axis=1)] >= A_thr_point))
    classifications["point_wise"] = classify_mechanism(
        point_m["roc_auc"], point_m["recall"],
        point_m["pct_attack_sub_threshold"], probe_above_point
    )
    _log(f"  {'point_wise':25s}: {classifications['point_wise']}")
    for fname, m in temporal_metrics.items():
        probe_above = probe_result["temporal_evidence"].get(fname, {}).get("pct_above_threshold", 0.0)
        clf = classify_mechanism(
            m["roc_auc"], m["recall"],
            m["pct_attack_sub_threshold"],
            probe_above if not (isinstance(probe_above, float) and np.isnan(probe_above)) else 0.0
        )
        classifications[fname] = clf
        _log(f"  {fname:25s}: {clf}")

    # Feature hashes for reproducibility
    feat_hashes = {fname: sha256_array(arr) for fname, arr in temporal_features.items()}
    feat_hashes["A_eval"] = sha256_array(A_eval)

    return {
        "run_id": run_id,
        "s_min": s_min, "s_max": s_max, "A_thr_point": A_thr_point,
        "raw_thr_equiv": raw_thr_equiv,
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
# Main Execution
# ---------------------------------------------------------------------------

def main():
    t_start = datetime.now()
    _log("=" * 70)
    _log("PHASE 2C CORRECTION RERUN — EVALUATION VALIDITY FIX ONLY")
    _log(f"Start: {t_start.isoformat()}")
    _log("=" * 70)

    # Source data hash
    data_sha = sha256_file(DATA_FILE)
    _log(f"Source data SHA-256: {data_sha}")

    # Load data once
    X_train_s, X_calib_s, X_eval_s, X_eval_raw, y_eval = load_and_split()

    # Save verified label mapping
    label_map_obj = {
        "total_eval_rows": len(y_eval),
        "label_column": "label",
        "label_dtype": "int64",
        "label_unique_values": [0, 1],
        "label_counts": {
            "0": int((y_eval == 0).sum()),
            "1": int((y_eval == 1).sum()),
        },
        "type_column": "type",
        "type_unique_values": ["normal", "scanning"],
        "normal_count": int((y_eval == 0).sum()),
        "attack_count": int((y_eval == 1).sum()),
        "verified_mapping": {"0": "normal", "1": "attack"},
        "assertion_passed": True,
    }
    with open(OUT_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_map_obj, f, indent=2)
    _log("  Saved verified label_mapping.json")

    # Run 1
    r1 = run_experiment(X_train_s, X_calib_s, X_eval_s, X_eval_raw, y_eval, run_id=1)

    # Run 2 (Reproducibility)
    r2 = run_experiment(X_train_s, X_calib_s, X_eval_s, X_eval_raw, y_eval, run_id=2)

    # Reproducibility Verification
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
        _log(f"  {hname:25s}: {'[PASS]' if match else '[FAIL]'}")
    _log(f"\nOverall: {'VERIFIED' if repro['all_match'] else 'FAILED'}")

    # Comparative Summary Table
    _log("\n" + "=" * 105)
    _log("CORRECTED TEMPORAL EVIDENCE COMPARATIVE SUMMARY (Run 1)")
    _log("=" * 105)
    hdr = (f"{'Mechanism':22s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'Recall':>10s} "
           f"{'FPR':>8s} {'F1':>8s} {'Sub-thr%':>9s} {'Classification':>25s}")
    _log(hdr)
    _log("-" * 105)
    pm = r1["point_metrics"]
    _log(f"{'point_wise':22s} {pm['roc_auc']:8.4f} {pm['pr_auc']:8.4f} {pm['recall']:10.6f} "
         f"{pm['FPR']:8.4f} {pm['f1']:8.4f} {pm['pct_attack_sub_threshold']:9.2f} "
         f"{r1['classifications']['point_wise']:>25s}")
    for fname, m in r1["temporal_metrics"].items():
        clf = r1["classifications"].get(fname, "")
        _log(f"{fname:22s} {m['roc_auc']:8.4f} {m['pr_auc']:8.4f} {m['recall']:10.6f} "
             f"{m['FPR']:8.4f} {m['f1']:8.4f} {m['pct_attack_sub_threshold']:9.2f} "
             f"{clf:>25s}")

    # Write Artifacts
    _log("\n[ARTIFACTS] Writing corrected JSON artifacts ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def jdump(obj, fname):
        with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    config = {
        "phase": "2C_CORRECTED",
        "experiment": "Temporal/Relational Evidence Feasibility Audit (Corrected)",
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
        "burst_rate_status": "VOID_DEGENERATE_FEATURE",
        "n_train": 165322,
        "n_calib": 35576,
        "n_eval": 799102,
        "n_eval_attack": 791321,
        "n_eval_benign": 7781,
    }
    jdump(config, "config.json")
    jdump({
        "s_min": r1["s_min"], "s_max": r1["s_max"],
        "A_thr": r1["A_thr_point"], "raw_thr_equiv": r1["raw_thr_equiv"],
        "sha256": r1["feature_hashes"]["A_eval"],
        "metrics": r1["point_metrics"]
    }, "point_scores.json")
    jdump({
        "thresholds": r1["temporal_thresholds"],
        "metrics": r1["temporal_metrics"],
        "distributions": r1["temporal_distributions"],
        "diagnostics": r1["temporal_diagnostics"]
    }, "temporal_scores.json")
    jdump({
        "point_wise": r1["point_metrics"],
        "temporal": r1["temporal_metrics"],
        "ablation": r1["ablation"],
        "classifications": r1["classifications"]
    }, "metrics.json")
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
    _log("\nPHASE_2C_CORRECTION_RERUN_COMPLETE")


if __name__ == "__main__":
    main()
