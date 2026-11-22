"""
Phase 2A.1 — Temporal Feature Representation Audit
===================================================
Project: C:\\Users\\Shreyas A\\PROJECTS\\BAZT-TON_IoT\\TON_IoT_project
Protocol: E1 Chronological Split (Continuous April 23-29 evaluation interval)

Evaluates whether a predefined set of causal temporal features provides
materially more informative anomaly scores than the four instantaneous
Modbus register features.

Representations:
  A. STATIC BASELINE (4 instantaneous features)
  B. TEMPORAL REPRESENTATION (29 causal temporal features)

Detector:
  IsolationForest(n_estimators=300, max_samples=256, max_features=1.0, bootstrap=False, random_state=42)
  Fitted strictly on benign TRAIN.
  RobustScaler fitted strictly on benign TRAIN.
  Threshold A_thr from 99th percentile of benign CALIBRATION.
  Evaluation over complete 173,506 rows.
"""

import sys
import os
import json
import hashlib
import logging
import traceback
import warnings
from datetime import datetime
from pathlib import Path

# Safe stdout reconfigure
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
EDGEIIOT_ROOT = PROJECT_ROOT / "edge-iiot-anomaly-detection"
sys.path.insert(0, str(EDGEIIOT_ROOT))

from config.settings import config
from src.data.ton_iot_adapter import TonIoTDataAdapter, TonIoTChronologicalSplitter, TON_IOT_FEATURE_COLUMNS

OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = OUT_DIR / "run_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("phase2a1_audit")

MODBUS_CSV = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_audit\iot_processed\IoT_Modbus.csv")

# ── Frozen Constants ──────────────────────────────────────────────────────────
EXPECTED_TRAIN = 100_940
EXPECTED_CALIB = 11_691
EXPECTED_EVAL = 173_506
EXPECTED_EVAL_BENIGN = 109_167
EXPECTED_EVAL_ATTACK = 64_339

STATIC_FEATURES = [
    "FC1_Read_Input_Register",
    "FC2_Read_Discrete_Value",
    "FC3_Read_Holding_Register",
    "FC4_Read_Coil",
]

DETECTOR_CFG = {
    "n_estimators": 300,
    "max_samples": 256,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}


def sha256_of_array(arr: np.ndarray) -> str:
    arr64 = np.asarray(arr, dtype=np.float64)
    return hashlib.sha256(arr64.tobytes()).hexdigest()


def score_distribution_stats(scores: np.ndarray, label: str) -> dict:
    return {
        "label": label,
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "max": float(np.max(scores)),
    }


def lag1_autocorr(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def first_diff_volatility(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return float("nan")
    return float(np.std(np.diff(arr)))


def run_above_thr_runs(arr: np.ndarray, a_thr: float):
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] >= a_thr:
            start = i
            while i < n and arr[i] >= a_thr:
                i += 1
            runs.append((start, i - 1, i - start))
        else:
            i += 1
    return runs


def run_episodes(labels: np.ndarray, scores: np.ndarray, a_thr: float):
    episodes = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            start = i
            while i < n and labels[i] == 1:
                i += 1
            end = i
            ep_scores = scores[start:end]
            pre_score = float(scores[start - 1]) if start > 0 else float("nan")
            episodes.append({
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "duration": int(end - start),
                "pre_onset_score": pre_score,
                "mean_score": float(np.mean(ep_scores)),
                "max_score": float(np.max(ep_scores)),
                "any_above_thr": bool(np.any(ep_scores >= a_thr)),
                "peak_offset": int(np.argmax(ep_scores)),
            })
        else:
            i += 1
    return episodes


def build_temporal_features(df_partition: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Construct strictly causal temporal features within a single partition.
    Boundary rule: History begins strictly at the start of df_partition.
    No future rows used. No cross-partition rows used.
    
    Features constructed:
      1. 4 current register values (x_t)
      2. 4 first differences (x_t - x_{t-1}, t=0 filled with 0.0)
      3. 4 absolute first differences (|x_t - x_{t-1}|, t=0 filled with 0.0)
      4. 1 inter-arrival time (delta_t = ts_t - ts_{t-1}, t=0 filled with 0.0)
      5. 4 rolling means over short window (w=5, min_periods=1)
      6. 4 rolling variances over short window (w=5, min_periods=1, t=0 var filled with 0.0)
      7. 4 rolling means over medium window (w=20, min_periods=1)
      8. 4 rolling variances over medium window (w=20, min_periods=1, t=0 var filled with 0.0)
    Total = 29 features.
    """
    feats = pd.DataFrame(index=df_partition.index)

    # 1. Instantaneous register values
    for col in STATIC_FEATURES:
        feats[col] = pd.to_numeric(df_partition[col], errors="raise").astype(float)

    # 2. First difference & 3. Absolute first difference
    for col in STATIC_FEATURES:
        diff_s = feats[col].diff().fillna(0.0)
        feats[f"diff_{col}"] = diff_s
        feats[f"abs_diff_{col}"] = diff_s.abs()

    # 4. Inter-arrival time (in seconds)
    ts = pd.to_datetime(df_partition["timestamp"])
    dt_seconds = ts.diff().dt.total_seconds().fillna(0.0)
    feats["inter_arrival_time"] = dt_seconds

    # 5. Short window (w=5) rolling mean and variance
    for col in STATIC_FEATURES:
        feats[f"roll_mean_w5_{col}"] = feats[col].rolling(window=5, min_periods=1).mean()
        feats[f"roll_var_w5_{col}"] = feats[col].rolling(window=5, min_periods=1).var().fillna(0.0)

    # 6. Medium window (w=20) rolling mean and variance
    for col in STATIC_FEATURES:
        feats[f"roll_mean_w20_{col}"] = feats[col].rolling(window=20, min_periods=1).mean()
        feats[f"roll_var_w20_{col}"] = feats[col].rolling(window=20, min_periods=1).var().fillna(0.0)

    feature_names = list(feats.columns)
    return feats, feature_names


def evaluate_representation(
    name: str,
    X_train_df: pd.DataFrame,
    X_calib_df: pd.DataFrame,
    X_eval_df: pd.DataFrame,
    y_eval: np.ndarray,
    seed: int,
) -> dict:
    """Run full fitting, calibration, normalization, and evaluation for one representation."""
    log.info(f"\n--- Evaluating Representation: {name} (N_features={X_train_df.shape[1]}) ---")

    # 1. Preprocessing: Fit RobustScaler strictly on TRAIN
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_calib_scaled = scaler.transform(X_calib_df)
    X_eval_scaled = scaler.transform(X_eval_df)

    # 2. Detector Fitting: Train Isolation Forest strictly on TRAIN
    cfg = {**DETECTOR_CFG, "random_state": seed}
    model = IsolationForest(**cfg)
    model.fit(X_train_scaled)

    # 3. Raw Scores (raw = -score_samples)
    raw_train = -model.score_samples(X_train_scaled)
    raw_calib = -model.score_samples(X_calib_scaled)
    raw_eval = -model.score_samples(X_eval_scaled)

    # 4. Score Normalization bounds from TRAIN ONLY
    s_min = float(np.min(raw_train))
    s_max = float(np.max(raw_train))
    if s_max <= s_min:
        s_max = s_min + 1e-10

    A_train = np.clip((raw_train - s_min) / (s_max - s_min), 0.0, 1.0)
    A_calib = np.clip((raw_calib - s_min) / (s_max - s_min), 0.0, 1.0)
    A_eval = np.clip((raw_eval - s_min) / (s_max - s_min), 0.0, 1.0)

    # 5. Threshold A_thr from 99th percentile of benign CALIBRATION
    A_thr = float(np.percentile(A_calib, 99.0, method="linear"))
    raw_thr_equiv = float(A_thr * (s_max - s_min) + s_min)

    # 6. Evaluation metrics (post-hoc joining labels)
    y_pred = (A_eval >= A_thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
    precision = float(precision_score(y_eval, y_pred, zero_division=0))
    recall = float(recall_score(y_eval, y_pred, zero_division=0))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_eval, y_pred))
    roc_auc = float(roc_auc_score(y_eval, A_eval))
    pr_auc = float(average_precision_score(y_eval, A_eval))

    A_eval_benign = A_eval[y_eval == 0]
    A_eval_attack = A_eval[y_eval == 1]

    stats_benign = score_distribution_stats(A_eval_benign, f"{name} eval_benign")
    stats_attack = score_distribution_stats(A_eval_attack, f"{name} eval_attack")

    sub_atk_mask = (y_eval == 1) & (A_eval < A_thr)
    above_atk_mask = (y_eval == 1) & (A_eval >= A_thr)
    n_sub_atk = int(sub_atk_mask.sum())
    n_above_atk = int(above_atk_mask.sum())
    pct_sub_atk = float(100.0 * n_sub_atk / len(A_eval_attack))

    score_hash = sha256_of_array(A_eval)

    # Temporal & Episode analysis
    ac_all = lag1_autocorr(A_eval)
    vol_all = first_diff_volatility(A_eval)
    above_runs = run_above_thr_runs(A_eval, A_thr)
    n_above_runs = len(above_runs)
    run_lens = [r[2] for r in above_runs]
    mean_run_len = float(np.mean(run_lens)) if run_lens else 0.0
    max_run_len = int(np.max(run_lens)) if run_lens else 0

    episodes = run_episodes(y_eval, A_eval, A_thr)
    n_ep = len(episodes)
    ep_with_any_above = sum(1 for e in episodes if e["any_above_thr"])
    ep_durations = [e["duration"] for e in episodes]
    ep_means = [e["mean_score"] for e in episodes]
    ep_maxs = [e["max_score"] for e in episodes]

    log.info(f"  [{name}] ROC-AUC={roc_auc:.6f}, PR-AUC={pr_auc:.6f}, F1={f1:.6f}, Recall={recall:.6f}")
    log.info(f"  [{name}] Benign Mean A={stats_benign['mean']:.6f}, Attack Mean A={stats_attack['mean']:.6f}")
    log.info(f"  [{name}] Sub-threshold attacks: {n_sub_atk:,} ({pct_sub_atk:.2f}%)")
    log.info(f"  [{name}] Episodes crossing A_thr: {ep_with_any_above:,}/{n_ep:,} ({100*ep_with_any_above/n_ep:.1f}%)")
    log.info(f"  [{name}] Score Hash: {score_hash}")

    return {
        "representation_name": name,
        "n_features": int(X_train_df.shape[1]),
        "feature_names": list(X_train_df.columns),
        "s_min": s_min,
        "s_max": s_max,
        "A_thr": A_thr,
        "raw_thr_equiv": raw_thr_equiv,
        "point_metrics": {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "precision": precision, "recall": recall, "f1": f1,
            "FPR": fpr, "FNR": fnr, "specificity": specificity,
            "balanced_accuracy": bal_acc, "roc_auc": roc_auc, "pr_auc": pr_auc,
        },
        "score_distributions": {
            "eval_benign": stats_benign,
            "eval_attack": stats_attack,
        },
        "sub_threshold": {
            "n_attack_sub_thr": n_sub_atk,
            "n_attack_above_thr": n_above_atk,
            "pct_attack_sub_thr": pct_sub_atk,
        },
        "temporal_characteristics": {
            "lag1_autocorr": ac_all,
            "first_diff_volatility": vol_all,
            "n_above_thr_runs": n_above_runs,
            "mean_run_length": mean_run_len,
            "max_run_length": max_run_len,
        },
        "episode_analysis": {
            "n_episodes": n_ep,
            "episodes_crossing_A_thr": ep_with_any_above,
            "episodes_entirely_sub_thr": n_ep - ep_with_any_above,
            "pct_episodes_crossing_A_thr": float(100.0 * ep_with_any_above / n_ep),
            "median_episode_mean_A": float(np.median(ep_means)) if ep_means else 0.0,
            "median_episode_max_A": float(np.median(ep_maxs)) if ep_maxs else 0.0,
        },
        "score_sha256": score_hash,
    }


def run_experiment(run_id: int, seed: int) -> dict:
    log.info(f"\n{'='*70}")
    log.info(f"PHASE 2A.1 EXPERIMENT RUN {run_id} (seed={seed})")
    log.info(f"{'='*70}")

    # 1. Load data & reconstruct timestamps
    log.info("[1] Loading raw data & reconstructing timestamps...")
    adapter = TonIoTDataAdapter(filepath=MODBUS_CSV)
    df_raw = adapter.load_raw()
    df_sorted = adapter.reconstruct_timestamps(df_raw)

    # 2. Chronological Split
    log.info("[2] Partitioning via TonIoTChronologicalSplitter...")
    splitter = TonIoTChronologicalSplitter()
    split_res = splitter.split(df_sorted)

    # Reconstruct partition DataFrames with timestamps
    df_train = pd.concat([split_res.meta_train, split_res.X_train, split_res.labels_train], axis=1)
    df_calib = pd.concat([split_res.meta_calib, split_res.X_calib, split_res.labels_calib], axis=1)
    df_eval = pd.concat([split_res.meta_eval, split_res.X_eval, split_res.labels_eval], axis=1)

    # Verify counts
    assert len(df_train) == EXPECTED_TRAIN, f"Train count mismatch: {len(df_train)} != {EXPECTED_TRAIN}"
    assert len(df_calib) == EXPECTED_CALIB, f"Calib count mismatch: {len(df_calib)} != {EXPECTED_CALIB}"
    assert len(df_eval) == EXPECTED_EVAL, f"Eval count mismatch: {len(df_eval)} != {EXPECTED_EVAL}"
    assert int((df_train["label"] == 1).sum()) == 0, "Train contains attacks!"
    assert int((df_calib["label"] == 1).sum()) == 0, "Calib contains attacks!"
    log.info("  [OK] Data partition counts and benign purity verified.")

    y_eval = df_eval["label"].values.astype(int)

    # 3. Representation A: Static Baseline (4 features)
    log.info("[3] Preparing Representation A: Static Baseline (4 features)...")
    X_train_static = df_train[STATIC_FEATURES].copy()
    X_calib_static = df_calib[STATIC_FEATURES].copy()
    X_eval_static = df_eval[STATIC_FEATURES].copy()

    res_static = evaluate_representation("STATIC_BASELINE", X_train_static, X_calib_static, X_eval_static, y_eval, seed)

    # 4. Representation B: Temporal Representation (29 features)
    log.info("[4] Building Representation B: Causal Temporal Features (29 features)...")
    X_train_temporal, t_names = build_temporal_features(df_train)
    X_calib_temporal, _ = build_temporal_features(df_calib)
    X_eval_temporal, _ = build_temporal_features(df_eval)

    log.info(f"  [OK] Temporal feature names ({len(t_names)}): {t_names}")

    # Check for NaNs
    assert not X_train_temporal.isna().any().any(), "Train temporal features contain NaN!"
    assert not X_calib_temporal.isna().any().any(), "Calib temporal features contain NaN!"
    assert not X_eval_temporal.isna().any().any(), "Eval temporal features contain NaN!"
    log.info("  [OK] Zero NaNs verified across all temporal partitions.")

    res_temporal = evaluate_representation("TEMPORAL_REPRESENTATION", X_train_temporal, X_calib_temporal, X_eval_temporal, y_eval, seed)

    return {
        "run_id": run_id,
        "seed": seed,
        "static_baseline": res_static,
        "temporal_representation": res_temporal,
    }


def main():
    log.info("Starting Phase 2A.1 — Temporal Feature Representation Audit")
    log.info(f"Start time: {datetime.now().isoformat()}")

    # Run 1
    r1 = run_experiment(run_id=1, seed=42)

    # Run 2 (reproducibility check)
    r2 = run_experiment(run_id=2, seed=42)

    # ── Cross-Run Reproducibility Checks ──────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("CROSS-RUN REPRODUCIBILITY VERIFICATION")
    log.info("="*70)

    checks = [
        ("static.s_min", r1["static_baseline"]["s_min"], r2["static_baseline"]["s_min"]),
        ("static.s_max", r1["static_baseline"]["s_max"], r2["static_baseline"]["s_max"]),
        ("static.A_thr", r1["static_baseline"]["A_thr"], r2["static_baseline"]["A_thr"]),
        ("static.score_sha256", r1["static_baseline"]["score_sha256"], r2["static_baseline"]["score_sha256"]),
        ("temporal.s_min", r1["temporal_representation"]["s_min"], r2["temporal_representation"]["s_min"]),
        ("temporal.s_max", r1["temporal_representation"]["s_max"], r2["temporal_representation"]["s_max"]),
        ("temporal.A_thr", r1["temporal_representation"]["A_thr"], r2["temporal_representation"]["A_thr"]),
        ("temporal.score_sha256", r1["temporal_representation"]["score_sha256"], r2["temporal_representation"]["score_sha256"]),
    ]

    all_match = True
    for name, v1, v2 in checks:
        match = (v1 == v2)
        if not match:
            all_match = False
        log.info(f"  {'[PASS]' if match else '[FAIL]':10s} {name}: {v1}")

    log.info(f"Overall Reproducibility: {'VERIFIED' if all_match else 'FAILED'}")

    # ── Primary Comparison & Conclusion ───────────────────────────────────────
    st = r1["static_baseline"]
    tp = r1["temporal_representation"]

    st_pm = st["point_metrics"]
    tp_pm = tp["point_metrics"]
    st_sd = st["score_distributions"]
    tp_sd = tp["score_distributions"]
    st_sub = st["sub_threshold"]
    tp_sub = tp["sub_threshold"]
    st_ep = st["episode_analysis"]
    tp_ep = tp["episode_analysis"]

    log.info("\n" + "="*70)
    log.info("PRIMARY COMPARISON: STATIC vs TEMPORAL REPRESENTATION")
    log.info("="*70)
    log.info(f"{'Metric':<35} {'STATIC (4 feats)':<20} {'TEMPORAL (29 feats)':<20} {'Delta':<15}")
    log.info("-" * 90)

    comparisons = [
        ("Features Count", st["n_features"], tp["n_features"]),
        ("ROC-AUC", st_pm["roc_auc"], tp_pm["roc_auc"]),
        ("PR-AUC", st_pm["pr_auc"], tp_pm["pr_auc"]),
        ("Recall", st_pm["recall"], tp_pm["recall"]),
        ("F1-Score", st_pm["f1"], tp_pm["f1"]),
        ("Precision", st_pm["precision"], tp_pm["precision"]),
        ("FPR", st_pm["FPR"], tp_pm["FPR"]),
        ("Balanced Accuracy", st_pm["balanced_accuracy"], tp_pm["balanced_accuracy"]),
        ("Benign Mean A", st_sd["eval_benign"]["mean"], tp_sd["eval_benign"]["mean"]),
        ("Attack Mean A", st_sd["eval_attack"]["mean"], tp_sd["eval_attack"]["mean"]),
        ("Score Separation (Atk - Ben)", st_sd["eval_attack"]["mean"] - st_sd["eval_benign"]["mean"], tp_sd["eval_attack"]["mean"] - tp_sd["eval_benign"]["mean"]),
        ("Sub-Threshold Attack %", st_sub["pct_attack_sub_thr"], tp_sub["pct_attack_sub_thr"]),
        ("Episodes Crossing A_thr", st_ep["episodes_crossing_A_thr"], tp_ep["episodes_crossing_A_thr"]),
        ("Episodes Crossing A_thr %", st_ep["pct_episodes_crossing_A_thr"], tp_ep["pct_episodes_crossing_A_thr"]),
    ]

    for label, v_st, v_tp in comparisons:
        delta = v_tp - v_st if isinstance(v_st, (int, float)) and isinstance(v_tp, (int, float)) else "N/A"
        delta_str = f"{delta:+.6f}" if isinstance(delta, float) else f"{delta:+d}" if isinstance(delta, int) else "N/A"
        log.info(f"{label:<35} {v_st:<20.6f} {v_tp:<20.6f} {delta_str:<15}")

    # ── Conclusion Decision ───────────────────────────────────────────────────
    # Decision criteria:
    # 1. Does ROC-AUC materially move away from ~0.50 (e.g. >= 0.65)?
    # 2. Does score separation become positive and substantial?
    # 3. Does PR-AUC exceed attack prevalence (0.3708) by meaningful margin?
    # 4. Does sub-threshold attack % decrease substantially?
    # 5. Does episode detection increase?

    auc_diff = tp_pm["roc_auc"] - st_pm["roc_auc"]
    sep_diff = (tp_sd["eval_attack"]["mean"] - tp_sd["eval_benign"]["mean"]) - (st_sd["eval_attack"]["mean"] - st_sd["eval_benign"]["mean"])

    if tp_pm["roc_auc"] >= 0.70 and (tp_sd["eval_attack"]["mean"] > tp_sd["eval_benign"]["mean"] + 0.05):
        verdict = "TEMPORAL_SIGNAL_IMPROVED"
        verdict_rationale = (
            f"Predefined temporal features significantly improve detector discrimination: "
            f"ROC-AUC improved from {st_pm['roc_auc']:.4f} to {tp_pm['roc_auc']:.4f}, "
            f"PR-AUC improved from {st_pm['pr_auc']:.4f} to {tp_pm['pr_auc']:.4f}, and "
            f"score separation improved by {sep_diff:+.4f}."
        )
    elif tp_pm["roc_auc"] >= 0.55 or (tp_sd["eval_attack"]["mean"] > tp_sd["eval_benign"]["mean"] + 0.01) or (tp_ep["pct_episodes_crossing_A_thr"] > st_ep["pct_episodes_crossing_A_thr"] + 10.0):
        verdict = "TEMPORAL_SIGNAL_PARTIALLY_IMPROVED"
        verdict_rationale = (
            f"Temporal features yield marginal or partial improvements: "
            f"ROC-AUC moved from {st_pm['roc_auc']:.4f} to {tp_pm['roc_auc']:.4f} (delta={auc_diff:+.4f}), "
            f"but remains insufficient or limited for robust downstream zero-trust trust calibration."
        )
    else:
        verdict = "TEMPORAL_SIGNAL_NOT_IMPROVED"
        verdict_rationale = (
            f"Temporal features do not materially improve discrimination: "
            f"ROC-AUC is {tp_pm['roc_auc']:.4f} (baseline was {st_pm['roc_auc']:.4f}), "
            f"PR-AUC is {tp_pm['pr_auc']:.4f} vs prevalence 0.3708, "
            f"score separation remains negligible ({tp_sd['eval_attack']['mean'] - tp_sd['eval_benign']['mean']:+.6f}). "
            f"The feature space limitation persists under these predefined temporal windows."
        )

    log.info("\n" + "="*70)
    log.info(f"AUDIT VERDICT: {verdict}")
    log.info(f"RATIONALE: {verdict_rationale}")
    log.info("="*70)

    # ── Leakage Audit ─────────────────────────────────────────────────────────
    leakage_audit = {
        "future_values_used": False,
        "cross_partition_temporal_leakage": False,
        "attack_labels_used_during_training": False,
        "attack_labels_used_during_calibration": False,
        "evaluation_labels_used_for_threshold": False,
        "scaler_fit_only_on_benign_train": True,
        "normalization_bounds_derived_only_from_train": True,
        "A_thr_derived_only_from_benign_calibration": True,
        "leakage_detected": False,
    }

    # ── Save JSON Artifacts ───────────────────────────────────────────────────
    log.info("\n[ARTIFACTS] Writing JSON artifacts to experiments/ton_iot/phase2a1_temporal_representation/...")

    temporal_feature_names = r1["temporal_representation"]["feature_names"]
    with open(OUT_DIR / "feature_specification.json", "w") as f:
        json.dump({
            "static_feature_count": len(STATIC_FEATURES),
            "static_feature_names": STATIC_FEATURES,
            "temporal_feature_count": len(temporal_feature_names),
            "temporal_feature_names": temporal_feature_names,
            "short_window_size": 5,
            "medium_window_size": 20,
            "boundary_handling": "History starts strictly at partition row 0. First differences filled with 0.0. Rolling variance at t=0 filled with 0.0. min_periods=1 for rolling means. Zero cross-partition lookback.",
        }, f, indent=2)

    with open(OUT_DIR / "detector_config.json", "w") as f:
        json.dump(DETECTOR_CFG, f, indent=2)

    with open(OUT_DIR / "normalization_and_thresholds.json", "w") as f:
        json.dump({
            "static": {
                "s_min": st["s_min"],
                "s_max": st["s_max"],
                "A_thr": st["A_thr"],
                "raw_thr_equiv": st["raw_thr_equiv"],
            },
            "temporal": {
                "s_min": tp["s_min"],
                "s_max": tp["s_max"],
                "A_thr": tp["A_thr"],
                "raw_thr_equiv": tp["raw_thr_equiv"],
            },
        }, f, indent=2)

    with open(OUT_DIR / "metrics_comparison.json", "w") as f:
        json.dump({
            "static_baseline": st_pm,
            "temporal_representation": tp_pm,
        }, f, indent=2)

    with open(OUT_DIR / "score_distributions.json", "w") as f:
        json.dump({
            "static_baseline": st_sd,
            "temporal_representation": tp_sd,
        }, f, indent=2)

    with open(OUT_DIR / "temporal_signal_analysis.json", "w") as f:
        json.dump({
            "static_baseline": st["temporal_characteristics"],
            "temporal_representation": tp["temporal_characteristics"],
            "sub_threshold_static": st_sub,
            "sub_threshold_temporal": tp_sub,
        }, f, indent=2)

    with open(OUT_DIR / "episode_analysis.json", "w") as f:
        json.dump({
            "static_baseline": st_ep,
            "temporal_representation": tp_ep,
        }, f, indent=2)

    with open(OUT_DIR / "leakage_audit.json", "w") as f:
        json.dump(leakage_audit, f, indent=2)

    with open(OUT_DIR / "reproducibility.json", "w") as f:
        json.dump({
            "run1_static_hash": r1["static_baseline"]["score_sha256"],
            "run2_static_hash": r2["static_baseline"]["score_sha256"],
            "run1_temporal_hash": r1["temporal_representation"]["score_sha256"],
            "run2_temporal_hash": r2["temporal_representation"]["score_sha256"],
            "checks": {c[0]: {"run1": c[1], "run2": c[2], "match": (c[1] == c[2])} for c in checks},
            "all_match": all_match,
        }, f, indent=2)

    with open(OUT_DIR / "audit_conclusion.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "rationale": verdict_rationale,
            "comparisons": {item[0]: {"static": item[1], "temporal": item[2]} for item in comparisons},
        }, f, indent=2)

    def _json_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(OUT_DIR / "full_results.json", "w") as f:
        json.dump({"run1": r1, "reproducibility": all_match, "verdict": verdict}, f, indent=2, default=_json_safe)

    log.info("[ARTIFACTS] All artifacts written successfully.")
    log.info("\nPHASE_2A1_TEMPORAL_REPRESENTATION_AUDIT_COMPLETE")
    return r1, r2, all_match, verdict, verdict_rationale


if __name__ == "__main__":
    try:
        r1, r2, all_match, verdict, verdict_rationale = main()
        sys.exit(0)
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
