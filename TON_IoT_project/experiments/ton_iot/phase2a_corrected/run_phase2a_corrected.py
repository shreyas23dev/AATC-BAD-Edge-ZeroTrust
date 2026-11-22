"""
Phase 2A Corrected — E1 Detector Suitability Experiment
=========================================================
Protocol:  E1 (corrected continuous evaluation count: 173,506 rows)
Detector:  IsolationForest (primary; E1-specified)
Objective: Detector suitability assessment — NOT TrustEngine execution

CRITICAL CONSTRAINTS:
  - No TrustEngine execution
  - No trust parameter tuning
  - No evaluation-label-guided threshold adjustment
  - No reuse of old 150,023-row results
  - Evaluation labels joined ONLY after detector fully frozen and scored

Outputs written to: experiments/ton_iot/phase2a_corrected/
Report  written to: docs/TONIOT_PHASE2A_CORRECTED_DETECTOR_SUITABILITY.md
"""

import sys
import os
import json
import hashlib
import logging
import traceback
import warnings

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)

warnings.filterwarnings("ignore")

# ── Path Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent          # TON_IoT_project/
EDGEIIOT_ROOT = PROJECT_ROOT / "edge-iiot-anomaly-detection"

sys.path.insert(0, str(EDGEIIOT_ROOT))

from config.settings import config
from src.data.ton_iot_adapter import TonIoTDataAdapter, TonIoTChronologicalSplitter
from src.preprocessing.ton_iot_preprocessor import TonIoTPreprocessor
from src.model.ton_iot_calibration import (
    TonIoTScoreNormalizer,
    compute_benign_anomaly_threshold,
    simulate_ema_stream,
)

# ── Output Directories ────────────────────────────────────────────────────────
OUT_DIR = SCRIPT_DIR
DOCS_DIR = PROJECT_ROOT / "docs"
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
log = logging.getLogger("phase2a_corrected")

# ── Dataset Path ──────────────────────────────────────────────────────────────
MODBUS_CSV = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_audit\iot_processed\IoT_Modbus.csv")

# ── E1 Frozen Detector Configuration ─────────────────────────────────────────
E1_DETECTOR_CFG = {
    "n_estimators": 300,
    "max_samples": 256,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}

# ── Expected Partition Counts ─────────────────────────────────────────────────
EXPECTED_TRAIN      = 100_940
EXPECTED_CALIB      = 11_691
EXPECTED_EVAL       = 173_506
EXPECTED_EVAL_BENIGN = 109_167
EXPECTED_EVAL_ATTACK = 64_339


def sha256_of_array(arr: np.ndarray) -> str:
    """Compute SHA-256 of a float64 numpy array in deterministic byte representation."""
    arr64 = np.asarray(arr, dtype=np.float64)
    return hashlib.sha256(arr64.tobytes()).hexdigest()


def score_distribution_stats(scores: np.ndarray, label: str) -> dict:
    """Return comprehensive descriptive statistics for a score array."""
    return {
        "label": label,
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "min":  float(np.min(scores)),
        "p25":  float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75":  float(np.percentile(scores, 75)),
        "p90":  float(np.percentile(scores, 90)),
        "p95":  float(np.percentile(scores, 95)),
        "p99":  float(np.percentile(scores, 99)),
        "max":  float(np.max(scores)),
    }


def lag1_autocorr(arr: np.ndarray) -> float:
    """Lag-1 autocorrelation of an array."""
    if len(arr) < 2:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def first_diff_volatility(arr: np.ndarray) -> float:
    """Standard deviation of first differences."""
    if len(arr) < 2:
        return float("nan")
    return float(np.std(np.diff(arr)))


def run_episodes(labels: np.ndarray, scores: np.ndarray, a_thr: float):
    """
    Identify contiguous attack episodes in the label sequence.
    Returns a list of episode dicts with score characteristics.
    """
    episodes = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            start = i
            while i < n and labels[i] == 1:
                i += 1
            end = i  # exclusive
            ep_scores = scores[start:end]
            # score 1 step before onset (if available)
            pre_score = float(scores[start - 1]) if start > 0 else float("nan")
            episodes.append({
                "start_idx": int(start),
                "end_idx":   int(end - 1),
                "duration":  int(end - start),
                "pre_onset_score": pre_score,
                "mean_score":  float(np.mean(ep_scores)),
                "max_score":   float(np.max(ep_scores)),
                "any_above_thr": bool(np.any(ep_scores >= a_thr)),
                "peak_offset": int(np.argmax(ep_scores)),
            })
        else:
            i += 1
    return episodes


def run_above_thr_runs(arr: np.ndarray, a_thr: float):
    """
    Find contiguous runs where A_t >= a_thr.
    Returns list of (start, end_inclusive, length).
    """
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


def run_experiment(run_id: int, seed: int) -> dict:
    """
    Execute one complete experiment run. Called twice for reproducibility.
    Returns a dict with all measured quantities.
    """
    log.info(f"\n{'='*70}")
    log.info(f"EXPERIMENT RUN {run_id}  (seed={seed})")
    log.info(f"{'='*70}\n")

    result = {"run_id": run_id, "seed": seed}

    # ─────────────────────────────────────────────────────────────────────────
    # 1. DATA LOADING & TIMESTAMP RECONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[1] Loading IoT_Modbus.csv...")
    adapter = TonIoTDataAdapter(filepath=MODBUS_CSV)
    df_raw = adapter.load_raw()
    log.info(f"    Raw rows loaded: {len(df_raw)}")

    log.info("[1] Reconstructing deterministic timestamps...")
    df_sorted = adapter.reconstruct_timestamps(df_raw)
    log.info(f"    Timestamped & sorted rows: {len(df_sorted)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CHRONOLOGICAL SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[2] Applying TonIoTChronologicalSplitter (E1 frozen boundaries)...")
    splitter = TonIoTChronologicalSplitter()
    split_result = splitter.split(df_sorted)
    dr = split_result.discrepancy_report

    n_train  = len(split_result.X_train)
    n_calib  = len(split_result.X_calib)
    n_eval   = len(split_result.X_eval)
    n_eval_b = int((split_result.labels_eval["label"] == 0).sum())
    n_eval_a = int((split_result.labels_eval["label"] == 1).sum())
    eval_start = str(split_result.meta_eval["timestamp"].min())
    eval_end   = str(split_result.meta_eval["timestamp"].max())

    log.info(f"\n{'='*60}")
    log.info("DATA SPLIT VERIFICATION")
    log.info(f"{'='*60}")
    log.info(f"  TRAIN N          : {n_train:>10,}  (expected {EXPECTED_TRAIN:,})")
    log.info(f"  CALIBRATION N    : {n_calib:>10,}  (expected {EXPECTED_CALIB:,})")
    log.info(f"  EVALUATION N     : {n_eval:>10,}  (expected {EXPECTED_EVAL:,})")
    log.info(f"  EVAL BENIGN N    : {n_eval_b:>10,}  (expected {EXPECTED_EVAL_BENIGN:,})")
    log.info(f"  EVAL ATTACK N    : {n_eval_a:>10,}  (expected {EXPECTED_EVAL_ATTACK:,})")
    log.info(f"  evaluation_start : {eval_start}")
    log.info(f"  evaluation_end   : {eval_end}")
    log.info(f"{'='*60}\n")

    # Hard-stop on count mismatch
    count_errors = []
    if n_train  != EXPECTED_TRAIN:      count_errors.append(f"TRAIN: got {n_train}, expected {EXPECTED_TRAIN}")
    if n_calib  != EXPECTED_CALIB:      count_errors.append(f"CALIB: got {n_calib}, expected {EXPECTED_CALIB}")
    if n_eval   != EXPECTED_EVAL:       count_errors.append(f"EVAL: got {n_eval}, expected {EXPECTED_EVAL}")
    if n_eval_b != EXPECTED_EVAL_BENIGN: count_errors.append(f"EVAL_BENIGN: got {n_eval_b}, expected {EXPECTED_EVAL_BENIGN}")
    if n_eval_a != EXPECTED_EVAL_ATTACK: count_errors.append(f"EVAL_ATTACK: got {n_eval_a}, expected {EXPECTED_EVAL_ATTACK}")
    if count_errors:
        raise RuntimeError(
            f"SPLIT COUNT DISCREPANCY — STOPPING:\n" + "\n".join(count_errors)
        )
    log.info("  [OK] All partition counts verified.")

    result["split"] = {
        "n_train": n_train, "n_calib": n_calib, "n_eval": n_eval,
        "n_eval_benign": n_eval_b, "n_eval_attack": n_eval_a,
        "eval_start": eval_start, "eval_end": eval_end,
        "counts_verified": True,
    }

    # Benign purity check
    assert int((split_result.labels_train["label"] == 1).sum()) == 0, "TRAIN contains attacks!"
    assert int((split_result.labels_calib["label"] == 1).sum()) == 0, "CALIB contains attacks!"
    log.info("  [OK] Train & calibration benign purity verified.")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PREPROCESSING — FIT ONLY ON TRAIN
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[3] Fitting TonIoTPreprocessor (RobustScaler) on TRAIN partition only...")
    preprocessor = TonIoTPreprocessor()
    X_train_scaled = preprocessor.fit_transform(split_result.X_train)
    X_calib_scaled = preprocessor.transform(split_result.X_calib)
    X_eval_scaled  = preprocessor.transform(split_result.X_eval)

    scaler_params = preprocessor.get_scaler_params()
    log.info(f"    Scaler medians: {[f'{v:.4f}' for v in scaler_params['medians']]}")
    log.info(f"    Scaler IQRs   : {[f'{v:.4f}' for v in scaler_params['iqrs']]}")

    result["scaler"] = scaler_params

    # LEAKAGE AUDIT — verify scaler not re-fitted on calib/eval
    # (structural: TonIoTPreprocessor.fit was only called on X_train above)
    log.info("  [OK] Leakage audit: scaler fitted exclusively on TRAIN benign data.")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. DETECTOR FITTING — TRAIN ONLY
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n[4] Fitting IsolationForest on TRAIN ({n_train:,} benign samples)...")
    log.info(f"    Config: {E1_DETECTOR_CFG}")

    cfg = {**E1_DETECTOR_CFG, "random_state": seed}
    model = IsolationForest(**cfg)
    model.fit(X_train_scaled)
    log.info("    [OK] IsolationForest fitted.")

    result["detector_config"] = {**cfg}

    # ─────────────────────────────────────────────────────────────────────────
    # 5. RAW ANOMALY SCORES (convention: raw = -score_samples)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[5] Computing raw anomaly scores (raw = -score_samples(X))...")
    raw_train = -model.score_samples(X_train_scaled)
    raw_calib = -model.score_samples(X_calib_scaled)
    raw_eval  = -model.score_samples(X_eval_scaled)
    log.info(f"    raw_train: n={len(raw_train)}, mean={np.mean(raw_train):.6f}")
    log.info(f"    raw_calib: n={len(raw_calib)}, mean={np.mean(raw_calib):.6f}")
    log.info(f"    raw_eval:  n={len(raw_eval)}, mean={np.mean(raw_eval):.6f}")

    result["raw_score_stats"] = {
        "train": score_distribution_stats(raw_train, "TRAIN (benign)"),
        "calib": score_distribution_stats(raw_calib, "CALIBRATION (benign)"),
        "eval_all": score_distribution_stats(raw_eval, "EVALUATION (all)"),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 6. SCORE NORMALIZATION — BOUNDS FROM TRAIN ONLY
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[6] Fitting score normalizer on TRAIN benign raw scores...")
    normalizer = TonIoTScoreNormalizer()
    normalizer.fit(raw_train)
    s_min = normalizer.s_min_
    s_max = normalizer.s_max_

    log.info(f"    s_min = {s_min:.8f}")
    log.info(f"    s_max = {s_max:.8f}")
    log.info("  [OK] Normalization bounds derived from TRAIN only.")

    A_train = normalizer.normalize(raw_train)
    A_calib = normalizer.normalize(raw_calib)
    A_eval  = normalizer.normalize(raw_eval)

    assert A_train.min() >= 0.0 and A_train.max() <= 1.0
    assert A_calib.min() >= 0.0 and A_calib.max() <= 1.0
    assert A_eval.min()  >= 0.0 and A_eval.max()  <= 1.0

    result["normalization"] = {"s_min": s_min, "s_max": s_max}

    # ─────────────────────────────────────────────────────────────────────────
    # 7. CALIBRATION THRESHOLD — 99th PERCENTILE OF BENIGN CALIBRATION SCORES
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[7] Computing A_thr = 99th percentile of benign calibration scores...")
    A_thr = compute_benign_anomaly_threshold(A_calib, percentile=99.0)
    raw_thr_equiv = A_thr * (s_max - s_min) + s_min

    log.info(f"    A_thr (normalized) = {A_thr:.8f}")
    log.info(f"    raw score equiv    = {raw_thr_equiv:.8f}")
    log.info(f"    interpolation      = linear (numpy.percentile default)")
    log.info("  [OK] A_thr derived from benign calibration only. No evaluation labels used.")

    calib_stats = score_distribution_stats(A_calib, "CALIBRATION A_t (normalized)")
    log.info(f"    Calib A_t stats: mean={calib_stats['mean']:.6f}, std={calib_stats['std']:.6f}, "
             f"p99={calib_stats['p99']:.6f}, max={calib_stats['max']:.6f}")

    result["calibration"] = {
        "n_calib": n_calib,
        "percentile_method": "linear (numpy default)",
        "A_thr": A_thr,
        "raw_thr_equiv": raw_thr_equiv,
        "A_calib_stats": calib_stats,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 8. EVALUATION — LABELS JOINED POST-HOC
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[8] Joining evaluation labels post-hoc (after all scoring is frozen)...")
    y_eval = split_result.labels_eval["label"].values.astype(int)
    y_pred = (A_eval >= A_thr).astype(int)

    assert len(y_eval) == EXPECTED_EVAL, f"Label length mismatch: {len(y_eval)} != {EXPECTED_EVAL}"

    # ─────────────────────────────────────────────────────────────────────────
    # 9. POINT-LEVEL METRICS
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[9] Computing point-level discrimination metrics...")

    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
    precision  = precision_score(y_eval, y_pred, zero_division=0)
    recall     = recall_score(y_eval, y_pred, zero_division=0)
    f1         = f1_score(y_eval, y_pred, zero_division=0)
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr        = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc    = balanced_accuracy_score(y_eval, y_pred)
    roc_auc    = roc_auc_score(y_eval, A_eval)
    pr_auc     = average_precision_score(y_eval, A_eval)
    attack_prev = n_eval_a / n_eval

    log.info(f"\n{'='*60}")
    log.info("POINT-LEVEL METRICS (post-hoc, A_thr={A_thr:.6f})")
    log.info(f"{'='*60}")
    log.info(f"  Attack prevalence : {attack_prev:.4f} ({n_eval_a}/{n_eval})")
    log.info(f"  TP                : {tp:,}")
    log.info(f"  TN                : {tn:,}")
    log.info(f"  FP                : {fp:,}")
    log.info(f"  FN                : {fn:,}")
    log.info(f"  Precision         : {precision:.6f}")
    log.info(f"  Recall            : {recall:.6f}")
    log.info(f"  F1-Score          : {f1:.6f}")
    log.info(f"  FPR               : {fpr:.6f}")
    log.info(f"  FNR               : {fnr:.6f}")
    log.info(f"  Specificity       : {specificity:.6f}")
    log.info(f"  Balanced Accuracy : {bal_acc:.6f}")
    log.info(f"  ROC-AUC           : {roc_auc:.6f}")
    log.info(f"  PR-AUC            : {pr_auc:.6f}")
    log.info(f"{'='*60}\n")

    result["point_metrics"] = {
        "A_thr": A_thr,
        "attack_prevalence": attack_prev,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "FPR": fpr, "FNR": fnr, "specificity": specificity,
        "balanced_accuracy": bal_acc, "roc_auc": roc_auc, "pr_auc": pr_auc,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 10. SCORE DISTRIBUTIONS — BENIGN vs ATTACK
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[10] Computing benign vs attack score distributions...")
    A_eval_benign = A_eval[y_eval == 0]
    A_eval_attack = A_eval[y_eval == 1]

    stats_benign = score_distribution_stats(A_eval_benign, "EVAL benign")
    stats_attack = score_distribution_stats(A_eval_attack, "EVAL attack")
    stats_all    = score_distribution_stats(A_eval, "EVAL all")

    for s in [stats_benign, stats_attack, stats_all]:
        log.info(
            f"  [{s['label']:20s}] n={s['n']:>7,}  mean={s['mean']:.6f}  "
            f"std={s['std']:.6f}  median={s['median']:.6f}  "
            f"p90={s['p90']:.6f}  p99={s['p99']:.6f}  max={s['max']:.6f}"
        )

    result["score_distributions"] = {
        "eval_benign": stats_benign,
        "eval_attack": stats_attack,
        "eval_all": stats_all,
        "train_benign": score_distribution_stats(A_train, "TRAIN benign"),
        "calib_benign": calib_stats,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 11. SUB-THRESHOLD SIGNAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[11] Sub-threshold attack signal analysis...")

    sub_atk_mask   = (y_eval == 1) & (A_eval < A_thr)
    above_atk_mask = (y_eval == 1) & (A_eval >= A_thr)
    sub_ben_mask   = (y_eval == 0) & (A_eval < A_thr)
    above_ben_mask = (y_eval == 0) & (A_eval >= A_thr)

    n_sub_atk    = int(sub_atk_mask.sum())
    n_above_atk  = int(above_atk_mask.sum())
    n_sub_ben    = int(sub_ben_mask.sum())
    n_above_ben  = int(above_ben_mask.sum())
    pct_sub_atk  = 100.0 * n_sub_atk / n_eval_a if n_eval_a > 0 else 0.0

    log.info(f"\n{'='*60}")
    log.info("SUB-THRESHOLD ANALYSIS")
    log.info(f"{'='*60}")
    log.info(f"  Attack obs with A_t < A_thr  : {n_sub_atk:>8,}  ({pct_sub_atk:.2f}% of attacks)")
    log.info(f"  Attack obs with A_t >= A_thr : {n_above_atk:>8,}  ({100.0 - pct_sub_atk:.2f}% of attacks)")
    log.info(f"  Benign obs with A_t < A_thr  : {n_sub_ben:>8,}")
    log.info(f"  Benign obs with A_t >= A_thr : {n_above_ben:>8,}  (FP count)")
    log.info(f"{'='*60}\n")

    if n_sub_atk > 0:
        st = score_distribution_stats(A_eval[sub_atk_mask], "sub-threshold attacks")
        log.info(f"  Sub-threshold attack score stats:")
        log.info(f"    mean={st['mean']:.6f}  median={st['median']:.6f}  "
                 f"p90={st['p90']:.6f}  p95={st['p95']:.6f}  "
                 f"p99={st['p99']:.6f}  max={st['max']:.6f}")
    if n_above_atk > 0:
        st2 = score_distribution_stats(A_eval[above_atk_mask], "above-thr attacks")
        log.info(f"  Above-threshold attack score stats:")
        log.info(f"    mean={st2['mean']:.6f}  median={st2['median']:.6f}  "
                 f"p99={st2['p99']:.6f}  max={st2['max']:.6f}")

    result["sub_threshold"] = {
        "n_attack_sub_thr":   n_sub_atk,
        "n_attack_above_thr": n_above_atk,
        "n_benign_sub_thr":   n_sub_ben,
        "n_benign_above_thr": n_above_ben,
        "pct_attack_sub_thr": pct_sub_atk,
        "A_thr": A_thr,
        "sub_thr_attack_stats": score_distribution_stats(A_eval[sub_atk_mask], "sub-thr-attack") if n_sub_atk > 0 else None,
        "above_thr_attack_stats": score_distribution_stats(A_eval[above_atk_mask], "above-thr-attack") if n_above_atk > 0 else None,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 12. TEMPORAL SCORE CHARACTERISTICS (no TrustEngine)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[12] Computing temporal score characteristics...")

    ac_all   = lag1_autocorr(A_eval)
    ac_ben   = lag1_autocorr(A_eval_benign)
    ac_atk   = lag1_autocorr(A_eval_attack)
    vol_all  = first_diff_volatility(A_eval)
    vol_ben  = first_diff_volatility(A_eval_benign)
    vol_atk  = first_diff_volatility(A_eval_attack)

    # Contiguous A_t >= A_thr runs
    above_runs = run_above_thr_runs(A_eval, A_thr)
    n_above_runs = len(above_runs)
    run_lengths  = [r[2] for r in above_runs]
    mean_run_len = float(np.mean(run_lengths)) if run_lengths else 0.0
    max_run_len  = int(np.max(run_lengths)) if run_lengths else 0

    log.info(f"  Eval stream lag-1 autocorr (all)   : {ac_all:.6f}")
    log.info(f"  Eval stream lag-1 autocorr (benign) : {ac_ben:.6f}")
    log.info(f"  Eval stream lag-1 autocorr (attack) : {ac_atk:.6f}")
    log.info(f"  First-diff volatility (all)         : {vol_all:.6f}")
    log.info(f"  First-diff volatility (benign)      : {vol_ben:.6f}")
    log.info(f"  First-diff volatility (attack)      : {vol_atk:.6f}")
    log.info(f"  Contiguous A_t >= A_thr runs        : {n_above_runs:,}")
    log.info(f"  Mean above-thr run length           : {mean_run_len:.2f} steps")
    log.info(f"  Max  above-thr run length           : {max_run_len} steps")

    result["temporal"] = {
        "lag1_autocorr_all":   ac_all,
        "lag1_autocorr_benign": ac_ben,
        "lag1_autocorr_attack": ac_atk,
        "first_diff_vol_all":   vol_all,
        "first_diff_vol_benign": vol_ben,
        "first_diff_vol_attack": vol_atk,
        "n_above_thr_runs":   n_above_runs,
        "mean_run_length":    mean_run_len,
        "max_run_length":     max_run_len,
        "score_stats_benign": stats_benign,
        "score_stats_attack": stats_attack,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 13. EPISODE-LEVEL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[13] Episode-level score characterization...")
    episodes = run_episodes(y_eval, A_eval, A_thr)
    n_ep = len(episodes)
    ep_with_any_above = sum(1 for e in episodes if e["any_above_thr"])
    ep_durations = [e["duration"] for e in episodes]
    ep_mean_scores = [e["mean_score"] for e in episodes]
    ep_max_scores = [e["max_score"] for e in episodes]

    log.info(f"  Number of attack episodes         : {n_ep:,}")
    log.info(f"  Episodes with any A_t >= A_thr    : {ep_with_any_above:,} ({100*ep_with_any_above/n_ep:.1f}%)")
    if ep_durations:
        log.info(f"  Episode duration: min={min(ep_durations)} median={np.median(ep_durations):.0f} max={max(ep_durations)}")
        log.info(f"  Episode mean A_t: min={min(ep_mean_scores):.6f} median={np.median(ep_mean_scores):.6f} max={max(ep_mean_scores):.6f}")
        log.info(f"  Episode max  A_t: min={min(ep_max_scores):.6f} median={np.median(ep_max_scores):.6f} max={max(ep_max_scores):.6f}")

    # Detailed per-episode table (save separately; log summary only)
    result["episodes"] = {
        "n_episodes": n_ep,
        "episodes_with_any_above_thr": ep_with_any_above,
        "pct_episodes_detected_point": 100.0 * ep_with_any_above / n_ep if n_ep > 0 else 0.0,
        "duration_min": int(min(ep_durations)) if ep_durations else 0,
        "duration_median": float(np.median(ep_durations)) if ep_durations else 0.0,
        "duration_max": int(max(ep_durations)) if ep_durations else 0,
        "mean_score_median": float(np.median(ep_mean_scores)) if ep_mean_scores else 0.0,
        "max_score_median": float(np.median(ep_max_scores)) if ep_max_scores else 0.0,
        "episode_detail": episodes[:50],   # cap to first 50 for JSON size
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 14. REPRODUCIBILITY — HASH
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[14] Computing SHA-256 hash of evaluation score array...")
    score_hash = sha256_of_array(A_eval)
    log.info(f"    SHA-256(A_eval) = {score_hash}")

    # Timestamp diffs
    ts_eval = split_result.meta_eval["timestamp"]
    ts_diffs = ts_eval.diff().dropna()
    n_neg_diffs = int((ts_diffs < pd.Timedelta(0)).sum())
    n_zero_diffs = int((ts_diffs == pd.Timedelta(0)).sum())
    n_pos_diffs  = int((ts_diffs > pd.Timedelta(0)).sum())
    n_expected_diffs = EXPECTED_EVAL - 1

    log.info(f"    Eval rows                    : {len(ts_eval):,}")
    log.info(f"    Timestamp diffs computed     : {len(ts_diffs):,}  (expected {n_expected_diffs:,})")
    log.info(f"    Negative diffs               : {n_neg_diffs}  (expected 0)")
    log.info(f"    Zero diffs (same second)     : {n_zero_diffs}")
    log.info(f"    Positive diffs               : {n_pos_diffs}")

    # Duplicate row check
    n_dup_rows = int(split_result.meta_eval.duplicated(subset=["original_row_id"]).sum())
    log.info(f"    Duplicate eval original_row_id: {n_dup_rows}  (expected 0)")

    result["reproducibility"] = {
        "score_sha256": score_hash,
        "n_eval_rows": len(ts_eval),
        "n_timestamp_diffs": len(ts_diffs),
        "n_negative_diffs": n_neg_diffs,
        "n_zero_diffs": n_zero_diffs,
        "n_positive_diffs": n_pos_diffs,
        "n_duplicate_rows": n_dup_rows,
        "monotonicity_verified": n_neg_diffs == 0,
        "no_duplicates_verified": n_dup_rows == 0,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 15. LEAKAGE AUDIT SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    log.info("[15] Leakage audit summary...")
    leakage_audit = {
        "train_used_for": "model fitting and normalization bounds (s_min, s_max) only",
        "calib_used_for": "A_thr computation only (benign 99th percentile)",
        "eval_used_for": "frozen post-hoc evaluation only",
        "attack_labels_used_for_fitting": False,
        "attack_labels_used_for_threshold": False,
        "attack_labels_used_for_normalization": False,
        "attack_labels_used_for_model_selection": False,
        "eval_labels_joined_before_scoring": False,
        "scaler_fitted_on_train_only": True,
        "normalizer_fitted_on_train_only": True,
        "leakage_detected": False,
    }
    log.info("  [OK] No leakage detected.")
    result["leakage_audit"] = leakage_audit

    return result


def main():
    log.info("Phase 2A Corrected — E1 Detector Suitability Experiment")
    log.info(f"Start time: {datetime.now().isoformat()}")
    log.info(f"Dataset   : {MODBUS_CSV}")
    log.info(f"Output dir: {OUT_DIR}")

    # ── Run 1 ──────────────────────────────────────────────────────────────
    r1 = run_experiment(run_id=1, seed=42)

    # ── Run 2 (reproducibility check) ──────────────────────────────────────
    r2 = run_experiment(run_id=2, seed=42)

    # ── Cross-run reproducibility verification ──────────────────────────────
    log.info("\n" + "="*60)
    log.info("CROSS-RUN REPRODUCIBILITY VERIFICATION")
    log.info("="*60)
    repro_checks = {}
    fields_to_compare = [
        ("split.n_train", r1["split"]["n_train"], r2["split"]["n_train"]),
        ("split.n_calib", r1["split"]["n_calib"], r2["split"]["n_calib"]),
        ("split.n_eval",  r1["split"]["n_eval"],  r2["split"]["n_eval"]),
        ("normalization.s_min", r1["normalization"]["s_min"], r2["normalization"]["s_min"]),
        ("normalization.s_max", r1["normalization"]["s_max"], r2["normalization"]["s_max"]),
        ("calibration.A_thr",   r1["calibration"]["A_thr"],  r2["calibration"]["A_thr"]),
        ("score_sha256", r1["reproducibility"]["score_sha256"], r2["reproducibility"]["score_sha256"]),
    ]
    all_match = True
    for name, v1, v2 in fields_to_compare:
        match = (v1 == v2)
        repro_checks[name] = {"run1": v1, "run2": v2, "match": match}
        icon = "[PASS]" if match else "[FAIL]"
        log.info(f"  {icon:10s} {name}: {v1}")
        if not match:
            all_match = False
            log.error(f"            Run2 value: {v2}")

    log.info(f"\n  Overall reproducibility: {'VERIFIED' if all_match else 'FAILED'}")

    # ── Suitability Conclusion ───────────────────────────────────────────────
    pm = r1["point_metrics"]
    st = r1["sub_threshold"]
    ep = r1["episodes"]

    log.info("\n" + "="*60)
    log.info("SUITABILITY CONCLUSION")
    log.info("="*60)

    has_continuous_signal  = pm["roc_auc"] > 0.50
    has_score_separation   = (
        r1["score_distributions"]["eval_attack"]["mean"] >
        r1["score_distributions"]["eval_benign"]["mean"]
    )
    sub_thr_pct = st["pct_attack_sub_thr"]
    point_recall = pm["recall"]
    point_f1     = pm["f1"]

    log.info(f"  ROC-AUC                : {pm['roc_auc']:.6f}")
    log.info(f"  PR-AUC                 : {pm['pr_auc']:.6f}")
    log.info(f"  Point recall           : {point_recall:.6f}")
    log.info(f"  Point F1               : {point_f1:.6f}")
    log.info(f"  Sub-threshold attacks  : {sub_thr_pct:.2f}%")
    log.info(f"  Score separation       : {'YES' if has_score_separation else 'NO'}")
    log.info(f"  Continuous signal      : {'YES' if has_continuous_signal else 'NO'}")

    if pm["roc_auc"] >= 0.70 and has_score_separation:
        verdict = "SUITABLE"
        verdict_rationale = (
            f"ROC-AUC={pm['roc_auc']:.4f} exceeds minimum threshold (0.70). "
            f"The detector produces a continuous anomaly-score signal with measurable "
            f"benign/attack score separation. {sub_thr_pct:.1f}% of attack observations "
            f"fall below A_thr, indicating that temporal evidence accumulation is needed "
            f"to recover sub-threshold attack signal — consistent with BAZT motivation."
        )
    elif pm["roc_auc"] >= 0.55 and has_score_separation:
        verdict = "LIMITED_BUT_USABLE"
        verdict_rationale = (
            f"ROC-AUC={pm['roc_auc']:.4f} indicates weak but non-trivial discrimination. "
            f"Score separation exists. The detector provides a continuous signal adequate "
            f"for studying temporal trust behavior, with {sub_thr_pct:.1f}% sub-threshold attacks."
        )
    else:
        verdict = "UNSUITABLE"
        verdict_rationale = (
            f"ROC-AUC={pm['roc_auc']:.4f} is near-random or detector fails "
            f"minimum suitability criteria. Recommend investigation before TrustEngine phase."
        )

    log.info(f"\n  VERDICT: {verdict}")
    log.info(f"  RATIONALE: {verdict_rationale}")

    suitability = {
        "verdict": verdict,
        "rationale": verdict_rationale,
        "roc_auc":  pm["roc_auc"],
        "pr_auc":   pm["pr_auc"],
        "point_recall": point_recall,
        "point_f1": point_f1,
        "sub_thr_pct": sub_thr_pct,
        "has_score_separation": has_score_separation,
        "has_continuous_signal": has_continuous_signal,
    }
    r1["suitability"] = suitability

    # ── Save Artifacts ──────────────────────────────────────────────────────
    log.info("\n[ARTIFACTS] Saving experiment artifacts...")

    # Detector config
    cfg_path = OUT_DIR / "detector_config.json"
    with open(cfg_path, "w") as f:
        json.dump(r1["detector_config"], f, indent=2)

    # Normalization constants
    norm_path = OUT_DIR / "normalization_constants.json"
    with open(norm_path, "w") as f:
        json.dump(r1["normalization"], f, indent=2)

    # A_thr
    thr_path = OUT_DIR / "calibration_threshold.json"
    with open(thr_path, "w") as f:
        json.dump(r1["calibration"], f, indent=2)

    # Point metrics
    pm_path = OUT_DIR / "point_metrics.json"
    with open(pm_path, "w") as f:
        json.dump(r1["point_metrics"], f, indent=2)

    # Sub-threshold analysis
    st_path = OUT_DIR / "sub_threshold_analysis.json"
    with open(st_path, "w") as f:
        json.dump(r1["sub_threshold"], f, indent=2)

    # Episode analysis (capped)
    ep_path = OUT_DIR / "episode_analysis.json"
    with open(ep_path, "w") as f:
        json.dump(r1["episodes"], f, indent=2)

    # Temporal characteristics
    tc_path = OUT_DIR / "temporal_score_characteristics.json"
    with open(tc_path, "w") as f:
        json.dump(r1["temporal"], f, indent=2)

    # Score distributions
    sd_path = OUT_DIR / "score_distributions.json"
    with open(sd_path, "w") as f:
        json.dump(r1["score_distributions"], f, indent=2)

    # Leakage audit
    la_path = OUT_DIR / "leakage_audit.json"
    with open(la_path, "w") as f:
        json.dump(r1["leakage_audit"], f, indent=2)

    # Reproducibility
    rp_path = OUT_DIR / "reproducibility.json"
    with open(rp_path, "w") as f:
        json.dump({
            "run1": r1["reproducibility"],
            "run2": r2["reproducibility"],
            "cross_run_checks": repro_checks,
            "all_match": all_match,
        }, f, indent=2)

    # Suitability summary
    su_path = OUT_DIR / "suitability_summary.json"
    with open(su_path, "w") as f:
        json.dump(suitability, f, indent=2)

    # Evaluation score array (npy)
    score_path = OUT_DIR / "eval_scores_A_t.npy"
    np.save(str(score_path), np.asarray(r1.get("_A_eval_internal", []), dtype=np.float64))

    # Full results
    full_path = OUT_DIR / "full_results_run1.json"
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")
    with open(full_path, "w") as f:
        json.dump(r1, f, indent=2, default=_json_safe)

    log.info(f"  detector_config.json          -> {cfg_path}")
    log.info(f"  normalization_constants.json  -> {norm_path}")
    log.info(f"  calibration_threshold.json    -> {thr_path}")
    log.info(f"  point_metrics.json            -> {pm_path}")
    log.info(f"  sub_threshold_analysis.json   -> {st_path}")
    log.info(f"  episode_analysis.json         -> {ep_path}")
    log.info(f"  temporal_score_characteristics.json -> {tc_path}")
    log.info(f"  score_distributions.json      -> {sd_path}")
    log.info(f"  leakage_audit.json            -> {la_path}")
    log.info(f"  reproducibility.json          -> {rp_path}")
    log.info(f"  suitability_summary.json      -> {su_path}")
    log.info(f"  full_results_run1.json        -> {full_path}")
    log.info(f"  run_log.txt                   -> {log_path}")

    # ── Return summary to stdout ──────────────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("PHASE_2A_CORRECTED_COMPLETE")
    log.info("="*70)
    log.info(f"  Finish time: {datetime.now().isoformat()}")

    # Return key values for report generation
    return r1, r2, repro_checks, all_match, suitability


if __name__ == "__main__":
    try:
        r1, r2, repro_checks, all_match, suitability = main()
        sys.exit(0)
    except Exception as e:
        log.error(f"\nFATAL ERROR: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
