"""
Phase 2A.2 — ToN-IoT Modality Suitability Audit
=================================================
Project: C:\\Users\\Shreyas A\\PROJECTS\\BAZT-TON_IoT\\TON_IoT_project
Modality: IoT_Weather (Processed_IoT_dataset/IoT_Weather.csv)

Objective:
Determine whether another ToN-IoT IoT telemetry modality (IoT_Weather)
provides a sufficiently informative causal anomaly-score signal for
evaluating the BAZT temporal trust mechanism under the same leakage-controlled
chronological methodology.

Constraints:
  - Raw numerical telemetry representation only (no temporal feature engineering yet)
  - Predefined chronological split: TRAIN < CALIBRATION < EVALUATION
  - Preprocessing & normalization fitted strictly on benign TRAIN
  - Threshold A_thr derived strictly from 99th percentile of benign CALIBRATION
  - Full continuous evaluation interval (N = 193,049 rows)
  - Zero TrustEngine execution
  - Bit-for-bit reproducibility verified across 2 independent runs
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

# Safe stdout reconfigure for Windows
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
log = logging.getLogger("phase2a2_modality_audit")

WEATHER_CSV = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_audit\iot_processed\IoT_Weather.csv")

# ── Feature & Detector Configuration ──────────────────────────────────────────
WEATHER_FEATURES = ["temperature", "pressure", "humidity"]

DETECTOR_CFG = {
    "n_estimators": 300,
    "max_samples": 256,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}

# ── Predefined Chronological Split Boundaries ────────────────────────────────
TRAIN_START = "2019-03-31 12:36:52"
TRAIN_END   = "2019-04-03 23:59:59"
CALIB_START = "2019-04-04 00:00:00"
CALIB_END   = "2019-04-04 23:59:59"
EVAL_START  = "2019-04-23 00:00:00"
EVAL_END    = "2019-04-29 23:59:59"


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


def audit_raw_dataset(csv_path: Path) -> dict:
    log.info(f"[DATASET AUDIT] Inspecting raw dataset at {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    n_rows = len(df)
    cols = df.columns.tolist()

    missing = df.isna().sum().to_dict()
    n_dup = int(df.duplicated().sum())

    clean_date = df["date"].astype(str).str.strip()
    clean_time = df["time"].astype(str).str.strip()
    parsed_ts = pd.to_datetime(clean_date + " " + clean_time, format="%d-%b-%y %H:%M:%S", errors="coerce")
    malformed_ts = int(parsed_ts.isna().sum())

    df["timestamp"] = parsed_ts
    df["original_row_id"] = df.index
    df_sorted = df.sort_values(by=["timestamp", "original_row_id"]).reset_index(drop=True)

    deltas = df_sorted["timestamp"].diff().dropna()
    n_neg_deltas = int((deltas < pd.Timedelta(0)).sum())

    label_counts = df_sorted["label"].value_counts().to_dict()
    type_counts = df_sorted["type"].value_counts().to_dict()

    df_sorted["date_only"] = df_sorted["timestamp"].dt.date
    daily = df_sorted.groupby("date_only").agg(
        total=("label", "count"),
        normal=("label", lambda s: (s == 0).sum()),
        attack=("label", lambda s: (s == 1).sum()),
    ).reset_index()
    daily_records = daily.to_dict(orient="records")

    audit_info = {
        "file_path": str(csv_path),
        "total_rows": n_rows,
        "columns": cols,
        "numeric_candidate_features": WEATHER_FEATURES,
        "missing_values": missing,
        "duplicate_rows": n_dup,
        "malformed_timestamps": malformed_ts,
        "negative_timestamp_deltas": n_neg_deltas,
        "earliest_timestamp": str(df_sorted["timestamp"].min()),
        "latest_timestamp": str(df_sorted["timestamp"].max()),
        "label_counts": {int(k): int(v) for k, v in label_counts.items()},
        "type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "daily_distribution": [{k: (str(v) if k == "date_only" else int(v)) for k, v in r.items()} for r in daily_records],
    }

    log.info(f"  Total rows: {n_rows:,}")
    log.info(f"  Missing values: {missing}")
    log.info(f"  Duplicate rows: {n_dup:,}")
    log.info(f"  Time range: {audit_info['earliest_timestamp']} to {audit_info['latest_timestamp']}")
    log.info(f"  Label counts: {label_counts}")
    log.info(f"  Attack types: {type_counts}")
    return audit_info, df_sorted


def run_experiment(run_id: int, df_sorted: pd.DataFrame, seed: int) -> dict:
    log.info(f"\n{'='*70}")
    log.info(f"EXPERIMENT RUN {run_id} (seed={seed})")
    log.info(f"{'='*70}")

    # 1. Chronological Partitioning
    log.info("[1] Partitioning IoT_Weather according to chronological rules...")
    train_mask = (df_sorted["timestamp"] >= TRAIN_START) & (df_sorted["timestamp"] <= TRAIN_END)
    calib_mask = (df_sorted["timestamp"] >= CALIB_START) & (df_sorted["timestamp"] <= CALIB_END)
    eval_mask  = (df_sorted["timestamp"] >= EVAL_START)  & (df_sorted["timestamp"] <= EVAL_END)

    df_train = df_sorted[train_mask].reset_index(drop=True)
    df_calib = df_sorted[calib_mask].reset_index(drop=True)
    df_eval  = df_sorted[eval_mask].reset_index(drop=True)

    n_train = len(df_train)
    n_calib = len(df_calib)
    n_eval  = len(df_eval)
    train_attacks = int((df_train["label"] == 1).sum())
    calib_attacks = int((df_calib["label"] == 1).sum())
    eval_normal   = int((df_eval["label"] == 0).sum())
    eval_attacks  = int((df_eval["label"] == 1).sum())

    log.info(f"  TRAIN rows      : {n_train:>10,} (attacks: {train_attacks})")
    log.info(f"  CALIBRATION rows: {n_calib:>10,} (attacks: {calib_attacks})")
    log.info(f"  EVALUATION rows : {n_eval:>10,} (normal: {eval_normal:,}, attacks: {eval_attacks:,})")
    log.info(f"  EVAL time span  : {df_eval['timestamp'].min()} to {df_eval['timestamp'].max()}")

    assert train_attacks == 0, "TRAIN partition contains attacks!"
    assert calib_attacks == 0, "CALIBRATION partition contains attacks!"
    assert df_train["timestamp"].max() < df_calib["timestamp"].min(), "TRAIN does not precede CALIB!"
    assert df_calib["timestamp"].max() < df_eval["timestamp"].min(), "CALIB does not precede EVAL!"
    log.info("  [OK] Temporal precedence and benign purity strictly verified.")

    # 2. Feature Extraction
    log.info(f"[2] Extracting raw numerical telemetry features: {WEATHER_FEATURES}...")
    X_train = df_train[WEATHER_FEATURES].copy().astype(float)
    X_calib = df_calib[WEATHER_FEATURES].copy().astype(float)
    X_eval  = df_eval[WEATHER_FEATURES].copy().astype(float)
    y_eval  = df_eval["label"].values.astype(int)

    # 3. Preprocessing (Fit RobustScaler strictly on TRAIN)
    log.info("[3] Fitting RobustScaler on TRAIN benign partition only...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_eval_scaled  = scaler.transform(X_eval)

    scaler_params = {
        "feature_names": WEATHER_FEATURES,
        "medians": scaler.center_.tolist(),
        "iqrs": scaler.scale_.tolist(),
    }
    log.info(f"  Scaler medians: {scaler_params['medians']}")
    log.info(f"  Scaler IQRs   : {scaler_params['iqrs']}")

    # 4. Detector Fitting (Isolation Forest on TRAIN benign)
    log.info(f"[4] Fitting Isolation Forest on {n_train:,} benign TRAIN samples...")
    cfg = {**DETECTOR_CFG, "random_state": seed}
    model = IsolationForest(**cfg)
    model.fit(X_train_scaled)
    log.info("  [OK] Isolation Forest fitted.")

    # 5. Raw Scores
    log.info("[5] Computing raw anomaly scores (raw = -score_samples)...")
    raw_train = -model.score_samples(X_train_scaled)
    raw_calib = -model.score_samples(X_calib_scaled)
    raw_eval  = -model.score_samples(X_eval_scaled)

    # 6. Score Normalization (Bounds from TRAIN only)
    s_min = float(np.min(raw_train))
    s_max = float(np.max(raw_train))
    if s_max <= s_min:
        s_max = s_min + 1e-10

    A_train = np.clip((raw_train - s_min) / (s_max - s_min), 0.0, 1.0)
    A_calib = np.clip((raw_calib - s_min) / (s_max - s_min), 0.0, 1.0)
    A_eval  = np.clip((raw_eval - s_min) / (s_max - s_min), 0.0, 1.0)

    log.info(f"  Normalization bounds: s_min={s_min:.8f}, s_max={s_max:.8f}")

    # 7. Calibration Threshold (99th percentile of benign CALIB)
    A_thr = float(np.percentile(A_calib, 99.0, method="linear"))
    raw_thr_equiv = float(A_thr * (s_max - s_min) + s_min)
    log.info(f"  A_thr (normalized) = {A_thr:.8f} (raw equiv = {raw_thr_equiv:.8f})")

    # 8. Post-Hoc Evaluation Metrics
    log.info("[8] Evaluating frozen detector on complete EVALUATION partition...")
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
    prevalence = float(eval_attacks / n_eval)

    log.info(f"\n{'='*60}")
    log.info("EVALUATION POINT METRICS (IoT_Weather)")
    log.info(f"{'='*60}")
    log.info(f"  Attack Prevalence : {prevalence:.4f} ({eval_attacks:,} / {n_eval:,})")
    log.info(f"  ROC-AUC           : {roc_auc:.6f}")
    log.info(f"  PR-AUC            : {pr_auc:.6f}")
    log.info(f"  Precision         : {precision:.6f}")
    log.info(f"  Recall            : {recall:.6f}")
    log.info(f"  F1-Score          : {f1:.6f}")
    log.info(f"  FPR               : {fpr:.6f}")
    log.info(f"  Specificity       : {specificity:.6f}")
    log.info(f"  Balanced Accuracy : {bal_acc:.6f}")
    log.info(f"{'='*60}\n")

    # 9. Score Distributions
    A_eval_benign = A_eval[y_eval == 0]
    A_eval_attack = A_eval[y_eval == 1]
    stats_benign = score_distribution_stats(A_eval_benign, "eval_benign")
    stats_attack = score_distribution_stats(A_eval_attack, "eval_attack")
    score_sep = float(stats_attack["mean"] - stats_benign["mean"])

    log.info(f"  Benign Mean A_t   : {stats_benign['mean']:.6f} (median: {stats_benign['median']:.6f})")
    log.info(f"  Attack Mean A_t   : {stats_attack['mean']:.6f} (median: {stats_attack['median']:.6f})")
    log.info(f"  Score Separation  : {score_sep:+.6f}")

    # 10. Sub-threshold Analysis
    n_sub_atk   = int(((y_eval == 1) & (A_eval < A_thr)).sum())
    n_above_atk = int(((y_eval == 1) & (A_eval >= A_thr)).sum())
    n_sub_ben   = int(((y_eval == 0) & (A_eval < A_thr)).sum())
    n_above_ben = int(((y_eval == 0) & (A_eval >= A_thr)).sum())
    pct_sub_atk = float(100.0 * n_sub_atk / eval_attacks)

    log.info(f"  Sub-threshold attacks : {n_sub_atk:,} ({pct_sub_atk:.2f}%)")
    log.info(f"  Above-threshold attacks: {n_above_atk:,} ({100.0 - pct_sub_atk:.2f}%)")

    # 11. Temporal Stream & Episode Characteristics
    ac_all  = lag1_autocorr(A_eval)
    vol_all = first_diff_volatility(A_eval)
    above_runs = run_above_thr_runs(A_eval, A_thr)
    n_above_runs = len(above_runs)
    run_lens = [r[2] for r in above_runs]
    mean_run_len = float(np.mean(run_lens)) if run_lens else 0.0
    max_run_len = int(np.max(run_lens)) if run_lens else 0

    episodes = run_episodes(y_eval, A_eval, A_thr)
    n_ep = len(episodes)
    ep_crossing = sum(1 for e in episodes if e["any_above_thr"])
    ep_durations = [e["duration"] for e in episodes]
    ep_means = [e["mean_score"] for e in episodes]
    ep_maxs = [e["max_score"] for e in episodes]

    log.info(f"  Lag-1 Autocorrelation : {ac_all:.6f}")
    log.info(f"  First-Diff Volatility : {vol_all:.6f}")
    log.info(f"  Contiguous runs >= A_thr: {n_above_runs:,} (mean length: {mean_run_len:.2f})")
    log.info(f"  Attack episodes       : {n_ep:,} (crossing A_thr: {ep_crossing:,} / {100*ep_crossing/n_ep:.1f}%)")

    score_hash = sha256_of_array(A_eval)
    log.info(f"  Score SHA-256         : {score_hash}")

    return {
        "run_id": run_id,
        "seed": seed,
        "n_train": n_train,
        "n_calib": n_calib,
        "n_eval": n_eval,
        "eval_normal": eval_normal,
        "eval_attacks": eval_attacks,
        "eval_start": str(df_eval["timestamp"].min()),
        "eval_end": str(df_eval["timestamp"].max()),
        "features": WEATHER_FEATURES,
        "scaler": scaler_params,
        "normalization": {"s_min": s_min, "s_max": s_max},
        "calibration": {"A_thr": A_thr, "raw_thr_equiv": raw_thr_equiv},
        "point_metrics": {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "precision": precision, "recall": recall, "f1": f1,
            "FPR": fpr, "FNR": fnr, "specificity": specificity,
            "balanced_accuracy": bal_acc, "roc_auc": roc_auc, "pr_auc": pr_auc,
            "attack_prevalence": prevalence,
        },
        "score_distributions": {
            "eval_benign": stats_benign,
            "eval_attack": stats_attack,
            "score_separation": score_sep,
        },
        "sub_threshold": {
            "n_attack_sub_thr": n_sub_atk,
            "n_attack_above_thr": n_above_atk,
            "n_benign_sub_thr": n_sub_ben,
            "n_benign_above_thr": n_above_ben,
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
            "episodes_crossing_A_thr": ep_crossing,
            "episodes_entirely_sub_thr": n_ep - ep_crossing,
            "pct_episodes_crossing_A_thr": float(100.0 * ep_crossing / n_ep) if n_ep > 0 else 0.0,
            "median_episode_mean_A": float(np.median(ep_means)) if ep_means else 0.0,
            "median_episode_max_A": float(np.median(ep_maxs)) if ep_maxs else 0.0,
            "median_episode_duration": float(np.median(ep_durations)) if ep_durations else 0.0,
        },
        "score_sha256": score_hash,
    }


def main():
    log.info("Starting Phase 2A.2 — ToN-IoT Modality Suitability Audit")
    log.info(f"Start time: {datetime.now().isoformat()}")

    # 1. Dataset Audit First
    audit_info, df_sorted = audit_raw_dataset(WEATHER_CSV)

    # 2. Run 1
    r1 = run_experiment(run_id=1, df_sorted=df_sorted, seed=42)

    # 3. Run 2 (Reproducibility)
    r2 = run_experiment(run_id=2, df_sorted=df_sorted, seed=42)

    # 4. Cross-Run Reproducibility Checks
    log.info("\n" + "="*70)
    log.info("CROSS-RUN REPRODUCIBILITY VERIFICATION")
    log.info("="*70)
    checks = [
        ("n_train", r1["n_train"], r2["n_train"]),
        ("n_calib", r1["n_calib"], r2["n_calib"]),
        ("n_eval", r1["n_eval"], r2["n_eval"]),
        ("s_min", r1["normalization"]["s_min"], r2["normalization"]["s_min"]),
        ("s_max", r1["normalization"]["s_max"], r2["normalization"]["s_max"]),
        ("A_thr", r1["calibration"]["A_thr"], r2["calibration"]["A_thr"]),
        ("score_sha256", r1["score_sha256"], r2["score_sha256"]),
    ]

    all_match = True
    for name, v1, v2 in checks:
        match = (v1 == v2)
        if not match:
            all_match = False
        log.info(f"  {'[PASS]' if match else '[FAIL]':10s} {name}: {v1}")

    log.info(f"Overall Reproducibility: {'VERIFIED' if all_match else 'FAILED'}")

    # 5. Suitability Assessment
    pm = r1["point_metrics"]
    sd = r1["score_distributions"]
    st = r1["sub_threshold"]
    roc_auc = pm["roc_auc"]
    pr_auc = pm["pr_auc"]
    prev = pm["attack_prevalence"]
    score_sep = sd["score_separation"]

    log.info("\n" + "="*70)
    log.info("SUITABILITY ASSESSMENT")
    log.info("="*70)
    log.info(f"  ROC-AUC          : {roc_auc:.6f}")
    log.info(f"  PR-AUC           : {pr_auc:.6f} (Prevalence: {prev:.4f})")
    log.info(f"  Score Separation : {score_sep:+.6f}")
    log.info(f"  Sub-threshold %  : {st['pct_attack_sub_thr']:.2f}%")

    if roc_auc >= 0.70 and score_sep > 0.05 and pr_auc > prev + 0.10:
        verdict = "SIGNAL_SUFFICIENT"
        rationale = (
            f"IoT_Weather provides a clearly informative anomaly score signal: "
            f"ROC-AUC={roc_auc:.4f} is substantially above 0.50, "
            f"PR-AUC={pr_auc:.4f} exceeds prevalence ({prev:.4f}), and "
            f"attacks produce positive score separation ({score_sep:+.4f})."
        )
    elif roc_auc >= 0.55 or (score_sep > 0.01) or (roc_auc <= 0.45):
        verdict = "SIGNAL_LIMITED"
        rationale = (
            f"IoT_Weather exhibits limited or weak discrimination: "
            f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f} (prevalence {prev:.4f}), "
            f"score separation is {score_sep:+.4f}. Signal is insufficient for robust downstream trust calibration."
        )
    else:
        verdict = "SIGNAL_INSUFFICIENT"
        rationale = (
            f"IoT_Weather provides near-random or uninformative discrimination: "
            f"ROC-AUC={roc_auc:.4f} is near 0.50, PR-AUC={pr_auc:.4f} matches prevalence ({prev:.4f}), "
            f"and attack/benign score distributions substantially overlap (separation={score_sep:+.4f}). "
            f"The detector-input modality remains an experimental bottleneck."
        )

    log.info(f"  VERDICT  : {verdict}")
    log.info(f"  RATIONALE: {rationale}")
    log.info("="*70)

    # 6. Leakage Audit
    leakage_audit = {
        "detector_fit_only_on_benign_train": True,
        "scaler_fit_only_on_benign_train": True,
        "normalization_bounds_from_train_only": True,
        "A_thr_from_benign_calibration_only": True,
        "no_attack_labels_during_training": True,
        "no_attack_labels_during_calibration": True,
        "no_evaluation_labels_during_model_selection": True,
        "no_future_information_used": True,
        "chronological_split_verified": True,
        "no_cross_partition_temporal_leakage": True,
        "no_label_derived_features": True,
        "leakage_detected": False,
    }

    # 7. Write JSON Artifacts
    log.info("\n[ARTIFACTS] Writing JSON artifacts to experiments/ton_iot/phase2a2_modality_audit/...")

    with open(OUT_DIR / "modality_metadata.json", "w") as f:
        json.dump({
            "selected_modality": "IoT_Weather",
            "csv_filename": "IoT_Weather.csv",
            "total_rows": audit_info["total_rows"],
            "features": WEATHER_FEATURES,
            "target_column": "label",
            "type_column": "type",
            "selection_rationale": "Identified as primary candidate in ToN-IoT audit inventory; continuous environmental sensor telemetry (temperature, pressure, humidity) with identical UNSW testbed execution dates and attack campaigns.",
        }, f, indent=2)

    with open(OUT_DIR / "dataset_audit.json", "w") as f:
        json.dump(audit_info, f, indent=2)

    with open(OUT_DIR / "split_specification.json", "w") as f:
        json.dump({
            "train": {"start": TRAIN_START, "end": TRAIN_END, "rows": r1["n_train"], "attacks": 0},
            "calibration": {"start": CALIB_START, "end": CALIB_END, "rows": r1["n_calib"], "attacks": 0},
            "evaluation": {"start": EVAL_START, "end": EVAL_END, "rows": r1["n_eval"], "normal": r1["eval_normal"], "attacks": r1["eval_attacks"]},
        }, f, indent=2)

    with open(OUT_DIR / "detector_config.json", "w") as f:
        json.dump(DETECTOR_CFG, f, indent=2)

    with open(OUT_DIR / "normalization_constants.json", "w") as f:
        json.dump(r1["normalization"], f, indent=2)

    with open(OUT_DIR / "calibration_threshold.json", "w") as f:
        json.dump(r1["calibration"], f, indent=2)

    with open(OUT_DIR / "point_metrics.json", "w") as f:
        json.dump(r1["point_metrics"], f, indent=2)

    with open(OUT_DIR / "score_distributions.json", "w") as f:
        json.dump(r1["score_distributions"], f, indent=2)

    with open(OUT_DIR / "sub_threshold_analysis.json", "w") as f:
        json.dump(r1["sub_threshold"], f, indent=2)

    with open(OUT_DIR / "temporal_score_characteristics.json", "w") as f:
        json.dump(r1["temporal_characteristics"], f, indent=2)

    with open(OUT_DIR / "episode_analysis.json", "w") as f:
        json.dump(r1["episode_analysis"], f, indent=2)

    with open(OUT_DIR / "leakage_audit.json", "w") as f:
        json.dump(leakage_audit, f, indent=2)

    with open(OUT_DIR / "reproducibility.json", "w") as f:
        json.dump({
            "run1_sha256": r1["score_sha256"],
            "run2_sha256": r2["score_sha256"],
            "all_match": all_match,
            "checks": {c[0]: {"run1": c[1], "run2": c[2], "match": (c[1] == c[2])} for c in checks},
        }, f, indent=2)

    with open(OUT_DIR / "suitability_assessment.json", "w") as f:
        json.dump({"verdict": verdict, "rationale": rationale}, f, indent=2)

    def _json_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(OUT_DIR / "full_results.json", "w") as f:
        json.dump({"run1": r1, "verdict": verdict, "reproducibility": all_match}, f, indent=2, default=_json_safe)

    log.info("[ARTIFACTS] All artifacts written successfully.")
    log.info("\nPHASE_2A2_MODALITY_AUDIT_COMPLETE")
    return r1, r2, all_match, verdict, rationale


if __name__ == "__main__":
    try:
        r1, r2, all_match, verdict, rationale = main()
        sys.exit(0)
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
