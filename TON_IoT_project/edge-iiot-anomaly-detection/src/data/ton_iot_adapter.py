"""ToN-IoT Modbus data adapter and chronological splitter matching E1 protocol."""
import logging
from pathlib import Path
from typing import NamedTuple, Any

import numpy as np
import pandas as pd

from config.settings import config, TonIoTDatasetConfig

logger = logging.getLogger(__name__)

# Exact frozen Modbus feature columns (in deterministic order)
TON_IOT_FEATURE_COLUMNS = [
    "FC1_Read_Input_Register",
    "FC2_Read_Discrete_Value",
    "FC3_Read_Holding_Register",
    "FC4_Read_Coil",
]

# Required raw columns in IoT_Modbus.csv
TON_IOT_REQUIRED_COLUMNS = [
    "date",
    "time",
    "FC1_Read_Input_Register",
    "FC2_Read_Discrete_Value",
    "FC3_Read_Holding_Register",
    "FC4_Read_Coil",
    "label",
    "type",
]

TON_IOT_LABEL_COLUMNS = ["label", "type"]
TON_IOT_METADATA_COLUMNS = ["date", "time", "timestamp", "original_row_id"]


class TonIoTSplitResult(NamedTuple):
    """Encapsulates split datasets ensuring complete isolation of features from labels."""

    # Model features only (strictly the 4 numeric Modbus registers)
    X_train: pd.DataFrame
    X_calib: pd.DataFrame
    X_eval: pd.DataFrame

    # Temporal metadata (timestamps, original row identities)
    meta_train: pd.DataFrame
    meta_calib: pd.DataFrame
    meta_eval: pd.DataFrame

    # Ground truth labels (reserved strictly for evaluation reporting; NEVER passed to model)
    labels_train: pd.DataFrame
    labels_calib: pd.DataFrame
    labels_eval: pd.DataFrame

    # Comprehensive verification & discrepancy report
    discrepancy_report: dict[str, Any]


class TonIoTDataAdapter:
    """Adapter for ingesting and validating ToN-IoT SCADA Modbus telemetry."""

    def __init__(self, filepath: Path | str | None = None):
        if filepath is not None:
            self.filepath = Path(filepath)
        else:
            self.filepath = config.paths.data_raw / config.ton_iot_dataset.filename

    def load_raw(self) -> pd.DataFrame:
        """Load raw CSV file from disk."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"ToN-IoT dataset file not found at: {self.filepath}")
        logger.info(f"Loading raw ToN-IoT Modbus CSV from {self.filepath}...")
        df = pd.read_csv(self.filepath, low_memory=False)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validate presence of all required columns and non-null constraints."""
        missing = [col for col in TON_IOT_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"ToN-IoT dataset schema violation! Missing required columns: {missing}")

        # Check for missing values in required columns
        null_counts = df[TON_IOT_REQUIRED_COLUMNS].isna().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise ValueError(
                f"ToN-IoT dataset contains missing values in columns: {cols_with_nulls.to_dict()}. "
                "Per protocol, missing values must not be silently ignored."
            )

    def reconstruct_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministically parse and validate timestamps, enforcing monotonic chronological order."""
        logger.info("Reconstructing deterministic timestamps from 'date' + 'time'...")
        self.validate_schema(df)

        df_out = df.copy()

        # Preserve original raw row identity
        df_out["original_row_id"] = df_out.index.copy()

        # Deterministic string cleaning
        clean_date = df_out["date"].astype(str).str.strip()
        clean_time = df_out["time"].astype(str).str.strip()
        datetime_str = clean_date + " " + clean_time

        # Parse with strict format %d-%b-%y %H:%M:%S
        parsed_ts = pd.to_datetime(datetime_str, format="%d-%b-%y %H:%M:%S", errors="coerce")

        # Detect malformed / unparseable timestamps
        malformed_mask = parsed_ts.isna()
        if malformed_mask.any():
            malformed_count = int(malformed_mask.sum())
            malformed_indices = df_out.index[malformed_mask].tolist()[:10]
            raise ValueError(
                f"Malformed timestamps detected in {malformed_count} rows! "
                f"First unparseable row indices: {malformed_indices}. Malformed data cannot be silently discarded."
            )

        df_out["timestamp"] = parsed_ts

        # Sort chronologically, breaking ties deterministically with original_row_id
        df_sorted = df_out.sort_values(by=["timestamp", "original_row_id"]).reset_index(drop=True)

        # Verify monotonicity: ensure no negative chronological differences after sorting
        deltas = df_sorted["timestamp"].diff().dropna()
        negative_deltas = deltas[deltas < pd.Timedelta(0)]
        if not negative_deltas.empty:
            raise ValueError(
                f"Negative chronological delta detected after sorting! "
                f"Count: {len(negative_deltas)}, Minimum delta: {negative_deltas.min()}"
            )

        logger.info(
            f"Timestamp reconstruction successful. "
            f"Earliest: {df_sorted['timestamp'].min()}, Latest: {df_sorted['timestamp'].max()}"
        )
        return df_sorted

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract strictly the 4 continuous Modbus numeric registers in deterministic order."""
        for feat in TON_IOT_FEATURE_COLUMNS:
            if feat not in df.columns:
                raise KeyError(f"Feature '{feat}' not found in DataFrame.")

        # Ensure correct column ordering and numeric dtypes
        features_df = df[TON_IOT_FEATURE_COLUMNS].copy()
        for col in TON_IOT_FEATURE_COLUMNS:
            features_df[col] = pd.to_numeric(features_df[col], errors="raise")

        return features_df

    @staticmethod
    def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
        """Extract timestamp and original row identifiers."""
        return df[["timestamp", "date", "time", "original_row_id"]].copy()

    @staticmethod
    def extract_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Extract ground truth labels (binary label and multiclass type)."""
        return df[["label", "type"]].copy()


class TonIoTChronologicalSplitter:
    """Partitions ToN-IoT telemetry into strictly chronological TRAIN, CALIBRATION, and EVALUATION sets."""

    def __init__(self, dataset_config: TonIoTDatasetConfig | None = None):
        self.cfg = dataset_config or config.ton_iot_dataset

        self.train_start = pd.Timestamp(self.cfg.train_start)
        self.train_end = pd.Timestamp(self.cfg.train_end)
        self.calib_start = pd.Timestamp(self.cfg.calib_start)
        self.calib_end = pd.Timestamp(self.cfg.calib_end)
        self.eval_start = pd.Timestamp(self.cfg.eval_start)
        self.eval_end = pd.Timestamp(self.cfg.eval_end)

    def split(self, df_sorted: pd.DataFrame) -> TonIoTSplitResult:
        """Execute chronological partitioning and comprehensive protocol verification."""
        if "timestamp" not in df_sorted.columns:
            raise KeyError("DataFrame must contain 'timestamp' column. Run TonIoTDataAdapter.reconstruct_timestamps() first.")

        logger.info("Executing chronological partitioning according to frozen E1 boundaries...")

        # Partition 1: Training (31-Mar-2019 12:36:52 to 03-Apr-2019 23:59:59)
        train_mask = (df_sorted["timestamp"] >= self.train_start) & (df_sorted["timestamp"] <= self.train_end)
        df_train = df_sorted[train_mask].reset_index(drop=True)

        # Partition 2: Calibration (04-Apr-2019 00:00:00 to 04-Apr-2019 23:59:59)
        calib_mask = (df_sorted["timestamp"] >= self.calib_start) & (df_sorted["timestamp"] <= self.calib_end)
        df_calib = df_sorted[calib_mask].reset_index(drop=True)

        # Partition 3: Evaluation (23-Apr-2019 00:00:00 to 29-Apr-2019 23:59:58)
        eval_mask = (df_sorted["timestamp"] >= self.eval_start) & (df_sorted["timestamp"] <= self.eval_end)
        df_eval = df_sorted[eval_mask].reset_index(drop=True)

        # Verification 1: Strict Temporal Precedence: TRAIN < CALIBRATION < EVALUATION
        assert df_train["timestamp"].max() < df_calib["timestamp"].min(), (
            f"Temporal precedence violation: Train max ({df_train['timestamp'].max()}) "
            f"is not strictly less than Calib min ({df_calib['timestamp'].min()})"
        )
        assert df_calib["timestamp"].max() < df_eval["timestamp"].min(), (
            f"Temporal precedence violation: Calib max ({df_calib['timestamp'].max()}) "
            f"is not strictly less than Eval min ({df_eval['timestamp'].min()})"
        )

        # Verification 2: Zero Temporal Overlap
        train_ids = set(df_train["original_row_id"])
        calib_ids = set(df_calib["original_row_id"])
        eval_ids = set(df_eval["original_row_id"])
        assert train_ids.isdisjoint(calib_ids), "Train and Calibration partitions share overlapping rows!"
        assert calib_ids.isdisjoint(eval_ids), "Calibration and Evaluation partitions share overlapping rows!"
        assert train_ids.isdisjoint(eval_ids), "Train and Evaluation partitions share overlapping rows!"

        # Verification 3: Benign Purity of Train and Calibration
        train_attacks = int((df_train["label"] != self.cfg.normal_label).sum())
        calib_attacks = int((df_calib["label"] != self.cfg.normal_label).sum())
        if train_attacks > 0:
            raise ValueError(f"Protocol violation: Training partition contains {train_attacks} attack observations! Must be 100% normal.")
        if calib_attacks > 0:
            raise ValueError(f"Protocol violation: Calibration partition contains {calib_attacks} attack observations! Must be 100% normal.")

        # Verification 4: Sample Counts & Discrepancy Auditing
        actual_train = len(df_train)
        actual_calib = len(df_calib)
        actual_eval = len(df_eval)

        # Legacy recommendation check (Apr 23-29 excluding Apr 24 + including Apr 17)
        legacy_eval_mask = (
            ((df_sorted["timestamp"] >= self.eval_start) & (df_sorted["timestamp"] <= self.eval_end) & (df_sorted["timestamp"].dt.date != pd.to_datetime("2019-04-24").date()))
            | (df_sorted["timestamp"].dt.date == pd.to_datetime("2019-04-17").date())
        )
        legacy_eval_count = int(legacy_eval_mask.sum())

        discrepancy_report = {
            "train": {
                "start": str(df_train["timestamp"].min()),
                "end": str(df_train["timestamp"].max()),
                "expected_count": self.cfg.expected_train_count,
                "actual_count": actual_train,
                "count_match": (actual_train == self.cfg.expected_train_count),
                "normal_samples": int((df_train["label"] == self.cfg.normal_label).sum()),
                "attack_samples": train_attacks,
                "benign_purity_percent": 100.0 if train_attacks == 0 else 0.0,
            },
            "calibration": {
                "start": str(df_calib["timestamp"].min()),
                "end": str(df_calib["timestamp"].max()),
                "expected_count": self.cfg.expected_calib_count,
                "actual_count": actual_calib,
                "count_match": (actual_calib == self.cfg.expected_calib_count),
                "normal_samples": int((df_calib["label"] == self.cfg.normal_label).sum()),
                "attack_samples": calib_attacks,
                "benign_purity_percent": 100.0 if calib_attacks == 0 else 0.0,
            },
            "evaluation": {
                "start": str(df_eval["timestamp"].min()),
                "end": str(df_eval["timestamp"].max()),
                "expected_continuous_count": self.cfg.expected_eval_count_continuous,
                "actual_continuous_count": actual_eval,
                "continuous_count_match": (actual_eval == self.cfg.expected_eval_count_continuous),
                "expected_legacy_count": self.cfg.expected_eval_count_legacy,
                "actual_legacy_count": legacy_eval_count,
                "legacy_count_match": (legacy_eval_count == self.cfg.expected_eval_count_legacy),
                "normal_samples": int((df_eval["label"] == self.cfg.normal_label).sum()),
                "attack_samples": int((df_eval["label"] == self.cfg.attack_label).sum()),
                "attack_categories": df_eval[df_eval["label"] == self.cfg.attack_label]["type"].value_counts().to_dict(),
            },
            "temporal_precedence_verified": True,
            "zero_overlap_verified": True,
        }

        # Check expected counts
        if actual_train != self.cfg.expected_train_count:
            logger.warning(f"Train row count discrepancy! Expected {self.cfg.expected_train_count}, got {actual_train}")
        if actual_calib != self.cfg.expected_calib_count:
            logger.warning(f"Calibration row count discrepancy! Expected {self.cfg.expected_calib_count}, got {actual_calib}")

        # Separate features, metadata, and labels
        X_train = TonIoTDataAdapter.extract_features(df_train)
        X_calib = TonIoTDataAdapter.extract_features(df_calib)
        X_eval = TonIoTDataAdapter.extract_features(df_eval)

        meta_train = TonIoTDataAdapter.extract_metadata(df_train)
        meta_calib = TonIoTDataAdapter.extract_metadata(df_calib)
        meta_eval = TonIoTDataAdapter.extract_metadata(df_eval)

        labels_train = TonIoTDataAdapter.extract_labels(df_train)
        labels_calib = TonIoTDataAdapter.extract_labels(df_calib)
        labels_eval = TonIoTDataAdapter.extract_labels(df_eval)

        logger.info(
            f"Chronological split complete: "
            f"Train={len(X_train)} (100% normal), "
            f"Calib={len(X_calib)} (100% normal), "
            f"Eval={len(X_eval)} ({discrepancy_report['evaluation']['attack_samples']} attacks)"
        )

        return TonIoTSplitResult(
            X_train=X_train,
            X_calib=X_calib,
            X_eval=X_eval,
            meta_train=meta_train,
            meta_calib=meta_calib,
            meta_eval=meta_eval,
            labels_train=labels_train,
            labels_calib=labels_calib,
            labels_eval=labels_eval,
            discrepancy_report=discrepancy_report,
        )
