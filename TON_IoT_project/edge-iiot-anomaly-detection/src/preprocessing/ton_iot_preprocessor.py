"""Preprocessing pipeline for ToN-IoT Modbus telemetry matching E1 protocol."""
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.ton_iot_adapter import TON_IOT_FEATURE_COLUMNS, TON_IOT_LABEL_COLUMNS, TON_IOT_METADATA_COLUMNS

logger = logging.getLogger(__name__)


class TonIoTPreprocessor:
    """Dedicated preprocessor for ToN-IoT SCADA Modbus 16-bit register telemetry.

    Protocol Invariants:
    1. RobustScaler is fitted STRICTLY on the training partition (benign only).
    2. Calibration and Evaluation partitions are transformed using the frozen TRAIN scaler parameters.
    3. Ground truth labels (label, type) and metadata (date, time, timestamp) are strictly forbidden.
    4. Deterministic feature ordering is strictly enforced.
    5. Zero Edge-IIoT feature drops or imputation rules are applied.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names_in_: list[str] = list(TON_IOT_FEATURE_COLUMNS)
        self.is_fitted_: bool = False
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _validate_input_columns(self, X: pd.DataFrame, context: str) -> None:
        """Enforce strict column isolation: forbid labels/metadata and verify feature presence."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{context} input must be a pandas DataFrame, got {type(X)}")

        # Check for forbidden label columns
        forbidden_present = [col for col in TON_IOT_LABEL_COLUMNS if col in X.columns]
        if forbidden_present:
            raise ValueError(
                f"Protocol violation! Ground truth columns {forbidden_present} found in {context} features. "
                "Labels must never be exposed to the preprocessor or detector."
            )

        # Check for forbidden temporal metadata columns
        meta_present = [col for col in TON_IOT_METADATA_COLUMNS if col in X.columns]
        if meta_present:
            raise ValueError(
                f"Protocol violation! Metadata columns {meta_present} found in {context} features. "
                "Timestamps must not be passed into the numeric scaler."
            )

        # Check that all expected Modbus register features are present
        missing_feats = [col for col in self.feature_names_in_ if col not in X.columns]
        if missing_feats:
            raise KeyError(f"{context} DataFrame is missing required Modbus features: {missing_feats}")

    def fit(self, X_train: pd.DataFrame) -> "TonIoTPreprocessor":
        """Fit RobustScaler strictly on the training partition."""
        logger.info(f"Fitting TonIoTPreprocessor (RobustScaler) on {len(X_train)} samples...")
        self._validate_input_columns(X_train, "fit(X_train)")

        # Enforce exact frozen column order
        X_ordered = X_train[self.feature_names_in_].copy()

        # Fit RobustScaler
        self.scaler.fit(X_ordered)
        self.center_ = self.scaler.center_.copy()
        self.scale_ = self.scaler.scale_.copy()
        self.is_fitted_ = True

        logger.info("TonIoTPreprocessor successfully fitted on training partition.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the frozen training scaler parameters."""
        if not self.is_fitted_:
            raise RuntimeError("TonIoTPreprocessor must be fitted on TRAIN before calling transform().")

        self._validate_input_columns(X, "transform(X)")

        # Enforce exact frozen column order
        X_ordered = X[self.feature_names_in_].copy()

        scaled_values = self.scaler.transform(X_ordered)
        X_transformed = pd.DataFrame(scaled_values, columns=self.feature_names_in_, index=X_ordered.index)
        return X_transformed

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit strictly on X_train and transform it."""
        self.fit(X_train)
        return self.transform(X_train)

    def get_scaler_params(self) -> dict[str, Any]:
        """Return learned median and IQR parameters for auditing and reproducibility."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor is not fitted.")
        return {
            "feature_names": self.feature_names_in_,
            "medians": self.center_.tolist(),
            "iqrs": self.scale_.tolist(),
        }

    def save(self, filepath: Path | str) -> None:
        """Serialize preprocessor artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved TonIoTPreprocessor artifact to {path}")

    @classmethod
    def load(cls, filepath: Path | str) -> "TonIoTPreprocessor":
        """Load serialized preprocessor artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {path}")
        return joblib.load(path)
