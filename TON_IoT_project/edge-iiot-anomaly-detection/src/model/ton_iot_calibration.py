"""Score normalization and benign calibration interfaces matching E1 protocol."""
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TonIoTScoreNormalizer:
    """Normalizes raw Isolation Forest scores to [0, 1] using training score bounds.

    Protocol Invariants:
    1. raw_score = -1.0 * IsolationForest.score_samples(X)
    2. s_min and s_max are derived STRICTLY from the benign Training partition.
    3. Normalized score A_t = clip((raw_score - s_min) / (s_max - s_min), 0.0, 1.0).
    4. Edge-IIoTset normalization constants are strictly forbidden.
    """

    def __init__(self, s_min: float | None = None, s_max: float | None = None):
        self.s_min_: float | None = s_min
        self.s_max_: float | None = s_max
        self.is_fitted_: bool = (s_min is not None and s_max is not None)

    def fit(self, raw_scores_train: np.ndarray) -> "TonIoTScoreNormalizer":
        """Learn empirical score boundaries from benign training raw scores."""
        if len(raw_scores_train) == 0:
            raise ValueError("Cannot fit score normalizer on empty raw scores array.")

        scores_arr = np.asarray(raw_scores_train, dtype=float)
        if not np.all(np.isfinite(scores_arr)):
            raise ValueError("Raw training anomaly scores contain non-finite values (NaN/Inf).")

        self.s_min_ = float(np.min(scores_arr))
        self.s_max_ = float(np.max(scores_arr))

        if self.s_max_ <= self.s_min_:
            logger.warning(
                f"Degenerate score bounds detected: s_min={self.s_min_}, s_max={self.s_max_}. "
                "Adding epsilon to prevent division by zero."
            )
            self.s_max_ = self.s_min_ + 1e-10

        self.is_fitted_ = True
        logger.info(f"TonIoTScoreNormalizer fitted: s_min={self.s_min_:.6f}, s_max={self.s_max_:.6f}")
        return self

    def normalize(self, raw_scores: np.ndarray) -> np.ndarray:
        """Apply min-max normalization clipped to [0, 1]."""
        if not self.is_fitted_ or self.s_min_ is None or self.s_max_ is None:
            raise RuntimeError("TonIoTScoreNormalizer must be fitted on benign TRAIN scores before normalizing.")

        scores_arr = np.asarray(raw_scores, dtype=float)
        normalized = (scores_arr - self.s_min_) / (self.s_max_ - self.s_min_)
        return np.clip(normalized, 0.0, 1.0)

    def to_dict(self) -> dict[str, float | None]:
        """Export normalization bounds for logging and audit persistence."""
        return {
            "s_min": self.s_min_,
            "s_max": self.s_max_,
        }


def compute_benign_anomaly_threshold(
    normalized_scores_calib: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """Calculate the anomaly decision threshold A_thr from benign calibration scores.

    Protocol Invariants:
    1. Uses strictly linear interpolation: numpy.percentile(..., method="linear").
    2. Computed ONLY on benign calibration data (100% attack-label-free).
    3. Establishes a nominal 1% calibration-tail operating point without claiming future guarantees.
    """
    scores_arr = np.asarray(normalized_scores_calib, dtype=float)
    if len(scores_arr) == 0:
        raise ValueError("Cannot calculate threshold on empty calibration scores.")
    if not np.all((scores_arr >= 0.0) & (scores_arr <= 1.0)):
        raise ValueError("Scores must be normalized within [0, 1] prior to thresholding.")

    threshold = float(np.percentile(scores_arr, percentile, method="linear"))
    return float(np.clip(threshold, 0.0, 1.0))


def compute_benign_gamma(
    ema_scores_calib: np.ndarray,
    multiplier: float = 3.05,
) -> tuple[float, float, float]:
    """Calculate the EMA accumulator energy activation threshold gamma.

    Protocol Invariants:
    1. gamma = mu_calibration + 3.05 * sigma_calibration.
    2. Controls recursive EMA temporal-evidence activation (E_t > gamma).
    3. Evaluated for benign accumulator dormancy (does NOT control point-anomaly FAR).
    4. Computed ONLY on benign calibration data.

    Returns:
        tuple of (gamma, mu_calib, sigma_calib)
    """
    ema_arr = np.asarray(ema_scores_calib, dtype=float)
    if len(ema_arr) == 0:
        raise ValueError("Cannot calculate gamma on empty EMA scores.")

    mu_calib = float(np.mean(ema_arr))
    sigma_calib = float(np.std(ema_arr))
    gamma = float(mu_calib + multiplier * sigma_calib)

    return gamma, mu_calib, sigma_calib


def simulate_ema_stream(
    normalized_scores: np.ndarray,
    beta: float = 0.40,
    initial_ema: float = 0.0,
) -> np.ndarray:
    """Stream normalized scores through recursive EMA filter: E_t = beta * A_t + (1 - beta) * E_{t-1}."""
    scores_arr = np.asarray(normalized_scores, dtype=float)
    n = len(scores_arr)
    ema_out = np.zeros(n, dtype=float)

    prev_e = initial_ema
    for t in range(n):
        curr_e = beta * scores_arr[t] + (1.0 - beta) * prev_e
        ema_out[t] = curr_e
        prev_e = curr_e

    return ema_out
