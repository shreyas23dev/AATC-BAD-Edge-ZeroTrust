#!/usr/bin/env python3
"""
export_onnx.py — Export the trained Isolation Forest to ONNX format.

Usage:
    python export_onnx.py \
        --model experiments/candidate5/models/isolation_forest.pkl \
        --output rust-edge/model.onnx \
        --n-features 61

Requirements:
    pip install scikit-learn skl2onnx onnx
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def export(model_path: Path, output_path: Path, n_features: int) -> None:
    # ── Load trained model ───────────────────────────────────────────────────
    print(f"[1/4] Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Verify it's an Isolation Forest
    model_type = type(model).__name__
    print(f"      Model type : {model_type}")
    if "IsolationForest" not in model_type:
        print(f"WARNING: Expected IsolationForest, got {model_type}")

    # ── Convert to ONNX ──────────────────────────────────────────────────────
    print(f"[2/4] Converting to ONNX (n_features={n_features})")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        sys.exit("ERROR: skl2onnx not installed. Run: pip install skl2onnx")

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        # Request score output (not just label)
        options={type(model): {"score_samples": True}},
    )

    # ── Validate ─────────────────────────────────────────────────────────────
    print("[3/4] Validating ONNX model")
    try:
        import onnx
        onnx.checker.check_model(onnx_model)
        print("      ONNX check: PASSED")
    except ImportError:
        print("      (onnx not installed — skipping validation)")

    # Quick numerical sanity check
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(onnx_model.SerializeToString())
        dummy = np.zeros((1, n_features), dtype=np.float32)
        out = sess.run(None, {"float_input": dummy})
        print(f"      Dummy inference output shapes: {[o.shape for o in out]}")
    except ImportError:
        print("      (onnxruntime not installed — skipping inference check)")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] Saving to {output_path}")
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = output_path.stat().st_size / 1024
    print(f"\n✓ Export complete — {size_kb:.1f} KB")
    print(f"  Pass to aatc-edge binary: --model {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Isolation Forest → ONNX")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("experiments/candidate5/models/isolation_forest.pkl"),
        help="Path to pickled sklearn IsolationForest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rust-edge/model.onnx"),
        help="Output .onnx path",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=61,
        help="Feature vector dimension (must match training)",
    )
    args = parser.parse_args()
    export(args.model, args.output, args.n_features)
