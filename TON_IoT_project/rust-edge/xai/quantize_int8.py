#!/usr/bin/env python3
"""
quantize_int8.py — Phase 2: Weight-Quantized INT8 Autoencoder
=============================================================
Quantizes Linear weights (MatMul) to 8-bit integers while preserving
full Float32 precision for the elementwise squared error and reduction
attributions (avoiding underflow on small reconstruction errors).
"""

import sys, os, json, platform
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

if platform.system() == "Linux" and os.path.exists("/mnt/c"):
    XAI_DIR = Path("/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge/xai")
else:
    XAI_DIR = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_project\rust-edge\xai")

MODELS_DIR = XAI_DIR / "models"
MODEL_F32 = MODELS_DIR / "autoencoder_f32.onnx"
MODEL_INT8 = MODELS_DIR / "autoencoder_int8.onnx"
META_PATH = XAI_DIR / "xai_meta.json"

print("=" * 65)
print("AATC-BAD XAI Extension — Phase 2: INT8 Model Quantization")
print("=" * 65)

if not MODEL_F32.exists():
    sys.exit(f"ERROR: {MODEL_F32} does not exist. Run train_autoencoder.py first.")

with open(META_PATH, "r") as f:
    meta = json.load(f)

print("\nQuantizing linear layers to INT8...")
quantize_dynamic(
    model_input=str(MODEL_F32),
    model_output=str(MODEL_INT8),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=['MatMul']
)

f32_kb = MODEL_F32.stat().st_size / 1024
int8_kb = MODEL_INT8.stat().st_size / 1024

print(f"\nModel Size:")
print(f"      Float32 model : {f32_kb:.1f} KB")
print(f"      INT8 model    : {int8_kb:.1f} KB")
print(f"      Reduction     : {(1.0 - int8_kb / f32_kb) * 100:.1f}%")

meta["onnx_int8"] = str(MODEL_INT8)
meta["quantization"] = {
    "format": "INT8_Dynamic_Linear",
    "f32_size_kb": f32_kb,
    "int8_size_kb": int8_kb,
    "calibrated_mse_threshold": 0.0040,
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Updated metadata in {META_PATH}")
print("Phase 2 Complete: INT8 model ready.")
