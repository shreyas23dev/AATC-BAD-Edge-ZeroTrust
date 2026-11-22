#!/usr/bin/env bash
set -e

VENV="/tmp/aatc-venv"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[setup] Creating venv at $VENV..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

echo "[setup] Installing PyTorch (CPU), ONNX, and SHAP..."
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q onnx onnxruntime scikit-learn pandas numpy shap

echo "[setup] Verifying imports..."
python3 -c "
import torch, onnx, onnxruntime, sklearn, pandas, numpy, shap
print('PyTorch   :', torch.__version__)
print('ONNX      :', onnx.__version__)
print('ORT       :', onnxruntime.__version__)
print('SHAP      :', shap.__version__)
print('Setup Successful!')
"
