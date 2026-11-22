#!/usr/bin/env bash
set -e
VENV="/home/shreyas/aatc-venv"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating persistent venv at $VENV..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -q --upgrade pip
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q onnx onnxruntime scikit-learn pandas numpy shap onnxscript
else
    source "$VENV/bin/activate"
fi

echo "Persistent Venv ready at $VENV"
