#!/usr/bin/env bash
# export_and_build.sh
# Run from WSL: bash /mnt/c/Users/Shreyas\ A/.../rust-edge/scripts/export_and_build.sh
set -e

VENV="/tmp/aatc-venv"
PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"

# ── Python venv ──────────────────────────────────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
  echo "[venv] Creating at $VENV ..."
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install -q skl2onnx onnx onnxruntime scikit-learn pandas numpy
else
  source "$VENV/bin/activate"
  echo "[venv] Activated: $(python --version)"
fi

# ── Export Pipeline ONNX ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " Step 1/2 — Export Pipeline ONNX"
echo "════════════════════════════════════════════════════"
cd "$PROJ"
python retrain_and_export.py

# ── Rust toolchain path ──────────────────────────────────────────────────────
export PATH="$HOME/.cargo/bin:$PATH"

# ── Rust build ───────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " Step 2/2 — cargo check (fast) + cargo build --release"
echo "════════════════════════════════════════════════════"
# .cargo/config.toml now handles OPENSSL_* vars — no prefix needed
cargo check 2>&1
cargo build --release 2>&1

echo ""
echo "✓ All done."
echo "  Binary : $PROJ/target/release/aatc-edge"
echo "  Model  : $PROJ/model_pipeline.onnx"
