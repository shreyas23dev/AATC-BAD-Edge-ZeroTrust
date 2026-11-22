#!/usr/bin/env bash
set -e
export PATH="$HOME/.cargo/bin:$PATH"
PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"

echo "=== cargo build --release ==="
cd "$PROJ"
cargo build --release 2>&1

echo ""
echo "=== Running replay (17,459 rows) ==="
bash "$PROJ/scripts/replay.sh"
