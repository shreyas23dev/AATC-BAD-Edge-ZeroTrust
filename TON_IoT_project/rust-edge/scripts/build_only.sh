#!/usr/bin/env bash
set -e
export PATH="$HOME/.cargo/bin:$PATH"
PROJ="/mnt/c/Users/Shreyas A/PROJECTS/BAZT-TON_IoT/TON_IoT_project/rust-edge"
cd "$PROJ"
echo "=== cargo check ==="
cargo check 2>&1
echo "=== cargo build --release ==="
cargo build --release 2>&1
echo "Done. Binary: $PROJ/target/release/aatc-edge"
