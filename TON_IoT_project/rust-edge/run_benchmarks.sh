#!/usr/bin/env bash
set -e
export PATH="$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")"

echo "=== Running Cargo Bench for aatc-engine ==="
cargo bench -p aatc-engine --bench trust_update
