# AATC-BAD Edge Runtime

> **Single-binary, zero-Python IoT edge IDS** implementing the AATC-BAD
> countermeasure from *"When Trust Recovery Creates Moral Hazard: Strategic
> Evasion and Policy-Layer Defense in Zero-Trust IoT"* (InCIS 2027) and its
> journal extension featuring **Zero-Sampling Single-Pass INT8 Autoencoder XAI**.

## Architecture

```
Training & Quantization Plane (Python, offline)
  scikit-learn IsolationForest → export_onnx.py → model_pipeline.onnx (InCIS 2027)
  PyTorch Symmetric AE (88-64-32-16-32-64-88) → ONNX → quantize_int8.py → autoencoder_int8.onnx (Journal)
                                                          │
                                                          ▼
Edge Runtime (Rust, single binary)
  CSV / live network telemetry
       │
       ▼
  crates/aatc-edge
  ├── inference.rs        Isolation Forest anomaly scorer (InCIS 2027)
  ├── xai_inference.rs    AeXaiModel: Single-Pass INT8 Autoencoder XAI (Journal Extension)
  ├── window.rs           16-frame sliding-window ring buffer (88-dim mean & std)
  ├── replay.rs           Streaming evaluation with audit tuple R_t = (A_t, alpha_t, D_t, C_t, d_t, Phi_t)
  └── bench.rs            Throughput benchmark
  
  crates/aatc-engine      (pure library, zero heap-allocation hot path)
  ├── config.rs           Frozen paper parameters (Table 1)
  ├── engine.rs           AATC-BAD state machine (D_t, C_t update loop)
  ├── policy.rs           ALLOW / VERIFY / BLOCK decision type
  └── telemetry.rs        TelemetryFrame input type
```

---

## Performance Summary (Empirical Micro-benchmarks)

| Component | Operation | Latency | Throughput |
|---|---|---|---|
| **Sliding Window** | Ring Buffer `push_and_evict` (44 floats) | **12.3 ns** | 81.4 M ops/sec |
| **Feature Extraction** | Window Aggregation ($16 \times 44 \to 88$, ddof=0) | **371.9 ns** | 2.69 M ops/sec |
| **Trust Engine** | Trust state transition ($\alpha_t, D_t, C_t, d_t$) | **53.3 ns** | 18.76 M ops/sec |
| **Combined Non-ML** | Total Edge Gateway Pre- & Post-Processing | **437.4 ns** | **2.29 M frames/sec** |
| **INT8 Autoencoder** | Single-pass Anomaly $A_t$ + Feature Attribution $\Phi_t$ | **0.056 ms** | **17,933.5 fps** |

---

## Quick Start

### 1. Build the Release Binary

```bash
cd rust-edge
cargo build --release
# Executable at: target/release/aatc-edge
```

### 2. Run Replay Modes

#### Mode A: Single-Pass INT8 Autoencoder XAI (Journal Extension)
Processes telemetry frames, computes 16-frame rolling means and standard deviations, extracts zero-sampling local feature attributions $\Phi_t$, and records full decision audit logs:

```bash
./target/release/aatc-edge replay \
    --csv multi_attack_eval.csv \
    --model xai/models/autoencoder_int8.onnx \
    --meta xai/xai_meta.json \
    --xai \
    --output decisions_multi_attack.jsonl
```

#### Mode B: Isolation Forest Baseline (InCIS 2027)
```bash
./target/release/aatc-edge replay \
    --csv eval_set.csv \
    --model model_pipeline.onnx \
    --output decisions_eval.jsonl
```

### 3. Run Criterion Nanosecond Micro-benchmarks

```bash
cargo bench -p aatc-engine --bench trust_update
```

### 4. Regenerate Explainability Fingerprint Figures

```bash
python3 xai/plot_fingerprints.py
# Outputs:
#   xai/xai_attribution_fingerprints.png (300 DPI)
#   xai/xai_attribution_fingerprints.pdf (Vector for LaTeX)
```

---

## Frozen System Parameters (InCIS 2027 Table 1)

| Parameter | Value | Description |
|---|---|---|
| `beta` | 0.40 | Trust decay rate under anomaly |
| `gamma` | 0.335 | Trust recovery rate under healthy telemetry |
| `theta` | 1.00 | Exponential trust decay steepness |
| `lambda` | 0.55 | Moving average smoothing weight |
| `delta` | 0.05 | Clean-observation anomaly tolerance |
| `alpha_0` | 0.2057| Cold-start initialization trust |
| `t_allow` | 0.80 | ALLOW policy boundary threshold |
| `t_quar` | 0.30 | QUARANTINE / BLOCK boundary threshold |
| `delta_d` | 0.10 | Authorization debt increment penalty |
| `k_allow` | 1.70 | Cumulative debt ceiling threshold |
| `eta_c` | 0.30 | Recovery ceiling linear decay slope |
| `c_floor` | 0.70 | Minimum ceiling floor ($C_{\min}$) |

---

## Key Artifacts & Results

- **Figure 5 (Attribution Fingerprints):** `xai/xai_attribution_fingerprints.png`
- **Table 5 (Latency & Model Footprint):** `xai/benchmark_results.json`
- **Table 6 (Cross-Threat Robustness):** `xai/cross_attack_results.json`
- **Table 7 (Micro-architectural Profiling):** Target criterion statistical logs in `results.md`
