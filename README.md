# Resilient and Explainable Zero-Trust Architecture for Industrial IoT

[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org/)
[![Benchmarks](https://img.shields.io/badge/criterion.rs-76.50_ns-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Special Issue](https://img.shields.io/badge/Springer_Nature-Information_Systems_Frontiers-purple.svg)](https://www.springer.com/journal/10796)

Official repository for the manuscript:  
**"Resilient and Explainable Zero-Trust Architecture for Industrial IoT: Mitigating Strategic Moral Hazard via Authorization Debt and In-Flight Attribution"**  
Submitted to *Information Systems Frontiers* (Springer Nature), Special Issue on *"Advancing towards a Responsible and Resilient Digital Future"* (Collection Code: `ffhgdbhbad`).

---

## 📌 Executive Summary

Continuous Zero-Trust Architecture (ZTA) in Cyber-Physical Systems (CPS) relies on Continuous Dynamic Trust Models (CDTM) to authenticate edge devices. However, standard CDTMs harbor an exploitable **strategic moral hazard**: unconditional trust rehabilitation allows pacing adversaries to interleave borderline sub-threshold attack bursts with nominal rest intervals, harvesting elevated permissions indefinitely while pristine traffic purges accumulated suspicion.

**AATC-BAD** (**A**daptive **A**symptotic **T**rust **C**eilings with **B**orderline **A**uthorization **D**ebt) resolves this exploit at the policy layer:
1. **Authorization Debt ($D_t$):** Accrues non-linearly whenever borderline anomaly scores ($A_t \in [q_{\mathrm{low}}, A_{\mathrm{thr}})$) coincide with full access ($\mathtt{ALLOW}$).
2. **Asymptotic Recovery Ceiling ($C_t$):** Dynamically ratchets downward once debt exceeds an allowance buffer ($K_{\mathrm{allow}}$), permanently quenching evasive cycles without heuristic mode switching.
3. **Availability Parity Theorem:** Compliant devices experience **$0.00\,\text{pp}$** marginal availability penalty relative to baseline CDTM ($\Delta \mathtt{ALLOW} = 0.00\,\text{pp}$, $\Delta \mathtt{BLOCK} = 0.00\,\text{pp}$).
4. **Zero-Sampling INT8 Explainability ($\mathbf{\Phi}_t$):** An embedded deep autoencoder derives exact local feature attributions in **$115\,\mu\text{s}$** ($8{,}687\,\text{fps}$), achieving a **$102\times$ speedup** over KernelSHAP to satisfy NIST AI RMF and IEC 62443 auditability mandates.

---

## 🔬 Key Empirical Results Across 46,011 Streaming Frames

| Evaluation Metric | Edge-IIoTset Testbed ($20{,}000$ Frames) | UNB CIC-IoT-2023 Generalization ($26{,}011$ Frames) |
|---|:---:|:---:|
| **Benign Stream Size** | $10{,}000$ frames | $10{,}000$ frames |
| **Benign $\mathtt{ALLOW}$ Rate ($B_0$ vs. AATC-BAD)** | $87.67\%$ vs. $87.67\%$ ($\mathbf{\Delta = 0.00\,\text{pp}}$) | $87.51\%$ vs. $87.51\%$ ($\mathbf{\Delta = 0.00\,\text{pp}}$) |
| **Benign Pure $\mathtt{BLOCK}$ Rate (FPR)** | $10.90\%$ vs. $10.90\%$ ($\mathbf{\Delta = 0.00\,\text{pp}}$) | $11.27\%$ vs. $11.27\%$ ($\mathbf{\Delta = 0.00\,\text{pp}}$) |
| **Aggregate Threat Mitigation** | **$98.55\%$** ($86.04\%$ Pure $\mathtt{BLOCK}$) | **$66.75\%$** ($66.54\%$ Pure $\mathtt{BLOCK}$) |
| **Detectable High-Impact Threats** | $100.00\%$ (UDP, ICMP), $99.85\%$–$99.95\%$ | **$99.92\%$** ($100\%$ SYN/SlowLoris, $99.95\%$ ICMP/PortScan) |
| **Mirai Botnet Mitigation** | N/A | **$34.42\%$** ($33.22\%$ Pure $\mathtt{BLOCK}$) |
| **Monte Carlo Yield Reduction ($\bar{M}_{\mathrm{cum}}$)** | $\mathbf{9.26\% \text{ to } 20.84\%}$ ($p < 0.001$, Wilcoxon) | $\mathbf{1.09\% \text{ to } 12.34\%}$ ($p < 0.001$, Wilcoxon) |
| **State Update Latency (Criterion.rs)** | **$76.50\,\text{ns}$** ($13.07\,\text{M}$ ops/s) | **$76.50\,\text{ns}$** |
| **$1{,}000$ Concurrent Devices Batch** | **$79.34\,\mu\text{s}$** ($79.3\,\text{ns}$/device) | **$79.34\,\mu\text{s}$** |

---

## 📂 Repository Structure

```
AATC-BAD-Edge-ZeroTrust/
├── TON_IoT_project/
│   ├── rust-edge/                  # Native Rust edge engine (L1-cache optimized, <80ns latency)
│   │   ├── crates/                 # Core trust kinetics, policy state machines, and streaming runner
│   │   ├── benches/                # Criterion.rs micro-benchmarks & summary estimates
│   │   ├── models/                 # Quantized INT8 autoencoder models (ONNX/Tract)
│   │   └── xai/                    # Attribution fingerprints, SHAP benchmarks & 100-seed Monte Carlo
│   ├── dynamic-trust-engine/       # Baseline CDTM & AATC-BAD reference implementations
│   │   ├── src/                    # Trust evaluation kinetics, state machine, and REST API
│   │   └── config/                 # Trust parameters and simulation configurations
│   ├── edge-iiot-anomaly-detection/# ML anomaly detection pipelines & feature extraction
│   ├── zero-trust-engine/          # Policy decision point (PDP) and ZTA constraint validators
│   ├── experiments/candidate5/     # Complete evaluation pipeline
│   │   ├── phase8_failure_diagnosis/          # Empirical demonstration of CDTM moral hazard
│   │   ├── phase9_countermeasure/             # AATC-BAD parameter calibration
│   │   ├── phase10_countermeasure_evaluation/ # Comparative replay evaluation across 46k frames
│   │   ├── phase11_envelope_analysis/         # 20-configuration sensitivity grid
│   │   └── adaptive_adversary/                # Closed-loop adaptive adversary evaluation (Table 8)
│   └── data/                       # Dataset schema & preparation scripts for Edge-IIoTset & CIC-IoT-2023
├── .gitignore                      # Excludes raw datasets, build caches, and local drafts
├── LICENSE                         # MIT License
└── README.md                       # Repository overview and guide
```

> **Note on Directory Naming & Intermediate Outputs:**  
> - The module directory `TON_IoT_project/` is retained as the project namespace from early exploratory architectural baselines. As detailed in the manuscript, all formal evaluations, streaming replays, and benchmarks presented in this paper were conducted on the realistic **Edge-IIoTset** ($20{,}000$ frames) and **UNB CIC-IoT-2023** ($26{,}011$ frames) testbeds.  
> - The `.tex` files located under `TON_IoT_project/experiments/candidate5/tables/` (`table1_*.tex`–`table8_*.tex`) are intermediate raw outputs exported automatically by the experimental evaluation scripts; their numbering follows the internal pipeline execution order rather than the published manuscript table numbering.

---

## 🚀 Quickstart & Reproducibility Guide

### 1. Prerequisites
- **Rust Toolchain:** Stable 1.75+ (`rustup default stable`)
- **Python:** 3.10+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`

### 2. Building and Benchmarking the Rust Edge Engine
```bash
cd TON_IoT_project/rust-edge

# Run unit and integration tests
cargo test --release

# Run Criterion.rs micro-benchmarks (isolated kernel latencies)
cargo bench
```

### 3. Running Monte Carlo Stochastic Benchmarks (100 Seeds)
```bash
cd TON_IoT_project/rust-edge/xai
python benchmark_monte_carlo_pulsing.py
```

### 4. Running Closed-Loop Adaptive Adversary Simulations
```bash
cd TON_IoT_project/experiments/candidate5/adaptive_adversary
python run_adaptive_adversary.py
```

---

## 📜 Authors & Contact

* **Shreyas A** (*Corresponding Author*)  
  Department of Computer Science & Engineering, Sai Vidya Institute of Technology, Bengaluru, India  
  Email: `shreyasvasista05@gmail.com`
* **Dr. Arun Kumar B R**  
  Department of Computer Science & Engineering, BMS Institute of Technology & Management, Bengaluru, India  
  Email: `arunkumarbr@bmsit.in`

---

## 📄 License
This project is open-sourced under the [MIT License](LICENSE).
