# ToN-IoT Experimental Evaluation Archive

**Branch Status:** `CLOSED` (as of 2026-09-29)  
**Machine-Readable Status:** [`BRANCH_STATUS.json`](BRANCH_STATUS.json)  
**Formal Closure Document:** [`../../docs/TONIOT_NETWORK_BRANCH_CLOSURE.md`](../../docs/TONIOT_NETWORK_BRANCH_CLOSURE.md)

---

## 1. Branch Closure Summary

The **ToN-IoT Network Modality Evaluation Branch** (`ton_iot_network`) has concluded and is **officially closed**.

Across Phases 2A.3, 2B, 2C, and the authoritative Phase 2C Correction rerun, empirical evaluation demonstrated that:
- The 8-feature leakage-controlled network flow representation (`duration`, `src_bytes`, `dst_bytes`, `src_pkts`, `dst_pkts`, `src_ip_bytes`, `dst_ip_bytes`, `missed_bytes`) lacks sufficient discriminative signal when evaluated under unsupervised clean-corpus novelty detection against the scanning-dominated evaluation stream.
- A single homogeneous attack vector—consisting of 482,518 zero-payload single-packet TCP SYN probes (60.38% of all evaluation records, 100% attack prevalence)—forms an inlier-like cluster in feature space that all tested detector families score as sub-threshold.
- Point-wise OneClassSVM achieved ROC-AUC = 0.8784, but operational recall at the benign-calibrated threshold was only 0.0082% (99.99% of attacks remained sub-threshold).
- Causal temporal aggregation (rolling mean, rolling max, exceedance count, persistence, EWMA) failed to rescue detection: 99.81% to 100.00% of dense-probe flows remained sub-threshold across all 16 valid temporal mechanisms, with temporal smoothing actively degrading discriminative ranking.

**Core Directives for Future Research:**
1. **No TrustEngine Integration:** Under no circumstances should the anomaly score streams from this network evaluation branch be fed into the BAZT `TrustEngine`. An uninformative anomaly input cannot drive valid dynamic trust decay.
2. **No Post-Hoc Tuning:** No further detector family exploration, threshold adjustment, or temporal window tuning should be attempted within this 8-feature representation.
3. **Requirement for New Studies:** Any future investigation utilizing the ToN-IoT network dataset must formulate a materially distinct representation (e.g., entity-level connection rate accumulators, bidirectional TCP state tracking, protocol payload parsing) or an alternative research question. Any such study must be chartered as an entirely **NEW** experiment, and must not be treated as a continuation or modification of Phases 2A through 2C.

---

## 2. Manifest of Preserved Historical Artifacts

All experimental directories are preserved as immutable scientific records. **Do not overwrite, recompute, or modify existing JSON artifacts.**

| Directory | Experiment Phase | Status | Key Report / Outputs |
|---|---|---|---|
| [`phase2a_corrected/`](phase2a_corrected/) | Phase 2A: Modbus Modality Suitability Audit (Corrected Counts) | Historical | `docs/TONIOT_PHASE2A_CORRECTED_DETECTOR_SUITABILITY.md` |
| [`phase2a1_temporal_representation/`](phase2a1_temporal_representation/) | Phase 2A.1: Modbus Temporal Representation Audit | Historical | `docs/TONIOT_PHASE2A1_TEMPORAL_REPRESENTATION_AUDIT.md` |
| [`phase2a2_modality_audit/`](phase2a2_modality_audit/) | Phase 2A.2: IoT Weather Modality Audit | Historical | `docs/TONIOT_PHASE2A2_MODALITY_SUITABILITY_AUDIT.md` |
| [`phase2a3_network_modality_audit/`](phase2a3_network_modality_audit/) | Phase 2A.3: Network Modality Leakage & Suitability Audit | Historical | `docs/TONIOT_PHASE2A3_NETWORK_MODALITY_AUDIT.md` |
| [`phase2b_detector_family_audit/`](phase2b_detector_family_audit/) | Phase 2B: Unsupervised Detector-Family Audit | Historical | `docs/TONIOT_PHASE2B_DETECTOR_FAMILY_AUDIT.md` |
| [`phase2c_temporal_evidence_audit/`](phase2c_temporal_evidence_audit/) | Phase 2C: Initial Temporal Feasibility Audit | Superceded | Quantitative evaluation invalidated by label parser bug |
| [`phase2c_temporal_evidence_audit_corrected/`](phase2c_temporal_evidence_audit_corrected/) | Phase 2C: Corrected Temporal Feasibility Audit | **Authoritative** | `docs/TONIOT_PHASE2C_TEMPORAL_EVIDENCE_AUDIT_CORRECTED.md` |
