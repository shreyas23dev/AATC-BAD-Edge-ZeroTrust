#!/usr/bin/env python3
"""
Adaptive Adversary Closed-Loop Evaluation for AATC-BAD
======================================================
Evaluates four closed-loop adaptive adversary policies against Baseline CDTM
and the proposed AATC-BAD framework over T = 1,000 transaction epochs.

Policies Evaluated:
-------------------
1. Greedy Reactive Pacer:
   Attacks at borderline amplitude (A_t = 0.450) whenever granted ALLOW.
   Upon demotion (VERIFY or BLOCK), pauses transmission and emits clean telemetry
   (A_t = 0.050) until ALLOW is restored.

2. Multi-Burst Pacer (m = 2):
   Transmits 2 consecutive borderline pulses (A_t = 0.450) per opportunity,
   then rests until access is restored.

3. Adaptive Cooldown Backoff (tau_r Learner):
   Starts at tau_r = 4. Increments rest duration by +2 epochs whenever demoted,
   searching for the debt-neutral boundary Delta D_net <= 0.

4. Online Amplitude Throttling (A_t Learner):
   Holds rest fixed (tau_r = 6) and attenuates attack perturbation amplitude
   downward by Delta A = 0.02 upon demotion, attempting to evade debt accumulation.

Additionally performs a systematic recovery-sweep over tau_r in [5, 69] with
10-frame attack bursts over 600 steps to evaluate debt ceiling dynamics.

Dependencies: Python 3 standard library + numpy.
"""

import json
import os
from pathlib import Path
import numpy as np

# =============================================================================
# Authoritative Paper Parameters (Frozen)
# =============================================================================
A_THR   = 0.4554
BETA    = 0.40
GAMMA   = 0.335
THETA   = 1.00
LAMBDA_ = 0.55
DELTA   = 0.05
P_PEN   = 0.25
ALPHA_0 = 0.2057
T_ALLOW = 0.80
T_QUAR  = 0.30

# AATC-BAD Parameters
Q_LOW   = 0.35
DENOM   = A_THR - Q_LOW  # 0.1054
DELTA_D = 0.10
K_ALLOW = 1.70
ETA_C   = 0.30
C_FLOOR = 0.70

# =============================================================================
# State Transition Functions
# =============================================================================
def step_aatc(a: float, state: dict) -> dict:
    """Exact AATC-BAD policy transition (Equations 1-4, 15-18)."""
    T, E, SB, D, C = state["T"], state["E"], state["SB"], state["D"], state["C"]

    # Stage 1: Debt increment charged only under prior ALLOW and borderline score
    if (T >= T_ALLOW) and (Q_LOW <= a < A_THR):
        delta_D = (a - Q_LOW) / DENOM
    else:
        delta_D = 0.0

    # Stage 2: Debt update
    if a < Q_LOW:
        D_next = max(0.0, D - DELTA_D)
    else:
        D_next = D + delta_D

    # Stage 3: Dynamic ceiling update
    excess_debt = max(0.0, D_next - K_ALLOW)
    C_next = max(C_FLOOR, 1.0 - ETA_C * excess_debt)

    # Stage 4: Deviation smoothing and slow-burn accumulator
    E_next = BETA * a + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_next = SB + LAMBDA_
    else:
        SB_next = max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0

    # Stage 5: Dynamic trust state transition capped by C_next
    if a >= A_THR:
        T_next = T * (1.0 - a)
    elif I_next == 0:
        T_next = min(C_next, T + ALPHA_0 * (C_next - T))
    else:
        T_next = max(0.0, T - P_PEN * (a / A_THR))

    # Stage 6: 3-tier policy decision
    if T_next >= T_ALLOW:
        pol = "ALLOW"
    elif T_next >= T_QUAR:
        pol = "VERIFY"
    else:
        pol = "BLOCK"

    return {
        "T": float(T_next),
        "E": float(E_next),
        "SB": float(SB_next),
        "D": float(D_next),
        "C": float(C_next),
        "I": I_next,
        "policy": pol,
    }


def step_b0(a: float, state: dict) -> dict:
    """Baseline CDTM policy transition without authorization debt."""
    T, E, SB = state["T"], state["E"], state["SB"]

    E_next = BETA * a + (1.0 - BETA) * E
    if E_next > GAMMA:
        SB_next = SB + LAMBDA_
    else:
        SB_next = max(0.0, SB - DELTA)
    I_next = 1 if SB_next > THETA else 0

    if a >= A_THR:
        T_next = T * (1.0 - a)
    elif I_next == 0:
        T_next = T + ALPHA_0 * (1.0 - T)
    else:
        T_next = max(0.0, T - P_PEN * (a / A_THR))

    if T_next >= T_ALLOW:
        pol = "ALLOW"
    elif T_next >= T_QUAR:
        pol = "VERIFY"
    else:
        pol = "BLOCK"

    return {
        "T": float(T_next),
        "E": float(E_next),
        "SB": float(SB_next),
        "I": I_next,
        "policy": pol,
    }


# =============================================================================
# Closed-Loop Policy Runners
# =============================================================================
def run_greedy_reactive(step_fn, is_aatc: bool, t_max: int = 1000):
    state = {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0} if is_aatc else {"T": 1.0, "E": 0.0, "SB": 0.0}
    attempts, allows, verifies, blocks = 0, 0, 0, 0
    t_quench = None

    for t in range(1, t_max + 1):
        curr_allow = state["T"] >= T_ALLOW
        a = 0.450 if curr_allow else 0.050
        was_attack = curr_allow
        if was_attack:
            attempts += 1

        state = step_fn(a, state)

        if was_attack:
            if state["policy"] == "ALLOW":
                allows += 1
            elif state["policy"] == "VERIFY":
                verifies += 1
            else:
                blocks += 1

        if is_aatc and t_quench is None and state["C"] <= T_ALLOW:
            t_quench = t

    return {
        "strategy": "Greedy Reactive Pacer",
        "attempts": attempts,
        "allows": allows,
        "verifies": verifies,
        "blocks": blocks,
        "allow_rate_pct": round(allows / max(1, attempts) * 100, 2),
        "effective_duty_pct": round(attempts / t_max * 100, 2),
        "t_quench": t_quench,
    }


def run_multi_burst(step_fn, is_aatc: bool, m: int = 2, t_max: int = 1000):
    state = {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0} if is_aatc else {"T": 1.0, "E": 0.0, "SB": 0.0}
    attempts, allows, verifies, blocks = 0, 0, 0, 0
    burst_rem = 0
    t_quench = None

    for t in range(1, t_max + 1):
        curr_allow = state["T"] >= T_ALLOW
        if curr_allow and burst_rem == 0:
            burst_rem = m

        if burst_rem > 0:
            a = 0.450
            attempts += 1
            burst_rem -= 1
            was_attack = True
        else:
            a = 0.050
            was_attack = False

        state = step_fn(a, state)

        if was_attack:
            if state["policy"] == "ALLOW":
                allows += 1
            elif state["policy"] == "VERIFY":
                verifies += 1
            else:
                blocks += 1

        if is_aatc and t_quench is None and state["C"] <= T_ALLOW:
            t_quench = t

    return {
        "strategy": f"Multi-Burst Pacer (m={m})",
        "attempts": attempts,
        "allows": allows,
        "verifies": verifies,
        "blocks": blocks,
        "allow_rate_pct": round(allows / max(1, attempts) * 100, 2),
        "effective_duty_pct": round(attempts / t_max * 100, 2),
        "t_quench": t_quench,
    }


def run_cooldown_backoff(step_fn, is_aatc: bool, t_max: int = 1000):
    state = {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0} if is_aatc else {"T": 1.0, "E": 0.0, "SB": 0.0}
    tau_r = 4
    attempts, allows, verifies, blocks = 0, 0, 0, 0
    t = 0

    while t < t_max:
        # Attack frame
        a = 0.450
        attempts += 1
        t += 1
        state = step_fn(a, state)

        if state["policy"] == "ALLOW":
            allows += 1
        else:
            if state["policy"] == "VERIFY":
                verifies += 1
            else:
                blocks += 1
            tau_r += 2

        # Cooldown rest frames
        for _ in range(tau_r):
            if t >= t_max:
                break
            t += 1
            state = step_fn(0.050, state)

    return {
        "strategy": "Adaptive Cooldown Backoff",
        "attempts": attempts,
        "allows": allows,
        "verifies": verifies,
        "blocks": blocks,
        "allow_rate_pct": round(allows / max(1, attempts) * 100, 2),
        "effective_duty_pct": round(attempts / t_max * 100, 2),
        "final_tau_r": tau_r,
    }


def run_amplitude_throttling(step_fn, is_aatc: bool, tau_r: int = 6, t_max: int = 1000):
    state = {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0} if is_aatc else {"T": 1.0, "E": 0.0, "SB": 0.0}
    a_val = 0.450
    attempts, allows, verifies, blocks = 0, 0, 0, 0
    t = 0

    while t < t_max:
        attempts += 1
        t += 1
        state = step_fn(a_val, state)

        if state["policy"] == "ALLOW":
            allows += 1
        else:
            if state["policy"] == "VERIFY":
                verifies += 1
            else:
                blocks += 1
            a_val = max(0.340, a_val - 0.02)

        for _ in range(tau_r):
            if t >= t_max:
                break
            t += 1
            state = step_fn(0.050, state)

    return {
        "strategy": "Online Amplitude Throttling",
        "attempts": attempts,
        "allows": allows,
        "verifies": verifies,
        "blocks": blocks,
        "allow_rate_pct": round(allows / max(1, attempts) * 100, 2),
        "effective_duty_pct": round(attempts / t_max * 100, 2),
        "final_amplitude": round(a_val, 3),
    }


# =============================================================================
# Systematic Recovery Sweep over tau_r in [5, 69]
# =============================================================================
def run_recovery_sweep(tau_r_list=[5, 6, 8, 10, 15, 20, 69], n_steps=600):
    results = []
    for tau_r in tau_r_list:
        state = {"T": 1.0, "E": 0.0, "SB": 0.0, "D": 0.0, "C": 1.0}
        max_D = 0.0
        exceeds_K = False
        atk_attempts = 0
        atk_allows = 0

        step = 0
        while step < n_steps:
            # 1 attack frame at borderline score 0.450
            step += 1
            atk_attempts += 1
            state = step_aatc(0.450, state)
            max_D = max(max_D, state["D"])
            if state["D"] > K_ALLOW:
                exceeds_K = True
            if state["policy"] == "ALLOW":
                atk_allows += 1

            # tau_r rest frames at nominal 0.050
            for _ in range(tau_r):
                if step >= n_steps:
                    break
                step += 1
                state = step_aatc(0.050, state)
                max_D = max(max_D, state["D"])
                if state["D"] > K_ALLOW:
                    exceeds_K = True

        allow_pct = round(atk_allows / max(1, atk_attempts) * 100, 2)
        results.append({
            "tau_r": tau_r,
            "duty_cycle_pct": round(1.0 / (1.0 + tau_r) * 100, 2),
            "max_debt": round(max_D, 4),
            "exceeds_K_allow": exceeds_K,
            "attack_attempts": atk_attempts,
            "attack_allows": atk_allows,
            "attack_allow_pct": allow_pct,
        })
    return results


# =============================================================================
# Main Execution and Artifact Generation
# =============================================================================
def main():
    print("=" * 80)
    print(" Closed-Loop Adaptive Adversary Benchmark against CDTM and AATC-BAD")
    print("=" * 80)

    # 1. Evaluate Four Adaptive Policies
    policies = [
        ("Greedy Reactive Pacer", run_greedy_reactive),
        ("Multi-Burst Pacer (m=2)", run_multi_burst),
        ("Adaptive Cooldown Backoff", run_cooldown_backoff),
        ("Online Amplitude Throttling", run_amplitude_throttling),
    ]

    comparative_results = []
    for name, runner in policies:
        b0_res = runner(step_b0, is_aatc=False)
        aatc_res = runner(step_aatc, is_aatc=True)
        comparative_results.append({
            "strategy": name,
            "baseline_cdtm": b0_res,
            "aatc_bad": aatc_res,
        })
        print(f"\n[Strategy] {name}")
        print(f"  Baseline CDTM : Attempts={b0_res['attempts']:3d}, ALLOW={b0_res['allows']:3d} ({b0_res['allow_rate_pct']}%), Duty={b0_res['effective_duty_pct']}%, Verify/Block={b0_res['verifies']}/{b0_res['blocks']}")
        print(f"  AATC-BAD      : Attempts={aatc_res['attempts']:3d}, ALLOW={aatc_res['allows']:3d} ({aatc_res['allow_rate_pct']}%), Duty={aatc_res['effective_duty_pct']}%, Verify/Block={aatc_res['verifies']}/{aatc_res['blocks']}")

    # 2. Recovery Sweep
    print("\n" + "=" * 80)
    print(" Recovery Sweep over tau_r in [5, 69] (600 steps, single-pulse)")
    print("=" * 80)
    sweep_results = run_recovery_sweep()
    for row in sweep_results:
        print(f"  tau_r={row['tau_r']:2d} | Duty={row['duty_cycle_pct']:5.2f}% | Max D={row['max_debt']:.4f} | Exceeds K={str(row['exceeds_K_allow']):5s} | ALLOW={row['attack_allows']}/{row['attack_attempts']} ({row['attack_allow_pct']}%)")

    # 3. Export JSON Results
    out_dir = Path("TON_IoT_project/experiments/candidate5/adaptive_adversary")
    out_dir.mkdir(parents=True, exist_ok=True)

    export_data = {
        "title": "Closed-Loop Adaptive Adversary Benchmark",
        "parameters": {
            "A_thr": A_THR, "beta": BETA, "gamma": GAMMA, "theta": THETA,
            "lambda": LAMBDA_, "delta": DELTA, "P_pen": P_PEN, "alpha_0": ALPHA_0,
            "T_allow": T_ALLOW, "T_quar": T_QUAR, "q_low": Q_LOW,
            "delta_D": DELTA_D, "K_allow": K_ALLOW, "eta_C": ETA_C, "C_floor": C_FLOOR
        },
        "comparative_policies": comparative_results,
        "recovery_sweep": sweep_results
    }

    out_file = out_dir / "adaptive_adversary_results.json"
    with open(out_file, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")


if __name__ == "__main__":
    main()
