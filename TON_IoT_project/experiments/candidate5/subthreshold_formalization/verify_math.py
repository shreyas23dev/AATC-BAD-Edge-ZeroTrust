"""
Phase 5 Math Verification Script
Checks all closed-form derivations for the sub-threshold region formalization.
No empirical experiments. Pure analytical computation.
"""

A_THR = 0.4554
GAMMA = 0.335
BETA = 0.40
ALPHA_0 = 0.2057
DELTA = 0.05
LAMBDA_ = 0.55

# --- R_sub boundary verification ---
A_atk = 0.4100
E_0 = 0.1656

E_after_atk = BETA * A_atk + (1.0 - BETA) * E_0
print(f"E_after_attack = {BETA}*{A_atk} + {1.0-BETA}*{E_0} = {E_after_atk:.6f}")
print(f"A_atk < A_thr: {A_atk < A_THR}")
print(f"E_after_atk < gamma: {E_after_atk < GAMMA}")

# Boundary line: beta*A + (1-beta)*E = gamma  =>  E = (gamma - beta*A)/(1-beta)
E_bound_A0 = GAMMA / (1.0 - BETA)
E_bound_Athr = (GAMMA - BETA * A_THR) / (1.0 - BETA)
E_bound_A41 = (GAMMA - BETA * 0.41) / (1.0 - BETA)
print(f"Boundary line E = (gamma - beta*A)/(1-beta)")
print(f"  At A=0:     E_bound = {E_bound_A0:.6f}")
print(f"  At A=A_thr: E_bound = {E_bound_Athr:.6f}")
print(f"  At A=0.41:  E_bound = {E_bound_A41:.6f}")

# Consecutive pulses: how many before E > gamma?
print("\nConsecutive pulse accumulation from E_0=0.1656:")
E = E_0
for i in range(1, 8):
    E = BETA * A_atk + (1.0 - BETA) * E
    exceeds = "EXCEEDS" if E > GAMMA else "below"
    print(f"  Pulse {i}: E = {E:.6f}  ({exceeds} gamma={GAMMA})")
    if E > GAMMA:
        print(f"  => Exceeds gamma after {i} consecutive identical pulses")
        break

# Fixed point of E under constant A=0.41
E_fixed = A_atk
print(f"\nFixed point of E under constant A={A_atk}: E* = {E_fixed} (> gamma = {GAMMA})")
print(f"Cycle A uses 1 pulse then 10 rest steps -- E never reaches fixed point.")

# Energy during rest steps (A_rest = 0.1656)
A_rest = 0.1656
E_rest_start = E_after_atk
print(f"\nRest trajectory from E={E_rest_start:.6f} (A_rest={A_rest}):")
E = E_rest_start
for i in range(1, 12):
    E = BETA * A_rest + (1.0 - BETA) * E
    print(f"  Rest step {i}: E = {E:.6f}")

# Fixed point of E during rest
E_rest_fixed = A_rest
print(f"\nFixed point of E during rest (A={A_rest}): E* = {E_rest_fixed:.4f}")

# State aliasing construction
print("\n--- State Aliasing Verification ---")
# Both histories start at (T=1.0, E=0.1656, SB=0.0)
# H_attack: 1 adversarial pulse at A=0.4100
# H_benign: 1 benign pulse at A_B producing same E

# After 1 attack pulse: T stays 1.0 (alpha*(1-1.0)=0), E=E_after_atk, SB=max(0,SB-delta)=0
T_atk = 1.0
E_atk = E_after_atk
SB_atk = 0.0
print(f"After H_attack (1 pulse A=0.41): T={T_atk}, E={E_atk:.6f}, SB={SB_atk}")

# Any single benign pulse A_B that produces same E?
# E_b = BETA*A_B + (1-BETA)*0.1656 = 0.40*A_B + 0.0994
# For E_b = E_after_atk = 0.2634:
# 0.40*A_B = 0.2634 - 0.0994 = 0.1640
# A_B = 0.4100
# Same A! So in one step the aliasing is trivial (same A, same outcome)
# Need to demonstrate across different histories

# Demonstrate: After k cycles of (1 atk + 10 rest), the final state
# is indistinguishable from some benign rest history from the same starting state
import math

# Simulate 1 cycle of Cycle A
T = 1.0; E = 0.1656; SB = 0.0
# Attack step
E = BETA * 0.41 + (1.0 - BETA) * E
SB = max(0.0, SB - DELTA)
T = T + ALPHA_0 * (1.0 - T)
print(f"\nAfter 1 atk pulse: T={T:.6f}, E={E:.6f}, SB={SB:.6f}")

# 10 rest steps
for _ in range(10):
    E = BETA * A_rest + (1.0 - BETA) * E
    SB = max(0.0, SB - DELTA)
    T = T + ALPHA_0 * (1.0 - T)
print(f"After 1 atk + 10 rest: T={T:.6f}, E={E:.6f}, SB={SB:.6f}")

# Now simulate pure rest from initial state
T2 = 1.0; E2 = 0.1656; SB2 = 0.0
# After 11 pure rest steps
for _ in range(11):
    E2 = BETA * A_rest + (1.0 - BETA) * E2
    SB2 = max(0.0, SB2 - DELTA)
    T2 = T2 + ALPHA_0 * (1.0 - T2)
print(f"After 11 pure rest steps: T={T2:.6f}, E={E2:.6f}, SB={SB2:.6f}")

# Find how many rest steps from a different E start produce the same (T,E,SB)
# After attack: E = 0.2634. After 10 rest: E converges toward 0.1656
# The key aliasing: state at END of 1-attack-10-rest cannot be confused with pure rest
# unless E_start is chosen carefully.
# Let's find E_start for pure rest to get same E as after 1-atk+10-rest:
E_target = E   # approx 0.166 after 10 rest steps
E_target_val = E
# After 1 atk + 10 rest: T=1.0, E=~0.1659, SB=0.0
# Same as pure rest from ANY E_start: after enough steps, E -> 0.1656

# For aliasing: show that EXACTLY 1-atk + 10-rest and 0-atk + some-rest lead to same state
# Solve: 0.1656 after n rest steps from E_start = E_target_val from 0 atk + n rest:
# E_n = A_rest + (E_0 - A_rest) * (1-BETA)^n
# When E_0 = 0.1656 (fixed point), E_n = 0.1656 for all n
# So pure rest from E_0=0.1656 gives same E as after atk+rest in ~10 steps
# T aliasing: T=1.0 in both cases (T was already 1.0 and doesn't change in BENIGN_RECOVERY when T<1)
# Actually T DOES change: T = T + alpha*(1-T), so T increases toward 1.0
# At T=1.0 it's exactly 1.0, stays there.

print("\n--- Summary ---")
print(f"E_after 1 attack + 10 rest: {E:.8f}")
print(f"E_after 11 pure rest steps: {E2:.8f}")
print(f"Delta E: {abs(E - E2):.8f}")
print("B0 state after 1-atk+10-rest vs 11-pure-rest: DISTINGUISHABLE by E (attack E > rest E)")
print(f"But this gap closes. After {100} rest steps:")
T3 = 1.0; E3 = E_after_atk; SB3 = 0.0
for _ in range(100):
    E3 = BETA * A_rest + (1.0 - BETA) * E3
print(f"  E after 1-atk+100-rest = {E3:.8f}")
T4 = 1.0; E4 = 0.1656; SB4 = 0.0
for _ in range(100):
    E4 = BETA * A_rest + (1.0 - BETA) * E4
print(f"  E after 100-pure-rest  = {E4:.8f}")
print(f"  Delta E: {abs(E3-E4):.10f} (effectively zero)")

print("\nConclusion:")
print("B0 state after N atk-cycles CONVERGES to same state as pure rest")
print("after sufficient additional rest. The aliasing is ASYMPTOTIC.")
print("Formally: states become epsilon-aliased for any epsilon>0 as N_rest -> infinity.")
print("The current state (T,E,SB) does not encode 'number of prior attack cycles'.")
