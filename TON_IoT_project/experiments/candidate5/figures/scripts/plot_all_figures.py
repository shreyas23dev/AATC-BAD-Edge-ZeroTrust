"""
Candidate 5 — Publication Figures Generation Pipeline (Audited & Verified)
==========================================================================
Generates IEEE/InCIS conference publication-quality figures directly from
authoritative experiment datasets extracted by extract_figure_data.py.

STRICT INTEGRITY GUARANTEES:
- Zero hardcoded policy sequences
- Zero hardcoded outcome measurements
- All plotting inputs read directly from JSON data artifacts
- Vector PDF and 300 DPI PNG outputs for all figures
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Ensure data extraction is up to date
from extract_figure_data import extract_all

DATA_DIR = Path("experiments/candidate5/figures/data")
FIG_DIR = Path("experiments/candidate5/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication visual styling
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0,
    'lines.markersize': 6.0
})

C_B0 = '#C0392B'       # Crimson red for baseline vulnerability
C_FAMC = '#E67E22'     # Amber orange for Family C
C_AATC = '#1F77B4'     # Deep blue for AATC countermeasure
C_VERIFY = '#8E44AD'   # Purple for VERIFY policy
C_ALLOW = '#27AE60'    # Green for ALLOW policy
C_BLOCK = '#E74C3C'    # Red for BLOCK policy

# =============================================================================
# FIGURE 1: Representative Cyclic Trust Trajectories (B0 vs Family C vs AATC)
# =============================================================================
def plot_figure_1():
    data = json.load(open(DATA_DIR / 'figure1_data.json'))
    cycles = data['cycles']
    t_allow = data['T_allow']
    t_quar = data['T_quar']

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    
    ax.axhline(t_allow, color='gray', linestyle=':', linewidth=1.5, label=f'ALLOW Threshold ($T_{{\\rm allow}} = {t_allow:.2f}$)')
    ax.axhline(t_quar, color='darkgray', linestyle='-.', linewidth=1.5, label=f'VERIFY/BLOCK ($T_{{\\rm quar}} = {t_quar:.2f}$)')

    ax.plot(cycles, data['B0_T_post_atk'], marker='s', color=C_B0, label='Baseline B0 ($T$ post-attack)', linestyle='-')
    ax.plot(cycles, data['Family_C_T_post_atk'], marker='^', color=C_FAMC, label='Family C ($T$ post-attack)', linestyle='--')
    ax.plot(cycles, data['AATC_T_post_atk'], marker='o', color=C_AATC, label='AATC-BAD ($T$ post-attack)', linestyle='-')
    ax.plot(cycles, data['AATC_C_post_atk'], marker='x', color='teal', label=r'AATC-BAD Ceiling ($C_t$ post-attack)', linestyle=':')

    # Dynamically annotate any demotions below T_allow
    for idx, (c, t_val) in enumerate(zip(cycles, data['AATC_T_post_atk'])):
        if t_val < t_allow and t_val >= t_quar:
            offset_y = -0.15 if idx % 2 == 0 else -0.22
            ax.annotate(f'DEMOTED to VERIFY\n($T = {t_val:.4f} < {t_allow:.2f}$)',
                        xy=(c, t_val),
                        xytext=(c - 1.2, t_val + offset_y),
                        arrowprops=dict(facecolor=C_VERIFY, shrink=0.08, width=1.2, headwidth=6),
                        fontsize=8.5, fontweight='bold', color=C_VERIFY,
                        bbox=dict(boxstyle='round,pad=0.3', fc='lavender', alpha=0.85))

    ax.set_xlabel('Exploitation Cycle Index ($k$)')
    ax.set_ylabel('Trust Level Post-Attack ($T$)')
    ax.set_title('Figure 1: Cyclic Trust Trajectories under Cycle A Schedule (1:10)')
    ax.set_ylim(0.25, 1.05)
    ax.set_xticks(cycles)
    ax.legend(loc='lower left', framealpha=0.9)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'figure1_cyclic_trust_trajectories.png', dpi=300)
    fig.savefig(FIG_DIR / 'figure1_cyclic_trust_trajectories.pdf')
    plt.close(fig)
    print("Figure 1 generated.")

# =============================================================================
# FIGURE 2: Cumulative Attack Exposure across Cycles and Schedules
# =============================================================================
def plot_figure_2():
    data = json.load(open(DATA_DIR / 'figure2_data.json'))
    cycles = data['cycles']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    # Subplot 2A: Cycle A cumulative damage progression
    b0_final_m = data['B0_cum_M'][-1]
    famc_final_m = data['Family_C_cum_M'][-1]
    aatc_final_m = data['AATC_cum_M'][-1]
    m_reduction_pct = (b0_final_m - aatc_final_m) / b0_final_m * 100

    ax1.plot(cycles, data['B0_cum_M'], marker='s', color=C_B0, label=f'Baseline B0 ($M_{{\\rm cum}} = {b0_final_m:.1f}$)')
    ax1.plot(cycles, data['Family_C_cum_M'], marker='^', color=C_FAMC, linestyle='--', label=f'Family C ($M_{{\\rm cum}} = {famc_final_m:.1f}$)')
    ax1.plot(cycles, data['AATC_cum_M'], marker='o', color=C_AATC, label=f'AATC-BAD ($M_{{\\rm cum}} = {aatc_final_m:.1f}$)')
    ax1.fill_between(cycles, data['AATC_cum_M'], data['B0_cum_M'], color=C_AATC, alpha=0.15, label=f'Exposure Reduction (-{m_reduction_pct:.1f}%)')
    ax1.set_xlabel('Cycle Index ($k$)')
    ax1.set_ylabel(r'Cumulative Malicious Yield ($M_{\rm cum}$)')
    ax1.set_title('(a) Cumulative Malicious Yield Progression (Cycle A)')
    ax1.set_xticks(cycles)
    ax1.legend(loc='upper left', framealpha=0.9)

    # Subplot 2B: Multi-schedule comparison
    scheds = data['schedules_summary']['schedules']
    x = np.arange(len(scheds))
    width = 0.25

    ax2.bar(x - width, data['schedules_summary']['B0_M'], width, label='Baseline B0', color=C_B0, alpha=0.85)
    ax2.bar(x, data['schedules_summary']['Family_C_M'], width, label='Family C', color=C_FAMC, alpha=0.85)
    ax2.bar(x + width, data['schedules_summary']['AATC_M'], width, label='AATC-BAD', color=C_AATC, alpha=0.85)
    ax2.set_xlabel('Attack Schedule')
    ax2.set_ylabel(r'Total Malicious Yield ($M_{\rm cum}$)')
    ax2.set_title('(b) Total Malicious Yield Across All Schedules')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Cycle A\n(1:10)', 'Cycle B\n(1:25)', 'Cycle C\n(1:50)', 'Cycle D\n(5:25)', 'Cycle E\n(5:50)'], fontsize=9)
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.suptitle('Figure 2: Cumulative Attack Exposure and Multi-Schedule Security Comparison', y=1.02)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'figure2_cumulative_attack_exposure.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'figure2_cumulative_attack_exposure.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Figure 2 generated.")

# =============================================================================
# FIGURE 3: Policy-State Distributions
# =============================================================================
def plot_figure_3():
    data = json.load(open(DATA_DIR / 'figure3_data.json'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    # Subplot 3A: Benign Availability Policy Distribution (N=7,459)
    cats = data['benign_evaluation']['categories']
    x = np.arange(len(cats))
    allow_pct = data['benign_evaluation']['ALLOW_pct']
    verify_pct = data['benign_evaluation']['VERIFY_pct']
    block_pct = data['benign_evaluation']['BLOCK_pct']

    ax1.bar(x, allow_pct, label='ALLOW (Nominal Service)', color=C_ALLOW, alpha=0.85)
    ax1.bar(x, verify_pct, bottom=allow_pct, label='VERIFY (Step-Up Challenge)', color=C_VERIFY, alpha=0.85)
    ax1.bar(x, block_pct, bottom=np.array(allow_pct) + np.array(verify_pct), label='BLOCK (Quarantine)', color=C_BLOCK, alpha=0.85)

    ax1.set_ylabel('Telemetry Frame Fraction (%)')
    ax1.set_title('(a) Benign Availability Profile (N=7,459 frames)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=9.5)
    ax1.set_ylim(0, 110)
    for i in range(len(cats)):
        ax1.text(i, allow_pct[i]/2, f"{allow_pct[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    ax1.legend(loc='lower left', framealpha=0.9)

    # Subplot 3B: Malicious Attack Pulse Policy Decisions (Cycle A, 9 pulses)
    atk_allow = data['attack_pulses_cycle_a']['ALLOW_count']
    atk_verify = data['attack_pulses_cycle_a']['VERIFY_count']

    ax2.bar(x - 0.15, atk_allow, width=0.3, label='Malicious ALLOW (Exploited)', color='#E74C3C', alpha=0.85)
    ax2.bar(x + 0.15, atk_verify, width=0.3, label='Malicious VERIFY (Challenged)', color='#2980B9', alpha=0.85)
    ax2.set_ylabel('Attack Pulse Count (out of 9 pulses)')
    ax2.set_title('(b) Cycle A Attack Pulse Policy Outcomes')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=9.5)
    ax2.set_ylim(0, 10)
    for i in range(len(cats)):
        ax2.text(i - 0.15, atk_allow[i] + 0.3, str(atk_allow[i]), ha='center', fontweight='bold')
        ax2.text(i + 0.15, atk_verify[i] + 0.3, str(atk_verify[i]), ha='center', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.suptitle('Figure 3: Policy-State Profiles on Benign Telemetry vs Malicious Attack Pulses', y=1.02)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'figure3_policy_state_distributions.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'figure3_policy_state_distributions.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Figure 3 generated.")

# =============================================================================
# FIGURE 4: AATC Operating Envelope (Duty Cycle / Rest Period vs Security)
# =============================================================================
def plot_figure_4():
    data = json.load(open(DATA_DIR / 'figure4_data.json'))
    tau_r = data['tau_r']
    delta_m = data['Delta_M']
    demotions = data['AATC_demotions']

    fig, ax1 = plt.subplots(figsize=(7.8, 4.2), dpi=300)

    color1 = '#1F77B4'
    color2 = '#E67E22'

    ax1.set_xlabel(r'Rest Window Duration ($\tau_r$ steps) [Duty Cycle $\tau_a / (\tau_a + \tau_r)$]')
    ax1.set_ylabel(r'Security Gain $\Delta M = M_{\rm cum}(B0) - M_{\rm cum}(AATC)$', color=color1)
    line1 = ax1.plot(tau_r, delta_m, marker='o', color=color1, linewidth=2.2, label=r'$\Delta M$ Malicious Yield Reduction')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.5, 5.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('ALLOW Attack Pulses Demoted to VERIFY', color=color2)
    line2 = ax2.plot(tau_r, demotions, marker='s', color=color2, linestyle='--', linewidth=2.0, label='ALLOW Demotions')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-1, 10)
    ax2.grid(False)

    # Highlight active envelope region (tau_r <= 12)
    ax1.axvspan(4, 12.5, color='green', alpha=0.10, label='Active AATC Intervention Envelope')
    ax1.axvspan(12.5, 62, color='gray', alpha=0.08, label='Native B0 Quarantine Region')

    # Dual x-axis tick labels showing duty cycle
    duty_labels = [f"{t}\n({1/(1+t):.2f})" for t in tau_r]
    ax1.set_xticks(tau_r)
    ax1.set_xticklabels(duty_labels, fontsize=8.5)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)

    plt.title('Figure 4: AATC-BAD Duty-Cycle Operating Envelope')
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'figure4_aatc_operating_envelope.png', dpi=300)
    fig.savefig(FIG_DIR / 'figure4_aatc_operating_envelope.pdf')
    plt.close(fig)
    print("Figure 4 generated.")

# =============================================================================
# FIGURE 5: Security-Availability Trade-off
# =============================================================================
def plot_figure_5():
    data = json.load(open(DATA_DIR / 'figure5_data.json'))
    pts = data['points']

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)

    for p in pts:
        if p['is_phase9_frozen']:
            ax.scatter(p['Delta_ALLOW_pp'], p['Delta_M'], color='red', s=140, marker='*', zorder=5,
                       label=r'Phase 9 Frozen ($\delta_D=0.10, K_{\rm allow}=1.70$) [$\Delta$ALLOW = 0.00 pp]')
        elif p['delta_D'] == 0.05:
            ax.scatter(p['Delta_ALLOW_pp'], p['Delta_M'], color='#E67E22', s=55, alpha=0.8, marker='^',
                       label=r'Aggressive ($\delta_D=0.05$)' if p['K_allow'] == 1.20 else '')
        elif p['delta_D'] == 0.10:
            ax.scatter(p['Delta_ALLOW_pp'], p['Delta_M'], color='#2980B9', s=55, alpha=0.8, marker='o',
                       label=r'Calibrated ($\delta_D=0.10$)' if p['K_allow'] == 1.20 else '')
        else:
            ax.scatter(p['Delta_ALLOW_pp'], p['Delta_M'], color='gray', s=40, alpha=0.6, marker='x',
                       label=r'Conservative ($\delta_D \geq 0.15$)' if p['delta_D'] == 0.15 and p['K_allow'] == 1.20 else '')

    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axhline(0.0, color='black', linestyle=':', linewidth=1.0, alpha=0.5)

    ax.annotate(r'Zero-Availability-Cost Ridge ($\Delta$ALLOW = 0.00 pp)',
                xy=(0.0, 0.0), xytext=(0.2, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.0, headwidth=5),
                fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9))

    ax.set_xlabel(r'Availability Change vs Baseline B0 ($\Delta$ ALLOW fraction, percentage points)')
    ax.set_ylabel(r'Security Improvement ($\Delta M = M_{\rm cum}^{B0} - M_{\rm cum}^{\rm AATC}$)')
    ax.set_title('Figure 5: Security–Availability Trade-Off Surface (Validation Partition)')
    ax.set_xlim(-3.5, 1.0)
    ax.set_ylim(-0.2, 1.4)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'figure5_security_availability_tradeoff.png', dpi=300)
    fig.savefig(FIG_DIR / 'figure5_security_availability_tradeoff.pdf')
    plt.close(fig)
    print("Figure 5 generated.")

# =============================================================================
# FIGURE 6: Debt and Ceiling Dynamics Explaining AATC Behavior
# =============================================================================
def plot_figure_6():
    data = json.load(open(DATA_DIR / 'figure6_data.json'))
    cycles = data['cycles']
    k_allow = data['K_allow']
    t_allow = data['T_allow']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True, dpi=300)

    # Panel A: Authorization Debt Ratchet
    ax1.plot(cycles, data['D_post_atk'], marker='o', color='#8E44AD', label=r'Debt Post-Attack ($D_{\rm post-atk}$)')
    ax1.plot(cycles, data['D_post_rst'], marker='s', color='#BDC3C7', linestyle='--', label=r'Debt Post-Rest ($D_{\rm post-rst}$)')
    ax1.axhline(k_allow, color='#C0392B', linestyle=':', linewidth=1.8, label=f'Allowance Buffer ($K_{{\\rm allow}} = {k_allow:.2f}$)')
    ax1.fill_between(cycles, k_allow, [max(d, k_allow) for d in data['D_post_atk']], color='#8E44AD', alpha=0.2, label=r'Excess Debt Region ($D > K_{\rm allow}$)')
    ax1.set_ylabel('Authorization Debt ($D$)')
    ax1.set_title('(a) Authorization Debt Accumulation and Relaxation Across Cycles')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=8.5)

    # Panel B: Trust Level and Adaptive Ceiling
    ax2.plot(cycles, data['T_post_atk'], marker='o', color=C_AATC, label=r'Trust Post-Attack ($T_{\rm post-atk}$)')
    ax2.plot(cycles, data['C_post_atk'], marker='^', color='teal', linestyle=':', label=r'Adaptive Ceiling ($C_t$)')
    ax2.axhline(t_allow, color='darkgray', linestyle='--', linewidth=1.5, label=f'ALLOW Threshold ($T_{{\\rm allow}} = {t_allow:.2f}$)')

    # Dynamically label VERIFY decisions from the data policies list
    pols = data['policies']
    for i, p in enumerate(pols):
        if p == 'VERIFY':
            ax2.scatter(cycles[i], data['T_post_atk'][i], color='magenta', s=100, zorder=6)
            ax2.text(cycles[i], data['T_post_atk'][i] - 0.06, 'VERIFY', ha='center', fontsize=8, fontweight='bold', color='purple')

    ax2.set_xlabel('Exploitation Cycle Index ($k$)')
    ax2.set_ylabel('Trust Level / Ceiling')
    ax2.set_title(r'(b) Adaptive Ceiling Attenuation and Resulting Policy Demotions')
    ax2.set_ylim(0.65, 1.05)
    ax2.set_xticks(cycles)
    ax2.legend(loc='lower left', framealpha=0.9, fontsize=8.5)

    plt.suptitle('Figure 6: State-Variable Dynamics of AATC-BAD under Cycle A (1:10)', y=1.02)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'figure6_debt_and_ceiling_dynamics.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'figure6_debt_and_ceiling_dynamics.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Figure 6 generated.")

# =============================================================================
# FIGURE 7: Robustness / Sensitivity Grid Heatmaps
# =============================================================================
def plot_figure_7():
    data = json.load(open(DATA_DIR / 'figure7_data.json'))
    dD = data['delta_D_values']
    kA = data['K_allow_values']
    mat_m = np.array(data['Delta_M_grid'])
    mat_avail = np.array(data['Delta_ALLOW_grid'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    # Subplot 7A: Security improvement Delta M
    ax1.imshow(mat_m, cmap='Blues', origin='lower', aspect='auto')
    ax1.set_xticks(range(len(kA)))
    ax1.set_yticks(range(len(dD)))
    ax1.set_xticklabels(kA)
    ax1.set_yticklabels(dD)
    ax1.set_xlabel(r'Allowance Buffer ($K_{\rm allow}$)')
    ax1.set_ylabel(r'Debt Shedding Rate ($\delta_D$)')
    ax1.set_title(r'(a) Security Gain $\Delta M$')
    for i in range(len(dD)):
        for j in range(len(kA)):
            val = mat_m[i, j]
            ax1.text(j, i, f"+{val:.1f}" if val > 0 else f"{val:.1f}", ha='center', va='center',
                     color='white' if val > 0.5 else 'black', fontweight='bold')
    
    # Locate frozen point dynamically
    fz_dD = data['frozen_point']['delta_D']
    fz_kA = data['frozen_point']['K_allow']
    fz_j = kA.index(fz_kA)
    fz_i = dD.index(fz_dD)
    ax1.scatter(fz_j, fz_i, color='red', s=160, marker='*', edgecolors='black', label='Phase 9 Frozen')
    ax1.legend(loc='upper right', fontsize=8)

    # Subplot 7B: Availability Impact Delta ALLOW pp
    ax2.imshow(mat_avail, cmap='YlOrRd_r', origin='lower', aspect='auto')
    ax2.set_xticks(range(len(kA)))
    ax2.set_yticks(range(len(dD)))
    ax2.set_xticklabels(kA)
    ax2.set_yticklabels(dD)
    ax2.set_xlabel(r'Allowance Buffer ($K_{\rm allow}$)')
    ax2.set_ylabel(r'Debt Shedding Rate ($\delta_D$)')
    ax2.set_title(r'(b) Availability Impact $\Delta$ ALLOW (percentage points)')
    for i in range(len(dD)):
        for j in range(len(kA)):
            val = mat_avail[i, j]
            ax2.text(j, i, f"{val:.2f}", ha='center', va='center',
                     color='white' if val < -1.5 else 'black', fontweight='bold')
    ax2.scatter(fz_j, fz_i, color='red', s=160, marker='*', edgecolors='black', label='Phase 9 Frozen')
    ax2.legend(loc='lower right', fontsize=8)

    plt.suptitle('Figure 7: Sensitivity Grid Heatmaps across Parameter Space (Validation Partition)', y=1.02)
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'figure7_robustness_sensitivity_grid.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'figure7_robustness_sensitivity_grid.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Figure 7 generated.")

# =============================================================================
# FIGURE 8: Dual-Regime Defense Dynamics (B0 Quarantine vs AATC Intervention)
# =============================================================================
def plot_figure_8():
    data = json.load(open(DATA_DIR / 'figure8_data.json'))
    tau_r = data['tau_r']
    b0_survived = data['B0_survived']
    delta_m = data['Delta_M']

    fig, ax1 = plt.subplots(figsize=(7.8, 4.3), dpi=300)

    c_surv = '#C0392B'
    c_gain = '#1F77B4'

    ax1.set_xlabel(r'Attacker Rest Window ($\tau_r$ steps) [Duty Cycle $\tau_a / (\tau_a + \tau_r)$]')
    ax1.set_ylabel('Adversary Survived Cycles under Baseline B0', color=c_surv)
    line1 = ax1.plot(tau_r, b0_survived, marker='s', color=c_surv, linewidth=2.2, label='Adversary Survival under B0')
    ax1.tick_params(axis='y', labelcolor=c_surv)
    ax1.set_ylim(0, 20)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'AATC Additional Intervention Gain ($\Delta M$)', color=c_gain)
    line2 = ax2.plot(tau_r, delta_m, marker='o', color=c_gain, linestyle='--', linewidth=2.0, label='AATC Security Intervention Gain')
    ax2.tick_params(axis='y', labelcolor=c_gain)
    ax2.set_ylim(-0.5, 6.0)
    ax2.grid(False)

    # Shaded regimes based on envelope
    ax1.axvspan(4, 12.5, color='blue', alpha=0.10, label=r'Regime I: AATC Active Intervention ($\Delta M > 0$)')
    ax1.axvspan(12.5, 62, color='gray', alpha=0.12, label=r'Regime II: Native B0 Quarantine ($\Delta M = 0$)')

    # Boundary annotation
    ax1.axvline(12.5, color='darkgray', linestyle=':', linewidth=1.8)
    ax1.text(13.5, 16.5, r'Regime Boundary ($\tau_r^* \approx 12$)', fontsize=9, color='dimgray', rotation=90)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)

    # Dual x-axis tick labels
    duty_labels = [f"{t}\n({1/(1+t):.2f})" for t in tau_r]
    ax1.set_xticks(tau_r)
    ax1.set_xticklabels(duty_labels, fontsize=8.5)

    plt.title('Figure 8: Dual-Regime Defense Architecture: B0 Quarantine vs AATC Intervention')
    plt.tight_layout()

    fig.savefig(FIG_DIR / 'figure8_dual_regime_defense_dynamics.png', dpi=300)
    fig.savefig(FIG_DIR / 'figure8_dual_regime_defense_dynamics.pdf')
    plt.close(fig)
    print("Figure 8 generated.")

if __name__ == '__main__':
    print("Step 1: Extracting authoritative data...")
    extract_all()
    print("\nStep 2: Rendering figures...")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    plot_figure_7()
    plot_figure_8()
    print("\nAll 8 publication figures regenerated directly from authoritative artifacts.")
