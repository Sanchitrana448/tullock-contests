"""Experiment 4: heterogeneous valuations (RQ3).

Tests the Cornes and Hartley (2005) prediction that the player who values
the prize more exerts more effort at equilibrium. Figures 10-11.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import DATA_DIR, banner, save_json
from src.contest import TullockContest
from src.dynamics import run_synchronous
from src.plots import save_figure

N_SEEDS = 20
MAX_ITERS = 500

banner("Experiment 4: Heterogeneous player valuations")

rng = np.random.default_rng(0)


def run_case(valuations):
    """Mean equilibrium efforts across converged runs from random starts."""
    contest = TullockContest(n=2, r=1.0, valuations=valuations)
    converged = []
    efforts = []

    for _ in range(N_SEEDS):
        result = run_synchronous(contest, rng.uniform(0.1, 8.0, size=2),
                                 max_iterations=MAX_ITERS)
        converged.append(result['converged'])
        if result['converged']:
            efforts.append(result['final_efforts'])

    return {
        'convergence': np.mean(converged),
        'efforts': np.mean(efforts, axis=0) if efforts else np.full(2, np.nan),
    }


CASES = [
    ("Symmetric\nV1=V2=10", [10.0, 10.0]),
    ("Mild asymmetry\nV1=8, V2=12", [8.0, 12.0]),
    ("Strong asymmetry\nV1=2, V2=18", [2.0, 18.0]),
]

results = {}
for label, valuations in CASES:
    case = run_case(valuations)
    results[label] = case
    p1, p2 = case['efforts']
    verdict = "confirmed" if p2 > p1 else "not confirmed"
    print(f"V={valuations} | Conv: {case['convergence'] * 100:3.0f}% | "
          f"P1={p1:.4f} P2={p2:.4f} | Cornes & Hartley: {verdict}")

print("\nSweeping V2 with V1 = 10 fixed")
v2_values = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
p1_efforts = []
p2_efforts = []

for v2 in v2_values:
    case = run_case([10.0, v2])
    p1_efforts.append(case['efforts'][0])
    p2_efforts.append(case['efforts'][1])
    print(f"V2={v2:4.0f} | P1 effort: {p1_efforts[-1]:.4f} | "
          f"P2 effort: {p2_efforts[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(v2_values, p1_efforts, 'o-', color='steelblue', markersize=8,
        label='Player 1 (V1=10, fixed)')
ax.plot(v2_values, p2_efforts, 's-', color='darkorange', markersize=8,
        label='Player 2 (V2 varies)')
ax.axvline(10, color='grey', linestyle='--', alpha=0.5,
           label='Symmetric point (V2=10)')
ax.set_xlabel("Player 2 valuation (V2)")
ax.set_ylabel("Equilibrium effort")
ax.set_title("Equilibrium Effort vs. Valuation\n"
             "(tests Cornes & Hartley 2005 prediction)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig10_effort_vs_valuation.png")

fig, ax = plt.subplots(figsize=(8, 5))
labels = [label for label, _ in CASES]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width / 2, [results[k]['efforts'][0] for k in labels], width,
       label='Player 1', color='steelblue')
ax.bar(x + width / 2, [results[k]['efforts'][1] for k in labels], width,
       label='Player 2', color='darkorange')
ax.set_xlabel("Contest type")
ax.set_ylabel("Mean equilibrium effort")
ax.set_title("Equilibrium Efforts Under Symmetric vs Asymmetric Valuations")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig11_symmetric_vs_asymmetric.png")

summary = {
    'cases': {label: results[label] for label in labels},
    'valuation_sweep': {
        'v2_values': v2_values,
        'p1_efforts': p1_efforts,
        'p2_efforts': p2_efforts,
    },
}
output = save_json(summary, DATA_DIR / "experiment_04_summary.json")
print(f"\nData saved to {output}")
print("Experiment 4 complete.")
