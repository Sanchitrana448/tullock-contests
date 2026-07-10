"""Experiment 2: sweep the decisiveness parameter r (RQ1).

Two-player symmetric contest, synchronous updating from 20 random starts
per value of r. Figures 4-6.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import DATA_DIR, banner, save_json
from src.contest import TullockContest
from src.dynamics import run_synchronous
from src.plots import save_figure

R_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]
N_PLAYERS = 2
V = 10.0
N_SEEDS = 20
MAX_ITERS = 2000

banner("Experiment 2: Sweeping decisiveness parameter r")

rng = np.random.default_rng(0)
summary = {}

for r in R_VALUES:
    contest = TullockContest(n=N_PLAYERS, r=r, valuations=[V] * N_PLAYERS)
    converged = []
    iterations = []
    total_efforts = []

    for _ in range(N_SEEDS):
        initial = rng.uniform(0.1, V * 0.8, size=N_PLAYERS)
        result = run_synchronous(contest, initial, max_iterations=MAX_ITERS)

        converged.append(result['converged'])
        if result['converged']:
            iterations.append(result['iterations'])
            total_efforts.append(result['final_efforts'].sum())

    summary[r] = {
        'convergence_rate': np.mean(converged),
        'mean_iterations': np.mean(iterations) if iterations else np.nan,
        'mean_total_effort': np.mean(total_efforts) if total_efforts else np.nan,
    }

    print(f"r={r:.1f} | Conv. rate: {summary[r]['convergence_rate'] * 100:3.0f}% | "
          f"Mean iters: {summary[r]['mean_iterations']:8.1f} | "
          f"Mean total effort: {summary[r]['mean_total_effort']:.4f}")

labels = [str(r) for r in R_VALUES]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, [summary[r]['convergence_rate'] * 100 for r in R_VALUES],
       color='steelblue', edgecolor='white')
ax.set_xlabel("Decisiveness parameter r")
ax.set_ylabel("Convergence rate (%)")
ax.set_title("Convergence Rate of Synchronous BRD vs. r\n"
             f"(n={N_PLAYERS} players, V={V}, {N_SEEDS} random initialisations)")
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig04_convergence_rate_vs_r.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(labels, [summary[r]['mean_iterations'] for r in R_VALUES],
        'o-', color='darkorange', markersize=8)
ax.set_xlabel("Decisiveness parameter r")
ax.set_ylabel("Mean iterations to convergence")
ax.set_title("Convergence Speed vs. Decisiveness Parameter r\n"
             "(non-convergent values of r are absent)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig05_convergence_speed_vs_r.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(labels, [summary[r]['mean_total_effort'] for r in R_VALUES],
        's-', color='green', markersize=8)
ax.set_xlabel("Decisiveness parameter r")
ax.set_ylabel("Mean total equilibrium effort")
ax.set_title("Rent Dissipation vs. Decisiveness Parameter r\n"
             "(tests Nitzan 1994 non-monotonicity prediction)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig06_rent_dissipation_vs_r.png")

output = save_json(summary, DATA_DIR / "experiment_02_summary.json")
print(f"\nData saved to {output}")
print("Experiment 2 complete.")
