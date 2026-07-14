"""Experiment 3: sweep the number of players n (RQ2).

r fixed at 1.0, synchronous updating from 20 random starts per value of n.
Figures 7-9.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import DATA_DIR, banner, save_json
from src.contest import TullockContest
from src.dynamics import run_synchronous
from src.plots import save_figure

N_VALUES = [2, 3, 5, 10, 20]
R = 1.0
V = 10.0
N_SEEDS = 20
MAX_ITERS = 2000

banner("Experiment 3: Sweeping number of players n")

rng = np.random.default_rng(0)
summary = {}

def sweep_one_n(contest, initials, x_star):
    converged = []
    iterations = []
    total_efforts = []

    for initial in initials:
        result = run_synchronous(contest, initial, max_iterations=MAX_ITERS)

        converged.append(result['converged'])
        if result['converged']:
            iterations.append(result['iterations'])
            total_efforts.append(float(result['final_efforts'].sum()))

    return {
        'convergence_rate': float(np.mean(converged)),
        'mean_iterations': float(np.mean(iterations)) if iterations else np.nan,
        'mean_total_effort': (float(np.mean(total_efforts)) if total_efforts
                              else np.nan),
        'analytical_x_star': float(x_star),
        'analytical_total': float(x_star * contest.n),
    }


for n in N_VALUES:
    contest = TullockContest(n=n, r=R, valuations=[V] * n)
    x_star = contest.analytical_symmetric_equilibrium()

    initials = [rng.uniform(0.1, V * 0.8, size=n) for _ in range(N_SEEDS)]
    summary[n] = sweep_one_n(contest, initials, x_star)

    print(f"n={n:2d} | Conv. rate: {summary[n]['convergence_rate'] * 100:3.0f}% | "
          f"Mean iters: {summary[n]['mean_iterations']:8.1f} | "
          f"Total effort: {summary[n]['mean_total_effort']:7.4f} | "
          f"Analytical total: {x_star * n:.4f}")

labels = [str(n) for n in N_VALUES]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, [summary[n]['convergence_rate'] * 100 for n in N_VALUES],
       color='steelblue', edgecolor='white')
ax.set_xlabel("Number of players (n)")
ax.set_ylabel("Convergence rate (%)")
ax.set_title("Convergence Rate of Synchronous BRD vs. n\n"
             f"(r={R}, V={V}, {N_SEEDS} random initialisations)")
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig07_convergence_rate_vs_n.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(labels, [summary[n]['mean_iterations'] for n in N_VALUES],
        'o-', color='darkorange', markersize=8)
ax.set_xlabel("Number of players (n)")
ax.set_ylabel("Mean iterations to convergence")
ax.set_title("Convergence Speed vs. Number of Players n\n"
             "(non-convergent values of n are absent)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig08_convergence_speed_vs_n.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(labels, [summary[n]['mean_total_effort'] for n in N_VALUES],
        'o-', color='steelblue', markersize=8, label='Simulation')
ax.plot(labels, [summary[n]['analytical_total'] for n in N_VALUES],
        's--', color='darkorange', markersize=8, label='Analytical (Tullock 1980)')
ax.set_xlabel("Number of players (n)")
ax.set_ylabel("Total equilibrium effort")
ax.set_title("Total Effort vs. Number of Players\n"
             "(simulation shown only where dynamics converged)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig09_total_effort_vs_n.png")

output = save_json(summary, DATA_DIR / "experiment_03_summary.json")
print(f"\nData saved to {output}")
print("Experiment 3 complete.")
