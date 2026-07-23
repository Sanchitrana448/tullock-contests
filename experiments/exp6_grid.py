"""Experiment 6: full r x n convergence map.

Where synchronous best response dynamics reach an equilibrium and where
they do not. Figures 15-16.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import DATA_DIR, banner, save_json
from src.contest import TullockContest
from src.dynamics import run_synchronous
from src.plots import save_figure

R_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
N_VALUES = [2, 3, 5, 10, 20]
V = 10.0
N_SEEDS = 10
MAX_ITERS = 300

banner("Experiment 6: Combined r x n parameter sweep")
print(f"Grid: {len(R_VALUES)} r values x {len(N_VALUES)} n values, "
      f"{N_SEEDS} seeds per cell\n")

rng = np.random.default_rng(0)
conv_grid = np.zeros((len(N_VALUES), len(R_VALUES)))

for i, n in enumerate(N_VALUES):
    for j, r in enumerate(R_VALUES):
        contest = TullockContest(n=n, r=r, valuations=[V] * n)
        converged = [
            run_synchronous(contest, rng.uniform(0.1, V * 0.8, size=n),
                            max_iterations=MAX_ITERS)['converged']
            for _ in range(N_SEEDS)
        ]
        conv_grid[i, j] = np.mean(converged)
        print(f"n={n:2d}, r={r:.1f} | Conv. rate: {conv_grid[i, j] * 100:3.0f}%")

fig, ax = plt.subplots(figsize=(11, 6))
image = ax.imshow(conv_grid * 100, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(R_VALUES)), [str(r) for r in R_VALUES])
ax.set_yticks(range(len(N_VALUES)), [str(n) for n in N_VALUES])
ax.set_xlabel("Decisiveness parameter r")
ax.set_ylabel("Number of players (n)")
ax.set_title("Convergence Rate of Synchronous BRD\nAcross r x n Parameter Space (%)")

for i in range(len(N_VALUES)):
    for j in range(len(R_VALUES)):
        percent = conv_grid[i, j] * 100
        ax.text(j, i, f'{percent:.0f}%', ha='center', va='center', fontsize=9,
                fontweight='bold',
                color='white' if percent < 40 or percent > 80 else 'black')

fig.colorbar(image, ax=ax, label='Convergence rate (%)')
fig.tight_layout()
save_figure(fig, "fig15_convergence_heatmap.png")

fig, ax = plt.subplots(figsize=(9, 5))
for i, n in enumerate(N_VALUES):
    ax.plot(R_VALUES, conv_grid[i, :] * 100, 'o-', markersize=7, label=f'n={n}')
ax.set_xlabel("Decisiveness parameter r")
ax.set_ylabel("Convergence rate (%)")
ax.set_title("Convergence Rate vs. r for Different n Values")
ax.legend(title="Players")
ax.set_ylim(-5, 110)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig16_convergence_by_n_and_r.png")

print("\nSummary: convergence rate (%) across r x n")
print("-" * 55)
print("n\\r  " + "  ".join(f"{r:>5.1f}" for r in R_VALUES))
for i, n in enumerate(N_VALUES):
    print(f"n={n:2d} " + "  ".join(f"{conv_grid[i, j] * 100:>4.0f}%"
                                   for j in range(len(R_VALUES))))

output = save_json({'r_values': R_VALUES,
                    'n_values': N_VALUES,
                    'convergence_grid': conv_grid},
                   DATA_DIR / "experiment_06_summary.json")
print(f"\nData saved to {output}")
print("Experiment 6 complete.")
