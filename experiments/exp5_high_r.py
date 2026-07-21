"""Experiment 5: high decisiveness (RQ4).

Two-player contests have a pure strategy equilibrium only for r <= n/(n-1)
= 2. This locates where convergence breaks down, what the failure looks
like, and whether damping can recover it. Figures 12-14.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import DATA_DIR, banner, save_json
from src.contest import TullockContest
from src.dynamics import run_inertial, run_synchronous
from src.plots import save_figure

R_FINE = [1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5]
LAMBDAS = [0.8, 0.5, 0.3, 0.1]
N_SEEDS = 20
MAX_ITERS = 500

banner("Experiment 5: Probing high-r non-convergence")

rng = np.random.default_rng(0)

print("\nPart 1: Fine-grained r sweep")
conv_rates = []
for r in R_FINE:
    contest = TullockContest(n=2, r=r, valuations=[10.0, 10.0])
    converged = [
        run_synchronous(contest, rng.uniform(0.1, 8.0, size=2),
                        max_iterations=MAX_ITERS)['converged']
        for _ in range(N_SEEDS)
    ]
    conv_rates.append(np.mean(converged))
    print(f"r={r:.1f} | Conv. rate: {conv_rates[-1] * 100:3.0f}%")

print("\nPart 2: Trajectory at r=3.0")
contest_high = TullockContest(n=2, r=3.0, valuations=[10.0, 10.0])
trajectory = run_synchronous(contest_high, np.array([4.0, 1.0]),
                             max_iterations=100)['trajectory']
print("First 10 effort pairs (Player 1, Player 2):")
for t, (x1, x2) in enumerate(trajectory[:10]):
    print(f"  t={t}: ({x1:.4f}, {x2:.4f})")

print("\nPart 3: Can inertial dynamics rescue convergence at r=3.0?")
inertial_rates = []
for lam in LAMBDAS:
    converged = [
        run_inertial(contest_high, rng.uniform(0.1, 8.0, size=2), lam=lam,
                     max_iterations=MAX_ITERS)['converged']
        for _ in range(N_SEEDS)
    ]
    inertial_rates.append(np.mean(converged))
    print(f"lambda={lam:.1f} | Conv. rate: {inertial_rates[-1] * 100:3.0f}%")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot([str(r) for r in R_FINE], [rate * 100 for rate in conv_rates],
        'o-', color='steelblue', markersize=8)
ax.axvline(2.0, color='red', linestyle='--', alpha=0.6,
           label='r = n/(n-1): pure-equilibrium existence bound')
ax.set_xlabel("Decisiveness parameter r")
ax.set_ylabel("Convergence rate (%)")
ax.set_title("Fine-Grained Convergence Rate vs. r\n"
             "(identifying exact breakdown point)")
ax.set_ylim(-5, 110)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig12_fine_r_convergence.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(trajectory[:, 0], color='steelblue', label='Player 1')
ax.plot(trajectory[:, 1], color='darkorange', label='Player 2')
ax.set_xlabel("Iteration")
ax.set_ylabel("Effort")
ax.set_title("Non-Convergent Trajectory at r=3.0\n"
             "(synchronous dynamics, 100 iterations)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig13_nonconvergent_trajectory.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([str(lam) for lam in LAMBDAS], [rate * 100 for rate in inertial_rates],
       color='green', edgecolor='white')
ax.set_xlabel("Learning rate (lambda)")
ax.set_ylabel("Convergence rate (%)")
ax.set_title("Inertial Dynamics at r=3.0\n"
             "(can slower updating rescue convergence?)")
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save_figure(fig, "fig14_inertial_rescue.png")

summary = {
    'r_fine': R_FINE,
    'conv_rates': conv_rates,
    'lambdas': LAMBDAS,
    'inertial_rates': inertial_rates,
    'trajectory_r3': trajectory,
}
output = save_json(summary, DATA_DIR / "experiment_05_summary.json")
print(f"\nData saved to {output}")
print("Experiment 5 complete.")
