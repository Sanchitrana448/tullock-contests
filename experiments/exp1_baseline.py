"""Experiment 1: baseline two-player symmetric contest.

Checks all three update rules against x* = 2.5 and produces figures 1-3.
"""

import numpy as np

from common import banner
from src.contest import TullockContest
from src.dynamics import run_asynchronous, run_inertial, run_synchronous
from src.plots import plot_convergence_trajectory, plot_phase_portrait_2player

banner("Experiment 1: Baseline 2-player symmetric contest")

contest = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0])
initial = np.array([8.0, 1.0])

runs = {
    "Synchronous": run_synchronous(contest, initial),
    "Asynchronous": run_asynchronous(contest, initial, seed=42),
    "Inertial (l=0.5)": run_inertial(contest, initial, lam=0.5),
}

print(f"Analytical equilibrium: x* = {contest.analytical_symmetric_equilibrium()}")
for name, result in runs.items():
    print(f"{name:18} {result['iterations']:>4} iterations, "
          f"converged={result['converged']}, "
          f"final={np.round(result['final_efforts'], 4)}")

print("\nProducing figures...")

plot_convergence_trajectory(
    runs["Synchronous"], contest,
    title="Synchronous BRD - 2-player symmetric (V=10, r=1)",
    filename="fig01_sync_2player_convergence.png",
)
plot_convergence_trajectory(
    runs["Inertial (l=0.5)"], contest,
    title="Inertial BRD (l=0.5) - 2-player symmetric (V=10, r=1)",
    filename="fig02_inertial_2player_convergence.png",
)
plot_phase_portrait_2player(contest, filename="fig03_phase_portrait_r1.png")

print("\nExperiment 1 complete.")
