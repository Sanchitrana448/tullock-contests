"""Plots shared between experiments. One-off figures live in their script."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.best_response import compute_all_best_responses

FIGURE_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
})


def save_figure(fig, filename):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_convergence_trajectory(result, contest, title=None, filename=None):
    """Effort paths over time, with the analytical equilibrium marked."""
    trajectory = result['trajectory']
    n_players = trajectory.shape[1]
    palette = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n_players):
        ax.plot(trajectory[:, i], color=palette(i % 10), label=f'Player {i + 1}')

    try:
        x_star = contest.analytical_symmetric_equilibrium()
        ax.axhline(x_star, color='black', linestyle='--', linewidth=1.5,
                   label=f'Nash equilibrium (x*={x_star:.3f})')
    except ValueError:
        pass

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Effort")
    ax.set_title(title or
                 f"Best-Response Dynamics (n={contest.n}, r={contest.r})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filename:
        save_figure(fig, filename)
    return fig


def plot_phase_portrait_2player(contest, grid_size=20, filename=None):
    """Best response displacement field. Arrows vanish at the equilibrium."""
    if contest.n != 2:
        raise ValueError(f"phase portrait needs 2 players, got {contest.n}")

    x_max = contest.valuations.max() * 0.8
    axis = np.linspace(0.01, x_max, grid_size)
    x1, x2 = np.meshgrid(axis, axis)

    dx1 = np.zeros_like(x1)
    dx2 = np.zeros_like(x2)
    for i in range(grid_size):
        for j in range(grid_size):
            br = compute_all_best_responses(contest, np.array([x1[i, j], x2[i, j]]))
            dx1[i, j] = br[0] - x1[i, j]
            dx2[i, j] = br[1] - x2[i, j]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.quiver(x1, x2, dx1, dx2, alpha=0.7, color='steelblue')

    try:
        x_star = contest.analytical_symmetric_equilibrium()
        ax.plot(x_star, x_star, 'r*', markersize=15,
                label=f'Nash equilibrium ({x_star:.2f}, {x_star:.2f})')
        ax.legend()
    except ValueError:
        pass

    ax.set_xlabel("Player 1 effort (x1)")
    ax.set_ylabel("Player 2 effort (x2)")
    ax.set_title(f"Phase Portrait - 2-player contest (r={contest.r})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filename:
        save_figure(fig, filename)
    return fig
