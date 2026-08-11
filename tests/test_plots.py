"""Plotting helpers.

The figures get judged by eye, so these check what the eye misses: that a
plot uses the data it claims to, that the equilibrium marker appears only
when a closed form exists, that the phase portrait field vanishes at the
equilibrium, and that files land where the experiment scripts expect.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.contest import TullockContest  # noqa: E402
from src.dynamics import run_synchronous  # noqa: E402
from src.plots import (plot_convergence_trajectory,  # noqa: E402
                       plot_phase_portrait_2player, save_figure)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def converged_run():
    contest = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0])
    result = run_synchronous(contest, np.array([1.0, 3.0]),
                             max_iterations=200)
    return contest, result


# --- save_figure ----------------------------------------------------------

def test_save_figure_writes_a_png_and_closes_it(tmp_path, monkeypatch):
    monkeypatch.setattr("src.plots.FIGURE_DIR", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    save_figure(fig, "fig99_test.png")

    written = tmp_path / "fig99_test.png"
    assert written.exists()
    assert written.stat().st_size > 0
    assert not plt.get_fignums()


def test_save_figure_creates_the_directory(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "figures"
    monkeypatch.setattr("src.plots.FIGURE_DIR", target)
    fig, _ = plt.subplots()

    save_figure(fig, "fig98_test.png")

    assert (target / "fig98_test.png").exists()


# --- trajectory plot ------------------------------------------------------

def test_trajectory_plot_draws_one_line_per_player(converged_run):
    contest, result = converged_run

    fig = plot_convergence_trajectory(result, contest)
    ax = fig.axes[0]

    player_lines = [line for line in ax.get_lines()
                    if line.get_label().startswith('Player')]
    assert len(player_lines) == contest.n


def test_trajectory_plot_uses_the_actual_trajectory(converged_run):
    contest, result = converged_run

    fig = plot_convergence_trajectory(result, contest)
    line = fig.axes[0].get_lines()[0]

    assert line.get_ydata() == pytest.approx(result['trajectory'][:, 0])


def test_trajectory_plot_marks_the_analytical_equilibrium(converged_run):
    contest, result = converged_run
    x_star = contest.analytical_symmetric_equilibrium()

    fig = plot_convergence_trajectory(result, contest)

    marked = [line.get_ydata()[0] for line in fig.axes[0].get_lines()
              if 'Nash' in line.get_label()]
    assert marked == pytest.approx([x_star])


def test_trajectory_plot_omits_the_marker_without_a_closed_form():
    """r != 1 has no closed form, so there is no equilibrium line to draw."""
    contest = TullockContest(n=2, r=2.0, valuations=[10.0, 10.0])
    result = run_synchronous(contest, np.array([1.0, 2.0]), max_iterations=50)

    fig = plot_convergence_trajectory(result, contest)

    assert not any('Nash' in line.get_label()
                   for line in fig.axes[0].get_lines())


def test_trajectory_plot_honours_a_custom_title(converged_run):
    contest, result = converged_run

    fig = plot_convergence_trajectory(result, contest, title="Custom title")

    assert fig.axes[0].get_title() == "Custom title"


def test_trajectory_plot_writes_a_file_when_asked(converged_run, tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr("src.plots.FIGURE_DIR", tmp_path)
    contest, result = converged_run

    plot_convergence_trajectory(result, contest, filename="fig97_test.png")

    assert (tmp_path / "fig97_test.png").exists()


# --- phase portrait -------------------------------------------------------

def test_phase_portrait_rejects_more_than_two_players():
    contest = TullockContest(n=3, r=1.0, valuations=[10.0] * 3)

    with pytest.raises(ValueError, match="needs 2 players"):
        plot_phase_portrait_2player(contest)


def test_phase_portrait_field_vanishes_at_the_equilibrium():
    """The arrow at x* is the null vector: nobody wants to move."""
    contest = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0])
    x_star = contest.analytical_symmetric_equilibrium()

    fig = plot_phase_portrait_2player(contest, grid_size=8)
    quiver = fig.axes[0].collections[0]

    positions = quiver.get_offsets()
    displacements = np.column_stack([quiver.U, quiver.V])
    nearest = int(np.argmin(np.linalg.norm(positions - x_star, axis=1)))

    assert np.linalg.norm(displacements[nearest]) < 1.0


def test_phase_portrait_grid_size_controls_the_arrow_count():
    contest = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0])

    fig = plot_phase_portrait_2player(contest, grid_size=6)

    assert len(fig.axes[0].collections[0].get_offsets()) == 36
