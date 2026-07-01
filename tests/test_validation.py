"""Validation against Tullock's (1980) closed form.

For a symmetric contest with linear costs and r = 1 the unique Nash
equilibrium is x* = V(n-1)/n^2. Every update rule has to land on it. This
is what licenses using the simulation where no closed form exists.
"""

import numpy as np
import pytest

from src.contest import TullockContest
from src.dynamics import run_asynchronous, run_inertial, run_synchronous


@pytest.mark.parametrize("n, expected", [
    (2, 2.5),
    (3, 10.0 * 2 / 9),
    (5, 1.6),
])
def test_analytical_formula(n, expected):
    contest = TullockContest(n=n, r=1.0, valuations=[10.0] * n)
    assert contest.analytical_symmetric_equilibrium() == pytest.approx(expected)


@pytest.mark.parametrize("run", [
    run_synchronous,
    lambda c, x0: run_asynchronous(c, x0, seed=42),
    lambda c, x0: run_inertial(c, x0, lam=0.5),
], ids=["synchronous", "asynchronous", "inertial"])
def test_two_player_equilibrium(run):
    contest = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0])
    result = run(contest, np.array([8.0, 1.0]))

    assert result['converged']
    assert result['final_efforts'] == pytest.approx(2.5, abs=1e-4)


def test_three_player_equilibrium():
    contest = TullockContest(n=3, r=1.0, valuations=[10.0] * 3)
    result = run_synchronous(contest, np.array([5.0, 1.0, 3.0]),
                             max_iterations=500)

    assert result['converged']
    assert result['final_efforts'] == pytest.approx(10.0 * 2 / 9, abs=1e-4)


def test_five_player_equilibrium_reached_without_synchronous_updating():
    """x* exists at n = 5 but synchronous updating cannot reach it.

    The spectral radius is about 1.5 here, so the equilibrium repels under
    simultaneous updating. Asynchronous and damped updating still find it.
    """
    contest = TullockContest(n=5, r=1.0, valuations=[10.0] * 5)
    x0 = np.array([5.0, 1.0, 3.0, 2.0, 4.0])

    assert not run_synchronous(contest, x0, max_iterations=1000)['converged']

    for result in (run_asynchronous(contest, x0, max_iterations=1000, seed=42),
                   run_inertial(contest, x0, lam=0.5, max_iterations=1000)):
        assert result['converged']
        assert result['final_efforts'] == pytest.approx(1.6, abs=1e-4)


def test_equilibrium_is_a_fixed_point():
    """At x* nobody wants to move."""
    from src.best_response import compute_all_best_responses

    contest = TullockContest(n=3, r=1.0, valuations=[10.0] * 3)
    x_star = np.full(3, contest.analytical_symmetric_equilibrium())

    assert compute_all_best_responses(contest, x_star) == pytest.approx(x_star,
                                                                        abs=1e-6)


def test_analytical_formula_rejects_cases_it_does_not_cover():
    with pytest.raises(ValueError):
        TullockContest(n=2, r=2.0,
                       valuations=[10.0, 10.0]).analytical_symmetric_equilibrium()
    with pytest.raises(ValueError):
        TullockContest(n=2, r=1.0,
                       valuations=[8.0, 12.0]).analytical_symmetric_equilibrium()
    with pytest.raises(ValueError):
        TullockContest(n=2, r=1.0, valuations=[10.0, 10.0],
                       alpha=2.0).analytical_symmetric_equilibrium()
