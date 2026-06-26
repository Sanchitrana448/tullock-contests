"""Best response solver.

The payoff in own effort stops being concave once r > 1: a local maximum
near zero can coexist with an interior one. These pin the solver against a
brute force global maximum across the r range used in the experiments.
"""

import numpy as np
import pytest

from src.best_response import (EFFORT_MIN, compute_all_best_responses,
                               compute_best_response, effort_ceiling)
from src.contest import TullockContest


def brute_force_best_response(contest, player_index, others, n_points=60_000):
    """Global maximum on a dense grid. Slow, but hard to fool."""
    ceiling = effort_ceiling(contest, player_index)
    grid = np.concatenate([
        np.geomspace(EFFORT_MIN, min(1.0, ceiling), n_points // 4),
        np.linspace(min(1.0, ceiling), ceiling, n_points),
    ])
    payoffs = contest.own_payoff_curve(player_index, others, grid)
    return grid[np.argmax(payoffs)]


@pytest.mark.parametrize("r", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
def test_finds_global_maximum(r):
    rng = np.random.default_rng(11)
    contest_cache = {}

    for _ in range(25):
        n = int(rng.integers(2, 8))
        contest = contest_cache.setdefault(
            n, TullockContest(n=n, r=r, valuations=[10.0] * n))
        others = rng.uniform(0.01, 9.0, size=n - 1)

        solver = compute_best_response(contest, 0, others)
        reference = brute_force_best_response(contest, 0, others)

        solver_payoff = contest.own_payoff_curve(0, others, np.array([solver]))[0]
        reference_payoff = contest.own_payoff_curve(0, others,
                                                    np.array([reference]))[0]

        # Compare payoffs, not locations: near-flat optima make the argmax
        # itself unstable while the achieved payoff is not.
        assert solver_payoff >= reference_payoff - 1e-6


def test_symmetric_best_response_matches_closed_form():
    """With everyone at x*, each player's best response is x* itself."""
    contest = TullockContest(n=4, r=1.0, valuations=[10.0] * 4)
    x_star = contest.analytical_symmetric_equilibrium()

    br = compute_best_response(contest, 0, np.full(3, x_star))
    assert br == pytest.approx(x_star, abs=1e-6)


def test_higher_valuation_gives_weakly_higher_effort():
    contest = TullockContest(n=2, r=1.0, valuations=[5.0, 15.0])
    rivals = np.array([2.0])

    assert (compute_best_response(contest, 1, rivals)
            > compute_best_response(contest, 0, rivals))


def test_all_best_responses_matches_individual_calls():
    contest = TullockContest(n=4, r=2.0, valuations=[10.0] * 4)
    efforts = np.array([1.0, 2.0, 0.5, 3.0])

    combined = compute_all_best_responses(contest, efforts)
    individual = [compute_best_response(contest, i, np.delete(efforts, i))
                  for i in range(4)]

    assert combined == pytest.approx(individual)


def test_best_response_stays_within_bounds():
    contest = TullockContest(n=2, r=5.0, valuations=[10.0, 10.0])
    br = compute_best_response(contest, 0, np.array([0.001]))

    assert EFFORT_MIN <= br <= effort_ceiling(contest, 0)


def test_ceiling_scales_with_valuation():
    """The search ceiling has to scale with V, not be a fixed constant.

    At V = 1000 the two-player equilibrium is V(n-1)/n^2 = 250, well past
    any ceiling that would be reasonable at V = 10.
    """
    contest = TullockContest(n=2, r=1.0, valuations=[1000.0, 1000.0])
    x_star = contest.analytical_symmetric_equilibrium()

    br = compute_best_response(contest, 0, np.array([x_star]))

    assert br == pytest.approx(x_star, rel=1e-6)
    assert br > 100.0


def test_derived_ceiling_bounds_the_optimum():
    """x <= V^(1/alpha) is sufficient: beyond it the payoff is negative."""
    for alpha in (1.0, 2.0):
        contest = TullockContest(n=3, r=2.0, valuations=[10.0] * 3,
                                 alpha=alpha)
        ceiling = effort_ceiling(contest, 0)
        beyond = contest.own_payoff_curve(
            0, np.array([1.0, 1.0]),
            np.linspace(ceiling * 1.001, ceiling * 5, 200))

        assert np.all(beyond < 0.0)
