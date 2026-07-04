"""Update rule semantics.

The properties that matter here are that a run stays within its budget,
that each rule uses only its own steps, and that a rule which fails to
converge says so instead of falling back to another one.
"""

import numpy as np
import pytest

from src.best_response import compute_all_best_responses
from src.contest import TullockContest
from src.dynamics import run_asynchronous, run_inertial, run_synchronous

# r = 3 with two symmetric players is the knife-edge case: synchronous
# dynamics oscillate here instead of settling.
OSCILLATING = TullockContest(n=2, r=3.0, valuations=[10.0, 10.0])


@pytest.mark.parametrize("run", [
    lambda c, x0, m: run_synchronous(c, x0, max_iterations=m),
    lambda c, x0, m: run_asynchronous(c, x0, max_iterations=m, seed=1),
    lambda c, x0, m: run_inertial(c, x0, lam=0.5, max_iterations=m),
], ids=["synchronous", "asynchronous", "inertial"])
def test_respects_iteration_budget(run):
    result = run(OSCILLATING, np.array([4.0, 1.0]), 20)

    assert result['iterations'] <= 20
    assert len(result['trajectory']) <= 21


def test_synchronous_trajectory_is_purely_synchronous():
    """Every step must be the best response to the step before it."""
    result = run_synchronous(OSCILLATING, np.array([4.0, 1.0]), max_iterations=15)
    trajectory = result['trajectory']

    for before, after in zip(trajectory, trajectory[1:]):
        assert compute_all_best_responses(OSCILLATING, before) == pytest.approx(after)


def test_non_convergence_is_reported_not_rescued():
    result = run_synchronous(OSCILLATING, np.array([4.0, 1.0]), max_iterations=200)

    assert not result['converged']
    assert result['iterations'] == 200


def test_inertial_with_lam_one_is_synchronous():
    contest = TullockContest(n=3, r=1.0, valuations=[10.0] * 3)
    x0 = np.array([1.0, 4.0, 2.0])

    inertial = run_inertial(contest, x0, lam=1.0, max_iterations=50)
    synchronous = run_synchronous(contest, x0, max_iterations=50)

    assert inertial['trajectory'] == pytest.approx(synchronous['trajectory'])


def test_damping_does_not_rescue_r_above_the_existence_bound():
    """No pure-strategy equilibrium exists at r = 3 with two players.

    Two-player Tullock contests have a pure equilibrium only for
    r <= n/(n-1) = 2, so there is nothing for any update rule to converge
    to. Damping changes the shape of the oscillation but cannot remove it.
    """
    x0 = np.array([4.0, 1.0])

    assert not run_synchronous(OSCILLATING, x0, max_iterations=400)['converged']
    for lam in (0.5, 0.1):
        assert not run_inertial(OSCILLATING, x0, lam=lam,
                                max_iterations=400)['converged']


def test_damping_rescues_an_equilibrium_synchronous_updating_overshoots():
    """Where an equilibrium does exist, damping reaches what jumping cannot."""
    contest = TullockContest(n=5, r=1.0, valuations=[10.0] * 5)
    x0 = np.array([5.0, 1.0, 3.0, 2.0, 4.0])

    assert not run_synchronous(contest, x0, max_iterations=600)['converged']
    assert run_inertial(contest, x0, lam=0.5, max_iterations=600)['converged']


def test_collapse_to_the_lower_bound_is_not_convergence():
    """A tiny step near zero is not evidence of an equilibrium.

    At r = 5 with ten players best responses collapse to the bottom of the
    search interval within two rounds, so every step after that is tiny. A
    step size test alone would call that convergence even though deviating
    to x = 0.0001 pays about 10 rather than about 0.
    """
    contest = TullockContest(n=10, r=5.0, valuations=[10.0] * 10)
    rng = np.random.default_rng(0)
    result = run_synchronous(contest, rng.uniform(0.1, 8.0, size=10),
                             max_iterations=300)

    assert not result['converged']


def test_convergence_implies_a_genuine_fixed_point():
    """Anything reported as converged must survive a residual check."""
    for contest, x0 in [
        (TullockContest(n=2, r=1.0, valuations=[10.0, 10.0]), [8.0, 1.0]),
        (TullockContest(n=3, r=1.0, valuations=[10.0] * 3), [5.0, 1.0, 3.0]),
        (TullockContest(n=2, r=1.0, valuations=[4.0, 16.0]), [1.0, 6.0]),
    ]:
        result = run_synchronous(contest, np.array(x0), max_iterations=2000)
        assert result['converged']

        residual = compute_all_best_responses(contest, result['final_efforts'])
        assert residual == pytest.approx(result['final_efforts'], abs=1e-6)


def test_result_fields_are_consistent():
    contest = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0])
    result = run_synchronous(contest, np.array([8.0, 1.0]))

    assert result['iterations'] == len(result['trajectory']) - 1
    assert result['final_efforts'] == pytest.approx(result['trajectory'][-1])


def test_starting_at_equilibrium_converges_immediately():
    contest = TullockContest(n=3, r=1.0, valuations=[10.0] * 3)
    x_star = np.full(3, contest.analytical_symmetric_equilibrium())

    assert run_synchronous(contest, x_star)['iterations'] == 1


def test_initial_efforts_are_not_mutated():
    contest = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0])
    x0 = np.array([8.0, 1.0])

    run_asynchronous(contest, x0, max_iterations=5, seed=0)
    assert x0 == pytest.approx([8.0, 1.0])


def test_inertial_rejects_invalid_lambda():
    contest = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0])

    for lam in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            run_inertial(contest, np.array([1.0, 1.0]), lam=lam)
