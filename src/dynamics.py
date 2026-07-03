"""Best response dynamics under three update rules.

    synchronous   everyone jumps to their best response at once
    asynchronous  players update in turn, seeing the updates before them
    inertial      players move a fraction lam of the way there

All three return the same dict: trajectory, converged, iterations,
final_efforts.

A run counts as converged when the profile it reached is a fixed point of
the best response map. A small last step is not enough on its own: efforts
collapsing towards the lower bound take tiny steps while sitting nowhere
near an equilibrium. Since x = (1-lam)x + lam*BR(x) exactly when x = BR(x),
the same test works for all three rules.
"""

import numpy as np

from src.best_response import compute_best_response, compute_all_best_responses


def _at_fixed_point(contest, efforts, tolerance):
    residual = compute_all_best_responses(contest, efforts) - efforts
    return np.max(np.abs(residual)) < tolerance


def _result(trajectory, converged):
    trajectory = np.array(trajectory)
    return {
        'trajectory': trajectory,
        'converged': converged,
        'iterations': len(trajectory) - 1,
        'final_efforts': trajectory[-1].copy(),
    }


def run_synchronous(contest, initial_efforts, max_iterations=1000,
                    tolerance=1e-6):
    """All players adopt their best response simultaneously each round."""
    efforts = np.array(initial_efforts, dtype=float)
    trajectory = [efforts.copy()]

    for _ in range(max_iterations):
        new_efforts = compute_all_best_responses(contest, efforts)
        max_change = np.max(np.abs(new_efforts - efforts))
        trajectory.append(new_efforts.copy())
        efforts = new_efforts

        if max_change < tolerance and _at_fixed_point(contest, efforts, tolerance):
            return _result(trajectory, True)

    return _result(trajectory, False)


def run_asynchronous(contest, initial_efforts, max_iterations=1000,
                     tolerance=1e-6, random_order=True, seed=None):
    """Players update one at a time. One iteration is a full sweep of all n."""
    rng = np.random.default_rng(seed)
    efforts = np.array(initial_efforts, dtype=float)
    trajectory = [efforts.copy()]

    for _ in range(max_iterations):
        previous = efforts.copy()
        order = rng.permutation(contest.n) if random_order else range(contest.n)

        for i in order:
            efforts[i] = compute_best_response(contest, i, np.delete(efforts, i))

        trajectory.append(efforts.copy())

        if (np.max(np.abs(efforts - previous)) < tolerance
                and _at_fixed_point(contest, efforts, tolerance)):
            return _result(trajectory, True)

    return _result(trajectory, False)


def run_inertial(contest, initial_efforts, lam=0.5, max_iterations=1000,
                 tolerance=1e-6):
    """x(t+1) = (1 - lam) * x(t) + lam * BR(x(t)).

    lam = 1 is the synchronous rule; smaller values damp the adjustment.
    """
    if not 0 < lam <= 1:
        raise ValueError(f"lam must be in (0, 1], got {lam}")

    efforts = np.array(initial_efforts, dtype=float)
    trajectory = [efforts.copy()]

    for _ in range(max_iterations):
        target = compute_all_best_responses(contest, efforts)
        new_efforts = (1 - lam) * efforts + lam * target
        max_change = np.max(np.abs(new_efforts - efforts))
        trajectory.append(new_efforts.copy())
        efforts = new_efforts

        if max_change < tolerance and _at_fixed_point(contest, efforts, tolerance):
            return _result(trajectory, True)

    return _result(trajectory, False)
