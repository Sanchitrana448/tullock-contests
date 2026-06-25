"""Best response of a single player to a fixed profile of rival efforts."""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar

# Strictly positive: the success function is undefined at an all-zero profile.
EFFORT_MIN = 1e-8


def effort_ceiling(contest, player_index):
    """Upper bound on player i's best response.

    A best response never loses money, so p_i * V_i - x^alpha >= 0 at the
    optimum, and p_i <= 1 gives x <= V_i^(1/alpha).
    """
    return float(contest.valuations[player_index] ** (1.0 / contest.alpha))


@lru_cache(maxsize=64)
def _scan_grid(x_min, x_max, n_log=160, n_linear=160):
    """Candidate efforts used to bracket the global maximum."""
    return np.unique(np.concatenate([
        np.geomspace(x_min, x_max, n_log),
        np.linspace(x_min, x_max, n_linear),
    ]))


def compute_best_response(contest, player_index, others_efforts,
                          x_min=EFFORT_MIN, x_max=None):
    """Effort maximising player i's payoff with rivals' efforts held fixed.

    For r > 1 the payoff is not concave in own effort: a local maximum near
    zero can sit alongside an interior one, and a unimodal search returns
    whichever it happens to land in. So the interval is scanned first to
    find the right basin, then refined inside it.
    """
    others_efforts = np.asarray(others_efforts, dtype=float)
    if x_max is None:
        x_max = effort_ceiling(contest, player_index)

    grid = _scan_grid(x_min, x_max)
    payoffs = contest.own_payoff_curve(player_index, others_efforts, grid)
    best = int(np.argmax(payoffs))

    lower = grid[best - 1] if best > 0 else grid[0]
    upper = grid[best + 1] if best < len(grid) - 1 else grid[-1]
    if upper <= lower:
        return float(grid[best])

    # Written out as a scalar rather than going through contest.payoff, which
    # would rebuild the whole effort vector on each of Brent's evaluations.
    rivals_powered = float(np.sum(others_efforts ** contest.r))
    valuation = float(contest.valuations[player_index])
    r, alpha, n = contest.r, contest.alpha, contest.n

    def negative_payoff(x):
        own_powered = x ** r
        total = own_powered + rivals_powered
        probability = own_powered / total if total > 0 else 1.0 / n
        return -(probability * valuation - x ** alpha)

    refined = minimize_scalar(negative_payoff, bounds=(lower, upper),
                              method="bounded", options={"xatol": 1e-10})

    if -refined.fun >= payoffs[best]:
        return float(refined.x)
    return float(grid[best])


def compute_all_best_responses(contest, efforts):
    """Best response for every player against the current profile."""
    return np.array([
        compute_best_response(contest, i, np.delete(efforts, i))
        for i in range(contest.n)
    ])
