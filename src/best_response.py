"""Best response of a single player to a fixed profile of rival efforts."""

import numpy as np
from scipy.optimize import minimize_scalar

# Strictly positive: the success function is undefined at an all-zero profile.
EFFORT_MIN = 1e-8
EFFORT_MAX = 100.0


def compute_best_response(contest, player_index, others_efforts,
                          x_min=EFFORT_MIN, x_max=EFFORT_MAX):
    """Effort maximising player i's payoff with rivals' efforts held fixed."""
    others_efforts = np.asarray(others_efforts, dtype=float)

    def negative_payoff(x):
        efforts = np.insert(others_efforts, player_index, x)
        return -contest.payoff(efforts, player_index)

    result = minimize_scalar(negative_payoff, bounds=(x_min, x_max),
                             method="bounded", options={"xatol": 1e-10})
    return float(result.x)


def compute_all_best_responses(contest, efforts):
    """Best response for every player against the current profile."""
    return np.array([
        compute_best_response(contest, i, np.delete(efforts, i))
        for i in range(contest.n)
    ])
