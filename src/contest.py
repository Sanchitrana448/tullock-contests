"""Tullock contest model.

n players choose efforts x_i >= 0. Player i wins with probability
x_i^r / sum_j x_j^r and pays a cost x_i^alpha.
"""

import numpy as np


class TullockContest:
    """A Tullock contest between n players.

    n           number of players (>= 2)
    r           decisiveness. Low r is lottery-like, high r winner-take-all.
    valuations  prize value V_i for each player
    alpha       cost exponent (1 = linear, 2 = quadratic)
    """

    def __init__(self, n, r, valuations, alpha=1.0):
        valuations = np.asarray(valuations, dtype=float)

        if n < 2:
            raise ValueError(f"need at least 2 players, got n={n}")
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
        if valuations.shape != (n,):
            raise ValueError(f"expected {n} valuations, got {valuations.shape[0]}")
        if np.any(valuations <= 0):
            raise ValueError("valuations must be positive")
        if alpha < 1:
            raise ValueError(f"alpha must be >= 1, got {alpha}")

        self.n = n
        self.r = r
        self.valuations = valuations
        self.alpha = alpha

    def winning_probability(self, efforts):
        """Winning probability for each player."""
        efforts = np.asarray(efforts, dtype=float)

        # The success function is 0/0 if nobody exerts any effort.
        if np.all(efforts == 0):
            return np.full(self.n, 1.0 / self.n)

        powered = efforts ** self.r
        return powered / powered.sum()

    def cost(self, effort):
        return effort ** self.alpha

    def payoff(self, efforts, player_index):
        """Payoff to one player."""
        probs = self.winning_probability(efforts)
        return (probs[player_index] * self.valuations[player_index]
                - self.cost(efforts[player_index]))

    def all_payoffs(self, efforts):
        efforts = np.asarray(efforts, dtype=float)
        probs = self.winning_probability(efforts)
        return probs * self.valuations - self.cost(efforts)

    def own_payoff_curve(self, player_index, others_efforts, own_efforts):
        """Payoff to player i over many candidate own efforts at once.

        Same result as calling payoff() in a loop, but vectorised. The best
        response solver evaluates a few hundred candidates per call.
        """
        own = np.asarray(own_efforts, dtype=float)
        rivals_powered = np.sum(np.asarray(others_efforts, dtype=float) ** self.r)
        own_powered = own ** self.r
        total = own_powered + rivals_powered

        probs = np.where(total > 0, own_powered / total, 1.0 / self.n)
        return probs * self.valuations[player_index] - own ** self.alpha

    def analytical_symmetric_equilibrium(self):
        """Tullock (1980) closed form, x* = V(n-1)/n^2.

        Only valid for equal valuations, r = 1 and linear costs.
        """
        if not np.allclose(self.valuations, self.valuations[0]):
            raise ValueError("closed form needs equal valuations")
        if self.r != 1.0:
            raise ValueError("closed form needs r = 1")
        if self.alpha != 1.0:
            raise ValueError("closed form needs alpha = 1")

        V = self.valuations[0]
        return V * (self.n - 1) / self.n ** 2
