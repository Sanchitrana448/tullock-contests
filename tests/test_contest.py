"""Contest model: payoffs, success function, input validation."""

import numpy as np
import pytest

from src.contest import TullockContest


def test_winning_probabilities_sum_to_one():
    contest = TullockContest(n=4, r=1.5, valuations=[10.0] * 4)
    probs = contest.winning_probability([1.0, 2.0, 3.0, 4.0])

    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs > 0)


def test_equal_efforts_give_equal_chances():
    contest = TullockContest(n=3, r=2.0, valuations=[10.0] * 3)
    assert contest.winning_probability([2.0] * 3) == pytest.approx(1 / 3)


def test_all_zero_efforts_split_the_prize_evenly():
    contest = TullockContest(n=4, r=1.0, valuations=[10.0] * 4)
    assert contest.winning_probability([0.0] * 4) == pytest.approx(0.25)


def test_higher_r_sharpens_the_advantage_of_leading():
    efforts = [3.0, 1.0]
    leader_share = [
        TullockContest(n=2, r=r, valuations=[10.0, 10.0])
        .winning_probability(efforts)[0]
        for r in (0.5, 1.0, 3.0)
    ]

    assert leader_share == sorted(leader_share)


def test_payoff_is_probability_times_value_minus_cost():
    contest = TullockContest(n=2, r=1.0, valuations=[10.0, 20.0])
    efforts = [2.0, 3.0]

    expected = 10.0 * (2.0 / 5.0) - 2.0
    assert contest.payoff(efforts, 0) == pytest.approx(expected)


def test_all_payoffs_matches_payoff_per_player():
    contest = TullockContest(n=3, r=1.5, valuations=[8.0, 10.0, 12.0])
    efforts = [1.0, 2.0, 3.0]

    assert contest.all_payoffs(efforts) == pytest.approx(
        [contest.payoff(efforts, i) for i in range(3)])


def test_own_payoff_curve_matches_payoff():
    """The vectorised curve and the scalar payoff must not drift apart."""
    rng = np.random.default_rng(3)

    for _ in range(20):
        n = int(rng.integers(2, 8))
        contest = TullockContest(n=n, r=float(rng.uniform(0.5, 5.0)),
                                 valuations=[10.0] * n)
        others = rng.uniform(0.01, 9.0, size=n - 1)
        candidates = rng.uniform(1e-6, 50.0, size=15)

        curve = contest.own_payoff_curve(0, others, candidates)
        pointwise = [contest.payoff(np.insert(others, 0, x), 0)
                     for x in candidates]

        assert curve == pytest.approx(pointwise, rel=1e-12, abs=1e-12)


def test_convex_costs_reduce_payoff_at_high_effort():
    linear = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0], alpha=1.0)
    convex = TullockContest(n=2, r=1.0, valuations=[10.0, 10.0], alpha=2.0)

    assert convex.payoff([4.0, 4.0], 0) < linear.payoff([4.0, 4.0], 0)


@pytest.mark.parametrize("kwargs", [
    dict(n=1, r=1.0, valuations=[10.0]),
    dict(n=2, r=0.0, valuations=[10.0, 10.0]),
    dict(n=2, r=-1.0, valuations=[10.0, 10.0]),
    dict(n=3, r=1.0, valuations=[10.0, 10.0]),
    dict(n=2, r=1.0, valuations=[10.0, 0.0]),
    dict(n=2, r=1.0, valuations=[10.0, -5.0]),
    dict(n=2, r=1.0, valuations=[10.0, 10.0], alpha=0.5),
])
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        TullockContest(**kwargs)
