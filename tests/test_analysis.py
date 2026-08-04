"""Statistics, stability and the full report sweep.

Each of these checks against something known independently of the
simulation: dissipation against the textbook (n-1)/n, the spectral radius
against its exact symmetric value (n-2)/2, and the equilibrium check
against a cell whose converged runs land on different equilibria.
"""

import numpy as np
import pytest

from src.analysis import (EQUILIBRIUM_TOLERANCE, benchmark_deviation,
                          best_response_residual, convergence_statistics,
                          eigenvalue_stability, generate_full_report,
                          print_report_summary, rent_dissipation_analysis)
from src.contest import TullockContest


def symmetric_contest(n, r=1.0, V=10.0):
    return TullockContest(n=n, r=r, valuations=[V] * n)


def exact_spectral_radius(n):
    """Spectral radius of the BR Jacobian at the symmetric equilibrium.

    With r = 1 and linear costs, BR_i = sqrt(V*S_i) - S_i where S_i is the
    sum of rival efforts, so dBR_i/dx_j = (2-n) / (2(n-1)) for every j != i
    and zero on the diagonal. That matrix has eigenvalues (n-1)a and -a,
    giving a spectral radius of (n-2)/2.
    """
    return (n - 2) / 2


# --- convergence statistics ----------------------------------------------

def test_statistics_describe_converged_runs_only():
    contest = symmetric_contest(2)
    stats = convergence_statistics(contest, n_seeds=8, max_iterations=200)

    assert stats['convergence_rate'] == 1.0
    assert stats['n_converged'] + stats['n_diverged'] == 8
    assert stats['mean_final_efforts'] == pytest.approx(
        np.full(2, contest.analytical_symmetric_equilibrium()), abs=1e-4)


def test_statistics_report_nan_when_nothing_converges():
    contest = symmetric_contest(20, r=5.0)
    stats = convergence_statistics(contest, n_seeds=3, max_iterations=20)

    if stats['n_converged'] == 0:
        assert np.isnan(stats['mean_iterations'])
        assert np.all(np.isnan(stats['mean_final_efforts']))


def test_benchmark_deviation_is_none_without_a_closed_form():
    contest = symmetric_contest(3, r=2.0)
    stats = convergence_statistics(contest, n_seeds=3, max_iterations=100)

    assert benchmark_deviation(contest, stats)['analytical_x_star'] is None


def test_benchmark_deviation_recovers_the_closed_form():
    contest = symmetric_contest(3)
    stats = convergence_statistics(contest, n_seeds=5, max_iterations=200)
    benchmark = benchmark_deviation(contest, stats)

    assert benchmark['relative_error_pct'] < 0.1


# --- rent dissipation ----------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_dissipation_matches_the_textbook_share(n):
    """Total effort at x* is V(n-1)/n, so the share of the prize is (n-1)/n.

    There is one prize, not n of them, so the comparison is against the
    prize value rather than the sum of the valuations.
    """
    contest = symmetric_contest(n)
    x_star = contest.analytical_symmetric_equilibrium()
    stats = {'mean_final_efforts': np.full(n, x_star)}

    result = rent_dissipation_analysis(contest, stats)

    assert result['dissipation_ratio'] == pytest.approx((n - 1) / n, rel=1e-9)
    assert not result['over_dissipation']


def test_dissipation_share_rises_with_the_field():
    ratios = [
        rent_dissipation_analysis(
            symmetric_contest(n),
            {'mean_final_efforts': np.full(
                n, symmetric_contest(n).analytical_symmetric_equilibrium())},
        )['dissipation_ratio']
        for n in (2, 3, 5, 10, 20)
    ]

    assert ratios == sorted(ratios)


def test_dissipation_flags_over_dissipation():
    contest = symmetric_contest(2)
    stats = {'mean_final_efforts': np.array([8.0, 8.0])}

    assert rent_dissipation_analysis(contest, stats)['over_dissipation']


def test_per_player_shares_sum_to_one():
    contest = TullockContest(n=3, r=1.0, valuations=[5.0, 10.0, 15.0])
    stats = {'mean_final_efforts': np.array([1.0, 2.0, 3.0])}

    shares = rent_dissipation_analysis(contest, stats)['per_player_share']

    assert shares.sum() == pytest.approx(1.0)


# --- Jacobian estimation -------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 10, 20])
def test_spectral_radius_matches_the_exact_value(n):
    contest = symmetric_contest(n)
    x_star = np.full(n, contest.analytical_symmetric_equilibrium())

    rho = eigenvalue_stability(contest, x_star)['spectral_radius']

    assert rho == pytest.approx(exact_spectral_radius(n), abs=2e-3)


def test_stability_boundary_sits_at_four_players():
    """rho = (n-2)/2 crosses 1 at n = 4, which is the reported threshold."""
    for n, expected in ((3, True), (5, False)):
        contest = symmetric_contest(n)
        x_star = np.full(n, contest.analytical_symmetric_equilibrium())

        assert eigenvalue_stability(contest, x_star)['is_stable'] is expected


def test_jacobian_is_square_and_finite():
    contest = symmetric_contest(3)
    x_star = np.full(3, contest.analytical_symmetric_equilibrium())

    jacobian = eigenvalue_stability(contest, x_star)['jacobian']

    assert jacobian.shape == (3, 3)
    assert np.all(np.isfinite(jacobian))


def test_central_differences_beat_one_sided_ones():
    """The estimate must not degrade when the step is halved.

    One-sided differences divide solver noise by delta; the symptom is an
    estimate that moves with delta rather than settling.
    """
    contest = symmetric_contest(10)
    x_star = np.full(10, contest.analytical_symmetric_equilibrium())
    exact = exact_spectral_radius(10)

    coarse = eigenvalue_stability(contest, x_star, delta=1e-4)
    fine = eigenvalue_stability(contest, x_star, delta=5e-5)

    assert abs(coarse['spectral_radius'] - exact) < 2e-3
    assert abs(fine['spectral_radius'] - exact) < 2e-3


# --- linearise only at genuine equilibria --------------------------------

def test_residual_is_zero_at_an_equilibrium():
    contest = symmetric_contest(3)
    x_star = np.full(3, contest.analytical_symmetric_equilibrium())

    assert best_response_residual(contest, x_star) < 1e-6


def test_residual_is_large_away_from_an_equilibrium():
    contest = symmetric_contest(3)

    assert best_response_residual(contest, np.array([5.0, 0.5, 0.5])) > 1e-2


def test_report_withholds_stability_where_the_mean_is_not_an_equilibrium():
    """n = 3, r = 1.5: the converged runs reach different equilibria.

    Their mean has a residual of about 1.76, so a spectral radius computed
    there would describe no equilibrium of this contest.
    """
    report = generate_full_report([1.5], [3], n_seeds=10, max_iterations=200)
    cell = report[(3, 1.5)]

    assert cell['stats']['n_converged'] > 0
    assert cell['eigenvalue']['residual'] > EQUILIBRIUM_TOLERANCE
    assert cell['eigenvalue']['is_equilibrium'] is False
    assert cell['eigenvalue']['is_stable'] is None
    assert np.isnan(cell['eigenvalue']['spectral_radius'])


def test_report_gives_stability_where_the_mean_is_an_equilibrium():
    report = generate_full_report([1.0], [2], n_seeds=5, max_iterations=200)
    cell = report[(2, 1.0)]

    assert cell['eigenvalue']['is_equilibrium'] is True
    assert cell['eigenvalue']['residual'] < EQUILIBRIUM_TOLERANCE
    assert cell['eigenvalue']['spectral_radius'] == pytest.approx(0.0, abs=2e-3)


def test_report_covers_every_cell():
    r_values, n_values = [1.0, 2.0], [2, 3]
    report = generate_full_report(r_values, n_values, n_seeds=3,
                                  max_iterations=100)

    assert set(report) == {(n, r) for n in n_values for r in r_values}
    for cell in report.values():
        assert set(cell) == {'stats', 'benchmark', 'dissipation', 'eigenvalue'}


def test_summary_prints_na_rather_than_a_misleading_number(capsys):
    report = generate_full_report([1.5], [3], n_seeds=10, max_iterations=200)
    print_report_summary(report, [1.5], [3])

    line = [row for row in capsys.readouterr().out.splitlines()
            if row.strip().startswith('1.5')][0]

    assert line.count('N/A') == 2
