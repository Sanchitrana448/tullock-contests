"""Convergence statistics over repeated runs, and equilibrium stability."""

import numpy as np

from src.best_response import EFFORT_MIN, compute_all_best_responses
from src.contest import TullockContest
from src.dynamics import run_synchronous


def convergence_statistics(contest, n_seeds=50, max_iterations=500,
                           tolerance=1e-6, seed=0):
    """Run synchronous dynamics from many random starts and summarise.

    Starting efforts are drawn uniformly from [0.1, 0.8V]^n. Runs that hit
    the iteration cap count towards the convergence rate only; averaging in
    their final position would be meaningless.
    """
    rng = np.random.default_rng(seed)
    V = contest.valuations.max()

    converged_iters = []
    final_efforts = []

    for _ in range(n_seeds):
        initial = rng.uniform(0.1, V * 0.8, size=contest.n)
        result = run_synchronous(contest, initial,
                                 max_iterations=max_iterations,
                                 tolerance=tolerance)
        if result['converged']:
            converged_iters.append(result['iterations'])
            final_efforts.append(result['final_efforts'])

    n_converged = len(converged_iters)

    if n_converged:
        iters = np.asarray(converged_iters, dtype=float)
        finals = np.asarray(final_efforts)
        mean_iterations = iters.mean()
        std_iterations = iters.std()
        min_iterations = iters.min()
        max_iterations_val = iters.max()
        mean_final_efforts = finals.mean(axis=0)
        std_final_efforts = finals.std(axis=0)
    else:
        mean_iterations = std_iterations = np.nan
        min_iterations = max_iterations_val = np.nan
        mean_final_efforts = np.full(contest.n, np.nan)
        std_final_efforts = np.full(contest.n, np.nan)

    return {
        'convergence_rate': n_converged / n_seeds,
        'mean_iterations': mean_iterations,
        'std_iterations': std_iterations,
        'min_iterations': min_iterations,
        'max_iterations_val': max_iterations_val,
        'mean_final_efforts': mean_final_efforts,
        'std_final_efforts': std_final_efforts,
        'n_converged': n_converged,
        'n_diverged': n_seeds - n_converged,
    }


def benchmark_deviation(contest, stats):
    """Gap between the simulated equilibrium and Tullock's closed form."""
    simulated_mean = np.mean(stats['mean_final_efforts'])

    try:
        x_star = contest.analytical_symmetric_equilibrium()
    except ValueError:
        return {
            'analytical_x_star': None,
            'simulated_mean': simulated_mean,
            'absolute_error': None,
            'relative_error_pct': None,
        }

    absolute_error = abs(simulated_mean - x_star)
    return {
        'analytical_x_star': x_star,
        'simulated_mean': simulated_mean,
        'absolute_error': absolute_error,
        'relative_error_pct': absolute_error / x_star * 100,
    }


def rent_dissipation_analysis(contest, stats):
    """Share of the prize burned as effort at the simulated equilibrium.

    There is one prize; the valuations are n views of it, not n prizes, so
    total effort is compared against the mean valuation rather than the sum.
    A ratio above 1 is over-dissipation.
    """
    efforts = stats['mean_final_efforts']
    total_effort = np.sum(efforts)
    prize_value = float(np.mean(contest.valuations))

    return {
        'total_effort': total_effort,
        'prize_value': prize_value,
        'dissipation_ratio': total_effort / prize_value,
        'over_dissipation': total_effort > prize_value,
        'per_player_effort': efforts,
        'per_player_share': (efforts / total_effort if total_effort > 0
                             else np.full(contest.n, np.nan)),
    }


def best_response_residual(contest, efforts):
    """max_i |BR_i(x) - x_i|. Zero exactly at a Nash equilibrium."""
    efforts = np.asarray(efforts, dtype=float)
    if np.any(~np.isfinite(efforts)):
        return np.inf
    return float(np.max(np.abs(compute_all_best_responses(contest, efforts)
                               - efforts)))


def eigenvalue_stability(contest, equilibrium_efforts, delta=1e-4):
    """Local stability from the spectral radius of the best response Jacobian.

    Below 1 the map is a local contraction and synchronous dynamics are
    locally attracting, which is the Szidarovszky and Okuguchi (1997)
    condition in computable form.

    J is estimated by central differences. One-sided differences divide the
    solver's own numerical noise by delta, which was enough to move the
    third decimal place. equilibrium_efforts must actually be a fixed point;
    check with best_response_residual first.
    """
    n = contest.n
    x0 = np.array(equilibrium_efforts, dtype=float)

    jacobian = np.zeros((n, n))
    for j in range(n):
        forward = x0.copy()
        backward = x0.copy()
        forward[j] += delta
        backward[j] = max(x0[j] - delta, EFFORT_MIN)  # efforts stay positive
        step = forward[j] - backward[j]
        jacobian[:, j] = (compute_all_best_responses(contest, forward)
                          - compute_all_best_responses(contest, backward)) / step

    eigenvalues = np.linalg.eigvals(jacobian)
    spectral_radius = np.max(np.abs(eigenvalues))

    return {
        'eigenvalues': eigenvalues,
        'spectral_radius': float(spectral_radius),
        'is_stable': bool(spectral_radius < 1.0),
        'jacobian': jacobian,
    }


EQUILIBRIUM_TOLERANCE = 1e-4


def generate_full_report(r_values, n_values, V=10.0, n_seeds=30,
                         max_iterations=500,
                         equilibrium_tolerance=EQUILIBRIUM_TOLERANCE):
    """Convergence, dissipation and stability for every (n, r) cell.

    The mean of the converged runs is only an equilibrium if those runs all
    reached the same one, so each cell is tested against the best response
    map before it is linearised and reported as unavailable if it fails.
    """
    report = {}

    unavailable = {
        'eigenvalues': None,
        'spectral_radius': np.nan,
        'is_stable': None,
        'jacobian': None,
        'residual': np.nan,
        'is_equilibrium': False,
    }

    for n in n_values:
        for r in r_values:
            contest = TullockContest(n=n, r=r, valuations=[V] * n)
            stats = convergence_statistics(contest, n_seeds=n_seeds,
                                           max_iterations=max_iterations)

            if stats['n_converged']:
                mean_profile = stats['mean_final_efforts']
                residual = best_response_residual(contest, mean_profile)

                if residual <= equilibrium_tolerance:
                    eigenvalue = eigenvalue_stability(contest, mean_profile)
                    eigenvalue['residual'] = residual
                    eigenvalue['is_equilibrium'] = True
                else:
                    eigenvalue = dict(unavailable, residual=residual)
            else:
                eigenvalue = dict(unavailable)

            report[(n, r)] = {
                'stats': stats,
                'benchmark': benchmark_deviation(contest, stats),
                'dissipation': rent_dissipation_analysis(contest, stats),
                'eigenvalue': eigenvalue,
            }

    return report


def print_report_summary(report, r_values, n_values):
    """Print the report as one table per player count.

    N/A means there is nothing to report: either nothing converged, or the
    converged runs reached different equilibria whose mean is not one.
    """
    for n in n_values:
        print(f"\nn = {n} players")
        print("-" * 75)
        print(f"{'r':>6} | {'Conv%':>6} | {'MeanIter':>9} | "
              f"{'TotalEffort':>12} | {'SpectralR':>10} | {'Stable':>7}")
        print("-" * 75)

        for r in r_values:
            cell = report[(n, r)]
            rate = cell['stats']['convergence_rate']

            if cell['stats']['n_converged']:
                mean_iter = f"{cell['stats']['mean_iterations']:.1f}"
                total_effort = f"{cell['dissipation']['total_effort']:.4f}"
            else:
                mean_iter = total_effort = "N/A"

            if cell['eigenvalue']['is_stable'] is None:
                rho = stable = "N/A"
            else:
                rho = f"{cell['eigenvalue']['spectral_radius']:.3f}"
                stable = "Yes" if cell['eigenvalue']['is_stable'] else "No"

            print(f"{r:>6.1f} | {rate * 100:>5.0f}% | {mean_iter:>9} | "
                  f"{total_effort:>12} | {rho:>10} | {stable:>7}")
