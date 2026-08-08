"""Individual configurations quoted in Chapter 4 that are not part of a sweep.

Three of them: the n = 4 row of Table 4.3, sitting on the stability
boundary; recovery of the n = 5 equilibrium by the non-synchronous rules;
and the spectral radius at the asymmetric two-player equilibrium of Section
4.4. Also the solver comparison in Section 3.2.

Same conventions as the experiments (V = 10, starts drawn from [0.1, 8.0],
seed 0).
"""

import numpy as np

from common import DATA_DIR, banner, save_json
from scipy.optimize import minimize_scalar

from src.analysis import best_response_residual, eigenvalue_stability
from src.best_response import (EFFORT_MIN, compute_best_response,
                               effort_ceiling)
from src.contest import TullockContest
from src.dynamics import run_asynchronous, run_inertial, run_synchronous

V = 10.0
MAX_ITERS = 3000
LAMBDA = 0.5
LAMBDAS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
ASYNC_SEED = 42

banner("Supplementary checks")

results = {}

# --- the stability boundary, and equilibria the synchronous rule misses ---

print("\nPart 1: equilibrium recovery under each update rule (r = 1.0)")
print(f"{'n':>4} | {'rho':>7} | {'sync':>6} | {'async':>6} | "
      f"{'inert 0.5':>9} | {'best lambda':>11} | {'its':>5}")
print("-" * 68)

recovery = {}
for n in (4, 5, 10, 20):
    contest = TullockContest(n=n, r=1.0, valuations=[V] * n)
    x_star = np.full(n, contest.analytical_symmetric_equilibrium())
    rho = eigenvalue_stability(contest, x_star)['spectral_radius']

    rng = np.random.default_rng(0)
    initial = rng.uniform(0.1, V * 0.8, size=n)

    runs = {
        'sync': run_synchronous(contest, initial.copy(),
                                max_iterations=MAX_ITERS),
        'async': run_asynchronous(contest, initial.copy(),
                                  max_iterations=MAX_ITERS, seed=ASYNC_SEED),
        'inertial': run_inertial(contest, initial.copy(), lam=LAMBDA,
                                 max_iterations=MAX_ITERS),
    }

    # How much damping is needed grows with the spectral radius, so record
    # the largest lambda that still converges rather than a single value.
    inertial_sweep = {}
    largest_working_lambda = None
    for lam in LAMBDAS:
        run = run_inertial(contest, initial.copy(), lam=lam,
                           max_iterations=MAX_ITERS)
        inertial_sweep[lam] = {'converged': run['converged'],
                               'iterations': run['iterations']}
        if run['converged'] and largest_working_lambda is None:
            largest_working_lambda = lam

    def cell(run):
        return f"{run['iterations']}" if run['converged'] else "no"

    best = (f"{largest_working_lambda}" if largest_working_lambda
            else "none")
    best_its = (inertial_sweep[largest_working_lambda]['iterations']
                if largest_working_lambda else 0)
    print(f"{n:>4} | {rho:>7.3f} | {cell(runs['sync']):>6} | "
          f"{cell(runs['async']):>6} | {cell(runs['inertial']):>9} | "
          f"{best:>11} | {best_its if largest_working_lambda else 'N/A':>5}")

    recovery[n] = {
        'spectral_radius': rho,
        'analytical_x_star': float(contest.analytical_symmetric_equilibrium()),
        'analytical_total': float(contest.analytical_symmetric_equilibrium() * n),
        'equilibrium_residual': best_response_residual(contest, x_star),
        'largest_converging_lambda': largest_working_lambda,
        'inertial_sweep': inertial_sweep,
        **{rule: {'converged': run['converged'],
                  'iterations': run['iterations']}
           for rule, run in runs.items()},
    }

results['recovery_by_rule'] = recovery

# --- the asymmetric two-player equilibrium of Section 4.4 -----------------

print("\nPart 2: asymmetric two-player contest, V = (2, 18), r = 1.0")

contest = TullockContest(n=2, r=1.0, valuations=[2.0, 18.0])
rescued = run_inertial(contest, np.array([1.0, 1.0]), lam=0.3,
                       max_iterations=5000)
equilibrium = rescued['final_efforts']
stability = eigenvalue_stability(contest, equilibrium)

print(f"  equilibrium: ({equilibrium[0]:.4f}, {equilibrium[1]:.4f})")
print(f"  best-response residual: {best_response_residual(contest, equilibrium):.2e}")
print(f"  spectral radius: {stability['spectral_radius']:.3f}")
print(f"  recovered by inertial updating (lambda = 0.3) in "
      f"{rescued['iterations']} iterations")

results['asymmetric_two_player'] = {
    'valuations': [2.0, 18.0],
    'equilibrium': equilibrium,
    'residual': best_response_residual(contest, equilibrium),
    'spectral_radius': stability['spectral_radius'],
    'inertial_lambda': 0.3,
    'inertial_iterations': rescued['iterations'],
}

# --- the solver comparison quoted in Section 3.2 --------------------------

print("\nPart 3: unimodal search vs scan-and-refine, against brute force")
print("  (share of rival profiles where the method misses the global maximum)")
print(f"{'r':>5} {'n':>4} | {'golden':>7} | {'bounded':>8} | {'scan+refine':>12}")
print("-" * 46)

N_PROFILES = 60
BRUTE_POINTS = 20_000


def brute_force_payoff(contest, others):
    ceiling = effort_ceiling(contest, 0)
    grid = np.concatenate([
        np.geomspace(EFFORT_MIN, min(1.0, ceiling), BRUTE_POINTS // 4),
        np.linspace(min(1.0, ceiling), ceiling, BRUTE_POINTS),
    ])
    return contest.own_payoff_curve(0, others, grid).max()


def payoff_at(contest, others, effort):
    return contest.own_payoff_curve(0, others, np.array([effort]))[0]


def unimodal_payoff(contest, others, method):
    """Best payoff a search that assumes unimodality manages to find."""
    ceiling = effort_ceiling(contest, 0)
    objective = (lambda x: -payoff_at(contest, others, x))
    try:
        if method == "golden":
            found = minimize_scalar(
                objective, method="golden",
                bracket=(EFFORT_MIN, ceiling / 2, ceiling))
        else:
            found = minimize_scalar(
                objective, method="bounded", bounds=(EFFORT_MIN, ceiling),
                options={"xatol": 1e-10})
        return -found.fun
    except Exception:
        return -np.inf


comparison = {}
for r in (3.0, 5.0):
    for n in (2, 3, 5, 7):
        contest = TullockContest(n=n, r=r, valuations=[V] * n)
        rng = np.random.default_rng(7)
        misses = {'golden': 0, 'bounded': 0, 'scan_and_refine': 0}

        for _ in range(N_PROFILES):
            others = rng.uniform(0.01, 0.9 * V, size=n - 1)
            reference = brute_force_payoff(contest, others)
            for method in ('golden', 'bounded'):
                if unimodal_payoff(contest, others, method) < reference - 1e-6:
                    misses[method] += 1
            solver = payoff_at(contest, others,
                               compute_best_response(contest, 0, others))
            if solver < reference - 1e-6:
                misses['scan_and_refine'] += 1

        rates = {k: 100 * v / N_PROFILES for k, v in misses.items()}
        comparison[f"r={r},n={n}"] = rates
        print(f"{r:>5} {n:>4} | {rates['golden']:>6.0f}% | "
              f"{rates['bounded']:>7.0f}% | {rates['scan_and_refine']:>11.0f}%")

results['solver_comparison'] = comparison

output = save_json(results, DATA_DIR / "supplementary_checks.json")
print(f"\nData saved to {output}")
print("Supplementary checks complete.")
