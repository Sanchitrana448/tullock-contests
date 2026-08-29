# Best response dynamics in Tullock contests

Simulation code for my MSc dissertation on whether players in a Tullock
contest actually reach the Nash equilibria that theory says exist.

In a Tullock contest, `n` players choose efforts `x_i` and win with
probability `x_i^r / sum_j x_j^r`, paying their effort as a cost. The
equilibrium is well understood. What is less clear is whether players who
repeatedly best-respond to whatever everyone else did last round end up
there. This answers that empirically over the `r x n` parameter space.

## Layout

```
src/            model and dynamics
  contest.py          payoffs and the contest success function
  best_response.py    single player best response solver
  dynamics.py         synchronous, asynchronous and inertial update rules
  analysis.py         convergence statistics and eigenvalue stability
  plots.py            plots shared between experiments
experiments/    one script per experiment, run from the command line
tests/          pytest suite, including validation against the closed form
results/        figures, saved data, and the dissertation
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the experiments

Each one is independent and writes figures to `results/figures/` and its raw
output to `results/data/`:

```bash
python experiments/exp1_baseline.py        # validation, figures 1-3
python experiments/exp2_r_sweep.py         # vary r,  figures 4-6
python experiments/exp3_n_sweep.py         # vary n,  figures 7-9
python experiments/exp4_heterogeneous.py   # unequal valuations, 10-11
python experiments/exp5_high_r.py          # high r in detail, 12-14
python experiments/exp6_grid.py            # full r x n map, 15-16
python experiments/analysis_report.py      # spectral radius report
python experiments/extra_checks.py         # one-off numbers from Chapter 4
```

Experiments 2, 3 and 6 take a few minutes each. The rest are quick. The long
sweeps cache each cell under `results/data/cache/`, so an interrupted run
picks up where it left off; delete that directory to force a clean run.

## Tests

```bash
pytest
```

`tests/test_validation.py` is the one that matters for trusting the rest: it
checks the simulation against Tullock's closed form `x* = V(n-1)/n^2`.
`test_best_response.py` checks the solver against a brute force global
maximum, and `test_dynamics.py` checks the update rules against each other.

## Two notes on the implementation

The best response solver scans before it refines. For `r > 1` the payoff in
own effort is not concave, so a local maximum near zero can sit alongside an
interior one. Golden section search and SciPy's bounded method both return
whichever basin they land in, with no indication that they have done so, and
get it wrong on a large share of rival profiles at high `r`. Scanning the
interval first and refining inside the best cell avoids this.

Convergence is decided by a residual test, not a step size test. Efforts
that collapse towards the bottom of the search interval take tiny steps
while sitting nowhere near an equilibrium, so what gets checked is whether
any player can still gain by moving.
