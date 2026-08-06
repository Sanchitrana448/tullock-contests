"""Analysis report across the r x n grid.

Convergence, dissipation and spectral radius for each cell. These are the
numbers quoted in Chapter 4.
"""

from common import DATA_DIR, banner, save_json
from src.analysis import generate_full_report, print_report_summary

R_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
N_VALUES = [2, 3, 5, 10, 20]
V = 10.0
N_SEEDS = 10
MAX_ITERS = 200

banner("Full analysis report")
print(f"Grid: {len(R_VALUES)} r values x {len(N_VALUES)} n values\n")

report = generate_full_report(R_VALUES, N_VALUES, V=V, n_seeds=N_SEEDS,
                              max_iterations=MAX_ITERS)
print_report_summary(report, R_VALUES, N_VALUES)

# JSON cannot use tuples as keys, so flatten to a list of records.
records = [dict(cell, n=n, r=r) for (n, r), cell in report.items()]
output = save_json(records, DATA_DIR / "full_analysis_report.json")
print(f"\nReport saved to {output}")
