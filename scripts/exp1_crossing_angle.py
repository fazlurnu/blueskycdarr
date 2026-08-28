"""JRESS Experiment 1 — recovery methods under isotropic Gaussian noise (Table 1).

Crossing angle {2, 4, ..., 180} x position CI95 {3, 10} m x velocity CI95 {1, 3} m/s
x recovery {Past-CPA, FTR, Probabilistic FTR (gamma 0.999)}, Gaussian noise, 20 kts
fixed, 100 runs x 100 pairs per condition. Produces the IPR-versus-angle profile per
condition (fig_crossing_angle_vs_ipr) and its median-CPA companion.

    .venv/bin/python scripts/exp1_crossing_angle.py                # dummy (7 angles, 2 runs)
    .venv/bin/python scripts/exp1_crossing_angle.py --production   # the paper's budget
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jress_common import (
    BASE_CONFIG,
    PAIRS_PER_RUN,
    angle_grid,
    models,
    parser,
    recovery_by_name,
    runs_for,
    save,
)

from blueskycdarr import MC, Fixed, Sweep, run_experiment

PRODUCTION_RUNS = 100


def main() -> None:
    args = parser(__doc__, PRODUCTION_RUNS).parse_args()
    runs = runs_for(args, PRODUCTION_RUNS)

    result = run_experiment(
        {
            "dpsi": Sweep(angle_grid(args.production)),
            "pos_ci95": Sweep([3.0, 10.0]),
            "vel_ci95": Sweep([1.0, 3.0]),
            "recovery": Sweep(
                ["pastcpa", "ftr", "probabilistic_ftr"], build=recovery_by_name
            ),
            "noise": Fixed("gaussian"),
        },
        models=models(),
        backend=MC(n_encounters=runs * PAIRS_PER_RUN),
        base_config=BASE_CONFIG,
        seed=args.seed,
        n_jobs=args.jobs,
        card_dir=Path("results/exp1/cards"),
    )
    save(result, "exp1")


if __name__ == "__main__":
    main()
