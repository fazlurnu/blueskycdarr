"""JRESS Experiment 2 — the confidence threshold gamma under Gaussian noise (Table 2).

Crossing angle {2, 4, ..., 180} x position CI95 {3, 10} m x velocity CI95 {1, 3} m/s
x recovery {FTR, Probabilistic FTR at gamma in {0.999, 0.99, 0.9, 0.75, 0.5}}, 20 kts
fixed, 100 runs x 100 pairs per condition. FTR is the gamma-independent reference; the
recovery axis is labelled by gamma (fig_gamma_comparison_ipr and its median-CPA
companion).

    .venv/bin/python scripts/exp2_gamma.py                # dummy (7 angles, 2 runs)
    .venv/bin/python scripts/exp2_gamma.py --production   # the paper's budget
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
    runs_for,
    save,
)

from cdarr import MC, Fixed, ProbabilisticFTR, Sweep, run_experiment
from cdarr.recovery import FTR

PRODUCTION_RUNS = 100
GAMMAS = [0.999, 0.99, 0.9, 0.75, 0.5]


def _recovery(level: str | float) -> FTR | ProbabilisticFTR:
    return FTR() if level == "ftr" else ProbabilisticFTR(gamma=float(level))


def main() -> None:
    args = parser(__doc__, PRODUCTION_RUNS).parse_args()
    runs = runs_for(args, PRODUCTION_RUNS)

    result = run_experiment(
        {
            "dpsi": Sweep(angle_grid(args.production)),
            "pos_ci95": Sweep([3.0, 10.0]),
            "vel_ci95": Sweep([1.0, 3.0]),
            "recovery": Sweep(["ftr", *GAMMAS], build=_recovery),
            "noise": Fixed("gaussian"),
        },
        models=models(),
        backend=MC(n_encounters=runs * PAIRS_PER_RUN),
        base_config=BASE_CONFIG,
        seed=args.seed,
        n_jobs=args.jobs,
        card_dir=Path("results/exp2/cards"),
    )
    save(result, "exp2")


if __name__ == "__main__":
    main()
