"""JRESS Experiment 3 — robustness to the six navigation-noise models (Table 3).

Crossing angle drawn U(0, 360) per pair and aggregated; position/velocity CI95 fixed at
10 m / 1 m/s; the six position-error models of Appendix A (all matched to the same
radial CI95) x recovery {Past-CPA, FTR, Probabilistic FTR (gamma 0.999)}; 20 kts fixed,
1000 runs x 100 pairs per condition — one aggregate IPR per condition (18 cells).

The six models: isotropic Gaussian; heavy-tail mixture (10% at 3x sigma); anisotropic
(along-track std 3x cross-track, i.e. variance ratio 9); latency bias (0.1 s broadcast
delay times ground speed, intruder-perceived only); and the anisotropic model combined
with the latency bias and with the heavy tail.

    .venv/bin/python scripts/exp3_noise_models.py                # dummy (2 runs)
    .venv/bin/python scripts/exp3_noise_models.py --production   # the paper's budget
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jress_common import (
    BASE_CONFIG,
    PAIRS_PER_RUN,
    models,
    parser,
    recovery_by_name,
    runs_for,
    save,
)

from cdarr import (
    MC,
    AnisotropicGaussian,
    AnisotropicMixtureGaussian,
    Fixed,
    Gaussian,
    LatencyBiased,
    MixtureGaussian,
    Sweep,
    run_experiment,
)
from cdarr.noise import NoiseShape

PRODUCTION_RUNS = 1000

# Along-track std 3x cross-track = variance ratio 9; tail: 10% at 3x sigma; delay 0.1 s
# (Schaefer & Jonas 2025 measured ~66 ms for ADS-B v2; the paper stress-tests 100 ms).
NOISE_MODELS: dict[str, NoiseShape] = {
    "gaussian": Gaussian(),
    "mixture": MixtureGaussian(tail_ratio=3.0, tail_weight=0.1),
    "anisotropic": AnisotropicGaussian(var_ratio=9.0),
    "latency": LatencyBiased(Gaussian(), delay_s=0.1),
    "anisotropic_latency": LatencyBiased(AnisotropicGaussian(var_ratio=9.0), delay_s=0.1),
    "anisotropic_mixture": AnisotropicMixtureGaussian(
        var_ratio=9.0, tail_ratio=3.0, tail_weight=0.1
    ),
}


def main() -> None:
    args = parser(__doc__, PRODUCTION_RUNS).parse_args()
    runs = runs_for(args, PRODUCTION_RUNS)

    result = run_experiment(
        {
            "noise": Sweep(list(NOISE_MODELS), name="noise_model",
                           build=NOISE_MODELS.__getitem__),
            "recovery": Sweep(
                ["pastcpa", "ftr", "probabilistic_ftr"], build=recovery_by_name
            ),
            "pos_ci95": Fixed(10.0),
            "vel_ci95": Fixed(1.0),
        },
        models=models(dpsi=None),  # U(0, 360) per pair, aggregated (Table 3)
        backend=MC(n_encounters=runs * PAIRS_PER_RUN),
        base_config=BASE_CONFIG,
        seed=args.seed,
        n_jobs=args.jobs,
        card_dir=Path("results/exp3/cards"),
    )
    save(result, "exp3")


if __name__ == "__main__":
    main()
