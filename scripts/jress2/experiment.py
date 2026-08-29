"""The three JRESS probabilistic-recovery experiments, in one script.

Everything the paper's Tables 1-3 declare lives here: the shared simulation environment
(05ExperimentSetup.tex §5.1) and one function per experiment.  Each is a single
``run_experiment`` call, so the tables read like the paper's DOE tables.

The dependent variable reported here is **P(LoS)** -- the fraction of encounters that
lose separation -- not IPR.  ``run_experiment`` already emits it as ``p_los_run``
alongside the raw counts, so the CSV is written straight out with no post-processing.
Efficiency stays the paper's median closest-approach distance (``median_min_sep``).

There are no command-line arguments: edit ``EXPERIMENTS`` below and run the file.

    .venv/bin/python scripts/jress2/experiment.py

Outputs land in ``results/jress2/<exp>.csv`` with a provenance card beside them.
"""

from __future__ import annotations

from pathlib import Path

from blueskycdarr import (
    MC,
    MULTIROTOR,
    AnisotropicGaussian,
    AnisotropicMixtureGaussian,
    Config,
    ConflictConfig,
    ExperimentResult,
    Fixed,
    Gaussian,
    LatencyBiased,
    MixtureGaussian,
    Models,
    PairwiseEncounter,
    PastCPA,
    ProbabilisticFTR,
    SimulationConfig,
    Sweep,
    run_experiment,
)
from blueskycdarr.recovery import FTR, Recovery

# --- what to run -----------------------------------------------------------------------

EXPERIMENTS = ("exp3",)  # any of "exp1", "exp2", "exp3"
SEED = 42
N_JOBS = -1  # episode workers; -1 = every core
OUT_DIR = Path("results/jress2")

# --- the shared environment (§5.1) -------------------------------------------------------

SPEED_20_KTS = 10.2889  # m/s, both aircraft
PAIRS_PER_RUN = 100  # one simulator run = 100 independent pairwise encounters
GAMMA_DEFAULT = 0.999

BASE_CONFIG = Config(
    # uncertainty comes from each experiment's axes; comm stays at the perfect 1 Hz default
    conflict=ConflictConfig(rpz=50.0, t_lookahead=120.0, resolution_margin=1.05),
    simulation=SimulationConfig(dt=0.2, cdr_dt=1.0, t_max=600.0, done_timeout=10.0),
)

# spawn at 1.5x the look-ahead, head-on miss distance 0, M600 turn limits from the catalog
MODELS = Models(
    aircraft=MULTIROTOR,
    scenario=PairwiseEncounter(
        speed=SPEED_20_KTS, dpsi=90.0, dcpa=0.0, tlos=180.0, pairs=(10, 10)
    ),
)

ANGLES = [float(a) for a in range(2, 181, 2)]  # {2, 4, ..., 180} deg

# Along-track std 3x cross-track = variance ratio 9; tail: 10% at 3x sigma; delay 0.1 s.
NOISE_MODELS = {
    "gaussian": Gaussian(),
    "mixture": MixtureGaussian(tail_ratio=3.0, tail_weight=0.1),
    "anisotropic": AnisotropicGaussian(var_ratio=9.0),
    "latency": LatencyBiased(Gaussian(), delay_s=0.1),
    "anisotropic_latency": LatencyBiased(AnisotropicGaussian(var_ratio=9.0), delay_s=0.1),
    "anisotropic_mixture": AnisotropicMixtureGaussian(
        var_ratio=9.0, tail_ratio=3.0, tail_weight=0.1
    ),
}


def recovery_by_name(name: str) -> Recovery:
    return {
        "pastcpa": PastCPA(),
        "ftr": FTR(),
        "probabilistic_ftr": ProbabilisticFTR(gamma=GAMMA_DEFAULT),
    }[name]


def gamma_level(level: str | float) -> Recovery:
    """Experiment 2's recovery axis: FTR is the gamma-independent reference."""
    return FTR() if level == "ftr" else ProbabilisticFTR(gamma=float(level))


def run(name: str, axes: dict, runs: int, dpsi: float | None = 90.0) -> ExperimentResult:
    """One experiment: `runs` x 100 pairs per condition, table and card under OUT_DIR."""
    from dataclasses import replace

    result = run_experiment(
        axes,
        models=replace(MODELS, scenario=replace(MODELS.scenario, dpsi=dpsi)),
        backend=MC(n_encounters=runs * PAIRS_PER_RUN),
        base_config=BASE_CONFIG,
        seed=SEED,
        n_jobs=N_JOBS,
        card_dir=OUT_DIR / "cards",
    )
    csv_path = result.write_csv(OUT_DIR / f"{name}.csv")
    print(f"\n{name}: {len(result)} condition(s) -> {csv_path}")
    print(f"provenance card -> {result.card_path}")
    return result


# --- Table 1: recovery methods under isotropic Gaussian noise ----------------------------


def exp1() -> ExperimentResult:
    """Crossing angle {2..180} x pos CI95 {3, 10} m x vel CI95 {1, 3} m/s
    x recovery {Past-CPA, FTR, Probabilistic FTR (gamma 0.999)}; 100 runs x 100 pairs."""
    return run(
        "exp1",
        {
            "dpsi": Sweep(ANGLES),
            "pos_ci95": Sweep([3.0, 10.0]),
            "vel_ci95": Sweep([1.0, 3.0]),
            "recovery": Sweep(
                ["pastcpa", "ftr", "probabilistic_ftr"], build=recovery_by_name
            ),
            "noise": Fixed(Gaussian()),
        },
        runs=100,
    )


# --- Table 2: the confidence threshold gamma ---------------------------------------------


def exp2() -> ExperimentResult:
    """Same grid as Experiment 1, but the recovery axis is
    {FTR, Probabilistic FTR at gamma in {0.999, 0.99, 0.9, 0.75, 0.5}}."""
    return run(
        "exp2",
        {
            "dpsi": Sweep(ANGLES),
            "pos_ci95": Sweep([3.0, 10.0]),
            "vel_ci95": Sweep([1.0, 3.0]),
            "recovery": Sweep(["ftr", 0.999, 0.99, 0.9, 0.75, 0.5], build=gamma_level),
            "noise": Fixed(Gaussian()),
        },
        runs=100,
    )


# --- Table 3: robustness to the six navigation-noise models ------------------------------


def exp3() -> ExperimentResult:
    """Crossing angle drawn U(0, 360) per pair and aggregated; uncertainty fixed at
    10 m / 1 m/s; six noise models x three recovery methods; 1000 runs x 100 pairs."""
    return run(
        "exp3",
        {
            "noise": Sweep(
                list(NOISE_MODELS), name="noise_model", build=NOISE_MODELS.__getitem__
            ),
            "recovery": Sweep(
                ["pastcpa", "ftr", "probabilistic_ftr"], build=recovery_by_name
            ),
            "pos_ci95": Fixed(10.0),
            "vel_ci95": Fixed(1.0),
        },
        runs=1000,
        dpsi=None,  # U(0, 360) per pair (Table 3)
    )


if __name__ == "__main__":
    for exp in EXPERIMENTS:
        {"exp1": exp1, "exp2": exp2, "exp3": exp3}[exp]()
