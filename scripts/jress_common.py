"""Shared setup for the JRESS probabilistic-recovery experiments (exp1-exp3).

One place for the paper's simulation environment (05ExperimentSetup.tex §5.1), so the
three scripts cannot drift apart:

- both aircraft at 20 kts (10.2889 m/s), M600 turn limits (15 deg/s, 10 deg/s^2);
- spawn at 1.5x the look-ahead (tlos 180 s), RPZ 50 m, look-ahead 120 s;
- perfect communication at 1 Hz (no loss, delay, jitter or range gate) — the study
  isolates *navigation* uncertainty;
- state-based detection + MVP (margin 1.05), gamma 0.999 / K_theta 256 defaults;
- 100 pairs per run, all-clear + 10 s to terminate, 600 s cap.

Every script takes ``--production`` for the paper's full budget and defaults to a small
dummy grid (the production runs belong on the server). Outputs: a tidy CSV with the
paper's dependent variables added (``ipr`` = 1 - p_los_run; ``median_cpa_m`` is the
median closest-approach distance, i.e. ``median_min_sep``) plus a provenance card,
under ``results/<exp>/`` — committed to the repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from blueskycdarr import (
    MULTIROTOR,
    Config,
    ConflictConfig,
    ExperimentResult,
    Models,
    PairwiseEncounter,
    PastCPA,
    ProbabilisticFTR,
    SimulationConfig,
)
from blueskycdarr.recovery import FTR, Recovery

SPEED_20_KTS = 10.2889  # m/s
PAIRS_PER_RUN = 100
GAMMA_DEFAULT = 0.999

BASE_CONFIG = Config(
    # uncertainty comes from each experiment's axes; comm stays at the perfect default
    conflict=ConflictConfig(rpz=50.0, t_lookahead=120.0, resolution_margin=1.05),
    simulation=SimulationConfig(dt=0.2, cdr_dt=1.0, t_max=600.0, done_timeout=10.0),
)


def scenario(dpsi: float | None) -> PairwiseEncounter:
    """The paper's encounter: fixed 20 kts, dcpa 0, spawn at 1.5x look-ahead."""
    return PairwiseEncounter(
        speed=SPEED_20_KTS, dpsi=dpsi, dcpa=0.0, tlos=180.0, pairs=(10, 10)
    )


def models(dpsi: float | None = None) -> Models:
    return Models(aircraft=MULTIROTOR, scenario=scenario(dpsi))


def recovery_by_name(name: str, gamma: float = GAMMA_DEFAULT) -> Recovery:
    return {
        "pastcpa": PastCPA(),
        "ftr": FTR(),
        "probabilistic_ftr": ProbabilisticFTR(gamma=gamma),
    }[name]


def parser(description: str, production_runs: int) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--production",
        action="store_true",
        help=f"the paper's full budget ({production_runs} runs/condition, full grids)",
    )
    p.add_argument("--runs", type=int, default=None,
                   help="override runs per condition (100 pairs each)")
    p.add_argument("--jobs", type=int, default=-1, help="episode workers; -1 = all cores")
    p.add_argument("--seed", type=int, default=42)
    return p


def runs_for(args: argparse.Namespace, production_runs: int, dummy_runs: int = 2) -> int:
    if args.runs is not None:
        return args.runs
    return production_runs if args.production else dummy_runs


def angle_grid(production: bool) -> list[float]:
    """The paper's 2..180 step-2 sweep; a 7-angle skeleton for the dummy runs."""
    if production:
        return [float(a) for a in range(2, 181, 2)]
    return [2.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]


def save(result: ExperimentResult, name: str) -> Path:
    """The paper's dependent variables beside the raw table, plus the card path note."""
    out_dir = Path("results") / name
    csv_path = result.write_csv(out_dir / f"{name}.csv")
    import pandas

    df = pandas.read_csv(csv_path)
    df["ipr"] = 1.0 - df["p_los_run"]
    df["median_cpa_m"] = df["median_min_sep"]
    df.to_csv(csv_path, index=False)
    print(f"\n{len(result)} condition(s) -> {csv_path}")
    print(f"provenance card -> {result.card_path}")
    return csv_path
