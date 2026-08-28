"""Locks for the declaration layer (``blueskycdarr/experiment.py``, ADR 0003).

Everything here runs without the engine: expansion order, the closed vocabulary, value
routing into config/models, run-file parsing, and the result tabulations over hand-built
estimates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blueskycdarr.aircraft import FIXEDWING, MULTIROTOR
from blueskycdarr.config import Config
from blueskycdarr.experiment import (
    MC,
    Condition,
    ExperimentResult,
    Fixed,
    Models,
    Sweep,
    _apply,
    expand,
    load_run,
    sweep_from_file,
)
from blueskycdarr.metrics import MonteCarloEstimate
from blueskycdarr.recovery import FTR, ProbabilisticFTR
from blueskycdarr.scenario import PairwiseEncounter

_RUN_FILE = """
seed: 7
aircraft: multirotor
scenario: {type: pairwise, speed: 15.0, dpsi: 90.0, dcpa: 0.0, tlos: 60.0, pairs: [2, 2]}
recovery: {type: probabilistic_ftr, gamma: 0.99}
uncertainty: {pos_ci95: 10.0, vel_ci95: 1.0}
comm: {reception_prob: 0.8, max_range_m: 1000.0, latency_s: 0.1, broadcast_jitter_s: 0.1}
estimate: {type: mc, n_encounters: 8}
sweep:
  aircraft: [multirotor, fixedwing]
  pos_ci95: [3.0, 92.6]
"""


def test_expand_cross_product_and_unknown_key_message() -> None:
    conditions = expand({"pos_ci95": Sweep([3.0, 10.0]), "vel_ci95": Sweep([1.0, 3.0])})
    levels = [dict(c.levels) for c in conditions]
    assert levels == [
        {"pos_ci95": 3.0, "vel_ci95": 1.0},
        {"pos_ci95": 3.0, "vel_ci95": 3.0},
        {"pos_ci95": 10.0, "vel_ci95": 1.0},
        {"pos_ci95": 10.0, "vel_ci95": 3.0},
    ]
    with pytest.raises(ValueError, match="not_a_parameter"):
        expand({"not_a_parameter": Fixed(1)})


def test_apply_routes_every_vocabulary_group() -> None:
    condition = Condition(
        levels=(),
        values=(
            ("pos_ci95", 30.0),
            ("reception_prob", 0.5),
            ("rpz", 60.0),
            ("t_max", 200.0),
            ("dpsi", 45.0),
            ("aircraft", "fixedwing"),
        ),
    )
    config, models = _apply(condition, Config(), Models(MULTIROTOR, PairwiseEncounter()))
    assert config.uncertainty.pos_ci95 == 30.0
    assert config.comm.reception_prob == 0.5
    assert config.conflict.rpz == 60.0
    assert config.simulation.t_max == 200.0
    assert models.scenario.dpsi == 45.0
    assert models.aircraft is FIXEDWING


def test_run_file_round_trip_and_sweep_block(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(_RUN_FILE)
    config, models, backend = load_run(path)
    assert config.seed == 7
    assert config.comm.reception_prob == 0.8
    assert models.aircraft is MULTIROTOR
    assert models.scenario.pairs == (2, 2)
    assert models.recovery == ProbabilisticFTR(gamma=0.99)
    assert backend == MC(n_encounters=8)

    axes = sweep_from_file(path)
    assert set(axes) == {"aircraft", "pos_ci95"}
    assert isinstance(axes["pos_ci95"], Sweep) and axes["pos_ci95"].values == (3.0, 92.6)


def test_a_recovery_sweep_routes_labels_and_typed_levels() -> None:
    condition = Condition(
        levels=(("recovery", "ftr"),),
        values=(("recovery", {"type": "probabilistic_ftr", "gamma": 0.9}),),
    )
    _, models = _apply(condition, Config(), Models(MULTIROTOR, PairwiseEncounter()))
    assert models.recovery == ProbabilisticFTR(gamma=0.9)
    _, by_label = _apply(
        Condition(levels=(), values=(("recovery", "ftr"),)),
        Config(),
        Models(MULTIROTOR, PairwiseEncounter()),
    )
    assert by_label.recovery == FTR()


def test_run_file_errors_name_the_problem(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(_RUN_FILE.replace("type: pairwise", "type: ring"))
    with pytest.raises(ValueError, match="pairwise"):
        load_run(bad)
    bad.write_text(_RUN_FILE.replace("pos_ci95: [3.0, 92.6]", "warp_factor: [9]"))
    with pytest.raises(ValueError, match="warp_factor"):
        sweep_from_file(bad)


def _result_with(levels_metrics: list[tuple[dict, int]]) -> ExperimentResult:
    conditions, results = [], []
    for levels, n_los in levels_metrics:
        conditions.append(Condition(levels=tuple(levels.items()), values=tuple(levels.items())))
        results.append(
            MonteCarloEstimate(
                n_encounters=100,
                n_los=n_los,
                min_sep=np.full(100, 80.0),
                detection_rate=1.0,
                n_unsettled=0,
            )
        )
    return ExperimentResult(
        backend=MC(n_encounters=100),
        seed=0,
        conditions=tuple(conditions),
        results=tuple(results),
        axes=tuple(levels_metrics[0][0]) if levels_metrics else (),
    )


def test_records_and_cell_and_csv(tmp_path: Path) -> None:
    result = _result_with([({"pos_ci95": 3.0}, 2), ({"pos_ci95": 10.0}, 7)])
    rows = result.records()
    assert rows[0]["pos_ci95"] == 3.0 and rows[0]["p_los_run"] == 0.02
    assert "p_los_lo" not in rows[0] and "p_los_hi" not in rows[0]  # no interval columns

    assert result.cell(pos_ci95=10.0).n_los == 7
    with pytest.raises(KeyError):
        result.cell(pos_ci95=99.0)
    with pytest.raises(KeyError):
        result.cell()  # ambiguous over two conditions

    csv = (result.write_csv(tmp_path / "out.csv")).read_text().splitlines()
    assert csv[0].startswith("pos_ci95,n_encounters,n_los,p_los_run")
    assert len(csv) == 3
