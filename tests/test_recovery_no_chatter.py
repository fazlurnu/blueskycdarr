"""Engine-backed lock: FTR-family recovery judges the return course, so it does not chatter.

FTR releases when the course the aircraft would *return to* clears the zone (CDaRR's
``get_desired_ownship_velocity`` reads the autopilot's ``ap.trk`` / ``ap.tas``, which the
resolver never writes). Judging the standing resolution command instead releases every
avoidance course one tick after issue -- it clears by construction -- and the aircraft
flips between resolution and nominal on every CDR tick (ADR 0006 update, 2026-11-04).

These need the CDaRR BlueSky fork; the whole module skips without it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bluesky")

from blueskycdarr.aircraft import MULTIROTOR  # noqa: E402
from blueskycdarr.config import (  # noqa: E402
    CommConfig, Config, ConflictConfig, SimulationConfig, UncertaintyConfig,
)
from blueskycdarr.episode import run_episode  # noqa: E402
from blueskycdarr.recovery import FTR, ProbabilisticFTR  # noqa: E402
from blueskycdarr.rng import root_seed_sequence  # noqa: E402
from blueskycdarr.scenario import PairwiseEncounter  # noqa: E402


@pytest.mark.parametrize("recovery", [FTR(), ProbabilisticFTR()])
def test_one_resolution_per_encounter_with_perfect_cns(recovery) -> None:
    config = Config(
        uncertainty=UncertaintyConfig(0.0, 0.0),
        comm=CommConfig(reception_prob=1.0),
        conflict=ConflictConfig(rpz=150.0, t_lookahead=300.0, resolution_margin=1.05),
        simulation=SimulationConfig(dt=0.2, cdr_dt=1.0, t_max=900.0, done_timeout=10.0),
    )
    scenario = PairwiseEncounter(speed=10.28888, dpsi=90.0, dcpa=0.0, tlos=400.0, pairs=(1, 1),
                                 aim="flown")
    speeds = []
    result = run_episode(scenario, MULTIROTOR, config, root_seed_sequence(7), recovery,
                         recorder=lambda t, truth, d: speeds.append(truth.gs[0]))
    changes = int(np.sum(np.abs(np.diff(speeds)) > 0.05))
    assert changes <= 4, f"speed changed {changes} times: resolve/release chatter"
    assert result.n_los == 0
