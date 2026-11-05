"""The ``aim="flown"`` spawn: the miss distance the pair actually flies is ``dcpa``.

``creconfs`` aims along flat-plane straight lines, and constant-heading flight on
BlueSky's sphere bows away from them, so a long spawn misses by more than ``dcpa``.
These lock both halves: the default placement still misses (the effect is real, and the
default is unchanged for past results), and the flown placement does not.

These need the CDaRR BlueSky fork; the whole module skips without it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bluesky")

from blueskycdarr.aircraft import FIXEDWING  # noqa: E402
from blueskycdarr.config import ConflictConfig, SimulationConfig  # noqa: E402
from blueskycdarr.engine import PairwiseWorld  # noqa: E402
from blueskycdarr.scenario import PairwiseEncounter  # noqa: E402

RPZ = 150.0
SPEED = 30.86664  # 60 kt: the longest spawns, so the largest bow


def _fly(aim: str, dpsi: float, dcpa: float) -> tuple[np.ndarray, float]:
    """(distance per step, dt) for one pair flown with no commands until past CPA."""
    scenario = PairwiseEncounter(speed=SPEED, dpsi=dpsi, dcpa=dcpa, dcpa_max=50.0,
                                 tlos=400.0, pairs=(1, 1), aim=aim)
    conflict = ConflictConfig(rpz=RPZ, t_lookahead=300.0, resolution_margin=1.0)
    simulation = SimulationConfig(dt=0.2, t_max=900.0)
    geometry = scenario.draw_geometry(np.random.default_rng(0))
    distances = []
    with PairwiseWorld(scenario, geometry, FIXEDWING, conflict, simulation) as world:
        for _ in range(int(600.0 / simulation.dt)):  # past CPA at 2 deg (539 s)
            distances.append(world.pair_distances()[0])
            world.step()
    return np.array(distances), simulation.dt


def test_creconfs_misses_a_fast_crossing() -> None:
    d, _ = _fly("creconfs", dpsi=135.0, dcpa=0.0)
    assert d.min() > 20.0


@pytest.mark.parametrize("dpsi", [2.0, 90.0, 135.0, 180.0])
@pytest.mark.parametrize("dcpa", [0.0, 30.0])
def test_flown_aim_flies_the_requested_miss(dpsi: float, dcpa: float) -> None:
    d, dt = _fly("flown", dpsi=dpsi, dcpa=dcpa)
    assert d.min() == pytest.approx(dcpa, abs=0.1)
    t_los = dt * np.argmax(d < RPZ)
    assert t_los == pytest.approx(400.0, abs=1.0)


def test_aim_is_validated() -> None:
    with pytest.raises(ValueError, match="aim"):
        PairwiseEncounter(aim="straight")
