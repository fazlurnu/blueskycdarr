"""Engine-backed locks: the episode loop end to end (``cdarr/episode.py``).

These are the expensive tests — small batches, short horizons — that pin the behaviours
everything else rests on: a clean resolve under perfect CNS, the ballistic null result
(blind aircraft collide at the spawned geometry), and bit-for-bit reproducibility of a
seeded episode. They need the CDaRR BlueSky fork; the whole module skips without it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bluesky")

from cdarr.aircraft import MULTIROTOR  # noqa: E402
from cdarr.config import CommConfig, Config, UncertaintyConfig  # noqa: E402
from cdarr.episode import run_episode  # noqa: E402
from cdarr.experiment import MC, Fixed, Models, run_experiment  # noqa: E402
from cdarr.rng import child, root_seed_sequence  # noqa: E402
from cdarr.scenario import PairwiseEncounter  # noqa: E402

_SCENARIO = PairwiseEncounter(pairs=(1, 2), tlos=45.0)
_SEQ = child(root_seed_sequence(0), 0)


def test_perfect_cns_resolves_every_pair() -> None:
    """The tracer bullet: dpsi 90, no noise or loss — detected, resolved, settled."""
    result = run_episode(_SCENARIO, MULTIROTOR, Config(), _SEQ)
    assert result.detected.all()
    assert result.n_los == 0
    assert (result.min_sep > 50.0).all()
    assert result.settled


def test_blind_aircraft_collide_at_the_spawned_geometry() -> None:
    """reception_prob 0 is the null experiment: ballistic flight through dcpa ~ 0."""
    blind = Config(comm=CommConfig(reception_prob=0.0))
    result = run_episode(_SCENARIO, MULTIROTOR, blind, _SEQ)
    assert not result.detected.any()
    assert result.n_los == _SCENARIO.n_pairs
    assert (result.min_sep < 5.0).all()  # creconfs delivered the requested miss distance


def test_a_seeded_episode_reproduces_bit_for_bit() -> None:
    config = Config(
        uncertainty=UncertaintyConfig(pos_ci95=10.0, vel_ci95=1.0),
        comm=CommConfig(reception_prob=0.8, latency_s=0.1, broadcast_jitter_s=0.1),
    )
    first = run_episode(_SCENARIO, MULTIROTOR, config, _SEQ)
    second = run_episode(_SCENARIO, MULTIROTOR, config, _SEQ)
    np.testing.assert_array_equal(first.min_sep, second.min_sep)
    assert first.n_los == second.n_los and first.t_end == second.t_end


def test_commanded_ground_speed_is_the_ground_speed_flown() -> None:
    """The engine converts ground -> CAS at the boundary; without it, BlueSky reads a
    ground-frame command as CAS and flies 0.48% fast at 100 m — and a loop that re-feeds
    measured ground speed compounds that factor per command (the CDaRR speed creep,
    demonstrated in notebooks/bluesky_speed_command.ipynb)."""
    from cdarr.engine import PairwiseWorld

    scenario = PairwiseEncounter(pairs=(1, 1), speed=10.0, tlos=45.0)
    geometry = scenario.draw_geometry(np.random.default_rng(0))
    config = Config()
    with PairwiseWorld(scenario, geometry, MULTIROTOR, config.conflict,
                       config.simulation) as world:
        for _ in range(25):  # 5 s: past any spawn transient
            world.step()
        truth = world.truth()
        np.testing.assert_allclose(truth.gs, 10.0, atol=0.01)


def test_run_experiment_single_condition_reports_whole_episodes() -> None:
    result = run_experiment(
        {"pos_ci95": Fixed(0.0)},
        models=Models(aircraft=MULTIROTOR, scenario=_SCENARIO),
        backend=MC(n_encounters=3),  # 2 pairs/episode -> rounds up to 4
        base_config=Config(),
        seed=0,
        progress=False,
    )
    estimate = result.cell()
    assert estimate.n_encounters == 4
    assert estimate.n_los == 0
