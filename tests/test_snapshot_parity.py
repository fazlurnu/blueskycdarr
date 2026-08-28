"""Engine-backed locks for the IPS particle operations: snapshot, restore, clone, resume.

Fixed-level splitting (OpenCDaRR's ADR 0017, ported here) stands on one invariant: a
particle — a :class:`~cdarr.engine.WorldSnapshot` beside an
:class:`~cdarr.episode.EpisodeState` — restored and advanced must reproduce, **bit for
bit**, what an uninterrupted run does. Any future-affecting value that escapes the
particle (a stale global, an uncopied array, a timer counter) diverges these runs — and
at rare-event probabilities that corruption would be invisible in the estimate, which is
why it is pinned here instead. The layers are locked separately so a failure names its
culprit: the raw world restore first, then the full particle round trip (including a
pickled particle resumed in a *different* world — the time-multiplex and the process
boundary the parallel scheduler will cross), then the guard rails and the divergence
that fresh streams must produce.

These need the CDaRR BlueSky fork; the whole module skips without it.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("bluesky")

from cdarr.aircraft import MULTIROTOR  # noqa: E402
from cdarr.config import CommConfig, Config, UncertaintyConfig  # noqa: E402
from cdarr.engine import PairwiseWorld  # noqa: E402
from cdarr.episode import (  # noqa: E402
    EpisodeStreams,
    advance,
    episode_context,
    episode_result,
    init_episode,
)
from cdarr.rng import child, generator, root_seed_sequence  # noqa: E402
from cdarr.scenario import PairwiseEncounter  # noqa: E402

# Short-horizon encounter, every stream exercised: GNSS noise feeds the CDR views (so
# clones can diverge), latency keeps messages in flight across snapshots, jitter and
# random phase draw from the schedule stream, reception < 1 draws per transmission.
_SCENARIO = PairwiseEncounter(pairs=(1, 2), tlos=20.0)
_CONFIG = Config(
    uncertainty=UncertaintyConfig(pos_ci95=10.0, vel_ci95=1.0),
    comm=CommConfig(
        reception_prob=0.8,
        latency_s=0.1,
        broadcast_jitter_s=0.1,
        broadcast_random_phase=True,
    ),
)
_SEQ = child(root_seed_sequence(11), 0)
_PRE_STEPS = 40  # 8 s at dt 0.2: mid-approach, conflicts detected, resolutions flying


def _world(geometry_child: int) -> PairwiseWorld:
    geometry = _SCENARIO.draw_geometry(generator(child(_SEQ, geometry_child)))
    return PairwiseWorld(
        _SCENARIO, geometry, MULTIROTOR, _CONFIG.conflict, _CONFIG.simulation
    )


def _record_truth(world: PairwiseWorld) -> np.ndarray:
    truth = world.truth()
    return np.concatenate(
        [truth.lat, truth.lon, truth.trk, truth.gs, truth.gs_east, truth.gs_north,
         world.pair_distances()]
    )


def _continue_to_end(world, ctx, state, streams) -> tuple:
    """Advance to the episode's end, recording the full truth trajectory."""
    traj: list[np.ndarray] = []
    while advance(world, ctx, state, streams,
                  recorder=lambda t, truth, d: traj.append(
                      np.concatenate([[t], truth.lat, truth.lon, truth.trk, d]))):
        pass
    return episode_result(state, ctx), np.asarray(traj)


def test_a_restored_world_continues_bit_identically() -> None:
    """The engine layer alone: restore into the dirty world and into a reset + respawned
    one (a different particle's geometry), and the continuation matches exactly —
    including the fork's turn-limiter memory, exercised by a mid-run heading command."""
    scenario = PairwiseEncounter(pairs=(1, 1), tlos=45.0)
    config = Config()
    geometry = scenario.draw_geometry(generator(child(_SEQ, 90)))
    with PairwiseWorld(scenario, geometry, MULTIROTOR, config.conflict,
                       config.simulation) as world:
        for k in range(30):
            if k == 10:  # engage the rate-limited turn so prev_turnrate is armed
                world.command((world.nominal_trk + 40.0) % 360.0, world.nominal_gs)
            world.step()
        snap = world.snapshot()
        tail_a = []
        for _ in range(30):
            world.step()
            tail_a.append(_record_truth(world))

        world.restore(snap)  # same world, now 30 steps dirtier
        tail_b = []
        for _ in range(30):
            world.step()
            tail_b.append(_record_truth(world))
    np.testing.assert_array_equal(np.asarray(tail_a), np.asarray(tail_b))

    # a fresh world of the same cell, flown somewhere else entirely, then overwritten
    geometry2 = scenario.draw_geometry(generator(child(_SEQ, 91)))
    with PairwiseWorld(scenario, geometry2, MULTIROTOR, config.conflict,
                       config.simulation) as world2:
        for _ in range(7):
            world2.step()
        world2.restore(snap)
        tail_c = []
        for _ in range(30):
            world2.step()
            tail_c.append(_record_truth(world2))
    np.testing.assert_array_equal(np.asarray(tail_a), np.asarray(tail_c))


def test_a_resumed_particle_reproduces_the_run_bit_for_bit() -> None:
    """The full particle contract, all three resume paths against one reference:

    the reference continuation runs the live state in the live world on fresh streams;
    a clone resumed after ``world.restore`` must match it (state copy + channel copy +
    the reconstructed command cache all correct); a *pickled* clone resumed in a
    **different world of the cell** must match it too (the time-multiplex and the
    worker-process boundary). Identical streams are guaranteed reconstructible because
    stream addressing is stateless (``child``), never consumed (``spawn``)."""
    ctx = episode_context(_SCENARIO.n_pairs, MULTIROTOR, _CONFIG)
    resume_seq = child(_SEQ, 99)  # the continuation legs' own stream subtree

    with _world(geometry_child=0) as world:
        state = init_episode(world, ctx, EpisodeStreams.from_episode_seq(_SEQ))
        streams = EpisodeStreams.from_episode_seq(_SEQ)
        for _ in range(_PRE_STEPS):
            assert advance(world, ctx, state, streams)
        world_half = world.snapshot()
        episode_half = state.copy()
        blob = pickle.dumps((world_half, episode_half))  # the worker-bound form

        ref_result, ref_traj = _continue_to_end(
            world, ctx, state, EpisodeStreams.from_episode_seq(resume_seq)
        )
        assert ref_result.settled  # the reference reached the all-clear, not the cap

        world.restore(world_half)
        clone_result, clone_traj = _continue_to_end(
            world, ctx, episode_half.copy(), EpisodeStreams.from_episode_seq(resume_seq)
        )

    np.testing.assert_array_equal(ref_traj, clone_traj)
    np.testing.assert_array_equal(ref_result.min_sep, clone_result.min_sep)
    np.testing.assert_array_equal(ref_result.detected, clone_result.detected)
    assert (ref_result.n_los, ref_result.t_end, ref_result.settled) == (
        clone_result.n_los, clone_result.t_end, clone_result.settled
    )

    unpickled_world, unpickled_state = pickle.loads(blob)
    with _world(geometry_child=1) as other_world:  # a different particle's spawn
        for _ in range(5):
            other_world.step()
        other_world.restore(unpickled_world)
        pickled_result, pickled_traj = _continue_to_end(
            other_world, ctx, unpickled_state, EpisodeStreams.from_episode_seq(resume_seq)
        )
    np.testing.assert_array_equal(ref_traj, pickled_traj)
    np.testing.assert_array_equal(ref_result.min_sep, pickled_result.min_sep)


def test_clones_on_different_streams_diverge() -> None:
    """The other half of the splitting contract: fresh streams must actually reach every
    stochastic input, or resampled clones would replay their parent and the ladder would
    concentrate nothing. Two resumes of one particle on different subtrees must part."""
    ctx = episode_context(_SCENARIO.n_pairs, MULTIROTOR, _CONFIG)
    with _world(geometry_child=0) as world:
        state = init_episode(world, ctx, EpisodeStreams.from_episode_seq(_SEQ))
        streams = EpisodeStreams.from_episode_seq(_SEQ)
        for _ in range(_PRE_STEPS):
            advance(world, ctx, state, streams)
        particle = (world.snapshot(), state.copy())

        endpoints = []
        for leg in (101, 102):
            world.restore(particle[0])
            clone = particle[1].copy()
            leg_streams = EpisodeStreams.from_episode_seq(child(_SEQ, leg))
            for _ in range(80):
                if not advance(world, ctx, clone, leg_streams):
                    break
            endpoints.append(_record_truth(world))
    assert not np.array_equal(endpoints[0], endpoints[1])


def test_snapshot_refuses_a_mid_tick_boundary() -> None:
    """Pending stack commands live in neither the traffic arrays nor the clock, so a
    snapshot taken across them would silently lose them — refused, not risked. The spawn
    itself leaves the initial nominal commands pending, so the guard also fires there."""
    scenario = PairwiseEncounter(pairs=(1, 1), tlos=45.0)
    config = Config()
    geometry = scenario.draw_geometry(generator(child(_SEQ, 92)))
    with PairwiseWorld(scenario, geometry, MULTIROTOR, config.conflict,
                       config.simulation) as world:
        with pytest.raises(ValueError, match="post-step boundary"):
            world.snapshot()  # spawn stacked the nominals; nothing consumed them yet
        world.step()
        snap = world.snapshot()  # clean boundary now

        world.command((world.nominal_trk + 15.0) % 360.0, world.nominal_gs)
        with pytest.raises(ValueError, match="post-step boundary"):
            world.snapshot()
        with pytest.raises(ValueError, match="post-step boundary"):
            world.restore(snap)
        world.step()
        world.restore(snap)  # consumed again — both directions clean

        # the one sanctioned bypass: explicitly discarding a dead world's leftovers
        from cdarr.engine import discard_pending_commands

        world.command((world.nominal_trk + 30.0) % 360.0, world.nominal_gs)
        discard_pending_commands()
        world.snapshot()  # guard satisfied — the pending commands were renounced
