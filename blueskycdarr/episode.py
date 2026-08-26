"""One seeded episode: a batch of pairs flown from spawn to all-clear (CDaRR's loop).

The cadence structure is CDaRR's ``get_ipr_stochastic_env``: the engine integrates at
``dt``; detection, resolution and recovery run at ``cdr_dt`` on *perceived* state, and
their commands hold between ticks; the broadcast channel runs on its own per-aircraft
schedule (ADR 0002) — three clocks, deliberately decoupled.

Perception per aircraft at a CDR tick:

- **self** — a fresh noisy measurement of its own state (GNSS; no communication effects,
  matching CDaRR's ownship node and OpenCDaRR's navigation/communication split);
- **counterpart** — the last *delivered* broadcast, stale by latency plus holdover.

The episode ends when every pair has been past its closest approach (on truth) for
``done_timeout`` seconds, or at ``t_max``. Loss of separation is a true minimum pairwise
distance below ``rpz`` at any sampled step.

Stream layout (ADR 0004): the episode's SeedSequence fans into
``(geometry, navigation, measurement, reception, schedule)``, in that order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from blueskycdarr.adsl import BroadcastChannel, ContactTable, noisy_snapshot
from blueskycdarr.aircraft import AircraftSpec, as_pair
from blueskycdarr.config import Config
from blueskycdarr.detection import detect, pairs_all_clear
from blueskycdarr.engine import PairwiseWorld
from blueskycdarr.geo import track_components
from blueskycdarr.noise import DEFAULT_NOISE, NoiseShape
from blueskycdarr.recovery import DEFAULT_RECOVERY, Recovery, recovered_mask, worldview_sigmas
from blueskycdarr.resolution import DEFAULT_RESOLVER, Resolver, resolve
from blueskycdarr.rng import child, generator
from blueskycdarr.scenario import PairwiseEncounter
from blueskycdarr.state import StateArrays, counterpart

_EPS = 1e-9


@dataclass(frozen=True)
class EpisodeResult:
    """What one episode contributes to the estimate."""

    min_sep: np.ndarray  # m, true minimum separation per pair
    n_los: int
    detected: np.ndarray  # bool per pair: was the conflict ever perceived by either side
    t_end: float  # s
    settled: bool  # False = the t_max cap ended the run, not the all-clear


def run_episode(
    scenario: PairwiseEncounter,
    aircraft: AircraftSpec,
    config: Config,
    episode_seq: np.random.SeedSequence,
    recovery: Recovery = DEFAULT_RECOVERY,
    resolver: Resolver = DEFAULT_RESOLVER,
    noise: NoiseShape = DEFAULT_NOISE,
    recorder: Callable[[float, StateArrays, np.ndarray], None] | None = None,
) -> EpisodeResult:
    """Fly one batch of pairs and score it.

    ``aircraft`` is one model for both roles or an (ownship, intruder) pair;
    ``recovery`` selects the resume-navigation model (ADR 0006), ``resolver`` the
    resolution algorithm and ``noise`` the position-error shape (ADR 0007).
    ``recorder``, when given, is called once per step with ``(t, truth,
    pair_distances)`` — the hook the validation figures record trajectories through. A
    recorder observes; it must not influence the run (nothing it is handed is written
    back).
    """
    # child(), not spawn(): addressing is stateless, so calling run_episode twice with
    # the same SeedSequence draws the same streams (spawn would hand out new children).
    rng_geometry, rng_nav, rng_meas, rng_rx, rng_sched = (
        generator(child(episode_seq, i)) for i in range(5)
    )
    geometry = scenario.draw_geometry(rng_geometry)
    own_model, intr_model = as_pair(aircraft)
    n_pairs = scenario.n_pairs
    n = 2 * n_pairs
    sim = config.simulation
    rpz = config.conflict.rpz
    margin = config.conflict.resolution_margin

    with PairwiseWorld(scenario, geometry, aircraft, config.conflict, sim) as world:
        channel = BroadcastChannel(
            comm=config.comm,
            uncertainty=config.uncertainty,
            rng_measurement=rng_meas,
            rng_reception=rng_rx,
            rng_schedule=rng_sched,
            shape=noise,
        )
        channel.initialise(n)
        contacts = ContactTable.empty(world.truth())

        v_min = np.empty(n)
        v_max = np.empty(n)
        v_min[0::2], v_min[1::2] = own_model.v_min, intr_model.v_min
        v_max[0::2], v_max[1::2] = own_model.v_max, intr_model.v_max
        all_idx = np.arange(n)
        perm = counterpart(all_idx)
        all_seen = np.ones(n, dtype=bool)

        cmd_trk = world.nominal_trk.copy()
        cmd_gs = world.nominal_gs.copy()
        resolving = np.zeros(n, dtype=bool)
        ever_in_conflict = np.zeros(n, dtype=bool)

        # The FTR family's second hypothesis: the counterpart's velocity when its
        # conflict started (CDaRR's _intr_init_vel), NaN while unrecorded.
        initial_other_ve = np.full(n, np.nan)
        initial_other_vn = np.full(n, np.nan)
        # Worldview uncertainty for the probabilistic criteria (declared CI95s when
        # set — the exp5 calibration mismatch — else the actual ones, ADR 0007).
        rel_pos_sigma, rel_vel_sigma = worldview_sigmas(config.uncertainty)

        min_sep = np.full(n_pairs, np.inf)
        next_cdr = 0.0
        done_since: float | None = None
        settled = False
        step = 0
        t = 0.0

        while True:
            t = step * sim.dt
            truth = world.truth()
            distances = world.pair_distances()
            min_sep = np.minimum(min_sep, distances)
            if recorder is not None:
                recorder(t, truth, distances)

            channel.transmit_due(t, truth)
            channel.deliver_due(t, contacts)

            if t + _EPS >= next_cdr:
                own_view = noisy_snapshot(truth, all_idx, config.uncertainty, rng_nav, noise)
                other_view, seen = contacts.view_of_counterparts()
                conflicts = detect(
                    own_view, other_view, seen, rpz, config.conflict.t_lookahead
                )
                ever_in_conflict |= conflicts.in_conflict
                newly = conflicts.in_conflict & ~resolving
                initial_other_ve[newly] = other_view.gs_east[newly]
                initial_other_vn[newly] = other_view.gs_north[newly]
                resolving |= conflicts.in_conflict

                # Recovery FIRST, on the commands currently being flown (last tick's).
                # This one-tick lag is CDaRR's, by construction there (its FTR criteria
                # read ap.trk, the previously *stacked* command), and it is load-bearing:
                # a fresh resolution command is always flown for one CDR period before
                # the release criteria may judge it. Deciding on the same tick's fresh
                # command lets FTR release avoidance courses that were never flown —
                # measured against CDaRR itself: P(LoS) 0.99 instead of 0.03 (ADR 0006).
                recovered = recovered_mask(
                    recovery,
                    resolving=resolving,
                    conflicts=conflicts,
                    own=own_view,
                    other=other_view,
                    commanded_v=track_components(cmd_trk, cmd_gs),
                    initial_other_v=(initial_other_ve, initial_other_vn),
                    rpz=rpz,
                    margin=margin,
                    rel_pos_sigma=rel_pos_sigma,
                    rel_vel_sigma=rel_vel_sigma,
                )
                resolving[recovered] = False
                initial_other_ve[recovered] = np.nan
                initial_other_vn[recovered] = np.nan
                cmd_trk[recovered] = world.nominal_trk[recovered]
                cmd_gs[recovered] = world.nominal_gs[recovered]

                # Resolution for the aircraft still resolving; a just-released aircraft
                # flies nominal until (at the earliest) the next tick re-engages it.
                command = resolve(resolver, conflicts, own_view, rpz, margin, v_min, v_max)
                still = resolving[command.idx]
                cmd_trk[command.idx[still]] = command.trk[still]
                cmd_gs[command.idx[still]] = command.gs[still]
                world.command(cmd_trk, cmd_gs)

                # Done bookkeeping on ground truth, latched over done_timeout (CDaRR).
                truth_conflicts = detect(
                    truth, truth.reindexed(perm), all_seen, rpz, config.conflict.t_lookahead
                )
                if pairs_all_clear(truth_conflicts):
                    done_since = t if done_since is None else done_since
                else:
                    done_since = None

                missed = (
                    int(np.floor((t - next_cdr) / sim.cdr_dt)) + 1 if t > next_cdr else 1
                )
                next_cdr += missed * sim.cdr_dt

            if done_since is not None and t - done_since >= sim.done_timeout:
                settled = True
                break
            if t + _EPS >= sim.t_max:
                break

            world.step()
            step += 1

    detected = ever_in_conflict[::2] | ever_in_conflict[1::2]
    return EpisodeResult(
        min_sep=min_sep,
        n_los=int(np.sum(min_sep < rpz)),
        detected=detected,
        t_end=t,
        settled=settled,
    )
