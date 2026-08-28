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

Decomposition (for rare-event splitting)
----------------------------------------
The loop is factored so an estimator can pause, clone and resume it mid-flight — the
particle operations fixed-level IPS needs (OpenCDaRR's ADR 0017, mirrored here):

- :class:`EpisodeContext` — per-cell constants (config, models, derived tables), shared
  read-only by every particle;
- :class:`EpisodeState` — everything the loop carries between steps, and nothing else;
  cloning a particle is :meth:`EpisodeState.copy` beside an
  :class:`~blueskycdarr.engine.WorldSnapshot` of the engine;
- :class:`EpisodeStreams` — the forward random streams, deliberately *outside* the
  state: clones share their past (the state) and diverge only in their future noise
  (fresh streams per particle per level);
- :func:`init_episode` / :func:`advance` / :func:`episode_result` — build, step, score.

:func:`run_episode` composes exactly these pieces, so the plain Monte-Carlo path and a
splitting estimator fly the same loop by construction — there is no second
implementation to drift. ``advance``'s running-minimum separation (``EpisodeState.
min_sep``, monotone non-increasing) is the level function splitting ladders on.
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
from blueskycdarr.noise import DEFAULT_NOISE, LatencyBiased, NoiseShape
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


@dataclass(frozen=True)
class EpisodeStreams:
    """The episode's forward random streams (ADR 0004 order, geometry excluded).

    A bundle of generators, no state of its own to clone: :func:`run_episode` builds one
    per episode from children 1–4 of the episode sequence (child 0 is geometry, drawn
    once at spawn); a splitting estimator builds a *fresh* bundle per particle per level,
    which is exactly what makes two clones of one survivor diverge (the state/streams
    split :class:`~blueskycdarr.adsl.BroadcastChannel` documents).
    """

    navigation: np.random.Generator
    measurement: np.random.Generator
    reception: np.random.Generator
    schedule: np.random.Generator

    @classmethod
    def from_episode_seq(cls, episode_seq: np.random.SeedSequence) -> EpisodeStreams:
        """The streams a plain episode flies with — children 1–4, addressed statelessly
        (``child``, not ``spawn``), so building the bundle twice draws the same numbers."""
        nav, meas, rx, sched = (generator(child(episode_seq, i)) for i in range(1, 5))
        return cls(navigation=nav, measurement=meas, reception=rx, schedule=sched)


@dataclass(frozen=True)
class EpisodeContext:
    """Everything about an episode that is *not* particle state: config, models, and the
    tables derived from them. Built once per cell by :func:`episode_context` and shared
    read-only by every particle — a clone must never copy what cannot change.
    """

    config: Config
    recovery: Recovery
    resolver: Resolver
    noise: NoiseShape  # the broadcast measurement shape (rides the channel)
    own_noise: NoiseShape  # own-navigation shape: latency bias never applies (CDaRR's rule)
    v_min: np.ndarray  # per-aircraft speed envelope, ownship/intruder interleaved
    v_max: np.ndarray
    rel_pos_sigma: float  # worldview uncertainty for the probabilistic criteria (ADR 0007)
    rel_vel_sigma: float
    n_pairs: int
    all_idx: np.ndarray  # arange(n): every aircraft, for the ownship measurement
    perm: np.ndarray  # the counterpart permutation i ^ 1
    all_seen: np.ndarray  # truth sees everything: the done-check's `seen`

    @property
    def n(self) -> int:
        return 2 * self.n_pairs


def episode_context(
    n_pairs: int,
    aircraft: AircraftSpec,
    config: Config,
    recovery: Recovery = DEFAULT_RECOVERY,
    resolver: Resolver = DEFAULT_RESOLVER,
    noise: NoiseShape = DEFAULT_NOISE,
) -> EpisodeContext:
    """Derive the per-cell constants from the models — one place, so the plain-MC path
    and a splitting estimator cannot disagree on them."""
    own_model, intr_model = as_pair(aircraft)
    n = 2 * n_pairs
    v_min = np.empty(n)
    v_max = np.empty(n)
    v_min[0::2], v_min[1::2] = own_model.v_min, intr_model.v_min
    v_max[0::2], v_max[1::2] = own_model.v_max, intr_model.v_max
    all_idx = np.arange(n)
    rel_pos_sigma, rel_vel_sigma = worldview_sigmas(config.uncertainty)
    # The latency bias models a broadcast delay, so it never touches an aircraft's
    # own measurement (CDaRR's intruder-only rule): own-nav uses the base shape.
    own_noise = noise.base if isinstance(noise, LatencyBiased) else noise
    return EpisodeContext(
        config=config,
        recovery=recovery,
        resolver=resolver,
        noise=noise,
        own_noise=own_noise,
        v_min=v_min,
        v_max=v_max,
        rel_pos_sigma=rel_pos_sigma,
        rel_vel_sigma=rel_vel_sigma,
        n_pairs=n_pairs,
        all_idx=all_idx,
        perm=np.asarray(counterpart(all_idx)),
        all_seen=np.ones(n, dtype=bool),
    )


@dataclass
class EpisodeState:
    """One episode's mutable world beyond the engine: what the loop carries between
    steps, and nothing else.

    This is the episode half of an IPS particle (the engine half is
    :class:`~blueskycdarr.engine.WorldSnapshot`), under OpenCDaRR's no-hidden-state invariant:
    **everything that influences the future must live here** — a future-affecting value
    kept in a local, a global or a closure would be silently shared between clones, the
    exact corruption that is invisible at rare-event probabilities. The one deliberate
    exception is the engine's command de-duplication cache, which is *reconstructed*
    (not carried) on restore because it equals ``cmd_trk``/``cmd_gs`` at every post-step
    boundary (:meth:`~blueskycdarr.engine.PairwiseWorld.restore` documents the argument;
    ``tests/test_snapshot_parity.py`` pins it).

    ``nominal_trk``/``nominal_gs`` are per-particle *constants* (the spawned geometry's
    targets, read back on recovery) — carried here rather than on the world handle so a
    restored particle brings its own nominals with it.
    """

    nominal_trk: np.ndarray
    nominal_gs: np.ndarray
    channel: BroadcastChannel  # transmit schedule + in-flight messages (state only)
    contacts: ContactTable  # the last delivered broadcast per aircraft
    cmd_trk: np.ndarray  # the commands currently being flown (held between ticks)
    cmd_gs: np.ndarray
    resolving: np.ndarray  # bool: aircraft currently flying an avoidance command
    ever_in_conflict: np.ndarray  # bool: the detection record behind `detected`
    initial_other_ve: np.ndarray  # the FTR family's second hypothesis: the counterpart's
    initial_other_vn: np.ndarray  # velocity when its conflict started (NaN = unrecorded)
    min_sep: np.ndarray  # m per pair, running minimum — the splitting level function
    next_cdr: float = 0.0
    done_since: float | None = None  # truth all-clear latch (None = not currently clear)
    settled: bool = False
    ended: bool = False  # the episode has stopped (settled or t_max); advance is done
    step: int = 0  # integration steps completed; t = step * dt

    def copy(self) -> EpisodeState:
        """An independent particle: every mutable field duplicated, nothing aliased."""
        return EpisodeState(
            nominal_trk=self.nominal_trk.copy(),
            nominal_gs=self.nominal_gs.copy(),
            channel=self.channel.copy(),
            contacts=self.contacts.copy(),
            cmd_trk=self.cmd_trk.copy(),
            cmd_gs=self.cmd_gs.copy(),
            resolving=self.resolving.copy(),
            ever_in_conflict=self.ever_in_conflict.copy(),
            initial_other_ve=self.initial_other_ve.copy(),
            initial_other_vn=self.initial_other_vn.copy(),
            min_sep=self.min_sep.copy(),
            next_cdr=self.next_cdr,
            done_since=self.done_since,
            settled=self.settled,
            ended=self.ended,
            step=self.step,
        )


def init_episode(
    world: PairwiseWorld, ctx: EpisodeContext, streams: EpisodeStreams
) -> EpisodeState:
    """A fresh state for a just-spawned world.

    The channel's phase draw (random first-transmission slots) is the one init-time use
    of randomness, from the schedule stream — the same stream its jitter draws come
    from, so the ADR 0004 layout is unchanged by the decomposition. Commands start at
    the nominals the spawn already stacked.
    """
    channel = BroadcastChannel(
        comm=ctx.config.comm, uncertainty=ctx.config.uncertainty, shape=ctx.noise
    )
    channel.initialise(ctx.n, streams.schedule)
    contacts = ContactTable.empty(world.truth())
    n = ctx.n
    return EpisodeState(
        nominal_trk=world.nominal_trk.copy(),
        nominal_gs=world.nominal_gs.copy(),
        channel=channel,
        contacts=contacts,
        cmd_trk=world.nominal_trk.copy(),
        cmd_gs=world.nominal_gs.copy(),
        resolving=np.zeros(n, dtype=bool),
        ever_in_conflict=np.zeros(n, dtype=bool),
        initial_other_ve=np.full(n, np.nan),
        initial_other_vn=np.full(n, np.nan),
        min_sep=np.full(ctx.n_pairs, np.inf),
    )


def _cdr_tick(
    world: PairwiseWorld,
    ctx: EpisodeContext,
    state: EpisodeState,
    streams: EpisodeStreams,
    t: float,
    truth: StateArrays,
) -> None:
    """One detect/resolve/recover tick on perceived state, plus the done bookkeeping."""
    rpz = ctx.config.conflict.rpz
    margin = ctx.config.conflict.resolution_margin
    own_view = noisy_snapshot(
        truth, ctx.all_idx, ctx.config.uncertainty, streams.navigation, ctx.own_noise
    )
    other_view, seen = state.contacts.view_of_counterparts()
    conflicts = detect(
        own_view, other_view, seen, rpz, ctx.config.conflict.t_lookahead
    )
    state.ever_in_conflict |= conflicts.in_conflict
    newly = conflicts.in_conflict & ~state.resolving
    state.initial_other_ve[newly] = other_view.gs_east[newly]
    state.initial_other_vn[newly] = other_view.gs_north[newly]
    state.resolving |= conflicts.in_conflict

    # Recovery FIRST, on the commands currently being flown (last tick's).
    # This one-tick lag is CDaRR's, by construction there (its FTR criteria
    # read ap.trk, the previously *stacked* command), and it is load-bearing:
    # a fresh resolution command is always flown for one CDR period before
    # the release criteria may judge it. Deciding on the same tick's fresh
    # command lets FTR release avoidance courses that were never flown —
    # measured against CDaRR itself: P(LoS) 0.99 instead of 0.03 (ADR 0006).
    recovered = recovered_mask(
        ctx.recovery,
        resolving=state.resolving,
        conflicts=conflicts,
        own=own_view,
        other=other_view,
        commanded_v=track_components(state.cmd_trk, state.cmd_gs),
        initial_other_v=(state.initial_other_ve, state.initial_other_vn),
        rpz=rpz,
        margin=margin,
        rel_pos_sigma=ctx.rel_pos_sigma,
        rel_vel_sigma=ctx.rel_vel_sigma,
    )
    state.resolving[recovered] = False
    state.initial_other_ve[recovered] = np.nan
    state.initial_other_vn[recovered] = np.nan
    state.cmd_trk[recovered] = state.nominal_trk[recovered]
    state.cmd_gs[recovered] = state.nominal_gs[recovered]

    # Resolution for the aircraft still resolving; a just-released aircraft
    # flies nominal until (at the earliest) the next tick re-engages it.
    command = resolve(ctx.resolver, conflicts, own_view, rpz, margin, ctx.v_min, ctx.v_max)
    still = state.resolving[command.idx]
    state.cmd_trk[command.idx[still]] = command.trk[still]
    state.cmd_gs[command.idx[still]] = command.gs[still]
    world.command(state.cmd_trk, state.cmd_gs)

    # Done bookkeeping on ground truth, latched over done_timeout (CDaRR).
    truth_conflicts = detect(
        truth, truth.reindexed(ctx.perm), ctx.all_seen, rpz,
        ctx.config.conflict.t_lookahead,
    )
    if pairs_all_clear(truth_conflicts):
        state.done_since = t if state.done_since is None else state.done_since
    else:
        state.done_since = None

    missed = (
        int(np.floor((t - state.next_cdr) / ctx.config.simulation.cdr_dt)) + 1
        if t > state.next_cdr
        else 1
    )
    state.next_cdr += missed * ctx.config.simulation.cdr_dt


def advance(
    world: PairwiseWorld,
    ctx: EpisodeContext,
    state: EpisodeState,
    streams: EpisodeStreams,
    recorder: Callable[[float, StateArrays, np.ndarray], None] | None = None,
) -> bool:
    """One integration step of the episode loop; ``False`` when the episode has ended.

    The pre-decomposition loop body, verbatim, as one unit: observe truth (the running
    minimum ``state.min_sep`` — the splitting level function — updates here), fire and
    land broadcasts, run the CDR tick when due, judge the two stop conditions, and step
    the engine. On a stop (settled all-clear or the ``t_max`` cap) ``state.ended`` is
    latched and the engine is **not** stepped past the end, so ``state.step`` still
    names the last observed instant. An ended state's world may hold commands its final
    tick stacked but never flew — the one boundary that is *not* snapshotable — which is
    why the estimator keys on ``ended`` and never snapshots a terminal particle: an
    ended particle's future is closed, so only its episode half is ever read again.
    A level-crossing stop, by contrast, follows an advance that returned ``True`` (the
    engine stepped, the stack is drained), a valid place to
    :meth:`~blueskycdarr.engine.PairwiseWorld.snapshot`.
    """
    if state.ended:  # a closed episode stays closed — nothing to observe or draw
        return False
    sim = ctx.config.simulation
    t = state.step * sim.dt
    truth = world.truth()
    distances = world.pair_distances()
    np.minimum(state.min_sep, distances, out=state.min_sep)
    if recorder is not None:
        recorder(t, truth, distances)

    state.channel.transmit_due(
        t, truth, streams.measurement, streams.reception, streams.schedule
    )
    state.channel.deliver_due(t, state.contacts)

    if t + _EPS >= state.next_cdr:
        _cdr_tick(world, ctx, state, streams, t, truth)

    if state.done_since is not None and t - state.done_since >= sim.done_timeout:
        state.settled = True
        state.ended = True
        return False
    if t + _EPS >= sim.t_max:
        state.ended = True
        return False

    world.step()
    state.step += 1
    return True


def episode_result(state: EpisodeState, ctx: EpisodeContext) -> EpisodeResult:
    """Score a finished (or abandoned) state. Pure read; the arrays are copied out so
    the result stays valid if the state advances further (a splitting tail leg)."""
    detected = state.ever_in_conflict[::2] | state.ever_in_conflict[1::2]
    return EpisodeResult(
        min_sep=state.min_sep.copy(),
        n_los=int(np.sum(state.min_sep < ctx.config.conflict.rpz)),
        detected=detected,
        t_end=state.step * ctx.config.simulation.dt,
        settled=state.settled,
    )


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

    Composed from the decomposed pieces — :func:`episode_context`, :func:`init_episode`,
    :func:`advance`, :func:`episode_result` — and bit-identical to the monolithic loop
    it replaced (the seeded-reproduction lock in ``tests/test_episode.py``).
    """
    # child(), not spawn(): addressing is stateless, so calling run_episode twice with
    # the same SeedSequence draws the same streams (spawn would hand out new children).
    rng_geometry = generator(child(episode_seq, 0))
    streams = EpisodeStreams.from_episode_seq(episode_seq)
    geometry = scenario.draw_geometry(rng_geometry)
    ctx = episode_context(scenario.n_pairs, aircraft, config, recovery, resolver, noise)

    with PairwiseWorld(
        scenario, geometry, aircraft, config.conflict, config.simulation
    ) as world:
        state = init_episode(world, ctx, streams)
        while advance(world, ctx, state, streams, recorder):
            pass
    return episode_result(state, ctx)
