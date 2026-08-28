"""Rare-event estimator — fixed-level interacting particle system (IPS) on BlueSky.

The port of OpenCDaRR's estimator (``opencdarr/estimate/ips.py``, their ADR 0017; Blom
et al. 2007) onto this package's particle machinery. Where plain Monte Carlo starves in
the rare regime — a 10⁴-encounter run can read *zero* events — IPS concentrates effort
on the trajectories heading toward the rare set and returns the small probability at a
usable cost.

**The method.** Nest the rare event in **levels** — shrinking shells of the running
minimum separation ``EpisodeState.min_sep`` (monotone non-increasing, so crossings are
one-way): a strictly decreasing sequence ``d_1 > … > d_m``, with ``d_m`` the rare
boundary (``rpz`` for loss of separation). Keep a fixed ``N`` particles; at each level,
evolve every particle until it either crosses (a *survivor*) or its encounter ends first
(*dropped*), then resample the survivors with replacement back to ``N``. The estimate is
the product of survival fractions ``prod_k (S_k / N)`` — no per-particle weights.

**A particle is one encounter** at a post-step boundary: an engine half
(:class:`~blueskycdarr.engine.WorldSnapshot`) beside an episode half
(:class:`~blueskycdarr.episode.EpisodeState`). The engine is a process-global singleton, so the
cloud is **time-multiplexed** through it: evolving a particle restores its snapshot,
copies its episode state, and advances; freezing a survivor snapshots the world again.
Clones of one survivor *share* the frozen particle and diverge through the **freshly
built** per-particle-per-level :class:`~blueskycdarr.episode.EpisodeStreams` — the initial
cloud samples geometry (and the broadcast phase), splitting acts on the forward CNS
noise (ADR 0017 §4), so IPS estimates the same per-encounter P(LoS) the Monte-Carlo
backend does. That is what makes cross-checking the two meaningful.

**Deviations from the OpenCDaRR original**, both consequences of this package being
pairwise by construction:

- ``scenario`` must spawn **one pair per episode** (``pairs=(1, 1)``): the batch grid is
  MC throughput — independent encounters flown together — not splitting structure, and a
  multi-pair particle would estimate P(any of k pairs breaches), a different number.
- **No tail leg.** OpenCDaRR flies final-cloud survivors past their first breach to
  observe K (losses per run) and A (aircraft in LoS), because a fleet can lose more than
  one pair. A pair cannot: crossing the ``rpz`` boundary *is* the loss, K ∈ {0, 1} is
  decided at the ladder's foot, and per-run and per-encounter rates coincide.

**Replications** (§5): within one run the particles interact through resampling (shared
ancestors), so a single run's spread understates the real one. ``reps`` independent runs
are averaged instead. No interval is reported — this package's outputs carry counts and
point estimates only, and agreement with the Monte-Carlo anchor (or with OpenCDaRR as
the independent second engine) is judged on the **ratio** of estimates, OpenCDaRR's
ADR 0022 mirrored.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from blueskycdarr.aircraft import AircraftSpec
from blueskycdarr.config import Config
from blueskycdarr.engine import (
    PairwiseWorld,
    WorldSnapshot,
    discard_pending_commands,
    reset_world,
)
from blueskycdarr.episode import (
    EpisodeContext,
    EpisodeState,
    EpisodeStreams,
    advance,
    episode_context,
    init_episode,
)
from blueskycdarr.noise import DEFAULT_NOISE, NoiseShape
from blueskycdarr.recovery import DEFAULT_RECOVERY, Recovery
from blueskycdarr.resolution import DEFAULT_RESOLVER, Resolver
from blueskycdarr.rng import child, generator, root_seed_sequence
from blueskycdarr.scenario import PairwiseEncounter

# What a stage of the ladder tells the outside world when it finishes: one line, no
# state. A rare-event cell is the run that is hours long and otherwise silent, and the
# survival fractions are exactly what say whether the ladder is collapsing before it
# reaches the boundary. Called after the work it reports, so it cannot influence it.
StageReport = Callable[[str], None]


def _no_report(note: str) -> None:
    """The default report: say nothing. One code path, instead of an ``if`` per stage."""


@dataclass(frozen=True)
class Particle:
    """One IPS particle: an encounter frozen at a post-step boundary.

    ``world is None`` marks an **ended** encounter (settled or ``t_max``): its future is
    closed, so no engine state is kept — such a particle rides through later levels as a
    value, surviving only by overshoot (its accumulated ``min_sep`` already below the
    target) and never touching the engine again. This is also why an end mid-tick — the
    one boundary that cannot be snapshotted, ``blueskycdarr.episode.advance`` documents it —
    never needs to be.

    Treat as immutable even though the episode half is technically mutable: resampling
    *shares* Particle values between clones, and the safety of that sharing rests on the
    evolution legs copying ``episode`` before advancing it. Plain data throughout, hence
    picklable — a particle can cross a process boundary to a worker with its own engine.
    """

    world: WorldSnapshot | None
    episode: EpisodeState

    @property
    def min_sep(self) -> float:
        """The encounter's running-minimum separation [m] — the level function."""
        return float(self.episode.min_sep[0])


@dataclass(frozen=True)
class IPSReplication:
    """One IPS replication: the estimate and the per-level survival fractions behind it."""

    prob: float  # P̂ = Π_k survival_k (0.0 if a level collapsed) — per-encounter P(LoS)
    levels: tuple[float, ...]  # the level distances d_1 … d_m [m]
    survival: tuple[float, ...]  # S_k / N per crossed level
    n_particles: int
    collapsed_at: int | None  # index of the level where S_k = 0, or None
    n_lineages: int  # distinct survivors behind the final cloud — its effective size


@dataclass(frozen=True, repr=False)
class IPSEstimate:
    """The replicated estimate: per-encounter P(LoS), averaged over independent runs."""

    p_los: float  # mean of the per-replication P̂
    reps: tuple[IPSReplication, ...]  # every replication, for inspection
    n_collapsed: int  # replications that hit an empty level (P̂ = 0)
    n_lineages: int  # distinct final-cloud lineages summed over replications

    def __repr__(self) -> str:
        """A labelled block, the estimate first — the generated dump would bury the few
        numbers the run was for under every replication's survival tuple (the idiom
        ``MonteCarloEstimate`` and ``Config`` set)."""
        if not self.reps:
            return f"{type(self).__name__}  no replications"
        depth = max(len(r.survival) for r in self.reps)
        means = [
            float(np.mean([r.survival[k] for r in self.reps if len(r.survival) > k]))
            for k in range(depth)
        ]
        levels = ", ".join(f"{d:g}" for d in self.reps[0].levels)
        lines = (
            f"survival   levels ({levels}) m | mean per level "
            + ", ".join(f"{s:.2g}" for s in means),
            f"collapsed  {self.n_collapsed} of {len(self.reps)} replications",
            f"lineages   {self.n_lineages} distinct survivors behind the final clouds "
            f"(per-replication record: reps)",
        )
        head = (f"{type(self).__name__}  P(LoS) {self.p_los:.3g} "
                f"over {len(self.reps)} replications")
        return head + "\n" + "\n".join(f"  {line}" for line in lines)


def _spawn_particle(
    seed: np.random.SeedSequence,
    scenario: PairwiseEncounter,
    aircraft: AircraftSpec,
    ctx: EpisodeContext,
) -> tuple[PairwiseWorld, EpisodeState]:
    """Materialise one initial-cloud particle: geometry spawned, episode state fresh.

    The particle seed has the *same internal layout as an MC episode seed* (ADR 0004):
    geometry from child 0, the broadcast phase from the schedule stream of children 1–4.
    Those are the only two initial-cloud draws — the init bundle's forward streams are
    never consumed here, because evolution draws from fresh per-level bundles instead.
    So the initial cloud samples exactly the encounter distribution MC integrates over,
    which is what keeps the two estimators cell-for-cell comparable (ADR 0017 §4).
    """
    geometry = scenario.draw_geometry(generator(child(seed, 0)))
    world = PairwiseWorld(
        scenario, geometry, aircraft, ctx.config.conflict, ctx.config.simulation
    )
    return world, init_episode(world, ctx, EpisodeStreams.from_episode_seq(seed))


def _evolve_to_level(
    world: PairwiseWorld,
    ctx: EpisodeContext,
    state: EpisodeState,
    streams: EpisodeStreams,
    target: float,
) -> None:
    """Advance until the running minimum crosses ``target`` or the encounter ends.

    Crossing is judged *between* whole steps, so a crossing registered by an advance
    that also ended the encounter still counts — survivor status is read off
    ``state.min_sep`` afterwards, never off which condition broke the loop (the
    OpenCDaRR check order, kept). A survivor's world sits one step past its crossing
    observation: whole-step granularity, the same overshoot the original accepts.
    """
    running = not state.ended
    while running and float(state.min_sep[0]) > target:
        running = advance(world, ctx, state, streams)


def _freeze(world: PairwiseWorld, state: EpisodeState) -> Particle:
    """Package an evolved state as a shareable particle; ended encounters keep no world.

    An ending exactly on a command-stacking CDR tick leaves the dead world's commands
    pending in the live global stack, where they would trip the *next* particle's
    restore guard — discarded here, the moment the ending is known (the regression the
    first MC-vs-IPS comparison run caught, pinned in ``tests/test_ips.py``).
    """
    if state.ended:
        discard_pending_commands()
        return Particle(world=None, episode=state)
    return Particle(world=world.snapshot(), episode=state)


def resample_level(
    evolved: Sequence[Particle],
    target: float,
    n_particles: int,
    seq: np.random.SeedSequence,
) -> tuple[float, list[Particle], int]:
    """One level's barrier: the survival fraction, the resampled cloud, its lineage count.

    Survivors are those that reached the level (``min_sep <= target``); they are drawn
    with replacement back up to ``n_particles``. An empty returned cloud means the level
    collapsed — the caller decides what to record. Independence between clones comes
    from the next level's fresh per-particle streams, not from this draw.

    The third value is how many **distinct** survivors the draw actually took — the
    cloud's effective size (``n_particles`` counts clones, not information). Taken from
    the draw itself rather than by de-duplicating the returned particles: clones *share*
    one Particle value, so counting by identity gives the right answer in-process and
    the wrong one after a worker pickles the cloud into several equal copies.
    """
    survivors = [p for p in evolved if p.min_sep <= target]
    fraction = len(survivors) / n_particles
    if not survivors:
        return fraction, [], 0
    idx = generator(seq).integers(0, len(survivors), size=n_particles)
    return fraction, [survivors[i] for i in idx], len(set(idx.tolist()))


def ips_once(
    scenario: PairwiseEncounter,
    aircraft: AircraftSpec,
    ctx: EpisodeContext,
    *,
    levels: Sequence[float],
    n_particles: int,
    seq: np.random.SeedSequence,
    on_stage: StageReport | None = None,
) -> IPSReplication:
    """One fixed-effort multilevel-splitting run: ``P̂ = Π_k S_k/N`` over ``levels``.

    ``levels`` is the decreasing sequence ``d_1 > … > d_m`` (``d_0 = ∞`` is implicit —
    every particle starts at ``min_sep = ∞``). The first level materialises the cloud
    (one spawn per particle seed); later levels restore survivors' snapshots into the
    one live world — a new spawn replaces the previous world in place, so no ``with``
    nesting, and the traffic is cleared once on the way out whatever happens. Returns
    ``prob = 0`` with ``collapsed_at`` set if some level has no survivors — a signal the
    levels are spaced too aggressively, not a real zero.

    A pure function of its arguments: ``seq`` is only ever *addressed* (``child``, the
    stateless form), never consumed, so calling it twice returns the identical
    replication — the reproducibility contract ``config + seed -> result`` (ADR 0004)
    extended to the ladder. Stream layout: child 0 of ``seq`` fans into per-particle
    init seeds; child 1 fans per level into ``n_particles`` evolution seeds plus one
    resampling seed.
    """
    report = on_stage if on_stage is not None else _no_report
    init_seq, evolve_seq = child(seq, 0), child(seq, 1)
    world: PairwiseWorld | None = None
    particles: list[Particle] = []
    survival: list[float] = []
    lineages = 0
    try:
        for k, target in enumerate(levels):
            level_seq = child(evolve_seq, k)
            evolved: list[Particle] = []
            for i in range(n_particles):
                streams = EpisodeStreams.from_episode_seq(child(level_seq, i))
                if k == 0:
                    world, state = _spawn_particle(
                        child(init_seq, i), scenario, aircraft, ctx
                    )
                elif particles[i].world is None or particles[i].min_sep <= target:
                    # ended (future closed) or overshot (already past this shell):
                    # decided by the accumulated minimum alone, no engine work
                    evolved.append(particles[i])
                    continue
                else:
                    world.restore(particles[i].world)
                    state = particles[i].episode.copy()
                _evolve_to_level(world, ctx, state, streams, target)
                evolved.append(_freeze(world, state))
            fraction, particles, lineages = resample_level(
                evolved, target, n_particles, child(level_seq, n_particles)
            )
            survival.append(fraction)
            if not particles:
                report(f"level {target:g} m collapsed")
                return IPSReplication(
                    prob=0.0, levels=tuple(levels), survival=tuple(survival),
                    n_particles=n_particles, collapsed_at=k, n_lineages=0,
                )
            report(f"level {target:g} m, {fraction:.1%} survived")
    finally:
        reset_world()
    return IPSReplication(
        prob=float(np.prod(survival)), levels=tuple(levels), survival=tuple(survival),
        n_particles=n_particles, collapsed_at=None, n_lineages=lineages,
    )


def combine_replications(results: Sequence[IPSReplication]) -> IPSEstimate:
    """Aggregate independent :func:`ips_once` results into the replicated point
    estimate: the mean of the per-replication ``P̂``, each unbiased. Collapsed
    replications (an empty level ⇒ ``P̂ = 0``) are counted, not hidden. No interval is
    computed — the replications ride along raw (``reps``), and agreement with an anchor
    is judged on the ratio of estimates (the module docstring says why)."""
    return IPSEstimate(
        p_los=float(np.mean([r.prob for r in results])),
        reps=tuple(results),
        n_collapsed=sum(1 for r in results if r.collapsed_at is not None),
        n_lineages=sum(r.n_lineages for r in results),
    )


def _validate(
    scenario: PairwiseEncounter, levels: Sequence[float], n_particles: int, reps: int
) -> tuple[float, ...]:
    if scenario.n_pairs != 1:
        raise ValueError(
            f"an IPS particle is one encounter, but this scenario spawns "
            f"{scenario.n_pairs} pairs per episode; use pairs=(1, 1) — the batch grid "
            "is MC throughput, not splitting structure (a multi-pair particle would "
            "estimate P(any pair breaches), a different number)"
        )
    ladder = tuple(float(d) for d in levels)
    if not ladder or any(d <= 0 for d in ladder) or any(
        a <= b for a, b in zip(ladder, ladder[1:], strict=False)
    ):
        raise ValueError(
            f"levels must be a strictly decreasing sequence of positive distances "
            f"(shrinking shells down to the rare boundary), got {list(levels)}"
        )
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    return ladder


def _stage_printer(prefix: str) -> StageReport:
    """A :data:`StageReport` in the experiment layer's progress idiom: one flushed line
    per finished stage, tagged with where it came from."""

    def report(note: str) -> None:
        print(f"{prefix} {note}", flush=True)

    return report


def estimate_rare_prob(
    scenario: PairwiseEncounter,
    aircraft: AircraftSpec,
    config: Config,
    recovery: Recovery = DEFAULT_RECOVERY,
    resolver: Resolver = DEFAULT_RESOLVER,
    noise: NoiseShape = DEFAULT_NOISE,
    *,
    levels: Sequence[float],
    n_particles: int,
    reps: int,
    seed: int,
    n_jobs: int = 1,
    progress: bool = False,
) -> IPSEstimate:
    """Estimate the rare per-encounter P(LoS) from ``reps`` independent IPS runs.

    The same argument shape as :func:`~blueskycdarr.episode.run_episode`, argument for
    argument — the same scenario (``pairs=(1, 1)``; refused otherwise), config and
    swappable models — so a per-cell setting reaches both estimators or neither, and
    the environment is assembled through the same composition root
    (:func:`~blueskycdarr.episode.episode_context`). What is IPS's own rides keyword-only: the
    ``levels`` ladder (strictly decreasing, ending at the rare boundary — ``rpz`` for
    loss of separation), the per-shell ``n_particles``, the ``reps`` replication count,
    and ``seed`` (the reproducibility root, as ``run_experiment`` takes it — replication
    ``r`` reads child ``r`` of the root, statelessly).

    ``n_jobs`` is scheduling only, never statistics: ``1`` (the default) runs the
    serial reference in-process; more fans **replications** out over joblib workers,
    each of which initialises its own engine on first use, exactly like the episode
    fan-out in :mod:`blueskycdarr.experiment`. Every replication is a pure function of its
    seed subtree, so the two paths return the identical estimate. ``progress`` prints
    one line per finished ladder stage (the experiment layer's idiom) — serial only,
    since worker stdout does not interleave usefully.
    """
    ladder = _validate(scenario, levels, n_particles, reps)
    ctx = episode_context(scenario.n_pairs, aircraft, config, recovery, resolver, noise)
    root = root_seed_sequence(seed)
    seqs = [child(root, r) for r in range(reps)]
    if n_jobs == 1:
        results = [
            ips_once(
                scenario, aircraft, ctx,
                levels=ladder, n_particles=n_particles, seq=sq,
                on_stage=_stage_printer(f"[rep {r + 1}/{reps}]") if progress else None,
            )
            for r, sq in enumerate(seqs)
        ]
        return combine_replications(results)
    from joblib import Parallel, delayed  # scheduling only, imported on use

    results = Parallel(n_jobs=n_jobs)(
        delayed(ips_once)(
            scenario, aircraft, ctx, levels=ladder, n_particles=n_particles, seq=sq
        )
        for sq in seqs
    )
    return combine_replications(results)
