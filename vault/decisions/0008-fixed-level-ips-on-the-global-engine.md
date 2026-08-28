# ADR 0008 — Fixed-level IPS on the global engine: a particle is a snapshot pair, and streams stay outside the state

- Status: accepted
- Date: 2026-08-28
- Deciders: Fazlur Rahman
- Extends: [[0001-bluesky-fork-is-the-engine]], [[0004-metric-seeding-and-crn]]
- Mirrors: OpenCDaRR ADR 0017 (fixed-level IPS) and ADR 0022 (no intervals; ratio
  judgment against the MC anchor)

## Context

The Monte-Carlo backend starves in the rare regime: a 10⁴-encounter cell can read zero
losses, and the MixedVarLSENew integration needs P(LoS) estimates well below that.
OpenCDaRR solved this with fixed-level interacting-particle splitting (their ADR 0017,
Blom et al. 2007), designed from day one around clonable state — their state module
names BlueSky's global ``bs.traf`` as exactly what splitting cannot clone. This package
runs on that global engine, so the port had one open question: can the whole BlueSky
world be copied and put back, bit for bit?

It can. BlueSky's own create/delete/reset machinery forces every per-aircraft variable
through the ``TrafficArrays`` registry — an unregistered array would break resizing —
so a generic walk over the entity tree is complete *by construction*, including the
fork's ``prev_turnrate`` turn-limiter memory. What lives outside the registry is small
and explicit: the ``simtime`` clock and timers, ``sim.simt``, and ``traf.cd``'s
conflict bookkeeping. Restore-then-step reproduces an uninterrupted run exactly
(``tests/test_snapshot_parity.py``), at ~one integration step's cost per snapshot.

## Decision

- **A particle is one encounter, frozen at a post-step boundary**: an engine half
  (``blueskycdarr.engine.WorldSnapshot``) beside an episode half
  (``blueskycdarr.episode.EpisodeState``). The cloud is time-multiplexed through the one
  global world — evolve restores, copies, advances; freeze snapshots again. Resampling
  *shares* particle values; every evolution leg copies before mutating. Particles are
  plain data, hence picklable: replications fan out over joblib workers, each with its
  own engine (the episode fan-out's pattern).
- **Snapshots only at post-step boundaries.** ``sim.step`` drains the command stack at
  its start, so right after a step the queue is provably empty; snapshot and restore
  refuse a non-empty queue rather than silently dropping commands. Two consequences,
  both deliberate: the *spawn itself* leaves the nominal commands pending, so the first
  legal snapshot is after the first advance (the estimator materialises level-1
  particles by spawning and evolving, never by snapshotting a fresh spawn); and an
  encounter that *ends* mid-tick may hold stacked commands it never flew — so an ended
  particle keeps no world half at all (``world=None``), its future being closed, and
  the dead world's pending commands are **discarded the moment the ending is known**
  (``engine.discard_pending_commands``, called by the estimator's freeze) so they
  cannot trip the next restore's guard — found by the first MC-vs-IPS comparison run,
  pinned in ``tests/test_ips.py``. ``reset_world`` likewise clears commands with the
  traffic they addressed.
- **State and streams are separate arguments** (the OpenCDaRR §4 contract as code):
  ``BroadcastChannel`` holds schedule and in-flight messages only, its RNGs became
  method parameters; ``EpisodeStreams`` rides beside ``EpisodeState``, never inside it.
  The initial cloud samples geometry and broadcast phase from the particle seed (laid
  out exactly like an MC episode seed, ADR 0004); splitting acts on the forward CNS
  noise through fresh per-particle-per-level bundles. IPS therefore estimates the same
  per-encounter P(LoS) the MC backend does — what makes the anchor comparison, and the
  OpenCDaRR cross-engine comparison, meaningful.
- **The loop is decomposed, not duplicated**: ``episode_context`` / ``init_episode`` /
  ``advance`` / ``episode_result``, with ``run_episode`` composed from them —
  bit-identical to the pre-decomposition monolith (verified against a golden run of the
  old code). MC and IPS fly the same loop by construction.
- **Pairwise only, no tail leg** — the two deviations from the original. A scenario
  must spawn one pair per episode (``pairs=(1, 1)`` enforced): the batch grid is MC
  throughput, and a multi-pair particle would estimate P(any pair breaches). And a
  pair's K is decided at the ``rpz`` boundary (K ∈ {0, 1}), so OpenCDaRR's
  continuation leg — a fleet instrument for observing multiple losses — has nothing to
  measure here. A future interacting-fleet backend would revisit both together.
- **No intervals** (ADR 0022 mirrored, and this package's standing rule): the estimate
  is the mean over independent replications, collapses counted rather than hidden, and
  agreement with an anchor is judged on the ratio.

## Consequences

- ``blueskycdarr/ips.py`` (``estimate_rare_prob``); locks in ``tests/test_ips.py`` — exact at
  both ends (certain event ⇒ every shell survives; unreachable shell ⇒ clean collapse),
  the MC-anchor ratio on a moderate-probability cell, bit-for-bit seed reproducibility.
- The particle contract itself is pinned in ``tests/test_snapshot_parity.py``,
  including a pickled particle resumed in a different world.
- Cost shape: level 1 is dominated by per-particle spawns (reset + cre + creconfs),
  ~100 ms each; snapshot+restore is ~0.2 ms — under one integration step. Deeper
  levels are nearly pure stepping.
