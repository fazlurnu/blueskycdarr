# Understanding this codebase

Start here. This folder is the guided tour; the deep truth lives in the module
docstrings (every file opens with *why it exists and what rule it enforces*), the
[ADRs](../vault/decisions/) (why the design is this way and not another), and the tests
(each one pins a behaviour to its definition — they are the locked specification).

## The 60-second mental model

BlueSkyCDaRR answers one question: **for a pair of drones spawned into a conflict, what
is the probability that CNS imperfections — GNSS noise, lost or late broadcasts — turn
the encounter into a loss of separation?** BlueSky (the CDaRR fork) flies the aircraft;
everything else — perception, detection, resolution, recovery, the broadcast channel —
is this package's own, pure NumPy over explicit state. Estimation happens at two
scales: plain Monte Carlo over batches of independent pairs, and fixed-level splitting
(IPS) for the rare regime where MC starves. A declarative experiment layer turns
parameter axes into tables, and a blackbox adapter feeds the results to MixedVarLSENew.

Four rules carry the whole design; almost every module is an enforcement of one of them:

1. **One engine seam.** Only [`engine.py`](../blueskycdarr/engine.py) imports
   `bluesky`. Everything above it is pure and testable without a simulator.
2. **Algorithms act on perceived state, never truth.** The CDR chain sees noisy
   own-measurements and stale broadcast contacts; only scoring reads truth.
3. **`config + seed -> result`, bit for bit.** Every stochastic component draws from
   its own addressed stream ([`rng.py`](../blueskycdarr/rng.py)); conditions share
   encounter draws (common random numbers); reruns reproduce exactly.
4. **Counts, not intervals.** Estimates report raw event counts; consumers derive
   whatever bounds they need. Rare-event agreement is judged on ratios to an anchor.

## Reading order

Each step assumes only the ones before it. Read the module docstring first — it is the
document; the code below it is the proof.

| # | Read | What you learn |
|---|------|----------------|
| 1 | [`config.py`](../blueskycdarr/config.py) | The vocabulary: frozen dataclasses, closed YAML schema, fail-fast validation |
| 2 | [`state.py`](../blueskycdarr/state.py), [`geo.py`](../blueskycdarr/geo.py) | `StateArrays` — the index-aligned table every view of the fleet uses; the `i ^ 1` counterpart convention |
| 3 | [`rng.py`](../blueskycdarr/rng.py) | The seed tree; why `child()` (stateless addressing) and not `spawn()` |
| 4 | [`engine.py`](../blueskycdarr/engine.py) | The BlueSky boundary: spawn, command, step — and the world snapshot machinery |
| 5 | [`scenario.py`](../blueskycdarr/scenario.py), [`aircraft.py`](../blueskycdarr/aircraft.py) | What gets spawned: encounter geometry slots, the airframe catalog |
| 6 | [`noise.py`](../blueskycdarr/noise.py), [`adsl.py`](../blueskycdarr/adsl.py) | How perception degrades: measurement shapes, the event-based broadcast channel |
| 7 | [`detection.py`](../blueskycdarr/detection.py), [`resolution.py`](../blueskycdarr/resolution.py), [`recovery.py`](../blueskycdarr/recovery.py) | The CDR chain, pure functions over perceived views |
| 8 | [`episode.py`](../blueskycdarr/episode.py) | Where everything composes — read [episode-anatomy.md](episode-anatomy.md) alongside |
| 9 | [`metrics.py`](../blueskycdarr/metrics.py), [`experiment.py`](../blueskycdarr/experiment.py), [`card.py`](../blueskycdarr/card.py), [`blackbox.py`](../blueskycdarr/blackbox.py) | The study layer: estimates, declarative sweeps, provenance, the mvlse adapter |
| 10 | [`ips.py`](../blueskycdarr/ips.py) | Rare events — read [rare-events.md](rare-events.md) alongside |

## The documents

- **[architecture.md](architecture.md)** — the layer diagram, the module dependency
  graph, the invariants table, and a map of the repository.
- **[episode-anatomy.md](episode-anatomy.md)** — one integration step under the
  microscope: the three clocks, the three views of the fleet, the CDR tick's ordering
  (and the one-tick lag that is load-bearing), and the context/state/streams
  decomposition that makes the loop pausable.
- **[rare-events.md](rare-events.md)** — the IPS estimator: what a particle is, how the
  cloud time-multiplexes through one global engine, how to design a level ladder, and
  what the 100k-encounter validation found.

## Where the proofs live

- [`vault/correctness.md`](../vault/correctness.md) — every modelled effect validated
  against its own definition, with figures.
- [`vault/decisions/`](../vault/decisions/) — ADRs 0001–0008. If a piece of code seems
  arbitrary, its ADR almost certainly says why it is not.
- [`tests/`](../tests/) — named as behaviour sentences; `pytest` runs the whole suite
  in seconds. The engine-backed modules (`test_episode`, `test_snapshot_parity`,
  `test_ips`) skip cleanly when the BlueSky fork is not installed.
- [`results/ips_mc_comparison/`](../results/ips_mc_comparison/) — the MC-vs-IPS
  validation runs, logs, scripts, and the 100k-encounter depth distribution.
