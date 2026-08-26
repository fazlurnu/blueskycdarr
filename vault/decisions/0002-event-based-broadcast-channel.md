# ADR 0002 — Broadcast jitter, latency, reception and surveillance range are one event-based channel

- Status: accepted
- Date: 2026-08-26
- Deciders: Fazlur Rahman

## Context

CDaRR modelled communication inside the CDR tick: every `asas_dt` the intruder's state
was re-measured, kept with probability `reception_prob` (else held over from the previous
tick), and latency was approximated as a deterministic along-track position bias of
`-latency_s * gs`. That was sufficient for its sweeps, but this study adds two variables
the tick-locked model cannot express: **broadcast jitter** (transmissions do not align
with CDR ticks at all once slots dither) and a **surveillance range** (a message can fail
for geometry, not only for chance — and an aircraft that has *never* been heard must be
representable). The MixedVarLSENew design space sweeps reception probability and range
directly, so the channel is the part of the model under test.

## Decision

An event-based channel (`cdarr/adsl.py`) with three decoupled clocks — engine
`dt`, CDR `cdr_dt`, and per-aircraft broadcast schedules:

- **Schedule.** Aircraft `i` transmits at gaps `interval + U(-jitter, +jitter)` (optional
  random first phase). Gap draws accumulate in continuous time, so slots drift across the
  step grid rather than aliasing to it.
- **Content.** A transmission is a noisy snapshot of the *transmit-time* truth: position
  error isotropic 2D Gaussian with per-axis sigma `ci95 / 2.448`, velocity error likewise
  on the east/north components (CDaRR's measurement model, unchanged).
- **Gates, decided at transmission.** The message is received iff Bernoulli(`p_rx`)
  succeeds **and** the true transmitter-receiver distance at transmit time is within
  `max_range_m`. Both are transmit-path physics; a failed message is never queued.
- **Latency as staleness, not bias.** A received message becomes usable `latency_s`
  after transmission and carries its transmit-time state, so what the receiver acts on is
  *old truth*, aged `latency + holdover`. For straight flight this reproduces CDaRR's
  `-latency * gs` along-track offset exactly; through manoeuvres it stays physical where
  the bias approximation bends.
- **Hold last, never extrapolate.** The receiver keeps the last delivered message
  (CDaRR's holdover). Dead reckoning on stale contacts is a different CNS design and out
  of scope.
- **Own state is navigation, not communication.** Each CDR tick every aircraft measures
  *itself* fresh (position/velocity noise, no reception, range or latency), exactly
  CDaRR's ownship node and OpenCDaRR's navigation/communication split.

Two deliberate deviations from CDaRR, both locked by tests:

- **D1 — self-consistent snapshots.** CDaRR noised only the `gseast`/`gsnorth` component
  fields, so its detector (reading `trk`/`gs`) saw noise-free velocity while its resolver
  (reading components) saw noise. Here `trk`/`gs` are recomputed from the noised
  components: one measured velocity, every consumer.
- **D2 — no guaranteed first contact.** CDaRR copied a full noisy state to everyone
  before the first tick. With a surveillance range that would fake an in-range sighting
  at spawn; here a contact is invalid until the first genuine delivery, and an unseen
  counterpart is simply not detectable. With the gate off and zero latency the first
  broadcast lands at t = 0 and the difference vanishes.

With `jitter = 0`, `latency = 0`, no gate and `interval = cdr_dt`, the channel reduces to
CDaRR's per-tick model (test: `test_reduces_to_cdarr_channel_without_jitter_latency_and_gate`).

## Alternatives rejected

- **Keep CDaRR's along-track latency bias.** Cheaper, but wrong through turns, silent
  about jitter, and it double-books latency once real message timing exists. Rejected.
- **Extrapolate stale contacts (dead reckoning).** Changes the CDR question being studied;
  CDaRR held last-known, and comparability wins. Rejected (a future CNS variant, not a
  default).
- **Range-gate at delivery time instead of transmit time.** Radio truth is decided on the
  transmit path; a latency-delayed message from inside range should not be lost because
  the pair separated meanwhile. Rejected.

## Consequences

**Good:** jitter, latency, reception and range are orthogonal knobs with physical
meaning; each is validated against its own definition in `tests/test_adsl.py`.
**Cost:** an in-flight message queue per episode (bounded: one entry per transmission in
one latency window); slightly more state than CDaRR's four-node copy dance.
**Obligation:** the CDaRR-reduction test must stay green — it is the bridge that lets
CDaRR intuition transfer to this package.

## Relations

- [[0004-metric-seeding-and-crn]] — the channel draws from three of the episode's five
  streams (measurement, reception, schedule).
- [[0001-bluesky-fork-is-the-engine]] — the channel is pure; only the engine steps time.
