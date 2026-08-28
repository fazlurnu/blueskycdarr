# ADR 0001 — The CDaRR BlueSky fork is the runtime engine, behind one boundary module

- Status: accepted
- Date: 2026-08-26
- Deciders: Fazlur Rahman

## Context

This project rebuilds CDaRR — the original BlueSky-based stochastic pairwise conflict
simulation — small and clean, to serve as the simulation backend for the MixedVarLSENew
level-set study. Its sibling OpenCDaRR decided the opposite (their ADR 0003: no BlueSky
runtime dependency) and re-derived the dynamics; the value of *this* package is precisely
that the answers come out of BlueSky's engine: the same OpenAP rotor performance, the
same `creconfs` conflict spawning, and the fork's turn-rate limiter that CDaRR's
published results were produced with. Two independently-built engines answering the same
design points is the cross-check the study wants.

CDaRR's README pins the engine: `fazlurnu/bluesky`, branch `CDaRR`. That branch adds two
things stock BlueSky lacks and this package needs: per-aircraft turn-rate limiter arrays
(`max_tr`, `max_dtr2`, applied inside `update_airspeed` — 15 deg/s and 10 deg/s² for the
M600), and `creconfs_dist`. Stock BlueSky would run the same code with airliner-style
bank-only turning and silently different results.

## Decision

- **Depend on the fork as an installable requirement**, pinned by branch:
  `bluesky-simulator @ git+https://github.com/fazlurnu/bluesky.git@CDaRR`.
- **Wrap it at exactly one seam**, `blueskycdarr/engine.py`: the only module that
  imports `bluesky`. It owns initialisation, creation, unit conversions (knots, NM),
  command stacking and stepping. Detection, resolution, recovery and the broadcast
  channel are pure numpy over state tables and never see `bs.*`.
- **Fail fast on the wrong engine.** `ensure_engine()` checks for the fork's limiter
  arrays after `bs.init` and raises with the install line. A silent fall-through to
  stock dynamics would be a result-changing configuration error, not a degraded mode.
- **One world per process.** BlueSky is a process-global singleton, so parallelism is
  joblib *processes* over episodes (CDaRR's `_joblib_inited` pattern); the boundary
  keeps a module flag and `PairwiseWorld` is a context manager that resets traffic.

## Alternatives rejected

- **Re-derive the dynamics, no BlueSky (the OpenCDaRR route).** Rejected: that project
  already exists at `~/Projects/OpenCDaRR`; duplicating it adds nothing, and the study
  loses the independent-engine cross-check.
- **Depend on stock `bluesky-simulator` and monkey-patch the turn limiter in.** Patching
  `update_airspeed` at runtime couples us to BlueSky internals more tightly than the
  fork does, invisibly to anyone reading the dependency list. Rejected.
- **Vendor the fork into this repository.** A frozen copy would drift from the branch
  CDaRR itself runs; the git-URL pin states the provenance and keeps one source. Rejected.
- **Use the user's local `~/Projects/bluesky` checkout.** It is shared by other projects
  and currently sits on `master` (no limiter); an editable dependency on a mutable
  checkout makes results irreproducible. Rejected.

## Consequences

**Good:** results carry BlueSky's dynamics by construction; every other module is
testable without the engine (40-test suite runs the pure chain in milliseconds).
**Cost:** installing from a git URL needs network and ~a minute of build; the fork's
branch can move (the pin is a branch, not a commit — an accepted looseness while the
fork is the same author's).
**Obligation:** anything unit-shaped (kts, NM, CAS-vs-ground-speed) must stay inside
`engine.py`; a `bs.` import anywhere else is a review failure.

## Relations

- [[0005-aircraft-catalog-two-airframes]] — the catalog is the policy the engine writes
  into the fork's limiter arrays.
- OpenCDaRR ADR 0003 (their vault) — the mirrored decision this one deliberately inverts.

## Update — 2026-08-26: the boundary also owns the CAS/ground-speed frame

Every speed BlueSky accepts (``cre``'s ``acspd``, ``creconfs``'s ``spd``, the ``SPD``
stack command) is **calibrated airspeed**, while this package computes and commands
ground speeds. At 100 m the frames differ by the air-density factor 1.0048 — harmless
once, but a loop that reads ground speed back and re-commands it through ``SPD``
*compounds* the factor per command. CDaRR's resolution path did exactly that: its drones
crept ~0.48% faster per re-command, the "ground speed keeps increasing" observation.
``notebooks/bluesky_speed_command.ipynb`` demonstrates both halves on the fork —
one command is perfectly stable; the feedback pattern ratchets ×1.15 in 30 s, matching
``1.00482^29`` analytically.

The engine now converts ground → CAS (``vtas2cas`` at the flight altitude) at creation
and in ``command()``, so a commanded ground speed is the ground speed flown, exactly
(locked by ``test_commanded_ground_speed_is_the_ground_speed_flown``). This is a
deliberate, stated results change for fixed seeds (the reproducibility invariant):
aircraft previously flew 1.0048x their configured speeds, and resolution speeds could
creep during long engagements. It is also a deliberate divergence from CDaRR, which
carries both artefacts.
