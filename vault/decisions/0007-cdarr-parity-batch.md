# ADR 0007 — The CDaRR parity batch: VO, noise shapes, speed ranges, declared accuracy, mixed pairs

- Status: accepted
- Date: 2026-08-26
- Deciders: Fazlur Rahman
- Extends: [[0003-declarative-experiments-opencdarr-style]], [[0006-recovery-family-ftr-probftr]]

## Context

With the recovery family in, the remaining gap to "this package fully replaces CDaRR"
was five capabilities its experiments used: the VO resolver (MVP-vs-VO comparisons),
the exp3/exp4 noise-shape sweep (mixture-Gaussian tails, along-track anisotropy),
per-pair heterogeneous speeds (exp3's U(10, 30) kts draws), the exp5 calibration
mismatch (an assumed accuracy decoupled from the true noise), and mixed
ownship/intruder aircraft types (CDaRR's ``aircraft_type_intruder``). Distance-based
spawning (``creconfs_dist``) was reviewed and dropped by decision — no forward-looking
study needs it.

## Decision

One batch, five additions, each ported from its authoritative source and validated
against its own definition (``tests/test_parity.py``):

- **VO resolver** (`resolution.py`): re-derived from **OpenCDaRR's ``cr/vo.py``** (their
  re-derivation of CDaRR's), reduced to the single cone our directed-pairwise world
  needs and vectorised. The preferred velocity is deliberately the *current* one — their
  measured finding: biasing toward the nominal re-enters the conflict; going home is the
  recovery layer's job. Resolution became a component (``MVP()`` / ``VO()``, registry
  ``mvp`` / ``vo``); both read ``resolution_margin`` from the conflict config (CDaRR's
  global ``asas_marh``), so the resolver markers stay parameter-free.
- **Noise shapes** (`noise.py`): CDaRR's ``noise_distributions.py`` ported whole —
  mixture-Gaussian, anisotropic (track-oriented), anisotropic-mixture — including its
  bisections, so **every shape delivers the configured radial CI95 exactly** (the
  containment guarantee; locked empirically for all four shapes). A component on the
  bundle (``noise:``, registry names), forwarded to both the broadcast snapshots and the
  own-navigation measurement, as CDaRR forwarded ``pos_dist`` to every node. Velocity
  noise stays Gaussian; latency is not a shape here (it is the channel's, ADR 0002).
  The default ``Gaussian`` is draw-stream-identical to the pre-shape code — existing
  seeded results do not move (locked).
- **Speed ranges** (`scenario.py`): ``speed`` / ``gs_intr`` accept ``(min, max)`` and
  draw per pair from the geometry stream (exp3/exp4's heterogeneity). Draw order is
  pinned after ``dpsi`` and ``dcpa``, so pinned-speed runs consume the same stream as
  before (locked).
- **Declared accuracy** (`config.py`): ``pos_ci95_declared`` / ``vel_ci95_declared`` on
  the uncertainty block — the probabilistic-FTR worldview reads them instead of the
  truth (CDaRR's ``assumed_confidence_interval``, exp5). Declaring one under any other
  recovery is refused at declaration time (OpenCDaRR's no-op guard: a mismatch study
  that nothing reads would look publishable and measure nothing).
- **Mixed pairs** (`aircraft.py`, `engine.py`): ``aircraft`` accepts one model or an
  ``(ownship, intruder)`` pair (``{ownship: ..., intruder: ...}`` in YAML); the engine
  applies creation types, turn-limiter arrays and resolver envelopes per role.

All five are declarable and sweepable through the existing vocabulary
(``resolver``, ``noise``, ``speed``, ``pos_ci95_declared``, ``aircraft``), so every
CDaRR experiment family (exp1–exp5) is now a declaration rather than a script.

## Alternatives rejected

- **Porting CDaRR's shapely-based ``cr_vo.py`` directly.** OpenCDaRR already re-derived
  it in pure numpy-compatible math and stabilised the preferred-velocity choice; porting
  the older geometry stack would add a dependency and re-inherit what they fixed.
- **Noise shape as a config number block.** A shape is an algorithm with parameters,
  not a number — it follows the component rule (``type:`` + params), like recovery.
- **Distance-based spawning.** Dropped by decision (see Context).
- **A general fleet (n aircraft per encounter).** The pairwise world is this package's
  scope statement; mixed *pairs* close the CDaRR gap without reopening the scope.

## Consequences

**Good:** CDaRR's exp1–exp5 are all reproducible from declarations; the LSE design
space can grow categorical axes (resolver, noise shape) without new code.
**Cost:** the models bundle is five fields; ``run_episode`` takes three component
parameters. The next component should prompt a bundle-shaped signature.
**Obligation:** every shape keeps the CI95 containment guarantee — a new shape lands
with its bisection and its calibration test, or not at all.

## Relations

- [[0004-metric-seeding-and-crn]] — the stream-stability locks added here defend it.
- [[0005-aircraft-catalog-two-airframes]] — the per-role engine application.
- OpenCDaRR ``cr/vo.py`` and its Phase-6 plan — the VO provenance.
