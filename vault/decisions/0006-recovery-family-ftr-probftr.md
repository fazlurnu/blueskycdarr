# ADR 0006 — Recovery is a three-model family; FTR's release timing is one tick behind, on purpose

- Status: accepted
- Date: 2026-08-26
- Deciders: Fazlur Rahman
- Extends: [[0003-declarative-experiments-opencdarr-style]]

## Context

The package shipped with past-CPA recovery only. CDaRR's studies (exp1–exp4) compare
three recoveries — Past-CPA, FTR ("free to return", the double criteria), and
Probabilistic FTR — and the request is to carry all three, so recovery must become a
swappable component like the aircraft.

The reference implementations are CDaRR's ``crr_resumenav_ftr.py`` (release when the
ownship's course clears the protected zone against **two intruder hypotheses** — it
keeps its current velocity, or it reverts to the velocity recorded when the conflict
started — both as *unconstrained line-CPA* miss distances, so an already-cleared pass
releases without waiting for CPA) and ``crr_resumenav_probabilistic_ftr.py`` (the same
two criteria as probabilities: release when ``P(DCPA > rpz) > gamma`` under Gaussian
uncertainty on relative position and velocity, computed analytically — a
projected-normal integral over the relative-velocity direction with a folded-normal
cross-line tail; their Appendix B validates it against Monte Carlo).

## Decision

- **Components.** `PastCPA(bouncing_guard=True)`, `FTR()`,
  `ProbabilisticFTR(gamma=0.999, k_theta=256)` in `cdarr/recovery.py`; a
  `recovery` slot on `Models`, a top-level `recovery:` run-file key (bare name or
  `{type: ..., params}` — the OpenCDaRR component rule), a declarable/sweepable
  `recovery` axis (labels, typed mappings, or instances; `gamma` sweeps via
  `Sweep(..., build=...)` in Python), and a `recovery=` parameter on `make_blackbox`.
- **The probability, ported not re-derived.** `_p_line_dcpa_exceeds` is CDaRR's
  `analytical_dcpa_prob_gt` specialised to the isotropic covariances this package
  produces and vectorised over aircraft, log-space weights included. Locked by a
  Monte-Carlo agreement test (60k samples, |Δp| < 0.01). Worldview sigmas follow
  CDaRR's runner: both aircraft at the configured CI95, combined in the relative frame
  (`sqrt(2) * ci95 / 2.448`); a *declared-vs-actual* split (their exp5) is deferred.
- **Release decisions lag the command by one tick — the load-bearing finding.** CDaRR's
  criteria read `ap.trk`/`ap.tas`, the command *stacked at the previous tick*, so a
  fresh resolution course is always flown for one CDR period before release may judge
  it. A first port evaluated the same tick's fresh command; the release then fired
  before the avoidance command was ever applied, and FTR collapsed. Measured on the
  identical condition (dpsi 90, tlos 180 s, lookahead 120 s, pos 10 m / vel 1 m/s,
  cruise 10.29 m/s, 300 encounters), against CDaRR run directly with an instrumented
  `get_desired_ownship_velocity`:

  | recovery | CDaRR (10k pairs) | same-tick port | lagged (final) |
  |---|---|---|---|
  | Past-CPA | 0.0004 | 0.0000 | 0.0000 |
  | FTR | 0.0211 | **0.9933** | 0.0267 |
  | Probabilistic FTR (γ 0.999) | 0.0005 | 0.0000 | 0.0000 |

  The episode therefore runs recovery *before* resolution each tick, on the standing
  commands; a just-released aircraft flies nominal until the next tick can re-engage it
  (CDaRR's `resopairs` removal has the same effect). Probabilistic FTR is insensitive
  to the ordering — at γ 0.999 its criteria demand ~3σ of clearance beyond the zone, so
  it never releases at the freshly-commanded boundary — which is why it, unlike FTR,
  looked fine under the buggy ordering: a warning about validating families member by
  member.
- **Conflict-start bookkeeping.** The counterpart's velocity at conflict onset is
  recorded per directed aircraft (CDaRR's `_intr_init_vel`), cleared on release, and
  re-recorded on re-engagement; unrecorded falls back to the current velocity.

## Alternatives rejected

- **Evaluate release on the same tick's fresh command.** Measured above: it changes
  FTR from a functioning (if aggressive) recovery into a non-recovery. Rejected on the
  cross-validation, and the ordering is now stated in the episode loop rather than left
  as an accident of plumbing.
- **Monte-Carlo clearance probabilities instead of the analytic integral.** Hundreds of
  draws per aircraft per tick per hypothesis; CDaRR built the analytic form precisely
  to avoid this, and validated it. Rejected.
- **A general covariance interface for the probabilistic criteria.** This package only
  produces isotropic measurement noise; carrying CDaRR's scalar/vector/matrix `_to_cov`
  generality here would be unread code (no unrequested generality). The isotropic
  specialisation is stated in the docstring.
- **Sharing one "criteria evaluator" between FTR and ProbabilisticFTR.** The
  deterministic and probabilistic forms read better side by side than behind an
  abstraction with a `gamma=None` mode. Duplication that helps the reader.

## Consequences

**Good:** the CDaRR recovery comparison (their exp1) is reproducible here — same
ordering of the three models, same failure signature (FTR shaves the margin: median
min-sep at the matched condition 54.7 m, just above the 50 m zone); the LSE study can
sweep recovery as a categorical axis.
**Cost:** recovery is stateful across ticks (the conflict-start record), carried as two
episode arrays; `recovered_mask` takes ten keyword arguments — flat and explicit rather
than a context object, revisit if a fourth model lands.
**Obligation:** engage/disengage behaviour is safety-relevant (the OpenCDaRR warning:
chatter, not noise, causes LoS) — any change to the tick ordering or the criteria must
re-run the matched-CDaRR comparison in `scripts/correctness_figures.py`.

## Relations

- [[0004-metric-seeding-and-crn]] — the worldview sigmas reuse the metric's CI95
  convention.
- [[0002-event-based-broadcast-channel]] — perceived geometry the criteria act on.
- vault/correctness.md §recovery — the comparison figures and the matched-CDaRR table.

## Update — 2026-08-26: the regression number after the CAS/ground frame fix

The table above was measured on bit-identical speed frames (both sides flying 1.0048x
their configured speeds). After the boundary conversion (ADR 0001 update) this package
flies configured speeds exactly, so the matched condition is deliberately no longer
bit-identical to CDaRR's: the standing regression now reads FTR 0.0400 (12/300) against
CDaRR's 0.0211, past-CPA and probabilistic FTR unchanged at 0/300. FTR's chatter is the
one model sensitive to the 0.5% speed difference; the conclusion and the ordering stand.
