# ADR 0004 — The metric is P(LoS) per encounter; seeds form a condition-invariant tree

- Status: accepted
- Date: 2026-08-26
- Deciders: Fazlur Rahman

## Context

CDaRR reported IPR (intrusion prevention rate); MixedVarLSENew thresholds on
`log10 P(LoS)`. The two are complements (`IPR = 1 - P(LoS)`), but the study consumes raw
counts — its Jeffreys-corrected estimator wants `n_los` and `n_encounters`, not a
pre-divided rate. Separately, a 144-condition factorial lives or dies on comparability:
if every condition draws fresh encounter noise, half of an observed difference between
neighbouring cells is sampling noise.

## Decision

- **Metric.** An encounter is one ownship-intruder pair; **loss of separation** is its
  true minimum pairwise distance over the episode dipping below `rpz`.
  `MonteCarloEstimate` carries `n_encounters`, `n_los`, the raw per-encounter `min_sep`
  array (store the result, derive the statistic), `detection_rate` (was the conflict ever
  *perceived* by either side — LoS without detection is the comm-starvation signature),
  and `n_unsettled` (episodes ended by `t_max`, reported rather than hidden).
  `p_los_run` and a **Wilson** 95% interval are derived properties — Wilson because the
  study operates near p = 0 where the normal interval collapses to zero width.
- **Whole episodes, honest counts.** `MC(n_encounters)` is a floor: episodes fly
  `rows x cols` pairs each, the count rounds up, and the estimate reports the encounters
  actually flown.
- **Seed tree** (`cdarr/rng.py`, OpenCDaRR's contract): one root per run;
  `child(root, j)` is episode *j*'s sequence; each episode fans into five leaves —
  `(geometry, navigation, measurement, reception, schedule)`. Addressing uses `child`
  (stateless), never `spawn`, so a worker rebuilds exactly its slice and re-running an
  episode with the same sequence is bit-for-bit identical (locked by
  `test_a_seeded_episode_reproduces_bit_for_bit` — a test that caught the real
  stateful-`spawn` bug during development).
- **Common random numbers.** Episode seeds hang off the root **by episode index alone,
  not by condition**: every condition replays the same encounter geometry and the same
  noise/reception/schedule draws. Differences between cells are then differences in the
  physics, to first order, not in the sample. (Streams are consumed per CDR tick, so
  conditions with different episode lengths share the prefix — accepted.)

## Alternatives rejected

- **Report IPR as the primary metric.** The consumer thresholds on P(LoS); publishing the
  complement invites a silent `1 -` mistake at the seam. IPR stays a one-liner away.
- **Per-condition seed branches** (`root -> condition -> episode`). Statistically valid
  and simpler to reason about, but it throws away the variance reduction exactly where
  the LSE compares neighbouring cells. Rejected.
- **Normal (Wald) intervals.** Degenerate at zero losses — the regime the study lives in.
  Rejected.
- **Trimming the overshoot pairs to hit `n_encounters` exactly.** A trimmed tail biases
  nothing but hides that the batch is the sampling unit; honest counts are simpler and
  the consumer takes `n` as data. Rejected.

## Consequences

**Good:** the estimate plugs into MixedVarLSENew's estimator unchanged
(`cdarr/blackbox.py` is ~40 lines); sweeps are comparable cell to cell; every
stochastic path is reconstructable from `(seed, episode index)`.
**Cost:** CRN couples conditions statistically — the per-cell Wilson interval is valid,
but *differences* between cells are correlated (conservative for the LSE's use).
**Obligation:** a change that alters results for a fixed seed is a finding, not noise —
it must be deliberate and stated (the OpenCDaRR reproducibility invariant).

## Relations

- [[0002-event-based-broadcast-channel]] — which stream each channel effect draws from.
- [[0003-declarative-experiments-opencdarr-style]] — where the estimate surfaces
  (`records()`, `cell()`, CSV).

## Update — 2026-08-26: interval columns removed

By decision, the ``p_los_lo`` / ``p_los_hi`` Wilson columns are removed from the
estimate, the records, the CSVs, the figures and the notebooks — everywhere, and they
are not coming back. Uncertainty reporting is the raw counts alone: ``n_los`` and
``n_encounters`` stay first-class, so any consumer that wants bounds (or the
MixedVarLSENew blackbox's standard error, which is unaffected — it is part of the
``mvlse`` contract and derives from the same counts) computes them downstream. The
"Wilson because Wald degenerates at zero" argument above is retained as history; the
decision that supersedes it is: report counts, not intervals.
