# ADR 0003 — Experiments are declared OpenCDaRR-style; run files may carry their sweep

- Status: accepted
- Date: 2026-08-26
- Deciders: Fazlur Rahman

## Context

CDaRR ran experiments as bespoke scripts (`experiments/exp*.py`): nested loops, module
constants, `.npz` outputs — every new study a new script. OpenCDaRR replaced that with a
declarative layer the user already knows and MixedVarLSENew already calls: axes as
`Fixed`/`Sweep`, a closed vocabulary that routes each name into the run, `records()` /
`cell()` on the result, YAML run files, provenance cards. The brief for this package
says: the experiment-running interface should be that one.

## Decision

Mirror OpenCDaRR's interface at this package's (much smaller) scale, in
`blueskycdarr/experiment.py`:

- **Declaration.** `run_experiment(axes, models=Models(aircraft, scenario),
  backend=MC(n), base_config=Config(...), seed, n_jobs, card_dir)`; axes are
  `Fixed(value)` or `Sweep(values, name=None, build=None)`; conditions are the cross
  product in declaration order; an unknown key fails with the declarable list.
- **Vocabulary.** Uncertainty (`pos_ci95`, `vel_ci95`), comm (`reception_prob`,
  `max_range_m`, `latency_s`, `broadcast_interval_s`, `broadcast_jitter_s`,
  `broadcast_random_phase`), conflict (`rpz`, `t_lookahead`, `resolution_margin`),
  timing (`dt`, `cdr_dt`, `t_max`, `done_timeout`), geometry slots (`speed`, `gs_intr`,
  `dpsi`, `dcpa`, `dcpa_max`, `tlos`, `pairs`), and the one component slot `aircraft`
  (a catalog label or an `AircraftModel`). Values substitute via `dataclasses.replace`,
  so every condition re-validates itself.
- **Results.** A frozen `ExperimentResult` holding the raw `MonteCarloEstimate` per
  condition; `records()`, `to_dataframe()` (lazy pandas), `cell(**levels)` — the method
  MixedVarLSENew's notebook calls — and `write_csv()` for the tidy one-row-per-condition
  table.
- **Run files.** `run_one_experiment(*load_run("configs/x.yaml"))` for the all-fixed
  case; the file schema is `configs/README.md`. **One extension of the OpenCDaRR
  format:** an optional `sweep:` block (parameter -> levels, full factorial), read by
  `sweep_from_file`, so the whole 144-condition study is reproducible as *file + seed*
  from `scripts/run_experiment.py` — this package's studies are factorial sweeps by
  nature, and a study that lives half in a file and half in a script is the CDaRR
  failure mode this layer exists to end.
- **Provenance cards, CSV out, nothing committed.** A card (`vault`-style Markdown:
  declaration, models, config block, results table) is written when `card_dir` is
  passed; raw outputs land in gitignored `results/`.
- **Only the MC backend.** MixedVarLSENew's Monte-Carlo oracle needs `n_encounters` and
  `n_los`; rare-event splitting (IPS/AMS) stays in OpenCDaRR, which has it. No
  speculative backends here (no unrequested generality).

## Alternatives rejected

- **CDaRR-style experiment scripts.** The failure mode is documented in CDaRR itself:
  per-study constants drift, outputs lose their provenance, and every sweep is a rewrite.
  Rejected.
- **Import OpenCDaRR's experiment package and only swap the backend.** Their layer is
  coupled to their `Models` bundle (detectors, wind, navigation objects) and their cell
  runner; the coupling would cost more than the ~400 lines it saves. Rejected.
- **Sweep-in-Python only (OpenCDaRR's strict rule).** Their reason — components cannot
  be named in YAML — barely applies here (one component, label-named). The file-declared
  sweep buys single-artifact reproducibility for exactly this study. Rejected in favour
  of the `sweep:` extension, kept optional.
- **A cache layer keyed on code fingerprints (OpenCDaRR's `cache=`).** Valuable at their
  scale; here a full smoke condition runs in seconds and MixedVarLSENew already caches
  at the design-point level (its JSONL store). Deferred until a run hurts.

## Consequences

**Good:** the MixedVarLSENew notebook's blackbox transfers by changing imports
(`blueskycdarr.blackbox.make_blackbox` ships the adapter); studies diff as YAML.
**Cost:** two spellings of a sweep (Python and file) that must stay in step — both parse
into the same `Sweep` objects and share `_KNOWN_KEYS`, tested together.
**Obligation:** the vocabulary stays closed; a new declarable name lands with its route,
its README row and a test in the same change.

## Relations

- [[0004-metric-seeding-and-crn]] — what a condition's estimate contains and how seeds
  are laid out across conditions.
- [[0002-event-based-broadcast-channel]] — the comm vocabulary this layer exposes.
