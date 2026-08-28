# Run files

A run file is one YAML mapping that fully determines a run: `file + seed -> result`.
The format follows OpenCDaRR's component rule (their ADR 0023, our ADR 0003): **plain
numbers live in flat, closed blocks; a pluggable component is declared as `type:` plus
its own constructor parameters.** Unknown keys fail immediately, at every level, with
the legal list in the message — the schema is closed on purpose.

## Blocks

| Block         | Keys                                                                 |
| ------------- | -------------------------------------------------------------------- |
| `seed`        | integer reproducibility root                                          |
| `scenario`    | `type: pairwise` + `speed`, `gs_intr`, `dpsi`, `dcpa`, `dcpa_max`, `tlos`, `pairs` |
| `aircraft`    | a catalog label — `multirotor` (M600) or `fixedwing` — or a mixed pair `{ownship: ..., intruder: ...}` (ADR 0007) |
| `recovery`    | `pastcpa` (default), `ftr`, `probabilistic_ftr` — bare name or `{type: ..., params}`, e.g. `{type: probabilistic_ftr, gamma: 0.999}` (ADR 0006) |
| `resolver`    | `mvp` (default) or `vo`; both read `resolution_margin` from `conflict` (ADR 0007) |
| `noise`       | `gaussian` (default), `mixture_gaussian`, `anisotropic_gaussian`, `anisotropic_mixture_gaussian` — bare name or `{type: ..., params}`; every shape delivers the configured radial CI95 (ADR 0007) |
| `uncertainty` | `pos_ci95` (m), `vel_ci95` (m/s) — 95% radial CIs; `pos_ci95_declared` / `vel_ci95_declared` deviate the probabilistic-FTR worldview from the truth (exp5 mismatch; refused unless the recovery reads them) |
| `comm`        | `reception_prob`, `max_range_m`, `latency_s`, `broadcast_interval_s`, `broadcast_jitter_s`, `broadcast_random_phase` |
| `conflict`    | `rpz`, `t_lookahead`, `resolution_margin`                             |
| `simulation`  | `dt`, `cdr_dt`, `t_max`, `done_timeout`                               |
| `estimate`    | `type: mc` + `n_encounters`                                           |
| `sweep`       | optional: parameter names -> lists of levels (full factorial)         |

Geometry slots (`dpsi`, `dcpa`, `gs_intr`) follow the OpenCDaRR convention: a number
pins the slot for every pair, `null` draws it per pair from the episode's seeded
geometry stream. `speed` and `gs_intr` additionally accept a 2-list `[min, max]` — a
per-pair U(min, max) draw, CDaRR's exp3/exp4 heterogeneous speeds (ADR 0007).

## The `sweep:` block

`sweep:` is this project's one deliberate extension of the OpenCDaRR format (which keeps
sweeps in Python): each entry names a declarable parameter and its levels, and the run is
the full factorial. It exists so the *entire* MixedVarLSENew study is reproducible from
one file — see [`mixedvarlse.yaml`](mixedvarlse.yaml), the fully-annotated reference file.
The declarable vocabulary is identical to the Python `Sweep`/`Fixed` API and closed the
same way. Component levels may be typed mappings, so a recovery sweep is expressible in
the file — `recovery: [pastcpa, ftr, {type: probabilistic_ftr, gamma: 0.999}]` — while a
readable scalar axis over a component parameter (e.g. a `gamma` sweep) uses the Python
spelling: `Sweep([0.9, 0.99, 0.999], name="gamma", build=lambda g: ProbabilisticFTR(g))`.

## Running

File-driven, from a shell:

```bash
.venv/bin/python scripts/run_experiment.py configs/mixedvarlse.yaml --jobs -1 --out results/
```

or the single-condition one-liner in Python:

```python
from blueskycdarr import load_run, run_one_experiment
run_one_experiment(*load_run("configs/mixedvarlse.yaml"))
```

Declared sweeps in Python reach the same machinery:

```python
from blueskycdarr import Fixed, Sweep, MC, Models, run_experiment
```
