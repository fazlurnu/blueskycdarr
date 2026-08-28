# BlueSkyCDaRR

BlueSkyCDaRR runs **CDaRR's stochastic pairwise conflict simulation** — BlueSky engine,
M600-class aircraft, state-based detection, both resolvers (MVP and VO), CDaRR's three
recovery models (past-CPA, FTR, probabilistic FTR), and its noise-shape family
(Gaussian, heavy-tail mixture, track-anisotropic) — rebuilt small and clean, with the
**declarative experiment interface of [OpenCDaRR](https://github.com/opencdarr)** on
top. Every CDaRR experiment family (exp1–exp5) is a declaration here, not a script:
heterogeneous per-pair speeds, declared-vs-actual accuracy mismatch, and mixed
ownship/intruder airframes included. Its job is to be the simulation
backend for the **MixedVarLSENew** level-set study: sweep aircraft type, position and
velocity uncertainty, reception probability and surveillance range; get P(LoS) per
condition, with provenance.

The simulation models the full ADS-L path between the two aircraft of an encounter:
measurement noise (95% CIs), a per-transmission reception probability, a hard
surveillance range, **broadcast jitter**, and **latency** carried as genuine message
staleness. Separation logic acts on *perceived* state only — what the channel delivered —
never on ground truth.

Relations, in one line each:

- **[CDaRR](https://github.com/fazlurnu/CDaRR)** — the original; this package is that
  simulation, restructured (same engine fork, same models, same conventions).
- **OpenCDaRR** (`~/Projects/OpenCDaRR`) — the engine-independent sibling; this package
  mirrors its experiment UI and vault conventions, and deliberately *keeps* the BlueSky
  runtime its ADR 0003 removed — two independent engines, one study.
- **MixedVarLSENew** (`~/Projects/MixedVarLSENew`) — the consumer;
  `blueskycdarr.blackbox.make_blackbox` implements its oracle contract.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,figures]"
```

The BlueSky engine installs from the CDaRR fork
(`git+https://github.com/fazlurnu/bluesky.git@CDaRR`) automatically — the fork's
turn-rate limiter is load-bearing, and the package refuses to run on stock BlueSky
rather than produce silently different dynamics (see
[`vault/decisions/0001`](vault/decisions/0001-bluesky-fork-is-the-engine.md)).

## Run an experiment

The whole MixedVarLSENew factorial, from one file:

```bash
python scripts/run_experiment.py configs/mixedvarlse.yaml --jobs -1
```

which writes `results/mixedvarlse.csv` (one row per condition: swept levels,
`n_encounters`, `n_los`, `p_los_run`, `detection_rate`, `median_min_sep`) and a
provenance card. The same study in Python:

```python
from blueskycdarr import Fixed, Sweep, MC, Models, Config, run_experiment
from blueskycdarr import MULTIROTOR, PairwiseEncounter

res = run_experiment(
    {"aircraft": Sweep(["multirotor", "fixedwing"]),
     "pos_ci95": Sweep([3.0, 10.0, 30.0, 92.6]),
     "vel_ci95": Sweep([1.0, 3.0]),
     "reception_prob": Fixed(0.8),
     "max_range_m": Fixed(3000.0),
     "recovery": Fixed("pastcpa")},   # or ftr / probabilistic_ftr — sweepable too
    models=Models(aircraft=MULTIROTOR, scenario=PairwiseEncounter()),
    backend=MC(n_encounters=300),
    base_config=Config(), seed=7, n_jobs=-1,
)
res.to_dataframe()
res.cell(aircraft="multirotor", pos_ci95=3.0, vel_ci95=1.0)   # the raw estimate
```

As the MixedVarLSENew oracle:

```python
from blueskycdarr.blackbox import make_blackbox
blackbox = make_blackbox(n_encounters=300, seed=7, n_jobs=-1)
# run_lse(space, blackbox, threshold=np.log10(0.02), store="...jsonl")
```

The run-file format is [`configs/README.md`](configs/README.md);
[`configs/mixedvarlse.yaml`](configs/mixedvarlse.yaml) is the fully-annotated reference.

## The JRESS experiments

The probabilistic-recovery paper's three experiments (its §5, Tables 1–3) live in
[`scripts/`](scripts/), one script each, sharing the paper's environment through
[`jress_common.py`](scripts/jress_common.py):

```bash
python scripts/exp1_crossing_angle.py   # recovery methods vs crossing angle (Table 1)
python scripts/exp2_gamma.py            # the confidence threshold gamma (Table 2)
python scripts/exp3_noise_models.py     # six navigation-noise models (Table 3)
```

Each defaults to a small **dummy** budget (sparse angles, 2 runs/condition) and takes
`--production` for the paper's full budget (100–1000 runs/condition — run those on the
server). Tables land in [`results/exp{1,2,3}/`](results/) with provenance cards, and
`results/` is committed: every table in the repo is traceable to a config, a seed, and
a code state.

## Test

```bash
pytest          # 81 tests: pure CDR chain, channel physics, recovery criteria
                # (incl. an analytic-vs-Monte-Carlo check), engine-backed episodes,
                # world-snapshot parity, and the IPS estimator locks
ruff check .    # lint, line length 99
```

## Documentation

- [`docs/`](docs/) — **start here to understand the code**: the guided reading order,
  the architecture and its diagrams, the anatomy of the episode loop, and the
  rare-event (IPS) estimator with level-design guidance.
- [`vault/correctness.md`](vault/correctness.md) — **the evidence**: every modelled
  effect validated against its own definition, with figures, including the recovery
  family cross-validated against CDaRR's own exp1 results.
- [`vault/decisions/`](vault/decisions/) — the eight ADRs behind the design (engine
  choice, channel model, experiment layer, metric and seeding, aircraft catalog,
  recovery family, CDaRR-parity batch, fixed-level IPS).
- [`notebooks/`](notebooks/) — executed notebooks:
  [`ips_walkthrough.ipynb`](notebooks/ips_walkthrough.ipynb) runs one rare-event
  splitting replication under the microscope — 50 particles, the culls, the
  resampling fan, the lineages that reach loss of separation — and proves it
  reproduces the packaged estimator bit for bit;
  [`experiment_example.ipynb`](notebooks/experiment_example.ipynb) walks the study
  axes (dpsi, pos/vel uncertainty, reception, surveillance range, aircraft type) one
  experiment at a time;
  [`bluesky_speed_command.ipynb`](notebooks/bluesky_speed_command.ipynb) demonstrates
  the CAS/ground-speed frame trap behind CDaRR's creeping ground speed, and the
  boundary conversion that fixes it here.

## License

MIT.
