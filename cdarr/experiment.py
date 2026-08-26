"""The experiment layer — declare axes, get a table (OpenCDaRR's interface, ADR 0003).

An experiment is a mapping of parameter names to axes: :class:`Fixed` pins a value for
every condition, :class:`Sweep` enumerates levels, and the conditions are the cross
product in declaration order. The vocabulary is closed — an unknown name fails
immediately with the declarable list — and every name routes into the run: uncertainty,
communication, conflict and timing fields into :class:`~cdarr.config.Config`
sections, geometry slots into the scenario, ``aircraft`` into the models bundle.

    res = run_experiment(
        {"aircraft": Sweep(["multirotor", "fixedwing"]),
         "pos_ci95": Sweep([3.0, 10.0, 30.0, 92.6]),
         "vel_ci95": Fixed(1.0)},
        models=Models(aircraft=MULTIROTOR, scenario=PairwiseEncounter()),
        backend=MC(n_encounters=300),
        base_config=Config(),
        seed=7,
    )
    res.records()      # one dict per condition
    res.cell(pos_ci95=3.0, aircraft="multirotor")   # the raw estimate

Episode seeds hang off the root by episode index alone, so every condition replays the
same encounters and noise draws — common random numbers across the sweep (ADR 0004).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from cdarr.aircraft import AircraftSpec, aircraft_from_spec
from cdarr.config import Config, config_from_mapping, load_mapping
from cdarr.episode import run_episode
from cdarr.metrics import MonteCarloEstimate, combine
from cdarr.noise import DEFAULT_NOISE, NoiseShape, noise_from_spec
from cdarr.recovery import PastCPA, ProbabilisticFTR, Recovery, recovery_from_spec
from cdarr.resolution import DEFAULT_RESOLVER, Resolver, resolver_from_spec
from cdarr.rng import child, root_seed_sequence
from cdarr.scenario import PairwiseEncounter

# --- declaration -----------------------------------------------------------------------


@dataclass(frozen=True)
class Fixed:
    """One parameter held at ``value`` for every condition."""

    value: Any


@dataclass(frozen=True)
class Sweep:
    """One parameter enumerated over ``values``; ``build`` maps a readable level onto the
    value the run needs (so a component can be swept over a scalar axis)."""

    values: Sequence[Any]
    name: str | None = None
    build: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))  # lists compare; tuples hash

    def resolve(self, level: Any) -> Any:
        return level if self.build is None else self.build(level)


Axis = Fixed | Sweep


@dataclass(frozen=True)
class Models:
    """The component half of a run: what flies (one model, or an (ownship, intruder)
    pair), the encounter, and the resolver / recovery / noise-shape algorithms."""

    aircraft: AircraftSpec
    scenario: PairwiseEncounter
    recovery: Recovery = PastCPA()
    resolver: Resolver = DEFAULT_RESOLVER
    noise: NoiseShape = DEFAULT_NOISE


@dataclass(frozen=True)
class MC:
    """Plain Monte Carlo: size the effort by encounters, not runs.

    ``n_encounters`` is a floor. Episodes fly a whole batch of pairs, so the count is
    rounded up to whole episodes and the estimate reports the encounters actually flown —
    honest counts beat a trimmed sample.
    """

    n_encounters: int

    def __post_init__(self) -> None:
        if self.n_encounters < 1:
            raise ValueError(f"n_encounters must be >= 1, got {self.n_encounters}")


# Where each declarable name routes. The vocabulary is the union of these tables.
_UNCERTAINTY_FIELDS = frozenset(
    {"pos_ci95", "vel_ci95", "pos_ci95_declared", "vel_ci95_declared"}
)
_COMM_FIELDS = frozenset(
    {
        "reception_prob",
        "max_range_m",
        "latency_s",
        "broadcast_interval_s",
        "broadcast_jitter_s",
        "broadcast_random_phase",
    }
)
_CONFLICT_FIELDS = frozenset({"rpz", "t_lookahead", "resolution_margin"})
_SIMULATION_FIELDS = frozenset({"dt", "cdr_dt", "t_max", "done_timeout"})
_GEOMETRY_SLOTS = frozenset({"speed", "gs_intr", "dpsi", "dcpa", "dcpa_max", "tlos", "pairs"})
_COMPONENTS = frozenset({"aircraft", "recovery", "resolver", "noise"})
_KNOWN_KEYS = (
    _UNCERTAINTY_FIELDS
    | _COMM_FIELDS
    | _CONFLICT_FIELDS
    | _SIMULATION_FIELDS
    | _GEOMETRY_SLOTS
    | _COMPONENTS
)


@dataclass(frozen=True)
class Condition:
    """One cell of the sweep: the identifying levels and every declared value."""

    levels: tuple[tuple[str, Any], ...]
    values: tuple[tuple[str, Any], ...]


def expand(independent_vars: Mapping[str, Axis]) -> tuple[Condition, ...]:
    """The cross product of every :class:`Sweep`, in declaration order."""
    unknown = sorted(set(independent_vars) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {unknown}. Declarable: {sorted(_KNOWN_KEYS)}"
        )
    swept = [(key, axis) for key, axis in independent_vars.items() if isinstance(axis, Sweep)]
    fixed = [
        (key, axis.value) for key, axis in independent_vars.items() if isinstance(axis, Fixed)
    ]

    conditions: list[Condition] = []
    for combo in itertools.product(*(axis.values for _, axis in swept)):
        levels = tuple(
            (axis.name or key, level) for (key, axis), level in zip(swept, combo, strict=True)
        )
        resolved = tuple(
            (key, axis.resolve(level)) for (key, axis), level in zip(swept, combo, strict=True)
        )
        conditions.append(Condition(levels=levels, values=tuple(fixed) + resolved))
    return tuple(conditions)


def _apply(condition: Condition, config: Config, models: Models) -> tuple[Config, Models]:
    """Substitute a condition's values into fresh config/models (re-validated by replace)."""
    sections: dict[str, dict[str, Any]] = {}
    scenario_fields: dict[str, Any] = {}
    aircraft = models.aircraft
    recovery = models.recovery
    resolver = models.resolver
    noise = models.noise
    for key, value in condition.values:
        if key in _UNCERTAINTY_FIELDS:
            sections.setdefault("uncertainty", {})[key] = value
        elif key in _COMM_FIELDS:
            sections.setdefault("comm", {})[key] = value
        elif key in _CONFLICT_FIELDS:
            sections.setdefault("conflict", {})[key] = value
        elif key in _SIMULATION_FIELDS:
            sections.setdefault("simulation", {})[key] = value
        elif key in _GEOMETRY_SLOTS:
            # YAML sweeps deliver lists; the tuple-shaped slots need real tuples
            listy = key == "pairs" or (key in ("speed", "gs_intr") and isinstance(value, list))
            scenario_fields[key] = tuple(value) if listy else value
        elif key == "aircraft":
            aircraft = aircraft_from_spec(value, source="declaration")
        elif key == "recovery":
            recovery = recovery_from_spec(value, source="declaration")
        elif key == "resolver":
            resolver = resolver_from_spec(value, source="declaration")
        elif key == "noise":
            noise = noise_from_spec(value, source="declaration")
    for name, fields_ in sections.items():
        config = replace(config, **{name: replace(getattr(config, name), **fields_)})
    scenario = (
        replace(models.scenario, **scenario_fields) if scenario_fields else models.scenario
    )
    return config, Models(
        aircraft=aircraft, scenario=scenario, recovery=recovery, resolver=resolver, noise=noise
    )


# --- running ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentResult:
    """The sweep's outcome: one raw estimate per condition, plus the tabulations."""

    backend: MC
    seed: int
    conditions: tuple[Condition, ...]
    results: tuple[MonteCarloEstimate, ...]
    axes: tuple[str, ...]
    card_path: Path | None = None

    def records(self) -> list[dict[str, Any]]:
        """One flat dict per condition: the swept levels, then the estimate's metrics.

        Uncertainty is reported as the raw counts (``n_los``, ``n_encounters``) alone —
        no interval columns, by decision (ADR 0004 update).
        """
        rows = []
        for condition, estimate in zip(self.conditions, self.results, strict=True):
            row: dict[str, Any] = dict(condition.levels)
            row.update(
                n_encounters=estimate.n_encounters,
                n_los=estimate.n_los,
                p_los_run=estimate.p_los_run,
                detection_rate=estimate.detection_rate,
                median_min_sep=estimate.median_min_sep,
                n_unsettled=estimate.n_unsettled,
            )
            rows.append(row)
        return rows

    def to_dataframe(self) -> Any:
        import pandas  # deliberately lazy: tables are optional, simulation is not

        return pandas.DataFrame(self.records())

    def cell(self, **levels: Any) -> MonteCarloEstimate:
        """The raw estimate of one condition; with no arguments, the single condition."""
        matches = [
            est
            for cond, est in zip(self.conditions, self.results, strict=True)
            if all(dict(cond.levels).get(k) == v for k, v in levels.items())
        ]
        if len(matches) != 1:
            raise KeyError(
                f"{levels or 'no selector'} matches {len(matches)} conditions, need exactly 1"
            )
        return matches[0]

    def write_csv(self, path: str | Path) -> Path:
        """The tidy table (one row per condition) — the file MixedVarLSENew-adjacent
        tooling and plotting scripts read."""
        rows = self.records()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(rows[0]) if rows else []
        with path.open("w") as f:
            f.write(",".join(columns) + "\n")
            for row in rows:
                f.write(",".join(_csv_cell(row[c]) for c in columns) + "\n")
        return path

    def __len__(self) -> int:
        return len(self.conditions)


def _csv_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def run_experiment(
    independent_vars: Mapping[str, Axis],
    *,
    models: Models,
    backend: MC,
    base_config: Config | None = None,
    seed: int | None = None,
    n_jobs: int = 1,
    progress: bool = True,
    card_dir: str | Path | None = None,
) -> ExperimentResult:
    """Expand the declaration, run every condition, and tabulate.

    ``seed`` overrides ``base_config.seed`` as the reproducibility root. ``n_jobs`` fans
    the episodes of each condition out over processes (joblib; ``-1`` = every core) —
    each worker hosts its own BlueSky. ``card_dir`` writes a provenance card.
    """
    config = base_config if base_config is not None else Config()
    run_seed = config.seed if seed is None else seed
    conditions = expand(independent_vars)
    axes = tuple(name for name, _ in conditions[0].levels) if conditions else ()
    root = root_seed_sequence(run_seed)

    results = []
    for i, condition in enumerate(conditions):
        cond_config, cond_models = _apply(condition, config, models)
        _validate_declared_accuracy_is_read(cond_config, cond_models)
        estimate = _run_condition(cond_config, cond_models, backend, root, n_jobs)
        results.append(estimate)
        if progress:
            label = ", ".join(f"{k}={v}" for k, v in condition.levels) or "single condition"
            print(
                f"[{i + 1}/{len(conditions)}] {label}: "
                f"p_los={estimate.p_los_run:.4g} ({estimate.n_los}/{estimate.n_encounters})",
                flush=True,
            )

    result = ExperimentResult(
        backend=backend,
        seed=run_seed,
        conditions=conditions,
        results=tuple(results),
        axes=axes,
    )
    if card_dir is not None:
        from cdarr.card import write_card

        path = write_card(result, config, models, Path(card_dir))
        result = replace(result, card_path=path)
    return result


def _run_condition(
    config: Config,
    models: Models,
    backend: MC,
    root: np.random.SeedSequence,
    n_jobs: int,
) -> MonteCarloEstimate:
    """All episodes of one condition. Episode ``j`` draws from ``child(root, j)`` in every
    condition — the common-random-numbers layout (ADR 0004)."""
    per_episode = models.scenario.n_pairs
    n_episodes = math.ceil(backend.n_encounters / per_episode)
    seqs = [child(root, j) for j in range(n_episodes)]

    if n_jobs == 1:
        episodes = [
            run_episode(
                models.scenario, models.aircraft, config, seq,
                models.recovery, models.resolver, models.noise,
            )
            for seq in seqs
        ]
    else:
        from joblib import Parallel, delayed  # scheduling concern, imported where used

        episodes = Parallel(n_jobs=n_jobs)(
            delayed(run_episode)(
                models.scenario, models.aircraft, config, seq,
                models.recovery, models.resolver, models.noise,
            )
            for seq in seqs
        )

    return combine(episodes)


# --- run files -------------------------------------------------------------------------


def load_run(path: str | Path) -> tuple[Config, Models, MC]:
    """Parse a YAML run file into its three halves (format: configs/README.md)."""
    raw = load_mapping(path)
    config = config_from_mapping(raw, source=path)
    models = _models_from_mapping(raw, source=path)
    backend = _backend_from_mapping(raw.get("estimate"), source=path)
    return config, models, backend


def sweep_from_file(path: str | Path) -> dict[str, Axis]:
    """The optional ``sweep:`` block as axes: each entry a :class:`Sweep`, full factorial.

    This is the file spelling of the Python declaration, so the whole study is
    reproducible as ``file + seed`` (ADR 0003; a deliberate extension of the OpenCDaRR
    format, which keeps sweeps in Python).
    """
    raw = load_mapping(path).get("sweep") or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: 'sweep' must map parameter names to lists of levels")
    unknown = sorted(set(raw) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown sweep parameter(s) {unknown}. "
                         f"Declarable: {sorted(_KNOWN_KEYS)}")
    return {key: Sweep(values) for key, values in raw.items()}


def run_one_experiment(
    config: Config,
    models: Models,
    backend: MC,
    *,
    card_dir: str | Path | None = None,
    n_jobs: int = 1,
    progress: bool = True,
) -> ExperimentResult:
    """The all-``Fixed`` case: ``run_one_experiment(*load_run("configs/x.yaml"))``."""
    return run_experiment(
        {},
        models=models,
        backend=backend,
        base_config=config,
        n_jobs=n_jobs,
        progress=progress,
        card_dir=card_dir,
    )


def _validate_declared_accuracy_is_read(config: Config, models: Models) -> None:
    """Refuse a declared CI95 that nothing reads (the OpenCDaRR no-op guard).

    Only the probabilistic-FTR worldview consumes the declared accuracies; declaring
    one under any other recovery would run a calibration-mismatch study that silently
    measures nothing.
    """
    if config.uncertainty.declares_accuracy and not isinstance(
        models.recovery, ProbabilisticFTR
    ):
        raise ValueError(
            "pos_ci95_declared / vel_ci95_declared are declared but the recovery is "
            f"{type(models.recovery).__name__}, which never reads them — use "
            "ProbabilisticFTR, or drop the declared accuracies"
        )


def _models_from_mapping(raw: Mapping[str, Any], *, source: object) -> Models:
    aircraft_raw = raw.get("aircraft")
    if not isinstance(aircraft_raw, str | dict | list):
        raise ValueError(
            f"{source}: 'aircraft' must be a catalog label or an "
            f"{{ownship, intruder}} mapping, got {aircraft_raw!r}"
        )
    scenario_raw = raw.get("scenario")
    if not isinstance(scenario_raw, Mapping):
        raise ValueError(f"{source}: 'scenario' must be a mapping with type: pairwise")
    scenario_raw = dict(scenario_raw)
    kind = scenario_raw.pop("type", None)
    if kind != "pairwise":
        raise ValueError(f"{source}: scenario type must be 'pairwise', got {kind!r}")
    if "pairs" in scenario_raw:
        scenario_raw["pairs"] = tuple(scenario_raw["pairs"])
    for slot in ("speed", "gs_intr"):  # a YAML 2-list is a per-pair range (ADR 0007)
        if isinstance(scenario_raw.get(slot), list):
            scenario_raw[slot] = tuple(scenario_raw[slot])
    legal = set(PairwiseEncounter.__dataclass_fields__)
    unknown = sorted(set(scenario_raw) - legal)
    if unknown:
        raise ValueError(f"{source}: unknown scenario key(s) {unknown}. Legal: {sorted(legal)}")
    return Models(
        aircraft=aircraft_from_spec(aircraft_raw, source=source),
        scenario=PairwiseEncounter(**scenario_raw),
        recovery=recovery_from_spec(raw.get("recovery", "pastcpa"), source=source),
        resolver=resolver_from_spec(raw.get("resolver", "mvp"), source=source),
        noise=noise_from_spec(raw.get("noise", "gaussian"), source=source),
    )


def _backend_from_mapping(raw: Any, *, source: object) -> MC:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: 'estimate' must be a mapping with type: mc")
    raw = dict(raw)
    kind = raw.pop("type", None)
    if kind != "mc":
        raise ValueError(f"{source}: estimate type must be 'mc', got {kind!r}")
    if set(raw) != {"n_encounters"}:
        raise ValueError(f"{source}: estimate takes exactly 'n_encounters', got {sorted(raw)}")
    return MC(n_encounters=int(raw["n_encounters"]))
