"""Run configuration — frozen dataclasses, a closed YAML schema, fail-fast validation.

The conventions are OpenCDaRR's (their ADR 0023, mirrored here by ADR 0003):

- **Validation lives on the dataclass** (``__post_init__``), not the loader, so a config
  built in Python is checked exactly like one parsed from a file — and
  ``dataclasses.replace`` (the sweep's per-condition substitution) re-validates for free.
- **Closed schema.** Unknown keys fail immediately, at every level, with the legal list in
  the message.
- **Plain numbers live in flat blocks**; the component slots (``aircraft``,
  ``scenario``, ``recovery``, ``resolver``, ``noise``) are parsed by
  :func:`blueskycdarr.experiment.load_run`, never stored here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class UncertaintyConfig:
    """How well an aircraft is measured — 95% radial confidence intervals.

    ``pos_ci95`` (m) and ``vel_ci95`` (m/s) drive both the ownship's own GNSS measurement
    and the state its broadcasts carry (ADR 0002). ``0`` means a perfect sensor.

    ``pos_ci95_declared`` / ``vel_ci95_declared`` are what the *system believes* instead
    (CDaRR's exp5 calibration mismatch, ADR 0007): only the probabilistic-FTR worldview
    reads them; ``None`` keeps the belief matched to the truth. Declaring them with a
    recovery that never reads them is refused at declaration time — a no-op mismatch
    study would look publishable and measure nothing.
    """

    pos_ci95: float = 0.0
    vel_ci95: float = 0.0
    pos_ci95_declared: float | None = None
    vel_ci95_declared: float | None = None

    def __post_init__(self) -> None:
        _check(
            {
                "pos_ci95 >= 0": self.pos_ci95 >= 0,
                "vel_ci95 >= 0": self.vel_ci95 >= 0,
                "pos_ci95_declared >= 0": (
                    self.pos_ci95_declared is None or self.pos_ci95_declared >= 0
                ),
                "vel_ci95_declared >= 0": (
                    self.vel_ci95_declared is None or self.vel_ci95_declared >= 0
                ),
            }
        )

    @property
    def declares_accuracy(self) -> bool:
        """True when a declared CI deviates the worldview from the truth."""
        return self.pos_ci95_declared is not None or self.vel_ci95_declared is not None


@dataclass(frozen=True)
class CommConfig:
    """The broadcast channel between the two aircraft of a pair (ADR 0002).

    Attributes
    ----------
    reception_prob:
        Probability a transmitted message reaches the receiver, evaluated per
        transmission (Bernoulli). CDaRR's per-tick reception, made per-broadcast.
    max_range_m:
        Hard surveillance range: a transmission from farther away than this (true
        distance at transmit time) is never received. ``None`` disables the gate.
    latency_s:
        Time from transmission to the message becoming usable by the receiver. The
        received state is the *transmit-time* state, so latency ages every contact.
    broadcast_interval_s:
        Nominal gap between an aircraft's transmissions (ADS-L cadence).
    broadcast_jitter_s:
        Per-transmission slot dither: each gap is ``interval + U(-jitter, +jitter)``.
        ``0`` gives fixed gaps.
    broadcast_random_phase:
        Draw each aircraft's first transmission in ``U(0, interval)`` instead of at
        ``t = 0``, so a fleet does not transmit in lockstep.
    """

    reception_prob: float = 1.0
    max_range_m: float | None = None
    latency_s: float = 0.0
    broadcast_interval_s: float = 1.0
    broadcast_jitter_s: float = 0.0
    broadcast_random_phase: bool = False

    def __post_init__(self) -> None:
        _check(
            {
                "0 <= reception_prob <= 1": 0.0 <= self.reception_prob <= 1.0,
                "max_range_m > 0": self.max_range_m is None or self.max_range_m > 0,
                "latency_s >= 0": self.latency_s >= 0,
                "broadcast_interval_s > 0": self.broadcast_interval_s > 0,
                "0 <= broadcast_jitter_s < broadcast_interval_s": (
                    0 <= self.broadcast_jitter_s < self.broadcast_interval_s
                ),
            }
        )

    @property
    def range_gate_m(self) -> float:
        """The surveillance range as a number (``inf`` when the gate is off)."""
        return math.inf if self.max_range_m is None else self.max_range_m


@dataclass(frozen=True)
class ConflictConfig:
    """The separation standard and the resolver's margin.

    ``rpz`` (m) is the protected-zone radius — loss of separation is closer than this.
    ``t_lookahead`` (s) is the detection horizon. ``resolution_margin`` is MVP's
    resolution-zone buffer (>= 1; CDaRR's ``asas_marh``).
    """

    rpz: float = 50.0
    t_lookahead: float = 120.0
    resolution_margin: float = 1.05

    def __post_init__(self) -> None:
        _check(
            {
                "rpz > 0": self.rpz > 0,
                "t_lookahead > 0": self.t_lookahead > 0,
                "resolution_margin >= 1": self.resolution_margin >= 1,
            }
        )


@dataclass(frozen=True)
class SimulationConfig:
    """Timing: integration step, CDR cadence, and the two stop conditions.

    ``dt`` (s) is BlueSky's integration step. ``cdr_dt`` (s) is the detect/resolve/recover
    cadence (CDaRR's ``asas_dt``); commands hold between CDR ticks. A run ends when every
    pair has been past its closest approach with no conflict ahead for ``done_timeout``
    seconds, or at ``t_max`` — a ``t_max`` termination means the run never settled.
    """

    dt: float = 0.2
    cdr_dt: float = 1.0
    t_max: float = 300.0
    done_timeout: float = 10.0

    def __post_init__(self) -> None:
        _check(
            {
                "dt > 0": self.dt > 0,
                "cdr_dt >= dt": self.cdr_dt >= self.dt,
                "t_max > 0": self.t_max > 0,
                "done_timeout >= 0": self.done_timeout >= 0,
            }
        )


@dataclass(frozen=True)
class Config:
    """The parameter half of a run: everything numeric, nothing component-shaped."""

    seed: int = 0
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def __post_init__(self) -> None:
        _check({"seed >= 0": self.seed >= 0})

    def to_mapping(self) -> dict[str, Any]:
        """The file shape back — paste-able as a run file's parameter half (for the card)."""
        return {
            "seed": self.seed,
            "uncertainty": asdict(self.uncertainty),
            "comm": asdict(self.comm),
            "conflict": asdict(self.conflict),
            "simulation": asdict(self.simulation),
        }


_SECTIONS: dict[str, type] = {
    "uncertainty": UncertaintyConfig,
    "comm": CommConfig,
    "conflict": ConflictConfig,
    "simulation": SimulationConfig,
}

# Keys load_run owns (components + backend + the optional sweep block); listed here so the
# closed-schema check can name everything legal in one message.
_COMPONENT_KEYS = frozenset(
    {"aircraft", "scenario", "recovery", "resolver", "noise", "estimate", "sweep"}
)


def _check(constraints: Mapping[str, bool]) -> None:
    """Named predicates, so the error names every failed constraint at once."""
    failed = [name for name, ok in constraints.items() if not ok]
    if failed:
        raise ValueError(f"config constraints violated: {'; '.join(failed)}")


def _section(cls: type, raw: Mapping[str, Any], name: str, source: object) -> Any:
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{source}: {name!r} must be a mapping of fields, got {type(raw).__name__}"
        )
    legal = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(raw) - legal)
    if unknown:
        raise ValueError(f"{source}: unknown key(s) {unknown} in {name!r}. Legal: {sorted(legal)}")
    return cls(**raw)


def config_from_mapping(raw: Mapping[str, Any], *, source: object = "<mapping>") -> Config:
    """Build a :class:`Config` from the parameter half of a run mapping."""
    legal = set(_SECTIONS) | {"seed"} | _COMPONENT_KEYS
    unknown = sorted(set(raw) - legal)
    if unknown:
        raise ValueError(
            f"{source}: unknown top-level key(s) {unknown}. "
            f"Legal: {sorted(legal)} (format reference: configs/README.md)"
        )
    sections = {
        name: _section(cls, raw[name], name, source)
        for name, cls in _SECTIONS.items()
        if name in raw
    }
    return Config(seed=int(raw.get("seed", 0)), **sections)


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Read a run file as a plain mapping (shared by config and component parsing)."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a run file must be a YAML mapping, got {type(raw).__name__}")
    return raw
