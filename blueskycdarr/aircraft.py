"""The aircraft catalog — what "aircraft performance/type" means here (ADR 0005).

One :class:`AircraftModel` per airframe: the BlueSky performance id it flies under, the
speed envelope the *resolver* may command, and the turn-authority policy the engine writes
into the fork's per-aircraft limiter arrays (``bs.traf.max_tr`` / ``max_dtr2``). The fork
provides the *mechanism* (rate-limited turn dynamics in ``update_airspeed``); this catalog
is the *policy*, so a new airframe is a new value here, not a BlueSky edit.

The two entries mirror OpenCDaRR's registry (``M600`` / ``SMALL_FIXEDWING``,
``multirotor`` / ``fixedwing``) so a MixedVarLSENew design point transfers by label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AircraftModel:
    """One airframe: BlueSky carrier type, resolver speed envelope, turn authority.

    Attributes
    ----------
    label:
        The categorical value MixedVarLSENew sweeps (``"multirotor"`` / ``"fixedwing"``).
    bs_actype:
        The BlueSky/OpenAP performance id the aircraft is created as. This decides which
        *envelope BlueSky itself* enforces during integration; the resolver's envelope
        below may be tighter (and is, for the fixed-wing — see ``FIXEDWING``).
    v_min, v_max:
        The speed window (m/s) the resolver may command. MVP caps its speed resolutions
        into this window, so for a fixed-wing ``v_min`` is the stall floor.
    max_turn_rate, max_turn_accel:
        Turn-rate limit (deg/s) and its rate-of-change limit (deg/s^2), written into the
        fork's limiter arrays after creation. ``None`` leaves the limiter off
        (``inf``), which selects BlueSky's bank-angle turn path — the natural
        coordinated-turn model for a fixed-wing (turn rate ``g tan(phi) / V``).
    bank_deg:
        Default bank limit (deg) for the bank-angle turn path; ignored when a turn-rate
        limiter is set. ``None`` keeps BlueSky's default (25 deg).
    """

    label: str
    bs_actype: str
    v_min: float
    v_max: float
    max_turn_rate: float | None = None
    max_turn_accel: float | None = None
    bank_deg: float | None = None

    def __post_init__(self) -> None:
        if not self.v_max > self.v_min:
            raise ValueError(
                f"{self.label}: v_max ({self.v_max}) must exceed v_min ({self.v_min})"
            )


# DJI Matrice 600, exactly as CDaRR ran it: OpenAP rotor envelope (v_max 18 m/s from the
# fork's rotor/aircraft.json) and the fork's hard-coded M600 limiter values re-stated as
# policy (bluesky/traffic/traffic.py: max_tr 15 deg/s, max_dtr2 10 deg/s^2). v_min 0: a
# multirotor may stop, and CDaRR's MVP inherited a no-op floor (OpenAP rotor v_min is
# negative); 0 keeps speed commands physical without changing behaviour.
MULTIROTOR = AircraftModel(
    label="multirotor",
    bs_actype="M600",
    v_min=0.0,
    v_max=18.0,
    max_turn_rate=15.0,
    max_turn_accel=10.0,
)

# A small fixed-wing UAV with OpenCDaRR's SMALL_FIXEDWING numbers (their performance.py,
# from Reyner & Liem, Drones 2026: stall ~12 m/s, envelope top 25 m/s, operational bank
# ~44 deg). BlueSky has no small fixed-wing performance entry, so it flies under the
# "Amzn" OpenAP rotor id — the one shipped envelope (|v| <= 44 m/s) wide enough that
# BlueSky's own integrator never fights the 12..25 m/s window; the fixed-wing character
# comes from this catalog: the stall floor on resolver commands and the bank-limited turn
# path (limiter off). See ADR 0005 for why this carrier trick beats patching BlueSky.
FIXEDWING = AircraftModel(
    label="fixedwing",
    bs_actype="Amzn",
    v_min=12.0,
    v_max=25.0,
    bank_deg=44.0,
)

CATALOG: dict[str, AircraftModel] = {m.label: m for m in (MULTIROTOR, FIXEDWING)}

# One model for both roles, or (ownship, intruder) — CDaRR's aircraft_type_intruder
# made a mixed pair declarable; here it is a pair value (ADR 0007).
AircraftSpec = AircraftModel | tuple[AircraftModel, AircraftModel]


def aircraft_by_label(label: str) -> AircraftModel:
    """Registry lookup; fails fast with the known labels in the message."""
    try:
        return CATALOG[label]
    except KeyError:
        raise ValueError(f"unknown aircraft {label!r}. Known: {sorted(CATALOG)}") from None


def as_pair(spec: AircraftSpec) -> tuple[AircraftModel, AircraftModel]:
    """(ownship, intruder) models — a single model flies both roles."""
    if isinstance(spec, AircraftModel):
        return spec, spec
    return spec


def aircraft_from_spec(spec: object, *, source: object = "<spec>") -> AircraftSpec:
    """An aircraft spec from its file spelling: a label, a model, an
    ``{ownship: ..., intruder: ...}`` mapping, or a 2-sequence (ownship first)."""
    if isinstance(spec, AircraftModel):
        return spec
    if isinstance(spec, str):
        return aircraft_by_label(spec)
    if isinstance(spec, dict):
        unknown = sorted(set(spec) - {"ownship", "intruder"})
        if unknown or set(spec) != {"ownship", "intruder"}:
            raise ValueError(
                f"{source}: an aircraft mapping needs exactly the keys "
                f"'ownship' and 'intruder', got {sorted(spec)}"
            )
        return (
            aircraft_from_one(spec["ownship"], source=source),
            aircraft_from_one(spec["intruder"], source=source),
        )
    if isinstance(spec, list | tuple) and len(spec) == 2:
        return (
            aircraft_from_one(spec[0], source=source),
            aircraft_from_one(spec[1], source=source),
        )
    raise ValueError(
        f"{source}: aircraft must be a catalog label, a model, an "
        f"{{ownship, intruder}} mapping, or a 2-sequence, got {spec!r}"
    )


def aircraft_from_one(value: object, *, source: object) -> AircraftModel:
    if isinstance(value, AircraftModel):
        return value
    if isinstance(value, str):
        return aircraft_by_label(value)
    raise ValueError(f"{source}: expected a catalog label or model, got {value!r}")
