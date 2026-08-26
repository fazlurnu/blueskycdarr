"""Conflict resolution — two resolvers behind one dispatch (ADR 0007).

:class:`MVP` is CDaRR's Modified Voltage Potential (``sim_models/cr_mvp.py``, itself
BlueSky's), horizontal branch: the predicted CPA offset is pushed out to
``rpz * margin`` and the velocity change spread over the time to CPA,

    dv = ((rpz_m / erratum - |dcpa|) * dcpa_vec) / (|tcpa| * |dcpa|)

with the ``erratum`` correction active while the intruder is outside the resolution
zone, and the exactly-head-on case deflected perpendicular to the line of sight.

:class:`VO` is the velocity-obstacle resolver, re-derived from OpenCDaRR's
``cr/vo.py`` (their re-derivation of CDaRR's): the set of ownship velocities leading to
an incursion is a cone in velocity space — apex at the intruder's velocity, axis along
the bearing to it, half-angle ``asin(rpz_eff / dist)`` — and the resolution is the
*shortest way out*: the velocity on the nearer cone edge closest to the current one.
The preferred velocity is deliberately the **current** velocity, not the nominal
(their measured finding: biasing toward the nominal re-enters the conflict and loses
separation — returning home is the recovery layer's job). With one intruder per
aircraft, the union-of-cones machinery reduces to the single cone ported here.

Each aircraft resolves from *its own* perceived geometry, so the two aircraft of a pair
manoeuvre cooperatively without coordination. Commanded speeds are capped into the
aircraft's catalog envelope (``v_min`` matters: a fixed-wing cannot resolve by
stopping). Both resolvers read ``resolution_margin`` from the conflict config — CDaRR's
global ``asas_marh``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blueskycdarr.detection import Conflicts
from blueskycdarr.geo import track_from_components
from blueskycdarr.state import StateArrays

_MIN_DCPA_M = 0.001  # CDaRR's threshold for the exactly-head-on degenerate case
_ANG_EPS = 1e-9  # rad: within this of the cone edge counts as outside (OpenCDaRR's)
_VEC_EPS = 1e-12  # m/s: a truly degenerate relative velocity is treated as not closing


@dataclass(frozen=True)
class MVP:
    """The Modified Voltage Potential resolver."""


@dataclass(frozen=True)
class VO:
    """The velocity-obstacle resolver (shortest way out of the cone)."""


Resolver = MVP | VO

DEFAULT_RESOLVER: Resolver = MVP()

_LABELS: dict[str, type] = {"mvp": MVP, "vo": VO}


def resolver_from_spec(spec: object, *, source: object = "<spec>") -> Resolver:
    """A resolver component from its file spelling: a bare name or ``{type: ...}``."""
    if isinstance(spec, Resolver):
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("type", None)
    else:
        raise ValueError(
            f"{source}: resolver must be a name or a {{type: ...}} mapping, got {spec!r}"
        )
    if name not in _LABELS:
        raise ValueError(f"{source}: unknown resolver {name!r}. Known: {sorted(_LABELS)}")
    try:
        return _LABELS[name](**params)
    except TypeError as err:
        raise ValueError(f"{source}: bad resolver parameters for {name!r}: {err}") from None


@dataclass
class ResolutionCommand:
    """Commanded track (deg) and ground speed (m/s) for the aircraft in ``idx``."""

    idx: np.ndarray
    trk: np.ndarray
    gs: np.ndarray


def resolve(
    resolver: Resolver,
    conflicts: Conflicts,
    own: StateArrays,
    rpz: float,
    margin: float,
    v_min: np.ndarray,
    v_max: np.ndarray,
) -> ResolutionCommand:
    """The selected resolver's commands for every aircraft flagged ``in_conflict``."""
    if isinstance(resolver, MVP):
        return resolve_mvp(conflicts, own, rpz, margin, v_min, v_max)
    return resolve_vo(conflicts, own, rpz, margin, v_min, v_max)


def resolve_mvp(
    conflicts: Conflicts,
    own: StateArrays,
    rpz: float,
    margin: float,
    v_min: np.ndarray,
    v_max: np.ndarray,
) -> ResolutionCommand:
    """MVP resolution commands for every aircraft flagged ``in_conflict``."""
    idx = np.flatnonzero(conflicts.in_conflict)
    if idx.size == 0:
        return ResolutionCommand(idx=idx, trk=np.empty(0), gs=np.empty(0))

    rpz_m = rpz * margin
    dist = conflicts.dist[idx]
    tcpa = conflicts.tcpa[idx]

    # Predicted CPA offset (east, north), self -> other, in the perceived frame:
    # dcpa_vec = drel + vrel * tcpa, with vrel = v_other - v_self = (du, dv).
    dcpa_e = conflicts.dx[idx] + conflicts.du[idx] * tcpa
    dcpa_n = conflicts.dy[idx] + conflicts.dv[idx] * tcpa
    dabs = np.hypot(dcpa_e, dcpa_n)

    # Degenerate head-on: no CPA offset to push on — deflect perpendicular to the line
    # of sight (CDaRR: dcpa[0] = dy/dist * eps, dcpa[1] = -dx/dist * eps).
    headon = dabs <= _MIN_DCPA_M
    if np.any(headon):
        dcpa_e = np.where(headon, conflicts.dy[idx] / dist * _MIN_DCPA_M, dcpa_e)
        dcpa_n = np.where(headon, -conflicts.dx[idx] / dist * _MIN_DCPA_M, dcpa_n)
        dabs = np.where(headon, _MIN_DCPA_M, dabs)

    # Intrusion to resolve, with the displaced-apex correction outside the zone.
    outside = (rpz_m < dist) & (dabs < dist)
    erratum = np.cos(
        np.arcsin(np.clip(rpz_m / dist, -1.0, 1.0)) - np.arcsin(np.clip(dabs / dist, -1.0, 1.0))
    )
    gain = np.where(outside, rpz_m / erratum - dabs, rpz_m - dabs)

    scale = gain / (np.abs(tcpa) * dabs)
    dv_e = scale * dcpa_e
    dv_n = scale * dcpa_n

    # CDaRR accumulates dv[idx1] -= dv_mvp and applies newv = v + dv, i.e. v - dv_mvp:
    # the ownship steers away from the predicted CPA offset.
    new_e = own.gs_east[idx] - dv_e
    new_n = own.gs_north[idx] - dv_n

    trk = track_from_components(new_e, new_n)
    gs = np.clip(np.hypot(new_e, new_n), v_min[idx], v_max[idx])
    return ResolutionCommand(idx=idx, trk=trk, gs=gs)


def resolve_vo(
    conflicts: Conflicts,
    own: StateArrays,
    rpz: float,
    margin: float,
    v_min: np.ndarray,
    v_max: np.ndarray,
) -> ResolutionCommand:
    """Single-cone shortest-way-out VO commands for every ``in_conflict`` aircraft.

    Vectorised port of OpenCDaRR's ``cr/vo.py``: cone apex at the counterpart's
    perceived velocity, axis along the bearing to it, half-angle
    ``asin(rpz * margin / dist)``. A current velocity already outside the cone — or a
    pair already inside the zone (no cone to leave), or a non-closing degenerate
    relative velocity — commands the current velocity (hold); an inside one is
    projected onto both cone edges and the nearer projection wins.
    """
    idx = np.flatnonzero(conflicts.in_conflict)
    if idx.size == 0:
        return ResolutionCommand(idx=idx, trk=np.empty(0), gs=np.empty(0))

    rpz_eff = rpz * margin
    dist = conflicts.dist[idx]
    pref_e = own.gs_east[idx]
    pref_n = own.gs_north[idx]
    # Relative to the cone apex (the counterpart's velocity, apex = own + (du, dv)),
    # the preferred velocity is own - other = -(du, dv).
    rel_e = -conflicts.du[idx]
    rel_n = -conflicts.dv[idx]

    axis_e = conflicts.dx[idx] / dist
    axis_n = conflicts.dy[idx] / dist
    half = np.arcsin(np.clip(rpz_eff / dist, 0.0, 1.0))

    along = rel_e * axis_e + rel_n * axis_n
    perp = rel_e * axis_n - rel_n * axis_e  # signed cross: + = right of the axis
    closing = (np.hypot(rel_e, rel_n) >= _VEC_EPS) & (along > 0.0)
    inside_cone = (
        closing & (dist > rpz_eff) & (np.arctan2(np.abs(perp), along) < half - _ANG_EPS)
    )

    # Both edge rays from the apex: the axis bearing rotated by +-half.
    bearing = np.arctan2(conflicts.dx[idx], conflicts.dy[idx])
    new_e = pref_e.copy()
    new_n = pref_n.copy()
    for sign in (-1.0, 1.0):
        ang = bearing + sign * half
        dir_e, dir_n = np.sin(ang), np.cos(ang)
        t = np.maximum(0.0, rel_e * dir_e + rel_n * dir_n)
        cand_e, cand_n = t * dir_e, t * dir_n
        d_sq = (cand_e - rel_e) ** 2 + (cand_n - rel_n) ** 2
        if sign < 0:
            best_sq, best_e, best_n = d_sq, cand_e, cand_n
        else:
            nearer = d_sq < best_sq
            best_e = np.where(nearer, cand_e, best_e)
            best_n = np.where(nearer, cand_n, best_n)
    apex_e = pref_e - rel_e
    apex_n = pref_n - rel_n
    new_e = np.where(inside_cone, apex_e + best_e, new_e)
    new_n = np.where(inside_cone, apex_n + best_n, new_n)

    trk = track_from_components(new_e, new_n)
    gs = np.clip(np.hypot(new_e, new_n), v_min[idx], v_max[idx])
    return ResolutionCommand(idx=idx, trk=trk, gs=gs)
