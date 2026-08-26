"""State-based conflict detection, per directed pair — CDaRR's detector without the matrix.

The math is CDaRR's ``sim_models/cd_statebased.py`` (itself BlueSky's state-based CD):
closest point of approach from relative position and relative velocity, conflict when the
predicted miss distance is inside the protected zone and the conflict window opens within
the lookahead. Two deliberate reductions (ADR 0002):

- **Directed 1-vs-1 instead of N x N.** Pairs are independent by construction, so each of
  the ``2 n`` aircraft is tested only against its counterpart, with *its own* perceived
  views — the asymmetric situational awareness CDaRR builds with its ADSL node pairs.
- **Horizontal only.** Every aircraft flies the same altitude with zero vertical speed, so
  CDaRR's vertical branch is always-true bookkeeping here and is dropped.

An aircraft whose counterpart was never heard (invalid contact) perceives no conflict —
that is the surveillance-range effect, not an edge case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cdarr.geo import enu_offset
from cdarr.state import StateArrays

_MIN_REL_SPEED_SQ = 1e-6  # CDaRR's floor against division by zero for parallel flight


@dataclass
class Conflicts:
    """Per directed aircraft ``i``: the CPA prediction of ``i`` against its counterpart.

    ``in_conflict`` is False wherever ``seen`` is False; the kinematic arrays are still
    populated from the (placeholder) contact values there and must be read under the mask.
    """

    in_conflict: np.ndarray  # bool
    seen: np.ndarray  # bool — contact available at all
    dx: np.ndarray  # m east, self -> other (perceived)
    dy: np.ndarray  # m north
    du: np.ndarray  # m/s east, other minus self (relative velocity of the other aircraft)
    dv: np.ndarray  # m/s north
    dist: np.ndarray  # m
    tcpa: np.ndarray  # s
    dcpa: np.ndarray  # m
    t_in: np.ndarray  # s, time to conflict-window entry (tLOS)


def detect(
    own: StateArrays,
    other: StateArrays,
    seen: np.ndarray,
    rpz: float,
    t_lookahead: float,
) -> Conflicts:
    """CPA-based conflict detection of every aircraft against its counterpart.

    ``own`` row ``i`` is aircraft ``i``'s view of itself; ``other`` row ``i`` its view of
    its counterpart (already reindexed); ``seen`` masks rows with a usable contact.
    """
    dx, dy = enu_offset(own.lat, own.lon, other.lat, other.lon)
    dist = np.hypot(dx, dy)

    # Relative velocity of the other aircraft with respect to self. This matches the
    # element convention of CDaRR's matrix detector (du[i, j] = v_j - v_i), where the
    # same tcpa formula below yields tcpa > 0 for a closing pair.
    du = other.gs_east - own.gs_east
    dv = other.gs_north - own.gs_north
    rel_speed_sq = np.maximum(du * du + dv * dv, _MIN_REL_SPEED_SQ)
    rel_speed = np.sqrt(rel_speed_sq)

    tcpa = -(du * dx + dv * dy) / rel_speed_sq
    dcpa_sq = np.abs(dist * dist - tcpa * tcpa * rel_speed_sq)
    dcpa = np.sqrt(dcpa_sq)

    inside_zone = dcpa_sq < rpz * rpz
    half_window = np.sqrt(np.maximum(0.0, rpz * rpz - dcpa_sq)) / rel_speed
    t_in = np.where(inside_zone, tcpa - half_window, np.inf)
    t_out = np.where(inside_zone, tcpa + half_window, -np.inf)

    in_conflict = inside_zone & (t_out > 0.0) & (t_in < t_lookahead) & seen
    return Conflicts(
        in_conflict=in_conflict,
        seen=np.asarray(seen, dtype=bool),
        dx=dx,
        dy=dy,
        du=du,
        dv=dv,
        dist=dist,
        tcpa=tcpa,
        dcpa=dcpa,
        t_in=t_in,
    )


def pairs_all_clear(truth_conflicts: Conflicts) -> bool:
    """The episode's done condition: every pair past its closest approach, on truth.

    CDaRR's ``_check_tcpa_tinhor_per_pair`` requires ``tcpa < 0`` and ``t_in < 0`` for all
    pairs; since the window entry never lies after CPA (``t_in <= tcpa``), the first
    condition implies the second and "all past CPA" is the whole test.
    """
    return bool(np.all(truth_conflicts.tcpa < 0.0))
