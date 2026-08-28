"""Locks for the pure CDR chain: detection, MVP resolution, past-CPA recovery.

Every test builds tiny hand-computable geometries as :class:`StateArrays` — no engine.
The expected numbers come from the CPA equations directly, not from the code under test.
"""

from __future__ import annotations

import numpy as np

from blueskycdarr.detection import detect, pairs_all_clear
from blueskycdarr.geo import track_components
from blueskycdarr.recovery import past_cpa_recovered
from blueskycdarr.resolution import resolve_mvp
from blueskycdarr.state import StateArrays

_LAT = 52.0
_M_PER_DEG_LAT = 111_320.0


def _pair(offset_east_m: float, offset_north_m: float, trk: tuple[float, float],
          gs: tuple[float, float]) -> tuple[StateArrays, StateArrays]:
    """Two directed views of one pair: aircraft 0 at the origin, aircraft 1 displaced."""
    coslat = np.cos(np.radians(_LAT))
    lat = np.array([_LAT, _LAT + offset_north_m / _M_PER_DEG_LAT])
    lon = np.array([4.0, 4.0 + offset_east_m / (_M_PER_DEG_LAT * coslat)])
    own = StateArrays.from_track_speed(lat, lon, np.array(trk, float), np.array(gs, float))
    other = own.reindexed(np.array([1, 0]))
    return own, other


def _seen() -> np.ndarray:
    return np.ones(2, dtype=bool)


# --- detection -------------------------------------------------------------------------


def test_head_on_pair_is_detected_with_the_analytic_tcpa() -> None:
    """1000 m apart, closing at 30 m/s head-on: tcpa 33.3 s, dcpa 0, both directions."""
    own, other = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    conflicts = detect(own, other, _seen(), rpz=50.0, t_lookahead=120.0)
    assert conflicts.in_conflict.all()
    np.testing.assert_allclose(conflicts.tcpa, 1000.0 / 30.0, rtol=1e-6)
    # dcpa comes from |dist^2 - tcpa^2 v^2|: catastrophic cancellation leaves ~1e-5 m of
    # float noise on an exact-zero miss, far below anything physical. Millimetre is exact.
    np.testing.assert_allclose(conflicts.dcpa, 0.0, atol=1e-3)


def test_a_wide_miss_is_no_conflict_and_a_diverging_pair_is_past_cpa() -> None:
    own, other = _pair(200.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    assert not detect(own, other, _seen(), rpz=50.0, t_lookahead=120.0).in_conflict.any()

    own_d, other_d = _pair(0.0, 1000.0, trk=(180.0, 0.0), gs=(15.0, 15.0))  # flying apart
    conflicts = detect(own_d, other_d, _seen(), rpz=50.0, t_lookahead=120.0)
    assert not conflicts.in_conflict.any()
    assert pairs_all_clear(conflicts)


def test_beyond_the_lookahead_is_not_yet_a_conflict() -> None:
    own, other = _pair(0.0, 6000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))  # tcpa 200 s
    assert not detect(own, other, _seen(), rpz=50.0, t_lookahead=120.0).in_conflict.any()


def test_an_unseen_counterpart_is_never_a_conflict() -> None:
    """No contact, no conflict — the surveillance-range effect must be representable."""
    own, other = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    conflicts = detect(own, other, np.zeros(2, dtype=bool), rpz=50.0, t_lookahead=120.0)
    assert not conflicts.in_conflict.any()


# --- resolution ------------------------------------------------------------------------


def _cpa_after(own: StateArrays, other: StateArrays, idx: int, trk_cmd: float,
               gs_cmd: float) -> float:
    """Predicted miss distance if aircraft ``idx`` flies the command and the other holds."""
    ve, vn = track_components(np.array([trk_cmd]), np.array([gs_cmd]))
    due = other.gs_east[idx] - ve[0]
    dvn = other.gs_north[idx] - vn[0]
    coslat = np.cos(np.radians(_LAT))
    dx = (other.lon[idx] - own.lon[idx]) * _M_PER_DEG_LAT * coslat
    dy = (other.lat[idx] - own.lat[idx]) * _M_PER_DEG_LAT
    rel_sq = max(due * due + dvn * dvn, 1e-6)
    tcpa = -(due * dx + dvn * dy) / rel_sq
    return float(np.hypot(dx + due * tcpa, dy + dvn * tcpa))


def test_mvp_pushes_the_predicted_miss_to_the_resolution_zone() -> None:
    """Flying the command against an unchanged intruder must lift dcpa to ~rpz*margin."""
    own, other = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    conflicts = detect(own, other, _seen(), rpz=50.0, t_lookahead=120.0)
    command = resolve_mvp(conflicts, own, rpz=50.0, margin=1.05,
                          v_min=np.zeros(2), v_max=np.full(2, 18.0))
    assert set(command.idx) == {0, 1}
    for slot, i in enumerate(command.idx):
        after = _cpa_after(own, other, int(i), float(command.trk[slot]), float(command.gs[slot]))
        assert after > 50.0 * 1.05 * 0.95  # one-sided manoeuvre already clears most of it


def test_mvp_caps_commanded_speed_into_the_envelope() -> None:
    own, other = _pair(0.0, 400.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    conflicts = detect(own, other, _seen(), rpz=50.0, t_lookahead=120.0)
    command = resolve_mvp(conflicts, own, rpz=50.0, margin=1.05,
                          v_min=np.full(2, 12.0), v_max=np.full(2, 25.0))
    assert (command.gs >= 12.0 - 1e-9).all() and (command.gs <= 25.0 + 1e-9).all()


# --- recovery --------------------------------------------------------------------------


def test_recovery_waits_for_past_cpa_and_separation() -> None:
    resolving = np.ones(2, dtype=bool)
    approaching = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    conflicts = detect(*approaching, _seen(), rpz=50.0, t_lookahead=120.0)
    assert not past_cpa_recovered(resolving, conflicts, *approaching, rpz=50.0,
                                  margin=1.05).any()

    receding = _pair(0.0, 1000.0, trk=(180.0, 0.0), gs=(15.0, 15.0))
    conflicts_r = detect(*receding, _seen(), rpz=50.0, t_lookahead=120.0)
    assert past_cpa_recovered(resolving, conflicts_r, *receding, rpz=50.0, margin=1.05).all()


def test_recovery_refuses_inside_los_and_bouncing_geometry() -> None:
    resolving = np.ones(2, dtype=bool)
    in_los = _pair(0.0, 30.0, trk=(180.0, 0.0), gs=(15.0, 15.0))  # past CPA but too close
    conflicts = detect(*in_los, _seen(), rpz=50.0, t_lookahead=120.0)
    assert not past_cpa_recovered(resolving, conflicts, *in_los, rpz=50.0, margin=1.05).any()

    # Near-parallel, just outside the zone, inside the margin: the bouncing guard holds.
    bouncing = _pair(0.0, 51.0, trk=(0.0, 10.0), gs=(15.0, 14.0))
    conflicts_b = detect(*bouncing, _seen(), rpz=50.0, t_lookahead=120.0)
    receding_mask = conflicts_b.dx * conflicts_b.du + conflicts_b.dy * conflicts_b.dv > 0
    guard = past_cpa_recovered(resolving, conflicts_b, *bouncing, rpz=50.0, margin=1.05)
    no_guard = past_cpa_recovered(resolving, conflicts_b, *bouncing, rpz=50.0, margin=1.05,
                                  bouncing_guard=False)
    assert not guard.any()
    assert (no_guard == (receding_mask & (conflicts_b.dist >= 50.0))).all()
