"""Locks for the recovery family (``blueskycdarr/recovery.py``, ADR 0006).

The FTR criteria are pinned on hand-computable geometries (line-CPA distances worked out
in the docstrings); the probabilistic clearance integral is validated against Monte
Carlo — CDaRR's own Appendix-B check, re-run here on the ported code. Engine-backed
episodes assert each model runs and reproduces; behavioural comparisons live in the
correctness doc, not in assertions, because FTR's early release is *allowed* to trade
separation for mission time.
"""

from __future__ import annotations

import numpy as np
import pytest

from blueskycdarr.detection import detect
from blueskycdarr.geo import track_components
from blueskycdarr.recovery import (
    FTR,
    PastCPA,
    ProbabilisticFTR,
    _line_dcpa,
    _p_line_dcpa_exceeds,
    recovered_mask,
    recovery_from_spec,
)
from blueskycdarr.state import StateArrays

_LAT = 52.0
_M_PER_DEG_LAT = 111_320.0


def _pair(offset_east_m: float, offset_north_m: float, trk: tuple[float, float],
          gs: tuple[float, float]) -> tuple[StateArrays, StateArrays]:
    coslat = np.cos(np.radians(_LAT))
    lat = np.array([_LAT, _LAT + offset_north_m / _M_PER_DEG_LAT])
    lon = np.array([4.0, 4.0 + offset_east_m / (_M_PER_DEG_LAT * coslat)])
    own = StateArrays.from_track_speed(lat, lon, np.array(trk, float), np.array(gs, float))
    return own, own.reindexed(np.array([1, 0]))


def _mask(recovery, own, other, cmd_trk, cmd_gs, init_v=None,
          sigmas=(0.0, 0.0)) -> np.ndarray:
    """recovered_mask over a fully-resolving pair with explicit commands."""
    conflicts = detect(own, other, np.ones(2, dtype=bool), rpz=50.0, t_lookahead=120.0)
    nan = np.full(2, np.nan)
    init = init_v if init_v is not None else (nan, nan)
    return recovered_mask(
        recovery,
        resolving=np.ones(2, dtype=bool),
        conflicts=conflicts,
        own=own,
        other=other,
        commanded_v=track_components(np.array(cmd_trk, float), np.array(cmd_gs, float)),
        initial_other_v=init,
        rpz=50.0,
        margin=1.05,
        rel_pos_sigma=sigmas[0],
        rel_vel_sigma=sigmas[1],
    )


# --- deterministic FTR -----------------------------------------------------------------


def test_ftr_releases_before_cpa_when_the_commanded_course_clears() -> None:
    """Head-on pair still 1000 m out, both commanded 10 deg off: the commanded relative
    line misses by ~96 m > rpz under both hypotheses, so FTR releases while past-CPA
    (still approaching) holds — the early-release property that defines FTR."""
    own, other = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    assert _mask(FTR(), own, other, cmd_trk=(10.0, 190.0), cmd_gs=(15.0, 15.0)).all()
    assert not _mask(PastCPA(), own, other, cmd_trk=(10.0, 190.0), cmd_gs=(15.0, 15.0)).any()


def test_ftr_holds_while_the_commanded_course_still_collides() -> None:
    own, other = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    assert not _mask(FTR(), own, other, cmd_trk=(0.0, 180.0), cmd_gs=(15.0, 15.0)).any()


def test_ftr_second_criterion_blocks_release_on_the_initial_velocity_hypothesis() -> None:
    """Aircraft 0 commanded east; the intruder currently flies parallel east (criterion 1
    clears by the full 1000 m), but its *recorded* conflict-start velocity (15, -15)
    makes the commanded relative line pass through it (dcpa 0) — so FTR must hold, and
    must release once the record is empty (NaN falls back to the current velocity)."""
    own, other = _pair(0.0, 1000.0, trk=(90.0, 90.0), gs=(15.0, 15.0))
    # index i holds the recorded velocity of aircraft i's *counterpart*
    init = (np.array([15.0, np.nan]), np.array([-15.0, np.nan]))
    held = _mask(FTR(), own, other, cmd_trk=(90.0, 90.0), cmd_gs=(15.0, 15.0), init_v=init)
    assert not held[0]
    free = _mask(FTR(), own, other, cmd_trk=(90.0, 90.0), cmd_gs=(15.0, 15.0))
    assert free[0]


def test_line_dcpa_is_direction_symmetric() -> None:
    d1 = _line_dcpa(np.array([0.0]), np.array([1000.0]), np.array([2.6]), np.array([29.8]))
    d2 = _line_dcpa(np.array([0.0]), np.array([1000.0]), np.array([-2.6]), np.array([-29.8]))
    np.testing.assert_allclose(d1, d2)


# --- probabilistic FTR -----------------------------------------------------------------


def test_probabilistic_ftr_with_negligible_uncertainty_matches_ftr() -> None:
    own, other = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    for cmd in ((10.0, 190.0), (0.0, 180.0)):  # clearing and colliding commands
        ftr = _mask(FTR(), own, other, cmd_trk=cmd, cmd_gs=(15.0, 15.0))
        prob = _mask(ProbabilisticFTR(gamma=0.99), own, other, cmd_trk=cmd,
                     cmd_gs=(15.0, 15.0), sigmas=(0.0, 0.0))
        np.testing.assert_array_equal(ftr, prob)


def test_probabilistic_ftr_confidence_orders_the_release() -> None:
    """A commanded line missing by ~60 m under sigma_r 30 m clears with probability
    ~0.63: gamma 0.5 releases, gamma 0.95 holds — higher confidence, longer engagement."""
    p = _p_line_dcpa_exceeds(
        50.0, np.array([0.0]), np.array([1000.0]),
        np.array([0.9]), np.array([15.0]),  # line dcpa ~ 60 m
        sigma_r=30.0, sigma_v=0.5, k_theta=256,
    )
    assert 0.5 < p[0] < 0.75

    own, other = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    # line dcpa = 1000 sin(alpha/2) for a lone deflection alpha against the head-on
    # intruder: 6.88 deg -> ~60 m, the same clearance as the direct check above.
    kwargs = dict(cmd_trk=(6.88, 180.0), cmd_gs=(15.0, 15.0), sigmas=(30.0, 0.5))
    relaxed = _mask(ProbabilisticFTR(gamma=0.5), own, other, **kwargs)
    strict = _mask(ProbabilisticFTR(gamma=0.95), own, other, **kwargs)
    assert relaxed[0] and not strict[0]


def test_analytical_clearance_probability_matches_monte_carlo() -> None:
    """CDaRR's Appendix-B check on the ported integral: P(line DCPA > x) for Gaussian
    relative position and velocity, against 60k direct samples."""
    rng = np.random.default_rng(4)
    mu_r, sigma_r = np.array([200.0, 300.0]), 40.0
    mu_v, sigma_v = np.array([-5.0, -12.0]), 1.5
    x = 50.0

    r = mu_r + sigma_r * rng.standard_normal((60_000, 2))
    v = mu_v + sigma_v * rng.standard_normal((60_000, 2))
    empirical = float(np.mean(_line_dcpa(r[:, 0], r[:, 1], v[:, 0], v[:, 1]) > x))

    analytic = _p_line_dcpa_exceeds(
        x, mu_r[:1], mu_r[1:], mu_v[:1], mu_v[1:], sigma_r, sigma_v, k_theta=256
    )[0]
    assert abs(analytic - empirical) < 0.01


# --- the component spec ----------------------------------------------------------------


def test_recovery_spec_parses_names_and_typed_mappings() -> None:
    assert recovery_from_spec("pastcpa") == PastCPA()
    assert recovery_from_spec("ftr") == FTR()
    assert recovery_from_spec({"type": "pastcpa", "bouncing_guard": False}) == PastCPA(False)
    assert recovery_from_spec({"type": "probabilistic_ftr", "gamma": 0.9}) == (
        ProbabilisticFTR(gamma=0.9)
    )
    with pytest.raises(ValueError, match="unknown recovery"):
        recovery_from_spec("resume_maybe")
    with pytest.raises(ValueError, match="bad recovery parameters"):
        recovery_from_spec({"type": "ftr", "gamma": 0.5})
    with pytest.raises(ValueError, match="gamma"):
        ProbabilisticFTR(gamma=1.5)


# --- engine-backed ---------------------------------------------------------------------


@pytest.mark.parametrize("recovery", [PastCPA(), FTR(), ProbabilisticFTR(gamma=0.999)])
def test_each_recovery_flies_settles_and_reproduces(recovery) -> None:
    pytest.importorskip("bluesky")
    from blueskycdarr.aircraft import MULTIROTOR
    from blueskycdarr.config import Config
    from blueskycdarr.episode import run_episode
    from blueskycdarr.rng import child, root_seed_sequence
    from blueskycdarr.scenario import PairwiseEncounter

    scenario = PairwiseEncounter(pairs=(1, 2), tlos=45.0)
    seq = child(root_seed_sequence(0), 0)
    first = run_episode(scenario, MULTIROTOR, Config(), seq, recovery)
    second = run_episode(scenario, MULTIROTOR, Config(), seq, recovery)
    assert first.detected.all()
    assert first.settled
    np.testing.assert_array_equal(first.min_sep, second.min_sep)
