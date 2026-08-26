"""Locks for the MixedVarLSENew adapter (``cdarr/blackbox.py``).

The estimator math is pinned against hand-computed values (the test owns the expected
numbers); the end-to-end oracle runs one tiny design point through the engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdarr.blackbox import log10_p_los, make_blackbox


def test_log10_p_los_matches_the_hand_computed_jeffreys_estimator() -> None:
    """k=3, n=100: p = 3.5/101, se = sqrt(p(1-p)/100) / (p ln 10)."""
    y, se = log10_p_los(3, 100)
    p = 3.5 / 101.0
    assert y == pytest.approx(np.log10(p))
    assert se == pytest.approx(np.sqrt(p * (1 - p) / 100.0) / (p * np.log(10.0)))


def test_zero_losses_stay_finite() -> None:
    y, se = log10_p_los(0, 300)
    assert np.isfinite(y) and np.isfinite(se) and se > 0


def test_the_oracle_answers_a_design_point_in_order() -> None:
    pytest.importorskip("bluesky")
    from cdarr.scenario import PairwiseEncounter

    blackbox = make_blackbox(
        n_encounters=2,
        seed=0,
        scenario=PairwiseEncounter(pairs=(1, 2), tlos=45.0),
    )
    points = [
        {"p_reception": 1.0, "max_range_m": 3000.0, "kinematics": "multirotor",
         "pos_ci95_m": 3.0, "vel_ci95_ms": 1.0},
        {"p_reception": 0.0, "max_range_m": 3000.0, "kinematics": "fixedwing",
         "pos_ci95_m": 3.0, "vel_ci95_ms": 1.0},
    ]
    out = blackbox(points)
    assert len(out) == 2
    (y_seen, se_seen), (y_blind, se_blind) = out
    assert all(np.isfinite(v) for v in (y_seen, se_seen, y_blind, se_blind))
    assert y_blind > y_seen  # blind aircraft lose separation; resolved ones do not
