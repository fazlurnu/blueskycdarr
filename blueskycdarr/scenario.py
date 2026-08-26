"""The pairwise encounter — CDaRR's scenario, with OpenCDaRR's geometry-slot semantics.

A batch of independent ownship/intruder pairs is spawned on a lat/lon grid wide enough
that pairs never interact (CDaRR's ``envs/pairwise_conflict.py``). Each pair's intruder is
created *in conflict* via the fork's ``creconfs``: crossing angle ``dpsi``, miss distance
``dcpa``, and ``tlos`` seconds to loss of separation at spawn.

Geometry slots follow OpenCDaRR's rule: a number pins the slot for every pair, ``null``
draws it per pair (``dpsi`` from U(0, 360), ``dcpa`` from U(0, dcpa_max)) from the
episode's geometry stream — the same draws in every condition (ADR 0004).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairGeometry:
    """The per-pair spawn values one episode runs with (all arrays of length n_pairs)."""

    dpsi: np.ndarray  # deg, intruder track relative to ownship's
    dcpa: np.ndarray  # m, miss distance at closest approach
    gs_own: np.ndarray  # m/s
    gs_intr: np.ndarray  # m/s


@dataclass(frozen=True)
class PairwiseEncounter:
    """The encounter model and its grid.

    Attributes
    ----------
    speed:
        Ownship cruise ground speed, m/s — a number pins it, a ``(min, max)`` pair draws
        it per pair from U(min, max) (CDaRR's exp3/exp4 heterogeneous speeds, ADR 0007).
    gs_intr:
        Intruder speed, m/s; same number-or-range semantics as ``speed``; ``None``
        matches the ownship's per-pair value.
    dpsi:
        Crossing angle, deg; ``None`` draws U(0, 360) per pair.
    dcpa:
        Miss distance, m; ``None`` draws U(0, ``dcpa_max``) per pair.
    dcpa_max:
        Upper bound of the ``dcpa`` draw; must stay within the protected zone for the
        spawn to be a genuine conflict.
    tlos:
        Time to loss of separation at spawn, s. Below the detection horizon the pair is
        spawned straight into a detectable conflict (the MixedVarLSENew setup).
    pairs:
        The spawn grid as (rows, cols); rows x cols pairs per episode.
    """

    speed: float | tuple[float, float] = 15.0
    gs_intr: float | tuple[float, float] | None = None
    dpsi: float | None = 90.0
    dcpa: float | None = 0.0
    dcpa_max: float = 50.0
    tlos: float = 90.0
    pairs: tuple[int, int] = (10, 10)

    def __post_init__(self) -> None:
        problems = []
        problems += _speed_problems("speed", self.speed)
        if self.gs_intr is not None:
            problems += _speed_problems("gs_intr", self.gs_intr)
        if self.dcpa_max < 0:
            problems.append("dcpa_max >= 0")
        if self.dcpa is not None and not 0 <= self.dcpa <= max(self.dcpa_max, 0):
            problems.append("0 <= dcpa <= dcpa_max")
        if self.tlos <= 0:
            problems.append("tlos > 0")
        if len(self.pairs) != 2 or min(self.pairs) < 1:
            problems.append("pairs = (rows >= 1, cols >= 1)")
        if problems:
            raise ValueError(f"scenario constraints violated: {'; '.join(problems)}")

    @property
    def n_pairs(self) -> int:
        return self.pairs[0] * self.pairs[1]

    def draw_geometry(self, rng: np.random.Generator) -> PairGeometry:
        """Resolve every slot to per-pair arrays; pinned slots consume no randomness.

        Draw order (dpsi, dcpa, speed, gs_intr) is fixed: a range added to one slot must
        not shift another slot's draws (the common-random-numbers layout, ADR 0004).
        """
        n = self.n_pairs
        dpsi = (
            np.full(n, float(self.dpsi)) if self.dpsi is not None else rng.uniform(0.0, 360.0, n)
        )
        dcpa = (
            np.full(n, float(self.dcpa))
            if self.dcpa is not None
            else rng.uniform(0.0, self.dcpa_max, n)
        )
        gs_own = _resolve_speed(self.speed, n, rng)
        gs_intr = (
            gs_own.copy() if self.gs_intr is None else _resolve_speed(self.gs_intr, n, rng)
        )
        return PairGeometry(dpsi=dpsi, dcpa=dcpa, gs_own=gs_own, gs_intr=gs_intr)


def _speed_problems(name: str, value: float | tuple[float, float]) -> list[str]:
    if isinstance(value, tuple):
        if len(value) != 2 or not 0 < value[0] < value[1]:
            return [f"{name} range must be (min, max) with 0 < min < max"]
        return []
    return [] if value > 0 else [f"{name} > 0"]


def _resolve_speed(
    value: float | tuple[float, float], n: int, rng: np.random.Generator
) -> np.ndarray:
    if isinstance(value, tuple):
        return rng.uniform(value[0], value[1], n)
    return np.full(n, float(value))
