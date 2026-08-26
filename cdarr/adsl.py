"""The ADS-L layer: measurement noise and the broadcast channel (ADR 0002).

CDaRR modelled communication as a per-CDR-tick Bernoulli reception plus an along-track
position bias for latency. This module keeps the same measurement model but makes the
channel *event-based*, which is what lets broadcast jitter, latency, and a surveillance
range coexist without approximation:

- each aircraft transmits on its own schedule, ``interval + U(-jitter, +jitter)`` apart;
- a transmission is a **noisy snapshot of the transmit-time state**;
- it is received iff a Bernoulli(``reception_prob``) succeeds **and** the true distance to
  the receiver at transmit time is within ``max_range_m``;
- a received message becomes usable ``latency_s`` after transmission, and the receiver
  holds the last usable message (a stale contact, never extrapolated — CDaRR's holdover).

With jitter 0, latency 0, no range gate and the interval equal to the CDR cadence, the
channel reduces to CDaRR's per-tick reception (locked by a test).

Noise: position error is an isotropic 2D Gaussian with per-axis sigma ``ci95 / 2.448``
(2.448 = sqrt of the chi-square 95% quantile with 2 dof — CDaRR's ``CI95_TO_STD_2D``);
velocity error likewise on the east/north components. Unlike CDaRR — which noised only
the component fields, so its detector saw noise-free velocity while its resolver saw
noise — a snapshot here is self-consistent: ``trk``/``gs`` are recomputed from the noised
components, so every consumer sees the same measured velocity (ADR 0002, deviation D1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cdarr.config import CommConfig, UncertaintyConfig
from cdarr.geo import displace, distance_m, track_from_components
from cdarr.noise import CI95_TO_STD_2D, DEFAULT_NOISE, NoiseShape
from cdarr.state import StateArrays, counterpart

_EPS = 1e-9  # time comparison tolerance


def noisy_snapshot(
    truth: StateArrays,
    idx: np.ndarray,
    uncertainty: UncertaintyConfig,
    rng: np.random.Generator,
    shape: NoiseShape = DEFAULT_NOISE,
) -> StateArrays:
    """A measured copy of ``truth`` rows ``idx`` (all rows keep truth except ``idx``).

    Returned as a full-size table so callers patch by index; only rows ``idx`` carry
    noise. ``shape`` selects the position-error distribution (ADR 0007); every shape
    delivers the configured radial CI95, and velocity noise stays Gaussian regardless.
    """
    out = truth.copy()
    k = int(idx.size)
    if k == 0:
        return out

    vel_std = uncertainty.vel_ci95 / CI95_TO_STD_2D

    if uncertainty.pos_ci95 > 0:
        xy = shape.draw(k, uncertainty.pos_ci95, rng, np.radians(truth.trk[idx]), truth.gs[idx])
        east_m, north_m = xy[:, 0], xy[:, 1]
    else:
        east_m, north_m = 0.0, 0.0
    out.lat[idx], out.lon[idx] = displace(truth.lat[idx], truth.lon[idx], east_m, north_m)

    v_east = truth.gs_east[idx]
    v_north = truth.gs_north[idx]
    if vel_std > 0:
        v_east = v_east + rng.normal(0.0, vel_std, k)
        v_north = v_north + rng.normal(0.0, vel_std, k)
    out.gs_east[idx] = v_east
    out.gs_north[idx] = v_north
    out.gs[idx] = np.hypot(v_east, v_north)
    out.trk[idx] = track_from_components(v_east, v_north)
    return out


@dataclass
class ContactTable:
    """Row ``i`` = the last delivered broadcast *from* aircraft ``i`` (held by ``i ^ 1``).

    A contact starts invalid — an aircraft that has never been heard does not exist to its
    counterpart, which is exactly what a surveillance-range gate must be able to express
    (ADR 0002, deviation D2 from CDaRR's guaranteed first copy).
    """

    states: StateArrays
    valid: np.ndarray  # bool, per subject aircraft
    t_tx: np.ndarray  # s, transmit time of the held state (staleness = now - t_tx)

    @classmethod
    def empty(cls, truth: StateArrays) -> ContactTable:
        return cls(
            states=truth.copy(),  # placeholder values; masked by valid=False
            valid=np.zeros(truth.n, dtype=bool),
            t_tx=np.full(truth.n, np.nan),
        )

    def view_of_counterparts(self) -> tuple[StateArrays, np.ndarray]:
        """(states, valid) reindexed so row ``i`` is what aircraft ``i`` knows of ``i ^ 1``."""
        perm = np.asarray(counterpart(np.arange(self.states.n)))
        return self.states.reindexed(perm), self.valid[perm]


@dataclass
class _Delivery:
    t_due: float
    idx: np.ndarray
    snapshot: StateArrays
    t_tx: float


@dataclass
class BroadcastChannel:
    """The per-episode broadcast machinery. Advance with :meth:`transmit_due`, then
    :meth:`deliver_due`; both are idempotent within a time step."""

    comm: CommConfig
    uncertainty: UncertaintyConfig
    rng_measurement: np.random.Generator
    rng_reception: np.random.Generator
    rng_schedule: np.random.Generator
    shape: NoiseShape = DEFAULT_NOISE
    next_tx: np.ndarray = field(init=False)
    _pending: deque[_Delivery] = field(init=False, default_factory=deque)

    def initialise(self, n: int) -> None:
        if self.comm.broadcast_random_phase:
            self.next_tx = self.rng_schedule.uniform(0.0, self.comm.broadcast_interval_s, n)
        else:
            self.next_tx = np.zeros(n)

    def transmit_due(self, t: float, truth: StateArrays) -> None:
        """Fire every broadcast scheduled at or before ``t``.

        The snapshot content is the current truth plus measurement noise; reception and
        the range gate are decided now (they are transmit-path physics), delivery lands
        ``latency_s`` later.
        """
        due = np.flatnonzero(self.next_tx <= t + _EPS)
        if due.size:
            snapshot = noisy_snapshot(
                truth, due, self.uncertainty, self.rng_measurement, self.shape
            )
            heard = self.rng_reception.random(due.size) <= self.comm.reception_prob
            rx = np.asarray(counterpart(due))
            in_range = (
                distance_m(truth.lat[due], truth.lon[due], truth.lat[rx], truth.lon[rx])
                <= self.comm.range_gate_m
            )
            delivered = due[heard & in_range]
            if delivered.size:
                self._pending.append(
                    _Delivery(t_due=t + self.comm.latency_s, idx=delivered, snapshot=snapshot,
                              t_tx=t)
                )
            jitter = self.comm.broadcast_jitter_s
            gaps = self.comm.broadcast_interval_s + (
                self.rng_schedule.uniform(-jitter, jitter, due.size) if jitter > 0 else 0.0
            )
            self.next_tx[due] += gaps

    def deliver_due(self, t: float, contacts: ContactTable) -> None:
        """Land every in-flight message whose latency has elapsed, oldest first."""
        while self._pending and self._pending[0].t_due <= t + _EPS:
            d = self._pending.popleft()
            contacts.states.overwrite_from(d.snapshot, d.idx)
            contacts.valid[d.idx] = True
            contacts.t_tx[d.idx] = d.t_tx
