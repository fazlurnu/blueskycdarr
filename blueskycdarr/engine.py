"""The BlueSky boundary — the only module that imports ``bluesky`` (ADR 0001).

Everything third-party lives behind this seam: process-wide initialisation, aircraft
creation (``cre`` / ``creconfs``), the fork's turn-rate limiter arrays, unit conversions
(BlueSky stacks speeds in knots, spawns miss distances in NM), and stepping. The CDR
chain and the channel never touch ``bs.*``, so they stay pure and testable.

One conversion here is a trap this module exists to contain: **every speed BlueSky
accepts** — ``cre``'s ``acspd``, ``creconfs``'s ``spd``, the ``SPD`` stack command — **is
calibrated airspeed**, while everything this package computes and commands is a ground
speed. At 100 m the two differ by the air-density factor 1.0048; harmless once, but a
loop that reads ground speed back and re-commands it as CAS *compounds* the factor per
command — CDaRR's resolution path did exactly that, and its drones crept ~0.5% faster
per re-command (``notebooks/bluesky_speed_command.ipynb`` demonstrates it). This module
therefore converts ground -> CAS at the boundary, so a commanded ground speed is the
ground speed flown, exactly.

The engine must be the CDaRR fork (branch ``CDaRR``): its per-aircraft ``max_tr`` /
``max_dtr2`` limiter arrays and ``creconfs`` are load-bearing. :func:`ensure_engine`
fails fast with the install line when they are missing, rather than running stock
dynamics silently.

BlueSky is a process-global singleton, so one process hosts one world at a time; the
joblib episode fan-out gives each worker process its own engine (CDaRR's
``_joblib_inited`` pattern, kept here as a module flag).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from blueskycdarr.aircraft import AircraftModel, AircraftSpec, as_pair
from blueskycdarr.config import ConflictConfig, SimulationConfig
from blueskycdarr.geo import distance_m
from blueskycdarr.scenario import PairGeometry, PairwiseEncounter
from blueskycdarr.state import StateArrays

M_TO_NM = 1.0 / 1852.0
MPS_TO_KTS = 1.0 / 0.514444

# CDaRR's spawn grid (envs/pairwise_params.json): pairs far enough apart that encounters
# never interact.
_GRID_LAT0 = 52.3
_GRID_LON0 = 4.7
_GRID_DELTA_DEG = 0.3
_ALT_M = 100.0

_initialised = False


@contextmanager
def _quiet() -> Iterator[None]:
    """BlueSky prints its banner on init; keep library imports silent (CDaRR's helper)."""
    with open(os.devnull, "w") as devnull:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def ensure_engine() -> None:
    """Initialise BlueSky once per process and verify it is the CDaRR fork."""
    global _initialised
    if _initialised:
        return
    import bluesky as bs

    with _quiet():
        bs.init(mode="sim", detached=True)
    if not hasattr(bs.traf, "max_tr") or not hasattr(bs.traf, "creconfs"):
        raise RuntimeError(
            "this BlueSky lacks the CDaRR fork's turn-rate limiter; install the engine "
            "with: pip install 'bluesky-simulator @ "
            "git+https://github.com/fazlurnu/bluesky.git@CDaRR'"
        )
    _initialised = True


@dataclass
class PairwiseWorld:
    """One spawned batch of conflict pairs inside the process-global BlueSky.

    Aircraft ``2k`` is pair ``k``'s ownship (track 000), ``2k + 1`` its intruder, spawned
    in conflict by the fork's ``creconfs`` with the pair's crossing angle, miss distance
    and time to loss of separation. Use as a context manager so the global traffic state
    is reset even when an episode raises.
    """

    scenario: PairwiseEncounter
    geometry: PairGeometry
    aircraft: AircraftSpec
    conflict: ConflictConfig
    simulation: SimulationConfig

    def __post_init__(self) -> None:
        ensure_engine()
        import bluesky as bs
        from bluesky.tools.aero import vtas2cas

        self._bs = bs
        self._to_cas = lambda gs_ms: float(vtas2cas(gs_ms, _ALT_M))  # ground -> CAS at 100 m
        own_model, intr_model = as_pair(self.aircraft)
        bs.traf.reset()
        bs.settings.asas_pzr = self.conflict.rpz * M_TO_NM  # creconfs reads the zone radius
        bs.stack.stack(f"DT {self.simulation.dt}")

        n = self.scenario.n_pairs
        rows, cols = self.scenario.pairs
        self.nominal_trk = np.empty(2 * n)
        self.nominal_gs = np.empty(2 * n)

        for k in range(n):
            lat = _GRID_LAT0 + (k // cols) * _GRID_DELTA_DEG
            lon = _GRID_LON0 + (k % cols) * _GRID_DELTA_DEG
            own_id, intr_id = f"OWN{k:03d}", f"INT{k:03d}"
            bs.traf.cre(
                acid=own_id,
                actype=own_model.bs_actype,
                aclat=lat,
                aclon=lon,
                achdg=0.0,
                acalt=_ALT_M,
                acspd=self._to_cas(float(self.geometry.gs_own[k])),
            )
            bs.traf.creconfs(
                acid=intr_id,
                actype=intr_model.bs_actype,
                targetidx=bs.traf.id2idx(own_id),
                dpsi=float(self.geometry.dpsi[k]),
                dcpa=float(self.geometry.dcpa[k]) * M_TO_NM,
                tlosh=float(self.scenario.tlos),
                spd=self._to_cas(float(self.geometry.gs_intr[k])),
            )
            self.nominal_trk[2 * k] = 0.0
            self.nominal_trk[2 * k + 1] = self.geometry.dpsi[k] % 360.0
            self.nominal_gs[2 * k] = self.geometry.gs_own[k]
            self.nominal_gs[2 * k + 1] = self.geometry.gs_intr[k]

        self._apply_turn_policy(own_model, slice(0, None, 2))
        self._apply_turn_policy(intr_model, slice(1, None, 2))
        self._last_trk = np.full(2 * n, np.nan)
        self._last_gs = np.full(2 * n, np.nan)
        self.command(self.nominal_trk, self.nominal_gs)

    def _apply_turn_policy(self, model: AircraftModel, role: slice) -> None:
        """Write one role's turn authority into the fork's limiter arrays (ADR 0005)."""
        traf = self._bs.traf
        traf.max_tr[role] = model.max_turn_rate if model.max_turn_rate is not None else np.inf
        traf.max_dtr2[role] = (
            model.max_turn_accel if model.max_turn_accel is not None else np.inf
        )
        if model.bank_deg is not None:
            traf.ap.bankdef[role] = np.radians(model.bank_deg)

    @property
    def n_aircraft(self) -> int:
        return int(self._bs.traf.ntraf)

    def truth(self) -> StateArrays:
        """The ground-truth state table, copied out of the engine."""
        traf = self._bs.traf
        return StateArrays(
            lat=np.array(traf.lat, dtype=float),
            lon=np.array(traf.lon, dtype=float),
            trk=np.array(traf.trk, dtype=float),
            gs=np.array(traf.gs, dtype=float),
            gs_east=np.array(traf.gseast, dtype=float),
            gs_north=np.array(traf.gsnorth, dtype=float),
        )

    def command(self, trk: np.ndarray, gs: np.ndarray) -> None:
        """Stack HDG/SPD for every aircraft whose command changed (BlueSky holds them).

        ``gs`` is a ground speed; the SPD stack value is CAS (see the module docstring),
        so it is converted here — never by callers, and never fed back from a reading.
        """
        traf = self._bs.traf
        stack = self._bs.stack.stack
        changed = np.flatnonzero(
            ~(np.isclose(trk, self._last_trk) & np.isclose(gs, self._last_gs))
        )
        for i in changed:
            stack(f"HDG {traf.id[i]} {trk[i]:.4f}")
            stack(f"SPD {traf.id[i]} {self._to_cas(gs[i]) * MPS_TO_KTS:.4f}")
        self._last_trk[changed] = trk[changed]
        self._last_gs[changed] = gs[changed]

    def step(self) -> None:
        self._bs.sim.step()

    def pair_distances(self) -> np.ndarray:
        """True ownship-intruder distance per pair, metres (length ``n_pairs``)."""
        traf = self._bs.traf
        own = slice(0, None, 2)
        intr = slice(1, None, 2)
        return distance_m(
            np.array(traf.lat[own]), np.array(traf.lon[own]),
            np.array(traf.lat[intr]), np.array(traf.lon[intr]),
        )

    def __enter__(self) -> PairwiseWorld:
        return self

    def __exit__(self, *exc: object) -> None:
        self._bs.traf.reset()
