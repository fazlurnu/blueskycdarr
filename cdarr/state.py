"""State tables — the horizontal state of all aircraft in one episode, index-aligned.

One :class:`StateArrays` holds lat/lon/track/speed for the ``2 * n_pairs`` aircraft of an
episode. Aircraft ``2k`` is pair ``k``'s ownship, ``2k + 1`` its intruder, so an
aircraft's encounter counterpart is ``i ^ 1`` (:func:`counterpart`).

The same container carries three different views of the same fleet (ADR 0002):

- the **truth** read from the engine,
- an aircraft's **own navigation** view (truth + fresh GNSS noise), and
- the **contact** view (the counterpart's last *delivered* broadcast — stale by design).

The CDR chain takes these views and never touches the engine, which is what CDaRR's
"algorithms act on perceived state, never ground truth" invariant looks like as code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cdarr.geo import track_components


@dataclass
class StateArrays:
    """Index-aligned horizontal state (deg, deg, deg, m/s and derived m/s components)."""

    lat: np.ndarray
    lon: np.ndarray
    trk: np.ndarray
    gs: np.ndarray
    gs_east: np.ndarray
    gs_north: np.ndarray

    @classmethod
    def from_track_speed(
        cls, lat: np.ndarray, lon: np.ndarray, trk: np.ndarray, gs: np.ndarray
    ) -> StateArrays:
        east, north = track_components(trk, gs)
        return cls(
            lat=np.asarray(lat, dtype=float).copy(),
            lon=np.asarray(lon, dtype=float).copy(),
            trk=np.asarray(trk, dtype=float).copy(),
            gs=np.asarray(gs, dtype=float).copy(),
            gs_east=east,
            gs_north=north,
        )

    @property
    def n(self) -> int:
        return int(self.lat.size)

    def copy(self) -> StateArrays:
        return StateArrays(
            lat=self.lat.copy(),
            lon=self.lon.copy(),
            trk=self.trk.copy(),
            gs=self.gs.copy(),
            gs_east=self.gs_east.copy(),
            gs_north=self.gs_north.copy(),
        )

    def overwrite_from(self, other: StateArrays, idx: np.ndarray) -> None:
        """Patch rows ``idx`` from ``other`` (a delivered broadcast updating a contact)."""
        for name in ("lat", "lon", "trk", "gs", "gs_east", "gs_north"):
            getattr(self, name)[idx] = getattr(other, name)[idx]

    def reindexed(self, perm: np.ndarray) -> StateArrays:
        """A copy with rows reordered by ``perm`` (row ``i`` becomes old row ``perm[i]``)."""
        return StateArrays(
            lat=self.lat[perm],
            lon=self.lon[perm],
            trk=self.trk[perm],
            gs=self.gs[perm],
            gs_east=self.gs_east[perm],
            gs_north=self.gs_north[perm],
        )


def counterpart(idx: np.ndarray | int) -> np.ndarray | int:
    """The encounter counterpart of aircraft ``idx``: ownship <-> intruder of the pair."""
    return idx ^ 1
