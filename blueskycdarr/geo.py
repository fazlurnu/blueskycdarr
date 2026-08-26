"""Flat-earth geometry between perceived positions — pure numpy, no BlueSky import.

The CDR chain (detection, resolution, recovery) works on *perceived* lat/lon and must be
testable without the engine, so the small-angle geometry it needs lives here rather than in
``bluesky.tools.geo`` (wrap third parties at a boundary; the boundary is ``engine.py``).

The approximation is the same one CDaRR's recovery used (``crr_resumenav_cpa.py``): a local
tangent plane with one metre-per-degree latitude constant and a ``cos(lat)`` longitude
correction. At the encounter scale this package simulates (a few km around 52 N) the error
against BlueSky's ``kwikqdrdist`` is far below the smallest position noise swept (3 m CI95).
"""

from __future__ import annotations

import numpy as np

M_PER_DEG_LAT = 111_320.0  # same constant as CDaRR's noise model and BlueSky's kwik functions


def enu_offset(
    lat_from: np.ndarray, lon_from: np.ndarray, lat_to: np.ndarray, lon_to: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(east, north) metres from one position to another on the local tangent plane."""
    mean_lat = np.radians(0.5 * (lat_from + lat_to))
    east = (lon_to - lon_from) * M_PER_DEG_LAT * np.cos(mean_lat)
    north = (lat_to - lat_from) * M_PER_DEG_LAT
    return east, north


def distance_m(
    lat_from: np.ndarray, lon_from: np.ndarray, lat_to: np.ndarray, lon_to: np.ndarray
) -> np.ndarray:
    """Horizontal distance in metres."""
    east, north = enu_offset(lat_from, lon_from, lat_to, lon_to)
    return np.hypot(east, north)


def displace(
    lat: np.ndarray, lon: np.ndarray, east_m: np.ndarray, north_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Position displaced by (east, north) metres — the inverse of :func:`enu_offset`."""
    coslat = np.maximum(np.cos(np.radians(lat)), 1e-6)  # avoid blow-up near the poles
    return lat + north_m / M_PER_DEG_LAT, lon + east_m / (M_PER_DEG_LAT * coslat)


def track_components(trk_deg: np.ndarray, gs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(east, north) velocity components from track (deg, 0 = north, clockwise) and speed."""
    trk = np.radians(trk_deg)
    return gs * np.sin(trk), gs * np.cos(trk)


def track_from_components(v_east: np.ndarray, v_north: np.ndarray) -> np.ndarray:
    """Track angle in degrees, aviation convention, from velocity components."""
    return np.degrees(np.arctan2(v_east, v_north)) % 360.0
