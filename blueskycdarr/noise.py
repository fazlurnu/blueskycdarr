"""Position-noise shapes — CDaRR's pluggable distributions as components (ADR 0007).

Ported from CDaRR's ``sim_models/noise_distributions.py`` (itself the exp3/exp4
noise-model sweep): every shape returns ``(n, 2)`` east/north measurement errors in
metres and **preserves the same containment guarantee** — the 95th percentile of the 2D
radial error equals the configured ``ci95`` exactly, solved by the same bisections as
the source. Only the *shape* of the error changes:

- :class:`Gaussian` — the isotropic default (draw-stream-identical to the pre-shape
  code, so existing seeded results are unchanged).
- :class:`MixtureGaussian` — a heavy tail: with probability ``tail_weight`` the draw
  comes from a component ``tail_ratio`` times wider.
- :class:`AnisotropicGaussian` — along-track variance ``var_ratio`` times cross-track,
  oriented per aircraft by its track (Schaefer & Jonas 2025 measured ADS-B along-track
  errors ~3x the cross-track stdev, i.e. ``var_ratio`` 9 — CDaRR's exp value; the
  module default 3 matches the source's).
- :class:`AnisotropicMixtureGaussian` — both effects combined.

Velocity noise stays isotropic Gaussian regardless (as in CDaRR). The latency member of
CDaRR's sweep is not a shape here — it is the channel's ``latency_s`` (ADR 0002).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

try:  # scipy is present transitively (BlueSky depends on it); fall back to math.erf
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover - exercised only without scipy
    _erf = np.vectorize(math.erf)

CI95_TO_STD_2D = 2.448  # sqrt(-2 ln 0.05): 95% radial CI -> per-axis sigma, 2D isotropic

_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass(frozen=True)
class Gaussian:
    """Zero-mean isotropic 2D normal error."""

    def draw(
        self, n: int, ci95: float, rng: np.random.Generator, trk_rad: np.ndarray
    ) -> np.ndarray:
        # (2, n) then transpose: byte-identical to the pre-shape draw order, so a
        # seeded run with the default shape reproduces exactly (ADR 0004 invariant).
        std = float(ci95) / CI95_TO_STD_2D
        return rng.normal(0.0, std, (2, n)).T


@dataclass(frozen=True)
class MixtureGaussian:
    """Two-component isotropic mixture: a core and a ``tail_ratio``-wider tail."""

    tail_ratio: float = 3.0
    tail_weight: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.tail_weight < 1.0:
            raise ValueError(f"tail_weight must be in (0, 1), got {self.tail_weight}")
        if self.tail_ratio <= 1.0:
            raise ValueError(f"tail_ratio must be > 1, got {self.tail_ratio}")

    def draw(
        self, n: int, ci95: float, rng: np.random.Generator, trk_rad: np.ndarray
    ) -> np.ndarray:
        s1 = _mixture_sigma1(round(float(ci95), 8), self.tail_ratio, self.tail_weight)
        use_tail = rng.random(n) < self.tail_weight
        sigmas = np.where(use_tail, self.tail_ratio * s1, s1).reshape(n, 1)
        return rng.standard_normal((n, 2)) * sigmas


@dataclass(frozen=True)
class AnisotropicGaussian:
    """Along-track variance ``var_ratio`` times cross-track, oriented by the track."""

    var_ratio: float = 3.0

    def __post_init__(self) -> None:
        if self.var_ratio <= 1.0:
            raise ValueError(f"var_ratio must be > 1, got {self.var_ratio}")

    def draw(
        self, n: int, ci95: float, rng: np.random.Generator, trk_rad: np.ndarray
    ) -> np.ndarray:
        std_ratio = math.sqrt(self.var_ratio)
        sigma_cross = _aniso_sigma_cross(round(float(ci95), 8), std_ratio)
        along = rng.standard_normal(n) * (std_ratio * sigma_cross)
        cross = rng.standard_normal(n) * sigma_cross
        return _rotate_to_enu(along, cross, trk_rad)


@dataclass(frozen=True)
class AnisotropicMixtureGaussian:
    """Anisotropic core plus a ``tail_ratio``-wider tail of the same shape."""

    var_ratio: float = 3.0
    tail_ratio: float = 3.0
    tail_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.var_ratio <= 1.0:
            raise ValueError(f"var_ratio must be > 1, got {self.var_ratio}")
        if not 0.0 < self.tail_weight < 1.0:
            raise ValueError(f"tail_weight must be in (0, 1), got {self.tail_weight}")
        if self.tail_ratio <= 1.0:
            raise ValueError(f"tail_ratio must be > 1, got {self.tail_ratio}")

    def draw(
        self, n: int, ci95: float, rng: np.random.Generator, trk_rad: np.ndarray
    ) -> np.ndarray:
        std_ratio = math.sqrt(self.var_ratio)
        s1 = _aniso_mixture_sigma_cross(
            round(float(ci95), 8), std_ratio, self.tail_ratio, self.tail_weight
        )
        use_tail = rng.random(n) < self.tail_weight
        sigma_cross = np.where(use_tail, self.tail_ratio * s1, s1)
        along = rng.standard_normal(n) * (std_ratio * sigma_cross)
        cross = rng.standard_normal(n) * sigma_cross
        return _rotate_to_enu(along, cross, trk_rad)


NoiseShape = Gaussian | MixtureGaussian | AnisotropicGaussian | AnisotropicMixtureGaussian

DEFAULT_NOISE: NoiseShape = Gaussian()

_LABELS: dict[str, type] = {
    "gaussian": Gaussian,
    "mixture_gaussian": MixtureGaussian,
    "anisotropic_gaussian": AnisotropicGaussian,
    "anisotropic_mixture_gaussian": AnisotropicMixtureGaussian,
}


def noise_from_spec(spec: object, *, source: object = "<spec>") -> NoiseShape:
    """A noise shape from its file spelling: a bare name or ``{type: ..., params}``."""
    if isinstance(spec, NoiseShape):
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("type", None)
    else:
        raise ValueError(
            f"{source}: noise must be a name or a {{type: ...}} mapping, got {spec!r}"
        )
    if name not in _LABELS:
        raise ValueError(f"{source}: unknown noise shape {name!r}. Known: {sorted(_LABELS)}")
    try:
        return _LABELS[name](**params)
    except TypeError as err:
        raise ValueError(f"{source}: bad noise parameters for {name!r}: {err}") from None


# --- the containment bisections (CDaRR's, cached per parameter set) --------------------


def _rotate_to_enu(
    along: np.ndarray, cross: np.ndarray, trk_rad: np.ndarray
) -> np.ndarray:
    trk = np.asarray(trk_rad, dtype=float)
    east = along * np.sin(trk) + cross * np.cos(trk)
    north = along * np.cos(trk) - cross * np.sin(trk)
    return np.stack([east, north], axis=1)


@lru_cache(maxsize=256)
def _mixture_sigma1(ci95: float, tail_ratio: float, tail_weight: float) -> float:
    """Core sigma of the isotropic mixture such that P(radial <= ci95) = 0.95.

    Solves ``p exp(-u) + (1-p) exp(-u/k^2) = 0.05`` with ``u = ci95^2 / (2 sigma1^2)``.
    """
    p, k = 1.0 - tail_weight, tail_ratio
    lo, hi = ci95 * 1e-5, ci95
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        u = ci95**2 / (2.0 * mid**2)
        val = p * math.exp(-u) + (1.0 - p) * math.exp(-u / k**2)
        if val < 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _radial_cdf(r: float, sigma_along: float, sigma_cross: float, n_grid: int = 4001) -> float:
    """P(radial <= r) for independent N(0, sigma_along^2) x N(0, sigma_cross^2).

    No closed form for unequal sigmas; numerical integration, as in the source.
    """
    if sigma_along == sigma_cross:
        return 1.0 - math.exp(-(r**2) / (2.0 * sigma_along**2))
    x = np.linspace(-r, r, n_grid)
    fx = np.exp(-0.5 * (x / sigma_along) ** 2) / (sigma_along * math.sqrt(2.0 * math.pi))
    y_bound = np.sqrt(np.maximum(r**2 - x**2, 0.0))
    integrand = fx * _erf(y_bound / (sigma_cross * math.sqrt(2.0)))
    return float(_trapz(integrand, x))


@lru_cache(maxsize=256)
def _aniso_sigma_cross(ci95: float, std_ratio: float) -> float:
    """Cross-track sigma such that the anisotropic radial 95th percentile is ci95."""
    lo, hi = ci95 * 1e-5, ci95
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        # CDF falls as sigma grows: bracket the opposite way from the mixture bisection.
        if _radial_cdf(ci95, std_ratio * mid, mid) < 0.95:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=256)
def _aniso_mixture_sigma_cross(
    ci95: float, std_ratio: float, tail_ratio: float, tail_weight: float
) -> float:
    """Core cross-track sigma of the anisotropic mixture, same containment guarantee."""
    p, k = 1.0 - tail_weight, tail_ratio
    lo, hi = ci95 * 1e-5, ci95
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        val = p * _radial_cdf(ci95, std_ratio * mid, mid) + (1.0 - p) * _radial_cdf(
            ci95, std_ratio * k * mid, k * mid
        )
        if val < 0.95:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
