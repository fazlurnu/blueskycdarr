"""Conflict recovery — when a resolving aircraft may resume its navigation (ADR 0006).

Three models, all CDaRR's, selectable as a component on the models bundle:

- :class:`PastCPA` (``crr_resumenav_cpa.py``) — geometric: past closest approach, not in
  loss of separation, and (optionally) not in a *bouncing* near-parallel geometry.
- :class:`FTR` (``crr_resumenav_ftr.py``) — the double criteria: release only when the
  aircraft's **commanded course** clears the protected zone against both intruder
  hypotheses — the intruder keeps its current velocity (criterion 1), or reverts to the
  velocity it had when the conflict started (criterion 2). Both use the *unconstrained*
  line-CPA miss distance, so an already-resolved geometry (passing behind) releases
  early — FTR's point — while a resolved *pass* keeps its cleared miss distance.
- :class:`ProbabilisticFTR` (``crr_resumenav_probabilistic_ftr.py``) — the same two
  criteria under the worldview uncertainty: release when
  ``P(DCPA > rpz | hypothesis) > gamma`` for both, with relative position and velocity
  Gaussian around the perceived values. The probability is CDaRR's analytical form:
  integrate over the *direction* of the relative velocity (a projected-normal density)
  and, per direction, a folded-normal tail for the cross-line miss — ported here
  vectorised and in log space, specialised to the isotropic covariances this package
  produces (``sigma^2 I``).

Every model reads the perceived geometry from the detection pass; the FTR family also
reads the current *commanded* velocity (what the aircraft is flying toward) and the
recorded conflict-start velocity of the counterpart, which the episode maintains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from blueskycdarr.config import UncertaintyConfig
from blueskycdarr.detection import Conflicts
from blueskycdarr.noise import CI95_TO_STD_2D
from blueskycdarr.state import StateArrays

try:  # scipy is present transitively (BlueSky depends on it); fall back to math.erf
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover - exercised only without scipy
    _erf = np.vectorize(math.erf)

_BOUNCING_TRK_DIFF_DEG = 30.0
_MIN_REL_SPEED_SQ = 1e-6  # CDaRR's floor against division by zero for parallel flight
_MIN_SIGMA = 1e-3  # CDaRR regularises covariances with ~1e-6 I; same floor as a std


# --- the components --------------------------------------------------------------------


@dataclass(frozen=True)
class PastCPA:
    """Geometric recovery: past CPA, separated, and not bouncing."""

    bouncing_guard: bool = True


@dataclass(frozen=True)
class FTR:
    """Free to return: the commanded course clears both intruder hypotheses."""


@dataclass(frozen=True)
class ProbabilisticFTR:
    """FTR with confidence: both clearance probabilities must exceed ``gamma``.

    ``k_theta`` is the angular quadrature resolution of the analytical probability
    (CDaRR's default 256; its Appendix B validates the integral against Monte Carlo).
    """

    gamma: float = 0.999
    k_theta: int = 256

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.k_theta < 8:
            raise ValueError(f"k_theta must be >= 8, got {self.k_theta}")


Recovery = PastCPA | FTR | ProbabilisticFTR

# The package default (CDaRR's sim_config default), importable as a singleton so
# function signatures can default to it without constructing in the argument list.
DEFAULT_RECOVERY: Recovery = PastCPA()

_LABELS: dict[str, type] = {
    "pastcpa": PastCPA,
    "ftr": FTR,
    "probabilistic_ftr": ProbabilisticFTR,
}


def recovery_from_spec(spec: object, *, source: object = "<spec>") -> Recovery:
    """A recovery component from its file spelling: a bare name or ``{type: ..., params}``.

    The registry names match OpenCDaRR's (``pastcpa``, ``ftr``, ``probabilistic_ftr``).
    """
    if isinstance(spec, PastCPA | FTR | ProbabilisticFTR):
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("type", None)
    else:
        raise ValueError(
            f"{source}: recovery must be a name or a {{type: ...}} mapping, got {spec!r}"
        )
    if name not in _LABELS:
        raise ValueError(f"{source}: unknown recovery {name!r}. Known: {sorted(_LABELS)}")
    try:
        return _LABELS[name](**params)
    except TypeError as err:
        raise ValueError(f"{source}: bad recovery parameters for {name!r}: {err}") from None


def worldview_sigmas(uncertainty: UncertaintyConfig) -> tuple[float, float]:
    """The relative-frame standard deviations the probabilistic criteria assume.

    Both aircraft measured at the *declared* CI95 when one is set (CDaRR's exp5
    calibration mismatch — the system believes an accuracy other than the truth,
    ADR 0007), else the actual CI95, combined over the two aircraft:
    ``sqrt(2) * ci95 / 2.448`` (CDaRR's sigma_r / sigma_v).
    """
    pos = (
        uncertainty.pos_ci95_declared
        if uncertainty.pos_ci95_declared is not None
        else uncertainty.pos_ci95
    )
    vel = (
        uncertainty.vel_ci95_declared
        if uncertainty.vel_ci95_declared is not None
        else uncertainty.vel_ci95
    )
    return (
        math.sqrt(2.0) * pos / CI95_TO_STD_2D,
        math.sqrt(2.0) * vel / CI95_TO_STD_2D,
    )


# --- dispatch --------------------------------------------------------------------------


def recovered_mask(
    recovery: Recovery,
    *,
    resolving: np.ndarray,
    conflicts: Conflicts,
    own: StateArrays,
    other: StateArrays,
    commanded_v: tuple[np.ndarray, np.ndarray],
    initial_other_v: tuple[np.ndarray, np.ndarray],
    rpz: float,
    margin: float,
    rel_pos_sigma: float,
    rel_vel_sigma: float,
) -> np.ndarray:
    """Boolean mask of aircraft that may resume navigation this tick.

    ``commanded_v`` is the (east, north) velocity the aircraft is *currently flying
    toward* — the command applied at the previous CDR tick, never the one about to be
    issued. The caller owns that ordering, and it is load-bearing: CDaRR's criteria read
    ``ap.trk`` (the previously stacked command), which guarantees a fresh resolution is
    flown for one CDR period before release may judge it; evaluating the same tick's
    fresh command instead lets FTR discard avoidance courses that were never flown
    (measured: P(LoS) 0.99 vs CDaRR's 0.03 on the identical condition — ADR 0006).

    ``initial_other_v`` is the counterpart velocity recorded when its conflict started
    (NaN where unrecorded — the FTR family falls back to the current velocity, exactly
    CDaRR's ``.get(conflict, current)``); ``rel_pos_sigma`` / ``rel_vel_sigma`` are the
    combined worldview standard deviations only :class:`ProbabilisticFTR` reads.
    """
    if isinstance(recovery, PastCPA):
        return past_cpa_recovered(
            resolving, conflicts, own, other, rpz, margin, recovery.bouncing_guard
        )
    active = np.asarray(resolving, dtype=bool) & conflicts.seen
    idx = np.flatnonzero(active)
    mask = np.zeros_like(active)
    if idx.size == 0:
        return mask

    du1, dv1, du2, dv2 = _hypothesis_velocities(idx, commanded_v, other, initial_other_v)
    dx = conflicts.dx[idx]
    dy = conflicts.dy[idx]

    if isinstance(recovery, FTR):
        release = (_line_dcpa(dx, dy, du1, dv1) > rpz) & (_line_dcpa(dx, dy, du2, dv2) > rpz)
    else:
        p1 = _p_line_dcpa_exceeds(
            rpz, dx, dy, du1, dv1, rel_pos_sigma, rel_vel_sigma, recovery.k_theta
        )
        p2 = _p_line_dcpa_exceeds(
            rpz, dx, dy, du2, dv2, rel_pos_sigma, rel_vel_sigma, recovery.k_theta
        )
        release = (p1 > recovery.gamma) & (p2 > recovery.gamma)

    mask[idx[release]] = True
    return mask


def past_cpa_recovered(
    resolving: np.ndarray,
    conflicts: Conflicts,
    own: StateArrays,
    other: StateArrays,
    rpz: float,
    margin: float,
    bouncing_guard: bool = True,
) -> np.ndarray:
    """The geometric model: past CPA, not in horizontal LoS, not bouncing."""
    # CDaRR tests dot(drel, v_self - v_other) < 0 with drel = self -> other. The
    # detection pass stores (du, dv) = v_other - v_self, so the same test here is the
    # dot product with the sign flipped.
    dot = conflicts.dx * conflicts.du + conflicts.dy * conflicts.dv
    past_cpa = dot > 0.0

    hor_los = conflicts.dist < rpz

    trk_diff = np.abs((own.trk - other.trk + 180.0) % 360.0 - 180.0)
    bouncing = (trk_diff < _BOUNCING_TRK_DIFF_DEG) & (conflicts.dist < rpz * margin)
    if not bouncing_guard:
        bouncing = np.zeros_like(past_cpa)

    clear = past_cpa & ~hor_los & ~bouncing
    return np.asarray(resolving, dtype=bool) & clear & conflicts.seen


# --- the FTR criteria ------------------------------------------------------------------


def _hypothesis_velocities(
    idx: np.ndarray,
    commanded_v: tuple[np.ndarray, np.ndarray],
    other: StateArrays,
    initial_other_v: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Relative velocities (self minus other) for the two intruder hypotheses."""
    cmd_e, cmd_n = commanded_v[0][idx], commanded_v[1][idx]
    cur_e, cur_n = other.gs_east[idx], other.gs_north[idx]
    init_e = np.where(np.isnan(initial_other_v[0][idx]), cur_e, initial_other_v[0][idx])
    init_n = np.where(np.isnan(initial_other_v[1][idx]), cur_n, initial_other_v[1][idx])
    return cmd_e - cur_e, cmd_n - cur_n, cmd_e - init_e, cmd_n - init_n


def _line_dcpa(
    dx: np.ndarray, dy: np.ndarray, du: np.ndarray, dv: np.ndarray
) -> np.ndarray:
    """Miss distance of the unconstrained relative *line* (CDaRR's ``calculate_dcpa``).

    Sign conventions cancel: only ``tcpa**2`` enters, so self-minus-other and
    other-minus-self relative velocities give the same distance.
    """
    rel_sq = np.maximum(du * du + dv * dv, _MIN_REL_SPEED_SQ)
    tcpa = -(du * dx + dv * dy) / rel_sq
    return np.sqrt(np.abs(dx * dx + dy * dy - tcpa * tcpa * rel_sq))


# --- the analytical clearance probability ----------------------------------------------


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return 0.5 * (1.0 + _erf(np.asarray(z) / math.sqrt(2.0)))


def _p_line_dcpa_exceeds(
    x: float,
    mu_re: np.ndarray,
    mu_rn: np.ndarray,
    mu_ve: np.ndarray,
    mu_vn: np.ndarray,
    sigma_r: float,
    sigma_v: float,
    k_theta: int,
) -> np.ndarray:
    """``P(line DCPA > x)`` for ``r ~ N(mu_r, sigma_r^2 I)``, ``v ~ N(mu_v, sigma_v^2 I)``.

    CDaRR's ``analytical_dcpa_prob_gt`` (``crr_resumenav_probabilistic_ftr.py``),
    specialised to isotropic covariances and vectorised over aircraft: the direction of
    the relative velocity carries a projected-normal density (evaluated in log space —
    the velocity signal-to-noise ratio makes ``exp(z^2/2)`` overflow otherwise), and per
    direction the cross-line component of ``r`` is Gaussian, so ``P(|D| > x)`` is a
    folded-normal tail. Weights are normalised over the ``k_theta`` grid.
    Validated against Monte Carlo in the tests (CDaRR's Appendix B check).
    """
    sigma_r = max(float(sigma_r), _MIN_SIGMA)
    sigma_v = max(float(sigma_v), _MIN_SIGMA)

    theta = np.linspace(0.0, 2.0 * math.pi, int(k_theta), endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Projected-normal log density over the velocity direction, isotropic case:
    # a = 1/sigma_v^2 is direction-independent, b = (u . mu_v)/sigma_v^2, z = b/sqrt(a).
    b = (np.outer(mu_ve, cos_t) + np.outer(mu_vn, sin_t)) / sigma_v**2  # (M, K)
    z = b * sigma_v
    log_a = -2.0 * math.log(sigma_v)

    log_term1 = -log_a
    log_phi_z = np.log(np.maximum(_phi(z), 1e-300))
    log_term2_abs = (
        np.log(np.maximum(np.abs(b), 1e-300))
        + 0.5 * math.log(2.0 * math.pi)
        - 1.5 * log_a
        + 0.5 * z * z
        + log_phi_z
    )
    log_term = np.where(
        b >= 0,
        np.logaddexp(log_term1, log_term2_abs),
        # b < 0: term = 1/a - |term2|, positive by theory; guarded like CDaRR.
        log_term1
        + np.log(
            np.maximum(1.0 - np.exp(np.minimum(log_term2_abs - log_term1, 500.0)), 1e-300)
        ),
    )
    log_term -= log_term.max(axis=1, keepdims=True)
    weights = np.exp(log_term)
    weights /= weights.sum(axis=1, keepdims=True)

    # Per direction, the cross-line miss is N(u_perp . mu_r, sigma_r^2); fold at +-x.
    m = np.outer(mu_re, -sin_t) + np.outer(mu_rn, cos_t)  # (M, K)
    inside = np.clip(_phi((x - m) / sigma_r) - _phi((-x - m) / sigma_r), 0.0, 1.0)
    return np.clip(np.sum(weights * (1.0 - inside), axis=1), 0.0, 1.0)
