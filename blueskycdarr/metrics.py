"""The Monte Carlo estimate — what one experiment condition reports (ADR 0004).

The single number the study runs on is ``p_los_run``: the fraction of encounters whose
true minimum separation dipped below the protected zone. MixedVarLSENew's blackbox reads
``n_encounters`` and ``n_los`` (its Jeffreys correction wants the raw counts), so both
are first-class fields, not derivables — and the raw counts are the *only* uncertainty
reporting: no interval columns (ADR 0004 update; a consumer that wants bounds derives
them from ``n_los`` / ``n_encounters``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from blueskycdarr.episode import EpisodeResult


@dataclass(frozen=True)
class MonteCarloEstimate:
    """One condition's estimate, aggregated over its episodes.

    ``min_sep`` keeps the raw per-encounter minima — the estimate stores the raw result,
    not a reduced metric, so a statistic added later needs no new simulation (the
    OpenCDaRR caching rule).
    """

    n_encounters: int
    n_los: int
    min_sep: np.ndarray
    detection_rate: float
    n_unsettled: int  # episodes ended by the t_max cap rather than the all-clear

    @property
    def p_los_run(self) -> float:
        return self.n_los / self.n_encounters if self.n_encounters else float("nan")

    @property
    def median_min_sep(self) -> float:
        return float(np.median(self.min_sep)) if self.min_sep.size else float("nan")


def combine(episodes: Sequence[EpisodeResult]) -> MonteCarloEstimate:
    """Fold a condition's episodes into one estimate."""
    min_sep = (
        np.concatenate([e.min_sep for e in episodes]) if episodes else np.empty(0)
    )
    detected = (
        np.concatenate([e.detected for e in episodes]) if episodes else np.empty(0, dtype=bool)
    )
    return MonteCarloEstimate(
        n_encounters=int(min_sep.size),
        n_los=int(sum(e.n_los for e in episodes)),
        min_sep=min_sep,
        detection_rate=float(np.mean(detected)) if detected.size else float("nan"),
        n_unsettled=int(sum(not e.settled for e in episodes)),
    )
