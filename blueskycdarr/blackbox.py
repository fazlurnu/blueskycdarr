"""The MixedVarLSENew adapter — this package as an ``mvlse`` blackbox.

``mvlse`` asks for a callable over a batch of points and one ``(y, se)`` per point, in
order, where a point is a mapping in physical units keyed by the design-space names:

    p_reception, max_range_m, kinematics, pos_ci95_m, vel_ci95_ms

(the space of ``examples/opencdarr_blackbox.ipynb``). The target is ``log10 P(LoS)``
with its standard error in decades: a Jeffreys-corrected proportion and the delta method,
exactly the notebook's Monte-Carlo oracle, so swapping the OpenCDaRR backend for this one
is a one-import change:

    from blueskycdarr.blackbox import make_blackbox
    blackbox = make_blackbox(n_encounters=300, seed=7, n_jobs=-1)
    run_lse(space, blackbox, threshold=np.log10(0.02), ...)

Batches loop conditions serially and spend ``n_jobs`` inside each condition, following
the mvlse guidance to parallelise inside the simulator rather than around it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from blueskycdarr.aircraft import aircraft_by_label
from blueskycdarr.config import Config
from blueskycdarr.experiment import MC, Fixed, Models, run_experiment
from blueskycdarr.noise import DEFAULT_NOISE, NoiseShape
from blueskycdarr.recovery import DEFAULT_RECOVERY, Recovery
from blueskycdarr.resolution import DEFAULT_RESOLVER, Resolver
from blueskycdarr.scenario import PairwiseEncounter

Point = Mapping[str, Any]


def make_blackbox(
    *,
    n_encounters: int,
    seed: int = 7,
    base_config: Config | None = None,
    scenario: PairwiseEncounter | None = None,
    recovery: Recovery = DEFAULT_RECOVERY,
    resolver: Resolver = DEFAULT_RESOLVER,
    noise: NoiseShape = DEFAULT_NOISE,
    n_jobs: int = 1,
) -> Callable[[Sequence[Point]], list[tuple[float, float]]]:
    """A ``points -> [(log10 p_los, se)]`` oracle over this package's simulation.

    ``base_config``, ``scenario`` and ``recovery`` pin everything the design space does
    not vary (defaults: the package defaults — the MixedVarLSENew geometry of dpsi 90,
    dcpa 0, past-CPA recovery).
    """
    config = base_config if base_config is not None else Config()
    encounter = scenario if scenario is not None else PairwiseEncounter()

    def blackbox(points: Sequence[Point]) -> list[tuple[float, float]]:
        out = []
        for p in points:
            result = run_experiment(
                {
                    "aircraft": Fixed(str(p["kinematics"])),
                    "pos_ci95": Fixed(float(p["pos_ci95_m"])),
                    "vel_ci95": Fixed(float(p["vel_ci95_ms"])),
                    "reception_prob": Fixed(float(p["p_reception"])),
                    "max_range_m": Fixed(float(p["max_range_m"])),
                },
                models=Models(
                    aircraft=aircraft_by_label(str(p["kinematics"])),
                    scenario=encounter,
                    recovery=recovery,
                    resolver=resolver,
                    noise=noise,
                ),
                backend=MC(n_encounters=n_encounters),
                base_config=config,
                seed=seed,
                n_jobs=n_jobs,
                progress=False,
            )
            estimate = result.cell()
            out.append(log10_p_los(estimate.n_los, estimate.n_encounters))
        return out

    return blackbox


def log10_p_los(n_los: int, n_encounters: int) -> tuple[float, float]:
    """``(log10 p, se)`` from raw counts: Jeffreys-corrected proportion, delta method.

    The correction keeps a zero-loss cell finite and the delta method carries the
    binomial standard error onto the log10 scale — the notebook's estimator, verbatim.
    """
    p_hat = (n_los + 0.5) / (n_encounters + 1.0)
    se_p = float(np.sqrt(p_hat * (1.0 - p_hat) / n_encounters))
    se_log = se_p / (p_hat * np.log(10.0))
    return float(np.log10(p_hat)), float(se_log)
