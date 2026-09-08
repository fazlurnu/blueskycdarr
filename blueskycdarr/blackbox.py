"""The MixedVarLSENew adapter — this package as an ``mvlse`` blackbox.

``mvlse`` asks for a callable over a batch of points and one ``(y, se)`` per point, in
order, where a point is a mapping in physical units. Every key ``run_experiment``
declares is accepted; the historical MixedVarLSENew spellings are translated:

    p_reception -> reception_prob      pos_ci95_m -> pos_ci95
    kinematics  -> aircraft            vel_ci95_ms -> vel_ci95

so both the ``examples/opencdarr_blackbox.ipynb`` space and a space that varies the
geometry (``dpsi``, ``speed``) or the conflict definition (``rpz``) work unchanged.
``fixed`` pins declarable values the design space does not vary (e.g. ``rpz``). The
target is ``log10 P(LoS)`` with its standard error in decades: a Jeffreys-corrected
proportion and the delta method, exactly the notebook's Monte-Carlo oracle:

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

from blueskycdarr.aircraft import MULTIROTOR, AircraftSpec
from blueskycdarr.config import Config
from blueskycdarr.experiment import MC, Fixed, Models, run_experiment
from blueskycdarr.noise import DEFAULT_NOISE, NoiseShape
from blueskycdarr.recovery import DEFAULT_RECOVERY, Recovery
from blueskycdarr.resolution import DEFAULT_RESOLVER, Resolver
from blueskycdarr.scenario import PairwiseEncounter

Point = Mapping[str, Any]

# Design-space name -> run_experiment name; anything else passes through verbatim.
_TO_EXPERIMENT = {
    "p_reception": "reception_prob",
    "pos_ci95_m": "pos_ci95",
    "vel_ci95_ms": "vel_ci95",
    "kinematics": "aircraft",
}


def make_blackbox(
    *,
    n_encounters: int,
    seed: int = 7,
    base_config: Config | None = None,
    scenario: PairwiseEncounter | None = None,
    aircraft: AircraftSpec = MULTIROTOR,
    recovery: Recovery = DEFAULT_RECOVERY,
    resolver: Resolver = DEFAULT_RESOLVER,
    noise: NoiseShape = DEFAULT_NOISE,
    fixed: Mapping[str, Any] | None = None,
    n_jobs: int = 1,
) -> Callable[[Sequence[Point]], list[tuple[float, float]]]:
    """A ``points -> [(log10 p_los, se)]`` oracle over this package's simulation.

    ``base_config``, ``scenario``, ``aircraft`` and ``recovery`` pin everything the
    design space does not vary (defaults: the package defaults — the MixedVarLSENew
    geometry of dpsi 90, dcpa 0, past-CPA recovery). ``fixed`` pins declarable
    experiment values instead (e.g. ``{"rpz": 150.0}``); a point key overrides it.
    ``aircraft`` is the airframe every point flies unless the point carries a
    ``kinematics`` label of its own.
    """
    config = base_config if base_config is not None else Config()
    encounter = scenario if scenario is not None else PairwiseEncounter()
    pinned = {key: Fixed(value) for key, value in (fixed or {}).items()}

    def blackbox(points: Sequence[Point]) -> list[tuple[float, float]]:
        out = []
        for p in points:
            axes = dict(pinned)
            axes.update((_TO_EXPERIMENT.get(k, k), Fixed(v)) for k, v in p.items())
            result = run_experiment(
                axes,
                models=Models(
                    aircraft=aircraft,
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
