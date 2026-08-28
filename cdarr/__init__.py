"""BlueSkyCDaRR — CDaRR's pairwise conflict simulation on BlueSky, made declarative.

The short path, by role:

- **Declare and run an experiment** — :func:`~cdarr.experiment.run_experiment`
  with :class:`~cdarr.experiment.Fixed` / :class:`~cdarr.experiment.Sweep`
  axes, or ``run_one_experiment(*load_run("configs/x.yaml"))`` for a file-driven run.
- **Values you construct** — :class:`~cdarr.config.Config` (and its sections),
  :class:`~cdarr.scenario.PairwiseEncounter`,
  :class:`~cdarr.experiment.Models`, :class:`~cdarr.experiment.MC`, the
  aircraft catalog (:data:`~cdarr.aircraft.MULTIROTOR`,
  :data:`~cdarr.aircraft.FIXEDWING`).
- **Estimate a rare P(LoS)** — :func:`~cdarr.ips.estimate_rare_prob`, fixed-level
  splitting for the probabilities plain MC starves on (ADR 0008).
- **Feed MixedVarLSENew** — :func:`~cdarr.blackbox.make_blackbox`.

Usage::

    from cdarr import Fixed, Sweep, MC, Models, Config, run_experiment
    from cdarr import MULTIROTOR, PairwiseEncounter

    res = run_experiment(
        {"pos_ci95": Sweep([3.0, 10.0, 30.0, 92.6]), "vel_ci95": Fixed(1.0)},
        models=Models(aircraft=MULTIROTOR, scenario=PairwiseEncounter()),
        backend=MC(n_encounters=300), seed=7,
    )

A submodule import still reaches everything; this list is the short path for the common
case, not a mirror of the tree. ``import cdarr`` costs numpy + pyyaml — BlueSky
itself loads on first engine use, joblib and pandas inside the functions that need them.
"""

from cdarr.aircraft import CATALOG, FIXEDWING, MULTIROTOR, AircraftModel
from cdarr.blackbox import make_blackbox
from cdarr.config import (
    CommConfig,
    Config,
    ConflictConfig,
    SimulationConfig,
    UncertaintyConfig,
)
from cdarr.experiment import (
    MC,
    ExperimentResult,
    Fixed,
    Models,
    Sweep,
    load_run,
    run_experiment,
    run_one_experiment,
    sweep_from_file,
)
from cdarr.ips import IPSEstimate, estimate_rare_prob
from cdarr.metrics import MonteCarloEstimate
from cdarr.noise import (
    AnisotropicGaussian,
    AnisotropicMixtureGaussian,
    Gaussian,
    LatencyBiased,
    MixtureGaussian,
)
from cdarr.recovery import FTR, PastCPA, ProbabilisticFTR
from cdarr.resolution import MVP, VO
from cdarr.scenario import PairwiseEncounter

__version__ = "0.0.0"

__all__ = [
    "CATALOG",
    "FIXEDWING",
    "FTR",
    "MC",
    "MULTIROTOR",
    "MVP",
    "VO",
    "AircraftModel",
    "AnisotropicGaussian",
    "AnisotropicMixtureGaussian",
    "CommConfig",
    "Config",
    "ConflictConfig",
    "ExperimentResult",
    "Fixed",
    "Gaussian",
    "IPSEstimate",
    "LatencyBiased",
    "MixtureGaussian",
    "Models",
    "MonteCarloEstimate",
    "PairwiseEncounter",
    "PastCPA",
    "ProbabilisticFTR",
    "SimulationConfig",
    "Sweep",
    "UncertaintyConfig",
    "estimate_rare_prob",
    "load_run",
    "make_blackbox",
    "run_experiment",
    "run_one_experiment",
    "sweep_from_file",
]
