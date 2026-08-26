"""BlueSkyCDaRR — CDaRR's pairwise conflict simulation on BlueSky, made declarative.

The short path, by role:

- **Declare and run an experiment** — :func:`~blueskycdarr.experiment.run_experiment`
  with :class:`~blueskycdarr.experiment.Fixed` / :class:`~blueskycdarr.experiment.Sweep`
  axes, or ``run_one_experiment(*load_run("configs/x.yaml"))`` for a file-driven run.
- **Values you construct** — :class:`~blueskycdarr.config.Config` (and its sections),
  :class:`~blueskycdarr.scenario.PairwiseEncounter`,
  :class:`~blueskycdarr.experiment.Models`, :class:`~blueskycdarr.experiment.MC`, the
  aircraft catalog (:data:`~blueskycdarr.aircraft.MULTIROTOR`,
  :data:`~blueskycdarr.aircraft.FIXEDWING`).
- **Feed MixedVarLSENew** — :func:`~blueskycdarr.blackbox.make_blackbox`.

Usage::

    from blueskycdarr import Fixed, Sweep, MC, Models, Config, run_experiment
    from blueskycdarr import MULTIROTOR, PairwiseEncounter

    res = run_experiment(
        {"pos_ci95": Sweep([3.0, 10.0, 30.0, 92.6]), "vel_ci95": Fixed(1.0)},
        models=Models(aircraft=MULTIROTOR, scenario=PairwiseEncounter()),
        backend=MC(n_encounters=300), seed=7,
    )

A submodule import still reaches everything; this list is the short path for the common
case, not a mirror of the tree. ``import blueskycdarr`` costs numpy + pyyaml — BlueSky
itself loads on first engine use, joblib and pandas inside the functions that need them.
"""

from blueskycdarr.aircraft import CATALOG, FIXEDWING, MULTIROTOR, AircraftModel
from blueskycdarr.blackbox import make_blackbox
from blueskycdarr.config import (
    CommConfig,
    Config,
    ConflictConfig,
    SimulationConfig,
    UncertaintyConfig,
)
from blueskycdarr.experiment import (
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
from blueskycdarr.metrics import MonteCarloEstimate
from blueskycdarr.noise import (
    AnisotropicGaussian,
    AnisotropicMixtureGaussian,
    Gaussian,
    MixtureGaussian,
)
from blueskycdarr.recovery import FTR, PastCPA, ProbabilisticFTR
from blueskycdarr.resolution import MVP, VO
from blueskycdarr.scenario import PairwiseEncounter

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
    "MixtureGaussian",
    "Models",
    "MonteCarloEstimate",
    "PairwiseEncounter",
    "PastCPA",
    "ProbabilisticFTR",
    "SimulationConfig",
    "Sweep",
    "UncertaintyConfig",
    "load_run",
    "make_blackbox",
    "run_experiment",
    "run_one_experiment",
    "sweep_from_file",
]
