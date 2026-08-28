"""Engine-backed locks for the fixed-level IPS estimator (``cdarr/ips.py``).

The estimator is validated against its own definition at the two ends where the answer
is exact — a certain event must survive every shell, an unreachable one must collapse
cleanly rather than fabricate a zero — and against the Monte-Carlo anchor on a
moderate-probability cell, judged on the **ratio** of the two estimates (the ADR 0022
convention carried over from OpenCDaRR: at these designs an interval would claim a
precision the level spacing does not support, so the anchor agreement is the evidence).
Reproducibility is the same contract every estimator here signs: ``config + seed ->
result``, bit for bit. These are the expensive tests of this suite — every particle is a
real spawned encounter — so the clouds are small and the horizons short.

These need the CDaRR BlueSky fork; the whole module skips without it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("bluesky")

from cdarr.aircraft import MULTIROTOR  # noqa: E402
from cdarr.config import CommConfig, Config, SimulationConfig, UncertaintyConfig  # noqa: E402
from cdarr.episode import run_episode  # noqa: E402
from cdarr.ips import estimate_rare_prob  # noqa: E402
from cdarr.rng import child, root_seed_sequence  # noqa: E402
from cdarr.scenario import PairwiseEncounter  # noqa: E402

# One encounter per particle, spawned on a collision course (dcpa 0) so the ladder has
# something to concentrate on; the CDR chain is what stands between spawn and LoS.
_IPS_SCENARIO = PairwiseEncounter(pairs=(1, 1), dcpa=0.0, tlos=20.0)

# A cell where the CDR fails often enough for plain MC to anchor the comparison:
# heavy GNSS noise, four broadcasts in five lost, half a second stale on arrival.
_NOISY = Config(
    uncertainty=UncertaintyConfig(pos_ci95=40.0, vel_ci95=3.0),
    comm=CommConfig(reception_prob=0.2, latency_s=0.5),
    simulation=SimulationConfig(t_max=90.0),
)


def test_a_certain_event_survives_every_shell() -> None:
    """Blind aircraft on a collision course breach with probability one, and the ladder
    must say exactly that: every survival fraction 1.0, no collapse, P̂ = 1."""
    blind = Config(
        comm=CommConfig(reception_prob=0.0), simulation=SimulationConfig(t_max=120.0)
    )
    est = estimate_rare_prob(
        _IPS_SCENARIO, MULTIROTOR, blind,
        levels=[100.0, 50.0], n_particles=6, reps=1, seed=0,
    )
    assert est.p_los == 1.0
    assert est.n_collapsed == 0
    assert est.reps[0].survival == (1.0, 1.0)


def test_an_unreachable_level_collapses_cleanly() -> None:
    """Under perfect CNS the resolver keeps every pair far outside 10 m, so that shell
    is empty: the replication must report the collapse (a signal the ladder is spaced
    too aggressively), not dress it up as a measured zero."""
    est = estimate_rare_prob(
        _IPS_SCENARIO, MULTIROTOR, Config(simulation=SimulationConfig(t_max=60.0)),
        levels=[10.0], n_particles=6, reps=1, seed=0,
    )
    assert est.p_los == 0.0
    assert est.n_collapsed == 1
    assert est.reps[0].collapsed_at == 0
    assert est.reps[0].survival == (0.0,)


def test_the_estimate_reproduces_bit_for_bit_from_its_seed() -> None:
    """config + seed -> result, the ladder included: the whole estimate — every
    replication's survival tuple — is a pure function of the seed (stream addressing is
    stateless), and a different seed actually reaches the draws."""
    kwargs = dict(levels=[100.0, 50.0], n_particles=6, reps=1)
    first = estimate_rare_prob(_IPS_SCENARIO, MULTIROTOR, _NOISY, seed=5, **kwargs)
    second = estimate_rare_prob(_IPS_SCENARIO, MULTIROTOR, _NOISY, seed=5, **kwargs)
    other = estimate_rare_prob(_IPS_SCENARIO, MULTIROTOR, _NOISY, seed=6, **kwargs)
    assert first == second
    assert first != other


def test_ips_agrees_with_the_monte_carlo_anchor() -> None:
    """The port's validation: on a cell MC can still measure, the two estimators — same
    scenario distribution, same config, same models, independent samplings — must land
    within a factor of 2.5 of each other (the ratio judgment; both are deterministic
    per seed, so this is a lock, not a flaky statistical hope)."""
    mc_scenario = PairwiseEncounter(pairs=(5, 4), dcpa=0.0, tlos=20.0)
    root = root_seed_sequence(21)
    n_los = n_encounters = 0
    for j in range(2):
        result = run_episode(mc_scenario, MULTIROTOR, _NOISY, child(root, j))
        n_los += result.n_los
        n_encounters += result.min_sep.size
    p_mc = n_los / n_encounters
    assert 0.0 < p_mc < 1.0  # the anchor actually measured something

    est = estimate_rare_prob(
        _IPS_SCENARIO, MULTIROTOR, _NOISY,
        levels=[100.0, 50.0], n_particles=16, reps=2, seed=5,
    )
    assert est.n_collapsed == 0
    assert 1 / 2.5 < est.p_los / p_mc < 2.5


def test_an_encounter_ending_mid_tick_does_not_poison_the_next_restore() -> None:
    """The regression the first MC-vs-IPS comparison run caught: an encounter that ends
    exactly on a command-stacking CDR tick (here forced by putting ``t_max`` on the
    tick grid, mid-conflict, with noisy views re-commanding every tick) leaves its dead
    world's commands pending — and the next particle's restore must not trip the
    boundary guard on them. The lock is that this completes; before the fix it raised
    from deep inside level 2."""
    capped = Config(
        uncertainty=UncertaintyConfig(pos_ci95=40.0, vel_ci95=3.0),
        comm=CommConfig(reception_prob=0.2, latency_s=0.5),
        simulation=SimulationConfig(t_max=6.0),
    )
    est = estimate_rare_prob(
        _IPS_SCENARIO, MULTIROTOR, capped,
        levels=[400.0, 40.0], n_particles=4, reps=1, seed=0,
    )
    assert 0.0 <= est.p_los <= 1.0  # the value is not the point; finishing is


def test_inputs_are_refused_at_the_boundary() -> None:
    """Fail-fast schema, the config convention: a multi-pair scenario estimates a
    different number, and a non-decreasing ladder is not a ladder."""
    good = dict(n_particles=4, reps=1, seed=0)
    with pytest.raises(ValueError, match="one encounter"):
        estimate_rare_prob(
            PairwiseEncounter(pairs=(1, 2), tlos=20.0), MULTIROTOR, Config(),
            levels=[50.0], **good,
        )
    for bad_levels in ([], [50.0, 100.0], [50.0, 50.0], [50.0, -5.0]):
        with pytest.raises(ValueError, match="strictly decreasing"):
            estimate_rare_prob(
                _IPS_SCENARIO, MULTIROTOR, Config(), levels=bad_levels, **good
            )
    with pytest.raises(ValueError, match="n_particles"):
        estimate_rare_prob(
            _IPS_SCENARIO, MULTIROTOR, Config(),
            levels=[50.0], n_particles=0, reps=1, seed=0,
        )
    with pytest.raises(ValueError, match="reps"):
        estimate_rare_prob(
            _IPS_SCENARIO, MULTIROTOR, Config(),
            levels=[50.0], n_particles=4, reps=0, seed=0,
        )
