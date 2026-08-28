"""Locks for the CDaRR-parity batch (ADR 0007): noise shapes, VO, speed ranges,
declared accuracy, mixed pairs.

Each addition is validated against its own definition: every noise shape must deliver
the configured radial CI95 (the containment guarantee the CDaRR source solves for); the
VO command must exit the cone it was inside; a speed range must draw inside itself
without disturbing the other geometry streams; a declared accuracy must reach the
probabilistic worldview and be refused where nothing reads it.
"""

from __future__ import annotations

import numpy as np
import pytest

from blueskycdarr.aircraft import FIXEDWING, MULTIROTOR, aircraft_from_spec, as_pair
from blueskycdarr.config import Config, UncertaintyConfig
from blueskycdarr.detection import detect
from blueskycdarr.experiment import (
    Condition,
    Models,
    _apply,
    _validate_declared_accuracy_is_read,
)
from blueskycdarr.noise import (
    AnisotropicGaussian,
    AnisotropicMixtureGaussian,
    Gaussian,
    LatencyBiased,
    MixtureGaussian,
    noise_from_spec,
)
from blueskycdarr.recovery import FTR, ProbabilisticFTR, worldview_sigmas
from blueskycdarr.resolution import MVP, VO, resolve_vo, resolver_from_spec
from blueskycdarr.rng import generator, root_seed_sequence
from blueskycdarr.scenario import PairwiseEncounter
from blueskycdarr.state import StateArrays

_LAT = 52.0
_M_PER_DEG_LAT = 111_320.0


def _pair(offset_east_m: float, offset_north_m: float, trk: tuple[float, float],
          gs: tuple[float, float]) -> tuple[StateArrays, StateArrays]:
    coslat = np.cos(np.radians(_LAT))
    lat = np.array([_LAT, _LAT + offset_north_m / _M_PER_DEG_LAT])
    lon = np.array([4.0, 4.0 + offset_east_m / (_M_PER_DEG_LAT * coslat)])
    own = StateArrays.from_track_speed(lat, lon, np.array(trk, float), np.array(gs, float))
    return own, own.reindexed(np.array([1, 0]))


# --- noise shapes ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        Gaussian(),
        MixtureGaussian(tail_ratio=3.0, tail_weight=0.1),
        AnisotropicGaussian(var_ratio=9.0),
        AnisotropicMixtureGaussian(var_ratio=9.0, tail_ratio=3.0, tail_weight=0.1),
    ],
)
def test_every_noise_shape_delivers_the_configured_radial_ci95(shape) -> None:
    """The containment guarantee CDaRR's bisections solve for, checked empirically."""
    rng = generator(root_seed_sequence(3))
    trk = rng.uniform(0.0, 2 * np.pi, 60_000)
    xy = shape.draw(60_000, 10.0, rng, trk, np.full(60_000, 10.3))
    radial95 = np.percentile(np.hypot(xy[:, 0], xy[:, 1]), 95)
    assert abs(radial95 - 10.0) < 0.15


def test_the_mixture_has_the_heavier_tail_and_the_anisotropy_has_the_declared_ratio() -> None:
    rng = generator(root_seed_sequence(4))
    n = 120_000
    trk = np.zeros(n)  # track north: along-track = north component

    plain = Gaussian().draw(n, 10.0, rng, trk, np.full(n, 10.3))
    mixed = MixtureGaussian(tail_ratio=3.0, tail_weight=0.1).draw(
        n, 10.0, rng, trk, np.full(n, 10.3)
    )
    tail = lambda xy: np.mean(np.hypot(xy[:, 0], xy[:, 1]) > 15.0)  # noqa: E731
    assert tail(mixed) > 2.0 * tail(plain)  # same CI95, far heavier beyond it

    aniso = AnisotropicGaussian(var_ratio=9.0).draw(n, 10.0, rng, trk, np.full(n, 10.3))
    ratio = np.var(aniso[:, 1]) / np.var(aniso[:, 0])  # north (along) / east (cross)
    assert 8.0 < ratio < 10.0


def test_gaussian_shape_reproduces_the_pre_shape_draw_stream() -> None:
    """The default shape must not move existing seeded results (ADR 0004 invariant)."""
    std = 10.0 / 2.448
    a = generator(root_seed_sequence(5))
    b = generator(root_seed_sequence(5))
    old_east, old_north = a.normal(0.0, std, (2, 7))
    xy = Gaussian().draw(7, 10.0, b, np.zeros(7), np.full(7, 10.3))
    np.testing.assert_array_equal(xy[:, 0], old_east)
    np.testing.assert_array_equal(xy[:, 1], old_north)


def test_latency_bias_lags_the_track_by_delay_times_speed() -> None:
    """The exp3 latency model: a deterministic along-track lag of delay_s * gs on top
    of the base shape's error, which keeps its own radial containment."""
    rng = generator(root_seed_sequence(9))
    n, gs = 60_000, np.full(60_000, 10.2889)
    trk = np.zeros(n)  # north: along-track = north component
    xy = LatencyBiased(Gaussian(), delay_s=0.1).draw(n, 10.0, rng, trk, gs)
    assert np.mean(xy[:, 1]) == pytest.approx(-0.1 * 10.2889, abs=0.05)  # lags behind
    assert abs(np.mean(xy[:, 0])) < 0.05  # no cross-track bias
    spec = noise_from_spec(
        {"type": "latency_biased", "delay_s": 0.1,
         "base": {"type": "anisotropic_gaussian", "var_ratio": 9.0}}
    )
    assert spec == LatencyBiased(AnisotropicGaussian(var_ratio=9.0), delay_s=0.1)


def test_noise_spec_parses_and_rejects() -> None:
    assert noise_from_spec("gaussian") == Gaussian()
    assert noise_from_spec({"type": "mixture_gaussian", "tail_weight": 0.2}) == (
        MixtureGaussian(tail_weight=0.2)
    )
    with pytest.raises(ValueError, match="unknown noise"):
        noise_from_spec("cauchy")
    with pytest.raises(ValueError, match="tail_weight"):
        MixtureGaussian(tail_weight=1.5)


# --- VO --------------------------------------------------------------------------------


def _line_dcpa_of(dx, dy, du, dv) -> float:
    rel_sq = max(du * du + dv * dv, 1e-6)
    tcpa = -(du * dx + dv * dy) / rel_sq
    return float(np.sqrt(abs(dx * dx + dy * dy - tcpa * tcpa * rel_sq)))


def test_vo_command_exits_the_cone_with_the_shortest_way_out() -> None:
    """Head-on inside the cone: the command's relative line must clear rpz * margin."""
    own, other = _pair(0.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    conflicts = detect(own, other, np.ones(2, dtype=bool), rpz=50.0, t_lookahead=120.0)
    command = resolve_vo(conflicts, own, rpz=50.0, margin=1.05,
                         v_min=np.zeros(2), v_max=np.full(2, 18.0))
    assert set(command.idx) == {0, 1}
    for slot, i in enumerate(command.idx):
        r = np.radians(command.trk[slot])
        ve, vn = command.gs[slot] * np.sin(r), command.gs[slot] * np.cos(r)
        du = ve - (own.gs_east[i] + conflicts.du[i])  # command minus other's velocity
        dv = vn - (own.gs_north[i] + conflicts.dv[i])
        after = _line_dcpa_of(conflicts.dx[i], conflicts.dy[i], du, dv)
        assert after > 50.0 * 1.05 * 0.999  # on the cone edge: grazing the margin zone
        # shortest way out: the speed barely changes, the heading does the work
        assert abs(command.gs[slot] - 15.0) < 1.0


def test_vo_holds_when_already_clear_or_already_inside_the_zone() -> None:
    clear, clear_other = _pair(300.0, 1000.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    conflicts = detect(clear, clear_other, np.ones(2, dtype=bool), rpz=50.0,
                       t_lookahead=120.0)
    # force evaluation despite no detected conflict by marking in_conflict
    conflicts.in_conflict = np.ones(2, dtype=bool)
    command = resolve_vo(conflicts, clear, rpz=50.0, margin=1.05,
                         v_min=np.zeros(2), v_max=np.full(2, 18.0))
    np.testing.assert_allclose(command.gs, 15.0, atol=1e-9)  # outside the cone: hold

    inside, inside_other = _pair(0.0, 30.0, trk=(0.0, 180.0), gs=(15.0, 15.0))
    conflicts_i = detect(inside, inside_other, np.ones(2, dtype=bool), rpz=50.0,
                         t_lookahead=120.0)
    conflicts_i.in_conflict = np.ones(2, dtype=bool)
    command_i = resolve_vo(conflicts_i, inside, rpz=50.0, margin=1.05,
                           v_min=np.zeros(2), v_max=np.full(2, 18.0))
    np.testing.assert_allclose(command_i.gs, 15.0, atol=1e-9)  # no cone to leave: hold


def test_resolver_spec_parses_and_rejects() -> None:
    assert resolver_from_spec("mvp") == MVP()
    assert resolver_from_spec("vo") == VO()
    with pytest.raises(ValueError, match="unknown resolver"):
        resolver_from_spec("orca")


# --- speed ranges ----------------------------------------------------------------------


def test_a_speed_range_draws_per_pair_inside_itself() -> None:
    scenario = PairwiseEncounter(speed=(8.0, 12.0), gs_intr=(10.0, 16.0), pairs=(4, 4))
    geometry = scenario.draw_geometry(generator(root_seed_sequence(6)))
    assert (geometry.gs_own >= 8.0).all() and (geometry.gs_own <= 12.0).all()
    assert (geometry.gs_intr >= 10.0).all() and (geometry.gs_intr <= 16.0).all()
    assert np.std(geometry.gs_own) > 0  # actually heterogeneous
    with pytest.raises(ValueError, match="speed range"):
        PairwiseEncounter(speed=(12.0, 8.0))


def test_a_pinned_speed_consumes_no_randomness_next_to_a_drawn_dpsi() -> None:
    """Adding the range feature must not shift existing fixed-speed streams (ADR 0004)."""
    drawn = PairwiseEncounter(dpsi=None, speed=15.0, pairs=(2, 2))
    a = drawn.draw_geometry(generator(root_seed_sequence(7)))
    b = drawn.draw_geometry(generator(root_seed_sequence(7)))
    np.testing.assert_array_equal(a.dpsi, b.dpsi)
    np.testing.assert_array_equal(a.gs_own, np.full(4, 15.0))
    np.testing.assert_array_equal(a.gs_intr, a.gs_own)


# --- declared accuracy -----------------------------------------------------------------


def test_declared_accuracy_reaches_the_worldview_and_defaults_to_the_truth() -> None:
    matched = worldview_sigmas(UncertaintyConfig(pos_ci95=10.0, vel_ci95=1.0))
    mismatched = worldview_sigmas(
        UncertaintyConfig(pos_ci95=10.0, vel_ci95=1.0, pos_ci95_declared=30.0)
    )
    assert mismatched[0] == pytest.approx(3.0 * matched[0])  # believes 30 m, truth 10 m
    assert mismatched[1] == matched[1]  # velocity belief untouched


def test_an_unread_declared_accuracy_is_refused() -> None:
    config = Config(uncertainty=UncertaintyConfig(pos_ci95=10.0, pos_ci95_declared=30.0))
    probftr = Models(MULTIROTOR, PairwiseEncounter(), recovery=ProbabilisticFTR())
    _validate_declared_accuracy_is_read(config, probftr)  # fine: probFTR reads it
    with pytest.raises(ValueError, match="never reads them"):
        _validate_declared_accuracy_is_read(
            config, Models(MULTIROTOR, PairwiseEncounter(), recovery=FTR())
        )


# --- mixed pairs and routing -----------------------------------------------------------


def test_aircraft_spec_accepts_labels_pairs_and_mappings() -> None:
    assert aircraft_from_spec("multirotor") is MULTIROTOR
    assert as_pair(MULTIROTOR) == (MULTIROTOR, MULTIROTOR)
    mixed = aircraft_from_spec({"ownship": "multirotor", "intruder": "fixedwing"})
    assert mixed == (MULTIROTOR, FIXEDWING)
    assert aircraft_from_spec(["fixedwing", "multirotor"]) == (FIXEDWING, MULTIROTOR)
    with pytest.raises(ValueError, match="ownship"):
        aircraft_from_spec({"ownship": "multirotor"})


def test_apply_routes_the_new_components_and_fields() -> None:
    condition = Condition(
        levels=(),
        values=(
            ("resolver", "vo"),
            ("noise", {"type": "anisotropic_gaussian", "var_ratio": 9.0}),
            ("aircraft", {"ownship": "multirotor", "intruder": "fixedwing"}),
            ("pos_ci95_declared", 30.0),
            ("speed", [8.0, 12.0]),
        ),
    )
    config, models = _apply(condition, Config(), Models(MULTIROTOR, PairwiseEncounter()))
    assert models.resolver == VO()
    assert models.noise == AnisotropicGaussian(var_ratio=9.0)
    assert models.aircraft == (MULTIROTOR, FIXEDWING)
    assert config.uncertainty.pos_ci95_declared == 30.0
    assert models.scenario.speed == (8.0, 12.0)


# --- engine-backed ---------------------------------------------------------------------


def test_vo_and_a_mixed_pair_fly_resolve_and_reproduce() -> None:
    pytest.importorskip("bluesky")
    from blueskycdarr.episode import run_episode
    from blueskycdarr.rng import child

    scenario = PairwiseEncounter(pairs=(1, 2), tlos=45.0, speed=(13.0, 16.0))
    seq = child(root_seed_sequence(0), 0)
    mixed = (MULTIROTOR, FIXEDWING)
    first = run_episode(scenario, mixed, Config(), seq, resolver=VO())
    second = run_episode(scenario, mixed, Config(), seq, resolver=VO())
    assert first.detected.all()
    assert first.n_los == 0
    assert (first.min_sep > 50.0).all()
    np.testing.assert_array_equal(first.min_sep, second.min_sep)
