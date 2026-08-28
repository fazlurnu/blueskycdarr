"""Locks for the ADS-L layer (``cdarr/adsl.py``, ADR 0002).

Each communication effect is validated against its own definition: noise against the
configured CI95, reception against its probability, jitter against its bounds, latency
against delivery timing, and the surveillance range against a hard cutoff. The final test
pins the reduction to CDaRR's channel (zero jitter/latency, no gate: one delivery per
interval, holdover on loss).
"""

from __future__ import annotations

import numpy as np

from cdarr.adsl import BroadcastChannel, ContactTable, noisy_snapshot
from cdarr.config import CommConfig, UncertaintyConfig
from cdarr.rng import generator, root_seed_sequence, spawn
from cdarr.state import StateArrays


def _fleet(n: int, spacing_m: float = 1000.0) -> StateArrays:
    """n aircraft on one latitude, ``spacing_m`` apart, flying north at 15 m/s."""
    lon = np.arange(n) * spacing_m / (111_320.0 * np.cos(np.radians(52.0)))
    return StateArrays.from_track_speed(
        lat=np.full(n, 52.0), lon=lon, trk=np.zeros(n), gs=np.full(n, 15.0)
    )


def _channel(
    comm: CommConfig, uncertainty: UncertaintyConfig, seed: int = 0
) -> tuple[BroadcastChannel, tuple[np.random.Generator, ...]]:
    """A channel plus its (measurement, reception, schedule) streams — the streams are
    method arguments now, not fields (the state/streams split IPS cloning rests on)."""
    streams = spawn(root_seed_sequence(seed), 3)
    rngs = tuple(generator(s) for s in streams)
    return BroadcastChannel(comm=comm, uncertainty=uncertainty), rngs


# --- measurement noise -----------------------------------------------------------------


def test_position_noise_matches_the_configured_ci95() -> None:
    """The 95th percentile of the radial position error is the configured CI95."""
    truth = _fleet(20_000)
    rng = generator(root_seed_sequence(1))
    snap = noisy_snapshot(truth, np.arange(truth.n), UncertaintyConfig(pos_ci95=10.0), rng)
    east = (snap.lon - truth.lon) * 111_320.0 * np.cos(np.radians(truth.lat))
    north = (snap.lat - truth.lat) * 111_320.0
    radial95 = np.percentile(np.hypot(east, north), 95)
    assert abs(radial95 - 10.0) < 0.3


def test_velocity_noise_is_self_consistent() -> None:
    """trk/gs are recomputed from the noised components — one velocity, every consumer
    (the CDaRR detector/resolver inconsistency this package removes, ADR 0002 D1)."""
    truth = _fleet(100)
    rng = generator(root_seed_sequence(2))
    snap = noisy_snapshot(truth, np.arange(truth.n), UncertaintyConfig(vel_ci95=3.0), rng)
    assert not np.allclose(snap.gs, truth.gs)  # noise reached the scalar fields
    np.testing.assert_allclose(snap.gs, np.hypot(snap.gs_east, snap.gs_north), rtol=1e-12)
    np.testing.assert_allclose(
        snap.gs_east, snap.gs * np.sin(np.radians(snap.trk)), atol=1e-9
    )


def test_zero_uncertainty_is_the_identity() -> None:
    truth = _fleet(10)
    snap = noisy_snapshot(
        truth, np.arange(truth.n), UncertaintyConfig(), generator(root_seed_sequence(3))
    )
    np.testing.assert_array_equal(snap.lat, truth.lat)
    np.testing.assert_array_equal(snap.gs, truth.gs)


# --- the channel -----------------------------------------------------------------------


def test_reception_probability_is_honoured_per_transmission() -> None:
    truth = _fleet(2)
    channel, rngs = _channel(CommConfig(reception_prob=0.3), UncertaintyConfig())
    channel.initialise(truth.n, rngs[2])
    contacts = ContactTable.empty(truth)
    delivered = 0
    for k in range(4000):
        t = float(k)
        channel.transmit_due(t, truth, *rngs)
        channel.deliver_due(t, contacts)
        delivered += int(contacts.t_tx[0] == t)  # a fresh contact carries this tick's stamp
    assert abs(delivered / 4000 - 0.3) < 0.03


def test_jitter_keeps_gaps_inside_the_configured_bounds() -> None:
    truth = _fleet(2)
    channel, rngs = _channel(
        CommConfig(broadcast_interval_s=1.0, broadcast_jitter_s=0.2), UncertaintyConfig()
    )
    channel.initialise(truth.n, rngs[2])
    gaps = []
    for _ in range(500):
        t = float(channel.next_tx[0])  # fire aircraft 0's slot exactly when it is due
        channel.transmit_due(t, truth, *rngs)
        gaps.append(float(channel.next_tx[0]) - t)
    gaps_arr = np.asarray(gaps)
    assert gaps_arr.min() >= 0.8 - 1e-9 and gaps_arr.max() <= 1.2 + 1e-9
    assert gaps_arr.std() > 0.05  # actually dithered, not constant


def test_latency_delays_usability_and_the_content_is_the_transmit_time_state() -> None:
    truth = _fleet(2)
    channel, rngs = _channel(CommConfig(latency_s=0.5), UncertaintyConfig())
    channel.initialise(truth.n, rngs[2])
    contacts = ContactTable.empty(truth)

    channel.transmit_due(0.0, truth, *rngs)
    channel.deliver_due(0.4, contacts)
    assert not contacts.valid.any()  # still in flight

    moved = _fleet(2)
    moved.lat += 0.01  # truth has moved on; the message must not reflect it
    channel.deliver_due(0.5, contacts)
    assert contacts.valid.all()
    np.testing.assert_array_equal(contacts.states.lat, truth.lat)
    assert contacts.t_tx[0] == 0.0  # staleness accounting starts at transmission


def test_surveillance_range_is_a_hard_gate_at_transmit_time() -> None:
    near = _fleet(2, spacing_m=400.0)
    far = _fleet(2, spacing_m=4000.0)
    for truth, expect in ((near, True), (far, False)):
        channel, rngs = _channel(CommConfig(max_range_m=1000.0), UncertaintyConfig())
        channel.initialise(truth.n, rngs[2])
        contacts = ContactTable.empty(truth)
        channel.transmit_due(0.0, truth, *rngs)
        channel.deliver_due(0.0, contacts)
        assert bool(contacts.valid.all()) is expect


def test_contacts_start_invalid_until_first_delivery() -> None:
    """An aircraft never heard does not exist to its counterpart (ADR 0002 D2)."""
    truth = _fleet(4)
    contacts = ContactTable.empty(truth)
    assert not contacts.valid.any()
    _, seen = contacts.view_of_counterparts()
    assert not seen.any()


def test_reduces_to_cdarr_channel_without_jitter_latency_and_gate() -> None:
    """interval = CDR cadence, no jitter/latency/gate: exactly one delivery per tick,
    and a lost transmission leaves the previous contact standing (the holdover)."""
    truth = _fleet(2)
    channel, rngs = _channel(CommConfig(reception_prob=0.5), UncertaintyConfig(), seed=9)
    channel.initialise(truth.n, rngs[2])
    contacts = ContactTable.empty(truth)
    seen_tx_times: list[float] = []
    for k in range(200):
        t = float(k)
        channel.transmit_due(t, truth, *rngs)
        channel.deliver_due(t, contacts)
        if contacts.valid[0]:
            seen_tx_times.append(float(contacts.t_tx[0]))
    held = np.array(seen_tx_times)
    assert (np.diff(held) >= 0).all()  # never goes backwards
    assert set(np.unique(held)) < set(float(k) for k in range(200))  # only tick-times
    assert (np.diff(held) > 1.0 + 1e-9).any()  # some losses -> holdover across ticks


def test_channel_and_contact_copies_share_the_past_but_not_the_future() -> None:
    """The IPS clone contract: a copy holds the same schedule, in-flight messages and
    contacts, and nothing done to one side reaches the other."""
    truth = _fleet(2)
    channel, rngs = _channel(CommConfig(latency_s=0.5), UncertaintyConfig())
    channel.initialise(truth.n, rngs[2])
    channel.transmit_due(0.0, truth, *rngs)  # one message now in flight
    dup = channel.copy()
    np.testing.assert_array_equal(dup.next_tx, channel.next_tx)

    # landing the copy's message must not drain the original's queue
    dup_contacts = ContactTable.empty(truth)
    dup.deliver_due(0.5, dup_contacts)
    assert dup_contacts.valid.all()
    original_contacts = ContactTable.empty(truth)
    channel.deliver_due(0.5, original_contacts)
    assert original_contacts.valid.all()

    # and a contact-table copy is independent storage, not a view
    clone = original_contacts.copy()
    clone.valid[:] = False
    clone.states.lat += 1.0
    assert original_contacts.valid.all()
    np.testing.assert_array_equal(original_contacts.states.lat, truth.lat)
