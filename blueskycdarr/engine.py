"""The BlueSky boundary — the only module that imports ``bluesky`` (ADR 0001).

Everything third-party lives behind this seam: process-wide initialisation, aircraft
creation (``cre`` / ``creconfs``), the fork's turn-rate limiter arrays, unit conversions
(BlueSky stacks speeds in knots, spawns miss distances in NM), stepping, and the world
snapshot (:class:`WorldSnapshot`) that rare-event splitting clones particles with. The
CDR chain and the channel never touch ``bs.*``, so they stay pure and testable.

One conversion here is a trap this module exists to contain: **every speed BlueSky
accepts** — ``cre``'s ``acspd``, ``creconfs``'s ``spd``, the ``SPD`` stack command — **is
calibrated airspeed**, while everything this package computes and commands is a ground
speed. At 100 m the two differ by the air-density factor 1.0048; harmless once, but a
loop that reads ground speed back and re-commands it as CAS *compounds* the factor per
command — CDaRR's resolution path did exactly that, and its drones crept ~0.5% faster
per re-command (``notebooks/bluesky_speed_command.ipynb`` demonstrates it). This module
therefore converts ground -> CAS at the boundary, so a commanded ground speed is the
ground speed flown, exactly.

The engine must be the CDaRR fork (branch ``CDaRR``): its per-aircraft ``max_tr`` /
``max_dtr2`` limiter arrays and ``creconfs`` are load-bearing. :func:`ensure_engine`
fails fast with the install line when they are missing, rather than running stock
dynamics silently.

BlueSky is a process-global singleton, so one process hosts one world at a time; the
joblib episode fan-out gives each worker process its own engine (CDaRR's
``_joblib_inited`` pattern, kept here as a module flag).
"""

from __future__ import annotations

import copy
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from blueskycdarr.aircraft import AircraftModel, AircraftSpec, as_pair
from blueskycdarr.config import ConflictConfig, SimulationConfig
from blueskycdarr.geo import distance_m
from blueskycdarr.scenario import PairGeometry, PairwiseEncounter
from blueskycdarr.state import StateArrays

M_TO_NM = 1.0 / 1852.0
MPS_TO_KTS = 1.0 / 0.514444

# CDaRR's spawn grid (envs/pairwise_params.json): pairs far enough apart that encounters
# never interact.
_GRID_LAT0 = 52.3
_GRID_LON0 = 4.7
_GRID_DELTA_DEG = 0.3
_ALT_M = 100.0

_initialised = False


@contextmanager
def _quiet() -> Iterator[None]:
    """BlueSky prints its banner on init; keep library imports silent (CDaRR's helper)."""
    with open(os.devnull, "w") as devnull:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def ensure_engine() -> None:
    """Initialise BlueSky once per process and verify it is the CDaRR fork."""
    global _initialised
    if _initialised:
        return
    import bluesky as bs

    with _quiet():
        bs.init(mode="sim", detached=True)
    if not hasattr(bs.traf, "max_tr") or not hasattr(bs.traf, "creconfs"):
        raise RuntimeError(
            "this BlueSky lacks the CDaRR fork's turn-rate limiter; install the engine "
            "with: pip install 'bluesky-simulator @ "
            "git+https://github.com/fazlurnu/bluesky.git@CDaRR'"
        )
    _initialised = True


# BlueSky's per-conflict bookkeeping on ``traf.cd`` lives in plain attributes, not in
# registered traffic arrays (it is sized by conflict pairs, not by aircraft), so the
# generic tree walk below misses it. Inert under this package's usage — the CDR chain is
# blueskycdarr's own and BlueSky's resolver is the do-nothing base class — but it is carried in
# the snapshot anyway so a restored world holds no stale record of another particle.
_CONFLICT_BOOKKEEPING = (
    "confpairs", "lospairs", "qdr", "dist", "dcpa", "tcpa", "tLOS",
    "confpairs_unique", "lospairs_unique", "confpairs_all", "lospairs_all",
)


def _entity_tree(node):  # bs.traf and its registered sub-entities, in registration order
    yield node
    for child in node._children:
        yield from _entity_tree(child)


def _copied(value):
    """A value copied deeply enough to be independent of the live world."""
    return value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)


def _require_empty_stack() -> None:
    """Refuse to snapshot or restore across pending stack commands.

    ``sim.step`` consumes the whole command queue at its start, so right after a step the
    queue is empty — that is the boundary this module snapshots at. A non-empty queue
    means the caller is mid-tick: snapshotting would silently drop the pending commands
    from the copy (they live in neither the traffic arrays nor the clock), and restoring
    would replay them into the restored timeline. Fail fast instead.
    """
    from bluesky.stack.stackbase import Stack

    if Stack.cmdstack:
        raise ValueError(
            f"world snapshot/restore only at a post-step boundary: {len(Stack.cmdstack)} "
            "stacked command(s) pending (sim.step consumes the stack at its start)"
        )


@dataclass(frozen=True)
class WorldSnapshot:
    """A complete copy of the process-global BlueSky world at a post-step boundary.

    The engine half of an IPS particle (the episode half is
    :class:`blueskycdarr.episode.EpisodeState`): :func:`restore_world` overwrites the live world
    with this value and stepping continues **bit-identically** — into the same world
    later, or into one that hosted a different particle of the same cell
    (``tests/test_snapshot_parity.py`` pins both).

    Why a generic walk is complete: BlueSky's own create/delete/reset machinery forces
    every per-aircraft variable through the ``TrafficArrays`` registry (an unregistered
    array would break resizing), so copying every registered array and list of every
    entity — ``nodes``, in registration order — captures all per-aircraft state by
    construction, including the fork's ``prev_turnrate`` turn-limiter memory. What lives
    outside the registry is carried explicitly: the aircraft count, ``traf.cd``'s
    conflict bookkeeping (:data:`_CONFLICT_BOOKKEEPING`), the ``simtime`` clock and its
    timers (they gate BlueSky's own timed updates), and ``sim.simt``. Out of scope, on
    purpose: plugin module state (none loaded in detached mode) and conditional stack
    commands (this package never issues any).

    Treat as immutable — it is what resampling *shares* between clones. Snapshot and
    restore both copy every array, so no live world ever aliases a snapshot. Plain data
    end to end, hence picklable: a particle can cross a process boundary to a worker
    that owns its own engine.
    """

    ntraf: int
    nodes: tuple[tuple[dict[str, np.ndarray], dict[str, list]], ...]
    conflict_lists: dict[str, object]
    clock: tuple  # simtime._clock: (t: Decimal, dt: Decimal, ft: float, fdt: float)
    timers: dict[str, tuple]  # per timer: (counter, readynext, tprev, rel_freq, dt_act, dt_req)
    simt: float
    simdt: float | None


def snapshot_world() -> WorldSnapshot:
    """Copy the process-global BlueSky world into a :class:`WorldSnapshot`."""
    ensure_engine()
    import bluesky as bs
    from bluesky.core import simtime

    _require_empty_stack()
    nodes = tuple(
        (
            {v: node.__dict__[v].copy() for v in node._ArrVars},
            {v: copy.deepcopy(node.__dict__[v]) for v in node._LstVars},
        )
        for node in _entity_tree(bs.traf)
    )
    cd = bs.traf.cd
    return WorldSnapshot(
        ntraf=bs.traf.ntraf,
        nodes=nodes,
        conflict_lists={
            k: _copied(getattr(cd, k)) for k in _CONFLICT_BOOKKEEPING if hasattr(cd, k)
        },
        clock=(simtime._clock.t, simtime._clock.dt, simtime._clock.ft, simtime._clock.fdt),
        timers={
            t.name: (t.counter, t.readynext, t.tprev, t.rel_freq, t.dt_act, t.dt_requested)
            for t in simtime.Timer.timers()
        },
        simt=bs.sim.simt,
        simdt=getattr(bs.sim, "simdt", None),
    )


def restore_world(snap: WorldSnapshot) -> None:
    """Overwrite the process-global BlueSky world with ``snap``.

    Pure array/list rebinding — BlueSky's own ``create`` rebinds arrays every spawn, so
    nothing in the engine may rely on array identity across steps. The entity tree must
    have the shape it had at snapshot time (same process, no plugins loaded in between);
    aircraft count may differ (a reset + respawned world restores fine).
    """
    ensure_engine()
    import bluesky as bs
    from bluesky.core import simtime

    _require_empty_stack()
    live = list(_entity_tree(bs.traf))
    if len(live) != len(snap.nodes):
        raise ValueError(
            f"the entity tree changed shape since the snapshot: {len(live)} live "
            f"entities vs {len(snap.nodes)} saved"
        )
    bs.traf.ntraf = snap.ntraf
    for node, (arrs, lsts) in zip(live, snap.nodes, strict=True):
        for v, a in arrs.items():
            node.__dict__[v] = a.copy()
        for v, lst in lsts.items():
            node.__dict__[v] = copy.deepcopy(lst)
    cd = bs.traf.cd
    for k, value in snap.conflict_lists.items():
        setattr(cd, k, _copied(value))
    (simtime._clock.t, simtime._clock.dt,
     simtime._clock.ft, simtime._clock.fdt) = snap.clock
    for timer in simtime.Timer.timers():
        saved = snap.timers.get(timer.name)
        if saved is not None:  # a timer born after the snapshot keeps its own state
            (timer.counter, timer.readynext, timer.tprev,
             timer.rel_freq, timer.dt_act, timer.dt_requested) = saved
    bs.sim.simt = snap.simt
    if snap.simdt is not None:
        bs.sim.simdt = snap.simdt


def discard_pending_commands() -> None:
    """Drop stacked-but-unflown commands — the ended-encounter cleanup.

    An encounter that ends exactly on a command-stacking CDR tick leaves commands its
    final tick stacked but never flew. They belong to no future — the engine's next use
    is a restore or a respawn that overwrites the world — yet they sit in the live
    global stack and would trip :func:`_require_empty_stack` on that next use (the
    first MC-vs-IPS comparison run hit exactly this). Discarding is the caller's
    explicit statement that a dead world's leftovers are meant to vanish; the guard
    stays strict for every path that did *not* say so.
    """
    ensure_engine()
    from bluesky.stack.stackbase import Stack

    Stack.cmdstack.clear()


def reset_world() -> None:
    """Clear the process-global traffic and any commands still addressed to it — what
    ``PairwiseWorld``'s context exit does.

    Also for callers that manage world lifecycles themselves: the IPS estimator spawns
    one world per particle *without* ``with`` (a new spawn replaces the previous world
    in place, and restores re-point the last spawn), so it calls this once when a
    replication ends rather than nesting a context per spawn. Commands are cleared with
    the traffic they addressed: leaving them would hand the next spawn a stale prefix
    that only works out because the spawn re-commands every aircraft afterwards — an
    accident of FIFO order nothing should rest on.
    """
    ensure_engine()
    import bluesky as bs

    bs.traf.reset()
    discard_pending_commands()


@dataclass
class PairwiseWorld:
    """One spawned batch of conflict pairs inside the process-global BlueSky.

    Aircraft ``2k`` is pair ``k``'s ownship (track 000), ``2k + 1`` its intruder, spawned
    in conflict by the fork's ``creconfs`` with the pair's crossing angle, miss distance
    and time to loss of separation. Use as a context manager so the global traffic state
    is reset even when an episode raises.
    """

    scenario: PairwiseEncounter
    geometry: PairGeometry
    aircraft: AircraftSpec
    conflict: ConflictConfig
    simulation: SimulationConfig

    def __post_init__(self) -> None:
        ensure_engine()
        import bluesky as bs
        from bluesky.tools.aero import vtas2cas

        self._bs = bs
        self._to_cas = lambda gs_ms: float(vtas2cas(gs_ms, _ALT_M))  # ground -> CAS at 100 m
        own_model, intr_model = as_pair(self.aircraft)
        bs.traf.reset()
        bs.settings.asas_pzr = self.conflict.rpz * M_TO_NM  # creconfs reads the zone radius
        bs.stack.stack(f"DT {self.simulation.dt}")

        n = self.scenario.n_pairs
        rows, cols = self.scenario.pairs
        self.nominal_trk = np.empty(2 * n)
        self.nominal_gs = np.empty(2 * n)

        for k in range(n):
            lat = _GRID_LAT0 + (k // cols) * _GRID_DELTA_DEG
            lon = _GRID_LON0 + (k % cols) * _GRID_DELTA_DEG
            own_id, intr_id = f"OWN{k:03d}", f"INT{k:03d}"
            bs.traf.cre(
                acid=own_id,
                actype=own_model.bs_actype,
                aclat=lat,
                aclon=lon,
                achdg=0.0,
                acalt=_ALT_M,
                acspd=self._to_cas(float(self.geometry.gs_own[k])),
            )
            bs.traf.creconfs(
                acid=intr_id,
                actype=intr_model.bs_actype,
                targetidx=bs.traf.id2idx(own_id),
                dpsi=float(self.geometry.dpsi[k]),
                dcpa=float(self.geometry.dcpa[k]) * M_TO_NM,
                tlosh=float(self.scenario.tlos),
                spd=self._to_cas(float(self.geometry.gs_intr[k])),
            )
            self.nominal_trk[2 * k] = 0.0
            self.nominal_trk[2 * k + 1] = self.geometry.dpsi[k] % 360.0
            self.nominal_gs[2 * k] = self.geometry.gs_own[k]
            self.nominal_gs[2 * k + 1] = self.geometry.gs_intr[k]

        self._apply_turn_policy(own_model, slice(0, None, 2))
        self._apply_turn_policy(intr_model, slice(1, None, 2))
        self._last_trk = np.full(2 * n, np.nan)
        self._last_gs = np.full(2 * n, np.nan)
        self.command(self.nominal_trk, self.nominal_gs)

    def _apply_turn_policy(self, model: AircraftModel, role: slice) -> None:
        """Write one role's turn authority into the fork's limiter arrays (ADR 0005)."""
        traf = self._bs.traf
        traf.max_tr[role] = model.max_turn_rate if model.max_turn_rate is not None else np.inf
        traf.max_dtr2[role] = (
            model.max_turn_accel if model.max_turn_accel is not None else np.inf
        )
        if model.bank_deg is not None:
            traf.ap.bankdef[role] = np.radians(model.bank_deg)

    @property
    def n_aircraft(self) -> int:
        return int(self._bs.traf.ntraf)

    def truth(self) -> StateArrays:
        """The ground-truth state table, copied out of the engine."""
        traf = self._bs.traf
        return StateArrays(
            lat=np.array(traf.lat, dtype=float),
            lon=np.array(traf.lon, dtype=float),
            trk=np.array(traf.trk, dtype=float),
            gs=np.array(traf.gs, dtype=float),
            gs_east=np.array(traf.gseast, dtype=float),
            gs_north=np.array(traf.gsnorth, dtype=float),
        )

    def command(self, trk: np.ndarray, gs: np.ndarray) -> None:
        """Stack HDG/SPD for every aircraft whose command changed (BlueSky holds them).

        ``gs`` is a ground speed; the SPD stack value is CAS (see the module docstring),
        so it is converted here — never by callers, and never fed back from a reading.
        """
        traf = self._bs.traf
        stack = self._bs.stack.stack
        changed = np.flatnonzero(
            ~(np.isclose(trk, self._last_trk) & np.isclose(gs, self._last_gs))
        )
        for i in changed:
            stack(f"HDG {traf.id[i]} {trk[i]:.4f}")
            stack(f"SPD {traf.id[i]} {self._to_cas(gs[i]) * MPS_TO_KTS:.4f}")
        self._last_trk[changed] = trk[changed]
        self._last_gs[changed] = gs[changed]

    def step(self) -> None:
        self._bs.sim.step()

    def snapshot(self) -> WorldSnapshot:
        """The engine half of a particle: the whole global world, copied (post-step)."""
        return snapshot_world()

    def restore(self, snap: WorldSnapshot) -> None:
        """Point this handle's world at ``snap``'s state (the IPS time-multiplex).

        The command cache is invalidated rather than restored: at every post-step
        boundary the cache equals the episode's commanded arrays, so the next
        ``command`` call re-stacks values identical to the restored autopilot state —
        a no-op on dynamics, pinned bit-for-bit by ``tests/test_snapshot_parity.py``.
        """
        if snap.ntraf != len(self._last_trk):
            raise ValueError(
                f"snapshot holds {snap.ntraf} aircraft but this world was spawned with "
                f"{len(self._last_trk)} — restore only within one cell's worlds"
            )
        restore_world(snap)
        self._last_trk.fill(np.nan)
        self._last_gs.fill(np.nan)

    def pair_distances(self) -> np.ndarray:
        """True ownship-intruder distance per pair, metres (length ``n_pairs``)."""
        traf = self._bs.traf
        own = slice(0, None, 2)
        intr = slice(1, None, 2)
        return distance_m(
            np.array(traf.lat[own]), np.array(traf.lon[own]),
            np.array(traf.lat[intr]), np.array(traf.lon[intr]),
        )

    def __enter__(self) -> PairwiseWorld:
        return self

    def __exit__(self, *exc: object) -> None:
        reset_world()
