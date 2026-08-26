"""Generate the validation figures for vault/correctness.md.

    .venv/bin/python scripts/correctness_figures.py            # ~2-3 min, all four
    .venv/bin/python scripts/correctness_figures.py --fast     # skip the response sweeps

Writes PNGs into vault/img/ and the response-sweep tables into results/. House plot
style: no grid, no figure title, concise subplot titles, elaboration in the caption
(which lives in correctness.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blueskycdarr import (
    MC,
    MULTIROTOR,
    CommConfig,
    Config,
    Fixed,
    Models,
    PairwiseEncounter,
    Sweep,
    UncertaintyConfig,
    run_experiment,
)
from blueskycdarr.adsl import BroadcastChannel, ContactTable, noisy_snapshot
from blueskycdarr.episode import run_episode
from blueskycdarr.geo import enu_offset
from blueskycdarr.rng import child, generator, root_seed_sequence, spawn
from blueskycdarr.state import StateArrays

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "vault" / "img"
RESULTS = ROOT / "results"

POS_LEVELS = [3.0, 10.0, 30.0, 92.6]
RPZ = 50.0


def _save(fig: plt.Figure, name: str) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    out = IMG / name
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# --- 1. resolution baseline ------------------------------------------------------------


def fig_resolution_baseline() -> None:
    """One pair, dpsi 90: resolved under perfect CNS vs blind (reception 0)."""
    scenario = PairwiseEncounter(pairs=(1, 1), tlos=60.0)
    seq = child(root_seed_sequence(0), 0)

    def fly(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ts, tracks, dists = [], [], []
        origin: dict[str, np.ndarray] = {}

        def record(t: float, truth: StateArrays, distances: np.ndarray) -> None:
            if not origin:  # one common origin (the ownship spawn), so geometry survives
                origin["lat"] = np.full_like(truth.lat, truth.lat[0])
                origin["lon"] = np.full_like(truth.lon, truth.lon[0])
            east, north = enu_offset(origin["lat"], origin["lon"], truth.lat, truth.lon)
            ts.append(t)
            tracks.append(np.stack([east, north], axis=1))
            dists.append(distances[0])

        run_episode(scenario, MULTIROTOR, config, seq, recorder=record)
        return np.array(ts), np.array(tracks), np.array(dists)

    t_res, xy_res, d_res = fly(Config())
    t_bli, xy_bli, d_bli = fly(Config(comm=CommConfig(reception_prob=0.0)))

    fig, (ax_map, ax_dist) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for i, (label, style) in enumerate((("ownship", "-"), ("intruder", "-"))):
        ax_map.plot(xy_res[:, i, 0], xy_res[:, i, 1], style, color=f"C{i}",
                    label=f"{label}, resolved")
        ax_map.plot(xy_bli[:, i, 0], xy_bli[:, i, 1], ":", color=f"C{i}", alpha=0.7,
                    label=f"{label}, blind")
    k = int(np.argmin(d_res))
    ax_map.plot(*xy_res[k].T, "k.", ms=4)
    ax_map.set_xlabel("east [m]")
    ax_map.set_ylabel("north [m]")
    ax_map.set_title("trajectories (displacement from spawn)")
    ax_map.axis("equal")
    ax_map.legend(fontsize=8, loc="upper left")

    ax_dist.plot(t_res, d_res, label=f"resolved (min {d_res.min():.0f} m)")
    ax_dist.plot(t_bli, d_bli, ":", label=f"blind (min {d_bli.min():.1f} m)")
    ax_dist.axhline(RPZ, color="k", lw=0.8, dashes=(4, 2))
    ax_dist.text(1.0, RPZ * 1.15, f"RPZ {RPZ:.0f} m", fontsize=8)
    ax_dist.set_xlabel("time [s]")
    ax_dist.set_ylabel("pair distance [m]")
    ax_dist.set_title("separation over time")
    ax_dist.legend(fontsize=8)

    _save(fig, "resolution-baseline.png")


# --- 2. noise calibration --------------------------------------------------------------


def fig_noise_calibration() -> None:
    """Empirical radial position error vs every configured CI95 level."""
    n = 40_000
    truth = StateArrays.from_track_speed(
        lat=np.full(n, 52.3), lon=np.full(n, 4.7), trk=np.zeros(n), gs=np.full(n, 15.0)
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, ci in enumerate(POS_LEVELS):
        rng = generator(child(root_seed_sequence(1), i))
        snap = noisy_snapshot(truth, np.arange(n), UncertaintyConfig(pos_ci95=ci), rng)
        east, north = enu_offset(truth.lat, truth.lon, snap.lat, snap.lon)
        radial = np.sort(np.hypot(east, north))
        p95 = radial[int(0.95 * n)]
        ax.plot(radial, np.linspace(0, 1, n), color=f"C{i}",
                label=f"CI95 {ci:g} m (measured {p95:.2f} m)")
        ax.axvline(ci, color=f"C{i}", lw=0.8, dashes=(4, 2))
    ax.axhline(0.95, color="k", lw=0.8, dashes=(1, 2))
    ax.set_xscale("log")
    ax.set_xlabel("radial position error [m]")
    ax.set_ylabel("empirical CDF")
    ax.set_title("measured error vs configured CI95 (dashed)")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "noise-calibration.png")


# --- 3. channel effects ----------------------------------------------------------------


def fig_channel_effects() -> None:
    """Jitter, latency, holdover and the surveillance gate, each against its definition."""
    comm = CommConfig(
        reception_prob=0.7, latency_s=0.4, broadcast_interval_s=1.0, broadcast_jitter_s=0.2
    )

    def channel_for(comm: CommConfig, truth: StateArrays, seed: int) -> tuple[
        BroadcastChannel, ContactTable
    ]:
        streams = spawn(root_seed_sequence(seed), 3)
        ch = BroadcastChannel(
            comm=comm,
            uncertainty=UncertaintyConfig(),
            rng_measurement=generator(streams[0]),
            rng_reception=generator(streams[1]),
            rng_schedule=generator(streams[2]),
        )
        ch.initialise(truth.n)
        return ch, ContactTable.empty(truth)

    # (a) contact age over time: sawtooth with a latency floor and holdover teeth.
    truth = StateArrays.from_track_speed(
        lat=np.array([52.3, 52.3]), lon=np.array([4.70, 4.705]),
        trk=np.zeros(2), gs=np.full(2, 15.0),
    )
    ch, contacts = channel_for(comm, truth, seed=2)
    times = np.arange(0.0, 30.0, 0.05)
    ages = np.full(times.size, np.nan)
    for i, t in enumerate(times):
        ch.transmit_due(float(t), truth)
        ch.deliver_due(float(t), contacts)
        if contacts.valid[0]:
            ages[i] = t - contacts.t_tx[0]

    # (b) realized inter-broadcast gaps.
    ch2, _ = channel_for(comm, truth, seed=3)
    gaps = []
    for _ in range(2000):
        t = float(ch2.next_tx[0])
        ch2.transmit_due(t, truth)
        gaps.append(float(ch2.next_tx[0]) - t)

    # (c) delivery rate vs pair distance under a 1000 m surveillance range.
    distances = np.linspace(100.0, 2000.0, 39)
    rate = []
    for j, d in enumerate(distances):
        pair = StateArrays.from_track_speed(
            lat=np.array([52.3, 52.3]),
            lon=np.array([4.7, 4.7 + d / (111_320.0 * np.cos(np.radians(52.3)))]),
            trk=np.zeros(2), gs=np.full(2, 15.0),
        )
        gate = CommConfig(reception_prob=0.7, max_range_m=1000.0)
        ch3, contacts3 = channel_for(gate, pair, seed=100 + j)
        got = 0
        for k in range(400):
            ch3.transmit_due(float(k), pair)
            ch3.deliver_due(float(k), contacts3)
            got += int(contacts3.t_tx[0] == float(k))
        rate.append(got / 400)

    fig, (ax_age, ax_gap, ax_range) = plt.subplots(1, 3, figsize=(12.5, 3.6))

    ax_age.plot(times, ages, lw=1.0)
    ax_age.axhline(comm.latency_s, color="k", lw=0.8, dashes=(4, 2))
    ax_age.text(0.3, comm.latency_s + 0.06, f"latency {comm.latency_s} s", fontsize=8)
    ax_age.set_xlabel("time [s]")
    ax_age.set_ylabel("contact age [s]")
    ax_age.set_title("staleness: latency floor + holdover")

    ax_gap.hist(gaps, bins=40, color="C0")
    for x in (comm.broadcast_interval_s - comm.broadcast_jitter_s,
              comm.broadcast_interval_s + comm.broadcast_jitter_s):
        ax_gap.axvline(x, color="k", lw=0.8, dashes=(4, 2))
    ax_gap.set_xlabel("inter-broadcast gap [s]")
    ax_gap.set_ylabel("count")
    ax_gap.set_title(f"jitter: gaps in {comm.broadcast_interval_s} ± "
                     f"{comm.broadcast_jitter_s} s")

    ax_range.plot(distances, rate, ".-")
    ax_range.axvline(1000.0, color="k", lw=0.8, dashes=(4, 2))
    ax_range.axhline(0.7, color="k", lw=0.8, dashes=(1, 2))
    ax_range.set_xlabel("pair distance [m]")
    ax_range.set_ylabel("delivery rate")
    ax_range.set_title("surveillance gate at 1000 m, p_rx 0.7")

    _save(fig, "channel-effects.png")


# --- 4. P(LoS) response ----------------------------------------------------------------


def fig_p_los_response(n_jobs: int) -> None:
    """The physics the study runs on: P(LoS) against the five design variables."""
    base = Config(
        comm=CommConfig(
            reception_prob=0.8, max_range_m=3000.0, latency_s=0.1,
            broadcast_interval_s=1.0, broadcast_jitter_s=0.1,
        )
    )
    models = Models(aircraft=MULTIROTOR, scenario=PairwiseEncounter())
    backend = MC(n_encounters=300)

    left = run_experiment(
        {
            "aircraft": Sweep(["multirotor", "fixedwing"]),
            "vel_ci95": Sweep([1.0, 3.0]),
            "pos_ci95": Sweep(POS_LEVELS),
        },
        models=models, backend=backend, base_config=base, seed=7, n_jobs=n_jobs,
    )
    left.write_csv(RESULTS / "correctness-uncertainty.csv")

    right = run_experiment(
        {
            "max_range_m": Sweep([500.0, 3000.0]),
            "reception_prob": Sweep([0.2, 0.4, 0.6, 0.8, 1.0]),
            "pos_ci95": Fixed(30.0),
            "vel_ci95": Fixed(3.0),
        },
        models=models, backend=backend, base_config=base, seed=7, n_jobs=n_jobs,
    )
    right.write_csv(RESULTS / "correctness-comm.csv")

    fig, (ax_u, ax_c) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    styles = {"multirotor": "-o", "fixedwing": "--s"}
    for aircraft in ("multirotor", "fixedwing"):
        for vel, color in ((1.0, "C0"), (3.0, "C3")):
            rows = [r for r in left.records()
                    if r["aircraft"] == aircraft and r["vel_ci95"] == vel]
            p = np.array([r["p_los_run"] for r in rows])
            ax_u.plot(POS_LEVELS, p, styles[aircraft], color=color, ms=4, lw=1.2,
                      label=f"{aircraft}, vel {vel:g} m/s")
    ax_u.set_xscale("log")
    ax_u.set_xticks(POS_LEVELS, [f"{v:g}" for v in POS_LEVELS])
    ax_u.set_xlabel("position CI95 [m]")
    ax_u.set_ylabel("P(LoS) per encounter")
    ax_u.set_title("uncertainty response (p_rx 0.8, range 3000 m)")
    ax_u.legend(fontsize=8)

    for rng_m, marker in ((500.0, "--s"), (3000.0, "-o")):
        rows = [r for r in right.records() if r["max_range_m"] == rng_m]
        probs = [r["reception_prob"] for r in rows]
        p = np.array([r["p_los_run"] for r in rows])
        ax_c.plot(probs, p, marker, ms=4, lw=1.2, label=f"range {rng_m:g} m")
    ax_c.set_xlabel("reception probability")
    ax_c.set_title("communication response (pos 30 m, vel 3 m/s)")
    ax_c.legend(fontsize=8)

    _save(fig, "p-los-response.png")


# --- 5. recovery comparison ------------------------------------------------------------


def fig_recovery_comparison(n_jobs: int) -> None:
    """The three recovery models against uncertainty, the gamma knob, and CDaRR itself."""
    from blueskycdarr import FTR, PastCPA, ProbabilisticFTR, SimulationConfig

    stressed = Config(
        uncertainty=UncertaintyConfig(vel_ci95=3.0),
        comm=CommConfig(
            reception_prob=0.8, max_range_m=3000.0, latency_s=0.1,
            broadcast_interval_s=1.0, broadcast_jitter_s=0.1,
        ),
    )
    backend = MC(n_encounters=300)
    recoveries = [("pastcpa", PastCPA()), ("ftr", FTR()),
                  ("probabilistic_ftr", ProbabilisticFTR(gamma=0.999))]

    # (a) P(LoS) vs position uncertainty per recovery.
    per_recovery = {}
    for label, rec in recoveries:
        res = run_experiment(
            {"pos_ci95": Sweep(POS_LEVELS), "vel_ci95": Fixed(3.0)},
            models=Models(MULTIROTOR, PairwiseEncounter(), rec),
            backend=backend, base_config=stressed, seed=7, n_jobs=n_jobs,
        )
        res.write_csv(RESULTS / f"correctness-recovery-{label}.csv")
        per_recovery[label] = res.records()

    # (b) the gamma knob at the hardest uncertainty level, FTR/PastCPA as references.
    gammas = [0.5, 0.9, 0.99, 0.999]
    gamma_res = run_experiment(
        {
            "recovery": Sweep(gammas, name="gamma",
                              build=lambda g: ProbabilisticFTR(gamma=g)),
            "pos_ci95": Fixed(92.6),
            "vel_ci95": Fixed(3.0),
        },
        models=Models(MULTIROTOR, PairwiseEncounter(), PastCPA()),
        backend=backend, base_config=stressed, seed=7, n_jobs=n_jobs,
    )
    gamma_res.write_csv(RESULTS / "correctness-recovery-gamma.csv")

    # (c) the matched-CDaRR cross-validation condition (ADR 0006's table).
    matched = Config(
        uncertainty=UncertaintyConfig(10.0, 1.0),
        simulation=SimulationConfig(dt=0.2, cdr_dt=1.0, t_max=600.0, done_timeout=10.0),
    )
    cdarr_reference = {"pastcpa": 0.0004, "ftr": 0.0211, "probabilistic_ftr": 0.0005}
    print("\nmatched-CDaRR condition (dpsi 90, tlos 180 s, pos 10 m / vel 1 m/s):")
    rows = []
    for label, rec in recoveries:
        res = run_experiment(
            {"pos_ci95": Fixed(10.0)},
            models=Models(
                MULTIROTOR,
                PairwiseEncounter(speed=10.2889, dpsi=90.0, dcpa=0.0, tlos=180.0),
                rec,
            ),
            backend=backend, base_config=matched, seed=7, n_jobs=n_jobs, progress=False,
        )
        e = res.cell()
        rows.append((label, e))
        print(f"  {label:18s} here={e.p_los_run:.4f} ({e.n_los}/{e.n_encounters})"
              f"  CDaRR exp1={cdarr_reference[label]:.4f}")
    with (RESULTS / "correctness-recovery-cdarr-match.csv").open("w") as f:
        f.write("recovery,p_los_run,n_los,n_encounters,median_min_sep,cdarr_exp1_p_los\n")
        for label, e in rows:
            f.write(f"{label},{e.p_los_run:.10g},{e.n_los},{e.n_encounters},"
                    f"{e.median_min_sep:.10g},{cdarr_reference[label]}\n")

    fig, (ax_u, ax_g) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    styles = {"pastcpa": ("-o", "C0", "past-CPA"),
              "ftr": ("--s", "C3", "FTR"),
              "probabilistic_ftr": ("-.^", "C2", "prob. FTR (γ 0.999)")}
    for label, (fmt, color, name) in styles.items():
        rows_r = per_recovery[label]
        p = np.array([r["p_los_run"] for r in rows_r])
        ax_u.plot(POS_LEVELS, p, fmt, color=color, ms=4, lw=1.2, label=name)
    ax_u.set_xscale("log")
    ax_u.set_xticks(POS_LEVELS, [f"{v:g}" for v in POS_LEVELS])
    ax_u.set_xlabel("position CI95 [m]")
    ax_u.set_ylabel("P(LoS) per encounter")
    ax_u.set_title("recovery vs uncertainty (vel 3 m/s, p_rx 0.8)")
    ax_u.legend(fontsize=8, loc="upper left")

    rows_g = gamma_res.records()
    p = np.array([r["p_los_run"] for r in rows_g])
    ax_g.plot(gammas, p, "-.^", color="C2", ms=4, lw=1.2, label="prob. FTR")
    for label, color, name in (("ftr", "C3", "FTR"), ("pastcpa", "C0", "past-CPA")):
        ref = per_recovery[label][-1]  # pos_ci95 = 92.6 row
        ax_g.axhline(ref["p_los_run"], color=color, lw=0.9, dashes=(4, 2), label=name)
    ax_g.set_xscale("logit")
    ax_g.set_xticks(gammas, [f"{g:g}" for g in gammas])
    ax_g.minorticks_off()
    ax_g.set_xlabel("confidence threshold γ")
    ax_g.set_title("the γ knob at pos 92.6 m, vel 3 m/s")
    ax_g.legend(fontsize=8, loc="upper right")

    _save(fig, "recovery-comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip the response sweeps")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    fig_resolution_baseline()
    fig_noise_calibration()
    fig_channel_effects()
    if not args.fast:
        fig_p_los_response(args.jobs)
        fig_recovery_comparison(args.jobs)


if __name__ == "__main__":
    main()
