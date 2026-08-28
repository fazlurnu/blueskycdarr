"""Rare-event comparison at the ~1e-4 order: MC with 100,000 encounters vs IPS.

Cell: pos_ci95 25 m, vel_ci95 3 m/s, reception 0.8, latency 0.3 s — resolution active
but sloppy, so breach *depth* is continuously graded. Rare boundary: min_sep < 15 m
(a near-collision for a 50 m protected zone). The IPS ladder passes through 50 and 25 m,
so prefix products give estimates at three thresholds from one run; the single MC run
anchors all three by counting its min_sep array at each threshold.

All wall-clock times recorded per arm and per chunk.
"""

import datetime
import time

import numpy as np

from cdarr import MC, Config, Fixed, MULTIROTOR, Models, PairwiseEncounter, run_experiment
from cdarr.config import CommConfig, SimulationConfig, UncertaintyConfig
from cdarr.ips import estimate_rare_prob

CONFIG = Config(
    uncertainty=UncertaintyConfig(pos_ci95=25.0, vel_ci95=3.0),
    comm=CommConfig(reception_prob=0.8, latency_s=0.3),
    simulation=SimulationConfig(t_max=90.0),
)
THRESHOLDS = (50.0, 25.0, 15.0)
LADDER = [95.0, 82.0, 71.0, 62.0, 55.0, 50.0, 44.0, 38.0, 32.0, 28.0, 25.0,
          21.0, 18.0, 15.0]
MC_CHUNKS = 10
MC_CHUNK_SIZE = 10_000

t_run = time.perf_counter()
print(f"rare-event MC vs IPS — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
print(f"cell: pos_ci95=25 vel_ci95=3 rx=0.8 lat=0.3 | dcpa=0 tlos=20 | rpz=50\n", flush=True)

# --- MC arm: 100,000 encounters in 10 chunks, 8 workers -------------------------------
t0 = time.perf_counter()
counts = {d: 0 for d in THRESHOLDS}
n_total = 0
for i in range(MC_CHUNKS):
    cell = run_experiment(
        {"pos_ci95": Fixed(25.0)},
        models=Models(
            aircraft=MULTIROTOR,
            scenario=PairwiseEncounter(pairs=(5, 4), dcpa=0.0, tlos=20.0),
        ),
        backend=MC(n_encounters=MC_CHUNK_SIZE),
        base_config=CONFIG,
        seed=100 + i,
        progress=False,
        n_jobs=8,
    ).cell()
    n_total += cell.n_encounters
    for d in THRESHOLDS:
        counts[d] += int(np.sum(cell.min_sep < d))
    print(f"MC chunk {i + 1:>2}/{MC_CHUNKS}: cumulative "
          + "  ".join(f"<{d:g}m: {counts[d]}/{n_total}" for d in THRESHOLDS)
          + f"   ({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
t_mc = time.perf_counter() - t0
p_mc = {d: counts[d] / n_total for d in THRESHOLDS}
print(f"MC arm done: {n_total} encounters in {t_mc:.0f}s "
      f"({n_total / t_mc:.0f} encounters/s)\n", flush=True)

# --- IPS arm: 14-level ladder to 15 m, N=256, 6 replications, 6 workers ---------------
t0 = time.perf_counter()
est = estimate_rare_prob(
    PairwiseEncounter(pairs=(1, 1), dcpa=0.0, tlos=20.0),
    MULTIROTOR,
    CONFIG,
    levels=LADDER,
    n_particles=256,
    reps=6,
    seed=7,
    n_jobs=6,
)
t_ips = time.perf_counter() - t0
print(est, flush=True)
print(f"IPS arm done in {t_ips:.0f}s "
      f"({6 * 256} spawned encounters + restored legs)\n", flush=True)

# prefix products: an unbiased estimate of P(min_sep <= d_k) at every ladder rung
p_ips = {}
for d in THRESHOLDS:
    k = LADDER.index(d) + 1
    per_rep = [float(np.prod(r.survival[:k])) if r.collapsed_at is None or r.collapsed_at >= k
               else 0.0
               for r in est.reps]
    p_ips[d] = float(np.mean(per_rep))

print("=" * 76)
print(f"{'threshold':>10} {'MC (100k)':>18} {'IPS':>12} {'ratio IPS/MC':>13}")
for d in THRESHOLDS:
    mc_str = f"{p_mc[d]:.3g} ({counts[d]})"
    ratio = p_ips[d] / p_mc[d] if p_mc[d] > 0 else float("inf")
    print(f"{d:>8.0f} m {mc_str:>18} {p_ips[d]:>12.3g} {ratio:>13.2f}")
print(f"\nwall: MC {t_mc:.0f}s (8 workers) | IPS {t_ips:.0f}s (6 workers) | "
      f"total {time.perf_counter() - t_run:.0f}s")
print(f"IPS collapsed replications: {est.n_collapsed}/{len(est.reps)}")
