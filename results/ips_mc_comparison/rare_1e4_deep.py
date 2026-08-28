"""The 1e-4 deliverable: MC 100k (full min-sep distribution kept) anchoring an IPS
ladder that descends to the empirical 1e-4 quantile of breach depth.

Same cell as the first comparison (pos_ci95 25, vel_ci95 3, rx 0.8, lat 0.3; dcpa 0,
tlos 20). The first run showed the severity tail is fat — p(<15 m) ~ 2e-3 — so the
1e-4 event in this cell is a *deep* near-collision. Here the MC arm (independent seeds
200–209) keeps every min_sep; the rare boundary d* is placed at the 10th-smallest of
the 100 000 (the empirical 1e-4 order statistic, rounded down to 0.1 m), the shells
between 15 m and d* are placed at order statistics chosen for ~0.45 conditional
survival each, and IPS then climbs that ladder with N=256, 6 replications.

All wall-clock times recorded.
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
UPPER_LADDER = [95.0, 82.0, 71.0, 62.0, 55.0, 50.0, 44.0, 38.0, 32.0, 28.0, 25.0,
                21.0, 18.0, 15.0]
MC_CHUNKS = 10
MC_CHUNK_SIZE = 10_000
OUT_NPY = ("/private/tmp/claude-2123284397/-Users-mfrahman-Projects-BlueSkyCDaRR/"
           "0c8e214b-4fba-4716-9d3c-cd6cc35ba3d2/scratchpad/min_sep_100k.npy")

t_run = time.perf_counter()
print(f"rare-event 1e-4 deep run — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

# --- MC arm: 100,000 encounters, full min_sep kept ------------------------------------
t0 = time.perf_counter()
all_ms = []
for i in range(MC_CHUNKS):
    cell = run_experiment(
        {"pos_ci95": Fixed(25.0)},
        models=Models(
            aircraft=MULTIROTOR,
            scenario=PairwiseEncounter(pairs=(5, 4), dcpa=0.0, tlos=20.0),
        ),
        backend=MC(n_encounters=MC_CHUNK_SIZE),
        base_config=CONFIG,
        seed=200 + i,
        progress=False,
        n_jobs=8,
    ).cell()
    all_ms.append(cell.min_sep)
    n = sum(m.size for m in all_ms)
    flat = np.concatenate(all_ms)
    print(f"MC chunk {i + 1:>2}/{MC_CHUNKS}: cumulative "
          f"<15m: {int(np.sum(flat < 15.0))}/{n}  <10m: {int(np.sum(flat < 10.0))}/{n}  "
          f"<5m: {int(np.sum(flat < 5.0))}/{n}   ({time.perf_counter() - t0:.0f}s)",
          flush=True)
min_sep = np.sort(np.concatenate(all_ms))
n_total = min_sep.size
t_mc = time.perf_counter() - t0
np.save(OUT_NPY, min_sep)
print(f"MC arm done: {n_total} encounters in {t_mc:.0f}s "
      f"({n_total / t_mc:.0f} encounters/s); min_sep saved to min_sep_100k.npy",
      flush=True)
print("smallest 12 min_sep [m]:", np.round(min_sep[:12], 2).tolist(), flush=True)

# --- place the rare boundary and the deep shells from the depth data ------------------
d_star = float(np.floor(min_sep[9] * 10.0) / 10.0)  # 10th smallest -> ~1e-4, tidied
count_15 = int(np.sum(min_sep < 15.0))
count_star = int(np.sum(min_sep < d_star))
# shells between 15 m and d*: order statistics at ~0.45 conditional steps
deep = []
c = count_15
while c * 0.45 > count_star + 2:
    c = int(round(c * 0.45))
    shell = float(np.floor(min_sep[c - 1] * 10.0) / 10.0)
    if (deep and shell >= deep[-1] - 0.2) or shell <= d_star + 0.2 or shell >= 14.8:
        continue
    deep.append(shell)
ladder = UPPER_LADDER + deep + [d_star]
print(f"\nrare boundary d* = {d_star} m ({count_star}/{n_total} below it)", flush=True)
print(f"deep shells from order statistics: {deep + [d_star]}", flush=True)

# --- IPS arm: down to d* ---------------------------------------------------------------
t0 = time.perf_counter()
est = estimate_rare_prob(
    PairwiseEncounter(pairs=(1, 1), dcpa=0.0, tlos=20.0),
    MULTIROTOR,
    CONFIG,
    levels=ladder,
    n_particles=256,
    reps=6,
    seed=9,
    n_jobs=6,
)
t_ips = time.perf_counter() - t0
print(est, flush=True)
print(f"IPS arm done in {t_ips:.0f}s ({6 * 256} spawned encounters + restored legs)\n",
      flush=True)

# --- three-plus-one-threshold table ----------------------------------------------------
thresholds = [50.0, 25.0, 15.0, d_star]
print("=" * 78)
print(f"{'threshold':>10} {'MC (100k)':>20} {'IPS':>12} {'ratio IPS/MC':>13}")
for d in thresholds:
    k = ladder.index(d) + 1
    per_rep = [float(np.prod(r.survival[:k])) for r in est.reps]
    p_ips = float(np.mean(per_rep))
    cnt = int(np.sum(min_sep < d))
    p_mc = cnt / n_total
    ratio = p_ips / p_mc if p_mc > 0 else float("inf")
    print(f"{d:>8.1f} m {f'{p_mc:.3g} ({cnt})':>20} {p_ips:>12.3g} {ratio:>13.2f}")
print(f"\nwall: MC {t_mc:.0f}s (8 workers) | IPS {t_ips:.0f}s (6 workers) | "
      f"total {time.perf_counter() - t_run:.0f}s")
print(f"IPS collapsed replications: {est.n_collapsed}/{len(est.reps)}")
print(f"finished {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
