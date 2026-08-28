"""MC-vs-IPS validation campaign over (pos_ci95, vel_ci95, crossing angle).

Per cell, both estimators price the same per-encounter P(LoS) — same config, same
encounter distribution, independent samplings — and the campaign judges the ratio
(ADR 0022's convention: a factor-of-two band, widening where the Monte-Carlo anchor
itself rests on few events; never intervals). The Monte-Carlo arm is one declarative
sweep (common random numbers across cells, ADR 0004); the IPS arm then reuses each
cell's MC min-sep distribution to place its level ladder at order statistics targeting
~0.45 conditional survival per shell — the recipe validated at the 1e-4 boundary
(vault/decisions/0008, results/ips_mc_comparison/).

Environment: the graded-degradation comm regime the estimator was validated in
(reception 0.8, latency 0.3 s), aircraft on a collision course (dcpa 0, tlos 20 s),
M600 both sides, rpz 50 m. Cells with strong CDR may ladder into a *cliff* (bimodal
min-sep — docs/rare-events.md); their collapses are recorded, not hidden.

    .venv/bin/python scripts/validation/mc_vs_ips_campaign.py               # dummy smoke
    .venv/bin/python scripts/validation/mc_vs_ips_campaign.py --production  # the server run

Dummy budgets smoke the *pipeline* (sweep, ladder construction, verdicts, outputs) in
about two minutes; at those budgets quantile shells are noisy and small clouds collapse
easily, so dummy verdicts are not evidence about the estimators. Judge agreement at
production budgets.

Outputs under results/validation/: a tidy CSV (one row per cell), a Markdown summary
with verdicts, and a JSONL with per-replication ladder detail. IPS parallelism is
min(--jobs, --reps) — replications fan out, particles within one replication are
serial — so on a big machine raise --reps toward the core count.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np

from blueskycdarr import (
    MC,
    MULTIROTOR,
    CommConfig,
    Config,
    Models,
    PairwiseEncounter,
    SimulationConfig,
    Sweep,
    UncertaintyConfig,
    run_experiment,
)
from blueskycdarr.ips import estimate_rare_prob

RPZ = 50.0
COMM = CommConfig(reception_prob=0.8, latency_s=0.3)
SIM = SimulationConfig(t_max=90.0)
BASE_CONFIG = Config(comm=COMM, simulation=SIM)
TLOS = 20.0
OUT_DIR = Path("results/validation")

# Grids: production covers the noise x geometry plane; the dummy grid is a smoke of the
# whole pipeline (MC sweep, ladder construction, IPS, verdicts) in about a minute.
PRODUCTION_GRID = dict(
    pos_ci95=[10.0, 25.0, 40.0], vel_ci95=[1.0, 3.0], dpsi=[45.0, 90.0, 135.0, 180.0]
)
DUMMY_GRID = dict(pos_ci95=[25.0, 40.0], vel_ci95=[3.0], dpsi=[90.0, 135.0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--production", action="store_true",
                   help="full grid and budgets (24 cells; the server run)")
    p.add_argument("--encounters", type=int, default=None,
                   help="MC encounters per cell (default: 3000 dummy, 20000 production)")
    p.add_argument("--particles", type=int, default=None,
                   help="IPS cloud size N (default: 48 dummy, 128 production)")
    p.add_argument("--reps", type=int, default=None,
                   help="IPS replications — also the IPS parallelism ceiling "
                        "(default: 2 dummy, 8 production)")
    p.add_argument("--jobs", type=int, default=-1, help="workers; -1 = all cores")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_ladder(min_sep_sorted: np.ndarray, target: float = 0.45,
                 max_shells: int = 8) -> list[float]:
    """Levels from a cell's own MC depth distribution, ending at the rpz boundary.

    Shells sit at order statistics spaced for ~``target`` conditional survival each —
    the geometry adapts itself to graded and cliff cells alike. The shell floor is the
    ~6-event quantile: below that the order statistic is too noisy to place furniture
    on. A cell with zero MC events at rpz still gets a ladder (the deepest resolvable
    shells, then rpz); its IPS may collapse at the boundary, which is the honest report.
    """
    n = min_sep_sorted.size
    floor_count = max(int(np.sum(min_sep_sorted < RPZ)), 6)
    fraction = floor_count / n
    if fraction >= target:  # common event: one shell (the boundary) is the whole ladder
        return [RPZ]
    m = int(np.clip(np.ceil(np.log(fraction) / np.log(target)), 1, max_shells))
    ratio = fraction ** (1.0 / m)
    shells: list[float] = []
    for k in range(1, m):
        count = max(int(round(n * ratio**k)), floor_count)
        shell = float(np.floor(min_sep_sorted[count - 1] * 10.0) / 10.0)
        if shell <= RPZ + 0.5 or (shells and shell >= shells[-1] - 0.2):
            continue
        shells.append(shell)
    return shells + [RPZ]


def verdict(ratio: float, n_events: int, reps: int) -> str:
    """The ADR 0022 judgment: factor two, widening as the anchor thins.

    Judged only when both arms carry enough evidence to judge: the MC anchor needs
    events, and the IPS estimate needs at least 4 replications — below that its own
    spread is unquantifiable, so a smoke run reports the ratio UNJUDGED rather than
    flapping between PASS and FAIL on replication noise.
    """
    if n_events == 0:
        return "NO_ANCHOR"
    if reps < 4:
        return "UNJUDGED"
    band = 2.0 if n_events >= 30 else (3.0 if n_events >= 10 else 5.0)
    return "PASS" if 1.0 / band <= ratio <= band else "FAIL"


def main() -> None:
    args = parse_args()
    grid = PRODUCTION_GRID if args.production else DUMMY_GRID
    n_encounters = args.encounters or (20_000 if args.production else 3_000)
    n_particles = args.particles or (128 if args.production else 48)
    reps = args.reps or (8 if args.production else 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_campaign = time.perf_counter()

    print(f"MC vs IPS validation campaign — {stamp}")
    print(f"grid {grid} | MC {n_encounters}/cell | IPS N={n_particles} reps={reps} "
          f"| seed {args.seed}\n")

    # --- MC arm: one declarative sweep, CRN across cells ------------------------------
    t0 = time.perf_counter()
    mc = run_experiment(
        {"dpsi": Sweep(grid["dpsi"]),
         "pos_ci95": Sweep(grid["pos_ci95"]),
         "vel_ci95": Sweep(grid["vel_ci95"])},
        models=Models(
            aircraft=MULTIROTOR,
            scenario=PairwiseEncounter(pairs=(5, 4), dcpa=0.0, tlos=TLOS),
        ),
        backend=MC(n_encounters=n_encounters),
        base_config=BASE_CONFIG,
        seed=args.seed,
        n_jobs=args.jobs,
        progress=True,
    )
    t_mc = time.perf_counter() - t0
    print(f"MC arm done in {t_mc:.0f}s\n")

    # --- IPS arm per cell, ladder from that cell's own MC quantiles -------------------
    rows: list[dict] = []
    detail_path = OUT_DIR / "mc_vs_ips_detail.jsonl"
    with detail_path.open("w") as detail:
        for i, (dpsi, pos, vel) in enumerate(
            (d, p, v)
            for d in grid["dpsi"] for p in grid["pos_ci95"] for v in grid["vel_ci95"]
        ):
            cell = mc.cell(dpsi=dpsi, pos_ci95=pos, vel_ci95=vel)
            p_mc = cell.n_los / cell.n_encounters
            ladder = build_ladder(np.sort(cell.min_sep))

            t0 = time.perf_counter()
            est = estimate_rare_prob(
                PairwiseEncounter(pairs=(1, 1), dpsi=dpsi, dcpa=0.0, tlos=TLOS),
                MULTIROTOR,
                Config(uncertainty=UncertaintyConfig(pos_ci95=pos, vel_ci95=vel),
                       comm=COMM, simulation=SIM),
                levels=ladder,
                n_particles=n_particles,
                reps=reps,
                seed=args.seed + 1_000 + i,
                n_jobs=args.jobs,
            )
            t_ips = time.perf_counter() - t0
            ratio = est.p_los / p_mc if p_mc > 0 else float("nan")
            row = dict(
                dpsi=dpsi, pos_ci95=pos, vel_ci95=vel,
                p_mc=p_mc, n_los=cell.n_los, n_encounters=cell.n_encounters,
                p_ips=est.p_los, ratio=ratio,
                verdict=verdict(ratio, cell.n_los, reps),
                n_levels=len(ladder), n_collapsed=est.n_collapsed, reps=reps,
                ips_wall_s=round(t_ips, 1),
            )
            rows.append(row)
            detail.write(json.dumps(dict(
                **{k: row[k] for k in ("dpsi", "pos_ci95", "vel_ci95")},
                levels=ladder,
                replications=[dict(survival=list(r.survival), collapsed_at=r.collapsed_at)
                              for r in est.reps],
            )) + "\n")
            print(f"[{i + 1}] dpsi={dpsi:g} pos={pos:g} vel={vel:g}: "
                  f"MC {p_mc:.3g} ({cell.n_los}) vs IPS {est.p_los:.3g} "
                  f"-> ratio {ratio:.2f} {row['verdict']}"
                  f"{' [' + str(est.n_collapsed) + ' collapsed]' if est.n_collapsed else ''}"
                  f"  ({t_ips:.0f}s, {len(ladder)} levels)", flush=True)

    # --- outputs ----------------------------------------------------------------------
    csv_path = OUT_DIR / "mc_vs_ips.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    n_fail = sum(r["verdict"] == "FAIL" for r in rows)
    n_anchorless = sum(r["verdict"] == "NO_ANCHOR" for r in rows)
    n_unjudged = sum(r["verdict"] == "UNJUDGED" for r in rows)
    total = time.perf_counter() - t_campaign
    md_path = OUT_DIR / "mc_vs_ips.md"
    with md_path.open("w") as f:
        f.write(f"# MC vs IPS validation — {stamp}\n\n")
        f.write(f"Seed {args.seed} · MC {n_encounters}/cell · IPS N={n_particles}, "
                f"{reps} reps · comm rx={COMM.reception_prob} lat={COMM.latency_s}s · "
                f"dcpa 0, tlos {TLOS:g}s, rpz {RPZ:g}m\n\n")
        f.write(f"**{n_pass} PASS / {n_fail} FAIL / {n_anchorless} NO_ANCHOR / "
                f"{n_unjudged} UNJUDGED** of {len(rows)} cells · MC arm {t_mc:.0f}s · "
                f"total {total:.0f}s\n\n")
        f.write("| dpsi | pos_ci95 | vel_ci95 | P(MC) | events | P(IPS) | ratio | "
                "collapsed | verdict |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['dpsi']:g} | {r['pos_ci95']:g} | {r['vel_ci95']:g} "
                    f"| {r['p_mc']:.3g} | {r['n_los']} | {r['p_ips']:.3g} "
                    f"| {r['ratio']:.2f} | {r['n_collapsed']}/{r['reps']} "
                    f"| {r['verdict']} |\n")
        f.write("\nVerdict bands (ADR 0022 style): ratio within 2x when the anchor has "
                ">=30 events, 3x for 10-29, 5x for 1-9; NO_ANCHOR when MC saw none; "
                "UNJUDGED when IPS ran fewer than 4 replications (smoke budgets).\n")

    print(f"\n{n_pass} PASS / {n_fail} FAIL / {n_anchorless} NO_ANCHOR / {n_unjudged} "
          f"UNJUDGED of {len(rows)} cells | total {total:.0f}s")
    print(f"-> {csv_path}\n-> {md_path}\n-> {detail_path}")


if __name__ == "__main__":
    main()
