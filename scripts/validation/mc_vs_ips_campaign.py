"""MC-vs-IPS validation at P ~ 1e-4, over (pos_ci95, vel_ci95, crossing angle).

Every cell is validated **at the same target probability** (default 1e-4), not at the
same distance: the Monte-Carlo arm (default 1,000,000 encounters per cell) keeps each
cell's full min-sep distribution, the rare boundary ``d*`` is placed at that cell's own
empirical target-probability order statistic (~100 events at 1e-4 — a tight anchor),
and the IPS ladder descends to ``d*`` on shells placed at order statistics targeting
~0.45 conditional survival each — the recipe validated at the 1e-4 boundary in
``results/ips_mc_comparison/`` (ratio 1.18 there). The 50 m protected-zone radius is
kept as an intermediate rung, so the classic P(LoS) comparison rides along for free as
a secondary column.

Why the boundary moves and the physics does not: in this CDR stack the *graded* failure
family bottoms out around P(LoS) ~ 1e-3 — push the cell physics rarer and the failure
mechanism turns discrete (never-detected encounters, a bimodal min-sep, the *cliff* of
``docs/rare-events.md``) where fixed-level ladders collapse regardless of spacing. The
1e-4 event that is graded in every cell is *depth of breach*, so that is what this
campaign ladders to. For the same reason the default grid keeps the graded corner
(pos_ci95 25/40 m); strong-CDR cells (pos_ci95 ~ 10 m) are cliff-structured and would
need AMS, not a finer fixed ladder.

Judgment is on the ratio, ADR 0022 bands (2x at >=30 anchor events, widening as the
anchor thins), verdicts only when the IPS arm ran >= 4 replications.

    .venv/bin/python scripts/validation/mc_vs_ips_campaign.py               # dummy smoke
    .venv/bin/python scripts/validation/mc_vs_ips_campaign.py --production  # the server run

Dummy budgets smoke the *pipeline* (sweep, boundary placement, ladders, verdicts,
outputs) in a few minutes at a shallower target (2e-3); dummy verdicts are UNJUDGED by
design. Production on ~100 cores: MC ~25-35 min (16 cells x 1M at ~8-10k encounters/s),
IPS ~20-30 min — and raise --reps toward the core count (IPS parallelism is
min(--jobs, --reps); more replications is also statistically the right dial).

Outputs under results/validation/: a tidy CSV (one row per cell, both thresholds), a
Markdown summary with verdicts, and a JSONL with the ladder and every replication's
survival fractions.
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

# The graded corner: cells whose breach *depth* is continuously distributed, so a
# target-probability boundary is reachable by a fixed ladder (module docstring).
PRODUCTION_GRID = dict(
    pos_ci95=[25.0, 40.0], vel_ci95=[1.0, 3.0], dpsi=[45.0, 90.0, 135.0, 180.0]
)
DUMMY_GRID = dict(pos_ci95=[25.0, 40.0], vel_ci95=[3.0], dpsi=[90.0, 135.0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--production", action="store_true",
                   help="full grid and budgets (16 cells; the server run)")
    p.add_argument("--encounters", type=int, default=None,
                   help="MC encounters per cell (default: 20000 dummy, 1000000 production)")
    p.add_argument("--target-p", type=float, default=None, dest="target_p",
                   help="probability the rare boundary d* is placed at "
                        "(default: 2e-3 dummy, 1e-4 production)")
    p.add_argument("--particles", type=int, default=None,
                   help="IPS cloud size N (default: 48 dummy, 256 production)")
    p.add_argument("--reps", type=int, default=None,
                   help="IPS replications — also the IPS parallelism ceiling "
                        "(default: 2 dummy, 8 production; raise toward the core count)")
    p.add_argument("--jobs", type=int, default=-1, help="workers; -1 = all cores")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def place_boundary(min_sep_sorted: np.ndarray, target_p: float) -> float:
    """The rare boundary d*: the cell's own empirical ``target_p`` order statistic,
    floored to 0.1 m. At 1e-4 and one million encounters this is the 100th-smallest
    minimum separation — deep enough to be rare, populated enough to anchor."""
    idx = max(int(round(target_p * min_sep_sorted.size)), 8)
    return float(np.floor(min_sep_sorted[idx - 1] * 10.0) / 10.0)


def build_ladder(min_sep_sorted: np.ndarray, d_star: float, target: float = 0.45,
                 max_shells: int = 16) -> list[float]:
    """Levels from the cell's own depth distribution, ending at ``d_star``.

    Shells sit at order statistics spaced for ~``target`` conditional survival each, so
    the geometry adapts to every cell; the 50 m protected-zone radius is forced in as a
    rung (when it lies above ``d_star``), which makes the classic P(LoS) a prefix
    product every replication reports for free.
    """
    n = min_sep_sorted.size
    floor_count = max(int(np.sum(min_sep_sorted < d_star)), 6)
    fraction = floor_count / n
    shells: list[float] = []
    if fraction < target:
        m = int(np.clip(np.ceil(np.log(fraction) / np.log(target)), 1, max_shells))
        ratio = fraction ** (1.0 / m)
        for k in range(1, m):
            count = max(int(round(n * ratio**k)), floor_count)
            shell = float(np.floor(min_sep_sorted[count - 1] * 10.0) / 10.0)
            if shell <= d_star + 0.5:
                continue
            shells.append(shell)
    if d_star + 0.5 < RPZ:
        shells.append(RPZ)  # the classic LoS boundary rides along as a rung
    rungs: list[float] = []
    for shell in sorted(set(shells), reverse=True):
        if not rungs or shell <= rungs[-1] - 0.2:
            rungs.append(shell)
    if rungs and rungs[-1] < d_star + 0.2:
        rungs.pop()
    return rungs + [d_star]


def prefix_p(replications, ladder: list[float], threshold: float) -> float:
    """Mean over replications of the prefix product through ``threshold``'s rung — an
    unbiased estimate of P(min_sep <= threshold); NaN if the rung is absent."""
    if threshold not in ladder:
        return float("nan")
    k = ladder.index(threshold) + 1
    return float(np.mean([float(np.prod(r.survival[:k])) for r in replications]))


def verdict(ratio: float, n_events: int, reps: int) -> str:
    """The ADR 0022 judgment: factor two, widening as the anchor thins; judged only
    when both arms carry enough evidence (>= 1 anchor event, >= 4 replications)."""
    if n_events == 0:
        return "NO_ANCHOR"
    if reps < 4:
        return "UNJUDGED"
    band = 2.0 if n_events >= 30 else (3.0 if n_events >= 10 else 5.0)
    return "PASS" if 1.0 / band <= ratio <= band else "FAIL"


def main() -> None:
    args = parse_args()
    grid = PRODUCTION_GRID if args.production else DUMMY_GRID
    n_encounters = args.encounters or (1_000_000 if args.production else 20_000)
    target_p = args.target_p or (1e-4 if args.production else 2e-3)
    n_particles = args.particles or (256 if args.production else 48)
    reps = args.reps or (8 if args.production else 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_campaign = time.perf_counter()

    print(f"MC vs IPS validation at target p ~ {target_p:g} — {stamp}")
    print(f"grid {grid} | MC {n_encounters}/cell | IPS N={n_particles} reps={reps} "
          f"| seed {args.seed}\n")

    # --- MC arm: one declarative sweep, CRN across cells, full depth kept -------------
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

    # --- IPS arm per cell: boundary at the cell's target-p quantile, ladder to it ----
    rows: list[dict] = []
    detail_path = OUT_DIR / "mc_vs_ips_detail.jsonl"
    with detail_path.open("w") as detail:
        for i, (dpsi, pos, vel) in enumerate(
            (d, p, v)
            for d in grid["dpsi"] for p in grid["pos_ci95"] for v in grid["vel_ci95"]
        ):
            cell = mc.cell(dpsi=dpsi, pos_ci95=pos, vel_ci95=vel)
            ms = np.sort(cell.min_sep)
            d_star = place_boundary(ms, target_p)
            ladder = build_ladder(ms, d_star)
            n_star = int(np.sum(ms < d_star))
            p_mc_star = n_star / ms.size
            p_mc_rpz = float(np.mean(ms < RPZ))

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
            p_ips_star = est.p_los
            p_ips_rpz = prefix_p(est.reps, ladder, RPZ)
            ratio_star = p_ips_star / p_mc_star if p_mc_star > 0 else float("nan")
            ratio_rpz = p_ips_rpz / p_mc_rpz if p_mc_rpz > 0 else float("nan")
            row = dict(
                dpsi=dpsi, pos_ci95=pos, vel_ci95=vel,
                d_star=d_star, n_events_star=n_star, n_encounters=ms.size,
                p_mc_star=p_mc_star, p_ips_star=p_ips_star, ratio_star=ratio_star,
                verdict=verdict(ratio_star, n_star, reps),
                p_mc_rpz=p_mc_rpz, p_ips_rpz=p_ips_rpz, ratio_rpz=ratio_rpz,
                n_levels=len(ladder), n_collapsed=est.n_collapsed, reps=reps,
                ips_wall_s=round(t_ips, 1),
            )
            rows.append(row)
            detail.write(json.dumps(dict(
                **{k: row[k] for k in ("dpsi", "pos_ci95", "vel_ci95", "d_star")},
                levels=ladder,
                replications=[dict(survival=list(r.survival), collapsed_at=r.collapsed_at)
                              for r in est.reps],
            )) + "\n")
            print(f"[{i + 1}] dpsi={dpsi:g} pos={pos:g} vel={vel:g}: d*={d_star:g}m "
                  f"MC {p_mc_star:.3g} ({n_star}) vs IPS {p_ips_star:.3g} "
                  f"-> ratio {ratio_star:.2f} {row['verdict']} | at rpz: {ratio_rpz:.2f}"
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
        f.write(f"# MC vs IPS validation at target p ~ {target_p:g} — {stamp}\n\n")
        f.write(f"Seed {args.seed} · MC {n_encounters}/cell · IPS N={n_particles}, "
                f"{reps} reps · comm rx={COMM.reception_prob} lat={COMM.latency_s}s · "
                f"dcpa 0, tlos {TLOS:g}s, rpz {RPZ:g}m · d* at each cell's "
                f"{target_p:g} depth quantile\n\n")
        f.write(f"**{n_pass} PASS / {n_fail} FAIL / {n_anchorless} NO_ANCHOR / "
                f"{n_unjudged} UNJUDGED** of {len(rows)} cells · MC arm {t_mc:.0f}s · "
                f"total {total:.0f}s\n\n")
        f.write("| dpsi | pos | vel | d* [m] | P_MC(d*) | events | P_IPS(d*) | ratio "
                "| ratio@rpz | collapsed | verdict |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['dpsi']:g} | {r['pos_ci95']:g} | {r['vel_ci95']:g} "
                    f"| {r['d_star']:g} | {r['p_mc_star']:.3g} | {r['n_events_star']} "
                    f"| {r['p_ips_star']:.3g} | {r['ratio_star']:.2f} "
                    f"| {r['ratio_rpz']:.2f} | {r['n_collapsed']}/{r['reps']} "
                    f"| {r['verdict']} |\n")
        f.write("\nPrimary verdicts at d* (the target-probability boundary); the rpz "
                "ratio is the classic P(LoS) read off the same ladders as a prefix "
                "product. Bands (ADR 0022 style): 2x at >=30 anchor events, 3x for "
                "10-29, 5x for 1-9; NO_ANCHOR when MC saw none; UNJUDGED below 4 "
                "replications (smoke budgets).\n")

    print(f"\n{n_pass} PASS / {n_fail} FAIL / {n_anchorless} NO_ANCHOR / {n_unjudged} "
          f"UNJUDGED of {len(rows)} cells | total {total:.0f}s")
    print(f"-> {csv_path}\n-> {md_path}\n-> {detail_path}")


if __name__ == "__main__":
    main()
