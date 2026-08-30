"""The JRESS paper's result figures, redrawn from ``results/jress2/*.csv``.

Same layout, colours and labels as the paper (06Results.tex), with one change the
paper's figures could not make: every safety figure now carries **both** dependent
variables side by side --- IPR on the left (linear, the paper's headline number) and
P(LoS) = 1 - IPR on the right (log, where the near-perfect methods separate). The log
panel replaces the paper's "zoom above 0.99" column: it shows the same tail, but over
four decades instead of one.

    .venv/bin/python scripts/jress2/figures.py            # PNG into results/jress2/figures/
    .venv/bin/python scripts/jress2/figures.py --pgf      # + .pgf for \\input{} in the paper

Cells with zero observed LoS cannot sit on a log axis. They are drawn as open
downward triangles on the floor line at 1/N (one LoS in 10^4 encounters) and the
P(LoS) line is broken there, so "no LoS observed" is never confused with a value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "jress2"
OUT = DATA / "figures"

# --- the paper's style (analysis/results_analysis_fulldpsi.py) ---------------------------

RPZ = 50.0
CI_LEVELS = [3.0, 10.0]
CIV_LEVELS = [1.0, 3.0]
UNCERTAINTY = [(ci, civ) for ci in CI_LEVELS for civ in CIV_LEVELS]
GAMMAS = ["0.999", "0.99", "0.9", "0.75", "0.5"]
DEFAULT_GAMMA = "0.999"

COLORS = {"pastcpa": "#1f77b4", "ftr": "#2ca02c", "probabilistic_ftr": "#ff7f0e"}
LABELS = {"pastcpa": "Past-CPA", "ftr": "FTR", "probabilistic_ftr": "Probabilistic FTR"}
METHODS = ["pastcpa", "ftr", "probabilistic_ftr"]

# light -> dark red for increasing gamma
GAMMA_COLORS = {
    "0.5": "#ffb3b3",
    "0.75": "#ff7f7f",
    "0.9": "#e63946",
    "0.99": "#c1121f",
    "0.999": "#780000",
}

NOISE_LABELS = {
    "gaussian": "Gaussian",
    "mixture": "Heavy-tail",
    "anisotropic": "Anisotropic",
    "latency": "Latency",
    "anisotropic_latency": "Aniso.\n+ latency",
    "anisotropic_mixture": "Aniso.\n+ heavy-tail",
}

USETEX = False


def style(pgf: bool) -> None:
    """The paper's rcParams; usetex only when writing .pgf (mathtext otherwise)."""
    global USETEX
    USETEX = pgf
    preamble = "\n".join([
        r"\usepackage{amsmath}",
        r"\newcommand{\norm}[1]{\lVert #1 \rVert}",
        r"\newcommand{\dCPA}{\mathbf{d}_{\mathrm{CPA}}}",
    ])
    matplotlib.rcParams.update({
        "text.usetex": pgf,
        "font.family": "serif",
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble": preamble,
        "pgf.preamble": preamble,
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })


def dcpa_label() -> str:
    return r"Median $\norm{\dCPA}$ [m]" if USETEX else r"Median $\|\mathbf{d}_{\mathrm{CPA}}\|$ [m]"


def subtitle(ci: float, civ: float) -> str:
    return (f"$CI_{{\\mathrm{{pos}}}}$ = {ci:g} m,  "
            f"$CI_{{\\mathrm{{vel}}}}$ = {civ:g} m/s")


def save(fig: plt.Figure, name: str, pgf: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {(OUT / f'{name}.png').relative_to(ROOT)}")
    if pgf:
        fig.savefig(OUT / f"{name}.pgf")
        print(f"wrote {(OUT / f'{name}.pgf').relative_to(ROOT)}")
    plt.close(fig)


# --- series extraction -------------------------------------------------------------------


def cell(df: pd.DataFrame, ci: float, civ: float, recovery: str) -> pd.DataFrame:
    """One (uncertainty, recovery) curve over the crossing-angle axis."""
    sub = df[(df.pos_ci95 == ci) & (df.vel_ci95 == civ) & (df.recovery == recovery)]
    return sub.sort_values("dpsi")


def plos_series(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Angles, P(LoS) with zeros masked to NaN, and the angles where n_los == 0."""
    angles = sub.dpsi.to_numpy(float)
    p = sub.p_los_run.to_numpy(float).copy()
    zeros = angles[sub.n_los.to_numpy() == 0]
    p[p == 0.0] = np.nan
    return angles, p, zeros


def floor_of(df: pd.DataFrame) -> float:
    """1/N: one LoS in the per-cell encounter budget."""
    return 1.0 / float(df.n_encounters.iloc[0])


def draw_zeros(ax, zeros: np.ndarray, color: str, floor: float) -> None:
    if len(zeros):
        ax.plot(zeros, np.full(len(zeros), floor), linestyle="none", marker="v",
                markersize=5, markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.2, alpha=0.9)


def finish_plos_axis(ax, floor: float) -> None:
    """Log P(LoS), with the counting-limited band below 1/N shaded out."""
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.55, 1.5)
    ax.axhspan(floor * 0.55, floor, color="0.5", alpha=0.13, linewidth=0, zorder=0)
    ax.axhline(floor, color="0.45", linestyle=":", linewidth=1.2, zorder=1)
    ax.set_ylabel("P(LoS)")


def floor_handles(n: int) -> list:
    """Legend entries explaining the resolution floor, shared by exp1 and exp2."""
    exp = int(round(np.log10(n)))
    return [
        plt.Line2D([], [], linestyle="none", marker="v", markersize=5,
                   markerfacecolor="none", markeredgecolor="0.3",
                   label=f"0 LoS in {n:,}"),
        plt.Line2D([], [], color="0.45", linestyle=":", linewidth=1.2,
                   label=f"floor: 1 LoS in $10^{{{exp}}}$"),
    ]


# --- Experiment 1: recovery method vs crossing angle -------------------------------------


def fig_crossing_angle_safety(e1: pd.DataFrame, pgf: bool) -> None:
    """Paper Fig. 'crossing_angle_vs_ipr', with the zoom column replaced by log P(LoS)."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    floor, n = floor_of(e1), int(e1.n_encounters.iloc[0])
    handles = []

    for row, (ci, civ) in enumerate(UNCERTAINTY):
        ax_ipr, ax_plos = axes[row, 0], axes[row, 1]

        for method in METHODS:
            sub = cell(e1, ci, civ, method)
            label = LABELS[method]
            if method == "probabilistic_ftr":
                label += f" ($\\gamma$={DEFAULT_GAMMA})"
            line, = ax_ipr.plot(sub.dpsi, 1.0 - sub.p_los_run, color=COLORS[method],
                                linewidth=1.5, alpha=0.85, label=label)
            if row == 0:
                handles.append(line)

            angles, p, zeros = plos_series(sub)
            ax_plos.plot(angles, p, color=COLORS[method], linewidth=1.5, alpha=0.85)
            draw_zeros(ax_plos, zeros, COLORS[method], floor)

        ax_ipr.set_ylim(0, 1.05)
        ax_ipr.set_ylabel("IPR")
        ax_ipr.set_title(subtitle(ci, civ))
        finish_plos_axis(ax_plos, floor)
        ax_plos.set_title(subtitle(ci, civ) + "  (log scale)")

        for ax in (ax_ipr, ax_plos):
            ax.set_xlabel("Crossing Angle [deg]")
            ax.set_xlim(0, 182)
            ax.grid(True, alpha=0.3)

    fig.legend(handles=handles + floor_handles(n), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig_crossing_angle_safety", pgf)


def fig_crossing_angle_dcpa(e1: pd.DataFrame, pgf: bool) -> None:
    """Paper Fig. 'crossing_angle_vs_dcpa_median' -- the efficiency counterpart."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    lo = e1.median_min_sep.min() * 0.95
    hi = e1.median_min_sep.max() * 1.05
    handles = []

    for idx, (ci, civ) in enumerate(UNCERTAINTY):
        ax = axes.flat[idx]
        for method in METHODS:
            sub = cell(e1, ci, civ, method)
            label = LABELS[method]
            if method == "probabilistic_ftr":
                label += f" ($\\gamma$={DEFAULT_GAMMA})"
            line, = ax.plot(sub.dpsi, sub.median_min_sep, color=COLORS[method],
                            linewidth=1.5, alpha=0.85, label=label)
            if idx == 0:
                handles.append(line)

        rpz = ax.axhline(RPZ, color="#d62728", linestyle="--", linewidth=1.5,
                         label=f"$R_{{\\mathrm{{PZ}}}}$ = {int(RPZ)} m")
        if idx == 0:
            handles.append(rpz)

        ax.set_ylim(lo, hi)
        ax.set_title(subtitle(ci, civ))
        ax.set_xlabel("Crossing Angle [deg]")
        ax.set_ylabel(dcpa_label())
        ax.set_xlim(0, 182)
        ax.grid(True, alpha=0.3)

    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "fig_crossing_angle_dcpa_median", pgf)


# --- Experiment 2: the confidence threshold gamma ----------------------------------------


def fig_gamma_safety(e2: pd.DataFrame, pgf: bool) -> None:
    """Paper Fig. 'gamma_comparison_ipr', with log P(LoS) beside it."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    floor, n = floor_of(e2), int(e2.n_encounters.iloc[0])
    handles = []

    for row, (ci, civ) in enumerate(UNCERTAINTY):
        ax_ipr, ax_plos = axes[row, 0], axes[row, 1]

        sub = cell(e2, ci, civ, "ftr")
        line, = ax_ipr.plot(sub.dpsi, 1.0 - sub.p_los_run, color=COLORS["ftr"],
                            linestyle="-", marker="*", markersize=6, linewidth=1.5,
                            alpha=0.8, label="FTR (deterministic)")
        angles, p, zeros = plos_series(sub)
        ax_plos.plot(angles, p, color=COLORS["ftr"], linestyle="-", marker="*",
                     markersize=6, linewidth=1.5, alpha=0.8)
        draw_zeros(ax_plos, zeros, COLORS["ftr"], floor)
        if row == 0:
            handles.append(line)

        for gamma in reversed(GAMMAS):  # legend reads 0.5 -> 0.999
            sub = cell(e2, ci, civ, gamma)
            line, = ax_ipr.plot(sub.dpsi, 1.0 - sub.p_los_run, color=GAMMA_COLORS[gamma],
                                linewidth=1.5, alpha=0.85, label=f"$\\gamma$ = {gamma}")
            angles, p, zeros = plos_series(sub)
            ax_plos.plot(angles, p, color=GAMMA_COLORS[gamma], linewidth=1.5, alpha=0.85)
            draw_zeros(ax_plos, zeros, GAMMA_COLORS[gamma], floor)
            if row == 0:
                handles.append(line)

        ax_ipr.set_ylim(0, 1.05)
        ax_ipr.set_ylabel("IPR")
        ax_ipr.set_title(subtitle(ci, civ))
        finish_plos_axis(ax_plos, floor)
        ax_plos.set_title(subtitle(ci, civ) + "  (log scale)")

        for ax in (ax_ipr, ax_plos):
            ax.set_xlabel("Crossing Angle [deg]")
            ax.set_xlim(0, 182)
            ax.grid(True, alpha=0.3)

    fig.legend(handles=handles + floor_handles(n), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "fig_gamma_comparison_safety", pgf)


def fig_gamma_dcpa(e2: pd.DataFrame, pgf: bool) -> None:
    """Paper Fig. 'gamma_comparison_dcpa_median'."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    lo = e2.median_min_sep.min() * 0.95
    hi = e2.median_min_sep.max() * 1.05
    handles = []

    for idx, (ci, civ) in enumerate(UNCERTAINTY):
        ax = axes.flat[idx]

        sub = cell(e2, ci, civ, "ftr")
        line, = ax.plot(sub.dpsi, sub.median_min_sep, color=COLORS["ftr"], linestyle="-",
                        marker="*", markersize=6, linewidth=1.5, alpha=0.8,
                        label="FTR (deterministic)")
        if idx == 0:
            handles.append(line)

        for gamma in reversed(GAMMAS):
            sub = cell(e2, ci, civ, gamma)
            line, = ax.plot(sub.dpsi, sub.median_min_sep, color=GAMMA_COLORS[gamma],
                            linewidth=1.5, alpha=0.85, label=f"$\\gamma$ = {gamma}")
            if idx == 0:
                handles.append(line)

        rpz = ax.axhline(RPZ, color="#d62728", linestyle="--", linewidth=1.5,
                         label=f"$R_{{\\mathrm{{PZ}}}}$ = {int(RPZ)} m")
        if idx == 0:
            handles.append(rpz)

        ax.set_ylim(lo, hi)
        ax.set_title(subtitle(ci, civ))
        ax.set_xlabel("Crossing Angle [deg]")
        ax.set_ylabel(dcpa_label())
        ax.set_xlim(0, 182)
        ax.grid(True, alpha=0.3)

    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "fig_gamma_comparison_dcpa_median", pgf)


# --- Experiment 3: robustness to the six navigation-noise models -------------------------


def fig_noise_models(e3: pd.DataFrame, pgf: bool) -> None:
    """Paper Fig. 'noise_dist_sensitivity': aggregate IPR, P(LoS) and median dCPA.

    The paper's bottom panel is a box plot of the CPA distribution; the campaign CSV
    keeps only the median, so this draws the median as a bar instead.
    """
    models = list(NOISE_LABELS)
    n = int(e3.n_encounters.iloc[0])
    floor = 1.0 / n
    x = np.arange(len(models), dtype=float)
    width = 0.26

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax_ipr, ax_plos, ax_dcpa = axes
    handles = []

    for k, method in enumerate(METHODS):
        offset = (k - 1) * width
        rows = e3[e3.recovery == method].set_index("noise_model").loc[models]
        label = LABELS[method]
        if method == "probabilistic_ftr":
            label += f" ($\\gamma$={DEFAULT_GAMMA})"

        ipr = 1.0 - rows.p_los_run.to_numpy()
        bars = ax_ipr.bar(x + offset, ipr, width, color=COLORS[method], alpha=0.85,
                          label=label)
        handles.append(bars)
        for xi, v in zip(x + offset, ipr):
            ax_ipr.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

        p = rows.p_los_run.to_numpy()
        ax_plos.bar(x + offset, np.where(p > 0, p, floor), width, color=COLORS[method],
                    alpha=0.85)
        for xi, v, c in zip(x + offset, p, rows.n_los.to_numpy()):
            ax_plos.text(xi, max(v, floor), f"{c:,} LoS", ha="center", va="bottom",
                         fontsize=9, rotation=90)

        med = rows.median_min_sep.to_numpy()
        ax_dcpa.bar(x + offset, med, width, color=COLORS[method], alpha=0.85)
        for xi, v in zip(x + offset, med):
            ax_dcpa.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax_ipr.set_ylim(0.90, 1.004)
    ax_ipr.set_ylabel("IPR")
    ax_ipr.set_title("bar labels: IPR (top), LoS count out of "
                     f"{n:,} (middle), median [m] (bottom)", fontsize=12, color="0.3")

    ax_plos.set_yscale("log")
    ax_plos.set_ylim(floor * 0.5, 2.0)  # headroom for the rotated count labels
    exp = int(round(np.log10(n)))
    floor_line = ax_plos.axhline(floor, color="0.45", linestyle=":", linewidth=1.2,
                                 label=f"floor: 1 LoS in $10^{{{exp}}}$")
    ax_plos.set_ylabel("P(LoS)")

    rpz = ax_dcpa.axhline(RPZ, color="#d62728", linestyle="--", linewidth=1.5,
                          label=f"$R_{{\\mathrm{{PZ}}}}$ = {int(RPZ)} m")
    ax_dcpa.set_ylabel(dcpa_label())
    ax_dcpa.set_ylim(0, e3.median_min_sep.max() * 1.18)

    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
    ax_dcpa.set_xticks(x)
    ax_dcpa.set_xticklabels([NOISE_LABELS[m] for m in models])
    ax_dcpa.set_xlabel("Navigation-noise model")

    fig.legend(handles=handles + [rpz, floor_line], loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig_noise_model_sensitivity", pgf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgf", action="store_true",
                    help="also write .pgf with the paper's usetex preamble")
    args = ap.parse_args()

    style(args.pgf)
    e1 = pd.read_csv(DATA / "exp1.csv")
    e2 = pd.read_csv(DATA / "exp2.csv", dtype={"recovery": str})
    e3 = pd.read_csv(DATA / "exp3.csv")

    fig_crossing_angle_safety(e1, args.pgf)
    fig_crossing_angle_dcpa(e1, args.pgf)
    fig_gamma_safety(e2, args.pgf)
    fig_gamma_dcpa(e2, args.pgf)
    fig_noise_models(e3, args.pgf)


if __name__ == "__main__":
    main()
