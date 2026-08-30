# JRESS result figures — `results/jress2/*.csv`

Regenerate with:

    .venv/bin/python scripts/jress2/figures.py          # PNG (mathtext, no LaTeX needed)
    .venv/bin/python scripts/jress2/figures.py --pgf    # + .pgf for \input{} in the paper

Layout, colours and labels follow `06Results.tex`. The one change: every safety figure
carries **both** dependent variables side by side — IPR on the left (linear, the paper's
headline number) and P(LoS) = 1 − IPR on the right (log). The log panel replaces the
paper's "zoom above 0.99" column: same tail, four decades instead of one.

| figure | paper counterpart | contents |
| --- | --- | --- |
| `fig_crossing_angle_safety` | `fig_crossing_angle_vs_ipr` | Exp 1, 4 uncertainty levels × (IPR \| P(LoS)) |
| `fig_crossing_angle_dcpa_median` | `fig_crossing_angle_vs_dcpa_median` | Exp 1 efficiency, unchanged |
| `fig_gamma_comparison_safety` | `fig_gamma_comparison_ipr` | Exp 2, γ ladder × (IPR \| P(LoS)) |
| `fig_gamma_comparison_dcpa_median` | `fig_gamma_comparison_dcpa_median` | Exp 2 efficiency, unchanged |
| `fig_noise_model_sensitivity` | `fig_homogenous_noise_dist_sensitivity` | Exp 3, IPR / P(LoS) / median ‖d_CPA‖ bars |

## Reading the log panels

Exp 1 and Exp 2 run 10,000 encounters per cell, so the finest resolvable P(LoS) is
1e-4. Cells with **zero** observed LoS cannot sit on a log axis: they are drawn as open
downward triangles on the floor line and the curve is broken there, so "no LoS observed"
is never read as a value. The band below 1/N is shaded — anything in it is a single-count
observation, not a measurement.

Exp 3 runs 100,000 encounters per cell (floor 1e-5) and every cell has at least 4 LoS,
so its bars need no floor handling; each bar is annotated with its raw LoS count.

## Deviation from the paper

The paper's Exp 3 bottom panel is a box plot of the CPA distribution. `exp3.csv` keeps
only `median_min_sep`, so the median is drawn as a bar instead. Restoring the box plot
needs per-encounter CPA retained by the campaign.
