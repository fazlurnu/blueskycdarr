"""Run a YAML-declared experiment: expand the sweep, run every condition, write the table.

    .venv/bin/python scripts/run_experiment.py configs/mixedvarlse.yaml --jobs -1

Outputs land under --out (default results/): a tidy CSV named after the run file, and a
provenance card beside it. Without a ``sweep:`` block the file runs as its single
condition — the ``run_one_experiment(*load_run(...))`` case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cdarr import load_run, run_experiment, sweep_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_file", type=Path, help="YAML run file (see configs/README.md)")
    parser.add_argument("--jobs", type=int, default=1, help="episode workers; -1 = all cores")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="override the file's seed")
    args = parser.parse_args()

    config, models, backend = load_run(args.run_file)
    axes = sweep_from_file(args.run_file)

    result = run_experiment(
        axes,
        models=models,
        backend=backend,
        base_config=config,
        seed=args.seed,
        n_jobs=args.jobs,
        card_dir=args.out / "cards",
    )

    csv_path = result.write_csv(args.out / f"{args.run_file.stem}.csv")
    print(f"\n{len(result)} condition(s) -> {csv_path}")
    print(f"provenance card -> {result.card_path}")


if __name__ == "__main__":
    main()
