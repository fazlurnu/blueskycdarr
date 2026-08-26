"""The provenance card — one Markdown file per run: config + seed -> result (ADR 0003).

Cards are the audit trail, not the data (OpenCDaRR's rule): raw outputs are gitignored
and regenerable, the card records exactly what produced them. Written only when the
caller passes ``card_dir``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from cdarr.config import Config
    from cdarr.experiment import ExperimentResult, Models


def write_card(
    result: ExperimentResult, config: Config, models: Models, card_dir: Path
) -> Path:
    """Write ``<card_dir>/<stamp>_seed<seed>.md`` and return its path."""
    card_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = card_dir / f"{stamp}_seed{result.seed}.md"

    lines = [
        f"# Experiment {stamp}",
        "",
        f"- backend: `MC(n_encounters={result.backend.n_encounters})`",
        f"- seed: {result.seed}",
        f"- conditions: {len(result.conditions)}",
        f"- swept axes: {list(result.axes)}",
        "",
        "## Declaration",
        "",
    ]
    for condition in result.conditions[:1]:  # the vocabulary; levels vary per row below
        for key, value in condition.values:
            role = "swept" if key in dict(condition.levels) else "fixed"
            lines.append(f"- {key}: {value!r} ({role})")
    lines += [
        "",
        "## Models",
        "",
        f"- aircraft: {_aircraft_line(models)}",
        f"- scenario: `{models.scenario!r}`",
        f"- resolver: `{models.resolver!r}`",
        f"- recovery: `{models.recovery!r}`",
        f"- noise: `{models.noise!r}`",
        "",
        "## Base config",
        "",
        "```yaml",
        yaml.safe_dump(config.to_mapping(), sort_keys=False).rstrip(),
        "```",
        "",
        "## Results",
        "",
    ]
    rows = result.records()
    if rows:
        columns = list(rows[0])
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "---|" * len(columns))
        for row in rows:
            lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path


def _aircraft_line(models: Models) -> str:
    from cdarr.aircraft import as_pair

    own, intr = as_pair(models.aircraft)
    if own is intr:
        return f"`{own.label}` (BlueSky type `{own.bs_actype}`)"
    return f"ownship `{own.label}` / intruder `{intr.label}`"


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
