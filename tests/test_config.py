"""Locks for the config schema (``blueskycdarr/config.py``, ADR 0003).

What matters: a config built in Python is checked exactly like one parsed from a file,
unknown keys fail with the legal list in the message, and ``to_mapping`` round-trips —
the provenance card's config block must be paste-able as a run file.
"""

from __future__ import annotations

import pytest

from blueskycdarr.config import (
    CommConfig,
    Config,
    ConflictConfig,
    SimulationConfig,
    UncertaintyConfig,
    config_from_mapping,
)


def test_defaults_are_a_valid_config() -> None:
    config = Config()
    assert config.conflict.rpz == 50.0
    assert config.comm.reception_prob == 1.0


def test_validation_names_every_failed_constraint() -> None:
    with pytest.raises(ValueError, match="reception_prob"):
        CommConfig(reception_prob=1.5)
    with pytest.raises(ValueError, match="broadcast_jitter_s"):
        CommConfig(broadcast_interval_s=1.0, broadcast_jitter_s=1.0)
    with pytest.raises(ValueError, match="resolution_margin"):
        ConflictConfig(resolution_margin=0.9)
    with pytest.raises(ValueError, match="cdr_dt >= dt"):
        SimulationConfig(dt=1.0, cdr_dt=0.5)
    with pytest.raises(ValueError, match="pos_ci95"):
        UncertaintyConfig(pos_ci95=-1.0)


def test_unknown_top_level_key_fails_with_the_legal_list() -> None:
    with pytest.raises(ValueError, match="typo_block") as err:
        config_from_mapping({"typo_block": {}})
    assert "conflict" in str(err.value)  # the message teaches the schema


def test_unknown_section_key_fails_with_the_legal_list() -> None:
    with pytest.raises(ValueError, match="jitter"):
        config_from_mapping({"comm": {"jitter": 0.1}})


def test_to_mapping_round_trips() -> None:
    config = Config(
        seed=7,
        uncertainty=UncertaintyConfig(pos_ci95=10.0, vel_ci95=1.0),
        comm=CommConfig(reception_prob=0.8, max_range_m=1000.0, latency_s=0.1),
    )
    assert config_from_mapping(config.to_mapping()) == config


def test_range_gate_reads_none_as_infinite() -> None:
    assert CommConfig(max_range_m=None).range_gate_m == float("inf")
    assert CommConfig(max_range_m=500.0).range_gate_m == 500.0
