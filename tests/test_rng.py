"""Locks for the RNG stream tree (``blueskycdarr/rng.py``, ADR 0004).

The load-bearing property: ``child`` addresses the same tree ``spawn`` enumerates, so a
parallel worker can rebuild exactly its slice of the episode fan-out — and the
common-random-numbers layout (episode streams keyed by index, not condition) rests on
children being reproducible from ``(root, index)`` alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from blueskycdarr.rng import child, generator, root_seed_sequence, spawn


def _draw(seq: np.random.SeedSequence) -> int:
    """One draw that fingerprints a stream (equal sequences draw the same number)."""
    return int(generator(seq).integers(0, 2**62))


def test_child_matches_spawn_by_index() -> None:
    """``child(p, i)`` is ``spawn(p, n)[i]`` — the same tree, addressed not enumerated."""
    root = root_seed_sequence(12345)
    kids = spawn(root, 8)
    assert [_draw(child(root, i)) for i in range(8)] == [_draw(k) for k in kids]


def test_child_leaves_the_parent_untouched() -> None:
    """Addressing must not advance the parent's spawn counter."""
    root = root_seed_sequence(5)
    child(root, 3)
    assert root.n_children_spawned == 0


def test_same_seed_same_streams_different_seed_different_streams() -> None:
    a, b, c = root_seed_sequence(1), root_seed_sequence(1), root_seed_sequence(2)
    assert _draw(child(a, 0)) == _draw(child(b, 0))
    assert _draw(child(a, 0)) != _draw(child(c, 0))


def test_invalid_arguments_fail_fast() -> None:
    with pytest.raises(ValueError):
        root_seed_sequence(-1)
    with pytest.raises(ValueError):
        spawn(root_seed_sequence(0), -1)
    with pytest.raises(ValueError):
        child(root_seed_sequence(0), -1)
