"""The RNG stream tree — one integer seed per run, `SeedSequence.spawn` below it.

Mirrors OpenCDaRR's ``rng.py`` contract (their ADR 0001): a run is reproducible as
``config + seed -> result`` because every stochastic component draws from its own
:class:`numpy.random.SeedSequence` child, and children are statistically independent
(``spawn``, not ``seed + k`` offsets, which carry no independence guarantee).

A ``SeedSequence`` plays exactly one of two roles, never both:

- *internal node* — fanned out with :func:`spawn` / addressed with :func:`child`;
- *leaf* — turned into a generator with :func:`generator` and drawn from.

``child(parent, i)`` equals ``spawn(parent, n)[i]`` for any ``n > i`` while the parent is
fresh; mixing the two on one parent re-uses indices, so pick one per parent.

The episode tree this package wires (ADR 0004): ``root(seed)`` -> episode ``j`` ->
``(geometry, navigation, measurement, reception, schedule)``. Episode streams hang off the
root by episode index alone — *not* by condition — so every condition replays the same
encounters and noise (common random numbers across a sweep).
"""

from __future__ import annotations

import numpy as np


def root_seed_sequence(seed: int) -> np.random.SeedSequence:
    """The root of the run's stream tree. One per run."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.SeedSequence(seed)


def spawn(parent: np.random.SeedSequence, n: int) -> list[np.random.SeedSequence]:
    """``n`` independent children of ``parent`` (stateful: continues where it left off)."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return parent.spawn(n)


def child(parent: np.random.SeedSequence, i: int) -> np.random.SeedSequence:
    """Child ``i`` of ``parent``, addressed absolutely and without touching the parent.

    This is what lets a parallel worker rebuild only its slice of a fan-out and still
    draw the numbers the serial run would.
    """
    if i < 0:
        raise ValueError(f"child index must be non-negative, got {i}")
    return np.random.SeedSequence(entropy=parent.entropy, spawn_key=(*parent.spawn_key, i))


def generator(seq: np.random.SeedSequence) -> np.random.Generator:
    """The one place a generator is created (PCG64). ``seq`` becomes a leaf."""
    return np.random.Generator(np.random.PCG64(seq))
