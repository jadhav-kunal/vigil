"""The detector's math (spec 4.2) — pure, deterministic, no I/O.

Operates on embeddings (numpy arrays) and labels that the caller has already produced, so every
function here is trivially unit-testable. Model loading and any async work live elsewhere
(embedder.py, analyzer.py).

Per-step quantities over the sliding window W (the last W steps, current step last):

    Sc       = mean cosine(embed(current), embed(prev_i)) over the other steps in the window
    H_mop    = Shannon entropy of the tool-name distribution in the window (tool diversity)
    P_state  = 0.30 if the current step mutated external state else 0.0
    S_final  = max(0, Sc - P_state)
    breach   = S_final > theta_sim AND H_mop < theta_ent

A trip is K consecutive breaching steps; the consecutive-streak bookkeeping is stateful and
lives in the analyzer, which calls `is_breach` here.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

STATE_MUTATION_PENALTY = 0.30


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector has no magnitude."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_similarity(current: np.ndarray, previous: Sequence[np.ndarray]) -> float:
    """Sc: mean cosine of the current step's embedding against each previous step in the window.

    0.0 when there is no prior step to compare against (a single step cannot be a loop).
    """
    if not previous:
        return 0.0
    sims = [cosine(current, p) for p in previous]
    return float(sum(sims) / len(sims))


def shannon_entropy(labels: Sequence[object]) -> float:
    """H_mop: Shannon entropy (natural log) of the label distribution.

    0.0 for an empty window or a window of one repeated label (no diversity). A read-only,
    text-only step contributes its tool label of None, which is a valid category.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p)
    return h


def state_penalty(caused_state_mutation: bool) -> float:
    return STATE_MUTATION_PENALTY if caused_state_mutation else 0.0


def final_score(sc: float, p_state: float) -> float:
    """S_final = max(0, Sc - P_state). The mutation penalty pulls a genuinely-working
    (state-changing) loop below the similarity ceiling so it is not flagged as stuck."""
    return max(0.0, sc - p_state)


def is_breach(s_final: float, h_mop: float, theta_sim: float, theta_ent: float) -> bool:
    """A single step breaches when it is both highly self-similar and low in tool diversity."""
    return s_final > theta_sim and h_mop < theta_ent
