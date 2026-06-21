"""Watchdog math (spec 4.2) — pure and deterministic, tested first."""

import math

import numpy as np

from vigil_proxy.watchdog import (
    STATE_MUTATION_PENALTY,
    cosine,
    final_score,
    is_breach,
    mean_similarity,
    shannon_entropy,
    state_penalty,
)


def test_cosine_identical_and_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine(a, a) == 1.0
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_cosine_zero_vector_is_zero():
    assert cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


def test_mean_similarity_no_previous_is_zero():
    assert mean_similarity(np.array([1.0, 0.0]), []) == 0.0


def test_mean_similarity_loop_is_one():
    # Current step identical to every recent step -> perfect self-similarity (a tight loop).
    cur = np.array([0.3, 0.7, 0.1])
    prev = [cur.copy(), cur.copy(), cur.copy()]
    assert math.isclose(mean_similarity(cur, prev), 1.0, rel_tol=1e-9)


def test_mean_similarity_is_the_mean():
    cur = np.array([1.0, 0.0])
    prev = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]  # cos 1.0 and 0.0
    assert mean_similarity(cur, prev) == 0.5


def test_shannon_entropy_all_same_is_zero():
    assert shannon_entropy(["a", "a", "a", "a"]) == 0.0
    assert shannon_entropy([None, None]) == 0.0  # text-only window, no diversity


def test_shannon_entropy_empty_is_zero():
    assert shannon_entropy([]) == 0.0


def test_shannon_entropy_two_equal_labels():
    # Natural-log Shannon entropy of a 50/50 split is ln(2).
    assert math.isclose(shannon_entropy(["a", "b"]), math.log(2), rel_tol=1e-9)


def test_shannon_entropy_diverse_exceeds_floor():
    # A diverse window (paginator scanning distinct tools) sits well above the 0.30 floor.
    assert shannon_entropy(["read", "next", "parse", "store", "read"]) > 0.30


def test_state_penalty():
    assert state_penalty(True) == STATE_MUTATION_PENALTY == 0.30
    assert state_penalty(False) == 0.0


def test_final_score_clamped_and_penalized():
    assert final_score(0.9, 0.0) == 0.9
    # a working, state-mutating loop is pulled down
    assert math.isclose(final_score(0.9, 0.3), 0.6, rel_tol=1e-9)
    assert final_score(0.2, 0.3) == 0.0  # never negative


def test_is_breach_trip_condition():
    # High similarity AND low diversity -> breach.
    assert is_breach(0.9, 0.0, 0.85, 0.30) is True
    # High similarity but diverse tools -> not a breach (healthy varied work).
    assert is_breach(0.9, 0.69, 0.85, 0.30) is False
    # Low similarity -> not a breach even if tools are uniform.
    assert is_breach(0.5, 0.0, 0.85, 0.30) is False
    # Exactly at the thresholds -> strict inequalities mean no breach.
    assert is_breach(0.85, 0.30, 0.85, 0.30) is False


def test_state_mutating_loop_does_not_breach():
    # A real write-loop: high raw similarity, zero tool diversity, but the state penalty
    # drops S_final below the ceiling, so it is correctly NOT flagged.
    sc = 0.95
    p = state_penalty(True)
    s_final = final_score(sc, p)  # 0.65
    assert is_breach(s_final, 0.0, 0.85, 0.30) is False
