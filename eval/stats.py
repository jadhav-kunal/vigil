"""Statistics for the benchmark (eval design §6): bootstrap CIs (robust to the right-skewed cost
distribution), a paired Wilcoxon signed-rank test (C0 vs C5), and the matched-pairs rank-biserial
effect size. scipy is used when available and falls back to a deterministic numpy implementation,
so the harness still runs without the optional `eval` extra (Invariant I2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # scipy is in the optional `eval` extra; degrade gracefully if absent.
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - exercised only without the extra installed
    _scipy_stats = None


@dataclass
class CI:
    mean: float
    lo: float
    hi: float

    def fmt(self, dollars: bool = False) -> str:
        u = "$" if dollars else ""
        p = 4 if dollars else 1
        return f"{u}{self.mean:.{p}f} [{u}{self.lo:.{p}f}, {u}{self.hi:.{p}f}]"


def bootstrap_ci(values: list[float], *, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> CI:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return CI(0.0, 0.0, 0.0)
    if arr.size == 1:
        return CI(float(arr[0]), float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(arr.mean()), float(lo), float(hi))


@dataclass
class PairedTest:
    statistic: float
    p_value: float
    rank_biserial: float
    n: int

    def fmt(self) -> str:
        return f"W={self.statistic:.1f}, p={self.p_value:.4g}, rank-biserial={self.rank_biserial:+.3f} (n={self.n})"


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> PairedTest:
    """Paired non-parametric test on a vs b (e.g. cost under C0 vs C5). Zero-difference pairs are
    dropped (Wilcoxon convention). Returns the matched-pairs rank-biserial effect size too."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    diff = x - y
    nz = diff[diff != 0]
    n = int(nz.size)
    if n == 0:
        return PairedTest(0.0, 1.0, 0.0, 0)

    ranks = _rankdata(np.abs(nz))
    r_plus = float(ranks[nz > 0].sum())
    r_minus = float(ranks[nz < 0].sum())
    total = r_plus + r_minus
    rank_biserial = (r_plus - r_minus) / total if total else 0.0

    if _scipy_stats is not None:
        res = _scipy_stats.wilcoxon(x, y, zero_method="wilcox", correction=False, mode="auto")
        return PairedTest(float(res.statistic), float(res.pvalue), rank_biserial, n)

    # Normal approximation with tie correction (deterministic fallback).
    w = min(r_plus, r_minus)
    mean_w = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(nz), return_counts=True)
    tie = float((counts**3 - counts).sum())
    var_w = (n * (n + 1) * (2 * n + 1) - tie / 2.0) / 24.0
    z = (w - mean_w) / np.sqrt(var_w) if var_w > 0 else 0.0
    p = float(2.0 * (1.0 - _norm_cdf(abs(z))))
    return PairedTest(float(w), min(1.0, p), rank_biserial, n)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties shared — matches scipy.stats.rankdata('average')."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _norm_cdf(x: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
