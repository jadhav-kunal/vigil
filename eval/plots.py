"""The benchmark's PNG plots (eval design §9). matplotlib lives in the optional `eval` extra; if
it is absent the harness skips plotting and says so (Invariant I2) — the json/csv/md artifacts are
the source of truth, the plots are a convenience."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from .conditions import CONDITIONS
from .engine import RunRecord
from .report import cache_hitrate_by_dataset, compression_ratio_by_dataset

ACCENT = "#6b8afd"
DARK = "#0f1117"


def render_plots(records: list[RunRecord], out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(name)

    by_cond: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_cond[r.condition].append(r)

    # cost_by_condition.png
    fig, ax = plt.subplots(figsize=(7, 4))
    ids = [c.id for c in CONDITIONS if c.id in by_cond]
    costs = [statistics.fmean([r.cost_usd for r in by_cond[i]]) for i in ids]
    ax.bar(ids, costs, color=ACCENT)
    ax.set_title("Mean cost per task by condition")
    ax.set_ylabel("USD / task")
    save(fig, "cost_by_condition.png")

    # tokens_before_after.png (compression, C3)
    c3 = [r for r in by_cond.get("C3", []) if r.tokens_before_compression]
    fig, ax = plt.subplots(figsize=(7, 4))
    before = statistics.fmean([r.tokens_before_compression for r in c3]) if c3 else 0
    after = statistics.fmean([r.tokens_after_compression for r in c3]) if c3 else 0
    ax.bar(["forwarded as-is", "after Vigil L1"], [before, after], color=["#888", ACCENT])
    ax.set_title("Context tokens before vs after compression (wire-measured)")
    ax.set_ylabel("tokens / task")
    save(fig, "tokens_before_after.png")

    # compression_ratio_by_dataset.png
    ratios = compression_ratio_by_dataset(records)
    if ratios:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(ratios.keys()), [v for v in ratios.values()], color=ACCENT)
        ax.axhline(1.0, color="#bbb", linestyle="--")
        ax.set_title("Compression ratio (after/before) by dataset — D1 loops vs D3 normal")
        ax.set_ylabel("ratio (lower = more removed)")
        save(fig, "compression_ratio_by_dataset.png")

    # cache_hitrate_by_dataset.png
    hits = cache_hitrate_by_dataset(records)
    if hits:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(hits.keys()), [100 * v for v in hits.values()], color=ACCENT)
        ax.set_title("Semantic-cache hit rate by dataset")
        ax.set_ylabel("hit rate %")
        save(fig, "cache_hitrate_by_dataset.png")

    # pr_curve.png — sweep the breaker trip threshold proxy via trip_step availability is not a
    # continuous score, so we plot the operating point (precision, recall) for math vs full.
    c1 = [r for r in by_cond.get("C1", []) if r.dataset in ("D1", "D2")]
    if c1:
        from .report import _confusion

        full = _confusion(c1, lambda r: r.opened)
        math_only = _confusion(c1, lambda r: r.opened and r.opened_by == "math")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            [math_only["recall"]], [math_only["precision"]], color="#888", label="math only", s=80
        )
        ax.scatter([full["recall"]], [full["precision"]], color=ACCENT, label="math + judge", s=80)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title("Detection operating points (D1∪D2)")
        ax.legend()
        save(fig, "pr_curve.png")

    return written
