"""Benchmark invariants (the eval harness is a first-class deliverable, so it is tested too):
D1 loops must trip, D2 healthy work must stay quiet, compression must shrink looping context more
than normal context, full-Vigil net saving must be positive, and the whole run must be
deterministic (identical results on a re-run)."""

from pathlib import Path

from eval.benchmark import run_benchmark
from eval.conditions import CONDITIONS
from eval.datasets import build_datasets
from eval.engine import run_scenario
from eval.report import compression_ratio_by_dataset
from vigil_proxy.pricing import DEFAULT_PRICE_TABLE

ALL = [c.id for c in CONDITIONS]


def _run(tmp_path, seeds=2):
    return run_benchmark(
        seeds=seeds,
        condition_ids=ALL,
        datasets=("D1", "D2", "D3"),
        out_dir=Path(tmp_path),
        make_plots=False,
    )


def test_d1_loops_trip_and_d2_stays_quiet(tmp_path):
    summary = _run(tmp_path)
    det = summary["detection"]  # confusion matrix over D1∪D2 under C1 (breaker only)
    assert det["fp"] == 0  # no healthy D2 session was wrongly halted
    assert det["fn"] == 0  # every D1 loop was caught (math + judge)
    assert det["recall"] == 1.0 and det["precision"] == 1.0


def test_tight_loop_is_caught_by_the_watchdog_math(tmp_path):
    # The cosine/entropy math (not the judge) must catch at least the blatant tight loop.
    scn = next(s for s in build_datasets(0) if s.archetype == "tight_loop")
    c1 = next(c for c in CONDITIONS if c.id == "C1")
    rec = run_scenario(scn, c1, seed=0, price_table=DEFAULT_PRICE_TABLE)
    assert rec.opened and rec.opened_by == "math"
    assert rec.steps_executed < rec.budget_steps  # halted before burning the full budget


def test_compression_shrinks_loops_more_than_normal_context(tmp_path):
    _run(tmp_path)
    summary = _run(tmp_path)  # ratios are computed from records; recompute via the report helper
    # Pull ratios straight from a fresh record set to avoid depending on summary internals.
    records = []
    for seed in range(2):
        for scn in build_datasets(seed):
            for c in CONDITIONS:
                records.append(run_scenario(scn, c, seed=seed, price_table=DEFAULT_PRICE_TABLE))
    ratios = compression_ratio_by_dataset(records)
    assert ratios["D1"] < 1.0  # looping context is compressed
    assert ratios["D1"] <= ratios["D3"]  # ... more than normal context (never conflated)
    assert summary["records"] == 2 * len(build_datasets(0)) * len(CONDITIONS)


def test_full_vigil_net_saving_is_positive_and_breaker_halts(tmp_path):
    records = []
    for seed in range(2):
        for scn in build_datasets(seed):
            for c in CONDITIONS:
                records.append(run_scenario(scn, c, seed=seed, price_table=DEFAULT_PRICE_TABLE))
    by_cond = {c.id: [r for r in records if r.condition == c.id] for c in CONDITIONS}
    paired: dict[str, dict[str, float]] = {}
    for r in records:
        paired.setdefault(r.scenario_id, {})[r.condition] = r.cost_usd
    diffs = [cond["C0"] - cond["C5"] for cond in paired.values() if "C0" in cond and "C5" in cond]
    assert sum(diffs) / len(diffs) > 0  # full Vigil is net cheaper than control
    # D3 outcome preserved: tasks still succeed under compression + routing.
    d3_full = [r for r in by_cond["C5"] if r.task_success is not None]
    assert d3_full and all(r.task_success for r in d3_full)


def test_artifacts_written(tmp_path):
    _run(tmp_path)
    for name in (
        "results.json",
        "savings_table.md",
        "savings_table.csv",
        "ablation.md",
        "net_savings.md",
        "detection_report.md",
    ):
        assert (Path(tmp_path) / name).exists()


def test_benchmark_is_deterministic(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a["detection"] == b["detection"]
    assert (Path(tmp_path) / "a" / "results.json").read_text() == (
        Path(tmp_path) / "b" / "results.json"
    ).read_text()
