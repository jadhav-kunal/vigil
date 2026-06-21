"""Vigil benchmark — the proof (eval design §10).

    python -m eval.benchmark [--conditions C0,C1,...] [--seeds 20] [--datasets D1,D2,D3] [--out DIR]

Fully deterministic and offline: a scripted mock upstream replays labeled trajectories, and the
ONLY thing that changes across conditions is which Vigil mechanisms are enabled (a paired design
on identical seeds). Emits the six artifacts of §9 to `eval/out/`. `--live` is reserved for a
future real-provider outcome-preservation run and is intentionally off by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vigil_proxy.pricing import DEFAULT_PRICE_TABLE

from .conditions import BY_ID, CONDITIONS
from .datasets import build_datasets
from .engine import RunRecord, run_scenario
from .plots import render_plots
from .report import write_all


def run_benchmark(
    *,
    seeds: int,
    condition_ids: list[str],
    datasets: tuple[str, ...],
    out_dir: Path,
    make_plots: bool = True,
) -> dict:
    conditions = [BY_ID[c] for c in condition_ids]
    records: list[RunRecord] = []
    for seed in range(seeds):
        scenarios = build_datasets(seed, datasets)
        for scenario in scenarios:
            for condition in conditions:
                records.append(
                    run_scenario(scenario, condition, seed=seed, price_table=DEFAULT_PRICE_TABLE)
                )
    summary = write_all(records, out_dir)
    summary["plots"] = render_plots(records, out_dir) if make_plots else []
    summary["seeds"] = seeds
    summary["conditions"] = condition_ids
    summary["datasets"] = list(datasets)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.benchmark", description="Vigil savings benchmark")
    p.add_argument("--conditions", default=",".join(c.id for c in CONDITIONS))
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--datasets", default="D1,D2,D3")
    p.add_argument("--out", default="eval/out")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--live",
        action="store_true",
        help="(reserved) run D3 against a real provider for outcome-preservation; not implemented",
    )
    args = p.parse_args(argv)

    if args.live:
        print(
            "--live is reserved for a future real-provider run and is not implemented; "
            "the deterministic benchmark below is the reproducible default."
        )

    condition_ids = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in condition_ids if c not in BY_ID]
    if unknown:
        p.error(f"unknown conditions: {unknown}; choose from {list(BY_ID)}")
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    out_dir = Path(args.out)

    summary = run_benchmark(
        seeds=args.seeds,
        condition_ids=condition_ids,
        datasets=datasets,
        out_dir=out_dir,
        make_plots=not args.no_plots,
    )

    det = summary["detection"]
    print(f"\nVigil benchmark complete — {summary['records']} runs over {args.seeds} seeds.")
    print(f"Artifacts written to {out_dir}/")
    print(
        f"  detection (C1): precision={det['precision']:.3f} recall={det['recall']:.3f} F1={det['f1']:.3f}"
    )
    print(f"  compression ratio by dataset: {summary['compression_ratio_by_dataset']}")
    print(f"  cache hit rate by dataset:    {summary['cache_hitrate_by_dataset']}")
    if summary["plots"]:
        print(f"  plots: {', '.join(summary['plots'])}")
    else:
        print("  plots: skipped (matplotlib not installed — `pip install -e .[eval]`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
