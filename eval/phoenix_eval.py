"""Phoenix/Arize closed-loop evaluation (eval design + spec 4.9 sponsor): judge -> tune θ ->
before/after F1.

This is the threshold-tuning half of the observability loop. It sweeps the watchdog thresholds
(θ_sim, θ_ent) over the labeled detection datasets (D1 loops ∪ D2 healthy), scores the
cosine/entropy MATH-only detector at each operating point, and reports the F1 at the shipped
defaults versus the best operating point found. It is fully offline and deterministic (reuses the
benchmark's scripted mock upstream); if a Phoenix/Arize backend is configured the summary can also
be exported as a span, but the tuning itself never needs the network.

    python -m eval.phoenix_eval [--seeds 10] [--out eval/out]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vigil_proxy.pricing import DEFAULT_PRICE_TABLE

from .conditions import BY_ID
from .datasets import build_datasets
from .engine import run_scenario
from .report import _confusion

SIM_GRID = [0.80, 0.85, 0.90, 0.95]
ENT_GRID = [0.20, 0.30, 0.40, 0.50]
DEFAULT_SIM, DEFAULT_ENT = 0.85, 0.30


def _f1_at(seeds: int, theta_sim: float, theta_ent: float) -> dict:
    c1 = BY_ID["C1"]
    records = []
    for seed in range(seeds):
        for scn in build_datasets(seed, ("D1", "D2")):
            records.append(
                run_scenario(
                    scn,
                    c1,
                    seed=seed,
                    price_table=DEFAULT_PRICE_TABLE,
                    theta_sim=theta_sim,
                    theta_ent=theta_ent,
                )
            )
    # Math-only detection, so the sweep tunes the cosine/entropy detector itself (not the judge).
    cm = _confusion(records, lambda r: r.opened and r.opened_by == "math")
    return {"theta_sim": theta_sim, "theta_ent": theta_ent, **cm}


def sweep(seeds: int) -> list[dict]:
    return [_f1_at(seeds, ts, te) for ts in SIM_GRID for te in ENT_GRID]


def _f1(row: dict) -> float:
    f1 = row["f1"]
    return f1 if f1 == f1 else -1.0  # NaN-safe ordering


def run_phoenix_eval(seeds: int, out_dir: Path) -> dict:
    rows = sweep(seeds)
    baseline = next(
        r for r in rows if r["theta_sim"] == DEFAULT_SIM and r["theta_ent"] == DEFAULT_ENT
    )
    best = max(rows, key=_f1)

    lines = [
        "# Phoenix closed-loop threshold tuning (judge -> tune θ -> before/after F1)",
        "",
        "Math-only (cosine + tool-entropy) detection over D1∪D2, swept across (θ_sim, θ_ent).",
        "",
        f"- Shipped defaults (θ_sim={DEFAULT_SIM}, θ_ent={DEFAULT_ENT}): "
        f"F1={baseline['f1']:.3f} (P={baseline['precision']:.3f} R={baseline['recall']:.3f})",
        f"- Best operating point (θ_sim={best['theta_sim']}, θ_ent={best['theta_ent']}): "
        f"F1={best['f1']:.3f} (P={best['precision']:.3f} R={best['recall']:.3f})",
        "",
        "| θ_sim | θ_ent | precision | recall | F1 | FPR |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['theta_sim']} | {r['theta_ent']} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['fpr']:.3f} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phoenix_eval.md").write_text("\n".join(lines) + "\n")
    (out_dir / "phoenix_eval.json").write_text(
        json.dumps({"baseline": baseline, "best": best, "rows": rows}, indent=2)
    )
    return {"baseline": baseline, "best": best, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.phoenix_eval")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="eval/out")
    args = p.parse_args(argv)
    result = run_phoenix_eval(args.seeds, Path(args.out))
    b, best = result["baseline"], result["best"]
    print(
        f"defaults F1={b['f1']:.3f} -> best F1={best['f1']:.3f} "
        f"at θ_sim={best['theta_sim']} θ_ent={best['theta_ent']} (wrote {args.out}/phoenix_eval.md)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
