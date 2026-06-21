"""Turn the per-run records into the six benchmark artifacts (eval design §9): results.json,
savings_table.{md,csv}, ablation.{md,csv}, net_savings.md, detection_report.md. Every aggregate
cost/token figure is a mean with a bootstrap 95% CI; the C0-vs-C5 headline carries a paired
Wilcoxon test. The two honesty moves are baked in: savings are reported NET of Vigil's own
overhead, and compression ratios are split by dataset (never conflated)."""

from __future__ import annotations

import dataclasses
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .conditions import CONDITIONS
from .engine import RunRecord
from .stats import bootstrap_ci, wilcoxon_signed_rank


def _by_condition(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    out: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        out[r.condition].append(r)
    return out


def _paired_cost(records: list[RunRecord]) -> dict[str, dict[str, float]]:
    """scenario_id -> {condition_id: cost_usd}. scenario_id already encodes the seed."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        out[r.scenario_id][r.condition] = r.cost_usd
    return out


def _aligned(
    paired: dict[str, dict[str, float]], a: str, b: str
) -> tuple[list[float], list[float]]:
    xa, xb = [], []
    for cond in paired.values():
        if a in cond and b in cond:
            xa.append(cond[a])
            xb.append(cond[b])
    return xa, xb


# --------------------------------------------------------------------------- savings table


def savings_table(records: list[RunRecord]) -> tuple[str, str]:
    by = _by_condition(records)
    paired = _paired_cost(records)
    rows = []
    for c in CONDITIONS:
        rs = by.get(c.id, [])
        if not rs:
            continue
        tokens = bootstrap_ci([float(r.total_tokens) for r in rs], seed=1)
        cost = bootstrap_ci([r.cost_usd for r in rs], seed=2)
        d3 = [r.task_success for r in rs if r.task_success is not None]
        completion = 100.0 * statistics.fmean(d3) if d3 else float("nan")
        d1 = [r for r in rs if r.is_loop]
        stopped = 100.0 * statistics.fmean([r.opened for r in d1]) if d1 else 0.0
        a0, ax = _aligned(paired, "C0", c.id)
        net = bootstrap_ci([x - y for x, y in zip(a0, ax, strict=False)], seed=3) if a0 else None
        rows.append((c, tokens, cost, completion, stopped, net))

    header = "| Condition | Tokens/task | Cost/task | Completion (D3) | Loops stopped (D1) | Net saving vs C0 |"
    sep = "|---|---|---|---|---|---|"
    md = [header, sep]
    csv = [
        "condition,tokens_mean,tokens_lo,tokens_hi,cost_mean,cost_lo,cost_hi,completion_pct,loops_stopped_pct,net_mean,net_lo,net_hi"
    ]
    for c, tokens, cost, completion, stopped, net in rows:
        comp = "—" if completion != completion else f"{completion:.0f}%"
        nets = net.fmt(dollars=True) if net else "—"
        md.append(
            f"| {c.id} {c.label} | {tokens.fmt()} | {cost.fmt(dollars=True)} | {comp} | {stopped:.0f}% | {nets} |"
        )
        nm = (net.mean if net else 0.0, net.lo if net else 0.0, net.hi if net else 0.0)
        csv.append(
            f"{c.id},{tokens.mean:.1f},{tokens.lo:.1f},{tokens.hi:.1f},{cost.mean:.6f},{cost.lo:.6f},"
            f"{cost.hi:.6f},{completion:.1f},{stopped:.1f},{nm[0]:.6f},{nm[1]:.6f},{nm[2]:.6f}"
        )
    return "\n".join(md) + "\n", "\n".join(csv) + "\n"


# --------------------------------------------------------------------------- ablation


def ablation(records: list[RunRecord]) -> tuple[str, str]:
    paired = _paired_cost(records)
    attrib = {"C1": "M2 breaker", "C2": "M3 governor", "C3": "M1 compression", "C4": "M4 cache"}
    md = ["| Mechanism | Condition | Mean saving vs C0 | 95% CI |", "|---|---|---|---|"]
    csv = ["mechanism,condition,mean_saving,lo,hi"]
    indiv_sum = 0.0
    for cid, name in attrib.items():
        a0, ax = _aligned(paired, "C0", cid)
        if not a0:
            continue
        ci = bootstrap_ci([x - y for x, y in zip(a0, ax, strict=False)], seed=4)
        indiv_sum += ci.mean
        md.append(f"| {name} | {cid} | ${ci.mean:.5f} | [${ci.lo:.5f}, ${ci.hi:.5f}] |")
        csv.append(f"{name},{cid},{ci.mean:.6f},{ci.lo:.6f},{ci.hi:.6f}")

    a0, a5 = _aligned(paired, "C0", "C5")
    if a0:
        c5 = bootstrap_ci([x - y for x, y in zip(a0, a5, strict=False)], seed=5)
        md.append(f"| **Full (measured)** | C5 | ${c5.mean:.5f} | [${c5.lo:.5f}, ${c5.hi:.5f}] |")
        csv.append(f"full_measured,C5,{c5.mean:.6f},{c5.lo:.6f},{c5.hi:.6f}")
        md.append("")
        md.append(
            f"> Sum of individual mechanism savings = ${indiv_sum:.5f}; measured C5 = ${c5.mean:.5f}. "
            "C5 < the sum because M2 (breaker) and M4 (cache) both attack loop redundancy — their "
            "effect is sub-additive, so the combined figure is reported as measured, never summed."
        )
    return "\n".join(md) + "\n", "\n".join(csv) + "\n"


# --------------------------------------------------------------------------- net savings


def net_savings(records: list[RunRecord]) -> str:
    by = _by_condition(records)
    paired = _paired_cost(records)
    c0 = by.get("C0", [])
    c5 = by.get("C5", [])
    gross_a, gross_b = _aligned(paired, "C0", "C5")
    gross = (
        statistics.fmean([a - b for a, b in zip(gross_a, gross_b, strict=False)])
        if gross_a
        else 0.0
    )
    overhead = statistics.fmean([r.overhead_usd for r in c5]) if c5 else 0.0
    llm_only = statistics.fmean([r.llm_cost_usd for r in c5]) if c5 else 0.0
    base = statistics.fmean([r.cost_usd for r in c0]) if c0 else 0.0
    # C0 has no Vigil overhead, so base == its LLM cost. Gross = the LLM-cost reduction; net then
    # subtracts Vigil's own overhead. The paired `gross` (C0-C5 of total cost) already equals net.
    gross_llm = base - llm_only
    net = gross  # cost_usd already folds in overhead, so the paired diff is already net
    pt = wilcoxon_signed_rank(gross_a, gross_b) if gross_a else None

    lines = [
        "# Net savings accounting (eval design §7)",
        "",
        "Gross savings overstate the truth because Vigil itself spends tokens and time. Below we "
        "show the gross LLM-cost reduction, then subtract Vigil's own overhead (goal-judge LLM "
        "calls + semantic-cache lookups) to get the NET figure — which equals the paired C0→C5 "
        "delta of total cost.",
        "",
        f"- Baseline cost/task (C0):        ${base:.5f}",
        f"- Full-Vigil LLM cost/task (C5):  ${llm_only:.5f}",
        f"- Gross LLM saving/task:          ${gross_llm:.5f}",
        f"- minus Vigil overhead/task:      ${overhead:.5f}  (goal-judge + cache lookups)",
        f"- = Net saving/task:              ${net:.5f}  ({100 * net / base if base else 0:.1f}% of baseline)",
    ]
    if pt:
        lines.append(f"- Paired Wilcoxon C0 vs C5:       {pt.fmt()}")
    lines += [
        "",
        "## Pre-empting the objections",
        "- *Router cost/latency*: the heuristic governor is the default — ~0 cost, ~0 latency. The "
        "LLM-classifier variant (~$0.0001/call) breaks even after a single avoided frontier step.",
        "- *Embedding every step*: runs async off the hot path (no added response latency) and "
        "locally (no API cost) — reported as wall-time, not dollars.",
        "- *Cache miss is pure overhead*: true and accounted — every lookup is charged, hit or miss.",
        "",
        "## Break-even regimes (§7.1)",
        "| Layer | Net-positive when | Break-even |",
        "|---|---|---|",
        "| M1 compression (Layer 2) | big/growing input, expensive model, reused context | "
        "`tokens_removed × input_price > TTC_price_per_token` |",
        "| M4 semantic cache | high repetition, output-heavy responses | "
        "`output_cost × hit_rate > lookup_overhead` |",
        "",
        "The breaker (M2) is **insurance**, not an average-case saving: its value is dominated by "
        "the rare runaway (the fat tail), so a mean-cost analysis understates it. See the P90/P99 "
        "of cost on D1 below.",
        "",
        _tail_block(by),
    ]
    return "\n".join(lines) + "\n"


def _tail_block(by: dict[str, list[RunRecord]]) -> str:
    def pct(rs: list[RunRecord], q: float) -> float:
        costs = sorted(r.cost_usd for r in rs if r.is_loop)
        if not costs:
            return 0.0
        k = min(len(costs) - 1, int(q * (len(costs) - 1)))
        return costs[k]

    c0, c1 = by.get("C0", []), by.get("C1", [])
    return (
        "### Cost tail on D1 loops (insurance value of the breaker)\n"
        f"- C0 (no breaker)  P90=${pct(c0, 0.9):.5f}  P99=${pct(c0, 0.99):.5f}\n"
        f"- C1 (breaker on)  P90=${pct(c1, 0.9):.5f}  P99=${pct(c1, 0.99):.5f}\n"
    )


# --------------------------------------------------------------------------- detection


def _confusion(records: list[RunRecord], detected) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for r in records:
        if r.dataset not in ("D1", "D2"):
            continue
        hit = detected(r)
        if r.is_loop and hit:
            tp += 1
        elif r.is_loop and not hit:
            fn += 1
        elif not r.is_loop and hit:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * prec * rec / (prec + rec)
        if prec == prec and rec == rec and (prec + rec)
        else float("nan")
    )
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
    }


def detection_report(records: list[RunRecord]) -> str:
    c1 = [r for r in records if r.condition == "C1"]
    full = _confusion(c1, lambda r: r.opened)
    math_only = _confusion(c1, lambda r: r.opened and r.opened_by == "math")

    trips = [r.trip_step for r in c1 if r.is_loop and r.opened and r.trip_step is not None]
    detected_loops = [
        r for r in c1 if r.is_loop and r.opened and r.trip_step is not None and r.budget_steps
    ]
    mstd = statistics.fmean(trips) if trips else float("nan")
    budget_eff = (
        100.0
        * statistics.fmean([1 - ((r.trip_step or 0) + 1) / r.budget_steps for r in detected_loops])
        if detected_loops
        else float("nan")
    )

    def fmt(c: dict) -> str:
        return (
            f"TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']} | "
            f"precision={c['precision']:.3f} recall={c['recall']:.3f} F1={c['f1']:.3f} FPR={c['fpr']:.3f}"
        )

    lines = [
        "# Detection report (eval design §6) — M2, condition C1 over D1∪D2",
        "",
        "Two confusion matrices, so the cosine/entropy math is not credited with what the "
        "goal-judge adds:",
        "",
        f"- **Watchdog math only** (cosine + tool-entropy): {fmt(math_only)}",
        f"- **Watchdog + goal-judge**:                      {fmt(full)}",
        "",
        f"- Mean Steps To Detection (MSTD): {mstd:.2f}",
        f"- Detection budget efficiency (mean % of budget left at trip): {budget_eff:.1f}%",
        "",
        "Per-archetype recall (full detector):",
    ]
    by_arch: dict[str, list[RunRecord]] = defaultdict(list)
    for r in c1:
        if r.is_loop:
            by_arch[r.archetype].append(r)
    for arch, rs in sorted(by_arch.items()):
        opened = sum(r.opened for r in rs)
        by_math = sum(r.opened and r.opened_by == "math" for r in rs)
        lines.append(
            f"  - {arch}: {opened}/{len(rs)} detected ({by_math} by math, {opened - by_math} by judge)"
        )
    lines += [
        "",
        "D2 (healthy-repetitive) must stay quiet; any FP above is a wrongly-halted healthy session "
        "and is reported, not hidden.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- aux stats for plots


def compression_ratio_by_dataset(records: list[RunRecord]) -> dict[str, float]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.condition in ("C3", "C5") and r.compression_ratio is not None:
            out[r.dataset].append(r.compression_ratio)
    return {ds: statistics.median(v) for ds, v in out.items() if v}


def cache_hitrate_by_dataset(records: list[RunRecord]) -> dict[str, float]:
    hits: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for r in records:
        if r.condition in ("C4", "C5"):
            hits[r.dataset] += r.cache_hits
            total[r.dataset] += r.cache_hits + r.cache_misses
    return {ds: (hits[ds] / total[ds]) for ds in total if total[ds]}


# --------------------------------------------------------------------------- driver


def write_all(records: list[RunRecord], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps([_record_json(r) for r in records], indent=2))
    st_md, st_csv = savings_table(records)
    (out_dir / "savings_table.md").write_text(st_md)
    (out_dir / "savings_table.csv").write_text(st_csv)
    ab_md, ab_csv = ablation(records)
    (out_dir / "ablation.md").write_text(ab_md)
    (out_dir / "ablation.csv").write_text(ab_csv)
    (out_dir / "net_savings.md").write_text(net_savings(records))
    (out_dir / "detection_report.md").write_text(detection_report(records))
    return {
        "records": len(records),
        "detection": _confusion([r for r in records if r.condition == "C1"], lambda r: r.opened),
        "compression_ratio_by_dataset": compression_ratio_by_dataset(records),
        "cache_hitrate_by_dataset": cache_hitrate_by_dataset(records),
    }


def _record_json(r: RunRecord) -> dict:
    d = dataclasses.asdict(r)
    d["cost_usd"] = r.cost_usd
    d["total_tokens"] = r.total_tokens
    d["compression_ratio"] = r.compression_ratio
    return d
