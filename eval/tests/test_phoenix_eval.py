"""The closed-loop threshold tuner (sponsor observability loop) is offline and deterministic, so
it is tested like the rest of the harness."""

from pathlib import Path

from eval.phoenix_eval import DEFAULT_ENT, DEFAULT_SIM, run_phoenix_eval, sweep


def test_sweep_covers_the_grid_and_includes_defaults():
    rows = sweep(seeds=2)
    assert len(rows) == 16  # 4 x 4 grid
    assert any(r["theta_sim"] == DEFAULT_SIM and r["theta_ent"] == DEFAULT_ENT for r in rows)


def test_best_is_no_worse_than_defaults_and_writes_artifacts(tmp_path):
    result = run_phoenix_eval(seeds=2, out_dir=Path(tmp_path))
    baseline_f1 = result["baseline"]["f1"]
    best_f1 = result["best"]["f1"]
    assert best_f1 >= baseline_f1  # tuning never picks a worse operating point
    assert (Path(tmp_path) / "phoenix_eval.md").exists()
    assert (Path(tmp_path) / "phoenix_eval.json").exists()


def test_phoenix_eval_is_deterministic():
    a = sweep(seeds=2)
    b = sweep(seeds=2)
    assert a == b
