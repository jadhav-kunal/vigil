"""Deterministic, seeded datasets for the Vigil benchmark (eval design §3).

Every scenario is a fully scripted trajectory: a list of `ScriptedStep`s that a deterministic
mock upstream replays (no network, no real LLM), so the whole benchmark is reproducible from a
seed. Each scenario carries ground-truth labels so detection quality is measurable.

  D1 — looping sessions (must trip):       tight / semantic-shift / meltdown / long-horizon-drift
  D2 — healthy repetitive (must NOT trip): paginator / table-scan / batch-format
  D3 — normal multi-step tasks:            verifiable success criterion, diverse context

The datasets are designed so the watchdog MATH (cosine + tool-entropy) genuinely separates D1
from D2 — healthy work varies its content/tools, loops do not. The harder loop archetypes
(semantic-shift, meltdown) are designed to slip past the cosine/entropy math and be caught by the
goal-judge instead, so the detection report can show the math-only gap honestly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
    "sigma tau upsilon phi chi psi omega quartz vortex cobalt ember lattice cipher harbor "
    "meadow falcon granite tunnel beacon"
).split()


def _sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n))


@dataclass
class ScriptedStep:
    assistant_text: str
    tool_name: str | None
    tool_args: dict | None
    tool_result: str  # the tool output appended to history (drives context growth + dedup)
    completion_tokens: int
    caused_state_mutation: bool
    progressed: bool  # ground truth: did this step advance the goal? (drives the harness judge)


@dataclass
class Scenario:
    scenario_id: str
    dataset: str  # "D1" | "D2" | "D3"
    archetype: str
    is_loop: bool  # ground-truth label
    inevitable_step: int | None  # step at which a loop becomes inevitable (budget efficiency)
    system_prompt: str
    user_goal: str
    steps: list[ScriptedStep]
    success_token: str | None = None  # D3: marker the final assistant text must contain
    tools: list[dict] = field(default_factory=list)

    @property
    def budget_steps(self) -> int:
        """What an unbroken run (C0) executes — the full scripted length."""
        return len(self.steps)


# --------------------------------------------------------------------------- D1 loops


def _tight_loop(rng: random.Random, n: int = 16) -> Scenario:
    result = "status=PENDING retries=exhausted queue=blocked " + _sentence(rng, 24)
    steps = [
        ScriptedStep(
            assistant_text="Checking the deploy status again.",
            tool_name="check_status",
            tool_args={"job": "deploy-1"},
            tool_result=result,
            completion_tokens=40,
            caused_state_mutation=False,
            progressed=(i == 0),
        )
        for i in range(n)
    ]
    return Scenario(
        "tight",
        "D1",
        "tight_loop",
        True,
        3,
        "You are a deploy agent.",
        "Get the deploy to succeed.",
        steps,
        tools=[{"type": "function", "function": {"name": "check_status"}}],
    )


def _semantic_shift(rng: random.Random, n: int = 16) -> Scenario:
    phrasings = [
        "Let me retry the failing migration once more.",
        "I'll attempt the database migration again.",
        "Re-running the schema migration step.",
        "Trying the migration a different way now.",
        "Another attempt at applying the migration.",
    ]
    steps = [
        ScriptedStep(
            assistant_text=phrasings[i % len(phrasings)],
            tool_name="run_migration",
            tool_args={"attempt": i},
            tool_result="migration failed: lock timeout " + _sentence(rng, 6),
            completion_tokens=45,
            caused_state_mutation=False,
            progressed=(i == 0),
        )
        for i in range(n)
    ]
    return Scenario(
        "semshift",
        "D1",
        "semantic_shift",
        True,
        4,
        "You are a DB agent.",
        "Apply the pending migration.",
        steps,
        tools=[{"type": "function", "function": {"name": "run_migration"}}],
    )


def _meltdown(rng: random.Random, n: int = 16) -> Scenario:
    tools = ["grep", "ls", "cat", "curl", "ping", "env", "ps", "kill", "find", "awk"]
    steps = [
        ScriptedStep(
            assistant_text="Trying something else: " + _sentence(rng, 14),
            tool_name=rng.choice(tools),
            tool_args={"x": rng.randint(0, 9999)},
            tool_result=_sentence(rng, 20),
            completion_tokens=50,
            caused_state_mutation=False,
            progressed=(i < 2),
        )
        for i in range(n)
    ]
    return Scenario(
        "meltdown",
        "D1",
        "meltdown",
        True,
        2,
        "You are a debug agent.",
        "Find the root cause of the outage.",
        steps,
        tools=[{"type": "function", "function": {"name": t}} for t in tools],
    )


def _long_drift(rng: random.Random, n: int = 18) -> Scenario:
    steps: list[ScriptedStep] = []
    for i in range(n):
        if i < 4:  # healthy progress, then it collapses into a repeated stuck check
            steps.append(
                ScriptedStep(
                    f"Step {i}: making progress. " + _sentence(rng, 12),
                    rng.choice(["read_file", "edit_file", "run_tests"]),
                    {"i": i},
                    _sentence(rng, 18),
                    48,
                    i % 2 == 0,
                    True,
                )
            )
        else:  # collapses into a repeated stuck check
            steps.append(
                ScriptedStep(
                    "Still trying to get the tests green.",
                    "run_tests",
                    {"suite": "all"},
                    "3 failing: test_auth test_cache test_io (unchanged)",
                    44,
                    False,
                    False,
                )
            )
    return Scenario(
        "drift",
        "D1",
        "long_horizon_drift",
        True,
        5,
        "You are a fixer agent.",
        "Make the test suite pass.",
        steps,
        tools=[{"type": "function", "function": {"name": "run_tests"}}],
    )


# --------------------------------------------------------------------------- D2 healthy


def _paginator(rng: random.Random, n: int = 12) -> Scenario:
    # Same tool every step, but each page is genuinely different content -> low self-similarity.
    steps = [
        ScriptedStep(
            f"Fetched page {i}; recording rows.",
            "get_page",
            {"page": i},
            "rows: " + _sentence(rng, 30),  # distinct content per page
            30,
            False,
            True,
        )
        for i in range(n)
    ]
    return Scenario(
        "paginate",
        "D2",
        "paginator",
        False,
        None,
        "You are a data agent.",
        "Export every page of the report.",
        steps,
        tools=[{"type": "function", "function": {"name": "get_page"}}],
    )


def _table_scan(rng: random.Random, n: int = 12) -> Scenario:
    steps = [
        ScriptedStep(
            f"Scanning column {i}.",
            "scan_column",
            {"col": f"c{i}"},
            f"col c{i}: " + _sentence(rng, 22),
            28,
            False,
            True,
        )
        for i in range(n)
    ]
    return Scenario(
        "tablescan",
        "D2",
        "table_scanner",
        False,
        None,
        "You are an analyst.",
        "Profile each column of the table.",
        steps,
        tools=[{"type": "function", "function": {"name": "scan_column"}}],
    )


def _batch_format(rng: random.Random, n: int = 12) -> Scenario:
    cycle = ["read_file", "format_doc", "write_file"]  # tool diversity -> high entropy, never trips
    steps = [
        ScriptedStep(
            f"{cycle[i % 3]} on item {i}.",
            cycle[i % 3],
            {"item": i},
            _sentence(rng, 16),
            26,
            cycle[i % 3] == "write_file",
            True,
        )
        for i in range(n)
    ]
    return Scenario(
        "batchfmt",
        "D2",
        "batch_formatter",
        False,
        None,
        "You are a formatter.",
        "Reformat every document in the batch.",
        steps,
        tools=[{"type": "function", "function": {"name": t}} for t in cycle],
    )


# --------------------------------------------------------------------------- D3 normal


def _normal_task(rng: random.Random, idx: int) -> Scenario:
    token = f"DONE-{idx}-{rng.randint(1000, 9999)}"
    plan = ScriptedStep(
        "Let me plan the approach and break this down.",
        None,
        None,
        "",
        80,
        False,
        True,
    )
    extract = ScriptedStep(
        "Extract the relevant fields from the document.",
        "read_file",
        {"path": f"doc{idx}.md"},
        "fields: " + _sentence(rng, 20),
        35,
        False,
        True,
    )
    tool = ScriptedStep(
        "Apply the transformation.",
        "transform",
        {"mode": "x"},
        _sentence(rng, 18),
        40,
        True,
        True,
    )
    verify = ScriptedStep(
        f"Verify the result is correct. Task complete: {token}",
        None,
        None,
        "",
        30,
        False,
        True,
    )
    return Scenario(
        f"normal{idx}",
        "D3",
        "normal_task",
        False,
        None,
        "You are a capable agent.",
        f"Complete task {idx} end to end.",
        [plan, extract, tool, verify],
        success_token=token,
        tools=[
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "transform"}},
        ],
    )


# --------------------------------------------------------------------------- assembly


def build_datasets(seed: int, which: tuple[str, ...] = ("D1", "D2", "D3")) -> list[Scenario]:
    """All scenarios for one seed. Each archetype is parameterized by a per-archetype RNG so the
    whole set is deterministic in `seed`."""

    def rng(tag: str) -> random.Random:
        # String seed -> random.Random hashes it via sha512 deterministically, so the dataset is
        # reproducible ACROSS processes/machines (builtin hash() is salted by PYTHONHASHSEED).
        return random.Random(f"{seed}:{tag}")

    out: list[Scenario] = []
    if "D1" in which:
        out += [
            _tight_loop(rng("tight")),
            _semantic_shift(rng("sem")),
            _meltdown(rng("melt")),
            _long_drift(rng("drift")),
        ]
    if "D2" in which:
        out += [_paginator(rng("pag")), _table_scan(rng("scan")), _batch_format(rng("fmt"))]
    if "D3" in which:
        out += [_normal_task(rng(f"n{i}"), i) for i in range(3)]
    # Stamp the seed into scenario ids so records are unique across seeds.
    for s in out:
        s.scenario_id = f"{s.scenario_id}-s{seed}"
    return out
