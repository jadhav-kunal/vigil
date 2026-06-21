"""Smoke tests for the Node CLI (`vigil init | prompt | demo`). They only assert the offline
commands produce the expected copy-paste output; the demo's full loop needs a running proxy and is
verified manually. Skipped cleanly if Node isn't on PATH."""

import shutil
import subprocess
from pathlib import Path

import pytest

CLI = Path(__file__).resolve().parents[3] / "cli" / "vigil.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _run(*args: str) -> str:
    out = subprocess.run(["node", str(CLI), *args], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_cli_file_exists():
    assert CLI.exists()


def test_init_prints_base_url_integration():
    out = _run("init")
    assert "http://localhost:8765/v1" in out  # OpenAI base_url
    assert "http://localhost:8765" in out  # Anthropic base_url
    assert "/health" in out


def test_init_respects_custom_base_url():
    out = _run("init", "--base-url", "http://vigil.internal:9000")
    assert "http://vigil.internal:9000/v1" in out


def test_prompt_is_copy_paste_agent_instructions():
    out = _run("prompt")
    assert "VIGIL_BASE_URL" in out
    assert "x-vigil-session-id" in out
    assert "passes it straight through" in out  # keys are never stored
