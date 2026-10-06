import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, cwd):
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "PINECONE_API_KEY"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_help_works_without_provider_credentials(tmp_path):
    result = cli("--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--topic" in result.stdout


def test_missing_credentials_produce_an_actionable_error(tmp_path):
    result = cli("--topic", "Research topic", cwd=tmp_path)
    assert result.returncode == 1
    assert "OPENAI_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


def test_demo_runs_without_keys_and_exports_markdown(tmp_path):
    output = tmp_path / "reports" / "demo.md"
    result = cli("--demo", "--output", str(output), "--db", str(tmp_path / "state.sqlite"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report = output.read_text()
    assert "Offline demonstration" in report
    assert "https://docs.langchain.com" in report
    assert "##" in report


def test_completed_session_can_be_read_after_restart(tmp_path):
    db = str(tmp_path / "state.sqlite")
    first = cli("--demo", "--thread-id", "portfolio", "--db", db, cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    second = cli("--demo", "--resume", "--thread-id", "portfolio", "--db", db, cwd=tmp_path)
    assert second.returncode == 0, second.stderr
    assert "Offline demonstration" in second.stdout
    assert "Saved report" in second.stderr


def test_resume_of_unknown_session_is_an_actionable_error(tmp_path):
    result = cli(
        "--demo", "--resume", "--thread-id", "missing", "--db", str(tmp_path / "state.sqlite"), cwd=tmp_path
    )
    assert result.returncode == 1
    assert "No saved session" in result.stderr
    assert "Traceback" not in result.stderr


def test_blank_topic_is_rejected(tmp_path):
    result = cli("--topic", "   ", cwd=tmp_path)
    assert result.returncode == 2
    assert "empty" in result.stderr.lower()


def test_demo_report_does_not_cite_skipped_research_steps(tmp_path):
    result = cli("--demo", "--max-steps", "1", "--db", str(tmp_path / "state.sqlite"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "https://docs.langchain.com/oss/python/langgraph/graph-api" in result.stdout
    assert "https://docs.langchain.com/oss/python/langgraph/persistence" not in result.stdout


def test_ingestion_without_credentials_fails_cleanly(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "ingest.py")],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if not key.endswith("API_KEY")},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "PINECONE_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr
