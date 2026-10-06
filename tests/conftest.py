import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Keep test credentials and checkpoints separate from user configuration."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
