# Contributing

Use Python 3.11 or newer. The test suite runs without real API keys or paid services.

```bash
uv sync --locked --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run research-assistant --demo --db /tmp/research-demo.sqlite
```

For a bug fix, add a test that demonstrates the broken behavior before changing the code.
Replace external model and retrieval calls in tests, while keeping the graph, SQLite,
document processing, and CLI behavior real. Never put credentials or personal documents
in test fixtures.

Keep the planner → researcher → responder workflow readable. A new retrieval tool should
return source links or local document paths, raise on failures, and declare its query input
with a type annotation and a useful docstring.

Update the README and changelog for user-visible changes. Describe the behavior and the
checks run in your pull request. Live provider and Pinecone tests should be opt-in;
they may make paid requests and must not run in ordinary CI.
