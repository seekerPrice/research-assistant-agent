# Changelog

## 0.2.0 — 2026-10-06

- Rename the repository from `srs_comp_research_agent` to `research-assistant-agent`.
- Keep every research step's evidence and reset it for each new topic.
- Separate retrieval failures from evidence; report when no usable evidence is available.
- Initialize model clients only on request; support either OpenAI or Gemini on its own.
- Preserve source URLs in web, Wikipedia, and arXiv results.
- Add one-shot CLI runs, Markdown export, explicit resume, and a key-free offline demo.
- Bound plan length and normalize provider text blocks.
- Make text/Markdown ingestion create a compatible Pinecone index and upsert stable chunk IDs.
- Add installable commands, a dependency lock, regression tests, CI, and architecture docs.

### CLI changes

`--topic` now runs once and exits. Use `--interactive` for multiple topics.
`--thread_id` remains available as an alias for `--thread-id`.
Use `--resume --thread-id ID` to continue an interrupted run or display its completed report.
