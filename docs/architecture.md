# Architecture

The application is a three-stage LangGraph workflow with explicit state and a small CLI.
Its purpose is to make planning, retrieval, synthesis, and recovery easy to inspect and test.

## Modules

| Module | Responsibility |
| --- | --- |
| `main.py` | Parse arguments, manage SQLite lifetime, resume sessions, render progress, export reports. |
| `graph.py` | Construct the graph and select live or demo dependencies. |
| `nodes.py` | Plan queries, collect evidence through tool calls, and synthesize a report. |
| `tools.py` | Read-only retrieval with source URLs, timeouts, and optional document support. |
| `state.py` | Describe the checkpointed state. |
| `ingest.py` | Load text/Markdown, split documents, validate/create a Pinecone index, upsert chunks. |
| `demo.py` | Provide fixed model responses and illustrative evidence without network access. |

## State and data flow

`AgentState` contains `topic`, `plan`, `current_step`, `findings`, `errors`, `messages`,
`answer`, `max_steps`, and `demo`.

1. **Planner:** uses a bounded slice of conversation history, asks for at most `max_steps`
   queries, trims enumeration prefixes, removes duplicates, and caps the result. It resets
   findings, errors, the answer, and the step counter for the new topic.
2. **Researcher:** asks a configured model to choose available tools, executes at most four
   tool calls for the current query, appends successful evidence to the existing findings,
   and increments `current_step`. Errors are stored separately. Local retrieval is offered
   only when the required embedding/Pinecone keys exist.
3. **Responder:** synthesizes all accumulated evidence into Markdown, asking for actual
   source links and a distinction between facts and inference. With no evidence it returns
   an explicit failure report rather than asking a model to fill the gap. Retrieval errors
   are appended as limitations.

Findings use normal overwrite semantics in the state schema. Each research update explicitly
returns the previous findings plus the new ones. This supports accumulation during a plan
and an uncomplicated reset between topics. Conversation messages use an append reducer.

## Persistence and recovery

The CLI owns a `SqliteSaver.from_conn_string(...)` context and closes its connection after
use. Every graph step is checkpointed under the selected session ID. Programmatic callers
can pass a checkpointer to `create_graph`; the default is an in-memory saver.

`--resume` inspects the saved state. Pending nodes continue with `app.stream(None, config)`;
a completed session displays its stored report without repeating searches. Demo/live mode
is saved and checked on resume. Starting a new topic with the same ID retains bounded
conversation context but resets evidence for that topic.

A failure during planning or synthesis is allowed to propagate, preserving a pending
checkpoint for a later retry. Retrieval failures are recorded and the plan continues.
An external request completed before a process died may repeat if its result was not yet
checkpointed. SQLite is suitable for the local CLI's single-user workload.

## Model and retrieval dependencies

Generation clients are created only when needed. OpenAI is preferred when its key exists;
Gemini is attempted next if configured. Google-only operation is supported. Failures across
all configured providers produce an actionable error. Provider text blocks are normalized
using LangChain's message text interface.

The live tools use DDGS for web snippets, MediaWiki's API for page introductions, and arXiv's
Atom API for abstracts. Results retain canonical links. They do not fetch arbitrary pages
or download PDFs. Web calls use a 10-second timeout; Wikipedia/arXiv requests use 15 seconds.
Model requests and embeddings have 30-second timeouts; provider clients allow one retry.
These are per-request limits, not a deadline for the entire research run.

Pinecone dependencies are optional and imported when document support is used. Ingestion
creates a missing serverless index, waits up to 60 seconds for readiness, and checks its
embedding dimension. Stable path/chunk IDs update existing chunks on repeated ingestion.
Old chunks from deleted files or shorter documents are not automatically pruned.

## Testing and the demo

Live and demo execution share all three nodes and graph edges. The demo injects a model
callable and a retrieval function at the existing dependency seams. It replaces external
calls, while preserving state transitions, evidence aggregation, checkpointing, and export.
The report identifies the evidence as fixed illustrative excerpts and includes only sources
from executed steps.

Regression tests cover lost findings, new-topic reset, retrieval failure handling, unsupported
model knowledge, plan limits, provider text blocks, fallback, CLI validation/export, completed
and interrupted sessions, source URLs, and optional document ingestion. External services
are replaced so the suite is deterministic and does not spend API credits.

## Tradeoffs

- Plain modules keep the prototype easy to navigate. There is no plugin framework or server.
- A fixed plan followed by sequential retrieval is understandable and bounded; it does not
  iteratively decide when research is sufficient or independently fact-check an answer.
- Linked snippets improve traceability, but do not prove the generated claims or citations.
- Prompt instructions treat source text as untrusted data. They reduce accidental instruction
  following but do not create a hardened security boundary.
- Provider/Pinecone integration is covered at its interfaces in offline tests; real account
  credentials, quotas, permissions, and service availability still require live validation.
