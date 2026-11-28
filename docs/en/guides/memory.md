---
title: Memory
summary: FTS5 + vector memory over the session store, embedding provider choice, and retrieval patterns.
tags:
  - guides
  - memory
  - embedding
---

# Memory

For readers who want to search past session events from the CLI, from Python, or from an agent at runtime.

A session's event log is also a small local knowledge base. Building a search index over it gives you FTS keyword search (free, fast), semantic search (needs an embedder), and hybrid search that fuses keyword and semantic rankings. Agents can query the current session's memory through the built-in `search_memory` tool.

Concept primer: [memory and compaction](../concepts/modules/memory-and-compaction.md), [sessions](sessions.md).

## What's searchable

Memory projects selected-lineage user input, trigger content, assistant text,
tool calls and nonempty tool results into searchable blocks. It follows the
same persisted branch view as Resume (or the default history selection).
Content eligibility does not depend on whether the Agent is processing:
late background results and synchronous follow-up calls are indexed too.
Distinct error and output payloads share the existing index text budget.
Round/block labels remain compatibility grouping metadata; source Anchors
identify exact physical events.

Search returns `SearchResult` records with:

- `content`: the matched text
- `agent`: which creature produced it
- `block_type`: `text` / `tool` / `trigger` / `user`
- `round_num`, `block_num`: compatibility grouping labels, not exact event locators
- `score`: match quality
- `ts`: timestamp

## Embedding providers

Three providers, pick one that matches your environment:

| Provider | What it needs | Notes |
|---|---|---|
| `model2vec` (default) | no torch, pure NumPy | Extremely fast, smallest install. Quality is good for keyword-adjacent retrieval, weaker for long-form semantic. |
| `sentence-transformer` | `torch` | Slower but much stronger semantic quality. GPU-friendly. |
| `api` | network + API key | Remote embedders (OpenAI, Jina, Gemini). Best quality, pay per call. |
| `auto` | (none) | Prefers `jina-v5-nano` if API available, else falls back to `model2vec`. |

Preset model names (portable across providers):

- `@tiny`: smallest/fastest
- `@base`: default balance
- `@retrieval`: tuned for retrieval
- `@best`: highest quality
- `@multilingual`, `@multilingual-best`: non-English sessions
- `@science`, `@nomic`, `@gemma`: specialized

You can also pass a Hugging Face path directly.

## Build an index

```bash
kt embedding ~/.kohakuterrarium/sessions/swe.kohakutr
```

With explicit options:

```bash
kt embedding swe.kohakutr \
  --provider sentence-transformer \
  --model @best \
  --dimensions 384
```

`--dimensions` is Matryoshka truncation; use it to shrink vectors on the fly when the model supports it.

Incremental: running `kt embedding` again only indexes new events.

## Rebuild an older Session's memory index

Close **all** readers and writers of the Session, including external processes.
Rebuild the whole Session, not individual agents:

```bash
kt memory rebuild SESSION
kt memory rebuild SESSION --dry-run
kt memory rebuild SESSION --vectors drop
kt memory rebuild SESSION --vectors rebuild --provider sentence-transformer --model @best
```

Existing vector rows require an explicit choice (or the interactive prompt):
recompute embeddings, which may download a model or incur API charges, or
drop the old vectors and disable semantic retrieval until re-embedded.
If no vectors exist, the default rebuild is FTS-only and preserves embedding
preferences.

Rebuild restores physical source references and content skipped by old
projection rules. It uses the selected branch view, so an old all-branch
index may become narrower. It does not modify original events, ancestry or
external artifacts. Missing source namespaces, unreadable events and
unsupported archive formats are refused before index replacement.

A verified logical SQLite backup includes prior indexes and committed WAL
data. Rebuild updates the existing database in place; an interrupted update
may leave derived indexes incomplete or invalid. Retry after resolving the
failure, keeping the backup. Do not delete WAL/SHM files manually.

Ordinary searches do not perform this explicit maintenance automatically.
A branch change may refresh the affected index through its normal lifecycle.

## Search from the CLI

```bash
kt search swe "auth bug"                # auto mode (hybrid if vectors exist, else fts)
kt search swe "auth bug" --mode fts     # keyword only
kt search swe "auth bug" --mode semantic
kt search swe "auth bug" --mode hybrid
kt search swe "auth bug" --agent swe -k 5
```

Modes:

- **`fts`**: BM25 over FTS5. No embeddings needed. Fastest. Great for literal phrases.
- **`semantic`**: pure vector similarity. Needs an index. Great for paraphrases.
- **`hybrid`**: Reciprocal-rank fusion of keyword and semantic rankings.
- **`auto`**: picks the richest mode the session supports.

`-k` caps results. `--agent` filters to one creature inside a terrarium session.

## Search from an agent

The built-in `search_memory` tool exposes the same engine to the controller:

```yaml
# creatures/my-agent/config.yaml
tools:
  - read
  - write
  - search_memory
memory:
  embedding:
    provider: model2vec
    model: "@base"
```

When the LLM calls `search_memory`, the tool runs over the *current* session's index. This is the seamless-memory primitive: agents can look up what they (or their teammates) said in earlier rounds without explicit RAG scaffolding.

Live searches keep model loading and encoding off the session's storage thread,
so a stalled embedder does not prevent history reads, event persistence, or graph
shutdown. `fts` never loads or calls an embedder. `auto` and `hybrid` fall back to
keywords if a model operation fails or takes more than 30 seconds; the tool result
reports that fallback. Explicit `semantic` tool calls return an error instead.
The Studio live-search adapter retains its existing semantic-to-FTS fallback and
includes a `warning` field when it degrades.

Only one model operation may remain outstanding per live store. Cancelling or
timing out a search does not forcibly terminate a native model call; later searches
use keywords while it remains busy. The isolated daemon worker owns no database
handles and cannot write late results into a closed session. Session database work
still drains in its original serial order. Keyword and vector indexing keep
separate progress, allowing semantic indexing to catch up after keyword searches.

Tool args (shape; concrete syntax depends on your `tool_format`; default bracket shown):

```
[/search_memory]
@@query=auth bug
@@mode=hybrid
@@k=5
@@agent=swe
[search_memory/]
```

### Navigate a search result

Pass a returned opaque anchor to `search_memory(anchor=..., before=3, after=3)`.
Follow neighboring anchors to walk through events, or use the returned
`next_offset` with the same anchor and scope to read a long event.

- `scope="selected"` (default) follows the anchor's pinned branch view.
  Unknown neighbors are skipped; a focus of unknown lineage requires raw scope.
- `scope="raw"` reads physical neighbors of the same agent, including other
  branches, with recorded turn/branch coordinates. Query-mode search remains
  selected-lineage only.

The tool excludes its own call/result echoes from ranked retrieval. Budget
exhaustion yields available hits or context with an incomplete-results notice,
not proof that no more evidence exists. Legacy hits without physical source
references remain searchable but need explicit rebuild for anchors.

For RAG over *external* sources, build a custom tool or a [plugin](plugins.md) that calls your vector store.

## Configuring memory in a creature

```yaml
memory:
  embedding:
    provider: model2vec       # or sentence-transformer, api, auto
    model: "@retrieval"       # preset or HF path
```

Live searches refresh the current agent's index as needed; FTS does not require an embedding provider.

## Inspecting programmatically

```python
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.memory.index import SessionMemory
from kohakuterrarium.session.memory.source_events import sourced_events

store = SessionStore("~/.kohakuterrarium/sessions/swe.kohakutr")
memory = SessionMemory(store.path, store=store)

memory.index_events("swe", sourced_events(store, "swe"))
results = memory.search("refactor", mode="fts", k=5)
for r in results:
    print(r.agent, r.source_ref, r.content[:120])

memory.close()
store.close()
```

## Troubleshooting

- **`No vectors in index`.** You ran `--mode semantic` without first running `kt embedding`. Either build the index or use `--mode fts`.
- **Slow `kt embedding`.** `sentence-transformer` is CPU-bound by default. Install torch with CUDA, or drop to `model2vec`.
- **Provider install fails.** `kt embedding --provider model2vec` has no native deps and works everywhere. `sentence-transformer` needs `torch`; `api` needs the provider SDK (`openai`, `google-generativeai`, etc.).
- **Hybrid mode returns noise.** Lower `-k` and prefer `semantic` over `hybrid` for paraphrase-heavy queries; prefer `fts` for literal-phrase queries.
- **Missing hits or anchors in an old index.** Try FTS and inspect the selected branch. Use `kt memory rebuild SESSION` to restore historical coverage or source metadata; incomplete notices and text caps can also limit results.

## See also

- [Sessions](sessions.md): the `.kohakutr` format that memory sits on top of.
- [Plugins](plugins.md): seamless-memory plugin pattern (`pre_llm_call` retrieval).
- [Reference / CLI](../reference/cli.md): `kt embedding`, `kt search`, and `kt memory rebuild` flags.
- [Concepts / memory and compaction](../concepts/modules/memory-and-compaction.md): the design rationale.
