---
name: search_memory
description: 'Search this session''s history by keyword or meaning. Use it to explore anchors; scope=raw reads physical history. Not for files - use grep.'
category: builtin
tags: [memory, search]
---

# search_memory

Searches the recorded event log of the current session.

## Arguments

| Arg | Type | Req | Description |
| --- | --- | --- | --- |
| query | string | conditional | Search text; supply exactly one of query or anchor |
| anchor | string | conditional | Opaque cursor from a preceding search/context window |
| before / after | integer | no | At most 10 neighboring events on either side (default 3 each) |
| offset | integer | no | Continue reading a long anchored event from a character offset |
| scope | string | no | Anchor context only: `selected` (default) or `raw` |
| k | integer | no | Maximum results (default 5) |
| mode | string | no | `auto` (default), `fts`, `semantic`, or `hybrid` |
| agent | string | no | Filter results by agent name |

## Behavior

- Search the current session's selected lineage, excluding search echoes.
- `fts` needs no model. `auto` and `hybrid` fall back to FTS on embedding
  failure or timeout; `semantic` reports an error.
- Use returned Anchors to inspect exact source events.

## Limits

- Context is bounded; incomplete results carry a notice.

## Reference

Recorded nonempty user/trigger input, assistant text, tool calls and tool
results are indexed independently of processing state. Distinct error/output
payloads contribute to the bounded indexed text. Search call/result echoes
are excluded from ranking but remain recorded. Embedding work does not block
session storage.

Anchors are process-local and may expire (up to 2048 cached). A window returns
up to 10 neighbors per side, with 900 characters per event slice.

Call `search_memory(anchor=..., before=3, after=3)` to inspect the exact
physical event and its neighbors. Follow returned Anchors to continue walking.
For a long event, pass its Anchor and `next_offset` as `offset` with
`before=0, after=0`; keep the same scope.

- `scope="selected"` (default) follows the branch view pinned at Anchor
  issuance. Unknown neighbors are skipped with an incomplete-context notice.
  An unknown or excluded focus requires raw scope.
- `scope="raw"` reads physical neighbors of the same agent, including other
  branches, with recorded turn/branch coordinates. It does not broaden query
  mode or bypass source verification.

Each direction examines at most 256 physical key positions. Legacy ancestry
inference is limited to the first 256 positions. `has_before=unknown` or
`has_after=unknown` means continuity could not be established within budget.
A verified Anchor never substitutes a different event for a missing source.

FTS and semantic candidate budgets are 8192 and 4096 respectively. If filtering
exhausts a budget, available ranked hits are returned with an incomplete-results
warning; zero hits then does not establish absence from the history. Long
indexed text may also be capped; inspect the source with an Anchor.

Legacy index hits without physical source references remain searchable but
cannot issue verified Anchors. Explicit offline `kt memory rebuild SESSION`
restores source metadata and previously omitted content. Invalid or stale
Anchors return an error; raw scope does not repair them.
