# Recovery reads

`read_restore_events()` builds temporary `EventDescriptor` mappings. Each
descriptor owns small scalar fields, shared ancestry, and a physical payload
locator. Nested values and long strings stay in the archive. Consumers borrow
these mappings read-only; `dict(event)` materializes one complete event.

Descriptors also retain a per-operation comparison hash. Different hashes avoid
rereading bodies during adjacent deduplication; equal hashes still compare full
signatures. Field names and event types share interned strings. No hash is
persisted or used to authenticate checkpoints.

The first pass still decodes historical records because metadata and bodies
share one stored value. SQL locators stream in physical order. Pages contain
at most 64 records and 1 MiB of encoded values, except that one oversized value
occupies its own page. Later payload reads fetch only the requested event and
retain at most that one decoded event. The encoded size budget bounds batches,
not decoded Python objects or RSS.

Existing replay planning resolves branches, compaction, canonical assistant
rounds and late ancestor tool results over descriptors. Body reads can revisit
events. This is not a single-pass fold or a cross-table database snapshot.
Cold recovery runs on the owning Store worker before runtime publication.
External writes during that operation are outside this consistency contract.
Each new operation rebuilds its descriptors; no persistent receipt bypasses
historical selection. Conversation checkpoints retain their existing authority,
including snapshot-only messages and summaries.

Checkpoint selection validates candidate bodies one at a time, retaining only
the best header and rank, then reloads the winner. Legacy inline catalogs still
require decoding one complete catalog value. Version 2 catalogs reference
independent message payloads. The latest conversation mirror remains compatible
with older readers.

After context installation, deferred output history retains only namespace,
logical watermark and physical append boundary, including opaque legacy keys.
Stdout prepares bounded previews from recorded display events on the Store
worker, without adding synthetic recovery announcements or interruptions.
Other projections retain recovery normalization by default. Legacy callbacks
and terminal consumers explicitly requesting complete event lists materialize
those lists and pay their memory cost. Normal checkpoints save snapshots and
runtime state without a third recovery context or historical authentication.
