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
share one stored value. Native keys stream in physical order, 64 keys per page.
Payload pages contain exactly one event, so decoding never overlaps multiple
large records. A single event can still exceed any encoded size budget. Later
payload reads fetch only the requested event and retain at most that one decoded
event. These bounds do not cap decoded Python objects or RSS. Reverse turn seeks
retain the namespace's keys before reading bodies; their key memory is O(N).

Live archives use only KohakuVault's native SQLite library. Python SQLite must
not open the same file in that process: the separate libraries can invalidate
WAL and locks on POSIX. Native scans also work in embedded mobile runtimes
without spawning a Python interpreter.

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

Archives produced by experimental recovery-cache builds can contain native
revision guards. Desktop and CLI writable opens remove only those triggers and
revision tables in a standalone stdlib-only process, using the launcher-aware
interpreter command. Newly created files skip this DDL. Read-only and mobile
opens preserve the guards; embedded mobile runtimes cannot spawn Python, and
the guards remain compatible with normal writes. State values remain opaque: their key names can
also identify plugin, run-tracker or child-namespace state. Recovery and normal
checkpoints do not read or update the old cache values; generic archive copying
can still copy them as State. Cleanup does not vacuum the archive.
