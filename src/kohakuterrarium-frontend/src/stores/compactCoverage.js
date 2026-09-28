const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

function originKey(value) {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    !uuid.test(value[0]) ||
    typeof value[1] !== "string" ||
    !value[1] ||
    !Number.isSafeInteger(value[2]) ||
    value[2] <= 0
  ) {
    throw new Error("Invalid compact event origin")
  }
  return JSON.stringify(value)
}

function coordinate(event) {
  const value = event._kt_event_coordinate_v1 ?? event._kt_event_origin_v1
  return value == null ? null : originKey(value)
}

export function compactProjection(events, selected, liveIds) {
  const visible = events.filter((row) => liveIds.has(row.event_id))
  let winner = null
  const terminals = new Set()
  const rounds = new Set()
  for (const event of events) {
    if (!["compact_replace", "compact_complete"].includes(event.type)) continue
    const versioned = Object.hasOwn(event, "compact_coverage_version")
    let origins = null
    let anchor = null
    if (versioned) {
      if (event.compact_coverage_version !== 1)
        throw new Error("Unsupported compact coverage version")
      if (event.coverage_mode === "exact") {
        if (!Array.isArray(event.replaced_event_origins))
          throw new Error("Invalid exact compact coverage")
        origins = new Set(event.replaced_event_origins.map(originKey))
        if (origins.size !== event.replaced_event_origins.length)
          throw new Error("Duplicate compact origin")
        if (origins.size || event.summary_anchor_origin != null)
          anchor = originKey(event.summary_anchor_origin)
      } else if (
        event.coverage_mode !== "legacy_fallback" ||
        Object.hasOwn(event, "replaced_event_origins")
      ) {
        throw new Error("Invalid compact coverage mode")
      }
    } else if (Object.hasOwn(event, "replaced_event_origins")) {
      throw new Error("Compact origins require a coverage version")
    }
    if (origins === null && Object.hasOwn(event, "_kt_event_origin_v1")) continue
    const path =
      Array.isArray(event.compact_path) && event.compact_path.length ? event.compact_path : null
    const rank = path ? (path.every(([turn, branch]) => selected.get(turn) === branch) ? 2 : 0) : 1
    const depth = path?.length || 0
    const first = event.replaced_from_event_id
    const last = event.replaced_to_event_id
    if (origins === null && (!Number.isInteger(first) || !Number.isInteger(last))) continue
    terminals.add(event)
    rounds.add(event.compact_round || event.round || 0)
    const members = new Set(
      visible
        .filter((row) => {
          const matches = origins
            ? origins.has(coordinate(row))
            : row.event_id >= first && row.event_id <= last
          return (
            matches &&
            (!path ||
              path.some(([turn, branch]) => row.turn_index === turn && row.branch_id === branch))
          )
        })
        .map((row) => row.event_id),
    )
    if (!rank || !members.size) continue
    const firstMember = visible.find((row) => members.has(row.event_id))?.event_id
    const anchorId = origins
      ? (visible.find((row) => coordinate(row) === anchor)?.event_id ?? firstMember)
      : firstMember
    if (!winner || rank > winner.rank || (rank === winner.rank && depth >= winner.depth)) {
      winner = { event, members, anchorId, rank, depth }
    }
  }
  return { winner, terminals, rounds }
}
