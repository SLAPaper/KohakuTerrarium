"""Pure lineage admission for local background completion consumption."""


def _path(value):
    if not isinstance(value, (list, tuple)):
        return None
    result = []
    previous = 0
    for pair in value:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(part) is not int or part < 1 for part in pair)
            or pair[0] <= previous
        ):
            return None
        result.append(list(pair))
        previous = pair[0]
    return result


def has_background_provenance(event):
    """Identify completion envelopes carrying explicit lineage or classification."""
    return event.type in ("tool_complete", "subagent_output") and any(
        key in (event.context or {}) for key in ("origin", "execution_class")
    )


def completion_identity(event, scope):
    """Return valid origin path and optional full delivery identity."""
    context = event.context or {}
    origin = context.get("origin")
    if not isinstance(origin, dict) or not isinstance(scope, str) or not scope:
        return None
    if (
        origin.get("runtime_scope_id") != scope
        or origin.get("execution_class") != "background"
        or context.get("execution_class", "background") != "background"
    ):
        return None
    path = _path(origin.get("origin_path"))
    if not path:
        return None
    fields = ("delivery_id", "turn_index", "branch_id", "parent_branch_path")
    if not any(key in origin for key in fields):
        return path, None
    if (
        type(origin.get("turn_index")) is not int
        or type(origin.get("branch_id")) is not int
        or path[-1] != [origin["turn_index"], origin["branch_id"]]
        or _path(origin.get("parent_branch_path")) != path[:-1]
        or not isinstance(event.job_id, str)
        or not event.job_id
        or origin.get("delivery_id") != f"{scope}:{event.job_id}"
    ):
        return None
    return path, origin["delivery_id"]


def lineage_accepts(agent, origin_path):
    """Accept the dispatch branch and its descendants, never ancestors or siblings."""
    parents = _path(getattr(agent, "_parent_branch_path", None))
    if parents is None:
        return False
    current = _path(
        [
            *parents,
            [getattr(agent, "_turn_index", None), getattr(agent, "_branch_id", None)],
        ]
    )
    return current is not None and current[: len(origin_path)] == origin_path
