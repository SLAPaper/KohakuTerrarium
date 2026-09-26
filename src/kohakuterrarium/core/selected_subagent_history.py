"""Strict detached subagent projection for selected runtime contexts."""

import json
from copy import deepcopy

from kohakuterrarium.session.history_feedback import subagent_feedback_message


def _name(event):
    value = event.get("name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Selected subagent history requires its recorded name")
    return value


def project_selected_subagents(events, *, allow_pending=False):
    """Validate lifecycle identity and project children through strict tool pairing."""
    projected, calls, terminals, text_results = [], {}, set(), {}
    for original in events:
        kind = original.get("type")
        if kind not in ("subagent_call", "subagent_result"):
            projected.append(original)
            continue
        event = deepcopy(original)
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("Selected subagent history requires a job ID")
        provider = event.get("tool_call_id")
        if provider is not None and (
            not isinstance(provider, str) or not provider.strip()
        ):
            raise ValueError("Selected subagent has invalid provider identity")
        if "call_id" in event and event["call_id"] not in (job_id, provider):
            raise ValueError("Selected subagent call identity disagrees")
        if "tool_name" in event and event["tool_name"] != _name(event):
            raise ValueError("Selected subagent tool name disagrees")
        if kind == "subagent_call":
            if job_id in calls:
                raise ValueError("Selected subagent has duplicate dispatch")
            name = _name(event)
            calls[job_id] = event
            if provider:
                arguments = event.get("tool_call_arguments")
                if not isinstance(arguments, str) or not isinstance(
                    json.loads(arguments), dict
                ):
                    raise ValueError(
                        "Selected native subagent requires recorded arguments"
                    )
            else:
                task = event.get("task")
                if not isinstance(task, str):
                    raise ValueError("Selected text subagent requires recorded task")
                arguments = json.dumps({"task": task}, ensure_ascii=False)
            if "args" in event:
                alias = (
                    json.loads(event["args"])
                    if isinstance(event["args"], str)
                    else event["args"]
                )
                if alias != json.loads(arguments):
                    raise ValueError("Selected subagent argument aliases disagree")
            projected.append(
                {
                    **event,
                    "type": "tool_call",
                    "call_id": provider or job_id,
                    "tool_name": name,
                    "args": arguments,
                }
            )
        else:
            call = calls.get(job_id)
            if call is None or job_id in terminals:
                raise ValueError("Selected subagent has orphan or duplicate result")
            if provider != call.get("tool_call_id"):
                raise ValueError("Selected subagent result provider identity disagrees")
            if _name(event) != _name(call):
                raise ValueError("Selected subagent result name disagrees")
            for key in ("turn_index", "branch_id", "parent_branch_path"):
                if (key in call) != (key in event) or event.get(key) != call.get(key):
                    raise ValueError("Selected subagent result lineage disagrees")
            if "origin" in call or "origin" in event:
                call_origin, result_origin = call.get("origin"), event.get("origin")
                if not isinstance(call_origin, dict) or not isinstance(
                    result_origin, dict
                ):
                    raise ValueError("Selected subagent origin evidence is incomplete")
                for key in (
                    "runtime_scope_id",
                    "delivery_id",
                    "origin_path",
                    "turn_index",
                    "branch_id",
                    "parent_branch_path",
                ):
                    if call_origin.get(key) != result_origin.get(key):
                        raise ValueError("Selected subagent origin identity disagrees")
            terminals.add(job_id)
            projected.append(
                {
                    **event,
                    "type": "tool_result",
                    "call_id": provider or job_id,
                    "tool_name": _name(call),
                    "output": event.get("error") or event.get("output", ""),
                }
            )
            if not provider:
                text_results[job_id] = subagent_feedback_message(event)
    if not allow_pending and set(calls) != terminals:
        raise ValueError(
            "Invalid structured subagent history: pending announcement has incomplete results"
        )
    if allow_pending:
        for job_id in set(calls) - terminals:
            if not calls[job_id].get("tool_call_id"):
                text_results[job_id] = None
    return projected, text_results


def restore_text_subagent_feedback(messages, text_results):
    """Keep bracket subagent feedback textual after strict temporary pairing."""
    restored, pending_feedback = [], []
    for message in messages:
        if message.get("role") != "tool" and pending_feedback:
            restored.extend(pending_feedback)
            pending_feedback.clear()
        if (
            message.get("role") == "tool"
            and message.get("tool_call_id") in text_results
        ):
            feedback = text_results[message["tool_call_id"]]
            if feedback is not None:
                pending_feedback.append(deepcopy(feedback))
            continue
        calls = message.get("tool_calls")
        if calls and any(call.get("id") in text_results for call in calls):
            message = deepcopy(message)
            remaining = [call for call in calls if call.get("id") not in text_results]
            if remaining:
                message["tool_calls"] = remaining
            else:
                message.pop("tool_calls", None)
                if not message.get("content") and not message.get("metadata"):
                    continue
        restored.append(message)
    restored.extend(pending_feedback)
    return restored
