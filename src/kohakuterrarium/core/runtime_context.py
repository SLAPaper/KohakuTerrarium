from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class RuntimeContextPreparationMode(Enum):
    EXACT_INSTALLED_LIVE_REUSE = "exact_installed_live_reuse"
    NORMALIZED_CANONICAL_REPLAY = "normalized_canonical_replay"
    PROVIDER_REPLACEMENT = "provider_replacement"


class RuntimeContextPreparationError(ValueError):
    """Invalid context preparation inputs or proofs."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _coordinate(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _current_coordinates(turn_index: object, branch_id: object) -> tuple[int, int]:
    if (
        isinstance(turn_index, bool)
        or not isinstance(turn_index, int)
        or turn_index < 0
    ):
        raise ValueError("turn_index must be a non-negative integer")
    if isinstance(branch_id, bool) or not isinstance(branch_id, int) or branch_id < 0:
        raise ValueError("branch_id must be a non-negative integer")
    if (turn_index == 0) != (branch_id == 0):
        raise ValueError("current coordinates must both be zero or both positive")
    return turn_index, branch_id


def _path(value: object, current_turn: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("parent_branch_path must be a list or tuple")

    result: list[tuple[int, int]] = []
    previous_turn = 0
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("parent_branch_path entries must be pairs")
        turn = _coordinate(entry[0], "parent turn")
        branch = _coordinate(entry[1], "parent branch")
        if turn <= previous_turn or turn >= current_turn:
            raise ValueError("parent turns must increase and precede current turn")
        result.append((turn, branch))
        previous_turn = turn
    return tuple(result)


def _branch_view(
    value: object,
    turn: int,
    branch: int,
    path: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    expected = () if turn == 0 else (*path, (turn, branch))
    if value is None:
        return expected
    if not isinstance(value, (list, tuple)):
        raise ValueError("branch_view must be a list or tuple")

    result: list[tuple[int, int]] = []
    previous_turn = 0
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("branch_view entries must be pairs")
        selected_turn = _coordinate(item[0], "branch turn")
        selected_branch = _coordinate(item[1], "branch id")
        if selected_turn <= previous_turn or selected_turn > turn:
            raise ValueError(
                "branch_view turns must strictly increase and not exceed current"
            )
        result.append((selected_turn, selected_branch))
        previous_turn = selected_turn

    if tuple(result) != expected:
        raise ValueError("branch_view is incoherent with current coordinates/path")
    return tuple(result)


def _mode(value: object) -> RuntimeContextPreparationMode:
    if not isinstance(value, RuntimeContextPreparationMode):
        raise ValueError("invalid preparation_mode")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeContextPreparation:
    agent_id: str
    runtime_scope_id: str
    provider_conversation: Any = field(compare=False, hash=False, repr=False)
    turn_index: int
    branch_id: int
    parent_branch_path: tuple[tuple[int, int], ...]
    preparation_mode: RuntimeContextPreparationMode
    branch_view: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        _text(self.agent_id, "agent_id")
        _text(self.runtime_scope_id, "runtime_scope_id")
        _current_coordinates(self.turn_index, self.branch_id)
        path = _path(self.parent_branch_path, self.turn_index)
        object.__setattr__(self, "parent_branch_path", path)
        object.__setattr__(
            self,
            "branch_view",
            _branch_view(self.branch_view, self.turn_index, self.branch_id, path),
        )
        _mode(self.preparation_mode)

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        runtime_scope_id: str,
        provider_conversation: object,
        turn_index: int,
        branch_id: int,
        parent_branch_path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        preparation_mode: RuntimeContextPreparationMode,
        branch_view: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    ) -> "RuntimeContextPreparation":
        return cls(
            agent_id=agent_id,
            runtime_scope_id=runtime_scope_id,
            provider_conversation=provider_conversation,
            turn_index=turn_index,
            branch_id=branch_id,
            parent_branch_path=parent_branch_path,
            preparation_mode=preparation_mode,
            branch_view=branch_view,
        )


@dataclass(frozen=True, slots=True)
class RuntimeContextId:
    allocator_namespace: UUID
    ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.allocator_namespace, UUID):
            raise ValueError("allocator_namespace must be UUID")
        _coordinate(self.ordinal, "ordinal")


@dataclass(frozen=True, slots=True)
class RuntimeContextRef:
    context_id: RuntimeContextId
    install_epoch: int
    agent_id: str
    runtime_scope_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.context_id, RuntimeContextId):
            raise ValueError("context_id must be RuntimeContextId")
        _coordinate(self.install_epoch, "install_epoch")
        _text(self.agent_id, "agent_id")
        _text(self.runtime_scope_id, "runtime_scope_id")


@dataclass(frozen=True, slots=True)
class RuntimeContextBundle:
    ref: RuntimeContextRef
    provider_conversation: Any = field(compare=False, hash=False, repr=False)
    turn_index: int
    branch_id: int
    parent_branch_path: tuple[tuple[int, int], ...]
    preparation_mode: RuntimeContextPreparationMode
    branch_view: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, RuntimeContextRef):
            raise ValueError("ref must be RuntimeContextRef")
        _current_coordinates(self.turn_index, self.branch_id)
        path = _path(self.parent_branch_path, self.turn_index)
        object.__setattr__(self, "parent_branch_path", path)
        object.__setattr__(
            self,
            "branch_view",
            _branch_view(self.branch_view, self.turn_index, self.branch_id, path),
        )
        _mode(self.preparation_mode)


class RuntimeContextAllocator:
    """Single-owner allocator for one in-process installation namespace."""

    __slots__ = (
        "_agent_id",
        "_runtime_scope_id",
        "_namespace",
        "_next_ordinal",
        "_next_install_epoch",
    )

    def __copy__(self) -> "RuntimeContextAllocator":
        raise TypeError("RuntimeContextAllocator is non-copyable")

    def __deepcopy__(self, memo: dict[int, object]) -> "RuntimeContextAllocator":
        raise TypeError("RuntimeContextAllocator is non-copyable")

    def __reduce__(self) -> object:
        raise TypeError("RuntimeContextAllocator is not serializable")

    def __init__(self, agent_id: str, runtime_scope_id: str) -> None:
        self._agent_id = _text(agent_id, "agent_id")
        self._runtime_scope_id = _text(runtime_scope_id, "runtime_scope_id")
        self._namespace = uuid4()
        self._next_ordinal = 1
        self._next_install_epoch = 1

    def install(self, preparation: RuntimeContextPreparation) -> RuntimeContextBundle:
        if not isinstance(preparation, RuntimeContextPreparation):
            raise ValueError("preparation must be RuntimeContextPreparation")
        if (
            preparation.agent_id != self._agent_id
            or preparation.runtime_scope_id != self._runtime_scope_id
        ):
            raise ValueError("preparation identity does not match allocator")

        ordinal = self._next_ordinal
        install_epoch = self._next_install_epoch
        self._next_ordinal += 1
        self._next_install_epoch += 1
        context_id = RuntimeContextId(
            allocator_namespace=self._namespace,
            ordinal=ordinal,
        )
        ref = RuntimeContextRef(
            context_id=context_id,
            install_epoch=install_epoch,
            agent_id=self._agent_id,
            runtime_scope_id=self._runtime_scope_id,
        )
        return RuntimeContextBundle(
            ref=ref,
            provider_conversation=preparation.provider_conversation,
            turn_index=preparation.turn_index,
            branch_id=preparation.branch_id,
            parent_branch_path=preparation.parent_branch_path,
            preparation_mode=preparation.preparation_mode,
            branch_view=preparation.branch_view,
        )
