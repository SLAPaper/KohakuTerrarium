import copy
import pickle
from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from kohakuterrarium.core.runtime_context import (
    RuntimeContextAllocator,
    RuntimeContextBundle,
    RuntimeContextId,
    RuntimeContextPreparation,
    RuntimeContextPreparationMode,
    RuntimeContextRef,
)

MODE = RuntimeContextPreparationMode.EXACT_INSTALLED_LIVE_REUSE


def make_preparation(**overrides) -> RuntimeContextPreparation:
    values = {
        "agent_id": "agent",
        "runtime_scope_id": "session",
        "provider_conversation": object(),
        "turn_index": 3,
        "branch_id": 2,
        "parent_branch_path": [(1, 1), (2, 1)],
        "preparation_mode": MODE,
    }
    values.update(overrides)
    return RuntimeContextPreparation.create(**values)


def make_ref(*, ordinal: int = 1) -> RuntimeContextRef:
    return RuntimeContextRef(
        context_id=RuntimeContextId(
            allocator_namespace=uuid4(),
            ordinal=ordinal,
        ),
        install_epoch=ordinal,
        agent_id="agent",
        runtime_scope_id="session",
    )


def make_bundle(**overrides) -> RuntimeContextBundle:
    values = {
        "ref": make_ref(),
        "provider_conversation": object(),
        "turn_index": 3,
        "branch_id": 2,
        "parent_branch_path": [(1, 1), (2, 1)],
        "preparation_mode": MODE,
    }
    values.update(overrides)
    return RuntimeContextBundle(**values)


def test_preparation_mode_has_exact_values():
    assert [(mode.name, mode.value) for mode in RuntimeContextPreparationMode] == [
        ("EXACT_INSTALLED_LIVE_REUSE", "exact_installed_live_reuse"),
        ("NORMALIZED_CANONICAL_REPLAY", "normalized_canonical_replay"),
        ("PROVIDER_REPLACEMENT", "provider_replacement"),
    ]


def test_zero_state_is_valid_only_with_empty_path():
    preparation = make_preparation(turn_index=0, branch_id=0, parent_branch_path=[])
    bundle = make_bundle(turn_index=0, branch_id=0, parent_branch_path=[])
    assert preparation.parent_branch_path == ()
    assert bundle.parent_branch_path == ()


@pytest.mark.parametrize("turn_index, branch_id", [(0, 1), (1, 0), (-1, 0), (0, -1)])
def test_preparation_rejects_mixed_or_negative_current_coordinates(
    turn_index, branch_id
):
    with pytest.raises(ValueError):
        make_preparation(
            turn_index=turn_index, branch_id=branch_id, parent_branch_path=[]
        )


def test_zero_state_rejects_nonempty_path():
    with pytest.raises(ValueError):
        make_preparation(turn_index=0, branch_id=0, parent_branch_path=[(1, 1)])


def test_preparation_normalizes_path_and_preserves_provider_identity():
    provider = object()

    result = make_preparation(
        provider_conversation=provider,
        parent_branch_path=[[1, 2], (2, 3)],
    )

    assert result.parent_branch_path == ((1, 2), (2, 3))
    assert result.provider_conversation is provider


@pytest.mark.parametrize("field", ["agent_id", "runtime_scope_id"])
@pytest.mark.parametrize("value", [None, 1, "", "  "])
def test_preparation_rejects_invalid_identity(field, value):
    with pytest.raises(ValueError):
        make_preparation(**{field: value})


@pytest.mark.parametrize("field", ["turn_index", "branch_id"])
@pytest.mark.parametrize("value", [True, "1", -1])
def test_preparation_rejects_invalid_current_coordinates(field, value):
    with pytest.raises(ValueError):
        make_preparation(**{field: value})


@pytest.mark.parametrize(
    "path",
    [
        "not-a-path",
        {"turn": 1, "branch": 1},
        [(1,)],
        [(1, 1, 1)],
        ["12"],
        [(True, 1)],
        [("1", 1)],
        [(0, 1)],
        [(-1, 1)],
        [(1, True)],
        [(1, "1")],
        [(1, 0)],
        [(1, -1)],
        [(1, 1), (1, 2)],
        [(2, 1), (1, 2)],
        [(3, 1)],
        [(4, 1)],
    ],
)
def test_preparation_rejects_invalid_parent_paths(path):
    with pytest.raises(ValueError):
        make_preparation(parent_branch_path=path)


def test_preparation_rejects_invalid_mode():
    with pytest.raises(ValueError):
        make_preparation(preparation_mode=MODE.value)


def test_direct_preparation_validates_and_normalizes():
    result = RuntimeContextPreparation(
        agent_id="agent",
        runtime_scope_id="scope",
        provider_conversation=object(),
        turn_index=2,
        branch_id=1,
        parent_branch_path=[[1, 2]],
        preparation_mode=MODE,
    )

    assert result.runtime_scope_id == "scope"
    assert result.parent_branch_path == ((1, 2),)


@pytest.mark.parametrize(
    "overrides",
    [
        {"agent_id": ""},
        {"runtime_scope_id": None},
        {"turn_index": True},
        {"branch_id": 0},
        {"parent_branch_path": [(3, 1)]},
        {"preparation_mode": "invalid"},
    ],
)
def test_direct_preparation_cannot_bypass_validation(overrides):
    with pytest.raises(ValueError):
        make_preparation(**overrides)


@pytest.mark.parametrize("field", ["agent_id", "runtime_scope_id"])
@pytest.mark.parametrize("value", [None, 1, "", "  "])
def test_allocator_rejects_invalid_identity(field, value):
    values = {"agent_id": "agent", "runtime_scope_id": "session"}
    values[field] = value

    with pytest.raises(ValueError):
        RuntimeContextAllocator(**values)


def test_allocator_uses_monotonic_ordinals_and_one_namespace():
    allocator = RuntimeContextAllocator("agent", "session")

    first = allocator.install(make_preparation())
    second = allocator.install(make_preparation())

    assert first.ref.context_id.ordinal == 1
    assert first.ref.install_epoch == 1
    assert second.ref.context_id.ordinal == 2
    assert second.ref.install_epoch == 2
    assert (
        first.ref.context_id.allocator_namespace
        == second.ref.context_id.allocator_namespace
    )
    assert first.ref.context_id != second.ref.context_id


def test_allocators_use_different_namespaces():
    first = RuntimeContextAllocator("agent", "session").install(make_preparation())
    second = RuntimeContextAllocator("agent", "session").install(make_preparation())

    assert (
        first.ref.context_id.allocator_namespace
        != second.ref.context_id.allocator_namespace
    )


def test_invalid_install_does_not_consume_ordinal():
    allocator = RuntimeContextAllocator("agent", "session")

    with pytest.raises(ValueError):
        allocator.install(object())
    with pytest.raises(ValueError):
        allocator.install(make_preparation(agent_id="other"))

    installed = allocator.install(make_preparation())

    assert installed.ref.context_id.ordinal == 1


def test_provider_is_opaque_for_equality_repr_and_hash():
    class Unhashable:
        __hash__ = None

    first = make_preparation(provider_conversation=Unhashable())
    second = make_preparation(provider_conversation=Unhashable())
    assert first == second
    assert "provider_conversation" not in repr(first)
    assert hash(first) == hash(second)

    first_bundle = make_bundle(provider_conversation=Unhashable())
    second_bundle = make_bundle(
        ref=first_bundle.ref,
        provider_conversation=Unhashable(),
    )
    assert first_bundle == second_bundle
    assert "provider_conversation" not in repr(first_bundle)
    assert hash(first_bundle) == hash(second_bundle)


def test_allocator_is_noncopyable():
    allocator = RuntimeContextAllocator("agent", "session")
    with pytest.raises(TypeError):
        copy.copy(allocator)
    with pytest.raises(TypeError):
        copy.deepcopy(allocator)


def test_allocator_is_not_serializable():
    allocator = RuntimeContextAllocator("agent", "session")

    with pytest.raises(TypeError):
        pickle.dumps(allocator)


def test_bundle_preserves_provider_preparation_and_ref_identity():
    provider = object()
    prepared = make_preparation(provider_conversation=provider)

    bundle = RuntimeContextAllocator("agent", "session").install(prepared)

    assert bundle.ref.agent_id == "agent"
    assert bundle.ref.runtime_scope_id == "session"
    assert bundle.provider_conversation is provider
    assert bundle.turn_index == prepared.turn_index
    assert bundle.branch_id == prepared.branch_id
    assert bundle.parent_branch_path == prepared.parent_branch_path
    assert bundle.preparation_mode is prepared.preparation_mode
    assert not hasattr(bundle, "agent_id")
    assert not hasattr(bundle, "runtime_scope_id")


def test_preparation_is_frozen():
    prepared = make_preparation()

    with pytest.raises(FrozenInstanceError):
        prepared.turn_index = 4


def test_context_id_is_frozen():
    context_id = RuntimeContextId(allocator_namespace=uuid4(), ordinal=1)

    with pytest.raises(FrozenInstanceError):
        context_id.ordinal = 2


def test_context_ref_is_frozen():
    ref = make_ref()

    with pytest.raises(FrozenInstanceError):
        ref.install_epoch = 2


def test_bundle_is_frozen():
    bundle = make_bundle()

    with pytest.raises(FrozenInstanceError):
        bundle.branch_id = 3


@pytest.mark.parametrize(
    ("namespace", "ordinal"),
    [
        ("not-a-uuid", 1),
        (uuid4(), True),
        (uuid4(), "1"),
        (uuid4(), 0),
        (uuid4(), -1),
    ],
)
def test_direct_context_id_rejects_invalid_values(namespace, ordinal):
    with pytest.raises(ValueError):
        RuntimeContextId(allocator_namespace=namespace, ordinal=ordinal)


def test_direct_context_ref_accepts_differing_ordinal_and_epoch():
    ref = make_ref(ordinal=7)
    ref = RuntimeContextRef(
        context_id=ref.context_id,
        install_epoch=3,
        agent_id="agent",
        runtime_scope_id="session",
    )
    assert ref.context_id.ordinal == 7
    assert ref.install_epoch == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_id": object()},
        {"install_epoch": True},
        {"install_epoch": "1"},
        {"install_epoch": 0},
        {"install_epoch": -1},
        {"agent_id": None},
        {"agent_id": " "},
        {"runtime_scope_id": 1},
        {"runtime_scope_id": ""},
    ],
)
def test_direct_context_ref_rejects_invalid_values(overrides):
    values = {
        "context_id": RuntimeContextId(uuid4(), 1),
        "install_epoch": 1,
        "agent_id": "agent",
        "runtime_scope_id": "session",
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        RuntimeContextRef(**values)


def test_direct_bundle_normalizes_path():
    bundle = make_bundle(parent_branch_path=[[1, 2], (2, 3)])

    assert bundle.parent_branch_path == ((1, 2), (2, 3))


@pytest.mark.parametrize(
    "overrides",
    [
        {"ref": object()},
        {"turn_index": True},
        {"turn_index": "3"},
        {"turn_index": 0},
        {"turn_index": -1},
        {"branch_id": True},
        {"branch_id": "2"},
        {"branch_id": 0},
        {"branch_id": -1},
        {"parent_branch_path": "invalid"},
        {"parent_branch_path": [(3, 1)]},
        {"preparation_mode": "invalid"},
    ],
)
def test_direct_bundle_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        make_bundle(**overrides)


def test_branch_view_defaults_to_parent_path_and_current_coordinate():
    preparation = make_preparation()
    bundle = RuntimeContextAllocator("agent", "session").install(preparation)

    assert preparation.branch_view == ((1, 1), (2, 1), (3, 2))
    assert bundle.branch_view == preparation.branch_view


def test_branch_view_normalizes_explicit_mixed_sequences():
    preparation = make_preparation(
        branch_view=[[1, 1], (2, 1), [3, 2]],
    )

    assert preparation.branch_view == ((1, 1), (2, 1), (3, 2))


@pytest.mark.parametrize(
    "branch_view",
    [
        "invalid",
        [(1,)],
        [(True, 1), (2, 1), (3, 2)],
        [(1, "1"), (2, 1), (3, 2)],
        [(0, 1), (2, 1), (3, 2)],
        [(1, 1), (1, 2), (3, 2)],
        [(2, 1), (1, 1), (3, 2)],
        [(1, 1), (2, 1)],
        [(1, 1), (2, 1), (3, 9)],
        [(1, 9), (2, 1), (3, 2)],
        [(1, 1), (2, 9), (3, 2)],
        [(1, 1), (2, 1), (4, 2)],
    ],
)
def test_branch_view_rejects_invalid_or_incoherent_values(branch_view):
    with pytest.raises(ValueError):
        make_preparation(branch_view=branch_view)


def test_zero_state_branch_view_must_be_empty():
    zero = make_preparation(
        turn_index=0,
        branch_id=0,
        parent_branch_path=[],
    )
    assert zero.branch_view == ()

    with pytest.raises(ValueError):
        make_preparation(
            turn_index=0,
            branch_id=0,
            parent_branch_path=[],
            branch_view=[(1, 1)],
        )


def test_branch_view_participates_in_bundle_equality_hash_and_freezing():
    first = make_bundle()
    same = make_bundle(ref=first.ref)

    assert first == same
    assert hash(first) == hash(same)
    with pytest.raises(FrozenInstanceError):
        first.branch_view = ()
