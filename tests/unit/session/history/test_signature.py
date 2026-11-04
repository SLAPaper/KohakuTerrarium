"""Path-signature reuse preserves legacy collisions and iterator isolation."""

import json

from kohakuterrarium.session.history.ancestry import BranchAncestry, SharedPath
from kohakuterrarium.session.history.replay import _event_signature_value
from kohakuterrarium.session.history.signature import SignatureValues


def test_repeated_owned_path_encodes_once_without_merging_different_trees():
    calls = []

    def encode(value):
        calls.append(value)
        return _event_signature_value(value)

    values = SignatureValues(encode)
    tree = BranchAncestry()
    path = tree.view(tree.intern([[1, 1]]))
    expected = "[[1,1]]"
    assert values(path) == values(SharedPath(tree, path.leaf)) == expected
    tree.intern([[1, 1], [2, 1]])
    assert values(path) == expected
    other = BranchAncestry()
    sibling = other.view(other.intern([[1, 2]]))
    assert values(sibling) == "[[1,2]]"
    assert len(calls) == 2


def test_cache_evicts_by_entry_count_and_encoded_byte_budget():
    tree = BranchAncestry()
    values = SignatureValues(_event_signature_value, path_limit=2, byte_limit=24)
    paths = [tree.view(tree.intern([[1, branch]])) for branch in range(1, 5)]
    for path in paths:
        assert values(path) == _event_signature_value(path)
        assert len(values._paths) <= 2
        assert values._bytes <= 24
    assert (tree, paths[0].leaf) not in values._paths
    long_path = tree.view(tree.intern([[turn, 1] for turn in range(1, 100)]))
    assert values(long_path) == _event_signature_value(long_path)
    assert (tree, long_path.leaf) not in values._paths
    assert values._bytes <= 24


def test_mutable_values_and_custom_paths_keep_fresh_legacy_signatures():
    class CustomPath(SharedPath):
        content = [["custom", False]]

        def __iter__(self):
            yield from self.content

    values = SignatureValues(_event_signature_value)
    raw = [[1, 1]]
    assert values(raw) == "[[1,1]]"
    raw[0][1] = 2
    assert values(raw) == "[[1,2]]"
    custom = CustomPath(BranchAncestry(), 0)
    assert values(custom) == '[["custom",false]]'
    custom.content[:] = [[2, 3]]
    assert values(custom) == "[[2,3]]"
    assert not values._paths
    assert values(((1, 1),)) == ((1, 1),)
    assert values("[[1,1]]") == "[[1,1]]"
    assert values({"text": "中文"}) == json.dumps(
        {"text": "中文"}, sort_keys=True, separators=(",", ":")
    )


def test_encoded_byte_limit_applies_before_the_entry_limit():
    tree = BranchAncestry()
    values = SignatureValues(_event_signature_value, path_limit=32, byte_limit=12)
    for branch in range(1, 5):
        assert values(tree.view(tree.intern([[1, branch]]))) == f"[[1,{branch}]]"
    assert len(values._paths) == 1
    unicode_values = SignatureValues(lambda path: "中文", byte_limit=4)
    assert unicode_values(tree.view(1)) == "中文"
    assert not unicode_values._paths
    disabled = SignatureValues(_event_signature_value, path_limit=0)
    assert disabled(tree.view(1)) == "[[1,1]]"
    assert not disabled._paths
