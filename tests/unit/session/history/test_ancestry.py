"""Compact ancestry shares prefixes without trusting malformed references."""

from copy import copy, deepcopy

import pytest

from kohakuterrarium.session.history.ancestry import BranchAncestry, SharedPath


def test_prefixes_share_nodes_and_round_trip():
    tree = BranchAncestry()
    a = tree.intern([[1, 1], [2, 1], [3, 2]])
    b = tree.intern([[1, 1], [2, 1], [3, 3]])
    assert len(tree.nodes) == 4
    assert tree.nodes[a - 1][0] == tree.nodes[b - 1][0]
    assert BranchAncestry(tree.nodes).path(b) == [[1, 1], [2, 1], [3, 3]]
    assert tree.path(0) == []
    for turn in range(4, 1004):
        b = tree.add(b, turn, 1)
    assert len(tree.nodes) == 1004
    assert len(tree.path(b)) == 1003


@pytest.mark.parametrize(
    "nodes",
    [
        [[1, 1, 1]],
        [[0, 0, 1]],
        [[0, True, 1]],
        [[0, 1]],
        [[0, 1, 1], [1, 1, 2]],
        [[0, 1, 1], [0, 1, 1]],
    ],
)
def test_invalid_nodes_rejected(nodes):
    with pytest.raises(ValueError):
        BranchAncestry(nodes)


@pytest.mark.parametrize(
    "path", [[[1, 1], [1, 2]], [[0, 1]], [[1, -1]], [[True, 1]], [[1]]]
)
def test_invalid_paths_rejected(path):
    with pytest.raises(ValueError):
        BranchAncestry().intern(path)
    with pytest.raises(ValueError):
        BranchAncestry().intern_with_shape(path)


@pytest.mark.parametrize(
    "path,canonical",
    [
        ([], True),
        ([[1, 1], [2, 3]], True),
        ([(1, 1), [2, 3]], False),
        (((1, 1), (2, 3)), False),
        (([1, 1],), False),
    ],
)
def test_path_validation_reports_container_shape_without_changing_nodes(
    path, canonical
):
    tree = BranchAncestry()
    leaf, shape = tree.intern_with_shape(path)
    assert shape is canonical
    assert leaf == tree.intern(path)
    assert tree.path(leaf) == [list(pair) for pair in path]


def test_path_subclasses_remain_valid_but_are_not_canonical_lists():
    class Pair(list):
        pass

    class Path(list):
        pass

    tree = BranchAncestry()
    for path in [Path([[1, 1]]), [Pair([1, 1])], [Pair([1, 1]), [2, 1]]]:
        leaf, canonical = tree.intern_with_shape(path)
        assert not canonical
        assert tree.path(leaf) == [list(pair) for pair in path]


@pytest.mark.parametrize("leaf", [-1, 1, True, "0"])
def test_invalid_leaf_rejected(leaf):
    with pytest.raises(ValueError):
        BranchAncestry().path(leaf)


@pytest.mark.parametrize("extension", [(-1, 1, 1), (1, 1, 1), (0, 0, 1), (0, True, 1)])
def test_invalid_extension_rejected(extension):
    with pytest.raises(ValueError):
        BranchAncestry().add(*extension)


def test_shared_views_are_immutable_and_do_not_retain_expanded_prefixes():
    tree = BranchAncestry()
    leaf = 0
    views = []
    for turn in range(1, 2001):
        leaf = tree.add(leaf, turn, 1)
        views.append(tree.view(leaf))
    assert len(tree.nodes) == len(tree._views) == 2000
    assert all(view.tree is tree and len(view) == i for i, view in enumerate(views, 1))
    last = views[-1]
    assert tree.view(leaf) is last
    assert list(last)[-1] == last[-1] == (2000, 1)
    assert last[:2] == ((1, 1), (2, 1))
    assert copy(last) is deepcopy(last) is last
    assert last == tuple(last) and tuple(last) == last
    assert hash(last) == hash(tuple(last))
    clone = BranchAncestry(tree.nodes).view(leaf)
    assert last == clone and hash(last) == hash(clone)
    assert last != list(last)
    assert last != tree.view(0)
    assert tree.view(0) < last
    assert sorted([((1, 2),), tree.view(1)]) == [tree.view(1), ((1, 2),)]
    assert sorted([tree.view(1), ((1, 2),)]) == [tree.view(1), ((1, 2),)]
    with pytest.raises(AttributeError):
        last.leaf = 0
    with pytest.raises(AttributeError):
        del last.tree
    with pytest.raises(IndexError):
        _ = last[2000]
    for bad in (-1, True, 2001):
        with pytest.raises(ValueError):
            tree.view(bad)
        with pytest.raises(ValueError):
            SharedPath(tree, bad)


def test_last_pair_lookup_does_not_expand_the_path(monkeypatch):
    tree = BranchAncestry()
    leaf = tree.intern([[turn, 1] for turn in range(1, 2001)])
    path = tree.view(leaf)

    def forbidden(_path):
        raise AssertionError("last-pair lookup expanded the path")

    with monkeypatch.context() as guard:
        guard.setattr(SharedPath, "__iter__", forbidden)
        assert path[-1] == path[1999] == (2000, 1)
        with pytest.raises(IndexError):
            tree.view(0)[-1]
    with pytest.raises(TypeError):
        path[-1.0]
