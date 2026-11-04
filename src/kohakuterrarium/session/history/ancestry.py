"""Intern branch lineage prefixes as parent references."""

from collections.abc import Sequence
from functools import total_ordering
from typing import Any


@total_ordering
class SharedPath(Sequence):
    """Immutable ancestry view; expansion is temporary and never cached."""

    __slots__ = ("tree", "leaf")

    def __init__(self, tree: "BranchAncestry", leaf: int):
        if type(leaf) is not int or not 0 <= leaf <= len(tree.nodes):
            raise ValueError("Invalid ancestry leaf")
        object.__setattr__(self, "tree", tree)
        object.__setattr__(self, "leaf", leaf)

    def __setattr__(self, name, value):
        raise AttributeError("Shared paths are immutable")

    def __delattr__(self, name):
        raise AttributeError("Shared paths are immutable")

    def __copy__(self):
        return self

    def __len__(self):
        return self.tree._depths[self.leaf]

    def __iter__(self):
        refs, leaf = [], self.leaf
        while leaf:
            refs.append(leaf)
            leaf = self.tree.nodes[leaf - 1][0]
        for ref in reversed(refs):
            _, turn, branch = self.tree.nodes[ref - 1]
            yield turn, branch

    def __getitem__(self, index):
        if type(index) is int and index in (-1, len(self) - 1):
            if not self.leaf:
                raise IndexError("Shared path index out of range")
            _, turn, branch = self.tree.nodes[self.leaf - 1]
            return turn, branch
        return tuple(self)[index]

    def __eq__(self, other):
        if isinstance(other, SharedPath) and other.tree is self.tree:
            return self.leaf == other.leaf
        if not isinstance(other, (SharedPath, tuple)):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __lt__(self, other):
        return tuple(self) < tuple(other)

    def __hash__(self):
        return hash(tuple(self))

    def __deepcopy__(self, memo: dict[int, Any]):
        return self


class BranchAncestry:
    """Share lineage nodes across recorded branches and event metadata."""

    def __init__(self, nodes: list | None = None):
        self.nodes: list[list[int]] = []
        self._ids: dict[tuple[int, int, int], int] = {}
        self._depths = [0]
        self._views: dict[int, SharedPath] = {}
        for raw in nodes or []:
            if not isinstance(raw, list) or len(raw) != 3:
                raise ValueError("Invalid ancestry node")
            parent, turn, branch = raw
            if any(type(value) is not int for value in raw):
                raise ValueError("Ancestry coordinates must be integers")
            if not 0 <= parent <= len(self.nodes) or turn <= 0 or branch <= 0:
                raise ValueError("Invalid ancestry reference")
            if parent and self.nodes[parent - 1][1] >= turn:
                raise ValueError("Ancestry turns must increase")
            size = len(self.nodes)
            self.add(parent, turn, branch)
            if len(self.nodes) == size:
                raise ValueError("Duplicate ancestry node")

    def add(self, parent: int, turn: int, branch: int) -> int:
        """Return the shared node for one lineage extension."""
        if (
            any(type(value) is not int for value in (parent, turn, branch))
            or not 0 <= parent <= len(self.nodes)
            or turn <= 0
            or branch <= 0
            or parent
            and self.nodes[parent - 1][1] >= turn
        ):
            raise ValueError("Invalid ancestry extension")
        key = parent, turn, branch
        if key not in self._ids:
            self.nodes.append(list(key))
            self._depths.append(self._depths[parent] + 1)
            self._ids[key] = len(self.nodes)
        return self._ids[key]

    def intern(self, path: Sequence[Sequence[int]]) -> int:
        """Return a leaf reference for a strictly increasing positive path."""
        return self.intern_with_shape(path)[0]

    def intern_with_shape(self, path: Sequence[Sequence[int]]) -> tuple[int, bool]:
        """Intern a path and report its exact list-of-lists container shape."""
        parent, previous = 0, 0
        canonical = type(path) is list
        for pair in path:
            pair_type = type(pair)
            if (
                pair_type is not list
                and pair_type is not tuple
                and not isinstance(pair, (list, tuple))
            ) or len(pair) != 2:
                raise ValueError("Invalid ancestry pair")
            canonical = canonical and pair_type is list
            turn, branch = pair
            if (
                type(turn) is not int
                or type(branch) is not int
                or turn <= previous
                or branch <= 0
            ):
                raise ValueError("Invalid ancestry coordinate")
            key = parent, turn, branch
            existing = self._ids.get(key)
            parent = self.add(*key) if existing is None else existing
            previous = turn
        return parent, canonical

    def view(self, leaf: int) -> SharedPath:
        """Return one shared immutable view of a validated leaf."""
        if type(leaf) is not int or not 0 <= leaf <= len(self.nodes):
            raise ValueError("Invalid ancestry leaf")
        if leaf not in self._views:
            self._views[leaf] = SharedPath(self, leaf)
        return self._views[leaf]

    def path(self, leaf: int) -> list[list[int]]:
        """Materialize one lineage at a legacy API boundary."""
        if type(leaf) is not int or not 0 <= leaf <= len(self.nodes):
            raise ValueError("Invalid ancestry leaf")
        result = []
        while leaf:
            leaf, turn, branch = self.nodes[leaf - 1]
            result.append([turn, branch])
        result.reverse()
        return result
