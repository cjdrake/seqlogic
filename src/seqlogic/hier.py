"""Logic hierarchy.

Implement the fundamental components of an N-ary tree.
A tree has branches and leaves.
Leaves must have a parent.
A branch may either have a parent, or no parent (the root).
"""

from __future__ import annotations

from collections.abc import Generator


class Hierarchy:
    """Any hierarchical design element."""

    def __init__(self, name: str, parent: Branch | None):
        """Initialize.

        Args:
            name: Name of this tree node.
            parent: Parent tree node, or None.
        """
        self._check_name(name)
        self._name = name
        self._parent = parent

    @property
    def name(self) -> str:
        """Return the design element name."""
        return self._name

    @property
    def parent(self) -> Branch | None:
        """Return the parent, or None."""
        return self._parent

    @property
    def qualname(self) -> str:
        """Return the design element's fully qualified name."""

    def iter_bfs(self) -> Generator[Hierarchy, None, None]:
        """Iterate through the design hierarchy in BFS order."""
        raise NotImplementedError()

    def iter_dfs(self) -> Generator[Hierarchy, None, None]:
        """Iterate through the design hierarchy in DFS order."""
        raise NotImplementedError()

    def iter_leaves(self) -> Generator[Leaf, None, None]:
        """Iterate through design leaves, left to right."""
        raise NotImplementedError()

    @staticmethod
    def _check_name(name: str):
        valid = name.isidentifier()
        if not valid:
            raise ValueError(f"Expected public identifier, got {name}")


class Branch(Hierarchy):
    """Design hierarchy branch node."""

    def __init__(self, name: str, parent: Branch | None = None):
        super().__init__(name, parent)
        self._children: list[Branch | Leaf] = []
        if parent is not None:
            parent.add_child(self)

    @property
    def qualname(self) -> str:
        if self._parent is None:
            return f"/{self._name}"
        return f"{self._parent.qualname}/{self._name}"

    def iter_bfs(self) -> Generator[Hierarchy, None, None]:
        yield self
        for child in self._children:
            yield from child.iter_bfs()

    def iter_dfs(self) -> Generator[Hierarchy, None, None]:
        for child in self._children:
            yield from child.iter_dfs()
        yield self

    def iter_leaves(self) -> Generator[Leaf, None, None]:
        for child in self._children:
            yield from child.iter_leaves()

    def add_child(self, child: Branch | Leaf):
        """Add child branch or leaf."""
        self._children.append(child)

    def is_root(self) -> bool:
        """Return True if this branch is the root of the tree."""
        return self._parent is None


class Leaf(Hierarchy):
    """Design hierarchy leaf node."""

    def __init__(self, name: str, parent: Branch):
        super().__init__(name, parent)
        parent.add_child(self)

        # Help type checker verify parent is not None
        self._parent: Branch

    @property
    def qualname(self) -> str:
        return f"{self._parent.qualname}/{self._name}"

    def iter_bfs(self) -> Generator[Hierarchy, None, None]:
        yield self

    def iter_dfs(self) -> Generator[Hierarchy, None, None]:
        yield self

    def iter_leaves(self) -> Generator[Leaf, None, None]:
        yield self
