"""Logic Design Hierarchy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator


class Hierarchy(ABC):
    """Any hierarchical design element."""

    def __init__(self, name: str, parent: Hierarchy | None):
        """TODO(cjdrake): Write docstring."""
        self._name = name
        self._parent = parent

    @property
    def name(self) -> str:
        """Return the design element name."""
        return self._name

    @property
    def parent(self) -> Hierarchy | None:
        """Return the parent, or None."""
        return self._parent

    @property
    @abstractmethod
    def qualname(self) -> str:
        """Return the design element's fully qualified name."""

    @abstractmethod
    def iter_bfs(self) -> Generator[Hierarchy, None, None]:
        """Iterate through the design hierarchy in BFS order."""

    @abstractmethod
    def iter_dfs(self) -> Generator[Hierarchy, None, None]:
        """Iterate through the design hierarchy in DFS order."""


class Branch(Hierarchy):
    """Design hierarchy branch node."""

    def __init__(self, name: str, parent: Branch | None = None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)
        self._children: list[Hierarchy] = []
        if parent is not None:
            parent.add_child(self)

    @property
    def qualname(self) -> str:
        """Return the branch's fully qualified name."""
        if self._parent is None:
            return f"/{self._name}"
        else:
            return f"{self._parent.qualname}/{self._name}"

    def iter_bfs(self) -> Generator[Hierarchy, None, None]:
        """TODO(cjdrake): Write docstring."""
        yield self
        for child in self._children:
            yield from child.iter_bfs()

    def iter_dfs(self) -> Generator[Hierarchy, None, None]:
        """TODO(cjdrake): Write docstring."""
        for child in self._children:
            yield from child.iter_dfs()
        yield self

    def add_child(self, child: Hierarchy):
        """Add child branch or leaf."""
        self._children.append(child)


class Leaf(Hierarchy):
    """Design hierarchy leaf node."""

    def __init__(self, name: str, parent: Branch):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)
        parent.add_child(self)

    @property
    def qualname(self) -> str:
        """Return the leaf's fully qualified name."""
        assert self._parent is not None
        return f"{self._parent.qualname}/{self._name}"

    def iter_bfs(self) -> Generator[Leaf, None, None]:
        """TODO(cjdrake): Write docstring."""
        yield self

    def iter_dfs(self) -> Generator[Leaf, None, None]:
        """TODO(cjdrake): Write docstring."""
        yield self
