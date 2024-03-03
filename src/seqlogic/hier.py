"""Logic Design Hierarchy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator

from vcd.writer import VCDWriter as VcdWriter

from .sim import notify


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

    def dump_waves(self, waves: defaultdict, pattern: str):
        """TODO(cjdrake): Write docstring."""

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """TODO(cjdrake): Write docstring."""


class Module(Hierarchy):
    """Design hierarchy branch node."""

    def __init__(self, name: str, parent: Module | None = None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)
        self._children: list[Hierarchy] = []
        if parent is not None:
            parent.add_child(self)
        # Processes
        self._procs = set()

    @property
    def qualname(self) -> str:
        """Return the module's fully qualified name."""
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

    @property
    def scope(self) -> str:
        """Return the module's full name using dot separator syntax."""
        if self._parent is None:
            return self.name
        assert isinstance(self._parent, Module)
        return f"{self._parent.scope}.{self.name}"

    @property
    def procs(self):
        """TODO(cjdrake): Write docstring."""
        return self._procs

    def connect(self, dst, src):
        """TODO(cjdrake): Write docstring."""

        async def proc():
            while True:
                await notify(src.changed)
                dst.next = src.next

        self._procs.add((proc, 0))

    def add_child(self, child: Hierarchy):
        """Add child module or variable."""
        self._children.append(child)

    def dump_waves(self, waves: defaultdict, pattern: str):
        """TODO(cjdrake): Write docstring."""
        for child in self._children:
            child.dump_waves(waves, pattern)

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """TODO(cjdrake): Write docstring."""
        for child in self._children:
            child.dump_vcd(vcdw, pattern)


class Variable(Hierarchy):
    """Design hierarchy leaf node."""

    def __init__(self, name: str, parent: Module):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)
        parent.add_child(self)

    @property
    def qualname(self) -> str:
        """Return the variable's fully qualified name."""
        assert self._parent is not None
        return f"{self._parent.qualname}/{self._name}"

    def iter_bfs(self) -> Generator[Variable, None, None]:
        """TODO(cjdrake): Write docstring."""
        yield self

    def iter_dfs(self) -> Generator[Variable, None, None]:
        """TODO(cjdrake): Write docstring."""
        yield self
