"""Logic Design Hierarchy."""

from __future__ import annotations

from abc import ABC, abstractmethod


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


class Module(Hierarchy):
    """Design hierarchy branch node."""

    def __init__(self, name: str, parent: Module | None = None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

    @property
    def qualname(self) -> str:
        """Return the module's fully qualified name."""
        match self._parent:
            case None:
                return f"/{self._name}"
            case List():
                return f"{self._parent.qualname}[{self._name}]"
            case Dict():
                return f"{self._parent.qualname}['{self._name}']"
            case Module():
                return f"{self._parent.qualname}/{self._name}"
            case _:  # pragma: no cover
                assert False


class HierVar(Hierarchy):
    """Design hierarchy leaf node."""

    def __init__(self, name: str, parent: Module):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

    @property
    def qualname(self) -> str:
        """Return the variable's fully qualified name."""
        match self._parent:
            case List():
                return f"{self._parent.qualname}[{self._name}]"
            case Dict():
                return f"{self._parent.qualname}['{self._name}']"
            case Module():
                return f"{self._parent.qualname}/{self._name}"
            case _:  # pragma: no cover
                assert False


class List(Module, list):
    """TODO(cjdrake): Write docstring."""


class Dict(Module, dict):
    """TODO(cjdrake): Write docstring."""
