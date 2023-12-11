"""Logic Design Hierarchy."""

from abc import ABC, abstractmethod


class Hierarchy(ABC):
    """Any hierarchical design element."""

    def __init__(self, name: str, parent):
        """TODO(cjdrake): Write docstring."""
        self._name = name
        self._parent = parent

    @property
    def name(self) -> str:
        """Return the design element name."""
        return self._name

    @property
    def parent(self):
        """Return the parent, or None."""
        return self._parent

    @property
    @abstractmethod
    def qualname(self) -> str:
        """Return the design element's fully qualified name."""


class Module(Hierarchy):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent=None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

    @property
    def qualname(self) -> str:
        """Return the module's fully qualified name."""
        if self._parent is None:
            return f"/{self._name}"
        return f"{self._parent.qualname}/{self._name}"


class HierVar(Hierarchy):
    """TODO(cjdrake): Write docstring."""

    @property
    def qualname(self) -> str:
        """Return the variable's fully qualified name."""
        return f"{self._parent.qualname}/{self._name}"
