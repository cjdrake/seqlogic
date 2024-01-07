"""
TODO(cjdrake): Write docstring.
"""

from .hier import HierVar, Module
from .logicvec import F, T, xes
from .sim import SimVar


class Logic(HierVar, SimVar):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, shape: tuple[int, ...]):
        HierVar.__init__(self, name, parent)
        SimVar.__init__(self, value=xes(shape))

    def changed(self) -> bool:
        return self.delta

    def posedge(self) -> bool:
        return self._value == F and self._next_value == T

    def negedge(self) -> bool:
        return self._value == T and self._next_value == F
