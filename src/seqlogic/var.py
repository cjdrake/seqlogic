"""TODO(cjdrake): Write docstring."""

import re
from collections import defaultdict

from .hier import HierVar, Module
from .logic import logic
from .logicvec import F, T, xes
from .sim import SimVar


class TraceVar(HierVar, SimVar):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, init):
        """TODO(cjdrake): Write docstring."""
        HierVar.__init__(self, name, parent)
        SimVar.__init__(self, value=init)
        self._waves_change = None

    def dump_waves(self, waves: defaultdict, pattern: str):
        """TODO(cjdrake): Write docstring."""
        if re.fullmatch(pattern, self.qualname):
            t = self._sim.time()
            waves[t][self] = self._value

            def change():
                t = self._sim.time()
                waves[t][self] = self._next_value

            self._waves_change = change

    def update(self):
        """TODO(cjdrake): Write docstring."""
        if self._waves_change and self.dirty():
            self._waves_change()
        super().update()


class LogicVar(TraceVar):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, init=logic.X)

    def posedge(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return (self._value is logic.F) and (self._next_value is logic.T)

    def negedge(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return (self._value is logic.T) and (self._next_value is logic.F)


class LogicVec(TraceVar):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, shape: tuple[int, ...]):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, init=xes(shape))


# TODO(cjdrake): Create a generic type for enums


class Logic(LogicVec):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, shape=(1,))

    def posedge(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return self._value == F and self._next_value == T

    def negedge(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return self._value == T and self._next_value == F
