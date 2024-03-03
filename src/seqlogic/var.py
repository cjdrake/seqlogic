"""TODO(cjdrake): Write docstring."""

import re
from collections import defaultdict

from vcd.writer import VarValue

from . import sim
from .hier import Module, Variable
from .logicvec import F, T, logicvec, xes


def vec2vcd(x: logicvec) -> VarValue:
    """Convert value to VCD variable."""
    bits = []
    for i in range(x.size):
        bits.append(str(x[i]))
    return "".join(reversed(bits))


class TraceVar(Variable, sim.Singular):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, value):
        """TODO(cjdrake): Write docstring."""
        Variable.__init__(self, name, parent)
        sim.Singular.__init__(self, value)
        self._waves_change = None
        self._vcd_change = None

    def dump_waves(self, waves: defaultdict, pattern: str):
        """TODO(cjdrake): Write docstring."""
        if re.fullmatch(pattern, self.qualname):
            t = self._sim.time()
            waves[t][self] = self._value

            def change():
                t = self._sim.time()
                waves[t][self] = self._next_value

            self._waves_change = change

    def dump_vcd(self, vcdw, pattern: str):
        """TODO(cjdrake): Write docstring."""
        assert isinstance(self._parent, Module)
        if re.match(pattern, self.qualname):
            var = vcdw.register_var(
                scope=self._parent.scope,
                name=self.name,
                var_type="reg",
                size=self._value.size,
                init=vec2vcd(self._value),
            )

            def change():
                t = self._sim.time()
                vcdw.change(var, t, vec2vcd(self._next_value))

            self._vcd_change = change

    def update(self):
        """TODO(cjdrake): Write docstring."""
        if self._waves_change and self.dirty():
            self._waves_change()
        if self._vcd_change and self.dirty():
            self._vcd_change()
        super().update()


class Bits(TraceVar):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, shape: tuple[int, ...]):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, value=xes(shape))


# TODO(cjdrake): Create a generic type for enums


class Bit(Bits):
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


class Array(Variable, sim.Aggregate):
    """TODO(cjdrake): Write docstring."""

    def __init__(
        self,
        name: str,
        parent: Module,
        packed_shape: tuple[int, ...],
        unpacked_shape: tuple[int, ...],
    ):
        """TODO(cjdrake): Write docstring."""
        Variable.__init__(self, name, parent)
        sim.Aggregate.__init__(self, shape=unpacked_shape, value=xes(packed_shape))
