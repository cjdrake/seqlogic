"""TODO(cjdrake): Write docstring."""

import re
from collections import defaultdict

from vcd.writer import VarValue

from . import sim
from .bits import F, T, logicvec, xes
from .hier import Module, Variable

_item2char = {
    0b00: "x",
    0b01: "0",
    0b10: "1",
    0b11: "x",
}


def vec2vcd(v: logicvec) -> VarValue:
    """Convert value to VCD variable."""
    # pylint: disable = protected-access
    return "".join(_item2char[v._w._get_item(i)] for i in range(v._w._n - 1, -1, -1))


class TraceSingular(Variable, sim.Singular):
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


class TraceAggregate(Variable, sim.Aggregate):
    """TODO(cjdrake): Write docstring."""

    def __init__(
        self,
        name: str,
        parent: Module,
        shape: tuple[int, ...],
        value,
    ):
        """TODO(cjdrake): Write docstring."""
        Variable.__init__(self, name, parent)
        sim.Aggregate.__init__(self, shape, value)

    def dump_waves(self, waves: defaultdict, pattern: str):
        """TODO(cjdrake): Write docstring."""

    def dump_vcd(self, vcdw, pattern: str):
        """TODO(cjdrake): Write docstring."""

    # def update(self):
    #    """TODO(cjdrake): Write docstring."""
    #    super().update()


class Bits(TraceSingular):
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


class Array(TraceAggregate):
    """TODO(cjdrake): Write docstring."""

    def __init__(
        self,
        name: str,
        parent: Module,
        unpacked_shape: tuple[int, ...],
        packed_shape: tuple[int, ...],
    ):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, unpacked_shape, value=xes(packed_shape))
