"""TODO(cjdrake): Write docstring."""

from __future__ import annotations

import inspect
import re
from abc import ABC
from collections import defaultdict
from collections.abc import Callable

from vcd.writer import VarValue
from vcd.writer import VCDWriter as VcdWriter

from . import bits, hier, sim
from .bits import F, T, xes
from .sim import Region, changed, get_loop

_item2char = {
    0b00: "x",
    0b01: "0",
    0b10: "1",
    0b11: "x",
}


def _bits2vcd(b: bits.Bits) -> VarValue:
    """Convert bit array to VCD value."""
    # pylint: disable = protected-access
    return "".join(_item2char[b._v._get_item(i)] for i in range(b._v._n - 1, -1, -1))


class _TraceIf(ABC):
    """TODO(cjdrake): Write docstring."""

    def dump_waves(self, waves: defaultdict, pattern: str):
        """TODO(cjdrake): Write docstring."""

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """TODO(cjdrake): Write docstring."""


class _ProcIf(ABC):
    """TODO(cjdrake): Write docstring."""

    def __init__(self):
        self._procs = []

        def is_proc(m) -> bool:
            match m:
                case [int(), Callable() as f] if inspect.iscoroutinefunction(f):
                    return True
                case _:
                    return False

        for _, (region, func) in inspect.getmembers(self, is_proc):
            self._procs.append((region, func, (), {}))

    @property
    def procs(self):
        return self._procs


class Module(hier.Branch, _TraceIf, _ProcIf):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None = None):
        """TODO(cjdrake): Write docstring."""
        hier.Branch.__init__(self, name, parent)
        _ProcIf.__init__(self)

    @property
    def scope(self) -> str:
        """Return the branch's full name using dot separator syntax."""
        if self._parent is None:
            return self.name
        assert isinstance(self._parent, Module)
        return f"{self._parent.scope}.{self.name}"

    def dump_waves(self, waves: defaultdict, pattern: str):
        """TODO(cjdrake): Write docstring."""
        for child in self._children:
            assert isinstance(child, _TraceIf)
            child.dump_waves(waves, pattern)

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """TODO(cjdrake): Write docstring."""
        for child in self._children:
            assert isinstance(child, _TraceIf)
            child.dump_vcd(vcdw, pattern)


class _TraceSingular(hier.Leaf, _TraceIf, sim.Singular, _ProcIf):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, value):
        """TODO(cjdrake): Write docstring."""
        hier.Leaf.__init__(self, name, parent)
        sim.Singular.__init__(self, value)
        _ProcIf.__init__(self)
        self._waves_change = None
        self._vcd_change = None

    def update(self):
        """TODO(cjdrake): Write docstring."""
        if self._waves_change and self.dirty():
            self._waves_change()
        if self._vcd_change and self.dirty():
            self._vcd_change()
        super().update()

    def dump_waves(self, waves: defaultdict, pattern: str):
        """TODO(cjdrake): Write docstring."""
        if re.fullmatch(pattern, self.qualname):
            t = self._sim.time
            waves[t][self] = self._value

            def change():
                t = self._sim.time
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
                init=_bits2vcd(self._value),
            )

            def change():
                t = self._sim.time
                vcdw.change(var, t, _bits2vcd(self._next_value))

            self._vcd_change = change

    def connect(self, src):
        """TODO(cjdrake): Write docstring."""

        async def proc():
            while True:
                await changed(src)
                self.next = src.next

        self._procs.append((Region(0), proc, (), {}))


class _TraceAggregate(hier.Leaf, _TraceIf, sim.Aggregate, _ProcIf):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, shape: tuple[int, ...], value):
        """TODO(cjdrake): Write docstring."""
        hier.Leaf.__init__(self, name, parent)
        sim.Aggregate.__init__(self, shape, value)
        _ProcIf.__init__(self)

    # def update(self):
    #    """TODO(cjdrake): Write docstring."""
    #    super().update()


class Bits(_TraceSingular):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, shape: tuple[int, ...]):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, value=xes(shape))


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


class Enum(_TraceSingular):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module, cls):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, value=cls.X)


class Array(_TraceAggregate):
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


def simify(d: Module | Bits | Enum | Array):
    """TODO(cjdrake): Write docstring."""
    loop = get_loop()
    for node in d.iter_bfs():
        assert isinstance(node, _ProcIf)
        for region, func, args, kwargs in node.procs:
            loop.add_proc(region, func, *args, **kwargs)
