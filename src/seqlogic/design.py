"""Design elements.

Combine elements from hierarchy, bit vectors, and simulation into a
straightforward API for creating a digital design.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC
from collections import defaultdict
from collections.abc import Callable

from vcd.writer import VarValue
from vcd.writer import VCDWriter as VcdWriter

from .hier import Branch, Leaf
from .lbool import Vec, VecEnum, ones, zeros
from .sim import Aggregate, Region, SimAwaitable, Singular, State, changed, get_loop

_item2char = {
    0b00: "x",
    0b01: "0",
    0b10: "1",
    0b11: "x",
}


def _vec2vcd(v: Vec) -> VarValue:
    """Convert bit array to VCD value."""
    # pylint: disable = protected-access
    return "".join(_item2char[v._get_item(i)] for i in range(len(v) - 1, -1, -1))


class _TraceIf(ABC):
    """Tracing interface.

    Implemented by components that support debug dump.
    """

    def dump_waves(self, waves: defaultdict, pattern: str):
        """Dump design elements w/ names matching pattern to waves dict."""

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """Dump design elements w/ names matching pattern to VCD file."""


class _ProcIf(ABC):
    """Process interface.

    Implemented by components that contain local simulator processes.
    """

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


class Module(Branch, _TraceIf, _ProcIf):
    """Hierarchical, branch-level design component.

    A module contains:
    * Submodules
    * Ports
    * Local variables
    * Local processes
    """

    def __init__(self, name: str, parent: Module | None = None):
        Branch.__init__(self, name, parent)
        _ProcIf.__init__(self)

    @property
    def scope(self) -> str:
        """Return the branch's full name using dot separator syntax."""
        if self._parent is None:
            return self.name
        assert isinstance(self._parent, Module)
        return f"{self._parent.scope}.{self.name}"

    def dump_waves(self, waves: defaultdict, pattern: str):
        for child in self._children:
            assert isinstance(child, _TraceIf)
            child.dump_waves(waves, pattern)

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        for child in self._children:
            assert isinstance(child, _TraceIf)
            child.dump_vcd(vcdw, pattern)


class _TraceSingular(Leaf, _TraceIf, Singular, _ProcIf):
    """Combine hierarchy and sim semantics for singular data types."""

    def __init__(self, name: str, parent: Module, dtype: type):
        Leaf.__init__(self, name, parent)
        Singular.__init__(self, dtype.xes())
        _ProcIf.__init__(self)
        self._dtype = dtype
        self._waves_change = None
        self._vcd_change = None

    def update(self):
        if self._waves_change and self.dirty():
            self._waves_change()
        if self._vcd_change and self.dirty():
            self._vcd_change()
        super().update()

    def dump_waves(self, waves: defaultdict, pattern: str):
        if re.fullmatch(pattern, self.qualname):
            t = self._sim.time
            waves[t][self] = self._value

            def change():
                t = self._sim.time
                waves[t][self] = self._next_value

            self._waves_change = change

    def dump_vcd(self, vcdw, pattern: str):
        assert isinstance(self._parent, Module)
        if issubclass(self._dtype, VecEnum):
            if re.match(pattern, self.qualname):
                var = vcdw.register_var(
                    scope=self._parent.scope,
                    name=self.name,
                    var_type="string",
                    init=self._value.name,
                )
                self._vcd_change = lambda: vcdw.change(var, self._sim.time, self._next_value.name)
        else:
            if re.match(pattern, self.qualname):
                var = vcdw.register_var(
                    scope=self._parent.scope,
                    name=self.name,
                    var_type="reg",
                    size=len(self._value),
                    init=_vec2vcd(self._value),
                )
                self._vcd_change = lambda: vcdw.change(
                    var, self._sim.time, _vec2vcd(self._next_value)
                )

    def connect(self, src):
        """Convenience function to reduce process boilerplate."""

        async def proc():
            while True:
                await changed(src)
                self.next = src.value

        self._procs.append((Region.REACTIVE, proc, (), {}))


class Bits(_TraceSingular):
    """Leaf-level bitvector design component."""


class Bit(Bits):
    """One-bit specialization of Bits that supports edge detection."""

    def __init__(self, name: str, parent: Module):
        super().__init__(name, parent, dtype=Vec[1])

    def is_neg(self) -> bool:
        """Return True when bit is stable 0 => 0."""
        return self._value == zeros(1) and self._next_value == zeros(1)

    def is_posedge(self) -> bool:
        """Return True when bit transitions 0 => 1."""
        return self._value == zeros(1) and self._next_value == ones(1)

    def is_negedge(self) -> bool:
        """Return True when bit transition 1 => 0."""
        return self._value == ones(1) and self._next_value == zeros(1)

    def is_pos(self) -> bool:
        """Return True when bit is stable 1 => 1."""
        return self._value == ones(1) and self._next_value == ones(1)

    async def posedge(self) -> State:
        """Suspend; resume execution at signal posedge."""
        self._sim.add_event(self, self.is_posedge)
        state = await SimAwaitable()
        return state

    async def negedge(self) -> State:
        """Suspend; resume execution at signal negedge."""
        self._sim.add_event(self, self.is_negedge)
        state = await SimAwaitable()
        return state


class _TraceAggregate(Leaf, _TraceIf, Aggregate, _ProcIf):
    """Combine hierarchy and sim semantics for aggregate data types."""

    def __init__(self, name: str, parent: Module, dtype: type):
        Leaf.__init__(self, name, parent)
        Aggregate.__init__(self, dtype.xes())
        _ProcIf.__init__(self)
        self._dtype = dtype


class _ArrayXPropItem:
    """Array X-Prop item helper."""

    def __init__(self, dtype: type):
        self._dtype = dtype

    def _get_value(self):
        return self._dtype.xes()

    value = property(fget=_get_value)

    def _set_next(self, value):
        pass

    next = property(fset=_set_next)


class Array(_TraceAggregate):
    """Leaf-level array of bitvector/enum/struct/union design components."""

    def __init__(self, name: str, parent: Module, shape: tuple[int, ...], dtype: type):
        assert len(shape) == 1
        super().__init__(name, parent, dtype)

    def __getitem__(self, key: int | Vec):
        match key:
            case int():
                return super().__getitem__(key)
            case Vec():
                try:
                    i = key.to_uint()
                except ValueError:
                    return _ArrayXPropItem(self._dtype)
                else:
                    return super().__getitem__(i)
            case _:
                assert False


def simify(d: Module | Bits | Bit | Array):
    """Add design processes to the simulator."""
    loop = get_loop()
    for node in d.iter_bfs():
        assert isinstance(node, _ProcIf)
        for region, func, args, kwargs in node.procs:
            loop.add_proc(region, func, *args, **kwargs)
