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
from .lbool import Vec, VecEnum, lit2vec
from .sim import Aggregate, Region, SimAwaitable, Singular, State, changed, get_loop, resume

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


class Module(Branch, _ProcIf, _TraceIf):
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

    def bits(self, name: str, dtype: type, port: bool = False) -> Bits:
        # TODO(cjdrake): Check name
        node = Bits(name, parent=self, dtype=dtype)
        setattr(self, f"_{name}", node)
        if port:
            setattr(self, name, node)
        return node

    def bit(self, name: str, port: bool = False) -> Bit:
        # TODO(cjdrake): Check name
        node = Bit(name, parent=self)
        setattr(self, f"_{name}", node)
        if port:
            setattr(self, name, node)
        return node

    def array(self, name: str, dtype: type) -> Array:
        # TODO(cjdrake): Check name
        node = Array(name, parent=self, dtype=dtype)
        setattr(self, f"_{name}", node)
        return node

    def submod(self, name: str, mod: type, **params) -> Module:
        # TODO(cjdrake): Check name
        node = mod(name, parent=self, **params)
        setattr(self, f"_{name}", node)
        return node

    def combi(self, y: Bits, f: Callable, *xs: Bits):
        """Combinational logic."""

        async def proc():
            while True:
                await changed(*xs)
                y.next = f(*[x.value for x in xs])

        self._procs.append((Region.REACTIVE, proc, (), {}))

    def combis(self, ys: list[Bits], f: Callable, *xs: Bits):
        """Combinational logic."""

        async def proc():
            while True:
                await changed(*xs)
                ret = f(*[x.value for x in xs])
                assert len(ys) == len(ret)
                for i, y in enumerate(ys):
                    y.next = ret[i]

        self._procs.append((Region.REACTIVE, proc, (), {}))

    def connect(self, y: Bits, x: Bits):
        """Connect input to output."""

        async def proc():
            while True:
                await changed(x)
                y.next = x.value

        self._procs.append((Region.REACTIVE, proc, (), {}))

    def dff(self, q: Bits, d: Bits, clk: Bit):
        """D Flip Flop."""

        async def proc():
            while True:
                state = await resume((clk, clk.is_posedge))
                if state is clk:
                    q.next = d.value
                else:
                    assert False

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def dff_ar(self, q: Bits, d: Bits, clk: Bit, rst: Bit, rval: Vec):
        """D Flip Flop with async reset."""

        async def proc():
            while True:
                state = await resume(
                    (rst, rst.is_posedge),
                    (clk, lambda: clk.is_posedge() and rst.is_neg()),
                )
                if state is rst:
                    q.next = rval
                elif state is clk:
                    q.next = d.value
                else:
                    assert False

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def dff_en_ar(self, q: Bits, d: Bits, en: Bit, clk: Bit, rst: Bit, rval: Vec):
        """D Flip Flop with enable, and async reset."""

        async def proc():
            while True:
                state = await resume(
                    (rst, rst.is_posedge),
                    (clk, lambda: clk.is_posedge() and rst.is_neg() and en.value == "1b1"),
                )
                if state is rst:
                    q.next = rval
                elif state is clk:
                    q.next = d.value
                else:
                    assert False

        self._procs.append((Region.ACTIVE, proc, (), {}))


class Bits(Leaf, Singular, _ProcIf, _TraceIf):
    """Leaf-level btvector design component."""

    def __init__(self, name: str, parent: Module, dtype: type):
        Leaf.__init__(self, name, parent)
        Singular.__init__(self, dtype.xes())
        _ProcIf.__init__(self)
        self._dtype = dtype
        self._waves_change = None
        self._vcd_change = None

    # Singular => State
    def set_next(self, value):
        if isinstance(value, str):
            value = lit2vec(value)
        assert isinstance(value, self._dtype)
        super().set_next(value)

    next = property(fset=set_next)

    def update(self):
        if self._waves_change and self.dirty():
            self._waves_change()
        if self._vcd_change and self.dirty():
            self._vcd_change()
        super().update()

    # TraceIf
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
        elif issubclass(self._dtype, Vec):
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


class Bit(Bits):
    """One-bit specialization of Bits that supports edge detection."""

    def __init__(self, name: str, parent: Module):
        super().__init__(name, parent, dtype=Vec[1])

    def is_neg(self) -> bool:
        """Return True when bit is stable 0 => 0."""
        return self._value == "1b0" and self._next_value == "1b0"

    def is_posedge(self) -> bool:
        """Return True when bit transitions 0 => 1."""
        return self._value == "1b0" and self._next_value == "1b1"

    def is_negedge(self) -> bool:
        """Return True when bit transition 1 => 0."""
        return self._value == "1b1" and self._next_value == "1b0"

    def is_pos(self) -> bool:
        """Return True when bit is stable 1 => 1."""
        return self._value == "1b1" and self._next_value == "1b1"

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


class Array(Leaf, Aggregate, _ProcIf, _TraceIf):
    """Leaf-level array of vec/enum/struct/union design components."""

    def __init__(self, name: str, parent: Module, dtype: type):
        Leaf.__init__(self, name, parent)
        Aggregate.__init__(self, dtype.xes())
        _ProcIf.__init__(self)
        self._dtype = dtype

    def __getitem__(self, key: int | Vec):
        if isinstance(key, int):
            return super().__getitem__(key)
        if isinstance(key, Vec):
            try:
                i = key.to_uint()
            except ValueError:
                return _ArrayXPropItem(self._dtype)
            return super().__getitem__(i)
        assert TypeError("Expected key to be int or Vec")


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


def simify(d: Module | Bits | Bit | Array):
    """Add design processes to the simulator."""
    loop = get_loop()
    for node in d.iter_bfs():
        assert isinstance(node, _ProcIf)
        for region, func, args, kwargs in node.procs:
            loop.add_proc(region, func, *args, **kwargs)
