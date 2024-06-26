"""Logic design components.

Combine hierarchy, bit vectors, and simulation semantics into a
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
from .sim import Aggregate, Region, SimAwaitable, Singular, State, Value, changed, get_loop, resume
from .vec import Vec, VecEnum, _lit2vec, cat

_item2char: dict[tuple[int, int], str] = {
    (0, 0): "x",
    (1, 0): "0",
    (0, 1): "1",
    (1, 1): "x",
}


def _vec2vcd(v: Vec) -> VarValue:
    """Convert bit array to VCD value."""
    return "".join(_item2char[v.get_item(i)] for i in range(len(v) - 1, -1, -1))


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

        # Ports: name => connected
        self._inputs = {}
        self._outputs = {}

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

    def input(self, name: str, dtype: type) -> Bits:
        self._check_name(name)
        assert issubclass(dtype, Vec) and dtype.n > 0
        if dtype.n == 1:
            node = Bit(name, parent=self)
        else:
            node = Bits(name, parent=self, dtype=dtype)
        self._inputs[name] = False
        setattr(self, name, node)
        return node

    def output(self, name: str, dtype: type) -> Bits:
        self._check_name(name)
        assert issubclass(dtype, Vec) and dtype.n > 0
        if dtype.n == 1:
            node = Bit(name, parent=self)
        else:
            node = Bits(name, parent=self, dtype=dtype)
        self._outputs[name] = False
        setattr(self, name, node)
        return node

    def connect(self, **ports):
        for name, rhs in ports.items():
            lhs = getattr(self, name)
            if isinstance(rhs, Bits):
                if name in self._inputs:
                    self.assign(lhs, rhs)
                elif name in self._outputs:
                    self.assign(rhs, lhs)
                else:
                    raise ValueError(f"Invalid port: {name}")
            elif isinstance(rhs, tuple):
                if name in self._inputs:
                    self.combi(lhs, rhs[0], *rhs[1:])
                elif name in self._outputs:
                    self.combi(rhs[1:], rhs[0], lhs)
                else:
                    raise ValueError(f"Invalid port: {name}")
            else:
                raise ValueError(f"Port {name} invalid connection")

    def bits(self, name: str, dtype: type, port: bool = False) -> Bits:
        self._check_name(name)
        node = Bits(name, parent=self, dtype=dtype)
        setattr(self, f"_{name}", node)
        if port:
            setattr(self, name, node)
        return node

    def bit(self, name: str, port: bool = False) -> Bit:
        self._check_name(name)
        node = Bit(name, parent=self)
        setattr(self, f"_{name}", node)
        if port:
            setattr(self, name, node)
        return node

    def array(self, name: str, dtype: type) -> Array:
        self._check_name(name)
        node = Array(name, parent=self, dtype=dtype)
        setattr(self, f"_{name}", node)
        return node

    def submod(self, name: str, mod: type[Module], **params) -> Module:
        self._check_name(name)
        node = mod(name, parent=self, **params)
        setattr(self, f"_{name}", node)
        return node

    def combi(self, ys: Value | tuple[Value, ...], f: Callable, *xs: Bits | Array):
        """Combinational logic."""

        # Pack outputs
        if not isinstance(ys, tuple):
            ys = (ys,)

        async def proc():
            while True:
                await changed(*xs)

                # Get sim var values
                vals = []
                for x in xs:
                    if isinstance(x, Bits):
                        vals.append(x.value)
                    elif isinstance(x, Array):
                        vals.append(x.values)
                    else:
                        raise TypeError("Expected x to be Bits or Array")

                # Apply f to inputs
                vals = f(*vals)

                # Pack inputs
                if not isinstance(vals, tuple):
                    vals = (vals,)

                assert len(ys) == len(vals)
                for y, val in zip(ys, vals):
                    y.next = val

        self._procs.append((Region.REACTIVE, proc, (), {}))

    def assign(self, y: Value, x):
        """Assign input to output."""
        # fmt: off
        if isinstance(x, Bits):
            async def proc1():
                while True:
                    await changed(x)
                    y.next = x.value
            self._procs.append((Region.REACTIVE, proc1, (), {}))
        else:
            async def proc2():
                y.next = x
            self._procs.append((Region.ACTIVE, proc2, (), {}))
        # fmt: on

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

    def dff_ar(self, q: Bits, d: Bits, clk: Bit, rst: Bit, rval):
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

    def dff_en(self, q: Bits, d: Bits, en: Bit, clk: Bit):
        """D Flip Flop with enable."""

        async def proc():
            while True:
                state = await resume(
                    (clk, lambda: clk.is_posedge() and en.value == "1b1"),
                )
                if state is clk:
                    q.next = d.value
                else:
                    assert False

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def dff_en_ar(self, q: Bits, d: Bits, en: Bit, clk: Bit, rst: Bit, rval):
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

    def mem_wr_en(self, mem: Array, addr: Bits, data: Bits, en: Bit, clk: Bit):
        """Memory with write enable."""

        async def proc():
            while True:
                state = await resume(
                    (clk, lambda: clk.is_posedge() and en.value == "1b1"),
                )
                assert not addr.value.has_unknown()
                if state is clk:
                    mem[addr.value].next = data.value
                else:
                    assert False

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def mem_wr_be(self, mem: Array, addr: Bits, data: Bits, en: Bit, be: Bits, clk: Bit):
        """Memory with write byte enable."""

        width = mem._dtype.n  # pylint: disable = protected-access
        if width % 8 != 0:
            raise ValueError("Expected data width to be multiple of 8")
        nbytes = width // 8

        async def proc():
            while True:
                state = await resume(
                    (clk, lambda: clk.is_posedge() and en.value == "1b1"),
                )
                assert not addr.value.has_unknown()
                assert not be.value.has_unknown()
                if state is clk:
                    xs = []
                    for i in range(nbytes):
                        m, n = 8 * i, 8 * (i + 1)
                        if be.value[i]:
                            xs.append(data.value[m:n])
                        else:
                            xs.append(mem[addr.value].value[m:n])
                    mem[addr.value].next = cat(*xs)
                else:
                    assert False

        self._procs.append((Region.ACTIVE, proc, (), {}))


class Bits(Leaf, Singular, _ProcIf, _TraceIf):
    """Leaf-level bitvector design component."""

    def __init__(self, name: str, parent: Module, dtype: type[Vec]):
        Leaf.__init__(self, name, parent)
        Singular.__init__(self, dtype.xes())
        _ProcIf.__init__(self)
        self._dtype = dtype
        self._waves_change = None
        self._vcd_change = None

    # Singular => State
    def set_next(self, value):
        if isinstance(value, str):
            value = _lit2vec(value)
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

                def f():
                    value = self._next_value.name
                    return vcdw.change(var, self._sim.time, value)

                self._vcd_change = f

        elif issubclass(self._dtype, Vec):
            if re.match(pattern, self.qualname):
                var = vcdw.register_var(
                    scope=self._parent.scope,
                    name=self.name,
                    var_type="reg",
                    size=len(self._value),
                    init=_vec2vcd(self._value),
                )

                def f():
                    value = _vec2vcd(self._next_value)
                    return vcdw.change(var, self._sim.time, value)

                self._vcd_change = f


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

    def __init__(self, name: str, parent: Module, dtype: type[Vec]):
        Leaf.__init__(self, name, parent)
        Aggregate.__init__(self, dtype.xes())
        _ProcIf.__init__(self)
        self._dtype = dtype


def simify(d: Module | Bits | Bit | Array):
    """Add design processes to the simulator."""
    loop = get_loop()
    for node in d.iter_bfs():
        assert isinstance(node, _ProcIf)
        for region, func, args, kwargs in node.procs:
            loop.add_proc(region, func, *args, **kwargs)
