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

from vcd.writer import VCDWriter as VcdWriter

from .bits import Bits, _lit2vec, stack
from .hier import Branch, Leaf
from .sim import Aggregate, Region, SimAwaitable, Singular, State, Value, changed, get_loop, resume


class DesignError(Exception):
    """Design Error."""


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

    def input(self, name: str, dtype: type) -> PackedLogic:
        self._check_name(name)
        assert issubclass(dtype, Bits) and dtype.size > 0
        node = PackedLogic(name, parent=self, dtype=dtype)
        self._inputs[name] = False
        setattr(self, name, node)
        return node

    def output(self, name: str, dtype: type) -> PackedLogic:
        self._check_name(name)
        assert issubclass(dtype, Bits) and dtype.size > 0
        node = PackedLogic(name, parent=self, dtype=dtype)
        self._outputs[name] = False
        setattr(self, name, node)
        return node

    def connect(self, **ports):
        for name, rhs in ports.items():
            lhs = getattr(self, name)
            if isinstance(rhs, PackedLogic):
                if name in self._inputs:
                    if not self._inputs[name]:
                        self.assign(lhs, rhs)
                        self._inputs[name] = True
                    else:
                        s = f"Input Port {name} already connected"
                        raise DesignError(s)
                elif name in self._outputs:
                    if not self._outputs[name]:
                        self.assign(rhs, lhs)
                        self._outputs[name] = True
                    else:
                        s = f"Output Port {name} already connected"
                        raise DesignError(s)
                else:
                    raise ValueError(f"Invalid port: {name}")
            elif isinstance(rhs, tuple):
                if name in self._inputs:
                    if not self._inputs[name]:
                        self.combi(lhs, rhs[0], *rhs[1:])
                        self._inputs[name] = True
                    else:
                        s = f"Input Port {name} already connected"
                        raise DesignError(s)
                elif name in self._outputs:
                    if not self._outputs[name]:
                        self.combi(rhs[1:], rhs[0], lhs)
                        self._outputs[name] = True
                    else:
                        s = f"Output Port {name} already connected"
                        raise DesignError(s)
                else:
                    raise ValueError(f"Invalid port: {name}")
            else:
                raise ValueError(f"Port {name} invalid connection")

    def logic(
        self, name: str, dtype: type, shape: tuple[int, ...] | None = None
    ) -> PackedLogic | UnpackedLogic:
        self._check_name(name)
        if shape is None:
            node = PackedLogic(name, parent=self, dtype=dtype)
        else:
            # TODO(cjdrake): Support > 1 unpacked dimensions
            assert len(shape) == 1
            node = UnpackedLogic(name, parent=self, dtype=dtype)
        setattr(self, f"_{name}", node)
        return node

    def submod(self, name: str, mod: type[Module], **params) -> Module:
        self._check_name(name)
        node = mod(name, parent=self, **params)
        setattr(self, f"_{name}", node)
        return node

    def combi(self, ys: Value | tuple[Value, ...], f: Callable, *xs: PackedLogic | UnpackedLogic):
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
                    if isinstance(x, PackedLogic):
                        vals.append(x.value)
                    elif isinstance(x, UnpackedLogic):
                        vals.append(x.values)
                    else:
                        s = "Expected x to be Logic"
                        raise TypeError(s)

                # Apply f to inputs
                vals = f(*vals)

                # Pack inputs
                if not isinstance(vals, tuple):
                    vals = (vals,)

                assert len(ys) == len(vals)
                for y, val in zip(ys, vals):
                    y.next = val

        self._procs.append((Region.REACTIVE, proc, (), {}))

    def assign(self, y: Value, x: Bits | str):
        """Assign input to output."""
        # fmt: off
        if isinstance(x, PackedLogic):
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

    def dff(self, q: PackedLogic, d: PackedLogic, clk: PackedLogic):
        """D Flip Flop."""

        async def proc():
            while True:
                state = await resume((clk, clk.is_posedge))
                if state is clk:
                    q.next = d.value
                else:
                    assert False  # pragma: no cover

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def dff_ar(
        self, q: PackedLogic, d: PackedLogic, clk: PackedLogic, rst: PackedLogic, rval: Bits | str
    ):
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
                    assert False  # pragma: no cover

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def dff_en(self, q: PackedLogic, d: PackedLogic, en: PackedLogic, clk: PackedLogic):
        """D Flip Flop with enable."""

        async def proc():
            while True:
                state = await resume(
                    (clk, lambda: clk.is_posedge() and en.value == "1b1"),
                )
                if state is clk:
                    q.next = d.value
                else:
                    assert False  # pragma: no cover

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def dff_en_ar(
        self,
        q: PackedLogic,
        d: PackedLogic,
        en: PackedLogic,
        clk: PackedLogic,
        rst: PackedLogic,
        rval: Bits | str,
    ):
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
                    assert False  # pragma: no cover

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def mem_wr_en(
        self,
        mem: UnpackedLogic,
        addr: PackedLogic,
        data: PackedLogic,
        en: PackedLogic,
        clk: PackedLogic,
    ):
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
                    assert False  # pragma: no cover

        self._procs.append((Region.ACTIVE, proc, (), {}))

    def mem_wr_be(
        self,
        mem: UnpackedLogic,
        addr: PackedLogic,
        data: PackedLogic,
        en: PackedLogic,
        be: PackedLogic,
        clk: PackedLogic,
    ):
        """Memory with write byte enable."""

        # Require mem/data to be Array[N,8]
        assert len(mem.dtype.shape) == 2 and mem.dtype.shape[1] == 8
        assert len(data.dtype.shape) == 2 and data.dtype.shape[1] == 8

        async def proc():
            while True:
                state = await resume(
                    (clk, lambda: clk.is_posedge() and en.value == "1b1"),
                )
                assert not addr.value.has_unknown()
                assert not be.value.has_unknown()
                if state is clk:
                    xs = []
                    for i, data_en in enumerate(be.value):
                        if data_en:
                            xs.append(data.value[i])
                        else:
                            xs.append(mem[addr.value].value[i])
                    mem[addr.value].next = stack(*xs)
                else:
                    assert False  # pragma: no cover

        self._procs.append((Region.ACTIVE, proc, (), {}))


class Logic(Leaf, _ProcIf, _TraceIf):
    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Leaf.__init__(self, name, parent)
        _ProcIf.__init__(self)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype


class PackedLogic(Logic, Singular):
    """Leaf-level bitvector design component."""

    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Logic.__init__(self, name, parent, dtype)
        Singular.__init__(self, dtype.xes())
        self._waves_change = None
        self._vcd_change = None

    # Singular => State
    def set_next(self, value: Bits | str):
        if isinstance(value, str):
            value = _lit2vec(value)
        super().set_next(self._dtype.cast(value))

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

        if re.match(pattern, self.qualname):
            var = vcdw.register_var(
                scope=self._parent.scope,
                name=self.name,
                var_type=self._value.vcd_var(),
                size=self._value.size,
                init=self._value.vcd_val(),
            )

            def change():
                value = self._next_value.vcd_val()
                return vcdw.change(var, self._sim.time, value)

            self._vcd_change = change

    def is_neg(self) -> bool:
        """Return True when bit is stable 0 => 0."""
        try:
            return not self._value and not self._next_value
        except ValueError:
            return False

    def is_posedge(self) -> bool:
        """Return True when bit transitions 0 => 1."""
        try:
            return not self._value and self._next_value
        except ValueError:
            return False

    def is_negedge(self) -> bool:
        """Return True when bit transition 1 => 0."""
        try:
            return self._value and not self._next_value
        except ValueError:
            return False

    def is_pos(self) -> bool:
        """Return True when bit is stable 1 => 1."""
        try:
            return self._value and self._next_value
        except ValueError:
            return False

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


class UnpackedLogic(Logic, Aggregate):
    """Leaf-level array of vec/enum/struct/union design components."""

    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Logic.__init__(self, name, parent, dtype)
        Aggregate.__init__(self, dtype.xes())


def simify(m: Module | PackedLogic | UnpackedLogic):
    """Add design processes to the simulator."""
    loop = get_loop()
    for node in m.iter_bfs():
        assert isinstance(node, _ProcIf)
        for region, func, args, kwargs in node.procs:
            loop.add_proc(region, func, *args, **kwargs)
