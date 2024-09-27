"""Logic design components.

Combine hierarchy, bit vectors, and simulation semantics into a
straightforward API for creating a digital design.
"""

# pylint: disable=exec-used

# PyLint is confused by metaclass implementation
# pylint: disable=protected-access

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Coroutine, Sequence

from vcd.writer import VCDWriter as VcdWriter

from .bits import Bits, _lit2vec, stack
from .expr import Expr, Variable
from .hier import Branch, Leaf
from .sim import Aggregate, ProcIf, Region, Singular, State, Task, Value, changed, resume


class DesignError(Exception):
    """Design Error."""


class _TraceIf:
    """Tracing interface.

    Implemented by components that support debug dump.
    """

    def dump_waves(self, waves: defaultdict, pattern: str):
        """Dump design elements w/ names matching pattern to waves dict."""

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """Dump design elements w/ names matching pattern to VCD file."""


def _mod_init_source(params) -> str:
    """Return source code for Module __init__ method w/ parameters."""
    lines = []
    s = ", ".join(f"{pn}=None" for pn, _, _ in params)
    lines.append(f"def init(self, name: str, parent: Module | None=None, {s}):\n")
    s = ", ".join(pn for pn, _, _ in params)
    lines.append(f"    _init_body(self, name, parent, {s})\n")
    return "".join(lines)


class _ModuleMeta(type):
    """Module metaclass, for parameterization."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "Module":
            return super().__new__(mcs, name, bases, attrs)

        # Get parameters
        params = []
        for pn, pt in attrs.get("__annotations__", {}).items():
            try:
                pv = attrs[pn]
            except KeyError as e:
                s = f"Module {name} param {pn} has no value"
                raise ValueError(s) from e
            if not isinstance(pv, pt):
                s = f"Module {name} param {pn} has invalid type"
                raise TypeError(s)
            params.append((pn, pt, pv))

        # Create Module class
        mod = super().__new__(mcs, name, bases + (Branch, ProcIf, _TraceIf), attrs)

        # Override Module.__init__ method
        def _init_body(self, name: str, parent: Module, *args):
            Branch.__init__(self, name, parent)
            ProcIf.__init__(self)

            for arg, (pn, _, pv) in zip(args, params):
                if arg is not None:
                    # TODO(cjdrake): Check input type?
                    setattr(self, pn, arg)
                else:
                    setattr(self, pn, pv)

            # Ports: name => connected
            self._inputs = {}
            self._outputs = {}

            self.build()

        source = _mod_init_source(params)
        globals_ = {"_init_body": _init_body}
        locals_ = {}
        exec(source, globals_, locals_)
        mod.__init__ = locals_["init"]

        return mod


class Module(metaclass=_ModuleMeta):
    """Hierarchical, branch-level design component.

    A module contains:
    * Submodules
    * Ports
    * Local variables
    * Local processes
    """

    def build(self):
        raise NotImplementedError()

    def elab(self):
        """Add design processes to the simulator."""
        for node in self.iter_bfs():
            assert isinstance(node, ProcIf)
            node.add_initial()

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

    def input(self, name: str, dtype: type[Bits]) -> Packed:
        # Require valid port name and type
        self._check_name(name)
        # Require ports to have unique name
        if hasattr(self, name):
            raise DesignError(f"Invalid input port name: {name}")
        # Create port
        node = Packed(name, parent=self, dtype=dtype)
        # Mark port unconnected
        self._inputs[name] = False
        # Save port in module namespace
        setattr(self, name, node)
        return node

    def output(self, name: str, dtype: type[Bits]) -> Packed:
        # Require valid port name and type
        self._check_name(name)
        # Require ports to have unique name
        if hasattr(self, name):
            raise DesignError(f"Invalid output port name: {name}")
        # Create port
        node = Packed(name, parent=self, dtype=dtype)
        # Mark port unconnected
        self._outputs[name] = False
        # Save port in module namespace
        setattr(self, name, node)
        return node

    def _connect_input(self, name: str, rhs):
        y = getattr(self, name)
        if self._inputs[name]:
            raise DesignError(f"Input port {name} already connected")
        match rhs:
            # y = x
            case Packed() as x:
                self.assign(y, x)
            # y = (f, x0, x1, ...)
            case Expr() as ex:
                f, xs = ex.to_func()
                self.combi(y, f, *xs)
            case [Callable() as f, *xs]:
                self.combi(y, f, *xs)
            case _:
                raise DesignError(f"Input port {name} invalid connection")
        # Mark port connected
        self._inputs[name] = True

    def _connect_output(self, name: str, rhs):
        x = getattr(self, name)
        if self._outputs[name]:
            raise DesignError(f"Output port {name} already connected")
        match rhs:
            # x = y
            case Packed() as y:
                self.assign(y, x)
            # x = (f, y0, y1, ...)
            case [Callable() as f, *ys]:
                self.combi(ys, f, x)
            case _:
                raise DesignError(f"Output port {name} invalid connection")
        # Mark port connected
        self._outputs[name] = True

    def connect(self, **ports):
        for name, rhs in ports.items():
            # Input Port
            if name in self._inputs:
                self._connect_input(name, rhs)
            # Output Port
            elif name in self._outputs:
                self._connect_output(name, rhs)
            else:
                raise DesignError(f"Invalid port connection: {name}")

    def logic(
        self,
        name: str,
        dtype: type[Bits],
        shape: tuple[int, ...] | None = None,
    ) -> Packed | Unpacked:
        # Require valid logic name
        self._check_name(name)
        local_name = f"_{name}"
        # Require logic to have unique name
        if hasattr(self, local_name):
            raise DesignError(f"Invalid logic name: {name}")
        # Create logic
        if shape is None:
            node = Packed(name, parent=self, dtype=dtype)
        else:
            # TODO(cjdrake): Support > 1 unpacked dimensions
            assert len(shape) == 1
            node = Unpacked(name, parent=self, dtype=dtype)
        # Save logic in module namespace
        setattr(self, local_name, node)
        return node

    def submod(self, name: str, mod: type[Module], **params) -> Module:
        # Require valid submodule name
        self._check_name(name)
        local_name = f"_{name}"
        # Require submodule to have unique name
        if hasattr(self, local_name):
            raise DesignError(f"Invalid submodule name: {name}")
        # Create submodule
        node = mod(name, parent=self, **params)
        # Save submodule in module namespace
        setattr(self, local_name, node)
        return node

    def drv(self, coro: Coroutine):
        self._initial.append(Task(coro, region=Region.ACTIVE))

    def mon(self, coro: Coroutine):
        self._initial.append(Task(coro, region=Region.INACTIVE))

    def _combi(self, ys: Sequence[Value], f: Callable, xs: Sequence[State]):

        async def cf():
            while True:
                await changed(*xs)

                # Apply f to inputs
                values = f(*[x.state for x in xs])

                # Pack outputs
                if not isinstance(values, (list, tuple)):
                    values = (values,)

                assert len(ys) == len(values)

                for y, value in zip(ys, values):
                    y.next = value

        self._initial.append(Task(cf(), region=Region.REACTIVE))

    def combi(
        self,
        ys: Value | list[Value] | tuple[Value, ...],
        f: Callable,
        *xs: Packed | Unpacked,
    ):
        """Combinational logic."""

        # Pack outputs
        if not isinstance(ys, (list, tuple)):
            ys = (ys,)

        self._combi(ys, f, xs)

    def expr(self, ys: Value | list[Value] | tuple[Value, ...], ex: Expr):
        """Expression logic."""

        # Pack outputs
        if not isinstance(ys, (list, tuple)):
            ys = (ys,)

        f, xs = ex.to_func()
        self._combi(ys, f, xs)

    def assign(self, y: Value, x: Packed | str):
        """Assign input to output."""
        # fmt: off
        if isinstance(x, str):
            async def cf():
                y.next = x
            self._initial.append(Task(cf(), region=Region.ACTIVE))
        elif isinstance(x, Packed):
            async def cf():
                while True:
                    await changed(x)
                    y.next = x.value
            self._initial.append(Task(cf(), region=Region.REACTIVE))
        else:
            raise TypeError("Expected x to be Packed or str")
        # fmt: on

    def dff(self, q: Packed, d: Packed, clk: Packed):
        """D Flip Flop.

        Args:
            q: output
            d: input
            clk: clock w/ positive edge trigger
        """

        async def cf():
            while True:
                state = await resume((clk, clk.is_posedge))
                if state is clk:
                    q.next = d.value
                else:
                    assert False  # pragma: no cover

        self._initial.append(Task(cf(), region=Region.ACTIVE))

    def dff_r(
        self,
        q: Packed,
        d: Packed,
        clk: Packed,
        rst: Packed,
        rval: Bits | str,
        rsync: bool = False,
        rneg: bool = False,
    ):
        """D Flip Flop with reset.

        Args:
            q: output
            d: output
            clk: clock w/ positive edge trigger
            rst: reset
            rval: reset value
            rsync: reset is edge triggered
            rneg: reset is active negative
        """

        # fmt: off
        if rneg:
            def clk_pred():
                return clk.is_posedge() and rst.is_pos()
        else:
            def clk_pred():
                return clk.is_posedge() and rst.is_neg()

        if rsync:
            async def cf():
                while True:
                    state = await resume((clk, clk_pred))
                    if state is clk:
                        ractive = (not rst.value) if rneg else bool(rst.value)
                        q.next = rval if ractive else d.value
                    else:
                        assert False  # pragma: no cover
        else:
            rst_pred = rst.is_negedge if rneg else rst.is_posedge

            async def cf():
                while True:
                    state = await resume((rst, rst_pred), (clk, clk_pred))
                    if state is rst:
                        q.next = rval
                    elif state is clk:
                        q.next = d.value
                    else:
                        assert False  # pragma: no cover

        self._initial.append(Task(cf(), region=Region.ACTIVE))
        # fmt: on

    def dff_en(self, q: Packed, d: Packed, en: Packed, clk: Packed):
        """D Flip Flop with enable.

        Args:
            q: output
            d: input
            en: enable
            clk: clock w/ positive edge trigger
        """

        def clk_pred():
            return clk.is_posedge() and en.value == "1b1"

        async def cf():
            while True:
                state = await resume((clk, clk_pred))
                if state is clk:
                    q.next = d.value
                else:
                    assert False  # pragma: no cover

        self._initial.append(Task(cf(), region=Region.ACTIVE))

    def dff_en_r(
        self,
        q: Packed,
        d: Packed,
        en: Packed,
        clk: Packed,
        rst: Packed,
        rval: Bits | str,
        rsync: bool = False,
        rneg: bool = False,
    ):
        """D Flip Flop with enable, and reset.

        Args:
            q: output
            d: input
            en: enable
            clk: clock w/ positive edge trigger
            rst: reset
            rval: reset value
            rsync: reset is edge triggered
            rneg: reset is active negative
        """

        # fmt: off
        if rneg:
            def clk_pred():
                return clk.is_posedge() and rst.is_pos() and en.value == "1b1"
        else:
            def clk_pred():
                return clk.is_posedge() and rst.is_neg() and en.value == "1b1"

        if rsync:
            async def cf():
                state = await resume((clk, clk_pred))
                if state is clk:
                    ractive = not rst.value if rneg else bool(rst.value)
                    q.next = rval if ractive else d.value
                else:
                    assert False  # pragma: no cover
        else:
            rst_pred = rst.is_negedge if rneg else rst.is_posedge

            async def cf():
                while True:
                    state = await resume((rst, rst_pred), (clk, clk_pred))
                    if state is rst:
                        q.next = rval
                    elif state is clk:
                        q.next = d.value
                    else:
                        assert False  # pragma: no cover

        self._initial.append(Task(cf(), region=Region.ACTIVE))
        # fmt: on

    def mem_wr_en(
        self,
        mem: Unpacked,
        addr: Packed,
        data: Packed,
        en: Packed,
        clk: Packed,
    ):
        """Memory with write enable."""

        def clk_pred():
            return clk.is_posedge() and en.value == "1b1"

        async def cf():
            while True:
                state = await resume((clk, clk_pred))
                assert not addr.value.has_unknown()
                if state is clk:
                    mem[addr.value].next = data.value
                else:
                    assert False  # pragma: no cover

        self._initial.append(Task(cf(), region=Region.ACTIVE))

    def mem_wr_be(
        self,
        mem: Unpacked,
        addr: Packed,
        data: Packed,
        en: Packed,
        be: Packed,
        clk: Packed,
    ):
        """Memory with write byte enable."""

        def clk_pred():
            return clk.is_posedge() and en.value == "1b1"

        # Require mem/data to be Array[N,8]
        assert len(mem.dtype.shape) == 2 and mem.dtype.shape[1] == 8
        assert len(data.dtype.shape) == 2 and data.dtype.shape[1] == 8

        async def cf():
            while True:
                state = await resume((clk, clk_pred))
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

        self._initial.append(Task(cf(), region=Region.ACTIVE))


class Logic(Leaf, ProcIf, _TraceIf):
    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Leaf.__init__(self, name, parent)
        ProcIf.__init__(self)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype


class Packed(Logic, Singular, Variable):
    """Leaf-level bitvector design component."""

    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Logic.__init__(self, name, parent, dtype)
        Singular.__init__(self, dtype.xes())
        Variable.__init__(self, name)
        self._waves_change = None
        self._vcd_change = None

    # Singular => State
    def _set_next(self, value):
        if isinstance(value, str):
            value = _lit2vec(value)
        super()._set_next(self._dtype.cast(value))

    next = property(fset=_set_next)

    def update(self):
        if self._waves_change and self.dirty():
            self._waves_change()
        if self._vcd_change and self.dirty():
            self._vcd_change()
        super().update()

    # TraceIf
    def dump_waves(self, waves: defaultdict, pattern: str):
        if re.fullmatch(pattern, self.qualname):
            # Initial time
            t = self._sim.time()
            waves[t][self] = self._value

            def change():
                t = self._sim.time()
                waves[t][self] = self._next_value

            self._waves_change = change

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        assert isinstance(self._parent, Module)

        if re.fullmatch(pattern, self.qualname):
            var = vcdw.register_var(
                scope=self._parent.scope,
                name=self.name,
                var_type=self._value.vcd_var(),
                size=self._value.size,
                init=self._value.vcd_val(),
            )

            def change():
                t = self._sim.time()
                value = self._next_value.vcd_val()
                return vcdw.change(var, t, value)

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
        """Return True when bit transitions 1 => 0."""
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

    def is_edge(self) -> bool:
        """Return True when bit transitions 0 => 1 or 1 => 0."""
        try:
            return (not self._value and self._next_value) or (self._value and not self._next_value)
        except ValueError:
            return False

    async def posedge(self) -> State:
        """Suspend; resume execution at signal posedge."""
        await resume((self, self.is_posedge))

    async def negedge(self) -> State:
        """Suspend; resume execution at signal negedge."""
        await resume((self, self.is_negedge))

    async def edge(self) -> State:
        """Suspend; resume execution at signal edge."""
        await resume((self, self.is_edge))


class Unpacked(Logic, Aggregate):
    """Leaf-level array of vec/enum/struct/union design components."""

    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Logic.__init__(self, name, parent, dtype)
        Aggregate.__init__(self, dtype.xes())
