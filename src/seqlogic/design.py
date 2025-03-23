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
from enum import IntEnum, auto

from bvwx import Bits, i2bv, lit2bv, stack, u2bv
from deltacycle import Aggregate, Loop, Singular
from deltacycle import Value as SimVal
from deltacycle import Variable as SimVar
from deltacycle import changed, create_task, now, resume
from vcd.writer import VCDWriter as VcdWriter

from .expr import Expr, Variable
from .hier import Branch, Leaf


class Region(IntEnum):
    # Coroutines that react to changes from Active region.
    # Used by combinational logic.
    REACTIVE = auto()

    # Coroutines that drive changes to model state.
    # Used by 1) testbench, and 2) sequential logic.
    ACTIVE = auto()

    # Coroutines that monitor model state.
    INACTIVE = auto()


class DesignError(Exception):
    """Design Error."""


class _ProcIf:
    """Process interface.

    Implemented by components that contain local simulator processes.
    """

    def __init__(self):
        self._initial: list[tuple[Coroutine, Region]] = []

    @property
    def initial(self):
        return self._initial


class _TraceIf:
    """Tracing interface.

    Implemented by components that support debug dump.
    """

    def dump_waves(self, waves: defaultdict, pattern: str):
        """Dump design elements w/ names matching pattern to waves dict."""

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """Dump design elements w/ names matching pattern to VCD file."""


_ModuleParams = defaultdict(dict)


def _get_pmod(mod: type[Module], params) -> type[Module]:
    """Return parameterized Module[p1=v1,p2=v2,...] type."""
    key = tuple(sorted(params.items()))
    try:
        return _ModuleParams[mod][key]
    except KeyError:
        s = ",".join(f"{k}={v}" for k, v in key)
        pmod = type(f"{mod.__name__}[{s}]", (mod,), params)
        _ModuleParams[mod][key] = pmod
        return pmod


def _mod_paramz_source(pntvs) -> str:
    """Return source code for Module paramz method."""
    lines = []
    s = ", ".join(f"{pn}=None" for pn, _, _ in pntvs)
    lines.append(f"def paramz(cls, {s}):\n")
    s = ", ".join(pn for pn, _, _ in pntvs)
    lines.append(f"    return _paramz_body(cls, {s})\n")
    return "".join(lines)


class _ModuleMeta(type):
    """Module metaclass, for parameterization."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "Module":
            return super().__new__(mcs, name, bases, attrs)

        # Get parameter names, types, default values
        pntvs = []
        for pn, pt in attrs.get("__annotations__", {}).items():
            try:
                pv = attrs[pn]
            except KeyError as e:
                s = f"Module {name} param {pn} has no default value"
                raise ValueError(s) from e
            if not isinstance(pv, pt):
                s = f"Module {name} param {pn} has invalid type"
                raise TypeError(s)
            pntvs.append((pn, pt, pv))

        # Create Module class
        mod = super().__new__(mcs, name, bases + (Branch, _ProcIf, _TraceIf), attrs)

        # Override Module.__init__ method
        def _init(self, name: str, parent: Module | None = None):
            Branch.__init__(self, name, parent)
            _ProcIf.__init__(self)

            # Ports: name => connected
            self._inputs = {}
            self._outputs = {}

            self.build()

        mod.__init__ = _init

        def _paramz_body(cls, *args):
            # No parameters
            if not pntvs:
                return cls

            # For all parameters, use either default or override value
            params = {}
            for arg, (pn, _, pv) in zip(args, pntvs):
                if arg is None:
                    params[pn] = pv  # default
                else:
                    # TODO(cjdrake): Check input type?
                    params[pn] = arg  # override

            return _get_pmod(cls, params)

        source = _mod_paramz_source(pntvs)
        globals_ = {"_paramz_body": _paramz_body}
        locals_ = {}
        exec(source, globals_, locals_)
        mod.paramz = classmethod(locals_["paramz"])
        mod.parameterize = mod.paramz

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
        raise NotImplementedError("Module requires a build method")

    def main(self) -> Coroutine:
        """Add design processes to the simulator."""

        async def cf():
            for node in self.iter_bfs():
                for coro, region in node.initial:
                    create_task(coro, region)

        return cf()

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
        # Require logic to have unique name
        if hasattr(self, name):
            raise DesignError(f"Invalid logic name: {name}")
        # Create logic
        if shape is None:
            node = Packed(name, parent=self, dtype=dtype)
        else:
            # TODO(cjdrake): Support > 1 unpacked dimensions
            assert len(shape) == 1
            node = Unpacked(name, parent=self, dtype=dtype)
        # Save logic in module namespace
        setattr(self, name, node)
        return node

    def float(self, name: str) -> Float:
        # Require valid logic name
        self._check_name(name)
        # Require logic to have unique name
        if hasattr(self, name):
            raise DesignError(f"Invalid float name: {name}")
        # Create logic
        node = Float(name, parent=self)
        # Save logic in module namespace
        setattr(self, name, node)
        return node

    def submod(self, name: str, mod: type[Module]) -> Module:
        # Require valid submodule name
        self._check_name(name)
        # Require submodule to have unique name
        if hasattr(self, name):
            raise DesignError(f"Invalid submodule name: {name}")
        # Create submodule
        node = mod(name, parent=self)
        # Save submodule in module namespace
        setattr(self, name, node)
        return node

    def drv(self, coro: Coroutine):
        self._initial.append((coro, Region.ACTIVE))

    def mon(self, coro: Coroutine):
        self._initial.append((coro, Region.INACTIVE))

    def _combi(self, ys: Sequence[SimVal], f: Callable, xs: Sequence[SimVar]):

        async def cf():
            while True:
                await changed(*xs)

                # Apply f to inputs
                values = f(*[x.value for x in xs])

                # Pack outputs
                if not isinstance(values, (list, tuple)):
                    values = (values,)

                assert len(ys) == len(values)

                for y, value in zip(ys, values):
                    y.next = value

        self._initial.append((cf(), Region.REACTIVE))

    def combi(
        self,
        ys: SimVal | list[SimVal] | tuple[SimVal, ...],
        f: Callable,
        *xs: Packed | Unpacked,
    ):
        """Combinational logic."""

        # Pack outputs
        if not isinstance(ys, (list, tuple)):
            ys = (ys,)

        self._combi(ys, f, xs)

    def expr(self, ys: SimVal | list[SimVal] | tuple[SimVal, ...], ex: Expr):
        """Expression logic."""

        # Pack outputs
        if not isinstance(ys, (list, tuple)):
            ys = (ys,)

        f, xs = ex.to_func()
        self._combi(ys, f, xs)

    def assign(self, y: SimVal, x: Packed | str):
        """Assign input to output."""
        # fmt: off
        if isinstance(x, str):
            async def cf():
                y.next = x
            self._initial.append((cf(), Region.ACTIVE))
        elif isinstance(x, Packed):
            async def cf():
                while True:
                    await changed(x)
                    y.next = x.value
            self._initial.append((cf(), Region.REACTIVE))
        else:
            raise TypeError("Expected x to be Packed or str")
        # fmt: on

    def dff(
        self,
        q: Packed,
        d: Packed,
        clk: Packed,
        en: Packed | None = None,
        rst: Packed | None = None,
        rval: Bits | str | None = None,
        rsync: bool = False,
        rneg: bool = False,
    ):
        """D Flip Flop with enable, and reset.

        Args:
            q: output
            d: input
            clk: clock w/ positive edge trigger
            en: enable
            rst: reset
            rval: reset value
            rsync: reset is edge triggered
            rneg: reset is active negative
        """
        # fmt: off
        if en is None:
            clk_en = clk.is_posedge
        else:
            def clk_en():
                return clk.is_posedge() and en.prev == "1b1"

        # No Reset
        if rst is None:
            async def cf():
                while True:
                    x = await resume((clk, clk_en))
                    if x is clk:
                        q.next = d.prev
                    else:
                        assert False  # pragma: no cover

        # Reset
        else:
            if rval is None:
                rval = q.dtype.zeros()

            # Synchronous Reset
            if rsync:
                if rneg:
                    async def cf():
                        x = await resume((clk, clk_en))
                        if x is clk:
                            q.next = rval if not rst.prev else d.prev
                        else:
                            assert False  # pragma: no cover
                else:
                    async def cf():
                        x = await resume((clk, clk_en))
                        if x is clk:
                            q.next = rval if rst.prev else d.prev
                        else:
                            assert False  # pragma: no cover

            # Asynchronous Reset
            else:
                if rneg:
                    rst_pred = rst.is_negedge
                    def clk_pred():  # noqa
                        return clk_en() and rst.is_pos()
                else:
                    rst_pred = rst.is_posedge
                    def clk_pred():  # noqa
                        return clk_en() and rst.is_neg()

                async def cf():
                    while True:
                        x = await resume((rst, rst_pred), (clk, clk_pred))
                        if x is rst:
                            q.next = rval
                        elif x is clk:
                            q.next = d.prev
                        else:
                            assert False  # pragma: no cover

        self._initial.append((cf(), Region.ACTIVE))
        # fmt: on

    def mem_wr(
        self,
        mem: Unpacked,
        addr: Packed,
        data: Packed,
        clk: Packed,
        en: Packed,
        be: Packed | None = None,
    ):
        """Memory with write enable."""

        def clk_pred():
            return clk.is_posedge() and en.prev == "1b1"

        # fmt: off
        if be is None:
            async def cf():
                while True:
                    x = await resume((clk, clk_pred))
                    assert not addr.prev.has_unknown()
                    if x is clk:
                        mem[addr.prev].next = data.prev
                    else:
                        assert False  # pragma: no cover
        else:
            # Require mem/data to be Array[N,8]
            assert len(mem.dtype.shape) == 2 and mem.dtype.shape[1] == 8
            assert len(data.dtype.shape) == 2 and data.dtype.shape[1] == 8

            async def cf():
                while True:
                    x = await resume((clk, clk_pred))
                    assert not addr.prev.has_unknown()
                    assert not be.prev.has_unknown()
                    if x is clk:
                        xs = []
                        for i, data_en in enumerate(be.prev):
                            if data_en:
                                xs.append(data.prev[i])
                            else:
                                xs.append(mem[addr.prev].prev[i])
                        mem[addr.prev].next = stack(*xs)
                    else:
                        assert False  # pragma: no cover

        self._initial.append((cf(), Region.ACTIVE))
        # fmt: on


class Logic(Leaf, _ProcIf, _TraceIf):
    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Leaf.__init__(self, name, parent)
        _ProcIf.__init__(self)
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

    # Singular => Variable
    def _set_next(self, value):
        if isinstance(value, str):
            value = lit2bv(value)
        elif isinstance(value, int):
            if value < 0:
                value = i2bv(value, size=self._dtype.size)
            else:
                value = u2bv(value, size=self._dtype.size)
        super()._set_next(self._dtype.cast(value))

    next = property(fset=_set_next)

    def update(self):
        if self._waves_change and (self._next != self._prev):
            self._waves_change()
        if self._vcd_change and (self._next != self._prev):
            self._vcd_change()
        super().update()

    # TraceIf
    def dump_waves(self, waves: defaultdict, pattern: str):
        if re.fullmatch(pattern, self.qualname):
            # Initial time
            waves[Loop.init_time][self] = self._prev

            def change():
                t = now()
                waves[t][self] = self._next

            self._waves_change = change

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        assert isinstance(self._parent, Module)

        if re.fullmatch(pattern, self.qualname):
            var = vcdw.register_var(
                scope=self._parent.scope,
                name=self.name,
                var_type=self._prev.vcd_var(),
                size=self._prev.size,
                init=self._prev.vcd_val(),
            )

            def change():
                t = now()
                value = self._next.vcd_val()
                return vcdw.change(var, t, value)

            self._vcd_change = change

    def is_neg(self) -> bool:
        """Return True when bit is stable 0 => 0."""
        try:
            return not self._prev and not self._next
        except ValueError:
            return False

    def is_posedge(self) -> bool:
        """Return True when bit transitions 0 => 1."""
        try:
            return not self._prev and self._next
        except ValueError:
            return False

    def is_negedge(self) -> bool:
        """Return True when bit transitions 1 => 0."""
        try:
            return self._prev and not self._next
        except ValueError:
            return False

    def is_pos(self) -> bool:
        """Return True when bit is stable 1 => 1."""
        try:
            return self._prev and self._next
        except ValueError:
            return False

    def is_edge(self) -> bool:
        """Return True when bit transitions 0 => 1 or 1 => 0."""
        try:
            return (not self._prev and self._next) or (self._prev and not self._next)
        except ValueError:
            return False

    async def posedge(self):
        """Suspend; resume execution at signal posedge."""
        await resume((self, self.is_posedge))

    async def negedge(self):
        """Suspend; resume execution at signal negedge."""
        await resume((self, self.is_negedge))

    async def edge(self):
        """Suspend; resume execution at signal edge."""
        await resume((self, self.is_edge))


class Unpacked(Logic, Aggregate):
    """Leaf-level array of vec/enum/struct/union design components."""

    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Logic.__init__(self, name, parent, dtype)
        Aggregate.__init__(self, dtype.xes())


class Float(Leaf, _ProcIf, _TraceIf, Singular):
    """Leaf-level float design component."""

    def __init__(self, name: str, parent: Module):
        Leaf.__init__(self, name, parent)
        _ProcIf.__init__(self)
        Singular.__init__(self, float())
        self._waves_change = None
        self._vcd_change = None

    def update(self):
        if self._waves_change and (self._next != self._prev):
            self._waves_change()
        if self._vcd_change and (self._next != self._prev):
            self._vcd_change()
        super().update()

    # TraceIf
    def dump_waves(self, waves: defaultdict, pattern: str):
        if re.fullmatch(pattern, self.qualname):
            # Initial time
            waves[Loop.init_time][self] = self._prev

            def change():
                t = now()
                waves[t][self] = self._next

            self._waves_change = change

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        assert isinstance(self._parent, Module)

        if re.fullmatch(pattern, self.qualname):
            var = vcdw.register_var(
                scope=self._parent.scope,
                name=self.name,
                var_type="real",
                init=float(),
            )

            def change():
                t = now()
                value = self._next
                return vcdw.change(var, t, value)

            self._vcd_change = change
