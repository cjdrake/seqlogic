"""Logic design components.

Combine hierarchy, bit vectors, and simulation semantics into a
straightforward API for creating a digital design.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable, Coroutine, Sequence
from enum import IntEnum

from bvwx import Bits, i2bv, lit2bv, stack, u2bv
from deltacycle import Aggregate, Loop, Singular, TaskGroup, any_var, get_running_loop, now
from deltacycle import Value as SimVal
from deltacycle import Variable as SimVar
from vcd.writer import VCDWriter as VcdWriter

from ._expr import Expr
from ._expr import Variable as ExprVar
from ._hier import Branch, Leaf

logger = logging.getLogger("deltacycle")


class Region(IntEnum):
    # Coroutines that react to changes from Active region.
    # Used by combinational logic.
    REACTIVE = -1

    # Coroutines that drive changes to model state.
    # Used by 1) testbench, and 2) sequential logic.
    ACTIVE = 0

    # Coroutines that monitor model state.
    INACTIVE = 1


class DesignError(Exception):
    """Design Error."""


class _ProcIf:
    """Process interface.

    Implemented by components that contain local simulator processes.
    """

    def __init__(self):
        self._reactive: list[Coroutine] = []
        self._active: list[Coroutine] = []
        self._inactive: list[Coroutine] = []

    @property
    def reactive(self) -> list[Coroutine]:
        return self._reactive

    @property
    def active(self) -> list[Coroutine]:
        return self._active

    @property
    def inactive(self) -> list[Coroutine]:
        return self._inactive


class _TraceIf:
    """Tracing interface.

    Implemented by components that support debug dump.
    """

    def dump_waves(self, waves: defaultdict, pattern: str):
        """Dump design elements w/ names matching pattern to waves dict."""

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        """Dump design elements w/ names matching pattern to VCD file."""


def _mod_factory_new(pntvs):
    # Create source for _new function
    lines = []
    s = ", ".join(f"{pn}=None" for pn, _, _ in pntvs)
    lines.append(f"def _new(cls, {s}):\n")
    s = ", ".join(pn for pn, _, _ in pntvs)
    lines.append(f"    return _new_body(cls, {s})\n")
    source = "".join(lines)

    def _new_body(cls, *args):
        # For all parameters, use either default or override value
        params = {}
        for arg, (pn, _, pv) in zip(args, pntvs):
            if arg is None:
                params[pn] = pv  # default
            else:
                # TODO(cjdrake): Check input type?
                params[pn] = arg  # override
        return _get_mod(cls, params)

    globals_ = {"_new_body": _new_body}
    locals_ = {}
    # Create _new method
    exec(source, globals_, locals_)
    return locals_["_new"]


_Modules = defaultdict(dict)


def _mod_name(cls, key) -> str:
    parts = []
    for name, value in key:
        if isinstance(value, type):
            parts.append(f"{name}={value.__name__}")
        else:
            parts.append(f"{name}={value}")
    return f"{cls.__name__}[{','.join(parts)}]"


def _mod_new(cls, name: str, parent: Module | None = None):
    return object.__new__(cls)


def _get_mod(cls, params) -> type[Module]:
    key = tuple(sorted(params.items()))
    try:
        mod = _Modules[cls][key]
    except KeyError:
        # Create extended Module
        name = _mod_name(cls, key)
        mod = type(name, (cls,), params)
        mod.__new__ = _mod_new
        _Modules[cls][key] = mod
    return mod


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

        if pntvs:
            # Create Module factory
            mod_factory = super().__new__(mcs, name, bases, attrs)
            mod_factory.__new__ = _mod_factory_new(pntvs)
            return mod_factory

        # Create base Module
        mod = super().__new__(mcs, name, bases + (Branch, _ProcIf, _TraceIf), attrs)
        mod.__new__ = _mod_new
        return mod


class Module(metaclass=_ModuleMeta):
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
        self._inputs: dict[str, bool] = {}
        self._outputs: dict[str, bool] = {}

        self.build()

    def build(self):
        raise NotImplementedError("Module requires a build method")

    def main(self) -> Coroutine:
        """Add design processes to the simulator."""
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        async def cf():
            async with TaskGroup() as tg:
                for node in self.iter_bfs():
                    assert isinstance(node, _ProcIf)
                    for coro in node.reactive:
                        tg.create_task(coro, priority=Region.REACTIVE)
                    for coro in node.active:
                        tg.create_task(coro, priority=Region.ACTIVE)
                    for coro in node.inactive:
                        tg.create_task(coro, priority=Region.INACTIVE)

        return cf()

    def _check_unique(self, name: str, description: str):
        if hasattr(self, name):
            raise DesignError(f"Invalid {description} name: {name}")

    @property
    def scope(self) -> str:
        """Return the branch's full name using dot separator syntax."""
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        if self._parent is None:
            return self.name
        assert isinstance(self._parent, Module)
        return f"{self._parent.scope}.{self.name}"

    def dump_waves(self, waves: defaultdict, pattern: str):
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        for child in self._children:
            assert isinstance(child, _TraceIf)
            child.dump_waves(waves, pattern)

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        for child in self._children:
            assert isinstance(child, _TraceIf)
            child.dump_vcd(vcdw, pattern)

    def input(self, name: str, dtype: type[Bits]) -> Packed:
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        # Require valid and unique name
        self._check_name(name)
        self._check_unique(name, "input port")

        # Create port
        node = Packed(name, parent=self, dtype=dtype)

        # Mark port unconnected
        self._inputs[name] = False

        # Save port in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def output(self, name: str, dtype: type[Bits]) -> Packed:
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        # Require valid and unique name
        self._check_name(name)
        self._check_unique(name, "output port")

        # Create port
        node = Packed(name, parent=self, dtype=dtype)

        # Mark port unconnected
        self._outputs[name] = False

        # Save port in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def _connect_input(self, name: str, rhs):
        y = getattr(self, name)

        if self._inputs[name]:
            raise DesignError(f"Input port {name} already connected")

        # Implement the connection
        match rhs:
            # y = x
            case Packed() as x:
                self.assign(y, x)
            case Expr() as ex:
                f, xs = ex.to_func()
                self.combi(y, f, *xs)
            # y = (f, x0, x1, ...)
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

        # Implement the connection
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
                raise DesignError(f"Invalid port name: {name}")

    def logic(
        self,
        name: str,
        dtype: type[Bits],
        shape: tuple[int, ...] | None = None,
    ) -> Packed | Unpacked:
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        # Require valid and unique name
        self._check_name(name)
        self._check_unique(name, "logic")

        # Create logic
        if shape is None:
            node = Packed(name, parent=self, dtype=dtype)
        else:
            # TODO(cjdrake): Support > 1 unpacked dimensions
            assert len(shape) == 1
            node = Unpacked(name, parent=self, dtype=dtype)

        # Save in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def float(self, name: str) -> Float:
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        # Require valid and unique name
        self._check_name(name)
        self._check_unique(name, "float")

        # Create float
        node = Float(name, parent=self)

        # Save in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def submod(self, name: str, mod: type[Module]) -> Module:
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        # Require valid and unique name
        self._check_name(name)
        self._check_unique(name, "submodule")

        # Create submodule
        node = mod(name, parent=self)

        # Save in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def drv(self, coro: Coroutine):
        assert isinstance(self, _ProcIf)
        self._active.append(coro)

    def mon(self, coro: Coroutine):
        assert isinstance(self, _ProcIf)
        self._inactive.append(coro)

    def _combi(self, ys: tuple[SimVal, ...], f: Callable, xs: list[SimVar]):
        assert isinstance(self, _ProcIf)

        async def cf():
            vps = {x: x.changed for x in xs}
            while True:
                await any_var(vps)

                # Apply f to inputs
                values = f(*[x.value for x in xs])

                # Pack outputs
                if not isinstance(values, (list, tuple)):
                    values = (values,)

                assert len(ys) == len(values)

                for y, value in zip(ys, values):
                    y.next = value

        coro = cf()
        self._reactive.append(coro)

    def combi(self, ys: SimVal | Sequence[SimVal], f: Callable, *xs: Logic):
        """Combinational logic."""

        # Pack outputs
        if isinstance(ys, SimVal):
            ys = (ys,)
        elif isinstance(ys, Sequence) and all(isinstance(y, SimVal) for y in ys):
            ys = tuple(ys)
        else:
            raise TypeError("Expected ys to be Simval or [SimVal]")

        self._combi(ys, f, xs)

    def expr(self, ys: SimVal | Sequence[SimVal], ex: Expr):
        """Expression logic."""

        # Pack outputs
        if isinstance(ys, SimVal):
            ys = (ys,)
        elif isinstance(ys, Sequence) and all(isinstance(y, SimVal) for y in ys):
            ys = tuple(ys)
        else:
            raise TypeError("Expected ys to be Simval or [SimVal]")

        f, xs = ex.to_func()
        self._combi(ys, f, xs)

    def assign(self, y: SimVal, x: Packed | str):
        """Assign input to output."""
        assert isinstance(self, _ProcIf)

        # fmt: off
        if isinstance(x, str):
            async def cf():
                y.next = x
            coro = cf()
            self._active.append(coro)
        elif isinstance(x, Packed):
            async def cf():
                while True:
                    await x
                    y.next = x.value
            coro = cf()
            self._reactive.append(coro)
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
        assert isinstance(self, _ProcIf)

        # fmt: off
        if en is None:
            clk_en = clk.is_posedge
        else:
            def clk_en() -> bool:
                return clk.is_posedge() and en.prev == "1b1"

        # No Reset
        if rst is None:
            async def cf():
                vps = {clk: clk_en}
                while True:
                    x = await any_var(vps)
                    assert x is clk
                    q.next = d.prev

        # Reset
        else:
            if rval is None:
                rval = q.dtype.zeros()

            # Synchronous Reset
            if rsync:
                if rneg:
                    async def cf():
                        vps = {clk: clk_en}
                        while True:
                            x = await any_var(vps)
                            assert x is clk
                            q.next = rval if not rst.prev else d.prev
                else:
                    async def cf():
                        vps = {clk: clk_en}
                        while True:
                            x = await any_var(vps)
                            assert x is clk
                            q.next = rval if rst.prev else d.prev

            # Asynchronous Reset
            else:
                if rneg:
                    rst_pred = rst.is_negedge
                    def clk_pred() -> bool:
                        return clk_en() and rst.is_pos()
                else:
                    rst_pred = rst.is_posedge
                    def clk_pred() -> bool:
                        return clk_en() and rst.is_neg()

                async def cf():
                    vps = {rst: rst_pred, clk: clk_pred}
                    while True:
                        x = await any_var(vps)
                        if x is rst:
                            q.next = rval
                        elif x is clk:
                            q.next = d.prev
                        else:
                            assert False  # pragma: no cover

        coro = cf()
        self._active.append(coro)
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
        assert isinstance(self, _ProcIf)

        def clk_pred() -> bool:
            return clk.is_posedge() and en.prev == "1b1"

        # fmt: off
        if be is None:
            async def cf():
                vps = {clk: clk_pred}
                while True:
                    x = await any_var(vps)
                    assert x is clk
                    assert not addr.prev.has_unknown()
                    mem[addr.prev].next = data.prev
        else:
            # Require mem/data to be Array[N,8]
            assert len(mem.dtype.shape) == 2 and mem.dtype.shape[1] == 8
            assert len(data.dtype.shape) == 2 and data.dtype.shape[1] == 8

            async def cf():
                vps = {clk: clk_pred}
                while True:
                    x = await any_var(vps)
                    assert x is clk
                    assert not addr.prev.has_unknown()
                    assert not be.prev.has_unknown()
                    xs = []
                    for i, data_en in enumerate(be.prev):
                        if data_en:
                            xs.append(data.prev[i])
                        else:
                            xs.append(mem[addr.prev].prev[i])
                    mem[addr.prev].next = stack(*xs)

        coro = cf()
        self._active.append(coro)
        # fmt: on

    def _check_func(
        self,
        C: type[Checker],
        name: str,
        p: Expr,
        f,
        xs: Sequence[SimVar],
        clk: Packed,
        rst: Packed,
        rsync: bool,
        rneg: bool,
        msg: str | None,
    ) -> Checker:
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        E = {Assumption: AssumeError, Assertion: AssertError}[C]

        # Require valid and unique name
        self._check_name(name)
        self._check_unique(name, "checker")

        node = C(name, parent=self)

        p_f, p_xs = p.to_func()

        def _check():
            y = f(*[x.value for x in xs])
            if not y:
                args = () if msg is None else (msg,)
                raise E(*args)

        # fmt: off

        # Synchronous Reset
        if rsync:
            raise NotImplementedError("Sync Reset not implemented yet")

        # Asynchronous Reset
        else:
            if rneg:
                rst_pred = rst.is_negedge
                def clk_pred() -> bool:
                    return clk.is_posedge() and rst.is_pos()
            else:
                rst_pred = rst.is_posedge
                def clk_pred() -> bool:
                    return clk.is_posedge() and rst.is_neg()

            async def cf():
                on = False
                vps = {rst: rst_pred, clk: clk_pred}
                while True:
                    x = await any_var(vps)
                    if x is rst:
                        on = True
                    elif x is clk:
                        if on:
                            en = p_f(*[x.value for x in p_xs])
                            if en:
                                _check()
                    else:
                        assert False  # pragma: no cover
        # fmt: on

        coro = cf()
        node._inactive.append(coro)

        # Save in module namespace
        setattr(self, name, node)
        return node

    def assume_expr(
        self,
        name: str,
        p: Expr,
        ex: Expr,
        clk: Packed,
        rst: Packed,
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assumption:
        f, xs = ex.to_func()
        return self._check_func(Assumption, name, p, f, xs, clk, rst, rsync, rneg, msg)

    def assert_expr(
        self,
        name: str,
        p: Expr,
        ex: Expr,
        clk: Packed,
        rst: Packed,
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assertion:
        f, xs = ex.to_func()
        return self._check_func(Assertion, name, p, f, xs, clk, rst, rsync, rneg, msg)

    def assume_func(
        self,
        name: str,
        p: Expr,
        f,
        xs: Sequence[SimVar],
        clk: Packed,
        rst: Packed,
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assumption:
        return self._check_func(Assumption, name, p, f, xs, clk, rst, rsync, rneg, msg)

    def assert_func(
        self,
        name: str,
        p: Expr,
        f,
        xs: Sequence[SimVar],
        clk: Packed,
        rst: Packed,
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assertion:
        return self._check_func(Assertion, name, p, f, xs, clk, rst, rsync, rneg, msg)

    def _check_seq(
        self,
        C: type[Checker],
        name: str,
        p: Expr,
        s,
        xs: Sequence[SimVar],
        clk: Packed,
        rst: Packed,
        rsync: bool,
        rneg: bool,
        msg: str | None,
    ) -> Checker:
        # Help type checker w/ metaclass
        assert isinstance(self, Branch)

        E = {Assumption: AssumeError, Assertion: AssertError}[C]

        # Require valid and unique name
        self._check_name(name)
        self._check_unique(name, "checker")

        node = C(name, parent=self)

        p_f, p_xs = p.to_func()

        async def _check():
            y = await s(*xs)
            if not y:
                args = () if msg is None else (msg,)
                raise E(*args)

        # fmt: off

        # Synchronous Reset
        if rsync:
            raise NotImplementedError("Sync Reset not implemented yet")

        # Asynchronous Reset
        else:
            if rneg:
                rst_pred = rst.is_negedge
                def clk_pred() -> bool:
                    return clk.is_posedge() and rst.is_pos()
            else:
                rst_pred = rst.is_posedge
                def clk_pred() -> bool:
                    return clk.is_posedge() and rst.is_neg()

            async def cf():
                loop = get_running_loop()
                task = loop.task()

                on = False
                vps = {rst: rst_pred, clk: clk_pred}
                while True:
                    x = await any_var(vps)
                    if x is rst:
                        on = True
                    elif x is clk:
                        if on:
                            en = p_f(*[x.value for x in p_xs])
                            if en:
                                task.group.create_task(_check(), priority=Region.INACTIVE)
                    else:
                        assert False  # pragma: no cover
        # fmt: on

        coro = cf()
        node._inactive.append(coro)

        # Save in module namespace
        setattr(self, name, node)
        return node

    def assume_seq(
        self,
        name: str,
        p: Expr,
        s,
        xs: Sequence[SimVar],
        clk: Packed,
        rst: Packed,
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assumption:
        return self._check_seq(Assumption, name, p, s, xs, clk, rst, rsync, rneg, msg)

    def assert_seq(
        self,
        name: str,
        p: Expr,
        s,
        xs: Sequence[SimVar],
        clk: Packed,
        rst: Packed,
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assertion:
        return self._check_seq(Assertion, name, p, s, xs, clk, rst, rsync, rneg, msg)


class Logic(Leaf, _ProcIf, _TraceIf):
    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Leaf.__init__(self, name, parent)
        _ProcIf.__init__(self)
        self._dtype = dtype

    @property
    def dtype(self) -> type[Bits]:
        return self._dtype


class Packed(Logic, Singular, ExprVar):
    """Leaf-level bitvector design component."""

    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Logic.__init__(self, name, parent, dtype)
        Singular.__init__(self, dtype.xes())
        ExprVar.__init__(self, name)
        self._waves_change = None
        self._vcd_change = None

        self._vps_e = {self: self.is_edge}
        self._vps_pe = {self: self.is_posedge}
        self._vps_ne = {self: self.is_negedge}

    # Singular => Variable
    def set_next(self, value):
        if isinstance(value, str):
            value = lit2bv(value)
        elif isinstance(value, int):
            if value < 0:
                value = i2bv(value, size=self._dtype.size)
            else:
                value = u2bv(value, size=self._dtype.size)
        value = self._dtype.cast(value)
        super().set_next(value)

    next = property(fset=set_next)

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
        await any_var(self._vps_pe)

    async def negedge(self):
        """Suspend; resume execution at signal negedge."""
        await any_var(self._vps_ne)

    async def edge(self):
        """Suspend; resume execution at signal edge."""
        await any_var(self._vps_e)


class Unpacked(Logic, Aggregate):
    """Leaf-level array of vec/enum/struct/union design components."""

    def __init__(self, name: str, parent: Module, dtype: type[Bits]):
        Logic.__init__(self, name, parent, dtype)
        Aggregate.__init__(self, dtype.xes())


class Checker(Leaf, _ProcIf, _TraceIf):
    def __init__(self, name: str, parent: Module):
        Leaf.__init__(self, name, parent)
        _ProcIf.__init__(self)


class Assumption(Checker):
    pass


class Assertion(Checker):
    pass


class CheckerError(Exception):
    pass


class AssumeError(CheckerError):
    pass


class AssertError(CheckerError):
    pass


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
