"""Logic design components.

Combine hierarchy, bit vectors, and simulation semantics into a
straightforward API for creating a digital design.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any, override

if sys.version_info >= (3, 14):
    from annotationlib import Format, get_annotate_from_class_namespace

from bvwx import Array, Scalar, i2bv, lit2bv, stack, u2bv
from deltacycle import (
    Aggregate,
    AnyOf,
    Kernel,
    Singular,
    TaskCoro,
    TaskGroup,
    get_running_kernel,
    now,
)
from deltacycle import Value as SimVal
from vcd.writer import VCDWriter as VcdWriter

from ._expr import Expr
from ._expr import Variable as ExprVar
from ._hier import Branch, Leaf
from ._proc_if import ProcIf
from ._trace_if import TraceIf


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


def _mod_factory_new(pntvs: list[tuple[str, type, Any]]):
    # Create source for _new function
    lines: list[str] = []
    s = ", ".join(f"{pn}=None" for pn, _, _ in pntvs)
    lines.append(f"def _new(cls, {s}):\n")
    s = ", ".join(pn for pn, _, _ in pntvs)
    lines.append(f"    return _new_body(cls, {s})\n")
    source = "".join(lines)

    def _new_body(cls, *args: Any):
        # For all parameters, use either default or override value
        params: dict[str, Any] = {}
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


type Key = tuple[tuple[str, Any], ...]
# Parameterized Module [params ...] => Module
_Modules: defaultdict[type[Module], dict[Key, type[Module]]] = defaultdict(dict)


def _mod_name(cls, key: Key) -> str:
    parts: list[str] = []
    for name, value in key:
        if isinstance(value, type):
            parts.append(f"{name}={value.__name__}")
        else:
            parts.append(f"{name}={value}")
    return f"{cls.__name__}[{','.join(parts)}]"


def _mod_new(cls, name: str, parent: Module | None = None):
    return object.__new__(cls)


def _get_mod(cls: type[Module], params: dict[str, Any]) -> type[Module]:
    key: Key = tuple(sorted(params.items()))
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

    @classmethod
    def _get_annotations(mcs, attrs: dict[str, Any]) -> dict[str, type]:
        if sys.version_info >= (3, 14):
            f = get_annotate_from_class_namespace(attrs)
            if f is not None:
                return f(Format.VALUE)
            else:
                return {}
        return attrs.get("__annotations__", {})

    def __new__(mcs, name: str, bases: tuple[type], attrs: dict[str, Any]):
        # Base case for API
        if name == "Module":
            return super().__new__(mcs, name, bases, attrs)

        # Do not support multiple inheritance
        assert len(bases) == 1

        # Get parameter names, types, default values
        pntvs: list[tuple[str, type, Any]] = []
        params = mcs._get_annotations(attrs)
        for pn, pt in params.items():
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
        mod = super().__new__(mcs, name, bases, attrs)
        mod.__new__ = _mod_new
        return mod


class Module(Branch, ProcIf, TraceIf, metaclass=_ModuleMeta):
    """Hierarchical, branch-level design component.

    A module contains:
    * Submodules
    * Ports
    * Local variables
    * Local processes
    """

    def __init__(self, name: str, parent: Module | None = None):
        Branch.__init__(self, name, parent)
        ProcIf.__init__(self)

        # Ports: name => connected
        self._inputs: dict[str, bool] = {}
        self._outputs: dict[str, bool] = {}

        self.build()

    def build(self) -> None:
        raise NotImplementedError("Module requires a build method")

    def main(self) -> TaskCoro:
        """Add design processes to the simulator."""

        async def cf():
            async with TaskGroup() as tg:
                for node in self.iter_bfs():
                    assert isinstance(node, ProcIf)
                    for cf, args, kwargs in node.reactive:
                        coro = cf(*args, **kwargs)
                        tg.create_task(coro, priority=Region.REACTIVE)
                    for cf, args, kwargs in node.active:
                        coro = cf(*args, **kwargs)
                        tg.create_task(coro, priority=Region.ACTIVE)
                    for cf, args, kwargs in node.inactive:
                        coro = cf(*args, **kwargs)
                        tg.create_task(coro, priority=Region.INACTIVE)

        return cf()

    def _check_unique(self, name: str, description: str):
        if hasattr(self, name):
            raise DesignError(f"Duplicate {description}: {name}")

    @property
    def scope(self) -> str:
        """Return the branch's full name using dot separator syntax."""
        if self._parent is None:
            return self.name
        assert isinstance(self._parent, Module)
        return f"{self._parent.scope}.{self.name}"

    def dump_waves(self, waves: defaultdict[int, dict], pattern: str):
        for child in self._children:
            assert isinstance(child, TraceIf)
            child.dump_waves(waves, pattern)

    def dump_vcd(self, vcdw: VcdWriter, pattern: str):
        for child in self._children:
            assert isinstance(child, TraceIf)
            child.dump_vcd(vcdw, pattern)

    def input[T: Array](self, name: str, dtype: T) -> Packed[T]:
        # Require valid and unique name
        self._check_unique(name, "input port")

        # Create port
        node = Packed(name, parent=self, dtype=dtype)

        # Mark port unconnected
        self._inputs[name] = False

        # Save port in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def output[T: Array](self, name: str, dtype: T) -> Packed[T]:
        # Require valid and unique name
        self._check_unique(name, "output port")

        # Create port
        node = Packed(name, parent=self, dtype=dtype)

        # Mark port unconnected
        self._outputs[name] = False

        # Save port in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def _connect_input(self, name: str, rhs: Packed | Expr):
        y = getattr(self, name)

        if self._inputs[name]:
            raise DesignError(f"Input port {name} already connected")

        # Implement the connection
        match rhs:
            # y = x
            case Packed() as x:
                self.assign(y, x)
            # y = x0 & x1 | ...
            case Expr() as ex:
                self.expr(y, ex)
            # y = (f, x0, x1, ...)
            # case [Callable() as f, *xs]:
            #    self._combi(y, f, *xs)
            case _:
                raise DesignError(f"Input port {name} invalid connection")

        # Mark port connected
        self._inputs[name] = True

    def _connect_output(self, name: str, rhs: Packed):
        x = getattr(self, name)

        if self._outputs[name]:
            raise DesignError(f"Output port {name} already connected")

        # Implement the connection
        match rhs:
            # x = y
            case Packed() as y:
                self.assign(y, x)
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

    def logic[T: Array](
        self, name: str, dtype: T, shape: tuple[int, ...] | None = None
    ) -> Packed[T] | Unpacked[T]:
        # Require valid and unique name
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
        # Require valid and unique name
        self._check_unique(name, "float")

        # Create float
        node = Float(name, parent=self)

        # Save in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def submod[T: Module](self, name: str, mod: T) -> T:
        # Require valid and unique name
        self._check_unique(name, "submodule")

        # Create submodule
        node = mod(name, parent=self)

        # Save in module namespace
        setattr(self, name, node)

        # Return a reference for local use
        return node

    def drv(self, cf: Callable[..., TaskCoro], *args: Any, **kwargs: Any):
        self._active.append((cf, args, kwargs))

    def mon(self, cf: Callable[..., TaskCoro], *args: Any, **kwargs: Any):
        self._inactive.append((cf, args, kwargs))

    def _combi(
        self,
        y: SimVal,
        f: Callable[..., Array | str],
        *xs: Packed | Unpacked,
    ):
        async def cf():
            while True:
                await AnyOf(*xs)
                y.next = f(*[x.value for x in xs])

        self._reactive.append((cf, (), {}))

    def _combis(
        self,
        ys: Sequence[SimVal],
        f: Callable[..., tuple[Array | str, ...]],
        *xs: Packed | Unpacked,
    ):
        async def cf():
            while True:
                await AnyOf(*xs)

                # Apply f to inputs
                values = f(*[x.value for x in xs])
                assert len(ys) == len(values)

                for y, value in zip(ys, values):
                    y.next = value

        self._reactive.append((cf, (), {}))

    def combi(
        self,
        ys: SimVal | Sequence[SimVal],
        f: Callable[..., Array | str | tuple[Array | str, ...]],
        *xs: Packed | Unpacked,
    ):
        """Combinational logic."""

        # Pack outputs
        if isinstance(ys, SimVal):
            self._combi(ys, f, *xs)
        elif isinstance(ys, Sequence) and all(isinstance(y, SimVal) for y in ys):
            self._combis(ys, f, *xs)
        else:
            raise TypeError("Expected ys to be Simval or [SimVal]")

    def expr(self, y: SimVal, ex: Expr):
        """Expression logic."""
        f, xs = ex.to_func()
        self._combi(y, f, *xs)

    def assign(self, y: SimVal, x: Packed | str):
        """Assign input to output."""
        # fmt: off
        if isinstance(x, str):
            async def cf_str():
                y.next = x
            self._active.append((cf_str, (), {}))
        elif isinstance(x, Packed):
            async def cf_packed():
                while True:
                    await x
                    y.next = x.value
            self._reactive.append((cf_packed, (), {}))
        else:
            raise TypeError("Expected x to be Packed or str")
        # fmt: on

    def dff[T: Array](
        self,
        q: Packed[T],
        d: Packed[T],
        clk: Packed[Scalar],
        en: Packed[Scalar] | None = None,
        rst: Packed[Scalar] | None = None,
        rval: T | str | None = None,
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
            def clk_en() -> bool:
                return clk.is_posedge() and en.prev == "1b1"

        # No Reset
        if rst is None:
            async def cf():
                while True:
                    v = await clk.pred(clk_en)
                    assert v is clk
                    q.next = d.prev

        # Reset
        else:
            if rval is None:
                rval = q.dtype.zeros()

            # Synchronous Reset
            if rsync:
                if rneg:
                    async def cf():
                        while True:
                            v = await clk.pred(clk_en)
                            assert v is clk
                            q.next = rval if not rst.prev else d.prev
                else:
                    async def cf():
                        while True:
                            v = await clk.pred(clk_en)
                            assert v is clk
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
                    while True:
                        v = await AnyOf(rst.pred(rst_pred), clk.pred(clk_pred))
                        if v is rst:
                            q.next = rval
                        elif v is clk:
                            q.next = d.prev
                        else:
                            assert False  # pragma: no cover
        # fmt: on

        self._active.append((cf, (), {}))

    def mem_wr[T: Array](
        self,
        mem: Unpacked[T],
        addr: Packed,
        data: Packed[T],
        clk: Packed[Scalar],
        en: Packed[Scalar],
        be: Packed | None = None,
    ):
        """Memory with write enable."""

        def clk_pred() -> bool:
            return clk.is_posedge() and en.prev == "1b1"

        # fmt: off
        if be is None:
            async def cf():
                while True:
                    v = await clk.pred(clk_pred)
                    assert v is clk
                    assert not addr.prev.has_unknown()
                    mem[addr.prev].next = data.prev
        else:
            # Require mem/data to be Array[N,8]
            assert len(mem.dtype.shape) == 2 and mem.dtype.shape[1] == 8
            assert len(data.dtype.shape) == 2 and data.dtype.shape[1] == 8

            async def cf():
                while True:
                    v = await clk.pred(clk_pred)
                    assert v is clk
                    assert not addr.prev.has_unknown()
                    assert not be.prev.has_unknown()
                    xs: list[T] = []
                    for i, data_en in enumerate(be.prev):
                        if data_en:
                            xs.append(data.prev[i])
                        else:
                            xs.append(mem[addr.prev].prev[i])
                    mem[addr.prev].next = stack(*xs)
        # fmt: on

        self._active.append((cf, (), {}))

    def _check_func[T: Checker](
        self,
        t: type[T],
        name: str,
        p: Expr,
        f,
        xs,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
        rsync: bool,
        rneg: bool,
        msg: str | None,
    ) -> T:
        E = {Assumption: AssumeError, Assertion: AssertError}[t]

        # Require valid and unique name
        self._check_unique(name, "checker")

        node = t(name, parent=self)

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
                while True:
                    v = await AnyOf(rst.pred(rst_pred), clk.pred(clk_pred))
                    if v is rst:
                        on = True
                    elif v is clk:
                        if on:
                            en = p_f(*[x.value for x in p_xs])
                            if en:
                                _check()
                    else:
                        assert False  # pragma: no cover
        # fmt: on

        node._inactive.append((cf, (), {}))

        # Save in module namespace
        setattr(self, name, node)
        return node

    def assume_expr(
        self,
        name: str,
        p: Expr,
        q: Expr,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assumption:
        f, xs = q.to_func()
        return self._check_func(Assumption, name, p, f, xs, clk, rst, rsync, rneg, msg)

    def assert_expr(
        self,
        name: str,
        p: Expr,
        q: Expr,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assertion:
        f, xs = q.to_func()
        return self._check_func(Assertion, name, p, f, xs, clk, rst, rsync, rneg, msg)

    def assume_func(
        self,
        name: str,
        p: Expr,
        f,
        xs,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
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
        xs,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assertion:
        return self._check_func(Assertion, name, p, f, xs, clk, rst, rsync, rneg, msg)

    def _check_seq[T: Checker](
        self,
        t: type[T],
        name: str,
        p: Expr,
        s,
        xs,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
        rsync: bool,
        rneg: bool,
        msg: str | None,
    ) -> T:
        E = {Assumption: AssumeError, Assertion: AssertError}[t]

        # Require valid and unique name
        self._check_unique(name, "checker")

        node = t(name, parent=self)

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
                kernel = get_running_kernel()
                task = kernel.task()

                on = False
                while True:
                    v = await AnyOf(rst.pred(rst_pred), clk.pred(clk_pred))
                    if v is rst:
                        on = True
                    elif v is clk:
                        if on:
                            en = p_f(*[x.value for x in p_xs])
                            if en:
                                assert task.group is not None
                                task.group.create_task(_check(), priority=Region.INACTIVE)
                    else:
                        assert False  # pragma: no cover
        # fmt: on

        node._inactive.append((cf, (), {}))

        # Save in module namespace
        setattr(self, name, node)
        return node

    def assume_seq(
        self,
        name: str,
        p: Expr,
        s,
        xs,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
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
        xs,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assertion:
        return self._check_seq(Assertion, name, p, s, xs, clk, rst, rsync, rneg, msg)

    def assume_next(
        self,
        name: str,
        p: Expr,
        q: Expr,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assumption:
        f, xs = q.to_func()

        async def s(*xs: Packed | Unpacked):
            await clk.posedge()
            return f(*[x.value for x in xs])

        return self._check_seq(Assumption, name, p, s, xs, clk, rst, rsync, rneg, msg)

    def assert_next(
        self,
        name: str,
        p: Expr,
        q: Expr,
        clk: Packed[Scalar],
        rst: Packed[Scalar],
        rsync: bool = False,
        rneg: bool = False,
        msg: str | None = None,
    ) -> Assertion:
        f, xs = q.to_func()

        async def s(*xs: Packed | Unpacked):
            await clk.posedge()
            return f(*[x.value for x in xs])

        return self._check_seq(Assertion, name, p, s, xs, clk, rst, rsync, rneg, msg)


class Logic[T: Array](Leaf, ProcIf, TraceIf):
    def __init__(self, name: str, parent: Module, dtype: T):
        Leaf.__init__(self, name, parent)
        ProcIf.__init__(self)
        self._dtype = dtype

    @property
    def dtype(self) -> T:
        return self._dtype


class Packed[T: Array](Logic[T], Singular[T], ExprVar):
    """Leaf-level bitvector design component."""

    def __init__(self, name: str, parent: Module, dtype: T):
        Logic.__init__(self, name, parent, dtype)
        Singular.__init__(self, dtype.xs())
        ExprVar.__init__(self, name)

        self._waves_change: Callable[[], None] | None = None
        self._vcd_change: Callable[[], None] | None = None

    # Singular => Variable
    @override
    def set_next(self, value: T | str | int):
        if isinstance(value, int):
            if value < 0:
                _value = i2bv(value, size=self._dtype.size)
            else:
                _value = u2bv(value, size=self._dtype.size)
        elif isinstance(value, str):
            _value = lit2bv(value)
        elif isinstance(value, Array):
            _value = value
        else:
            raise TypeError("Expected value to be Array, str literal, or int")
        _value = self._dtype.cast(_value)
        super().set_next(_value)

    next = property(fset=set_next)

    @override
    def update(self):
        if self._waves_change and (self._next != self._prev):
            self._waves_change()
        if self._vcd_change and (self._next != self._prev):
            self._vcd_change()
        super().update()

    # TraceIf
    def dump_waves(self, waves: defaultdict[int, dict], pattern: str):
        if re.fullmatch(pattern, self.qualname):
            # Initial time
            waves[Kernel.init_time][self] = self._prev

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
            return bool(not self._prev and not self._next)
        except ValueError:
            return False

    def is_posedge(self) -> bool:
        """Return True when bit transitions 0 => 1."""
        try:
            return bool(not self._prev and self._next)
        except ValueError:
            return False

    def is_negedge(self) -> bool:
        """Return True when bit transitions 1 => 0."""
        try:
            return bool(self._prev and not self._next)
        except ValueError:
            return False

    def is_pos(self) -> bool:
        """Return True when bit is stable 1 => 1."""
        try:
            return bool(self._prev and self._next)
        except ValueError:
            return False

    def is_edge(self) -> bool:
        """Return True when bit transitions 0 => 1 or 1 => 0."""
        try:
            pe = bool(not self._prev and self._next)
            ne = bool(self._prev and not self._next)
            return pe or ne
        except ValueError:
            return False

    async def posedge(self):
        """Suspend; resume execution at signal posedge."""
        await self.pred(self.is_posedge)

    async def negedge(self):
        """Suspend; resume execution at signal negedge."""
        await self.pred(self.is_negedge)

    async def edge(self):
        """Suspend; resume execution at signal edge."""
        await self.pred(self.is_edge)


class Unpacked[T: Array](Logic[T], Aggregate[T]):
    """Leaf-level array of vec/enum/struct/union design components."""

    def __init__(self, name: str, parent: Module, dtype: T):
        Logic.__init__(self, name, parent, dtype)
        Aggregate.__init__(self, dtype.xs())

    # NOTE: TraceIf not implemented


class Checker(Leaf, ProcIf, TraceIf):
    def __init__(self, name: str, parent: Module):
        Leaf.__init__(self, name, parent)
        ProcIf.__init__(self)

    # NOTE: TraceIf not implemented


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


class Float(Leaf, ProcIf, TraceIf, Singular[float]):
    """Leaf-level float design component."""

    def __init__(self, name: str, parent: Module):
        Leaf.__init__(self, name, parent)
        ProcIf.__init__(self)
        Singular.__init__(self, float())

        self._waves_change: Callable[[], None] | None = None
        self._vcd_change: Callable[[], None] | None = None

    @override
    def update(self):
        if self._waves_change and (self._next != self._prev):
            self._waves_change()
        if self._vcd_change and (self._next != self._prev):
            self._vcd_change()
        super().update()

    # TraceIf
    def dump_waves(self, waves: defaultdict[int, dict], pattern: str):
        if re.fullmatch(pattern, self.qualname):
            # Initial time
            waves[Kernel.init_time][self] = self._prev

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
