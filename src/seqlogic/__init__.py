"""Sequential Logic."""

from .bits import (
    Array,
    Bits,
    Empty,
    Enum,
    Scalar,
    Struct,
    Union,
    Vector,
    adc,
    add,
    and_,
    bits,
    cat,
    clz,
    decode,
    eq,
    ge,
    gt,
    i2bv,
    ite,
    le,
    lrot,
    lsh,
    lt,
    mux,
    nand,
    ne,
    neg,
    ngc,
    nor,
    not_,
    or_,
    pack,
    rep,
    rrot,
    rsh,
    sbc,
    sge,
    sgt,
    sle,
    slt,
    srsh,
    stack,
    sub,
    sxt,
    u2bv,
    uand,
    uor,
    uxnor,
    uxor,
    xnor,
    xor,
    xt,
)
from .design import DesignError, Module, Packed, Unpacked
from .expr import (
    EQ,
    GE,
    GT,
    ITE,
    LE,
    LT,
    NE,
    Add,
    And,
    Cat,
    GetAttr,
    GetItem,
    Lrot,
    Lsh,
    Mux,
    Nand,
    Neg,
    Nor,
    Not,
    Or,
    Rep,
    Rrot,
    Rsh,
    Srsh,
    Sub,
    Sxt,
    Uand,
    Uor,
    Uxnor,
    Uxor,
    Xnor,
    Xor,
    Xt,
)
from .sim import (
    EventLoop,
    Region,
    Task,
    changed,
    create_task,
    del_event_loop,
    finish,
    get_event_loop,
    get_running_loop,
    irun,
    new_event_loop,
    now,
    resume,
    run,
    set_event_loop,
    sleep,
)
from .util import clog2

# Alias Vector to Vec for brevity
Vec = Vector

__all__ = [
    # bits
    "Bits",
    "Empty",
    "Scalar",
    "Vector",
    "Vec",
    "Array",
    "Enum",
    "Struct",
    "Union",
    # bits: bitwise
    "not_",
    "nor",
    "or_",
    "nand",
    "and_",
    "xnor",
    "xor",
    "ite",
    "mux",
    # bits: unary
    "uor",
    "uand",
    "uxnor",
    "uxor",
    # bits: arithmetic
    "decode",
    "add",
    "adc",
    "sub",
    "sbc",
    "neg",
    "ngc",
    "lsh",
    "rsh",
    "srsh",
    # bits: word
    "xt",
    "sxt",
    "lrot",
    "rrot",
    "cat",
    "rep",
    "pack",
    "clz",
    # bits: predicate
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "slt",
    "sle",
    "sgt",
    "sge",
    # bits: factory
    "bits",
    "stack",
    "u2bv",
    "i2bv",
    # design
    "DesignError",
    "Module",
    "Packed",
    "Unpacked",
    # expr
    "Not",
    "Nor",
    "Or",
    "Nand",
    "And",
    "Xnor",
    "Xor",
    "ITE",
    "Mux",
    "Uor",
    "Uand",
    "Uxnor",
    "Uxor",
    "Add",
    "Sub",
    "Neg",
    "Lsh",
    "Rsh",
    "Srsh",
    "Xt",
    "Sxt",
    "Lrot",
    "Rrot",
    "Cat",
    "Rep",
    "LT",
    "LE",
    "EQ",
    "NE",
    "GT",
    "GE",
    "GetItem",
    "GetAttr",
    # sim
    "Region",
    "EventLoop",
    "Task",
    "create_task",
    "get_event_loop",
    "get_running_loop",
    "set_event_loop",
    "new_event_loop",
    "del_event_loop",
    "run",
    "irun",
    "now",
    "sleep",
    "changed",
    "resume",
    "finish",
    # util
    "clog2",
]
