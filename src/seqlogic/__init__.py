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
from .design import Module, Packed, Unpacked
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
    Lsh,
    Mux,
    Nand,
    Neg,
    Nor,
    Not,
    Or,
    Rsh,
    Srsh,
    Sub,
    Xnor,
    Xor,
)
from .sim import Region, Sim, changed, finish, get_loop, resume, sleep
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
    "Add",
    "Sub",
    "Neg",
    "Lsh",
    "Rsh",
    "Srsh",
    "Cat",
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
    "Sim",
    "get_loop",
    "sleep",
    "changed",
    "resume",
    "finish",
    # util
    "clog2",
]
