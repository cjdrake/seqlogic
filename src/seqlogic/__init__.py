"""Sequential Logic."""

from .bits import (
    AddResult,
    Array,
    Bits,
    Empty,
    Enum,
    Scalar,
    Struct,
    Union,
    Vector,
    add,
    and_,
    bits,
    cat,
    int2vec,
    nand,
    nor,
    or_,
    rep,
    stack,
    sub,
    uint2vec,
    vec,
    xnor,
    xor,
)
from .design import Module, Packed, Unpacked, simify
from .sim import active, changed, get_loop, reactive, resume, sleep
from .util import clog2

__all__ = [
    # bits
    "AddResult",
    "Array",
    "Bits",
    "Empty",
    "Enum",
    "Scalar",
    "Struct",
    "Union",
    "Vector",
    "add",
    "and_",
    "bits",
    "cat",
    "int2vec",
    "nand",
    "nor",
    "or_",
    "rep",
    "stack",
    "sub",
    "uint2vec",
    "vec",
    "xnor",
    "xor",
    # design
    "Module",
    "Packed",
    "Unpacked",
    "simify",
    # sim
    "active",
    "changed",
    "get_loop",
    "reactive",
    "resume",
    "sleep",
    # util
    "clog2",
]
