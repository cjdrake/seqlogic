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
    xnor,
    xor,
)
from .design import Module, Packed, Unpacked
from .sim import active, changed, get_loop, reactive, resume, sleep
from .util import clog2

# Alias Vector to Vec for brevity
Vec = Vector

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
    "Vec",
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
    "xnor",
    "xor",
    # design
    "Module",
    "Packed",
    "Unpacked",
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
