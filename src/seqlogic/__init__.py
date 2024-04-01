"""Sequential Logic."""

from .design import Array, Bit, Bits, Enum, Module, simify
from .sim import changed, get_loop, notify, resume, sleep
from .util import clog2

__all__ = [
    # design
    "Array",
    "Bit",
    "Bits",
    "Enum",
    "Module",
    "simify",
    # sim
    "changed",
    "get_loop",
    "notify",
    "resume",
    "sleep",
    # util
    "clog2",
]
