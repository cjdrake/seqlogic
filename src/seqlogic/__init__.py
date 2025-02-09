"""Sequential Logic."""

from bvwx import (
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
    div,
    encode_onehot,
    encode_priority,
    eq,
    ge,
    gt,
    i2bv,
    impl,
    ite,
    land,
    le,
    lor,
    lrot,
    lsh,
    lt,
    lxor,
    match,
    mod,
    mul,
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
    uxor,
    xnor,
    xor,
    xt,
)
from bvwx._util import clog2

from .design import DesignError, Float, Module, Packed, Unpacked
from .expr import (
    EQ,
    GE,
    GT,
    ITE,
    LE,
    LT,
    NE,
    Adc,
    Add,
    And,
    Cat,
    Expr,
    GetAttr,
    GetItem,
    Impl,
    Lrot,
    Lsh,
    Mul,
    Mux,
    Nand,
    Neg,
    Ngc,
    Nor,
    Not,
    Or,
    Rep,
    Rrot,
    Rsh,
    Sbc,
    Srsh,
    Sub,
    Sxt,
    Uand,
    Uor,
    Uxor,
    Xnor,
    Xor,
    Xt,
)
from .sim import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
    Aggregate,
    AggrValue,
    BoundedSemaphore,
    CancelledError,
    Event,
    EventLoop,
    FinishError,
    InvalidStateError,
    Lock,
    Semaphore,
    Singular,
    Task,
    TaskGroup,
    TaskState,
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
    wait,
)

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
    "impl",
    "ite",
    "mux",
    # bits: logical
    "lor",
    "land",
    "lxor",
    # bits: unary
    "uor",
    "uand",
    "uxor",
    # bits: arithmetic
    "decode",
    "encode_onehot",
    "encode_priority",
    "add",
    "adc",
    "sub",
    "sbc",
    "neg",
    "ngc",
    "mul",
    "div",
    "mod",
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
    # bits: predicate
    "match",
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
    "Float",
    # expr
    "Expr",
    "Not",
    "Nor",
    "Or",
    "Nand",
    "And",
    "Xnor",
    "Xor",
    "Impl",
    "ITE",
    "Mux",
    "Uor",
    "Uand",
    "Uxor",
    "Add",
    "Adc",
    "Sub",
    "Sbc",
    "Neg",
    "Ngc",
    "Mul",
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
    "CancelledError",
    "FinishError",
    "InvalidStateError",
    "Singular",
    "Aggregate",
    "AggrValue",
    "TaskState",
    "Task",
    "TaskGroup",
    "Event",
    "Semaphore",
    "BoundedSemaphore",
    "Lock",
    "EventLoop",
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
    "FIRST_COMPLETED",
    "FIRST_EXCEPTION",
    "ALL_COMPLETED",
    "wait",
    "finish",
    # util
    "clog2",
]
