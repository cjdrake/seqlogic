"""Lifted Boolean scalar and vector data types.

The conventional Boolean type consists of {False, True}, or {0, 1}.
You can pack Booleans using an int, e.g. hex(0xaa | 0x55) == '0xff'.

The "lifted" Boolean data type expands {0, 1} to {0, 1, Neither, DontCare}.

Zero and One retain their original meanings.

"Neither" means neither zero nor one.
This is an illogical or metastable value that always dominates other values.
We will use a shorthand character 'X'.

"DontCare" means either zero or one.
This is an unknown or uninitialized value that may either dominate or be
dominated by other values, depending on the operation.

The 'vec' class packs multiple lifted Booleans into a vector,
which enables efficient, bit-wise operations and arithmetic.
"""

# Scalars
_X = 0b00
_0 = 0b01
_1 = 0b10
_W = 0b11


_LNOT = (_X, _1, _0, _W)


def not_(x: int) -> int:
    """Lifted NOT function.

    f(x) -> y:
        X => X | 00 => 00
        0 => 1 | 01 => 10
        1 => 0 | 10 => 01
        - => - | 11 => 11
    """
    return _LNOT[x]


_LNOR = (
    (_X, _X, _X, _X),
    (_X, _1, _0, _W),
    (_X, _0, _0, _0),
    (_X, _W, _0, _W),
)


def nor(x0: int, x1: int) -> int:
    """Lifted NOR function.

    f(x0, x1) -> y:
        0 0 => 1
        1 - => 0
        X - => X
        - 0 => -

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
          +--+--+--+--+
       01 |00|10|11|01|
          +--+--+--+--+
       11 |00|11|11|01|  y0 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       10 |00|01|01|01|     | x0[1] & x1[1]
          +--+--+--+--+
    """
    return _LNOR[x0][x1]


_LOR = (
    (_X, _X, _X, _X),
    (_X, _0, _1, _W),
    (_X, _1, _1, _1),
    (_X, _W, _1, _W),
)


def or_(x0: int, x1: int) -> int:
    """Lifted OR function.

    f(x0, x1) -> y:
        0 0 => 0
        1 - => 1
        X - => X
        - 0 => -

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       01 |00|01|11|10|     | x0[1] & x1[1]
          +--+--+--+--+
       11 |00|11|11|10|  y0 = x0[0] & x1[0]
          +--+--+--+--+
       10 |00|10|10|10|
          +--+--+--+--+
    """
    return _LOR[x0][x1]


_LNAND = (
    (_X, _X, _X, _X),
    (_X, _1, _1, _1),
    (_X, _1, _0, _W),
    (_X, _1, _W, _W),
)


def nand(x0: int, x1: int) -> int:
    """Lifted NAND function.

    f(x0, x1) -> y:
        1 1 => 0
        0 - => 1
        X - => X
        - 1 => -

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
          +--+--+--+--+     | x0[0] & x1[1]
       01 |00|10|10|10|     | x0[1] & x1[0]
          +--+--+--+--+
       11 |00|10|11|11|  y0 = x0[1] & x1[1]
          +--+--+--+--+
       10 |00|10|11|01|
          +--+--+--+--+
    """
    return _LNAND[x0][x1]


_LAND = (
    (_X, _X, _X, _X),
    (_X, _0, _0, _0),
    (_X, _0, _1, _W),
    (_X, _0, _W, _W),
)


def and_(x0: int, x1: int) -> int:
    """Lifted AND function.

    f(x0, x1) -> y:
        1 1 => 1
        0 - => 0
        X - => X
        - 1 => -

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[1] & x1[1]
          +--+--+--+--+
       01 |00|01|01|01|
          +--+--+--+--+
       11 |00|01|11|11|  y0 = x0[0] & x1[0]
          +--+--+--+--+     | x0[0] & x1[1]
       10 |00|01|11|10|     | x0[1] & x1[0]
          +--+--+--+--+
    """
    return _LAND[x0][x1]


_LXNOR = (
    (_X, _X, _X, _X),
    (_X, _1, _0, _W),
    (_X, _0, _1, _W),
    (_X, _W, _W, _W),
)


def xnor(x0: int, x1: int) -> int:
    """Lifted XNOR function.

    f(x0, x1) -> y:
        0 0 => 1
        0 1 => 0
        1 0 => 0
        1 1 => 1
        X - => X
        - 0 => -
        - 1 => -

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
          +--+--+--+--+     | x0[1] & x1[1]
       01 |00|10|11|01|
          +--+--+--+--+
       11 |00|11|11|11|  y0 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       10 |00|01|11|10|
          +--+--+--+--+
    """
    return _LXNOR[x0][x1]


_LXOR = (
    (_X, _X, _X, _X),
    (_X, _0, _1, _W),
    (_X, _1, _0, _W),
    (_X, _W, _W, _W),
)


def xor(x0: int, x1: int) -> int:
    """Lifted XOR function.

    f(x0, x1) -> y:
        0 0 => 0
        0 1 => 1
        1 0 => 1
        1 1 => 0
        X - => X
        - 0 => -
        - 1 => -

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       01 |00|01|11|10|
          +--+--+--+--+
       11 |00|11|11|11|  y0 = x0[0] & x1[0]
          +--+--+--+--+     | x0[1] & x1[1]
       10 |00|10|11|01|
          +--+--+--+--+
    """
    return _LXOR[x0][x1]


_LIMPLIES = (
    (_X, _X, _X, _X),
    (_X, _1, _1, _1),
    (_X, _0, _1, _W),
    (_X, _W, _1, _W),
)


def implies(p: int, q: int) -> int:
    """Lifted IMPLIES function.

    f(p, q) -> y:
        0 0 => 1
        0 1 => 1
        1 0 => 0
        1 1 => 1
        N - => N
        0 - => 1
        1 - => -
        - 0 => -
        - 1 => 1
        - - => -

           q
           00 01 11 10
          +--+--+--+--+
     p 00 |00|00|00|00|  y1 = p[0] & q[0]
          +--+--+--+--+     | p[0] & q[1]
       01 |00|10|10|10|     | p[1] & q[1]
          +--+--+--+--+
       11 |00|11|11|10|  y0 = p[1] & q[0]
          +--+--+--+--+
       10 |00|01|11|10|
          +--+--+--+--+
    """
    return _LIMPLIES[p][q]
