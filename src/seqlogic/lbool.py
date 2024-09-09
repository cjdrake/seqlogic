"""Lifted Boolean constants."""

# Scalars
_X = (0, 0)
_0 = (1, 0)
_1 = (0, 1)
_W = (1, 1)


from_char: dict[str, tuple[int, int]] = {
    "X": _X,
    "0": _0,
    "1": _1,
    "-": _W,
}

to_char: dict[tuple[int, int], str] = {
    _X: "X",
    _0: "0",
    _1: "1",
    _W: "-",
}

to_vcd_char: dict[tuple[int, int], str] = {
    _X: "x",
    _0: "0",
    _1: "1",
    _W: "x",
}


def lnot(d: tuple[int, int]) -> tuple[int, int]:
    """Lifted NOT."""
    return d[1], d[0]


def lor(d0: tuple[int, int], d1: tuple[int, int]) -> tuple[int, int]:
    """Lifted OR.

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
    return (
        d0[0] & d1[0],
        d0[0] & d1[1] | d0[1] & d1[0] | d0[1] & d1[1],
    )


def land(d0: tuple[int, int], d1: tuple[int, int]) -> tuple[int, int]:
    """Lifted AND.

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
    return (
        d0[0] & d1[0] | d0[0] & d1[1] | d0[1] & d1[0],
        d0[1] & d1[1],
    )


def lxnor(d0: tuple[int, int], d1: tuple[int, int]) -> tuple[int, int]:
    """Lifted XNOR.

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
    return (
        d0[0] & d1[1] | d0[1] & d1[0],
        d0[0] & d1[0] | d0[1] & d1[1],
    )


def lxor(d0: tuple[int, int], d1: tuple[int, int]) -> tuple[int, int]:
    """Lifted XOR.

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
    return (
        d0[0] & d1[0] | d0[1] & d1[1],
        d0[0] & d1[1] | d0[1] & d1[0],
    )


def lite(s: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Lifted If-Then-Else.

    .i 6
    .o 2
    .ilb s0 s1 a0 a1 b0 b1
    .ob y0 y1
    .p 8
    1--11- 10 ; s0 & a1 & b0 ;
    1--1-1 01 ;              ; s0 & a1 & b1
    1-1-1- 10 ; s0 & a0 & b0 ;
    1-1--1 01 ;              ; s0 & a0 & b1
    -11--1 10 ; s1 & a0 & b1 ;
    -11-1- 10 ; s1 & a0 & b0 ;
    -1-1-1 01 ;              ; s1 & a1 & b1
    -1-11- 01 ;              ; s1 & a1 & b0
    .e
    """
    return (
        s[0] & a[1] & b[0] | s[0] & a[0] & b[0] | s[1] & a[0] & b[1] | s[1] & a[0] & b[0],
        s[0] & a[1] & b[1] | s[0] & a[0] & b[1] | s[1] & a[1] & b[1] | s[1] & a[1] & b[0],
    )
