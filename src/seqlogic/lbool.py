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
    r"""Lifted If-Then-Else.

    s=00  b                             s=01  b
        \ 00 01 11 10                       \ 00 01 11 10
         +--+--+--+--+                       +--+--+--+--+
    a 00 |00|00|00|00|                  a 00 |00|00|00|00|  s0 b0 a0
         +--+--+--+--+                       +--+--+--+--+  s0 b0 a1
      01 |00|00|00|00|                    01 |00|01|11|10|
         +--+--+--+--+                       +--+--+--+--+  s0 b1 a0
      11 |00|00|00|00|                    11 |00|01|11|10|  s0 b1 a1
         +--+--+--+--+                       +--+--+--+--+
      10 |00|00|00|00|                    10 |00|01|11|10|
         +--+--+--+--+                       +--+--+--+--+

    s=10  b                             s=11  b
        \ 00 01 11 10                       \ 00 01 11 10
         +--+--+--+--+                       +--+--+--+--+
    a 00 |00|00|00|00|  s1 a0 b0        a 00 |00|00|00|00|
         +--+--+--+--+  s1 a0 b1             +--+--+--+--+
      01 |00|01|01|01|                    01 |00|01|11|11|
         +--+--+--+--+  s1 a1 b0             +--+--+--+--+
      11 |00|11|11|11|  s1 a1 b1          11 |00|11|11|11|
         +--+--+--+--+                       +--+--+--+--+
      10 |00|10|10|10|                    10 |00|11|11|10|
         +--+--+--+--+                       +--+--+--+--+
    """
    a01 = a[0] | a[1]
    b01 = b[0] | b[1]
    return (
        s[1] & a[0] & b01 | s[0] & b[0] & a01,
        s[1] & a[1] & b01 | s[0] & b[1] & a01,
    )
