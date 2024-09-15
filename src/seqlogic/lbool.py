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


def _lmux(s: tuple[int, int], x0: tuple[int, int], x1: tuple[int, int]) -> tuple[int, int]:
    """Lifted 2:1 Mux."""
    x0_01 = x0[0] | x0[1]
    x1_01 = x1[0] | x1[1]
    return (
        s[0] & x0[0] & x1_01 | s[1] & x1[0] & x0_01,
        s[0] & x0[1] & x1_01 | s[1] & x1[1] & x0_01,
    )


def lmux(
    s: tuple[tuple[int, int], ...],
    xs: dict[int, tuple[int, int]],
    default: tuple[int, int],
) -> tuple[int, int]:
    """Lifted N:1 Mux."""
    n = 1 << len(s)

    if n == 1:
        x0 = default
        for i, x in xs.items():
            assert i < n
            x0 = x
        return x0

    if n == 2:
        x0, x1 = default, default
        for i, x in xs.items():
            assert i < n
            if i == 0:
                x0 = x
            else:
                x1 = x
        return _lmux(s[0], x0, x1)

    h = n >> 1
    mask = h - 1
    xs_0, xs_1 = {}, {}
    for i, x in xs.items():
        assert i < n
        if i < h:
            xs_0[i & mask] = x
        else:
            xs_1[i & mask] = x
    x0 = lmux(s[:-1], xs_0, default) if xs_0 else default
    x1 = lmux(s[:-1], xs_1, default) if xs_1 else default
    return _lmux(s[-1], x0, x1)
