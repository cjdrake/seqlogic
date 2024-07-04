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
