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

from_hexchar: dict[int, dict[str, tuple[int, int]]] = {
    1: {
        "0": _0,
        "1": _1,
    },
    2: {
        "0": (0b11, 0b00),
        "1": (0b10, 0b01),
        "2": (0b01, 0b10),
        "3": (0b00, 0b11),
    },
    3: {
        "0": (0b111, 0b000),
        "1": (0b110, 0b001),
        "2": (0b101, 0b010),
        "3": (0b100, 0b011),
        "4": (0b011, 0b100),
        "5": (0b010, 0b101),
        "6": (0b001, 0b110),
        "7": (0b000, 0b111),
    },
    4: {
        "0": (0b1111, 0b0000),
        "1": (0b1110, 0b0001),
        "2": (0b1101, 0b0010),
        "3": (0b1100, 0b0011),
        "4": (0b1011, 0b0100),
        "5": (0b1010, 0b0101),
        "6": (0b1001, 0b0110),
        "7": (0b1000, 0b0111),
        "8": (0b0111, 0b1000),
        "9": (0b0110, 0b1001),
        "a": (0b0101, 0b1010),
        "A": (0b0101, 0b1010),
        "b": (0b0100, 0b1011),
        "B": (0b0100, 0b1011),
        "c": (0b0011, 0b1100),
        "C": (0b0011, 0b1100),
        "d": (0b0010, 0b1101),
        "D": (0b0010, 0b1101),
        "e": (0b0001, 0b1110),
        "E": (0b0001, 0b1110),
        "f": (0b0000, 0b1111),
        "F": (0b0000, 0b1111),
    },
}
