"""Test seqlogic.bits module."""

import pytest

from seqlogic import (
    Scalar,
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
    sub,
    uint2vec,
    xnor,
    xor,
)
from seqlogic.lbconst import _W, _X, _0, _1

E = Vector[0](*_X)
X = Vector[1](*_X)
F = Vector[1](*_0)
T = Vector[1](*_1)
W = Vector[1](*_W)


def test_vec_class_getitem():
    # Negative values are illegal
    with pytest.raises(TypeError):
        _ = Vector[-1]

    vec_0 = Vector[0]
    assert vec_0.size == 0

    vec_4 = Vector[4]
    assert vec_4.size == 4

    # Always return the same class instance
    assert Vector[0] is vec_0
    assert Vector[4] is vec_4


def test_vec():
    # None/Empty
    assert bits() == E
    assert bits(None) == E
    assert bits([]) == E

    # Single bool input
    assert bits(False) == F
    assert bits(0) == F
    assert bits(True) == T
    assert bits(1) == T

    # Sequence of bools
    assert bits([False, True, 0, 1]) == Vector[4](0b0101, 0b1010)

    # String
    assert bits("4b-10X") == Vector[4](0b1010, 0b1100)

    # Invalid input type
    with pytest.raises(TypeError):
        bits(1.0e42)
    with pytest.raises(TypeError):
        bits([0, 0, 0, 42])


BIN_LITS = {
    "1b0": (1, 0b0),
    "1b1": (1, 0b1),
    "2b00": (2, 0b00),
    "2b01": (2, 0b01),
    "2b10": (2, 0b10),
    "2b11": (2, 0b11),
    "3b100": (3, 0b100),
    "3b101": (3, 0b101),
    "3b110": (3, 0b110),
    "3b111": (3, 0b111),
    "4b1000": (4, 0b1000),
    "4b1001": (4, 0b1001),
    "4b1010": (4, 0b1010),
    "4b1011": (4, 0b1011),
    "4b1100": (4, 0b1100),
    "4b1101": (4, 0b1101),
    "4b1110": (4, 0b1110),
    "4b1111": (4, 0b1111),
    "5b1_0000": (5, 0b1_0000),
    "5b1_1111": (5, 0b1_1111),
    "6b10_0000": (6, 0b10_0000),
    "6b11_1111": (6, 0b11_1111),
    "7b100_0000": (7, 0b100_0000),
    "7b111_1111": (7, 0b111_1111),
    "8b1000_0000": (8, 0b1000_0000),
    "8b1111_1111": (8, 0b1111_1111),
    "9b1_0000_0000": (9, 0b1_0000_0000),
    "9b1_1111_1111": (9, 0b1_1111_1111),
}


def test_vec_lit_bin():
    # Valid inputs w/o X
    for lit, (n, d1) in BIN_LITS.items():
        v = bits(lit)
        assert len(v) == n and v.data[1] == d1

    # Valid inputs w/ X
    v = bits("4b-1_0X")
    assert len(v) == 4 and v.data == (0b1010, 0b1100)
    v = bits("4bX01-")
    assert len(v) == 4 and v.data == (0b0101, 0b0011)

    # Not a literal
    with pytest.raises(ValueError):
        bits("invalid")

    # Size cannot be zero
    with pytest.raises(ValueError):
        bits("0b0")
    # Contains illegal characters
    with pytest.raises(ValueError):
        bits("4b1XW0")

    # Size is too big
    with pytest.raises(ValueError):
        bits("8b1010")

    # Size is too small
    with pytest.raises(ValueError):
        bits("4b1010_1010")


DEC_LITS = {
    "1d0": (1, 0),
    "1d1": (1, 1),
    "2d0": (2, 0),
    "2d1": (2, 1),
    "2d2": (2, 2),
    "2d3": (2, 3),
    "3d4": (3, 4),
    "3d5": (3, 5),
    "3d6": (3, 6),
    "3d7": (3, 7),
    "4d8": (4, 8),
    "4d9": (4, 9),
    "4d10": (4, 10),
    "4d11": (4, 11),
    "4d12": (4, 12),
    "4d13": (4, 13),
    "4d14": (4, 14),
    "4d15": (4, 15),
    "5d16": (5, 16),
    "5d31": (5, 31),
    "6d32": (6, 32),
    "6d63": (6, 63),
    "7d64": (7, 64),
    "7d127": (7, 127),
    "8d128": (8, 128),
    "8d255": (8, 255),
    "9d256": (9, 256),
    "9d511": (9, 511),
}


def test_lit2vec_dec():
    # Valid inputs
    for lit, (n, d1) in DEC_LITS.items():
        v = bits(lit)
        assert len(v) == n and v.data[1] == d1

    # Not a literal
    with pytest.raises(ValueError):
        bits("invalid")

    # Size cannot be zero
    with pytest.raises(ValueError):
        bits("0d0")
    # Contains illegal characters
    with pytest.raises(ValueError):
        bits("8hd3@d_b33f")

    # Size is too small
    with pytest.raises(ValueError):
        bits("8d256")


HEX_LITS = {
    "1h0": (1, 0x0),
    "1h1": (1, 0x1),
    "2h0": (2, 0x0),
    "2h1": (2, 0x1),
    "2h2": (2, 0x2),
    "2h3": (2, 0x3),
    "3h4": (3, 0x4),
    "3h5": (3, 0x5),
    "3h6": (3, 0x6),
    "3h7": (3, 0x7),
    "4h8": (4, 0x8),
    "4h9": (4, 0x9),
    "4hA": (4, 0xA),
    "4hB": (4, 0xB),
    "4hC": (4, 0xC),
    "4hD": (4, 0xD),
    "4hE": (4, 0xE),
    "4hF": (4, 0xF),
    "5h10": (5, 0x10),
    "5h1F": (5, 0x1F),
    "6h20": (6, 0x20),
    "6h3F": (6, 0x3F),
    "7h40": (7, 0x40),
    "7h7F": (7, 0x7F),
    "8h80": (8, 0x80),
    "8hFF": (8, 0xFF),
    "9h100": (9, 0x100),
    "9h1FF": (9, 0x1FF),
}


def test_lit2vec_hex():
    # Valid inputs
    for lit, (n, d1) in HEX_LITS.items():
        v = bits(lit)
        assert len(v) == n and v.data[1] == d1

    # Not a literal
    with pytest.raises(ValueError):
        bits("invalid")

    # Size cannot be zero
    with pytest.raises(ValueError):
        bits("0h0")
    # Contains illegal characters
    with pytest.raises(ValueError):
        bits("8hd3@d_b33f")

    # Size is too small
    with pytest.raises(ValueError):
        bits("8hdead")

    # Invalid characters
    with pytest.raises(ValueError):
        bits("3h8")  # Only 0..7 is legal
    with pytest.raises(ValueError):
        bits("5h20")  # Only 0..1F is legal


UINT2VEC_VALS = {
    0: "",
    1: "1b1",
    2: "2b10",
    3: "2b11",
    4: "3b100",
    5: "3b101",
    6: "3b110",
    7: "3b111",
    8: "4b1000",
}

UINT2VEC_N_VALS = {
    (0, 0): "",
    (0, 1): "1b0",
    (0, 2): "2b00",
    (1, 1): "1b1",
    (1, 2): "2b01",
    (1, 3): "3b001",
    (1, 4): "4b0001",
    (2, 2): "2b10",
    (2, 3): "3b010",
    (2, 4): "4b0010",
    (3, 2): "2b11",
    (3, 3): "3b011",
    (3, 4): "4b0011",
    (4, 3): "3b100",
    (4, 4): "4b0100",
    (4, 5): "5b0_0100",
}


def test_uint2vec():
    # Negative inputs are invalid
    with pytest.raises(ValueError):
        uint2vec(-1)

    for i, s in UINT2VEC_VALS.items():
        v = uint2vec(i)
        assert str(v) == s
        assert v.to_uint() == i

    for (i, n), s in UINT2VEC_N_VALS.items():
        v = uint2vec(i, n)
        assert str(v) == s
        assert v.to_uint() == i

    # Overflows
    with pytest.raises(ValueError):
        uint2vec(1, 0)
    with pytest.raises(ValueError):
        uint2vec(2, 0)
    with pytest.raises(ValueError):
        uint2vec(2, 1)
    with pytest.raises(ValueError):
        uint2vec(3, 0)
    with pytest.raises(ValueError):
        uint2vec(3, 1)


INT2VEC_VALS = {
    -8: "4b1000",
    -7: "4b1001",
    -6: "4b1010",
    -5: "4b1011",
    -4: "3b100",
    -3: "3b101",
    -2: "2b10",
    -1: "1b1",
    0: "1b0",
    1: "2b01",
    2: "3b010",
    3: "3b011",
    4: "4b0100",
    5: "4b0101",
    6: "4b0110",
    7: "4b0111",
    8: "5b0_1000",
}

INT2VEC_N_VALS = {
    (-5, 4): "4b1011",
    (-5, 5): "5b1_1011",
    (-5, 6): "6b11_1011",
    (-4, 3): "3b100",
    (-4, 4): "4b1100",
    (-4, 5): "5b1_1100",
    (-3, 3): "3b101",
    (-3, 4): "4b1101",
    (-3, 5): "5b1_1101",
    (-2, 2): "2b10",
    (-2, 3): "3b110",
    (-2, 4): "4b1110",
    (-1, 1): "1b1",
    (-1, 2): "2b11",
    (-1, 3): "3b111",
    (0, 1): "1b0",
    (0, 2): "2b00",
    (0, 3): "3b000",
    (1, 2): "2b01",
    (1, 3): "3b001",
    (1, 4): "4b0001",
    (2, 3): "3b010",
    (2, 4): "4b0010",
    (2, 5): "5b0_0010",
    (3, 3): "3b011",
    (3, 4): "4b0011",
    (3, 5): "5b0_0011",
    (4, 4): "4b0100",
    (4, 5): "5b0_0100",
    (4, 6): "6b00_0100",
}


def test_int2vec():
    for i, s in INT2VEC_VALS.items():
        v = int2vec(i)
        assert str(v) == s
        assert v.to_int() == i

    for (i, n), s in INT2VEC_N_VALS.items():
        v = int2vec(i, n)
        assert str(v) == s
        assert v.to_int() == i

    # Overflows
    with pytest.raises(ValueError):
        int2vec(-5, 3)
    with pytest.raises(ValueError):
        int2vec(-4, 2)
    with pytest.raises(ValueError):
        int2vec(-3, 2)
    with pytest.raises(ValueError):
        int2vec(-2, 1)
    with pytest.raises(ValueError):
        int2vec(-1, 0)
    with pytest.raises(ValueError):
        int2vec(0, 0)
    with pytest.raises(ValueError):
        int2vec(1, 1)
    with pytest.raises(ValueError):
        int2vec(2, 2)
    with pytest.raises(ValueError):
        int2vec(3, 2)
    with pytest.raises(ValueError):
        int2vec(4, 3)


def test_cat():
    v = bits("4b-10X")
    assert cat() == bits()
    assert cat(v) == v
    assert cat("2b0X", "2b-1") == bits("4b-10X")
    assert cat(bits("2b0X"), bits("2b-1")) == bits("4b-10X")
    assert cat(0, 1) == bits("2b10")

    with pytest.raises(TypeError):
        _ = cat(v, 42)


def test_rep():
    assert rep(bits(), 4) == bits()
    assert rep(bits("4b-10X"), 2) == bits("8b-10X_-10X")


def test_vec_getitem():
    v = bits("4b-10X")

    assert v[3] == "1b-"
    assert v[2] == "1b1"
    assert v[1] == "1b0"
    assert v[0] == "1bX"

    assert v[-1] == "1b-"
    assert v[-2] == "1b1"
    assert v[-3] == "1b0"
    assert v[-4] == "1bX"

    assert v[0:1] == "1bX"
    assert v[0:2] == "2b0X"
    assert v[0:3] == "3b10X"
    assert v[0:4] == "4b-10X"

    assert v[:-3] == "1bX"
    assert v[:-2] == "2b0X"
    assert v[:-1] == "3b10X"

    assert v[1:2] == "1b0"
    assert v[1:3] == "2b10"
    assert v[1:4] == "3b-10"

    assert v[-3:2] == "1b0"
    assert v[-3:3] == "2b10"
    assert v[-3:4] == "3b-10"

    assert v[2:3] == "1b1"
    assert v[2:4] == "2b-1"

    assert v[3:4] == "1b-"

    # Invalid index
    with pytest.raises(IndexError):
        _ = v[4]
    # Slice step not supported
    with pytest.raises(ValueError):
        _ = v[0:4:1]
    # Invalid index type
    with pytest.raises(TypeError):
        _ = v[1.0e42]


def test_vec_iter():
    v = bits("4b-10X")
    assert list(v) == ["1bX", "1b0", "1b1", "1b-"]


def test_vec_repr():
    assert repr(bits()) == "Empty(0b0, 0b0)"
    assert repr(bits("4b-10X")) == "Vector[4](0b1010, 0b1100)"


def test_vec_bool():
    assert bool(bits()) is False
    assert bool(bits("1b0")) is False
    assert bool(bits("1b1")) is True
    assert bool(bits("4b0000")) is False
    assert bool(bits("4b1010")) is True
    assert bool(bits("4b0101")) is True
    with pytest.raises(ValueError):
        bool(bits("4b110X"))
    with pytest.raises(ValueError):
        bool(bits("4b-100"))


def test_vec_int():
    assert int(bits()) == 0
    assert int(bits("1b0")) == 0
    assert int(bits("1b1")) == -1
    assert int(bits("4b0000")) == 0
    assert int(bits("4b1010")) == -6
    assert int(bits("4b0101")) == 5
    with pytest.raises(ValueError):
        int(bits("4b110X"))
    with pytest.raises(ValueError):
        int(bits("4b-100"))


def test_vec_hash():
    s = set()
    s.add(uint2vec(0))
    s.add(uint2vec(1))
    s.add(uint2vec(2))
    s.add(uint2vec(3))
    s.add(uint2vec(1))
    s.add(uint2vec(2))
    assert len(s) == 4


def test_vec_not():
    x = bits("4b-10X")
    assert x.not_() == bits("4b-01X")
    assert ~x == bits("4b-01X")


def test_vec_nor():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b-0-X_000X_-01X_XXXX"
    v0 = bits(x0)
    v1 = bits(x1)

    assert nor(v0, x1) == yy
    assert nor(v0, v1) == yy
    assert ~(v0 | x1) == yy
    assert ~(x0 | v1) == yy

    # Invalid rhs
    # with pytest.raises(TypeError):
    #    nor(v0, 1.0e42)
    with pytest.raises(TypeError):
        nor(v0, "1b0")


def test_vec_or():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b-1-X_111X_-10X_XXXX"
    v0 = bits(x0)
    v1 = bits(x1)

    assert or_(v0, x1) == yy
    assert or_(v0, v1) == yy
    assert v0 | x1 == yy
    assert x0 | v1 == yy

    # Invalid rhs
    # with pytest.raises(TypeError):
    #    or_(v0, 1.0e42)
    with pytest.raises(TypeError):
        or_(v0, "1b0")


def test_vec_nand():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b--1X_-01X_111X_XXXX"
    v0 = bits(x0)
    v1 = bits(x1)

    assert nand(v0, x1) == yy
    assert nand(v0, v1) == yy
    assert ~(v0 & x1) == yy
    assert ~(x0 & v1) == yy

    # Invalid rhs
    # with pytest.raises(TypeError):
    #    nand(v0, 1.0e42)
    with pytest.raises(TypeError):
        nand(v0, "1b0")


def test_vec_and():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b--0X_-10X_000X_XXXX"
    v0 = bits(x0)
    v1 = bits(x1)

    assert and_(v0, x1) == yy
    assert and_(v0, v1) == yy
    assert v0 & x1 == yy
    assert x0 & v1 == yy

    # Invalid rhs
    # with pytest.raises(TypeError):
    #    and_(v0, 1.0e42)
    with pytest.raises(TypeError):
        and_(v0, "1b0")


def test_vec_xnor():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b---X_-10X_-01X_XXXX"
    v0 = bits(x0)
    v1 = bits(x1)

    assert xnor(v0, x1) == yy
    assert xnor(v0, v1) == yy
    assert ~(v0 ^ x1) == yy
    assert ~(x0 ^ v1) == yy

    # Invalid rhs
    # with pytest.raises(TypeError):
    #    xnor(v0, 1.0e42)
    with pytest.raises(TypeError):
        xnor(v0, "1b0")


def test_vec_xor():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b---X_-01X_-10X_XXXX"
    v0 = bits(x0)
    v1 = bits(x1)

    assert xor(v0, x1) == yy
    assert xor(v0, v1) == yy
    assert v0 ^ x1 == yy
    assert x0 ^ v1 == yy

    # Invalid rhs
    # with pytest.raises(TypeError):
    #    xor(v0, 1.0e42)
    with pytest.raises(TypeError):
        xor(v0, "1b0")


UOR = {
    "2bXX": _X,
    "2bX0": _X,
    "2b0X": _X,
    "2bX1": _X,
    "2b1X": _X,
    "2bX-": _X,
    "2b-X": _X,
    "2b0-": _W,
    "2b-0": _W,
    "2b1-": _1,
    "2b-1": _1,
    "2b--": _W,
    "2b00": _0,
    "2b01": _1,
    "2b10": _1,
    "2b11": _1,
}


def test_vec_uor():
    for lit, (d0, d1) in UOR.items():
        assert bits(lit).uor() == Scalar(d0, d1)


UAND = {
    "2bXX": _X,
    "2bX0": _X,
    "2b0X": _X,
    "2bX1": _X,
    "2b1X": _X,
    "2bX-": _X,
    "2b-X": _X,
    "2b0-": _0,
    "2b-0": _0,
    "2b1-": _W,
    "2b-1": _W,
    "2b--": _W,
    "2b00": _0,
    "2b01": _0,
    "2b10": _0,
    "2b11": _1,
}


def test_vec_uand():
    for lit, (d0, d1) in UAND.items():
        assert bits(lit).uand() == Scalar(d0, d1)


UXNOR = {
    "2bXX": _X,
    "2bX0": _X,
    "2b0X": _X,
    "2bX1": _X,
    "2b1X": _X,
    "2bX-": _X,
    "2b-X": _X,
    "2b0-": _W,
    "2b-0": _W,
    "2b1-": _W,
    "2b-1": _W,
    "2b--": _W,
    "2b00": _1,
    "2b01": _0,
    "2b10": _0,
    "2b11": _1,
}


def test_vec_uxnor():
    for lit, (d0, d1) in UXNOR.items():
        assert bits(lit).uxnor() == Scalar(d0, d1)


UXOR = {
    "2bXX": _X,
    "2bX0": _X,
    "2b0X": _X,
    "2bX1": _X,
    "2b1X": _X,
    "2bX-": _X,
    "2b-X": _X,
    "2b0-": _W,
    "2b-0": _W,
    "2b1-": _W,
    "2b-1": _W,
    "2b--": _W,
    "2b00": _0,
    "2b01": _1,
    "2b10": _1,
    "2b11": _0,
}


def test_vec_uxor():
    for lit, (d0, d1) in UXOR.items():
        assert bits(lit).uxor() == Scalar(d0, d1)


EQ = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
    ("2b00", "2b00", T),
    ("2b00", "2b01", F),
    ("2b00", "2b0X", X),
    ("2b00", "2b0-", W),
    ("2b10", "2b10", T),
    ("2b10", "2b11", F),
    ("2b10", "2b1X", X),
    ("2b10", "2b1-", W),
]


def test_vec_eq():
    for a, b, y in EQ:
        assert bits(a).eq(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").eq("2b00")


NE = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
    ("2b00", "2b00", F),
    ("2b00", "2b01", T),
    ("2b00", "2b0X", X),
    ("2b00", "2b0-", W),
    ("2b10", "2b10", F),
    ("2b10", "2b11", T),
    ("2b10", "2b1X", X),
    ("2b10", "2b1-", W),
]


def test_vec_ne():
    for a, b, y in NE:
        assert bits(a).ne(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").ne("2b00")


LT = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", T),
    ("1b1", "1b0", F),
    ("1b1", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
]


def test_vec_lt():
    for a, b, y in LT:
        assert bits(a).lt(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").lt("2b00")


LE = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", T),
    ("1b1", "1b0", F),
    ("1b1", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
]


def test_vec_le():
    for a, b, y in LE:
        assert bits(a).le(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").le("2b00")


SLT = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", F),
    ("1b1", "1b0", T),
    ("1b1", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
]


def test_vec_slt():
    for a, b, y in SLT:
        assert bits(a).slt(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").slt("2b00")


SLE = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", F),
    ("1b1", "1b0", T),
    ("1b1", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
]


def test_vec_sle():
    for a, b, y in SLE:
        assert bits(a).sle(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").sle("2b00")


GT = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", F),
    ("1b1", "1b0", T),
    ("1b1", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
]


def test_vec_gt():
    for a, b, y in GT:
        assert bits(a).gt(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").gt("2b00")


GE = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", F),
    ("1b1", "1b0", T),
    ("1b1", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
]


def test_vec_ge():
    for a, b, y in GE:
        assert bits(a).ge(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").ge("2b00")


SGT = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", T),
    ("1b1", "1b0", F),
    ("1b1", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
]


def test_vec_sgt():
    for a, b, y in SGT:
        assert bits(a).sgt(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").sgt("2b00")


SGE = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", T),
    ("1b1", "1b0", F),
    ("1b1", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", W),
]


def test_vec_sge():
    for a, b, y in SGE:
        assert bits(a).sge(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        bits("1b0").sge("2b00")


def test_vec_xt():
    v = bits("4b1010")
    with pytest.raises(ValueError):
        v.xt(-1)
    assert v.xt(0) is v
    assert v.xt(4) == bits("8b0000_1010")


def test_vec_sxt():
    v1 = bits("4b1010")
    v2 = bits("4b0101")
    with pytest.raises(ValueError):
        v1.sxt(-1)
    assert v1.sxt(0) is v1
    assert v1.sxt(4) == bits("8b1111_1010")
    assert v2.sxt(0) is v2
    assert v2.sxt(4) == bits("8b0000_0101")


def test_vec_lsh():
    v = bits("4b1111")
    y, co = v.lsh(0)
    assert y is v and co == E
    assert v.lsh(1) == ("4b1110", "1b1")
    assert v.lsh(2) == ("4b1100", "2b11")
    assert v << 2 == "4b1100"
    assert "4b1111" << bits("2b10") == "4b1100"
    assert v.lsh(3) == ("4b1000", "3b111")
    assert v.lsh(4) == ("4b0000", "4b1111")

    with pytest.raises(ValueError):
        v.lsh(-1)
    with pytest.raises(ValueError):
        v.lsh(5)

    assert v.lsh(2, ci=bits("2b01")) == ("4b1101", "2b11")
    with pytest.raises(ValueError):
        v.lsh(2, ci=bits("3b000"))

    assert bits("2b01").lsh(bits("1bX")) == (bits("2bXX"), E)
    assert bits("2b01").lsh(bits("1b-")) == (bits("2b--"), E)
    assert bits("2b01").lsh(bits("1b1")) == (bits("2b10"), F)


def test_vec_lrot():
    v = bits("4b-10X")
    assert v.lrot(0) is v
    assert str(v.lrot(1)) == "4b10X-"
    assert str(v.lrot(2)) == "4b0X-1"
    assert str(v.lrot(bits("2b10"))) == "4b0X-1"
    assert str(v.lrot(bits("2b1-"))) == "4b----"
    assert str(v.lrot(bits("2b1X"))) == "4bXXXX"
    assert str(v.lrot(3)) == "4bX-10"

    with pytest.raises(ValueError):
        str(v.lrot(4))


def test_vec_rrot():
    v = bits("4b-10X")
    assert v.rrot(0) is v
    assert str(v.rrot(1)) == "4bX-10"
    assert str(v.rrot(2)) == "4b0X-1"
    assert str(v.rrot(bits("2b10"))) == "4b0X-1"
    assert str(v.rrot(bits("2b1-"))) == "4b----"
    assert str(v.rrot(bits("2b1X"))) == "4bXXXX"
    assert str(v.rrot(3)) == "4b10X-"

    with pytest.raises(ValueError):
        str(v.rrot(4))


def test_vec_rsh():
    v = bits("4b1111")
    y, co = v.rsh(0)
    assert y is v and co == E
    assert v.rsh(1) == ("4b0111", "1b1")
    assert v.rsh(2) == ("4b0011", "2b11")
    assert v >> 2 == "4b0011"
    assert "4b1111" >> bits("2b10") == "4b0011"
    assert v.rsh(3) == ("4b0001", "3b111")
    assert v.rsh(4) == ("4b0000", "4b1111")

    with pytest.raises(ValueError):
        v.rsh(-1)
    with pytest.raises(ValueError):
        v.rsh(5)

    assert v.rsh(2, ci=bits("2b10")) == ("4b1011", "2b11")
    with pytest.raises(ValueError):
        v.rsh(2, ci=bits("3b000"))

    assert bits("2b01").rsh(bits("1bX")) == (bits("2bXX"), E)
    assert bits("2b01").rsh(bits("1b-")) == (bits("2b--"), E)
    assert bits("2b01").rsh(bits("1b1")) == (bits("2b00"), T)


def test_vec_srsh():
    v = bits("4b1111")
    assert v.srsh(0) == ("4b1111", E)
    assert v.srsh(1) == ("4b1111", "1b1")
    assert v.srsh(2) == ("4b1111", "2b11")
    assert v.srsh(3) == ("4b1111", "3b111")
    assert v.srsh(4) == ("4b1111", "4b1111")

    v = bits("4b0111")
    assert v.srsh(0) == ("4b0111", E)
    assert v.srsh(1) == ("4b0011", "1b1")
    assert v.srsh(2) == ("4b0001", "2b11")
    assert v.srsh(3) == ("4b0000", "3b111")
    assert v.srsh(4) == ("4b0000", "4b0111")

    with pytest.raises(ValueError):
        v.srsh(-1)
    with pytest.raises(ValueError):
        v.srsh(5)

    assert bits("2b01").srsh(bits("1bX")) == (bits("2bXX"), E)
    assert bits("2b01").srsh(bits("1b-")) == (bits("2b--"), E)
    assert bits("2b01").srsh(bits("1b1")) == (bits("2b00"), T)


ADD_VALS = [
    ("2b00", "2b00", "1b0", "2b00", F),
    ("2b00", "2b01", "1b0", "2b01", F),
    ("2b00", "2b10", "1b0", "2b10", F),
    ("2b00", "2b11", "1b0", "2b11", F),
    ("2b01", "2b00", "1b0", "2b01", F),
    ("2b01", "2b01", "1b0", "2b10", F),  # overflow
    ("2b01", "2b10", "1b0", "2b11", F),
    ("2b01", "2b11", "1b0", "2b00", T),
    ("2b10", "2b00", "1b0", "2b10", F),
    ("2b10", "2b01", "1b0", "2b11", F),
    ("2b10", "2b10", "1b0", "2b00", T),  # overflow
    ("2b10", "2b11", "1b0", "2b01", T),  # overflow
    ("2b11", "2b00", "1b0", "2b11", F),
    ("2b11", "2b01", "1b0", "2b00", T),
    ("2b11", "2b10", "1b0", "2b01", T),  # overflow
    ("2b11", "2b11", "1b0", "2b10", T),
    ("2b00", "2b00", "1b1", "2b01", F),
    ("2b00", "2b01", "1b1", "2b10", F),  # overflow
    ("2b00", "2b10", "1b1", "2b11", F),
    ("2b00", "2b11", "1b1", "2b00", T),
    ("2b01", "2b00", "1b1", "2b10", F),  # overflow
    ("2b01", "2b01", "1b1", "2b11", F),  # overflow
    ("2b01", "2b10", "1b1", "2b00", T),
    ("2b01", "2b11", "1b1", "2b01", T),
    ("2b10", "2b00", "1b1", "2b11", F),
    ("2b10", "2b01", "1b1", "2b00", T),
    ("2b10", "2b10", "1b1", "2b01", T),  # overflow
    ("2b10", "2b11", "1b1", "2b10", T),
    ("2b11", "2b00", "1b1", "2b00", T),
    ("2b11", "2b01", "1b1", "2b01", T),
    ("2b11", "2b10", "1b1", "2b10", T),
    ("2b11", "2b11", "1b1", "2b11", T),
    ("2b0X", "2b00", F, "2bXX", X),
    ("2b00", "2b0X", F, "2bXX", X),
    ("2b00", "2b00", X, "2bXX", X),
    ("2b0-", "2b00", F, "2b--", W),
    ("2b00", "2b0-", F, "2b--", W),
    ("2b00", "2b00", W, "2b--", W),
]


def test_vec_add():
    """Test bits add method."""
    for a, b, ci, s, co in ADD_VALS:
        assert add(a, b, ci) == (s, co)
        if ci == F:
            assert bits(a) + b == cat(s, co)
            assert a + bits(b) == cat(s, co)


SUB_VALS = [
    ("2b00", "2b00", "2b00", T),
    ("2b00", "2b01", "2b11", F),
    ("2b00", "2b10", "2b10", F),
    ("2b00", "2b11", "2b01", F),
    ("2b01", "2b00", "2b01", T),
    ("2b01", "2b01", "2b00", T),
    ("2b01", "2b10", "2b11", F),
    ("2b01", "2b11", "2b10", F),
    ("2b10", "2b00", "2b10", T),
    ("2b10", "2b01", "2b01", T),
    ("2b10", "2b10", "2b00", T),
    ("2b10", "2b11", "2b11", F),
    ("2b11", "2b00", "2b11", T),
    ("2b11", "2b01", "2b10", T),
    ("2b11", "2b10", "2b01", T),
    ("2b11", "2b11", "2b00", T),
    ("2b0X", "2b00", "2bXX", X),
    ("2b00", "2b0X", "2bXX", X),
    ("2b0-", "2b00", "2b--", W),
    ("2b00", "2b0-", "2b--", W),
]


def test_vec_sub():
    for a, b, s, co in SUB_VALS:
        assert sub(a, b) == (s, co)
        assert bits(a) - b == cat(s, co)
        assert a - bits(b) == cat(s, co)


def test_vec_neg():
    assert -bits("3b000") == "4b1000"
    assert -bits("3b001") == "4b0111"
    assert -bits("3b111") == "4b0001"
    assert -bits("3b010") == "4b0110"
    assert -bits("3b110") == "4b0010"
    assert -bits("3b011") == "4b0101"
    assert -bits("3b101") == "4b0011"
    assert -bits("3b100") == "4b0100"


def test_count():
    v = bits("8b-10X_-10X")
    assert v.count_xes() == 2
    assert v.count_zeros() == 2
    assert v.count_ones() == 2
    assert v.count_dcs() == 2
    assert v.count_unknown() == 4

    assert not bits("4b0000").onehot()
    assert bits("4b1000").onehot()
    assert bits("4b0001").onehot()
    assert not bits("4b1001").onehot()
    assert not bits("4b1101").onehot()

    assert bits("4b0000").onehot0()
    assert bits("4b1000").onehot0()
    assert not bits("4b1010").onehot0()
    assert not bits("4b1011").onehot0()

    assert not bits("4b0000").has_x()
    assert bits("4b00X0").has_x()
    assert not bits("4b0000").has_dc()
    assert bits("4b00-0").has_dc()
    assert not bits("4b0000").has_unknown()
    assert bits("4b00X0").has_unknown()
    assert bits("4b00-0").has_unknown()
