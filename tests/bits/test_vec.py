"""Test seqlogic.bits module."""

import pytest

from seqlogic import (
    Scalar,
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
    ite,
    le,
    lrot,
    lsh,
    lt,
    mod,
    mul,
    mux,
    nand,
    ne,
    nor,
    not_,
    or_,
    rep,
    rrot,
    rsh,
    sbc,
    sge,
    sgt,
    sle,
    slt,
    srsh,
    sub,
    sxt,
    u2bv,
    uand,
    uor,
    uxnor,
    uxor,
    xnor,
    xor,
    xt,
)
from seqlogic._lbool import _W, _X, _0, _1

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


def test_lit2bv_dec():
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


def test_lit2bv_hex():
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


U2BV_VALS = {
    0: "[]",
    1: "1b1",
    2: "2b10",
    3: "2b11",
    4: "3b100",
    5: "3b101",
    6: "3b110",
    7: "3b111",
    8: "4b1000",
}

U2BV_N_VALS = {
    (0, 0): "[]",
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


def test_u2bv():
    # Negative inputs are invalid
    with pytest.raises(ValueError):
        u2bv(-1)

    for i, s in U2BV_VALS.items():
        v = u2bv(i)
        assert str(v) == s
        assert v.to_uint() == i

    for (i, n), s in U2BV_N_VALS.items():
        v = u2bv(i, n)
        assert str(v) == s
        assert v.to_uint() == i

    # Overflows
    with pytest.raises(ValueError):
        u2bv(1, 0)
    with pytest.raises(ValueError):
        u2bv(2, 0)
    with pytest.raises(ValueError):
        u2bv(2, 1)
    with pytest.raises(ValueError):
        u2bv(3, 0)
    with pytest.raises(ValueError):
        u2bv(3, 1)


I2BV_VALS = {
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

I2BV_N_VALS = {
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


def test_i2bv():
    for i, s in I2BV_VALS.items():
        v = i2bv(i)
        assert str(v) == s
        assert v.to_int() == i

    for (i, n), s in I2BV_N_VALS.items():
        v = i2bv(i, n)
        assert str(v) == s
        assert v.to_int() == i

    # Overflows
    with pytest.raises(ValueError):
        i2bv(-5, 3)
    with pytest.raises(ValueError):
        i2bv(-4, 2)
    with pytest.raises(ValueError):
        i2bv(-3, 2)
    with pytest.raises(ValueError):
        i2bv(-2, 1)
    with pytest.raises(ValueError):
        i2bv(-1, 0)
    with pytest.raises(ValueError):
        i2bv(0, 0)
    with pytest.raises(ValueError):
        i2bv(1, 1)
    with pytest.raises(ValueError):
        i2bv(2, 2)
    with pytest.raises(ValueError):
        i2bv(3, 2)
    with pytest.raises(ValueError):
        i2bv(4, 3)


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
    assert repr(bits()) == "bits([])"
    assert repr(bits("1b0")) == 'bits("1b0")'
    assert repr(bits("4b-10X")) == 'bits("4b-10X")'


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
    s.add(u2bv(0))
    s.add(u2bv(1))
    s.add(u2bv(2))
    s.add(u2bv(3))
    s.add(u2bv(1))
    s.add(u2bv(2))
    assert len(s) == 4


def test_vec_not():
    x = bits("4b-10X")
    assert not_(x) == bits("4b-01X")
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


ITE = (
    ("1bX", "1bX", "1bX", "1bX"),
    ("1bX", "1bX", "1bX", "1bX"),
    ("1bX", "1bX", "1b1", "1bX"),
    ("1bX", "1bX", "1b0", "1bX"),
    ("1bX", "1bX", "1b-", "1bX"),
    ("1bX", "1b1", "1bX", "1bX"),
    ("1bX", "1b1", "1b1", "1bX"),
    ("1bX", "1b1", "1b0", "1bX"),
    ("1bX", "1b1", "1b-", "1bX"),
    ("1bX", "1b0", "1bX", "1bX"),
    ("1bX", "1b0", "1b1", "1bX"),
    ("1bX", "1b0", "1b0", "1bX"),
    ("1bX", "1b0", "1b-", "1bX"),
    ("1bX", "1b-", "1bX", "1bX"),
    ("1bX", "1b-", "1b1", "1bX"),
    ("1bX", "1b-", "1b0", "1bX"),
    ("1bX", "1b-", "1b-", "1bX"),
    ("1b1", "1bX", "1bX", "1bX"),
    ("1b1", "1bX", "1b1", "1bX"),
    ("1b1", "1bX", "1b0", "1bX"),
    ("1b1", "1bX", "1b-", "1bX"),
    ("1b1", "1b1", "1bX", "1bX"),
    ("1b1", "1b1", "1b1", "1b1"),
    ("1b1", "1b1", "1b0", "1b1"),
    ("1b1", "1b1", "1b-", "1b1"),
    ("1b1", "1b0", "1bX", "1bX"),
    ("1b1", "1b0", "1b1", "1b0"),
    ("1b1", "1b0", "1b0", "1b0"),
    ("1b1", "1b0", "1b-", "1b0"),
    ("1b1", "1b-", "1bX", "1bX"),
    ("1b1", "1b-", "1b1", "1b-"),
    ("1b1", "1b-", "1b0", "1b-"),
    ("1b1", "1b-", "1b-", "1b-"),
    ("1b0", "1bX", "1bX", "1bX"),
    ("1b0", "1bX", "1b1", "1bX"),
    ("1b0", "1bX", "1b0", "1bX"),
    ("1b0", "1bX", "1b-", "1bX"),
    ("1b0", "1b1", "1bX", "1bX"),
    ("1b0", "1b1", "1b1", "1b1"),
    ("1b0", "1b1", "1b0", "1b0"),
    ("1b0", "1b1", "1b-", "1b-"),
    ("1b0", "1b0", "1bX", "1bX"),
    ("1b0", "1b0", "1b1", "1b1"),
    ("1b0", "1b0", "1b0", "1b0"),
    ("1b0", "1b0", "1b-", "1b-"),
    ("1b0", "1b-", "1bX", "1bX"),
    ("1b0", "1b-", "1b1", "1b1"),
    ("1b0", "1b-", "1b0", "1b0"),
    ("1b0", "1b-", "1b-", "1b-"),
    ("1b-", "1bX", "1bX", "1bX"),
    ("1b-", "1bX", "1b1", "1bX"),
    ("1b-", "1bX", "1b0", "1bX"),
    ("1b-", "1bX", "1b-", "1bX"),
    ("1b-", "1b1", "1bX", "1bX"),
    ("1b-", "1b1", "1b1", "1b1"),
    ("1b-", "1b1", "1b0", "1b-"),
    ("1b-", "1b1", "1b-", "1b-"),
    ("1b-", "1b0", "1bX", "1bX"),
    ("1b-", "1b0", "1b1", "1b-"),
    ("1b-", "1b0", "1b0", "1b0"),
    ("1b-", "1b0", "1b-", "1b-"),
    ("1b-", "1b-", "1bX", "1bX"),
    ("1b-", "1b-", "1b1", "1b-"),
    ("1b-", "1b-", "1b0", "1b-"),
    ("1b-", "1b-", "1b-", "1b-"),
)


def test_vec_ite():
    for s, a, b, y in ITE:
        assert ite(s, a, b) == y


def test_vec_mux():
    # Invalid x[n] argument name
    with pytest.raises(ValueError):
        mux("2b00", x4="4b0000")
    with pytest.raises(ValueError):
        mux("2b00", foo="4b0000")
    # Mismatching sizes
    with pytest.raises(TypeError):
        mux("2b00", x0="4b0000", x1="8h00")
    # No inputs
    with pytest.raises(ValueError):
        mux("2b00")

    assert mux(bits(), x0="4b1010") == "4b1010"

    assert mux("2b00", x0="4b10_00", x1="4b10_01", x2="4b10_10", x3="4b10_11") == "4b1000"
    assert mux("2b01", x0="4b10_00", x1="4b10_01", x2="4b10_10", x3="4b10_11") == "4b1001"
    assert mux("2b10", x0="4b10_00", x1="4b10_01", x2="4b10_10", x3="4b10_11") == "4b1010"
    assert mux("2b11", x0="4b10_00", x1="4b10_01", x2="4b10_10", x3="4b10_11") == "4b1011"

    assert mux("2b0-", x0="4b10_00", x1="4b10_01", x2="4b10_10", x3="4b10_11") == "4b100-"
    assert mux("2b-0", x0="4b10_00", x1="4b10_01", x2="4b10_10", x3="4b10_11") == "4b10-0"
    assert mux("2b1-", x0="4b10_00", x1="4b10_01", x2="4b10_10", x3="4b10_11") == "4b101-"
    assert mux("2b-1", x0="4b10_00", x1="4b10_01", x2="4b10_10", x3="4b10_11") == "4b10-1"


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
        assert uor(lit) == Scalar(d0, d1)


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
        assert uand(lit) == Scalar(d0, d1)


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
        assert uxnor(lit) == Scalar(d0, d1)


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
        assert uxor(lit) == Scalar(d0, d1)


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
        assert eq(bits(a), b) == y
        assert eq(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        eq("1b0", "2b00")


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
        assert ne(bits(a), b) == y
        assert ne(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        ne("1b0", "2b00")


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
        assert lt(bits(a), b) == y
        assert lt(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        lt("1b0", "2b00")


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
        assert le(bits(a), b) == y
        assert le(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        le("1b0", "2b00")


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
        assert slt(bits(a), b) == y
        assert slt(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        slt("1b0", "2b00")


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
        assert sle(bits(a), b) == y
        assert sle(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        sle("1b0", "2b00")


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
        assert gt(bits(a), b) == y
        assert gt(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        gt("1b0", "2b00")


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
        assert ge(bits(a), b) == y
        assert ge(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        ge("1b0", "2b00")


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
        assert sgt(bits(a), b) == y
        assert sgt(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        sgt("1b0", "2b00")


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
        assert sge(bits(a), b) == y
        assert sge(a, bits(b)) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        sge("1b0", "2b00")


def test_vec_xt():
    v = bits("4b1010")
    with pytest.raises(ValueError):
        xt(v, -1)
    assert xt(v, 0) is v
    assert xt(v, 4) == bits("8b0000_1010")


def test_vec_sxt():
    v1 = bits("4b1010")
    v2 = bits("4b0101")
    with pytest.raises(ValueError):
        sxt(v1, -1)
    assert sxt(v1, 0) is v1
    assert sxt(v1, 4) == bits("8b1111_1010")
    assert sxt(v2, 0) is v2
    assert sxt(v2, 4) == bits("8b0000_0101")


def test_vec_lsh():
    v = bits("4b1111")
    y = lsh(v, 0)
    assert y is v
    assert lsh(v, 1) == "4b1110"
    assert lsh(v, 2) == "4b1100"
    assert v << 2 == "4b1100"
    assert "4b1111" << bits("2b10") == "4b1100"
    assert lsh(v, 3) == "4b1000"
    assert lsh(v, 4) == "4b0000"

    with pytest.raises(ValueError):
        lsh(v, -1)
    with pytest.raises(ValueError):
        lsh(v, 5)
    with pytest.raises(TypeError):
        lsh(v, 0.5)

    assert lsh("2b01", "1bX") == bits("2bXX")
    assert lsh("2b01", "1b-") == bits("2b--")
    assert lsh("2b01", "1b1") == bits("2b10")


def test_vec_lrot():
    v = bits("4b-10X")
    assert lrot(v, 0) is v
    assert str(lrot(v, 1)) == "4b10X-"
    assert str(lrot(v, 2)) == "4b0X-1"
    assert str(lrot(v, "2b10")) == "4b0X-1"
    assert str(lrot(v, "2b1-")) == "4b----"
    assert str(lrot(v, "2b1X")) == "4bXXXX"
    assert str(lrot(v, 3)) == "4bX-10"

    with pytest.raises(ValueError):
        str(lrot(v, 4))


def test_vec_rrot():
    v = bits("4b-10X")
    assert rrot(v, 0) is v
    assert str(rrot(v, 1)) == "4bX-10"
    assert str(rrot(v, 2)) == "4b0X-1"
    assert str(rrot(v, "2b10")) == "4b0X-1"
    assert str(rrot(v, "2b1-")) == "4b----"
    assert str(rrot(v, "2b1X")) == "4bXXXX"
    assert str(rrot(v, 3)) == "4b10X-"

    with pytest.raises(ValueError):
        str(rrot(v, 4))


def test_vec_rsh():
    v = bits("4b1111")
    y = rsh(v, 0)
    assert y is v
    assert rsh(v, 1) == "4b0111"
    assert rsh(v, 2) == "4b0011"
    assert v >> 2 == "4b0011"
    assert "4b1111" >> bits("2b10") == "4b0011"
    assert rsh(v, 3) == "4b0001"
    assert rsh(v, 4) == "4b0000"

    with pytest.raises(ValueError):
        rsh(v, -1)
    with pytest.raises(ValueError):
        rsh(v, 5)

    assert rsh("2b01", "1bX") == bits("2bXX")
    assert rsh("2b01", "1b-") == bits("2b--")
    assert rsh("2b01", "1b1") == bits("2b00")


def test_vec_srsh():
    v = bits("4b1111")
    assert srsh(v, 0) == "4b1111"
    assert srsh(v, 1) == "4b1111"
    assert srsh(v, 2) == "4b1111"
    assert srsh(v, 3) == "4b1111"
    assert srsh(v, 4) == "4b1111"

    v = bits("4b0111")
    assert srsh(v, 0) == "4b0111"
    assert srsh(v, 1) == "4b0011"
    assert srsh(v, 2) == "4b0001"
    assert srsh(v, 3) == "4b0000"
    assert srsh(v, 4) == "4b0000"

    with pytest.raises(ValueError):
        srsh(v, -1)
    with pytest.raises(ValueError):
        srsh(v, 5)

    assert srsh("2b01", "1bX") == bits("2bXX")
    assert srsh("2b01", "1b-") == bits("2b--")
    assert srsh("2b01", "1b1") == bits("2b00")


DEC_VALS = [
    (bits(), "1b1"),
    ("1b0", "2b01"),
    ("1b1", "2b10"),
    ("2b00", "4b0001"),
    ("2b01", "4b0010"),
    ("2b10", "4b0100"),
    ("2b11", "4b1000"),
    ("1b-", "2b--"),
    ("1bX", "2bXX"),
]


def test_decode():
    for x, y in DEC_VALS:
        assert y == decode(x)


ENC_OH_VALS = [
    ("1b1", bits()),
    ("2b01", "1b0"),
    ("2b10", "1b1"),
    ("3b001", "2b00"),
    ("3b010", "2b01"),
    ("3b100", "2b10"),
    ("4b0001", "2b00"),
    ("4b0010", "2b01"),
    ("4b0100", "2b10"),
    ("4b1000", "2b11"),
    ("2b--", "1b-"),
    ("2bXX", "1bX"),
]


def test_encode_onehot():
    # Not a valid one-hot encoding
    with pytest.raises(ValueError):
        encode_onehot("1b0")
    with pytest.raises(ValueError):
        encode_onehot("2b00")

    for x, y in ENC_OH_VALS:
        assert y == encode_onehot(x)


ENC_PRI_VALS = [
    ("1b0", (E, "1b0")),
    ("1b1", (E, "1b1")),
    ("2b00", ("1b-", "1b0")),
    ("2b01", ("1b0", "1b1")),
    ("2b10", ("1b1", "1b1")),
    ("2b11", ("1b1", "1b1")),
    ("3b000", ("2b--", "1b0")),
    ("3b001", ("2b00", "1b1")),
    ("3b010", ("2b01", "1b1")),
    ("3b011", ("2b01", "1b1")),
    ("3b100", ("2b10", "1b1")),
    ("3b101", ("2b10", "1b1")),
    ("3b110", ("2b10", "1b1")),
    ("3b111", ("2b10", "1b1")),
    ("4b0000", ("2b--", "1b0")),
    ("4b0001", ("2b00", "1b1")),
    ("4b0010", ("2b01", "1b1")),
    ("4b0011", ("2b01", "1b1")),
    ("4b001-", ("2b01", "1b1")),
    ("4b0100", ("2b10", "1b1")),
    ("4b0101", ("2b10", "1b1")),
    ("4b0110", ("2b10", "1b1")),
    ("4b0111", ("2b10", "1b1")),
    ("4b01--", ("2b10", "1b1")),
    ("4b1000", ("2b11", "1b1")),
    ("4b1001", ("2b11", "1b1")),
    ("4b1010", ("2b11", "1b1")),
    ("4b1011", ("2b11", "1b1")),
    ("4b1100", ("2b11", "1b1")),
    ("4b1101", ("2b11", "1b1")),
    ("4b1110", ("2b11", "1b1")),
    ("4b1111", ("2b11", "1b1")),
    ("4b1---", ("2b11", "1b1")),
    # DC Propagation
    ("1b-", (E, "1b-")),
    ("2b--", ("1b-", "1b-")),
    ("2b0-", ("1b-", "1b-")),
    ("3b---", ("2b--", "1b-")),
    ("3b0--", ("2b--", "1b-")),
    ("3b00-", ("2b--", "1b-")),
    ("4b----", ("2b--", "1b-")),
    ("4b0---", ("2b--", "1b-")),
    ("4b00--", ("2b--", "1b-")),
    ("4b000-", ("2b--", "1b-")),
    # X Propagation
    ("2bXX", ("1bX", "1bX")),
    ("3b10X", ("2bXX", "1bX")),
]


def test_encode_priority():
    for x, y in ENC_PRI_VALS:
        assert y == encode_priority(x)


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
        assert adc(a, b, ci) == cat(s, co)
        assert add(a, b, ci) == s
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
        assert sbc(a, b) == cat(s, co)
        assert sub(a, b) == s
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


MUL_VALS = [
    ("1b0", "1b0", "2b00"),
    ("1b0", "1b1", "2b00"),
    ("1b1", "1b0", "2b00"),
    ("1b1", "1b1", "2b01"),
    ("2b00", "2b00", "4b0000"),
    ("2b00", "2b01", "4b0000"),
    ("2b00", "2b10", "4b0000"),
    ("2b00", "2b11", "4b0000"),
    ("2b01", "2b00", "4b0000"),
    ("2b01", "2b01", "4b0001"),
    ("2b01", "2b10", "4b0010"),
    ("2b01", "2b11", "4b0011"),
    ("2b10", "2b00", "4b0000"),
    ("2b10", "2b01", "4b0010"),
    ("2b10", "2b10", "4b0100"),
    ("2b10", "2b11", "4b0110"),
    ("2b11", "2b00", "4b0000"),
    ("2b11", "2b01", "4b0011"),
    ("2b11", "2b10", "4b0110"),
    ("2b11", "2b11", "4b1001"),
    ("2b0X", "2b00", "4bXXXX"),
    ("2b0-", "2b00", "4b----"),
]


def test_vec_mul():
    # Empty X Empty = Empty
    assert mul(bits(), bits()) == bits()

    for a, b, p in MUL_VALS:
        assert mul(a, b) == p
        assert bits(a) * b == p
        assert a * bits(b) == p


DIV_VALS = [
    ("1b0", "1b1", "1b0"),
    ("1b1", "1b1", "1b1"),
    ("8d42", "8d7", "8d6"),
    ("8d42", "8d6", "8d7"),
    ("8d42", "4d6", "8d7"),
    ("8d42", "8d8", "8d5"),
    ("8d42", "8d9", "8d4"),
    ("8d42", "8d10", "8d4"),
    ("8d42", "4bXXXX", "8bXXXX_XXXX"),
    ("8d42", "4b----", "8b----_----"),
]


def test_vec_div():
    # Cannot divide by empty
    with pytest.raises(ValueError):
        div("2b00", bits())
    # Cannot divide by zero
    with pytest.raises(ZeroDivisionError):
        div("8d42", "8d0")
    # Divisor cannot be wider than dividend
    with pytest.raises(ValueError):
        div("2b00", "8d42")

    for a, b, q in DIV_VALS:
        assert div(a, b) == q


MOD_VALS = [
    ("1b0", "1b1", "1b0"),
    ("1b1", "1b1", "1b0"),
    ("8d42", "8d7", "8d0"),
    ("8d42", "8d6", "8d0"),
    ("8d42", "4d6", "4d0"),
    ("8d42", "8d8", "8d2"),
    ("8d42", "8d9", "8d6"),
    ("8d42", "8d10", "8d2"),
    ("8d42", "4bXXXX", "4bXXXX"),
    ("8d42", "4b----", "4b----"),
]


def test_vec_mod():
    # Cannot divide by empty
    with pytest.raises(ValueError):
        mod("2b00", bits())
    # Cannot divide by zero
    with pytest.raises(ZeroDivisionError):
        mod("8d42", "8d0")
    # Divisor cannot be wider than dividend
    with pytest.raises(ValueError):
        mod("2b00", "8d42")

    for a, b, r in MOD_VALS:
        assert mod(a, b) == r


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


def test_reshape():
    v = bits("4b1010")
    assert v.reshape(v.shape) is v
    assert v.flatten() is v
    with pytest.raises(ValueError):
        v.reshape((5,))
    assert str(v.reshape((2, 2))) == "[2b10, 2b10]"
