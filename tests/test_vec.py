"""Test seqlogic.vec module."""

import pytest

from seqlogic.lbconst import _W, _X, _0, _1
from seqlogic.vec import Vec, cat, int2vec, rep, uint2vec, vec

E = Vec[0](*_X)
X = Vec[1](*_X)
F = Vec[1](*_0)
T = Vec[1](*_1)
W = Vec[1](*_W)


def test_vec_class_getitem():
    # Negative values are illegal
    with pytest.raises(ValueError):
        _ = Vec[-1]

    vec_0 = Vec[0]
    assert vec_0.n == 0 and vec_0.dmax == 0

    vec_4 = Vec[4]
    assert vec_4.n == 4 and vec_4.dmax == 15

    # Always return the same class instance
    assert Vec[0] is vec_0
    assert Vec[4] is vec_4


def test_vec():
    # None/Empty
    assert vec() == E
    assert vec(None) == E
    assert vec([]) == E

    # Single bool input
    assert vec(False) == F
    assert vec(0) == F
    assert vec(True) == T
    assert vec(1) == T

    # Sequence of bools
    assert vec([False, True, 0, 1]) == Vec[4](0b0101, 0b1010)

    # String
    assert vec("4b-10X") == Vec[4](0b1010, 0b1100)

    # Invalid input type
    with pytest.raises(TypeError):
        vec(1.0e42)


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
        v = vec(lit)
        assert len(v) == n and v.data == (d1 ^ v.dmax, d1)

    # Valid inputs w/ X
    v = vec("4b-1_0X")
    assert len(v) == 4 and v.data == (0b1010, 0b1100)
    v = vec("4bX01-")
    assert len(v) == 4 and v.data == (0b0101, 0b0011)

    # Not a literal
    with pytest.raises(ValueError):
        vec("invalid")

    # Size cannot be zero
    with pytest.raises(ValueError):
        vec("0b0")
    # Contains illegal characters
    with pytest.raises(ValueError):
        vec("4b1XW0")

    # Size is too big
    with pytest.raises(ValueError):
        vec("8b1010")

    # Size is too small
    with pytest.raises(ValueError):
        vec("4b1010_1010")


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
        v = vec(lit)
        assert len(v) == n and v.data == (d1 ^ v.dmax, d1)

    # Not a literal
    with pytest.raises(ValueError):
        vec("invalid")

    # Size cannot be zero
    with pytest.raises(ValueError):
        vec("0h0")
    # Contains illegal characters
    with pytest.raises(ValueError):
        vec("8hd3@d_b33f")

    # Size is too big
    with pytest.raises(ValueError):
        vec("16hdead_beef")

    # Size is too small
    with pytest.raises(ValueError):
        vec("8hdead")

    # Invalid characters
    with pytest.raises(ValueError):
        vec("3h8")  # Only 0..7 is legal
    with pytest.raises(ValueError):
        vec("5h20")  # Only 0..1F is legal


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
    v = vec("4b-10X")
    assert cat() == vec()
    assert cat(v) == v
    assert cat("2b0X", "2b-1") == vec("4b-10X")
    assert cat(vec("2b0X"), vec("2b-1")) == vec("4b-10X")
    assert cat(0, 1) == vec("2b10")

    with pytest.raises(TypeError):
        _ = cat(v, 42)


def test_rep():
    assert rep(vec(), 4) == vec()
    assert rep(vec("4b-10X"), 2) == vec("8b-10X_-10X")


def test_vec_getitem():
    v = vec("4b-10X")

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
        _ = v[1.0e42]  # pyright: ignore[reportArgumentType]


def test_vec_iter():
    v = vec("4b-10X")
    assert list(v) == ["1bX", "1b0", "1b1", "1b-"]


def test_vec_repr():
    assert repr(vec()) == "Vec[0](0b0, 0b0)"
    assert repr(vec("4b-10X")) == "Vec[4](0b1010, 0b1100)"


def test_vec_bool():
    assert bool(vec()) is False
    assert bool(vec("1b0")) is False
    assert bool(vec("1b1")) is True
    assert bool(vec("4b0000")) is False
    assert bool(vec("4b1010")) is True
    assert bool(vec("4b0101")) is True
    with pytest.raises(ValueError):
        bool(vec("4b110X"))
    with pytest.raises(ValueError):
        bool(vec("4b-100"))


def test_vec_int():
    assert int(vec()) == 0
    assert int(vec("1b0")) == 0
    assert int(vec("1b1")) == -1
    assert int(vec("4b0000")) == 0
    assert int(vec("4b1010")) == -6
    assert int(vec("4b0101")) == 5
    with pytest.raises(ValueError):
        int(vec("4b110X"))
    with pytest.raises(ValueError):
        int(vec("4b-100"))


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
    x = vec("4b-10X")
    assert x.not_() == vec("4b-01X")
    assert ~x == vec("4b-01X")


def test_vec_nor():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b-0-X_000X_-01X_XXXX"
    v0 = vec(x0)
    v1 = vec(x1)

    assert v0.nor(x1) == yy
    assert v0.nor(v1) == yy
    assert ~(v0 | x1) == yy
    assert ~(x0 | v1) == yy

    # Invalid rhs
    with pytest.raises(TypeError):
        v0.nor(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        v0.nor("1b0")


def test_vec_or():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b-1-X_111X_-10X_XXXX"
    v0 = vec(x0)
    v1 = vec(x1)

    assert v0.or_(x1) == yy
    assert v0.or_(v1) == yy
    assert v0 | x1 == yy
    assert x0 | v1 == yy

    # Invalid rhs
    with pytest.raises(TypeError):
        v0.nor(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        v0.nor("1b0")


def test_vec_nand():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b--1X_-01X_111X_XXXX"
    v0 = vec(x0)
    v1 = vec(x1)

    assert v0.nand(x1) == yy
    assert v0.nand(v1) == yy
    assert ~(v0 & x1) == yy
    assert ~(x0 & v1) == yy

    # Invalid rhs
    with pytest.raises(TypeError):
        v0.nor(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        v0.nor("1b0")


def test_vec_and():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b--0X_-10X_000X_XXXX"
    v0 = vec(x0)
    v1 = vec(x1)

    assert v0.and_(x1) == yy
    assert v0.and_(v1) == yy
    assert v0 & x1 == yy
    assert x0 & v1 == yy

    # Invalid rhs
    with pytest.raises(TypeError):
        v0.nor(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        v0.nor("1b0")


def test_vec_xnor():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b---X_-10X_-01X_XXXX"
    v0 = vec(x0)
    v1 = vec(x1)

    assert v0.xnor(x1) == yy
    assert v0.xnor(v1) == yy
    assert ~(v0 ^ x1) == yy
    assert ~(x0 ^ v1) == yy

    # Invalid rhs
    with pytest.raises(TypeError):
        v0.nor(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        v0.nor("1b0")


def test_vec_xor():
    x0 = "16b----_1111_0000_XXXX"
    x1 = "16b-10X_-10X_-10X_-10X"
    yy = "16b---X_-01X_-10X_XXXX"
    v0 = vec(x0)
    v1 = vec(x1)

    assert v0.xor(x1) == yy
    assert v0.xor(v1) == yy
    assert v0 ^ x1 == yy
    assert x0 ^ v1 == yy

    # Invalid rhs
    with pytest.raises(TypeError):
        v0.nor(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        v0.nor("1b0")


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
        assert vec(lit).uor() == Vec[1](d0, d1)


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
        assert vec(lit).uand() == Vec[1](d0, d1)


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
        assert vec(lit).uxnor() == Vec[1](d0, d1)


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
        assert vec(lit).uxor() == Vec[1](d0, d1)


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
        assert vec(a).eq(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").eq(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").eq("2b00")


NEQ = [
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


def test_vec_neq():
    for a, b, y in NEQ:
        assert vec(a).neq(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").neq(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").neq("2b00")


LTU = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", T),
    ("1b1", "1b0", F),
    ("1b1", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", X),
]


def test_vec_ltu():
    for a, b, y in LTU:
        assert vec(a).ltu(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").ltu(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").ltu("2b00")


LTEU = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", T),
    ("1b1", "1b0", F),
    ("1b1", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", X),
]


def test_vec_lteu():
    for a, b, y in LTEU:
        assert vec(a).lteu(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").lteu(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").lteu("2b00")


LT = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", F),
    ("1b1", "1b0", T),
    ("1b1", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", X),
]


def test_vec_lt():
    for a, b, y in LT:
        assert vec(a).lt(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").lt(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").lt("2b00")


LTE = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", F),
    ("1b1", "1b0", T),
    ("1b1", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", X),
]


def test_vec_lte():
    for a, b, y in LTE:
        assert vec(a).lte(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").lte(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").lte("2b00")


GTU = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", F),
    ("1b1", "1b0", T),
    ("1b1", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", X),
]


def test_vec_gtu():
    for a, b, y in GTU:
        assert vec(a).gtu(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").gtu(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").gtu("2b00")


GTEU = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", F),
    ("1b1", "1b0", T),
    ("1b1", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", X),
]


def test_vec_gteu():
    for a, b, y in GTEU:
        assert vec(a).gteu(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").gteu(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").gteu("2b00")


GT = [
    ("1b0", "1b0", F),
    ("1b0", "1b1", T),
    ("1b1", "1b0", F),
    ("1b1", "1b1", F),
    ("1b0", "1bX", X),
    ("1b0", "1b-", X),
]


def test_vec_gt():
    for a, b, y in GT:
        assert vec(a).gt(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").gt(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").gt("2b00")


GTE = [
    ("1b0", "1b0", T),
    ("1b0", "1b1", T),
    ("1b1", "1b0", F),
    ("1b1", "1b1", T),
    ("1b0", "1bX", X),
    ("1b0", "1b-", X),
]


def test_vec_gte():
    for a, b, y in GTE:
        assert vec(a).gte(b) == y

    # Invalid rhs
    with pytest.raises(TypeError):
        vec("1b0").gte(1.0e42)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        vec("1b0").gte("2b00")


def test_vec_zext():
    v = vec("4b1010")
    with pytest.raises(ValueError):
        v.zext(-1)
    assert v.zext(0) is v
    assert v.zext(4) == vec("8b0000_1010")


def test_vec_sext():
    v1 = vec("4b1010")
    v2 = vec("4b0101")
    with pytest.raises(ValueError):
        v1.sext(-1)
    assert v1.sext(0) is v1
    assert v1.sext(4) == vec("8b1111_1010")
    assert v2.sext(0) is v2
    assert v2.sext(4) == vec("8b0000_0101")


def test_vec_lsh():
    v = vec("4b1111")
    y, co = v.lsh(0)
    assert y is v and co == E
    assert v.lsh(1) == ("4b1110", "1b1")
    assert v.lsh(2) == ("4b1100", "2b11")
    assert v << 2 == "4b1100"
    assert "4b1111" << vec("2b10") == "4b1100"
    assert v.lsh(3) == ("4b1000", "3b111")
    assert v.lsh(4) == ("4b0000", "4b1111")

    with pytest.raises(ValueError):
        v.lsh(-1)
    with pytest.raises(ValueError):
        v.lsh(5)

    assert v.lsh(2, ci=vec("2b01")) == ("4b1101", "2b11")
    with pytest.raises(ValueError):
        v.lsh(2, ci=vec("3b000"))

    assert vec("2b01").lsh(vec("1bX")) == (vec("2bXX"), E)
    assert vec("2b01").lsh(vec("1b-")) == (vec("2b--"), E)
    assert vec("2b01").lsh(vec("1b1")) == (vec("2b10"), F)


def test_vec_rsh():
    v = vec("4b1111")
    y, co = v.rsh(0)
    assert y is v and co == E
    assert v.rsh(1) == ("4b0111", "1b1")
    assert v.rsh(2) == ("4b0011", "2b11")
    assert v >> 2 == "4b0011"
    assert "4b1111" >> vec("2b10") == "4b0011"
    assert v.rsh(3) == ("4b0001", "3b111")
    assert v.rsh(4) == ("4b0000", "4b1111")

    with pytest.raises(ValueError):
        v.rsh(-1)
    with pytest.raises(ValueError):
        v.rsh(5)

    assert v.rsh(2, ci=vec("2b10")) == ("4b1011", "2b11")
    with pytest.raises(ValueError):
        v.rsh(2, ci=vec("3b000"))

    assert vec("2b01").rsh(vec("1bX")) == (vec("2bXX"), E)
    assert vec("2b01").rsh(vec("1b-")) == (vec("2b--"), E)
    assert vec("2b01").rsh(vec("1b1")) == (vec("2b00"), T)


def test_vec_arsh():
    v = vec("4b1111")
    assert v.arsh(0) == ("4b1111", E)
    assert v.arsh(1) == ("4b1111", "1b1")
    assert v.arsh(2) == ("4b1111", "2b11")
    assert v.arsh(3) == ("4b1111", "3b111")
    assert v.arsh(4) == ("4b1111", "4b1111")

    v = vec("4b0111")
    assert v.arsh(0) == ("4b0111", E)
    assert v.arsh(1) == ("4b0011", "1b1")
    assert v.arsh(2) == ("4b0001", "2b11")
    assert v.arsh(3) == ("4b0000", "3b111")
    assert v.arsh(4) == ("4b0000", "4b0111")

    with pytest.raises(ValueError):
        v.arsh(-1)
    with pytest.raises(ValueError):
        v.arsh(5)

    assert vec("2b01").arsh(vec("1bX")) == (vec("2bXX"), E)
    assert vec("2b01").arsh(vec("1b-")) == (vec("2b--"), E)
    assert vec("2b01").arsh(vec("1b1")) == (vec("2b00"), T)


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
        assert vec(a).add(b, ci) == (s, co)
        if ci == F:
            assert vec(a) + b == s
            assert a + vec(b) == s


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
        assert vec(a).sub(b) == (s, co)
        assert vec(a) - b == s
        assert a - vec(b) == s


def test_vec_neg():
    assert -vec("3b000") == "3b000"
    assert -vec("3b001") == "3b111"
    assert -vec("3b111") == "3b001"
    assert -vec("3b010") == "3b110"
    assert -vec("3b110") == "3b010"
    assert -vec("3b011") == "3b101"
    assert -vec("3b101") == "3b011"
    assert -vec("3b100") == "3b100"


def test_count():
    v = vec("8b-10X_-10X")
    assert v.count_xes() == 2
    assert v.count_zeros() == 2
    assert v.count_ones() == 2
    assert v.count_dcs() == 2
    assert v.count_unknown() == 4

    assert not vec("4b0000").onehot()
    assert vec("4b1000").onehot()
    assert vec("4b0001").onehot()
    assert not vec("4b1001").onehot()
    assert not vec("4b1101").onehot()

    assert vec("4b0000").onehot0()
    assert vec("4b1000").onehot0()
    assert not vec("4b1010").onehot0()
    assert not vec("4b1011").onehot0()

    assert not vec("4b0000").has_x()
    assert vec("4b00X0").has_x()
    assert not vec("4b0000").has_dc()
    assert vec("4b00-0").has_dc()
    assert not vec("4b0000").has_unknown()
    assert vec("4b00X0").has_unknown()
    assert vec("4b00-0").has_unknown()
