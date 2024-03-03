"""Test seqlogic.lbool module."""

# pylint: disable = pointless-statement

import pytest

from seqlogic import lbool
from seqlogic.lbool import (
    int2vec,
    land,
    limplies,
    lnand,
    lnor,
    lnot,
    lor,
    lxnor,
    lxor,
    uint2vec,
    vec,
)

LNOT = {
    "?": "?",
    "0": "1",
    "1": "0",
    "X": "X",
}


def test_lnot():
    """Test seqlogic.lbool.lnot function."""
    for x, y in LNOT.items():
        x = lbool.from_char[x]
        y = lbool.from_char[y]
        assert lnot(x) == y
        assert ~vec(1, x) == vec(1, y)


LNOR = {
    "??": "?",
    "?0": "?",
    "?1": "?",
    "?X": "?",
    "0?": "?",
    "00": "1",
    "01": "0",
    "0X": "X",
    "1?": "?",
    "10": "0",
    "11": "0",
    "1X": "0",
    "X?": "?",
    "X0": "X",
    "X1": "0",
    "XX": "X",
}


def test_lnor():
    """Test seqlogic.lbool.lnor function."""
    for xs, y in LNOR.items():
        x0 = lbool.from_char[xs[0]]
        x1 = lbool.from_char[xs[1]]
        y = lbool.from_char[y]
        assert lnor(x0, x1) == y
        assert ~(vec(1, x0) | vec(1, x1)) == vec(1, y)


LOR = {
    "??": "?",
    "?0": "?",
    "?1": "?",
    "?X": "?",
    "0?": "?",
    "00": "0",
    "01": "1",
    "0X": "X",
    "1?": "?",
    "10": "1",
    "11": "1",
    "1X": "1",
    "X?": "?",
    "X0": "X",
    "X1": "1",
    "XX": "X",
}


def test_lor():
    """Test seqlogic.lbool.lor function."""
    for xs, y in LOR.items():
        x0 = lbool.from_char[xs[0]]
        x1 = lbool.from_char[xs[1]]
        y = lbool.from_char[y]
        assert lor(x0, x1) == y
        assert vec(1, x0) | vec(1, x1) == vec(1, y)


LNAND = {
    "??": "?",
    "?0": "?",
    "?1": "?",
    "?X": "?",
    "0?": "?",
    "00": "1",
    "01": "1",
    "0X": "1",
    "1?": "?",
    "10": "1",
    "11": "0",
    "1X": "X",
    "X?": "?",
    "X0": "1",
    "X1": "X",
    "XX": "X",
}


def test_lnand():
    """Test seqlogic.lbool.lnand function."""
    for xs, y in LNAND.items():
        x0 = lbool.from_char[xs[0]]
        x1 = lbool.from_char[xs[1]]
        y = lbool.from_char[y]
        assert lnand(x0, x1) == y
        assert ~(vec(1, x0) & vec(1, x1)) == vec(1, y)


LAND = {
    "??": "?",
    "?0": "?",
    "?1": "?",
    "?X": "?",
    "0?": "?",
    "00": "0",
    "01": "0",
    "0X": "0",
    "1?": "?",
    "10": "0",
    "11": "1",
    "1X": "X",
    "X?": "?",
    "X0": "0",
    "X1": "X",
    "XX": "X",
}


def test_land():
    """Test seqlogic.lbool.land function."""
    for xs, y in LAND.items():
        x0 = lbool.from_char[xs[0]]
        x1 = lbool.from_char[xs[1]]
        y = lbool.from_char[y]
        assert land(x0, x1) == y
        assert vec(1, x0) & vec(1, x1) == vec(1, y)


LXNOR = {
    "??": "?",
    "?0": "?",
    "?1": "?",
    "?X": "?",
    "0?": "?",
    "00": "1",
    "01": "0",
    "0X": "X",
    "1?": "?",
    "10": "0",
    "11": "1",
    "1X": "X",
    "X?": "?",
    "X0": "X",
    "X1": "X",
    "XX": "X",
}


def test_lxnor():
    """Test seqlogic.lbool.lnand function."""
    for xs, y in LXNOR.items():
        x0 = lbool.from_char[xs[0]]
        x1 = lbool.from_char[xs[1]]
        y = lbool.from_char[y]
        assert lxnor(x0, x1) == y
        assert ~(vec(1, x0) ^ vec(1, x1)) == vec(1, y)


LXOR = {
    "??": "?",
    "?0": "?",
    "?1": "?",
    "?X": "?",
    "0?": "?",
    "00": "0",
    "01": "1",
    "0X": "X",
    "1?": "?",
    "10": "1",
    "11": "0",
    "1X": "X",
    "X?": "?",
    "X0": "X",
    "X1": "X",
    "XX": "X",
}


def test_lxor():
    """Test seqlogic.lbool.land function."""
    for xs, y in LXOR.items():
        x0 = lbool.from_char[xs[0]]
        x1 = lbool.from_char[xs[1]]
        y = lbool.from_char[y]
        assert lxor(x0, x1) == y
        assert vec(1, x0) ^ vec(1, x1) == vec(1, y)


LIMPLIES = {
    "??": "?",
    "?0": "?",
    "?1": "?",
    "?X": "?",
    "0?": "?",
    "00": "1",
    "01": "1",
    "0X": "1",
    "1?": "?",
    "10": "0",
    "11": "1",
    "1X": "X",
    "X?": "?",
    "X0": "X",
    "X1": "1",
    "XX": "X",
}


def test_limplies():
    """Test seqlogic.lbool.limplies function."""
    for xs, y in LIMPLIES.items():
        x0 = lbool.from_char[xs[0]]
        x1 = lbool.from_char[xs[1]]
        y = lbool.from_char[y]
        assert limplies(x0, x1) == y


UINT2VEC_VALS = {
    0: "",
    1: "1",
    2: "01",
    3: "11",
    4: "001",
    5: "101",
    6: "011",
    7: "111",
    8: "0001",
}

UINT2VEC_N_VALS = {
    (0, 0): "",
    (0, 1): "0",
    (0, 2): "00",
    (1, 1): "1",
    (1, 2): "10",
    (1, 4): "1000",
    (2, 2): "01",
    (2, 3): "010",
    (2, 4): "0100",
    (3, 2): "11",
    (3, 3): "110",
    (3, 4): "1100",
    (4, 3): "001",
    (4, 4): "0010",
    (4, 5): "00100",
}


def test_uint2vec():
    """Test seqlogic.lbool.uint2vec function."""
    # Negative inputs are invalid
    with pytest.raises(ValueError):
        uint2vec(-1)

    for i, s in UINT2VEC_VALS.items():
        v = uint2vec(i)
        assert str(v) == s
        assert v.to_uint() == i

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

    for (i, n), s in UINT2VEC_N_VALS.items():
        v = uint2vec(i, n)
        assert str(v) == s
        assert v.to_uint() == i


INT2VEC_VALS = {
    -8: "0001",
    -7: "1001",
    -6: "0101",
    -5: "1101",
    -4: "001",
    -3: "101",
    -2: "01",
    -1: "1",
    0: "0",
    1: "10",
    2: "010",
    3: "110",
    4: "0010",
    5: "1010",
    6: "0110",
    7: "1110",
    8: "00010",
}

INT2VEC_N_VALS = {
    (-5, 4): "1101",
    (-5, 5): "11011",
    (-5, 6): "110111",
    (-4, 3): "001",
    (-4, 4): "0011",
    (-4, 5): "00111",
    (-3, 3): "101",
    (-3, 4): "1011",
    (-3, 5): "10111",
    (-2, 2): "01",
    (-2, 3): "011",
    (-2, 4): "0111",
    (-1, 1): "1",
    (-1, 2): "11",
    (-1, 3): "111",
    (0, 1): "0",
    (0, 2): "00",
    (0, 3): "000",
    (1, 2): "10",
    (1, 3): "100",
    (1, 4): "1000",
    (2, 3): "010",
    (2, 4): "0100",
    (2, 5): "01000",
    (3, 3): "110",
    (3, 4): "1100",
    (3, 5): "11000",
    (4, 4): "0010",
    (4, 5): "00100",
    (4, 6): "001000",
}


def test_int2vec():
    """Test seqlogic.lbool.int2vec function."""
    for i, s in INT2VEC_VALS.items():
        v = int2vec(i)
        assert str(v) == s
        assert v.to_int() == i

    for (i, n), s in INT2VEC_N_VALS.items():
        v = int2vec(i, n)
        assert str(v) == s
        assert v.to_int() == i

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


def test_vec_basic():
    """Test seqlogic.lbool.vec basic functionality."""
    # n is non-negative
    with pytest.raises(ValueError):
        vec(-1, 42)

    # data in [0, 2**nbits)
    with pytest.raises(ValueError):
        vec(4, -1)
    with pytest.raises(ValueError):
        vec(4, 2 ** (2 * 4))

    v = vec(4, 0b11_10_01_00)
    assert len(v) == 4

    assert v[3] == vec(1, 0b11)
    assert v[2] == vec(1, 0b10)
    assert v[1] == vec(1, 0b01)
    assert v[0] == vec(1, 0b00)

    assert v[0:1] == vec(1, 0b00)
    assert v[0:2] == vec(2, 0b01_00)
    assert v[0:3] == vec(3, 0b10_01_00)
    assert v[0:4] == vec(4, 0b11_10_01_00)
    assert v[1:2] == vec(1, 0b01)
    assert v[1:3] == vec(2, 0b10_01)
    assert v[1:4] == vec(3, 0b11_10_01)
    assert v[2:3] == vec(1, 0b10)
    assert v[2:4] == vec(2, 0b11_10)
    assert v[3:4] == vec(1, 0b11)

    with pytest.raises(TypeError):
        v["invalid"]  # pyright: ignore[reportArgumentType]

    assert list(v) == [vec(1, 0b00), vec(1, 0b01), vec(1, 0b10), vec(1, 0b11)]
