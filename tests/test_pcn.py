"""Test seqlogic.pcn module."""

import pytest

from seqlogic.pcn import (
    from_char,
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
)

LNOT = {
    "X": "X",
    "0": "1",
    "1": "0",
    "x": "x",
}


def test_lnot():
    """Test seqlogic.pcn.lnot function."""
    for x, y in LNOT.items():
        x = from_char[x]
        y = from_char[y]
        assert lnot(x) == y


LNOR = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "Xx": "X",
    "0X": "X",
    "00": "1",
    "01": "0",
    "0x": "x",
    "1X": "X",
    "10": "0",
    "11": "0",
    "1x": "0",
    "xX": "X",
    "x0": "x",
    "x1": "0",
    "xx": "x",
}


def test_lnor():
    """Test seqlogic.pcn.lnor function."""
    for xs, y in LNOR.items():
        x0 = from_char[xs[0]]
        x1 = from_char[xs[1]]
        y = from_char[y]
        assert lnor(x0, x1) == y


LOR = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "Xx": "X",
    "0X": "X",
    "00": "0",
    "01": "1",
    "0x": "x",
    "1X": "X",
    "10": "1",
    "11": "1",
    "1x": "1",
    "xX": "X",
    "x0": "x",
    "x1": "1",
    "xx": "x",
}


def test_lor():
    """Test seqlogic.pcn.lor function."""
    for xs, y in LOR.items():
        x0 = from_char[xs[0]]
        x1 = from_char[xs[1]]
        y = from_char[y]
        assert lor(x0, x1) == y


LNAND = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "Xx": "X",
    "0X": "X",
    "00": "1",
    "01": "1",
    "0x": "1",
    "1X": "X",
    "10": "1",
    "11": "0",
    "1x": "x",
    "xX": "X",
    "x0": "1",
    "x1": "x",
    "xx": "x",
}


def test_lnand():
    """Test seqlogic.pcn.lnand function."""
    for xs, y in LNAND.items():
        x0 = from_char[xs[0]]
        x1 = from_char[xs[1]]
        y = from_char[y]
        assert lnand(x0, x1) == y


LAND = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "Xx": "X",
    "0X": "X",
    "00": "0",
    "01": "0",
    "0x": "0",
    "1X": "X",
    "10": "0",
    "11": "1",
    "1x": "x",
    "xX": "X",
    "x0": "0",
    "x1": "x",
    "xx": "x",
}


def test_land():
    """Test seqlogic.pcn.land function."""
    for xs, y in LAND.items():
        x0 = from_char[xs[0]]
        x1 = from_char[xs[1]]
        y = from_char[y]
        assert land(x0, x1) == y


LXNOR = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "Xx": "X",
    "0X": "X",
    "00": "1",
    "01": "0",
    "0x": "x",
    "1X": "X",
    "10": "0",
    "11": "1",
    "1x": "x",
    "xX": "X",
    "x0": "x",
    "x1": "x",
    "xx": "x",
}


def test_lxnor():
    """Test seqlogic.pcn.lnand function."""
    for xs, y in LXNOR.items():
        x0 = from_char[xs[0]]
        x1 = from_char[xs[1]]
        y = from_char[y]
        assert lxnor(x0, x1) == y


LXOR = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "Xx": "X",
    "0X": "X",
    "00": "0",
    "01": "1",
    "0x": "x",
    "1X": "X",
    "10": "1",
    "11": "0",
    "1x": "x",
    "xX": "X",
    "x0": "x",
    "x1": "x",
    "xx": "x",
}


def test_lxor():
    """Test seqlogic.pcn.land function."""
    for xs, y in LXOR.items():
        x0 = from_char[xs[0]]
        x1 = from_char[xs[1]]
        y = from_char[y]
        assert lxor(x0, x1) == y


LIMPLIES = {
    "XX": "X",
    "X0": "X",
    "X1": "X",
    "Xx": "X",
    "0X": "X",
    "00": "1",
    "01": "1",
    "0x": "1",
    "1X": "X",
    "10": "0",
    "11": "1",
    "1x": "x",
    "xX": "X",
    "x0": "x",
    "x1": "1",
    "xx": "x",
}


def test_limplies():
    """Test seqlogic.pcn.limplies function."""
    for xs, y in LIMPLIES.items():
        x0 = from_char[xs[0]]
        x1 = from_char[xs[1]]
        y = from_char[y]
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
    """Test seqlogic.pcn.uint2vec function."""
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
    """Test seqlogic.pcn.int2vec function."""
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
