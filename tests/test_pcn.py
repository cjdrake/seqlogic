"""Test seqlogic.pcn module."""

import pytest

from seqlogic.pcn import int2vec, uint2vec

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
