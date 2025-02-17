"""Test symbolic predicate expressions"""

import pytest
from bvwx import bits

from seqlogic import EQ, GE, GT, LE, LT, NE
from seqlogic.expr import Variable

a = Variable(name="a")
b = Variable(name="b")


def test_eq():
    y = EQ("4b1010", "4b1100")
    assert str(y) == 'eq("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = EQ("2b10", 1)
    assert str(y) == 'eq("2b10", 1)'

    y = EQ("2b10", bits("2b01"))
    assert str(y) == 'eq("2b10", "2b01")'

    with pytest.raises(TypeError):
        EQ(4.2, 6.9)  # pyright: ignore[reportArgumentType]


def test_ne():
    y = NE("4b1010", "4b1100")
    assert str(y) == 'ne("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "1b1"

    y = NE("2b10", 1)
    assert str(y) == 'ne("2b10", 1)'

    y = NE("2b10", bits("2b01"))
    assert str(y) == 'ne("2b10", "2b01")'

    with pytest.raises(TypeError):
        NE(4.2, 6.9)  # pyright: ignore[reportArgumentType]


def test_gt():
    y = GT("4b1010", "4b1100")
    assert str(y) == 'gt("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = GT("2b10", 1)
    assert str(y) == 'gt("2b10", 1)'

    y = GT("2b10", bits("2b01"))
    assert str(y) == 'gt("2b10", "2b01")'

    with pytest.raises(TypeError):
        GT(4.2, 6.9)  # pyright: ignore[reportArgumentType]


def test_ge():
    y = GE("4b1010", "4b1100")
    assert str(y) == 'ge("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = GE("2b10", 1)
    assert str(y) == 'ge("2b10", 1)'

    y = GE("2b10", bits("2b01"))
    assert str(y) == 'ge("2b10", "2b01")'

    with pytest.raises(TypeError):
        GE(4.2, 6.9)  # pyright: ignore[reportArgumentType]


def test_lt():
    y = LT("4b1010", "4b1100")
    assert str(y) == 'lt("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "1b1"

    y = LT("2b10", 1)
    assert str(y) == 'lt("2b10", 1)'

    y = LT("2b10", bits("2b01"))
    assert str(y) == 'lt("2b10", "2b01")'

    with pytest.raises(TypeError):
        LT(4.2, 6.9)  # pyright: ignore[reportArgumentType]


def test_le():
    y = LE("4b1010", "4b1100")
    assert str(y) == 'le("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "1b1"

    y = LE("2b10", 1)
    assert str(y) == 'le("2b10", 1)'

    y = LE("2b10", bits("2b01"))
    assert str(y) == 'le("2b10", "2b01")'

    with pytest.raises(TypeError):
        LE(4.2, 6.9)  # pyright: ignore[reportArgumentType]
