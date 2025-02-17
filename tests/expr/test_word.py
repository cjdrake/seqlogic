"""Test symbolic word expressions"""

import pytest
from bvwx import bits

from seqlogic import Cat, Lrot, Rep, Rrot, Sxt, Xt
from seqlogic.expr import Variable

a = Variable(name="a")


def test_xt():
    y = Xt("4b1010", 2)
    assert str(y) == 'xt("4b1010", 2)'
    f, xs = y.to_func()
    assert f(*xs) == "6b00_1010"


def test_sxt():
    y = Sxt("4b1010", 2)
    assert str(y) == 'sxt("4b1010", 2)'
    f, xs = y.to_func()
    assert f(*xs) == "6b11_1010"


def test_lrot():
    y = Lrot("4b1001", 1)
    assert str(y) == 'lrot("4b1001", 1)'
    f, xs = y.to_func()
    assert f(*xs) == "4b0011"


def test_rrot():
    y = Rrot("4b1001", 1)
    assert str(y) == 'rrot("4b1001", 1)'
    f, xs = y.to_func()
    assert f(*xs) == "4b1100"


def test_cat():
    y = Cat(False, "1b1")
    assert str(y) == 'cat("1b0", "1b1")'
    f, xs = y.to_func()
    assert f(*xs) == "2b10"

    y = Cat(False, "1b1", bits(False), a)
    assert str(y) == 'cat("1b0", "1b1", "1b0", a)'


def test_rep():
    y = Rep(True, 4)
    assert str(y) == 'rep("1b1", 4)'
    f, xs = y.to_func()
    assert f(*xs) == "4b1111"

    with pytest.raises(TypeError):
        Rep(False, 4.2)  # pyright: ignore[reportArgumentType]
