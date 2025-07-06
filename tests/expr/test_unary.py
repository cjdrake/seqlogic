"""Test symbolic unary expressions"""

import pytest
from bvwx import bits

from seqlogic import Uand, Uor, Uxor, Variable

a = Variable(name="a")


def test_uor():
    y = Uor(False)
    assert str(y) == 'uor("1b0")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = Uor("4b1010")
    assert str(y) == 'uor("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "1b1"

    y = Uor(bits("4b1010"))
    assert str(y) == 'uor("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "1b1"

    y = Uor(a)
    assert str(y) == "uor(a)"

    with pytest.raises(TypeError):
        Uor(4.2)


def test_uand():
    y = Uand(False)
    assert str(y) == 'uand("1b0")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = Uand("4b1010")
    assert str(y) == 'uand("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = Uand(bits("4b1010"))
    assert str(y) == 'uand("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = Uand(a)
    assert str(y) == "uand(a)"

    with pytest.raises(TypeError):
        Uand(4.2)


def test_uxor():
    y = Uxor(False)
    assert str(y) == 'uxor("1b0")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = Uxor("4b1010")
    assert str(y) == 'uxor("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = Uxor(bits("4b1010"))
    assert str(y) == 'uxor("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = Uxor(a)
    assert str(y) == "uxor(a)"

    with pytest.raises(TypeError):
        Uxor(4.2)
