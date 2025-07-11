"""Test symbolic logical expressions"""

import pytest

from seqlogic import Land, Lor, Lxor, Variable

a = Variable(name="a")


def test_or():
    y = Lor(False, True)
    assert str(y) == 'lor("1b0", "1b1")'
    f, xs = y.to_func()
    assert f(*xs) == "1b1"

    with pytest.raises(TypeError):
        Lor(4.2)


def test_and():
    y = Land(False, True)
    assert str(y) == 'land("1b0", "1b1")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    with pytest.raises(TypeError):
        Land(4.2)


def test_xor():
    y = Lxor(False, True)
    assert str(y) == 'lxor("1b0", "1b1")'
    f, xs = y.to_func()
    assert f(*xs) == "1b1"

    with pytest.raises(TypeError):
        Lxor(4.2)
