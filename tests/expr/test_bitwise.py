"""Test symbolic bitwise expressions"""

import pytest
from bvwx import Scalar, bits

from seqlogic import ITE, And, Impl, Mux, Not, Or, Xor
from seqlogic.expr import BitsConst, Variable

a = Variable(name="a")
b = Variable(name="b")
c = Variable(name="c")


def test_misc():
    y = Or("2b00", "2b01", "2b10", "2b11")
    x = y.xs[1]
    assert isinstance(x, BitsConst)
    assert x.value == "2b01"


def test_not():
    y = Not(False)
    assert str(y) == 'not_("1b0")'
    f, xs = y.to_func()
    assert f(*xs) == "1b1"

    y = Not("4b1010")
    assert str(y) == 'not_("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0101"

    y = Not(bits("4b1010"))
    assert str(y) == 'not_("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0101"

    y = Not(a)
    assert str(y) == "not_(a)"

    y = ~a
    assert str(y) == "not_(a)"

    with pytest.raises(TypeError):
        Not(4.2)  # pyright: ignore[reportArgumentType]


def test_or():
    y = Or("4b1010", "4b1100")
    assert str(y) == 'or_("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "4b1110"

    y = Or("2b10", 1)
    assert str(y) == 'or_("2b10", 1)'

    y = Or("2b10", bits("2b01"))
    assert str(y) == 'or_("2b10", "2b01")'

    y = a | "4b1010"
    assert str(y) == 'or_(a, "4b1010")'

    y = "4b1010" | a
    assert str(y) == 'or_("4b1010", a)'

    with pytest.raises(TypeError):
        Or(4.2)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        Or("2b00", 4.2)  # pyright: ignore[reportArgumentType]


def test_and():
    y = And("4b1010", "4b1100")
    assert str(y) == 'and_("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "4b1000"

    y = And("2b10", 1)
    assert str(y) == 'and_("2b10", 1)'

    y = And("2b10", bits("2b01"))
    assert str(y) == 'and_("2b10", "2b01")'

    y = a & "4b1010"
    assert str(y) == 'and_(a, "4b1010")'

    y = "4b1010" & a
    assert str(y) == 'and_("4b1010", a)'

    with pytest.raises(TypeError):
        And(4.2)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        And("2b00", 4.2)  # pyright: ignore[reportArgumentType]


def test_xor():
    y = Xor("4b1010", "4b1100")
    assert str(y) == 'xor("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0110"

    y = Xor("2b10", 1)
    assert str(y) == 'xor("2b10", 1)'

    y = Xor("2b10", bits("2b01"))
    assert str(y) == 'xor("2b10", "2b01")'

    y = a ^ "4b1010"
    assert str(y) == 'xor(a, "4b1010")'

    y = "4b1010" ^ a
    assert str(y) == 'xor("4b1010", a)'

    with pytest.raises(TypeError):
        Xor(4.2)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        Xor("2b00", 4.2)  # pyright: ignore[reportArgumentType]


def test_impl():
    y = Impl("4b1010", "4b1100")
    assert str(y) == 'impl("4b1010", "4b1100")'
    f, xs = y.to_func()
    assert f(*xs) == "4b1101"

    y = Impl("2b10", 1)
    assert str(y) == 'impl("2b10", 1)'

    y = Impl("2b10", bits("2b01"))
    assert str(y) == 'impl("2b10", "2b01")'

    with pytest.raises(TypeError):
        Impl(4.2, 6.9)  # pyright: ignore[reportArgumentType]


def test_ite():
    y = ITE(False, "4b1010", "4b0101")
    assert str(y) == 'ite("1b0", "4b1010", "4b0101")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0101"

    s = bits("1b1")
    assert isinstance(s, Scalar)  # helping the type checker
    y = ITE(s, "4b1010", "4b0101")
    assert str(y) == 'ite("1b1", "4b1010", "4b0101")'

    y = ITE(a, "4b1010", "4b0101")
    assert str(y) == 'ite(a, "4b1010", "4b0101")'

    with pytest.raises(TypeError):
        ITE(4.2, 6.9, 9.6)  # pyright: ignore[reportArgumentType]


def test_mux():
    y = Mux(False, x0="4b1010", x1="4b0101")
    assert str(y) == 'mux("1b0", x0="4b1010", x1="4b0101")'
    f, xs = y.to_func()
    assert f(*xs) == "4b1010"

    y = Mux(a, x0=b, x1=c)
    f, xs = y.to_func()
    assert str(y) == "mux(a, x0=b, x1=c)"
    assert xs == [a, b, c]
