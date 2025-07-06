"""Test symbolic arithmetic expressions"""

import pytest
from bvwx import bits

from seqlogic import Adc, Add, Div, Lsh, Mod, Mul, Neg, Ngc, Rsh, Sbc, Srsh, Sub, Variable

a = Variable(name="a")


def test_add():
    y = Add("4b1010", "4b0101")
    assert str(y) == 'add("4b1010", "4b0101")'
    f, xs = y.to_func()
    assert f(*xs) == "4b1111"

    y = Add("4b1010", "4b0101", "1b1")
    assert str(y) == 'add("4b1010", "4b0101", "1b1")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0000"


def test_adc():
    y = Adc("4b1010", "4b0101")
    assert str(y) == 'adc("4b1010", "4b0101")'
    f, xs = y.to_func()
    assert f(*xs) == "5b0_1111"

    y = Adc("4b1010", "4b0101", "1b1")
    assert str(y) == 'adc("4b1010", "4b0101", "1b1")'
    f, xs = y.to_func()
    assert f(*xs) == "5b1_0000"


def test_sub():
    y = Sub("4b1010", "4b0101")
    assert str(y) == 'sub("4b1010", "4b0101")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0101"


def test_sbc():
    y = Sbc("4b1010", "4b0101")
    assert str(y) == 'sbc("4b1010", "4b0101")'
    f, xs = y.to_func()
    assert f(*xs) == "5b1_0101"


def test_neg():
    y = Neg(False)
    assert str(y) == 'neg("1b0")'
    f, xs = y.to_func()
    assert f(*xs) == "1b0"

    y = Neg("4b1010")
    assert str(y) == 'neg("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0110"

    y = Neg(bits("4b1010"))
    assert str(y) == 'neg("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0110"

    y = Neg(a)
    assert str(y) == "neg(a)"

    with pytest.raises(TypeError):
        Neg(4.2)


def test_ngc():
    y = Ngc(False)
    assert str(y) == 'ngc("1b0")'
    f, xs = y.to_func()
    assert f(*xs) == "2b10"

    y = Ngc("4b1010")
    assert str(y) == 'ngc("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "5b0_0110"

    y = Ngc(bits("4b1010"))
    assert str(y) == 'ngc("4b1010")'
    f, xs = y.to_func()
    assert f(*xs) == "5b0_0110"

    y = Ngc(a)
    assert str(y) == "ngc(a)"

    with pytest.raises(TypeError):
        Ngc(4.2)


def test_mul():
    y = Mul("4b0110", "4b0111")
    assert str(y) == 'mul("4b0110", "4b0111")'
    f, xs = y.to_func()
    assert f(*xs) == "8b0010_1010"


def test_div():
    y = Div("8b0010_1010", "4b0111")
    assert str(y) == 'div("8b0010_1010", "4b0111")'
    f, xs = y.to_func()
    assert f(*xs) == "8b0000_0110"


def test_mod():
    y = Mod("8b0010_1010", "4b0111")
    assert str(y) == 'mod("8b0010_1010", "4b0111")'
    f, xs = y.to_func()
    assert f(*xs) == "4b0000"


def test_lsh():
    y = Lsh("4b1010", 2)
    assert str(y) == 'lsh("4b1010", 2)'
    f, xs = y.to_func()
    assert f(*xs) == "4b1000"

    y = a << 2
    assert str(y) == "lsh(a, 2)"
    y = "4b1010" << a
    assert str(y) == 'lsh("4b1010", a)'

    y = Lsh("4b1010", "2b10")
    assert str(y) == 'lsh("4b1010", "2b10")'
    y = Lsh("4b1010", bits("2b10"))
    assert str(y) == 'lsh("4b1010", "2b10")'
    y = Lsh("4b1010", a)
    assert str(y) == 'lsh("4b1010", a)'

    with pytest.raises(TypeError):
        Lsh("4b1010", 4.2)


def test_rsh():
    y = Rsh("4b1010", 2)
    assert str(y) == 'rsh("4b1010", 2)'
    f, xs = y.to_func()
    assert f(*xs) == "4b0010"

    y = a >> 2
    assert str(y) == "rsh(a, 2)"
    y = "4b1010" >> a
    assert str(y) == 'rsh("4b1010", a)'


def test_srsh():
    y = Srsh("4b1010", 2)
    assert str(y) == 'srsh("4b1010", 2)'
    f, xs = y.to_func()
    assert f(*xs) == "4b1110"
