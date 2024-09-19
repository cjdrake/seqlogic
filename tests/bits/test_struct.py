"""Test seqlogic.bits.Struct class."""

# For error testing
# pylint: disable=unused-variable

import pytest

from seqlogic import Struct, Vector


def test_empty():
    with pytest.raises(ValueError):

        class EmptyStruct(Struct):
            pass


S1 = """\
Simple(
    a=2b00,
    b=3b000,
    c=4b0000,
)"""

R1 = """\
Simple(
    a=bits("2b00"),
    b=bits("3b000"),
    c=bits("4b0000"),
)"""


class Simple(Struct):
    a: Vector[2]
    b: Vector[3]
    c: Vector[4]


def test_simple():
    s = Simple(a="2b00", b="3b000", c="4b0000")

    assert str(s.a) == "2b00"
    assert str(s.b) == "3b000"
    assert str(s.c) == "4b0000"

    assert str(s) == S1
    assert repr(s) == R1


def test_init():
    s = Simple()
    assert str(s) == "Simple(\n    a=2bXX,\n    b=3bXXX,\n    c=4bXXXX,\n)"
    s = Simple(a="2b11")
    assert str(s) == "Simple(\n    a=2b11,\n    b=3bXXX,\n    c=4bXXXX,\n)"
    s = Simple(b="3b111")
    assert str(s) == "Simple(\n    a=2bXX,\n    b=3b111,\n    c=4bXXXX,\n)"
    s = Simple(c="4b1111")
    assert str(s) == "Simple(\n    a=2bXX,\n    b=3bXXX,\n    c=4b1111,\n)"

    assert str(Simple.xes()) == "Simple(\n    a=2bXX,\n    b=3bXXX,\n    c=4bXXXX,\n)"
    assert str(Simple.dcs()) == "Simple(\n    a=2b--,\n    b=3b---,\n    c=4b----,\n)"
