"""Test seqlogic.vec.VecStruct class."""

# pylint: disable = unused-variable

# PyRight is confused by MetaClass behavior
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

import pytest

from seqlogic.vec import Vec, VecStruct


def test_empty():
    with pytest.raises(ValueError):

        class EmptyStruct(VecStruct):
            pass


class Simple(VecStruct):
    a: Vec[2]
    b: Vec[3]
    c: Vec[4]


def test_simple():
    s = Simple(a="2b00", b="3b000", c="4b0000")

    assert str(s.a) == "2b00"
    assert str(s.b) == "3b000"
    assert str(s.c) == "4b0000"

    assert str(s) == "Simple(a=2b00, b=3b000, c=4b0000)"

    assert repr(s) == (
        "Simple(a=Vec[2](0b11, 0b00), b=Vec[3](0b111, 0b000), c=Vec[4](0b1111, 0b0000))"
    )

    assert len(s) == 9


def test_init():
    s = Simple()
    assert str(s) == "Simple(a=2bXX, b=3bXXX, c=4bXXXX)"
    s = Simple(a="2b11")
    assert str(s) == "Simple(a=2b11, b=3bXXX, c=4bXXXX)"
    s = Simple(b="3b111")
    assert str(s) == "Simple(a=2bXX, b=3b111, c=4bXXXX)"
    s = Simple(c="4b1111")
    assert str(s) == "Simple(a=2bXX, b=3bXXX, c=4b1111)"

    assert str(Simple.xes()) == "Simple(a=2bXX, b=3bXXX, c=4bXXXX)"
    assert str(Simple.dcs()) == "Simple(a=2b--, b=3b---, c=4b----)"
