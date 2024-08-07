"""Test seqlogic.bits.Union class."""

# For error testing
# pylint: disable=unused-variable

import pytest

from seqlogic import Union, Vector


def test_empty():
    with pytest.raises(ValueError):

        class EmptyUnion(Union):
            pass


class Simple(Union):
    a: Vector[2]
    b: Vector[3]
    c: Vector[4]


S1 = """\
Simple(
    a=2b00,
    b=3b000,
    c=4bX000,
)"""

R1 = """\
Simple(
    a=bits("2b00"),
    b=bits("3b000"),
    c=bits("4bX000"),
)"""


def test_simple():
    u = Simple("3b000")

    assert str(u.a) == "2b00"
    assert str(u.b) == "3b000"
    assert str(u.c) == "4bX000"

    assert str(u) == S1
    assert repr(u) == R1

    assert u.size == 4
