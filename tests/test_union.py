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


def test_simple():
    u = Simple("3b000")

    assert str(u.a) == "2b00"
    assert str(u.b) == "3b000"
    assert str(u.c) == "4bX000"

    assert u.size == 4
    assert u.shape is None
