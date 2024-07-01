"""Test seqlogic.vec.VecUnion class."""

# pylint: disable = unused-variable

# PyRight is confused by MetaClass behavior
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false

import pytest

from seqlogic.vec import Vec, VecUnion


def test_empty():
    with pytest.raises(ValueError):

        class EmptyUnion(VecUnion):
            pass


class Simple(VecUnion):
    a: Vec[2]
    b: Vec[3]
    c: Vec[4]


def test_simple():
    u = Simple("3b000")

    assert str(u.a) == "2b00"
    assert str(u.b) == "3b000"
    assert str(u.c) == "4bX000"

    assert len(u) == 4
