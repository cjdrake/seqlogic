"""Test seqlogic.enum module."""

import pytest

from seqlogic.lbool import VecEnum, ones, xes, zeros

# pylint: disable = no-value-for-parameter
# pyright: reportAttributeAccessIssue=false


class Color(VecEnum):
    """Boilerplate bit array enum."""

    RED = "2b00"
    GREEN = "2b01"
    BLUE = "2b10"


def test_basic():
    """Test basic usage model."""
    assert str(Color.RED) == "RED"
    assert str(Color.GREEN) == "GREEN"
    assert str(Color.BLUE) == "BLUE"

    assert Color("2b00") is Color.RED
    assert Color("2b01") is Color.GREEN
    assert Color("2b10") is Color.BLUE


def test_bits():
    """Test behaviors of bits subclass."""
    assert Color.GREEN[0] == ones(1)
    assert Color.GREEN[1] == zeros(1)


def test_x():
    """Test X item."""
    assert Color.X == xes(2)


def test_enum_error():
    """Test various enum spec errors."""
    with pytest.raises(TypeError):
        # The literal must be a str
        class InvalidType(VecEnum):
            FOO = 42

        _ = InvalidType()  # pyright: ignore[reportCallIssue]

    with pytest.raises(ValueError):
        _ = Color("2b11")
