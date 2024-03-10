"""Test seqlogic.enum module."""

import pytest

from seqlogic.bits import F, T, X
from seqlogic.enum import Enum

# pylint: disable = no-value-for-parameter
# pyright: reportAttributeAccessIssue=false


class Color(Enum):
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
    assert Color.GREEN[0] == T
    assert Color.GREEN[1] == F


def test_x():
    """Test X item."""
    assert Color.X[0] == X
    assert Color.X[1] == X


def test_enum_error():
    """Test various enum spec errors."""
    with pytest.raises(TypeError):
        # The literal must be a str
        class InvalidType(Enum):
            FOO = 42

        _ = InvalidType()  # pyright: ignore[reportCallIssue]

    with pytest.raises(ValueError):
        _ = Color("2b11")
