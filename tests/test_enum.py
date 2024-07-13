"""Test seqlogic.bits.Enum class."""

# PyLint is confused by Enum metaclass
# pylint: disable=invalid-unary-operand-type

# For error testing
# pylint: disable=unused-variable

import pytest

from seqlogic import Enum, Vector


class Color(Enum):
    """Boilerplate bit array enum."""

    RED = "2b00"
    GREEN = "2b01"
    BLUE = "2b10"


def test_empty():
    # Empty Enum is not supported
    with pytest.raises(ValueError):

        class EmptyEnum(Enum):
            pass


def test_basic():
    assert len(Color.RED) == 2
    assert Color.RED.name == "RED"
    assert Color.RED.data[1] == 0b00
    assert str(Color.RED) == "2b00"

    assert Color("2b00") is Color.RED
    assert Color(Vector[2](0b11, 0b00)) is Color.RED

    assert len(Color.GREEN) == 2
    assert Color.GREEN.name == "GREEN"
    assert Color.GREEN.data[1] == 0b01
    assert str(Color.GREEN) == "2b01"
    assert Color("2b01") is Color.GREEN
    assert Color(Vector[2](0b10, 0b01)) is Color.GREEN

    assert len(Color.BLUE) == 2
    assert Color.BLUE.name == "BLUE"
    assert Color.BLUE.data[1] == 0b10
    assert str(Color.BLUE) == "2b10"
    assert Color("2b10") is Color.BLUE
    assert Color(Vector[2](0b01, 0b10)) is Color.BLUE

    assert len(Color.X) == 2
    assert Color.X.name == "X"
    assert Color.X.data == (0, 0)
    assert str(Color.X) == "2bXX"
    assert Color("2bXX") is Color.X
    assert Color.xes() is Color.X
    assert Color(Vector[2](0, 0)) is Color.X

    assert len(Color.DC) == 2
    assert Color.DC.name == "DC"
    assert Color.DC.data == (0b11, 0b11)
    assert str(Color.DC) == "2b--"
    assert Color("2b--") is Color.DC
    assert Color.dcs() is Color.DC
    assert Color(Vector[2](0b11, 0b11)) is Color.DC

    assert str(Color("2b11").name) == "Color(2b11)"
    assert str(Color(Vector[2](0b00, 0b11)).name) == "Color(2b11)"

    # with pytest.raises(TypeError):
    #    _ = Color(1.0e42)


def test_typing():
    """Advanced type behavior."""
    assert ~Color.GREEN is Color.BLUE
    assert ~Color.BLUE is Color.GREEN

    assert isinstance(~Color.RED, Color)
    assert (~Color.RED).name == "Color(2b11)"

    assert (Color.GREEN << 1) is Color.BLUE
    assert (Color.BLUE >> 1) is Color.GREEN


def test_slicing():
    assert Color.GREEN[0] == "1b1"
    assert Color.GREEN[1] == "1b0"


def test_enum_error():
    """Test enum spec errors."""
    with pytest.raises(ValueError):

        class InvalidName(Enum):
            X = "4bXXXX"

        _ = InvalidName()

    with pytest.raises(ValueError):

        class InvalidData(Enum):
            FOO = "4bXXXX"

        _ = InvalidData()

    # The literal must be a str
    with pytest.raises(TypeError):

        class InvalidType(Enum):
            FOO = 42

        _ = InvalidType()

    with pytest.raises(ValueError):

        class InvalidMembers(Enum):
            A = "2b00"
            B = "3b000"

        _ = InvalidMembers()

    with pytest.raises(ValueError):

        class DuplicateMembers(Enum):
            A = "2b00"
            B = "2b01"
            C = "2b00"
