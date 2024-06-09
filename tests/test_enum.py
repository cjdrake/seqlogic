"""Test seqlogic.lbool.VecEnum."""

# pylint: disable = unused-variable

# pyright: reportAttributeAccessIssue = false
# pyright: reportCallIssue = false

import pytest

from seqlogic.lbool import Vec, VecEnum


class Color(VecEnum):
    """Boilerplate bit array enum."""

    RED = "2b00"
    GREEN = "2b01"
    BLUE = "2b10"


def test_empty():
    with pytest.raises(ValueError):

        class Empty(VecEnum):
            pass


def test_basic():
    assert len(Color.RED) == 2
    assert Color.RED.name == "RED"
    assert Color.RED.data == 0b0101
    assert str(Color.RED) == "2b00"
    assert Color("2b00") is Color.RED
    assert Color("2b0_0") is Color.RED
    assert Color(0b0101) is Color.RED
    assert Color(Vec[2](0b0101)) is Color.RED

    assert len(Color.GREEN) == 2
    assert Color.GREEN.name == "GREEN"
    assert Color.GREEN.data == 0b0110
    assert str(Color.GREEN) == "2b01"
    assert Color("2b01") is Color.GREEN
    assert Color("2b0_1") is Color.GREEN
    assert Color(0b0110) is Color.GREEN
    assert Color(Vec[2](0b0110)) is Color.GREEN

    assert len(Color.BLUE) == 2
    assert Color.BLUE.name == "BLUE"
    assert Color.BLUE.data == 0b1001
    assert str(Color.BLUE) == "2b10"
    assert Color("2b10") is Color.BLUE
    assert Color("2b1_0") is Color.BLUE
    assert Color(0b1001) is Color.BLUE
    assert Color(Vec[2](0b1001)) is Color.BLUE

    assert len(Color.X) == 2
    assert Color.X.name == "X"
    assert Color.X.data == 0
    assert str(Color.X) == "2bXX"
    assert Color("2bXX") is Color.X
    assert Color(0b0000) is Color.X
    assert Color.xes() is Color.X
    assert Color(Vec[2](0b0000)) is Color.X

    assert len(Color.DC) == 2
    assert Color.DC.name == "DC"
    assert Color.DC.data == 0b1111
    assert str(Color.DC) == "2b--"
    assert Color("2b--") is Color.DC
    assert Color(0b1111) is Color.DC
    assert Color.dcs() is Color.DC
    assert Color(Vec[2](0b1111)) is Color.DC

    assert str(Color("2b11").name) == "Color(2b11)"
    assert str(Color(0b1010).name) == "Color(2b11)"
    assert str(Color(Vec[2](0b1010)).name) == "Color(2b11)"

    with pytest.raises(TypeError):
        _ = Color(1.23e4)


def test_slicing():
    assert Color.GREEN[0] == "1b1"
    assert Color.GREEN[1] == "1b0"


def test_enum_error():
    """Test enum spec errors."""
    with pytest.raises(ValueError):

        class InvalidName(VecEnum):
            X = "4bXXXX"

        _ = InvalidName()

    with pytest.raises(ValueError):

        class InvalidData(VecEnum):
            FOO = "4bXXXX"

        _ = InvalidData()

    # The literal must be a str
    with pytest.raises(TypeError):

        class InvalidType(VecEnum):
            FOO = 42

        _ = InvalidType()

    with pytest.raises(ValueError):

        class InvalidMembers(VecEnum):
            A = "2b00"
            B = "3b000"

        _ = InvalidMembers()

    with pytest.raises(ValueError):

        class DuplicateMembers(VecEnum):
            A = "2b00"
            B = "2b01"
            C = "2b00"
