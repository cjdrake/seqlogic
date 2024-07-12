"""Test bit array data type."""

# pylint: disable = pointless-statement

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false

import pytest

from seqlogic.lbconst import _W, _X, _0, _1
from seqlogic.vec import Bits, Vec, add, and_, bits, nand, nor, or_, stack, sub, vec, xnor, xor

E = Vec[0](*_X)

X = Vec[1](*_X)
F = Vec[1](*_0)
T = Vec[1](*_1)
W = Vec[1](*_W)


def test_basic():
    # Degenerate dimensions
    assert Bits[None] is Vec[0]
    assert Bits[1] is Vec[1]
    assert Bits[(1,)] is Vec[1]
    assert Bits[2] is Vec[2]
    assert Bits[(2,)] is Vec[2]

    # Invalid dimension lens
    with pytest.raises(ValueError):
        _ = Bits[2, 0, 3]
    with pytest.raises(ValueError):
        _ = Bits[2, -1, 3]

    b = Bits[2, 3, 4](0, 0)

    # Class attributes
    assert b.shape == (2, 3, 4)
    assert b.size == 24

    # Instance attributes
    assert len(b) == 2

    # Basic methods
    assert b.flatten() == Vec[24](0, 0)
    assert b.reshape((4, 3, 2)) == Bits[4, 3, 2](0, 0)
    with pytest.raises(ValueError):
        b.reshape((4, 4, 4))
    assert list(b.flat) == [Vec[1](0, 0)] * 24


# Operators retain their shape
def test_not():
    b = bits(["4b----", "4b1111", "4b0000", "4bXXXX"])
    assert str(~b) == "bits([4b----, 4b0000, 4b1111, 4bXXXX])"


def test_nor():
    b0 = bits(["4b----", "4b1111", "4b0000", "4bXXXX"])
    b1 = bits(["4b-10X", "4b-10X", "4b-10X", "4b-10X"])
    assert str(nor(b0, b1)) == "bits([4b-0-X, 4b000X, 4b-01X, 4bXXXX])"


def test_or():
    b0 = bits(["4b----", "4b1111", "4b0000", "4bXXXX"])
    b1 = bits(["4b-10X", "4b-10X", "4b-10X", "4b-10X"])
    assert str(or_(b0, b1)) == "bits([4b-1-X, 4b111X, 4b-10X, 4bXXXX])"


def test_nand():
    b0 = bits(["4b----", "4b1111", "4b0000", "4bXXXX"])
    b1 = bits(["4b-10X", "4b-10X", "4b-10X", "4b-10X"])
    assert str(nand(b0, b1)) == "bits([4b--1X, 4b-01X, 4b111X, 4bXXXX])"


def test_and():
    b0 = bits(["4b----", "4b1111", "4b0000", "4bXXXX"])
    b1 = bits(["4b-10X", "4b-10X", "4b-10X", "4b-10X"])
    assert str(and_(b0, b1)) == "bits([4b--0X, 4b-10X, 4b000X, 4bXXXX])"


def test_xnor():
    b0 = bits(["4b----", "4b1111", "4b0000", "4bXXXX"])
    b1 = bits(["4b-10X", "4b-10X", "4b-10X", "4b-10X"])
    assert str(xnor(b0, b1)) == "bits([4b---X, 4b-10X, 4b-01X, 4bXXXX])"


def test_xor():
    b0 = bits(["4b----", "4b1111", "4b0000", "4bXXXX"])
    b1 = bits(["4b-10X", "4b-10X", "4b-10X", "4b-10X"])
    assert str(xor(b0, b1)) == "bits([4b---X, 4b-01X, 4b-10X, 4bXXXX])"


def test_add():
    b0 = bits(["4b1010", "4b0101"])
    b1 = bits(["4b0101", "4b1010"])
    assert str(add(b0, b1).s) == "bits([4b1111, 4b1111])"


def test_sub():
    b0 = bits(["4b1111", "4b1111"])
    b1 = bits(["4b0101", "4b1010"])
    assert str(sub(b0, b1).s) == "bits([4b1010, 4b0101])"


def test_xt():
    assert bits("4b1010").xt(4) == bits("8b0000_1010")
    # Zero extension on multi-dimensional array will flatten
    assert bits(["4b0000", "4b1111"]).xt(2) == bits("10b00_1111_0000")


def test_sxt():
    assert bits("4b1010").sxt(4) == bits("8b1111_1010")
    assert bits("4b0101").sxt(4) == bits("8b0000_0101")
    # Sign extension of multi-dimensional array will flatten
    assert bits(["4b0000", "4b1111"]).sxt(2) == bits("10b11_1111_0000")


def test_lsh():
    b = bits(["4b1111", "4b0000"])
    assert str(b << 2) == "bits([4b1100, 4b0011])"


def test_rsh():
    b = bits(["4b1111", "4b0000"])
    assert str(b >> 2) == "bits([4b0011, 4b0000])"


def test_srsh():
    b0 = bits(["4b1111", "4b0000"])
    y, _ = b0.srsh(2)
    assert str(y) == "bits([4b0011, 4b0000])"

    b1 = bits(["4b0000", "4b1111"])
    y, _ = b1.srsh(2)
    assert str(y) == "bits([4b1100, 4b1111])"


def test_rank2_errors():
    """Test bits function rank2 errors."""
    # Mismatched str literal
    with pytest.raises(TypeError):
        bits(["4b-10X", "3b10X"])
    # bits followed by some invalid type
    with pytest.raises(TypeError):
        bits(["4b-10X", 42])


R3VEC = """\
bits([[4b-10X, 4b-10X],
      [4b-10X, 4b-10X]])"""


def test_rank3_vec():
    """Test bits function w/ rank3 input."""
    b = bits(
        [
            ["4b-10X", "4b-10X"],
            ["4b-10X", "4b-10X"],
        ]
    )

    assert b.flatten() == bits("16b-10X_-10X_-10X_-10X")

    # Test __str__
    assert str(b) == R3VEC


R4VEC = """\
bits([[[4b-10X, 4b-10X],
       [4b-10X, 4b-10X]],

      [[4b-10X, 4b-10X],
       [4b-10X, 4b-10X]]])"""


def test_rank4_vec():
    """Test bits function w/ rank4 input."""
    b = bits(
        [
            [["4b-10X", "4b-10X"], ["4b-10X", "4b-10X"]],
            [["4b-10X", "4b-10X"], ["4b-10X", "4b-10X"]],
        ]
    )

    # Test __str__
    assert str(b) == R4VEC


def test_invalid_vec():
    """Test bits function invalid input."""
    with pytest.raises(TypeError):
        bits(42)


def test_slicing():
    """Test bits slicing behavior."""
    b = bits(
        [
            ["4b0000", "4b0001", "4b0010", "4b0011"],
            ["4b0100", "4b0101", "4b0110", "4b0111"],
            ["4b1000", "4b1001", "4b1010", "4b1011"],
            ["4b1100", "4b1101", "4b1110", "4b1111"],
        ]
    )

    assert b.shape == (4, 4, 4)

    with pytest.raises(IndexError):
        b[-5]
    with pytest.raises(TypeError):
        b["invalid"]  # pyright: ignore[reportArgumentType]
    # Slice step not supported
    with pytest.raises(ValueError):
        b[0:4:1]

    assert b == b[:]
    assert b == b[0:4]
    assert b == b[-4:]
    assert b == b[-5:]
    assert b == b[:, :]
    assert b == b[:, :, :]

    assert b[0] == b[0, :]
    assert b[0] == b[0, 0:4]
    assert b[0] == b[0, -4:]
    assert b[0] == b[0, -5:]
    assert b[0] == b[0, :, :]

    assert b[0] == bits(["4b0000", "4b0001", "4b0010", "4b0011"])
    assert b[1] == bits(["4b0100", "4b0101", "4b0110", "4b0111"])
    assert b[2] == bits(["4b1000", "4b1001", "4b1010", "4b1011"])
    assert b[3] == bits(["4b1100", "4b1101", "4b1110", "4b1111"])

    assert b[0, 0] == b[0, 0, :]
    assert b[0, 0] == b[0, 0, 0:4]
    assert b[0, 0] == b[0, 0, -4:]
    assert b[0, 0] == b[0, 0, -5:]

    assert b[0, 0] == bits("4b0000")
    assert b[1, 1] == bits("4b0101")
    assert b[2, 2] == bits("4b1010")
    assert b[3, 3] == bits("4b1111")

    assert b[0, :, 0] == bits("4b1010")
    assert b[1, :, 1] == bits("4b1100")
    assert b[2, :, 2] == bits("4b0000")
    assert b[3, :, 3] == bits("4b1111")

    assert b[0, 0, :-1] == bits("3b000")
    assert b[0, 0, :-2] == bits("2b00")
    assert b[0, 0, :-3] == bits("1b0")
    assert b[0, 0, :-4] == bits()

    assert b[0, 0, 0] == F
    # assert b[0, bits("2b00"), 0] == F
    assert b[-4, -4, -4] == F
    assert b[3, 3, 3] == T
    # assert b[3, bits("2b11"), 3] == T
    assert b[-1, -1, -1] == T

    with pytest.raises(ValueError):
        b[0, 0, 0, 0]
    with pytest.raises(TypeError):
        b["invalid"]  # pyright: ignore[reportArgumentType]


def test_bits():
    assert bits() == E
    assert bits(False) == "1b0"
    assert bits([0, 1, 0, 1]) == "4b1010"
    assert bits(["2b00", "2b01", "2b10", "2b11"]) == "8b11100100"
    assert bits([vec("2b00"), "2b01", "2b10", "2b11"]) == "8b11100100"

    with pytest.raises(TypeError):
        bits(42)
    with pytest.raises(TypeError):
        stack(["2b00", "1b0"])


def test_stack():
    assert stack() == E
    assert stack(False) == "1b0"
    assert stack(0, 1, 0, 1) == "4b1010"
    assert stack("2b00", "2b01", "2b10", "2b11") == "8b11100100"
    assert stack(vec("2b00"), "2b01", "2b10", "2b11") == "8b11100100"

    with pytest.raises(TypeError):
        stack(42)
    with pytest.raises(TypeError):
        stack("2b00", "1b0")
