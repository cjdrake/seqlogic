"""Test Gray Code Algorithm."""

from seqlogic import u2bv
from seqlogic.algorithm.gray import bin2gray, gray2bin

B2G_EXP = [
    0b0000,
    0b0001,
    0b0011,
    0b0010,
    0b0110,
    0b0111,
    0b0101,
    0b0100,
    0b1100,
    0b1101,
    0b1111,
    0b1110,
    0b1010,
    0b1011,
    0b1001,
    0b1000,
]


def test_bin2gray():
    """Test bin2gray function."""
    for b, g in enumerate(B2G_EXP):
        assert bin2gray(u2bv(b, 4)).to_uint() == g


def test_gray2bin():
    """Test gray2bin function."""
    for b, g in enumerate(B2G_EXP):
        temp = gray2bin(u2bv(g, 4)).to_uint()
        assert temp == b
