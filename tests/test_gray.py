"""Test Gray Code Algorithm."""

from seqlogic.algorithms.gray import bin2gray, gray2bin
from seqlogic.logicvec import uint2vec

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
        assert bin2gray(uint2vec(b, 4)).to_uint() == g


def test_gray2bin():
    """Test gray2bin function."""
    for b, g in enumerate(B2G_EXP):
        assert gray2bin(uint2vec(g, 4)).to_uint() == b
