"""Test seqlogic.lbool.VecStruct."""

from seqlogic.lbool import Vec, VecStruct, vec

# PyRight is confused by MetaClass behavior
# pyright: reportCallIssue=false


class Pixel(VecStruct):
    """TODO(cjdrake): Write docstring."""

    r: Vec[8]
    g: Vec[8]
    b: Vec[8]


def test_basic():
    p = Pixel(r=vec("8b0001_0001"), g=vec("8b0010_0010"), b=vec("8b0100_0100"))

    assert str(p.r) == "8b0001_0001"
    assert str(p.g) == "8b0010_0010"
    assert str(p.b) == "8b0100_0100"

    assert str(p) == "Pixel(r=8b0001_0001, g=8b0010_0010, b=8b0100_0100)"
    assert repr(p) == 'Pixel(r=vec("8b0001_0001"), g=vec("8b0010_0010"), b=vec("8b0100_0100"))'

    assert len(p) == 24
