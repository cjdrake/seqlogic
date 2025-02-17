"""Test symbolic variables"""

from seqlogic.expr import Variable

a = Variable(name="a")


def test_basic():
    assert repr(a) == "a"
    assert a.name == "a"
    assert set(a.iter_vars()) == {a}
