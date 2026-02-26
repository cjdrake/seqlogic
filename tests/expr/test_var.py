"""Test symbolic variables"""

from seqlogic import Variable

a = Variable(name="a")


def test_basic():
    assert a.name == "a"
    assert set(a.iter_vars()) == {a}
