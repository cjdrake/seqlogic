"""Test hierarchy module."""

from seqlogic import Module
from seqlogic.hier import Variable


def test_basic():
    """Test basic hierarchical naming."""
    a = Module(name="a")
    w = Variable(name="w", parent=a)
    b = Module(name="b", parent=a)
    x = Variable(name="x", parent=b)
    y = Variable(name="y", parent=b)

    assert a.name == "a"
    assert a.qualname == "/a"
    assert b.name == "b"
    assert b.qualname == "/a/b"

    assert w.name == "w"
    assert w.qualname == "/a/w"
    assert x.name == "x"
    assert x.qualname == "/a/b/x"
    assert y.name == "y"
    assert y.qualname == "/a/b/y"

    assert list(a.iter_bfs()) == [a, w, b, x, y]
    assert list(a.iter_dfs()) == [w, x, y, b, a]
