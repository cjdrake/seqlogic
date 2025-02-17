"""Test seqlogic.hier module."""

import pytest

from seqlogic.hier import Branch, Leaf


def test_basic():
    """Test basic hierarchical naming."""
    a = Branch(name="a")
    w = Leaf(name="w", parent=a)
    b = Branch(name="b", parent=a)
    x = Leaf(name="x", parent=b)
    y = Leaf(name="y", parent=b)

    with pytest.raises(ValueError):
        _ = Leaf(name="42", parent=a)

    assert a.is_root()

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
    assert list(a.iter_leaves()) == [w, x, y]
