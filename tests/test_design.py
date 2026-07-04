"""Test seqlogic.design module."""

import pytest
from bvwx import Array

from seqlogic import DesignError, Module


def test_duplicate_input():
    class Top(Module):
        def build(self):
            self.input(name="a", dtype=Array[8])
            with pytest.raises(DesignError):
                self.input(name="a", dtype=Array[8])

    _ = Top(name="top")


def test_duplicate_output():
    class Top(Module):
        def build(self):
            self.output(name="a", dtype=Array[8])
            with pytest.raises(DesignError):
                self.output(name="a", dtype=Array[8])

    _ = Top(name="top")


def test_duplicate_logic():
    # Duplicate names
    class Top1(Module):
        def build(self):
            self.logic(name="a", dtype=Array[8])
            with pytest.raises(DesignError):
                self.logic(name="a", dtype=Array[8])

    _ = Top1(name="top1")

    # Reserved names
    class Top2(Module):
        def build(self):
            with pytest.raises(DesignError):
                self.logic(name="name", dtype=Array[8])
            with pytest.raises(DesignError):
                self.logic(name="parent", dtype=Array[8])
            with pytest.raises(DesignError):
                self.logic(name="_children", dtype=Array[8])
            with pytest.raises(DesignError):
                self.logic(name="_reactive", dtype=Array[8])
            with pytest.raises(DesignError):
                self.logic(name="_active", dtype=Array[8])
            with pytest.raises(DesignError):
                self.logic(name="_inactive", dtype=Array[8])

    _ = Top2(name="top2")


def test_duplicate_submod():
    class Foo(Module):
        def build(self):
            pass

    class Top(Module):
        def build(self):
            self.submod(name="a", mod=Foo)
            with pytest.raises(DesignError):
                self.submod(name="a", mod=Foo)

    _ = Top(name="top")
