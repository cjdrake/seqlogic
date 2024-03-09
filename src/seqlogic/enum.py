"""Enum Logic Data Type."""

from . import lbool
from .bits import bits

# PyLint is confused by MetaClass behavior
# pylint: disable = no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false


class _EnumMeta(type):
    """Enum Metaclass: Create enum base classes."""

    def __new__(mcs, name, bases, attrs):
        base_attrs = {}
        lit2name = {}
        for key, val in attrs.items():
            if key.startswith("__"):
                base_attrs[key] = val
            else:
                lit2name[val] = key

        # Create enum class, save lit => name mapping
        cls = super().__new__(mcs, name, bases + (bits,), base_attrs)
        cls._lit2name = lit2name

        return cls

    def __init__(cls, unused_name, unused_bases, unused_attrs):
        # Populate the enum items
        for lit in cls._lit2name:
            _ = cls(lit)


class Enum(metaclass=_EnumMeta):
    """Enum Base Class: Create enums."""

    def __new__(cls, lit: str):
        """TODO(cjdrake): Write docstring."""
        if lit not in cls._lit2name:
            valid = ", ".join(cls._lit2name)
            s = f"Expected literal in {{{valid}}}, got {lit}"
            raise ValueError(s)
        name = cls._lit2name[lit]
        obj = getattr(cls, name, None)
        if obj is None:
            obj = super().__new__(cls)
            setattr(cls, name, obj)
        return obj

    def __init__(self, lit: str):
        """TODO(cjdrake): Write docstring."""
        super().__init__(lbool.lit2vec(lit))
        # Override string representation from base class
        self._name = self.__class__._lit2name[lit]

    def __str__(self) -> str:
        return self._name
