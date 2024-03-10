"""Enum Logic Data Type."""

from .bits import Bits
from .lbool import lit2vec

# PyLint is confused by MetaClass behavior
# pylint: disable = no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false


class _EnumMeta(type):
    """Enum Metaclass: Create enum base classes."""

    def __new__(mcs, name, bases, attrs):
        base_attrs = {}
        lit2name = {}
        size = None
        for key, val in attrs.items():
            if key.startswith("__"):
                base_attrs[key] = val
            else:
                if key == "X":
                    raise ValueError("Cannot use reserved name 'X'")
                if size is None:
                    size = len(lit2vec(val))
                else:
                    n = len(lit2vec(val))
                    if n != size:
                        s = f"Expected lit size {size}, got {n}"
                        raise ValueError(s)
                lit2name[val] = key

        # Add the 'X' item
        if size is not None:
            lit2name[f"{size}b" + "X" * size] = "X"

        # Create enum class, save lit => name mapping
        cls = super().__new__(mcs, name, bases + (Bits,), base_attrs)
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
        v = lit2vec(lit)
        super().__init__((len(v),), v.data)
        # Override string representation from base class
        self._name = self.__class__._lit2name[lit]

    def __str__(self) -> str:
        return self._name
