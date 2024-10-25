"""Bits Struct data type."""

# pylint: disable=exec-used
# pylint: disable=protected-access

from functools import partial

from .bits import Bits, Empty, Scalar, Vector, _expect_size, _vec_size
from .util import classproperty, mask


def _struct_init_source(fields: list[tuple[str, type]]) -> str:
    """Return source code for Struct __init__ method w/ fields."""
    lines = []
    s = ", ".join(f"{fn}=None" for fn, _ in fields)
    lines.append(f"def init(self, {s}):\n")
    s = ", ".join(fn for fn, _ in fields)
    lines.append(f"    _init_body(self, {s})\n")
    return "".join(lines)


class _StructMeta(type):
    """Struct Metaclass: Create struct base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "Struct":
            return super().__new__(mcs, name, bases, attrs)

        # Get field_name: field_type items
        try:
            fields = list(attrs["__annotations__"].items())
        except KeyError as e:
            raise ValueError("Empty Struct is not supported") from e

        # Add struct member base/size attributes
        offset = 0
        offsets = {}
        for field_name, field_type in fields:
            offsets[field_name] = offset
            offset += field_type.size

        # Create Struct class
        size = sum(field_type.size for _, field_type in fields)
        struct = super().__new__(mcs, name, bases + (Bits,), {})

        # Class properties
        struct.size = classproperty(lambda _: size)

        # Override Bits.__init__ method
        def _init_body(obj, *args):
            d0, d1 = 0, 0
            for arg, (fn, ft) in zip(args, fields):
                if arg is not None:
                    # TODO(cjdrake): Check input type?
                    x = _expect_size(arg, ft.size)
                    d0 |= x.data[0] << offsets[fn]
                    d1 |= x.data[1] << offsets[fn]
            obj._data = (d0, d1)

        source = _struct_init_source(fields)
        globals_ = {"_init_body": _init_body}
        locals_ = {}
        exec(source, globals_, locals_)
        struct.__init__ = locals_["init"]

        # Override Bits.__getitem__ method
        def _getitem(self, key: int | slice | Bits | str) -> Empty | Scalar | Vector:
            size, (d0, d1) = self._get_key(key)
            return _vec_size(size)(d0, d1)

        struct.__getitem__ = _getitem

        # Override Bits.__str__ method
        def _str(self):
            parts = [f"{name}("]
            for fn, _ in fields:
                x = getattr(self, fn)
                parts.append(f"    {fn}={x!s},")
            parts.append(")")
            return "\n".join(parts)

        struct.__str__ = _str

        # Override Bits.__repr__ method
        def _repr(self):
            parts = [f"{name}("]
            for fn, _ in fields:
                x = getattr(self, fn)
                parts.append(f"    {fn}={x!r},")
            parts.append(")")
            return "\n".join(parts)

        struct.__repr__ = _repr

        # Create Struct fields
        def _fget(fn: str, ft: type[Bits], self):
            m = mask(ft.size)
            d0 = (self._data[0] >> offsets[fn]) & m
            d1 = (self._data[1] >> offsets[fn]) & m
            return ft._cast_data(d0, d1)

        for fn, ft in fields:
            setattr(struct, fn, property(fget=partial(_fget, fn, ft)))

        return struct


class Struct(metaclass=_StructMeta):
    """Struct Base Class: Create struct."""
