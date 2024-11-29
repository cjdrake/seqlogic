"""Bits Union data type."""

# pylint: disable=protected-access

from functools import partial

from .bits import Bits, Vector, _lit2bv, _vec_size
from .util import classproperty, mask


class _UnionMeta(type):
    """Union Metaclass: Create union base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "Union":
            return super().__new__(mcs, name, bases, attrs)

        # Get field_name: field_type items
        try:
            fields = list(attrs["__annotations__"].items())
        except KeyError as e:
            raise ValueError("Empty Union is not supported") from e

        # Create Union class
        size = max(field_type.size for _, field_type in fields)
        union = super().__new__(mcs, name, bases + (Bits,), {})

        # Class properties
        union.size = classproperty(lambda _: size)

        # Override Bits.__init__ method
        def _init(self, arg: Bits | str):
            if isinstance(arg, str):
                x = _lit2bv(arg)
            else:
                x = arg
            ts = []
            for _, ft in fields:
                if ft not in ts:
                    ts.append(ft)
            if not isinstance(x, tuple(ts)):
                s = ", ".join(t.__name__ for t in ts)
                s = f"Expected arg to be {{{s}}}, or str literal"
                raise TypeError(s)
            self._data = x.data

        union.__init__ = _init

        # Override Bits.__getitem__ method
        def _getitem(self, key: int | slice | Bits | str) -> Vector:
            size, (d0, d1) = self._get_key(key)
            return _vec_size(size)(d0, d1)

        union.__getitem__ = _getitem

        # Override Bits.__str__ method
        def _str(self):
            parts = [f"{name}("]
            for fn, _ in fields:
                x = getattr(self, fn)
                parts.append(f"    {fn}={x!s},")
            parts.append(")")
            return "\n".join(parts)

        union.__str__ = _str

        # Override Bits.__repr__ method
        def _repr(self):
            parts = [f"{name}("]
            for fn, _ in fields:
                x = getattr(self, fn)
                parts.append(f"    {fn}={x!r},")
            parts.append(")")
            return "\n".join(parts)

        union.__repr__ = _repr

        # Create Union fields
        def _fget(ft: type[Bits], self):
            m = mask(ft.size)
            d0 = self._data[0] & m
            d1 = self._data[1] & m
            return ft._cast_data(d0, d1)

        for fn, ft in fields:
            setattr(union, fn, property(fget=partial(_fget, ft)))

        return union


class Union(metaclass=_UnionMeta):
    """User defined union data type.

    Compose a type from the union of other types.

    Extend from ``Union`` to define a struct:

    >>> from seqlogic import Vec
    >>> class Response(Union):
    ...     error: Vec[4]
    ...     data: Vec[8]

    Use the new type's constructor to create ``Union`` instances:

    >>> rsp = Response("8h0f")

    Access individual fields using attributes:

    >>> rsp.error
    bits("4b1111")
    >>> rsp.data
    bits("8b0000_1111")

    ``Unions`` have a ``size``, but no ``shape``.
    They do **NOT** implement a ``__len__`` method.

    >>> Response.size
    8

    ``Union`` slicing behaves like a ``Vector``:

    >>> rsp[3:5]
    bits("2b01")
    """
