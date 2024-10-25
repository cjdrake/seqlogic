"""Bits Enum data type."""

# pylint: disable=protected-access

from .bits import Bits, _expect_size, _parse_lit, _vec_size
from .util import mask


class _EnumMeta(type):
    """Enum Metaclass: Create enum base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "Enum":
            return super().__new__(mcs, name, bases, attrs)

        enum_attrs = {}
        data2key: dict[tuple[int, int], str] = {}
        size = None
        for key, val in attrs.items():
            if key.startswith("__"):
                enum_attrs[key] = val
            # NAME = lit
            else:
                if size is None:
                    size, data = _parse_lit(val)
                else:
                    size_i, data = _parse_lit(val)
                    if size_i != size:
                        s = f"Expected lit len {size}, got {size_i}"
                        raise ValueError(s)
                if key in ("X", "DC"):
                    raise ValueError(f"Cannot use reserved name = '{key}'")
                dmax = mask(size)
                if data in ((0, 0), (dmax, dmax)):
                    raise ValueError(f"Cannot use reserved value = {val}")
                if data in data2key:
                    raise ValueError(f"Duplicate value: {val}")
                data2key[data] = key

        # Empty Enum
        if size is None:
            raise ValueError("Empty Enum is not supported")

        # Add X/DC members
        data2key[(0, 0)] = "X"
        dmax = mask(size)
        data2key[(dmax, dmax)] = "DC"

        # Create Enum class
        vec = _vec_size(size)
        enum = super().__new__(mcs, name, bases + (vec,), enum_attrs)

        # Instantiate members
        for (d0, d1), key in data2key.items():
            setattr(enum, key, enum._cast_data(d0, d1))

        # Override Vector._cast_data method
        def _cast_data(cls, d0: int, d1: int) -> Bits:
            data = (d0, d1)
            try:
                obj = getattr(cls, data2key[data])
            except KeyError:
                obj = object.__new__(cls)
                obj._data = data
            return obj

        # Override Vector._cast_data method
        enum._cast_data = classmethod(_cast_data)

        # Override Vector.__new__ method
        def _new(cls, arg: Bits | str):
            x = _expect_size(arg, cls.size)
            return cls.cast(x)

        enum.__new__ = _new

        # Override Vector.__repr__ method
        def _repr(self):
            try:
                return f"{name}.{data2key[self._data]}"
            except KeyError:
                return f'{name}("{vec.__str__(self)}")'

        enum.__repr__ = _repr

        # Override Vector.__str__ method
        def _str(self):
            try:
                return f"{name}.{data2key[self._data]}"
            except KeyError:
                return f"{name}({vec.__str__(self)})"

        enum.__str__ = _str

        # Create name property
        def _name(self):
            try:
                return data2key[self._data]
            except KeyError:
                return f"{name}({vec.__str__(self)})"

        enum.name = property(fget=_name)

        # Override VCD methods
        enum.vcd_var = lambda _: "string"
        enum.vcd_val = _name

        return enum


class Enum(metaclass=_EnumMeta):
    """Enum Base Class: Create enums."""
