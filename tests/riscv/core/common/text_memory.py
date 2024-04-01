"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bits, Module, changed
from seqlogic.bits import xes
from seqlogic.sim import always_comb

from .. import WORD_BITS

WIDTH = 32
DEPTH = 1024


class TextMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.rd_addr = Bits(name="rd_addr", parent=self, shape=(14,))
        self.rd_data = Bits(name="rd_data", parent=self, shape=(WORD_BITS,))

        # State
        self._mem = Array(
            name="mem", parent=self, unpacked_shape=(DEPTH,), packed_shape=(WORD_BITS,)
        )

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.rd_addr, self._mem)
            try:
                i = self.rd_addr.next.to_uint()
            except ValueError:
                self.rd_data.next = xes((WORD_BITS,))
            else:
                self.rd_data.next = self._mem.get_next(i)
