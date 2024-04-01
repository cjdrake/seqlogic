"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bit, Bits, Module, changed, resume
from seqlogic.bits import T, cat, ones, xes
from seqlogic.sim import always_comb, always_ff

WORD_BITS = 32
WORD_BYTES = WORD_BITS // 8
DEPTH = 32 * 1024


class DataMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()

    def build(self):
        """Write docstring."""
        # Ports
        self.addr = Bits(name="addr", parent=self, shape=(15,))

        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_be = Bits(name="wr_be", parent=self, shape=(WORD_BYTES,))
        self.wr_data = Bits(name="wr_data", parent=self, shape=(WORD_BITS,))

        self.rd_data = Bits(name="rd_data", parent=self, shape=(WORD_BITS,))

        self.clock = Bit(name="clock", parent=self)

        # State
        self.mem = Array(
            name="mem", parent=self, unpacked_shape=(DEPTH,), packed_shape=(WORD_BITS,)
        )

    @always_ff
    async def p_f_0(self):
        """TODO(cjdrake): Write docstring."""

        def f():
            return self.clock.posedge() and self.wr_en.value == T

        while True:
            await resume((self.clock, f))
            word_addr = self.addr.value.to_uint()
            if self.wr_be.value == ones((WORD_BYTES,)):
                word = self.wr_data.value
            else:
                wr_val = self.wr_data.value.reshape((WORD_BYTES, 8))
                mem_val = self.mem.get_value(word_addr).reshape((WORD_BYTES, 8))
                bytes_ = [
                    wr_val[i] if self.wr_be.value[i] == T else mem_val[i] for i in range(WORD_BYTES)
                ]
                word = cat(bytes_, flatten=True)
            self.mem.set_next(word_addr, word)

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.addr, self.mem)
            try:
                i = self.addr.next.to_uint()
            except ValueError:
                self.rd_data.next = xes((WORD_BITS,))
            else:
                self.rd_data.next = self.mem.get_next(i)
