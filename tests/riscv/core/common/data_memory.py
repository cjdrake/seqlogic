"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bit, Bits, Module, changed, resume
from seqlogic.lbool import cat, ones, xes
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
        self._mem = Array(
            name="mem", parent=self, unpacked_shape=(DEPTH,), packed_shape=(WORD_BITS,)
        )

    @always_ff
    async def p_f_0(self):
        """TODO(cjdrake): Write docstring."""

        def f():
            return self.clock.is_posedge() and self.wr_en.value == ones(1)

        while True:
            await resume((self.clock, f))
            word_addr = self.addr.value.to_uint()
            if self.wr_be.value == ones(WORD_BYTES):
                word = self.wr_data.value
            else:
                wr_val = self.wr_data.value
                mem_val = self._mem.get_value(word_addr)
                # fmt: off
                wr_bytes = [wr_val[8*i:8*(i+1)] for i in range(WORD_BYTES)]  # noqa
                mem_bytes = [mem_val[8*i:8*(i+1)] for i in range(WORD_BYTES)]  # noqa
                # fmt: on
                bytes_ = [
                    wr_bytes[i] if self.wr_be.value[i] == ones(1) else mem_bytes[i]
                    for i in range(WORD_BYTES)
                ]
                word = cat(*bytes_)
            self._mem.set_next(word_addr, word)

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.addr, self._mem)
            try:
                i = self.addr.next.to_uint()
            except ValueError:
                self.rd_data.next = xes(WORD_BITS)
            else:
                self.rd_data.next = self._mem.get_next(i)
