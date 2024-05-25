"""Data Memory."""

from seqlogic import Array, Bit, Bits, Module, changed, resume
from seqlogic.lbool import Vec, cat
from seqlogic.sim import active, reactive

BYTE_BITS = 8


class DataMem(Module):
    """Data random access, read/write memory."""

    def __init__(
        self,
        name: str,
        parent: Module | None,
        word_addr_bits: int = 10,
        byte_addr_bits: int = 2,
    ):
        super().__init__(name, parent)
        self._word_addr_bits = word_addr_bits
        self._byte_addr_bits = byte_addr_bits
        # self._depth = 2**self._word_addr_bits
        self._word_bytes = 2**self._byte_addr_bits
        self._width = self._word_bytes * BYTE_BITS
        self._build()

    def _build(self):
        """Write docstring."""
        # Ports
        self.addr = Bits(name="addr", parent=self, dtype=Vec[self._word_addr_bits])
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_be = Bits(name="wr_be", parent=self, dtype=Vec[self._word_bytes])
        self.wr_data = Bits(name="wr_data", parent=self, dtype=Vec[self._width])
        self.rd_data = Bits(name="rd_data", parent=self, dtype=Vec[self._width])
        self.clock = Bit(name="clock", parent=self)

        # State
        self._mem = Array(name="mem", parent=self, dtype=Vec[self._width])

    @active
    async def p_f_0(self):
        def f():
            return self.clock.is_posedge() and self.wr_en.value == "1b1"

        while True:
            await resume((self.clock, f))
            addr = self.addr.value
            be = self.wr_be.value
            # If wr_en=1, addr/be must be known
            assert not addr.has_unknown() and not be.has_unknown()
            # fmt: off
            self._mem[addr].next = cat(*[
                self.wr_data.value[8*i:8*(i+1)] if be[i] else  # noqa
                self._mem[addr].value[8*i:8*(i+1)]  # noqa
                for i in range(self._word_bytes)
            ])
            # fmt: on

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.addr, self._mem)
            addr = self.addr.value
            self.rd_data.next = self._mem[addr].value
