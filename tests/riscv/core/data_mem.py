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
        self._depth = 2**self._word_addr_bits
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
        self._mem = Array(
            name="mem",
            parent=self,
            shape=(self._depth,),
            dtype=Vec[self._width],
        )

    @active
    async def p_f_0(self):
        def f():
            return self.clock.is_posedge() and self.wr_en.value == "1b1"

        while True:
            await resume((self.clock, f))
            addr = self.addr.value
            # If wr_en=1, address must be known
            assert not addr.has_unknown()
            wr_val = self.wr_data.value
            mem_val = self._mem[addr].value
            # fmt: off
            wr_bytes = [wr_val[8*i:8*(i+1)] for i in range(self._word_bytes)]  # noqa
            mem_bytes = [mem_val[8*i:8*(i+1)] for i in range(self._word_bytes)]  # noqa
            bytes_ = [
                self.wr_be.value[i].ite(wr_bytes[i], mem_bytes[i])
                for i in range(self._word_bytes)
            ]
            # fmt: on
            self._mem[addr].next = cat(*bytes_)

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.addr, self._mem)
            addr = self.addr.value
            self.rd_data.next = self._mem[addr].value
