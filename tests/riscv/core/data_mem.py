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

        # self._depth = 2**self._word_addr_bits
        word_bytes = 2**byte_addr_bits
        width = word_bytes * BYTE_BITS

        # Ports
        addr = Bits(name="addr", parent=self, dtype=Vec[word_addr_bits])
        wr_en = Bit(name="wr_en", parent=self)
        wr_be = Bits(name="wr_be", parent=self, dtype=Vec[word_bytes])
        wr_data = Bits(name="wr_data", parent=self, dtype=Vec[width])
        rd_data = Bits(name="rd_data", parent=self, dtype=Vec[width])
        clock = Bit(name="clock", parent=self)

        # State
        mem = Array(name="mem", parent=self, dtype=Vec[width])

        # TODO(cjdrake): Remove
        self.word_bytes = word_bytes
        self.addr = addr
        self.wr_en = wr_en
        self.wr_be = wr_be
        self.wr_data = wr_data
        self.rd_data = rd_data
        self.clock = clock
        self.mem = mem

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
            self.mem[addr].next = cat(*[
                self.wr_data.value[8*i:8*(i+1)] if be[i] else  # noqa
                self.mem[addr].value[8*i:8*(i+1)]  # noqa
                for i in range(self.word_bytes)
            ])
            # fmt: on

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.addr, self.mem)
            addr = self.addr.value
            self.rd_data.next = self.mem[addr].value
