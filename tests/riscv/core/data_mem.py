"""Data Memory."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Module, changed, resume
from seqlogic.lbool import Vec, cat
from seqlogic.sim import active, reactive


class DataMem(Module):
    """Data random access, read/write memory."""

    def __init__(self, name: str, parent: Module | None, word_addr_bits: int = 10):
        super().__init__(name, parent)

        # Ports
        self.bits(name="addr", dtype=Vec[word_addr_bits], port=True)
        self.bit(name="wr_en", port=True)
        self.bits(name="wr_be", dtype=Vec[4], port=True)
        self.bits(name="wr_data", dtype=Vec[32], port=True)
        self.bits(name="rd_data", dtype=Vec[32], port=True)
        self.bit(name="clock", port=True)

        # State
        self.array(name="mem", dtype=Vec[32])

    @active
    async def p_f_0(self):
        def f():
            return self._clock.is_posedge() and self._wr_en.value == "1b1"

        while True:
            await resume((self._clock, f))
            addr = self._addr.value
            be = self._wr_be.value
            # If wr_en=1, addr/be must be known
            assert not addr.has_unknown() and not be.has_unknown()
            # fmt: off
            value = cat(*[
                self._wr_data.value[8*i:8*(i+1)] if be[i] else  # noqa
                self._mem.values[addr][8*i:8*(i+1)]  # noqa
                for i in range(4)
            ])
            self._mem.set_next(addr, value)
            # fmt: on

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self._addr, self._mem)
            addr = self._addr.value
            self._rd_data.next = self._mem.values[addr]
