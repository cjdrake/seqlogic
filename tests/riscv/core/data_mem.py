"""Data Memory."""

# pyright: reportAttributeAccessIssue=false

import operator

from seqlogic import Module, resume
from seqlogic.lbool import Vec, cat
from seqlogic.sim import active


class DataMem(Module):
    """Data random access, read/write memory."""

    def __init__(self, name: str, parent: Module | None, word_addr_bits: int = 10):
        super().__init__(name, parent)

        # Ports
        addr = self.bits(name="addr", dtype=Vec[word_addr_bits], port=True)
        self.bit(name="wr_en", port=True)
        self.bits(name="wr_be", dtype=Vec[4], port=True)
        self.bits(name="wr_data", dtype=Vec[32], port=True)
        rd_data = self.bits(name="rd_data", dtype=Vec[32], port=True)
        self.bit(name="clock", port=True)

        # State
        mem = self.array(name="mem", dtype=Vec[32])

        # Read Port
        self.combi(rd_data, operator.getitem, mem, addr)

    # Write Port
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
            data = cat(*[
                self._wr_data.value[8*i:8*(i+1)] if be[i] else  # noqa
                self._mem[addr].value[8*i:8*(i+1)]  # noqa
                for i in range(4)
            ])
            self._mem[addr].next = data
            # fmt: on
