"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module
from seqlogic.hier import List
from seqlogic.logicvec import T, cat, xes
from seqlogic.sim import notify

from ..misc import COMBI, FLOP

NUM = 32 * 1024


class DataMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.addr = Bits(name="addr", parent=self, shape=(15,))

        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_be = Bits(name="wr_be", parent=self, shape=(4,))
        self.wr_data = Bits(name="wr_data", parent=self, shape=(32,))

        self.rd_data = Bits(name="rd_data", parent=self, shape=(32,))

        self.clock = Bit(name="clock", parent=self)

        # State
        self.mem = List(name="mem", parent=self)
        for i in range(NUM):
            reg = Bits(name=str(i), parent=self.mem, shape=(32,))
            self.mem.append(reg)

        self._procs.add((self.proc_wr_port, FLOP))
        self._procs.add((self.proc_rd_data, COMBI))

    async def proc_wr_port(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.clock.posedge)
            if self.wr_en.value == T:
                i = self.addr.value.to_uint()
                parts = []
                for j in range(4):
                    if self.wr_be.value[j] == T:
                        v = self.wr_data.value
                    else:
                        v = self.mem[i].value
                    a, b = 8 * j, 8 * (j + 1)
                    parts.append(v[a:b])
                self.mem[i].next = cat(parts, flatten=True)

    async def proc_rd_data(self):
        """TODO(cjdrake): Write docstring."""
        # TODO(cjdrake): This should include *all* variables in the mem
        others = [self.mem[i].changed for i in range(16)]
        while True:
            await notify(self.addr.changed, *others)
            try:
                i = self.addr.next.to_uint()
            except ValueError:
                self.rd_data.next = xes((32,))
            else:
                self.rd_data.next = self.mem[i].next
