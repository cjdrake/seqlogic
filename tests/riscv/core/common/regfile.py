"""TODO(cjdrake): Write docstring."""

from seqlogic import Module
from seqlogic.hier import List
from seqlogic.logicvec import T, xes, zeros
from seqlogic.sim import notify
from seqlogic.var import Bit, LogicVec

from ..misc import COMBI, FLOP, TASK

NUM = 32


class RegFile(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_addr = LogicVec(name="wr_addr", parent=self, shape=(5,))
        self.wr_data = LogicVec(name="wr_data", parent=self, shape=(32,))
        self.rs1_addr = LogicVec(name="rs1_addr", parent=self, shape=(5,))
        self.rs1_data = LogicVec(name="rs1_data", parent=self, shape=(32,))
        self.rs2_addr = LogicVec(name="rs2_addr", parent=self, shape=(5,))
        self.rs2_data = LogicVec(name="rs2_data", parent=self, shape=(32,))
        self.clock = Bit(name="clock", parent=self)

        # State
        self.regs = List(name="regs", parent=self)
        for i in range(NUM):
            reg = LogicVec(name=str(i), parent=self.regs, shape=(32,))
            self.regs.append(reg)

        self._procs.add((self.proc_init, TASK))
        self._procs.add((self.proc_wr_port, FLOP))
        self._procs.add((self.proc_rd1_port, COMBI))
        self._procs.add((self.proc_rd2_port, COMBI))

    async def proc_init(self):
        """TODO(cjdrake): Write docstring."""
        self.regs[0].next = zeros((32,))
        for i in range(1, NUM):
            self.regs[i].next = zeros((32,))

    async def proc_wr_port(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.clock.posedge)
            if self.wr_en.value == T:
                i = self.wr_addr.value.to_uint()
                if i != 0:
                    self.regs[i].next = self.wr_data.value

    async def proc_rd1_port(self):
        """TODO(cjdrake): Write docstring."""
        others = [self.regs[i].changed for i in range(1, NUM)]
        while True:
            await notify(self.rs1_addr.changed, *others)
            try:
                i = self.rs1_addr.next.to_uint()
            except ValueError:
                self.rs1_data.next = xes((32,))
            else:
                self.rs1_data.next = self.regs[i].next

    async def proc_rd2_port(self):
        """TODO(cjdrake): Write docstring."""
        others = [self.regs[i].changed for i in range(1, NUM)]
        while True:
            await notify(self.rs2_addr.changed, *others)
            try:
                i = self.rs2_addr.next.to_uint()
            except ValueError:
                self.rs2_data.next = xes((32,))
            else:
                self.rs2_data.next = self.regs[i].next
