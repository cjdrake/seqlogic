"""TODO(cjdrake): Add docstring."""

from seqlogic import Bits, Module
from seqlogic.sim import notify

from ..misc import COMBI


class InstructionDecoder(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.inst_funct7 = Bits(name="inst_funct7", parent=self, shape=(7,))
        self.inst_rs2 = Bits(name="inst_rs2", parent=self, shape=(5,))
        self.inst_rs1 = Bits(name="inst_rs1", parent=self, shape=(5,))
        self.inst_funct3 = Bits(name="inst_funct3", parent=self, shape=(3,))
        self.inst_rd = Bits(name="inst_rd", parent=self, shape=(5,))
        self.inst_opcode = Bits(name="inst_opcode", parent=self, shape=(7,))
        self.inst = Bits(name="inst", parent=self, shape=(32,))

        # Processes
        self._procs.add((self.proc_out, COMBI))

    async def proc_out(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.inst.changed)
            self.inst_funct7.next = self.inst.next[25:32]
            self.inst_rs2.next = self.inst.next[20:25]
            self.inst_rs1.next = self.inst.next[15:20]
            self.inst_funct3.next = self.inst.next[12:15]
            self.inst_rd.next = self.inst.next[7:12]
            self.inst_opcode.next = self.inst.next[0:7]
