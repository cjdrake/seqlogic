"""TODO(cjdrake): Write docstring."""

from seqlogic.hier import Module
from seqlogic.logicvec import xes
from seqlogic.sim import notify
from seqlogic.var import LogicVec

from ..misc import COMBI
from .constants import TEXT_BASE, TEXT_BITS, TEXT_SIZE
from .text_memory import TextMemory


class TextMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.rd_addr = LogicVec(name="rd_addr", parent=self, shape=(32,))
        self.rd_data = LogicVec(name="rd_data", parent=self, shape=(32,))

        # State
        self.text = LogicVec(name="text", parent=self, shape=(32,))

        # Submodules
        self.text_memory = TextMemory("text_memory", parent=self)
        self.connect(self.text, self.text_memory.rd_data)

        # Processes
        self._procs.add((self.proc_rd_data, COMBI))
        self._procs.add((self.proc_text_memory_rd_addr, COMBI))

    # output logic [31:0] rd_data
    async def proc_rd_data(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rd_addr.changed, self.text.changed)
            addr = self.rd_addr.next.to_uint()
            is_text = TEXT_BASE <= addr < (TEXT_BASE + TEXT_SIZE)
            if is_text:
                self.rd_data.next = self.text.next
            else:
                self.rd_data.next = xes((32,))

    # text_memory.rd_addr(rd_addr[...])
    async def proc_text_memory_rd_addr(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rd_addr.changed)
            self.text_memory.rd_addr.next = self.rd_addr.next[2:TEXT_BITS]
