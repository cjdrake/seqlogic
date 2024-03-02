"""TODO(cjdrake): Write docstring."""

from seqlogic import Module
from seqlogic.logicvec import F, T, xes
from seqlogic.sim import notify
from seqlogic.var import Bit, LogicVec

from ..misc import COMBI


class Mux2(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self._width = width

        # Ports
        self.out = LogicVec(name="out", parent=self, shape=(width,))
        self.sel = Bit(name="sel", parent=self)
        self.in0 = LogicVec(name="in0", parent=self, shape=(width,))
        self.in1 = LogicVec(name="in1", parent=self, shape=(width,))

        # Processes
        self._procs.add((self.proc_out, COMBI))

    async def proc_out(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.sel.changed, self.in0.changed, self.in1.changed)
            if self.sel.next == F:
                self.out.next = self.in0.next
            elif self.sel.next == T:
                self.out.next = self.in1.next
            else:
                self.out.next = xes((self._width,))
