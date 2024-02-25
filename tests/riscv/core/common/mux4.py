"""TODO(cjdrake): Write docstring."""

from seqlogic import Module
from seqlogic.logicvec import vec, xes
from seqlogic.sim import notify
from seqlogic.util import clog2
from seqlogic.var import LogicVec

from ..misc import COMBI


class Mux4(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self._width = width

        # Ports
        self.out = LogicVec(name="out", parent=self, shape=(width,))
        self.sel = LogicVec(name="sel", parent=self, shape=(clog2(4),))
        self.in0 = LogicVec(name="in0", parent=self, shape=(width,))
        self.in1 = LogicVec(name="in1", parent=self, shape=(width,))
        self.in2 = LogicVec(name="in2", parent=self, shape=(width,))
        self.in3 = LogicVec(name="in3", parent=self, shape=(width,))

        # Processes
        self._procs.add((self.proc_out, COMBI))

    async def proc_out(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(
                self.sel.changed,
                self.in0.changed,
                self.in1.changed,
                self.in2.changed,
                self.in3.changed,
            )
            if self.sel.next == vec("2b00"):
                self.out.next = self.in0.next
            elif self.sel.next == vec("2b01"):
                self.out.next = self.in1.next
            elif self.sel.next == vec("2b10"):
                self.out.next = self.in2.next
            elif self.sel.next == vec("2b11"):
                self.out.next = self.in3.next
            else:
                self.out.next = xes((self._width,))
