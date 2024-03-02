"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module
from seqlogic.logicvec import vec, xes
from seqlogic.sim import notify
from seqlogic.util import clog2

from ..misc import COMBI


class Mux8(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self._width = width

        # Ports
        self.out = Bits(name="out", parent=self, shape=(width,))
        self.sel = Bits(name="sel", parent=self, shape=(clog2(8),))
        self.in0 = Bits(name="in0", parent=self, shape=(width,))
        self.in1 = Bits(name="in1", parent=self, shape=(width,))
        self.in2 = Bits(name="in2", parent=self, shape=(width,))
        self.in3 = Bits(name="in3", parent=self, shape=(width,))
        self.in4 = Bits(name="in4", parent=self, shape=(width,))
        self.in5 = Bits(name="in5", parent=self, shape=(width,))
        self.in6 = Bits(name="in6", parent=self, shape=(width,))
        self.in7 = Bits(name="in7", parent=self, shape=(width,))

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
                self.in4.changed,
                self.in5.changed,
                self.in6.changed,
                self.in7.changed,
            )
            if self.sel.next == vec("3b000"):
                self.out.next = self.in0.next
            elif self.sel.next == vec("3b001"):
                self.out.next = self.in1.next
            elif self.sel.next == vec("3b010"):
                self.out.next = self.in2.next
            elif self.sel.next == vec("3b011"):
                self.out.next = self.in3.next
            elif self.sel.next == vec("3b100"):
                self.out.next = self.in4.next
            elif self.sel.next == vec("3b101"):
                self.out.next = self.in5.next
            elif self.sel.next == vec("3b110"):
                self.out.next = self.in6.next
            elif self.sel.next == vec("3b111"):
                self.out.next = self.in7.next
            else:
                self.out.next = xes((self._width,))
