"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, notify
from seqlogic.bits import F, T, logicvec

from ..misc import FLOP


class Register(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int, init: logicvec):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Parameters
        self.init = init

        # Ports
        self.q = Bits(name="q", parent=self, shape=(width,))
        self.en = Bit(name="en", parent=self)
        self.d = Bits(name="d", parent=self, shape=(width,))
        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # Processes
        self._procs.add((self.proc_q, FLOP))

    async def proc_q(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            v = await notify(self.clock.posedge, self.reset.posedge)
            if v is self.reset:
                self.q.next = self.init
            elif self.reset.value == F and self.en.value == T:
                self.q.next = self.d.value
