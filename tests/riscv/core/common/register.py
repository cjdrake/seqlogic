"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, notify
from seqlogic.bits import F, T
from seqlogic.sim import always_ff


class Register(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int, init):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Parameters
        self.width = width
        self.init = init

        self.build()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        self.q = Bits(name="q", parent=self, shape=(self.width,))
        self.en = Bit(name="en", parent=self)
        self.d = Bits(name="d", parent=self, shape=(self.width,))
        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

    @always_ff
    async def p_f_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            v = await notify(self.clock.posedge, self.reset.posedge)
            if v is self.reset:
                self.q.next = self.init
            elif self.reset.value == F and self.en.value == T:
                self.q.next = self.d.value
