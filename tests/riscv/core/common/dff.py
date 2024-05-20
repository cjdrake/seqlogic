"""D Flip Flops."""

from seqlogic import Bit, Bits, Module, resume
from seqlogic.sim import active


class DffEnAr(Module):
    """D Flip Flop with Enable and Async Reset."""

    def __init__(self, name: str, parent: Module | None, rst_val):
        super().__init__(name, parent)
        self._rst_val = rst_val
        self.build()

    def build(self):
        self.q = Bits(name="q", parent=self, dtype=type(self._rst_val))
        self.en = Bit(name="en", parent=self)
        self.d = Bits(name="d", parent=self, dtype=type(self._rst_val))
        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

    @active
    async def p_f_0(self):
        def f():
            return self.reset.is_neg() and self.clock.is_posedge() and self.en.value == "1b1"

        while True:
            state = await resume((self.reset, self.reset.is_posedge), (self.clock, f))
            match state:
                case self.reset:
                    self.q.next = self._rst_val
                case self.clock:
                    self.q.next = self.d.value
                case _:
                    assert False
