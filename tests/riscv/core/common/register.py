"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, resume
from seqlogic.lbool import ones
from seqlogic.sim import always_ff


class Register(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, init):
        super().__init__(name, parent)

        # Parameters
        self._init = init

        self.build()

    def build(self):
        dtype = type(self._init)
        self.q = Bits(name="q", parent=self, dtype=dtype)
        self.en = Bit(name="en", parent=self)
        self.d = Bits(name="d", parent=self, dtype=dtype)
        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

    @always_ff
    async def p_f_0(self):
        def f():
            return self.reset.is_neg() and self.clock.is_posedge() and self.en.value == ones(1)

        while True:
            state = await resume((self.reset, self.reset.is_posedge), (self.clock, f))
            match state:
                case self.reset:
                    self.q.next = self._init
                case self.clock:
                    self.q.next = self.d.value
                case _:
                    assert False
