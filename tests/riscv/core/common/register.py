"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, resume
from seqlogic.lbool import ones, zeros
from seqlogic.sim import always_ff


class Register(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, shape: tuple[int, ...], init):
        super().__init__(name, parent)

        # Parameters
        self._shape = shape
        self._init = init

        self.build()

    def build(self):
        self.q = Bits(name="q", parent=self, shape=self._shape)
        self.en = Bit(name="en", parent=self)
        self.d = Bits(name="d", parent=self, shape=self._shape)
        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

    @always_ff
    async def p_f_0(self):
        def f():
            return (
                self.clock.is_posedge()
                and self.reset.value == zeros(1)  # noqa
                and self.en.value == ones(1)  # noqa
            )

        while True:
            state = await resume((self.reset, self.reset.is_posedge), (self.clock, f))
            match state:
                case self.reset:
                    self.q.next = self._init
                case self.clock:
                    self.q.next = self.d.value
                case _:
                    assert False
