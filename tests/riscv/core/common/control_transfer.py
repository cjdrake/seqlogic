"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, notify
from seqlogic.bits import X
from seqlogic.sim import always_comb

from .constants import Funct3Branch


class ControlTransfer(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.take_branch = Bit(name="take_branch", parent=self)
        self.inst_funct3 = Bits(name="inst_funct3", parent=self, shape=(3,))
        self.result_equal_zero = Bit(name="result_equal_zero", parent=self)

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.inst_funct3.changed, self.result_equal_zero.changed)
            match self.inst_funct3.next:
                case Funct3Branch.EQ:
                    self.take_branch.next = ~(self.result_equal_zero.next)
                case Funct3Branch.NE:
                    self.take_branch.next = self.result_equal_zero.next
                case Funct3Branch.LT:
                    self.take_branch.next = ~(self.result_equal_zero.next)
                case Funct3Branch.GE:
                    self.take_branch.next = self.result_equal_zero.next
                case Funct3Branch.LTU:
                    self.take_branch.next = ~(self.result_equal_zero.next)
                case Funct3Branch.GEU:
                    self.take_branch.next = self.result_equal_zero.next
                case _:
                    self.take_branch.next = X
