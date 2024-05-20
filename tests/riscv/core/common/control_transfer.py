"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec
from seqlogic.sim import reactive

from .. import Funct3, Funct3Branch


class ControlTransfer(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self.build()

    def build(self):
        # Ports
        self.take_branch = Bit(name="take_branch", parent=self)
        self.inst_funct3 = Bits(name="inst_funct3", parent=self, dtype=Funct3)
        self.result_equal_zero = Bit(name="result_equal_zero", parent=self)

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.inst_funct3, self.result_equal_zero)
            match self.inst_funct3.value.branch:
                case Funct3Branch.EQ:
                    self.take_branch.next = ~(self.result_equal_zero.value)
                case Funct3Branch.NE:
                    self.take_branch.next = self.result_equal_zero.value
                case Funct3Branch.LT:
                    self.take_branch.next = ~(self.result_equal_zero.value)
                case Funct3Branch.GE:
                    self.take_branch.next = self.result_equal_zero.value
                case Funct3Branch.LTU:
                    self.take_branch.next = ~(self.result_equal_zero.value)
                case Funct3Branch.GEU:
                    self.take_branch.next = self.result_equal_zero.value
                case _:
                    self.take_branch.next = Vec[1].dcs()
