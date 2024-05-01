"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import zeros
from seqlogic.sim import always_comb

from .. import AluOp


class Alu(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        self.build()

    def build(self):
        # Ports
        self.result = Bits(name="result", parent=self, shape=(32,))
        self.result_equal_zero = Bit(name="result_equal_zero", parent=self)
        self.alu_function = Bits(name="alu_function", parent=self, shape=(5,))
        self.op_a = Bits(name="op_a", parent=self, shape=(32,))
        self.op_b = Bits(name="op_b", parent=self, shape=(32,))

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.alu_function, self.op_a, self.op_b)
            match self.alu_function.next:
                case AluOp.ADD:
                    self.result.next = self.op_a.next + self.op_b.next
                case AluOp.SUB:
                    self.result.next = self.op_a.next - self.op_b.next
                case AluOp.SLL:
                    self.result.next = self.op_a.next << self.op_b.next[0:5]
                case AluOp.SRL:
                    self.result.next = self.op_a.next >> self.op_b.next[0:5]
                case AluOp.SRA:
                    self.result.next, _ = self.op_a.next.arsh(self.op_b.next[0:5])
                case AluOp.SEQ:
                    self.result.next = self.op_a.next.eq(self.op_b.next).zext(32 - 1)
                case AluOp.SLT:
                    self.result.next = self.op_a.next.lt(self.op_b.next).zext(32 - 1)
                case AluOp.SLTU:
                    self.result.next = self.op_a.next.ltu(self.op_b.next).zext(32 - 1)
                case AluOp.XOR:
                    self.result.next = self.op_a.next ^ self.op_b.next
                case AluOp.OR:
                    self.result.next = self.op_a.next | self.op_b.next
                case AluOp.AND:
                    self.result.next = self.op_a.next & self.op_b.next
                case _:
                    self.result.next = zeros(32)

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.result)
            self.result_equal_zero.next = self.result.next.eq(zeros(32))
