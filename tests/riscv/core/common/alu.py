"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec, zeros
from seqlogic.sim import always_comb

from .. import AluOp


class Alu(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        self.build()

    def build(self):
        # Ports
        self.result = Bits(name="result", parent=self, dtype=Vec[32])
        self.result_equal_zero = Bit(name="result_equal_zero", parent=self)
        self.alu_function = Bits(name="alu_function", parent=self, dtype=Vec[5])
        self.op_a = Bits(name="op_a", parent=self, dtype=Vec[32])
        self.op_b = Bits(name="op_b", parent=self, dtype=Vec[32])

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.alu_function, self.op_a, self.op_b)
            match self.alu_function.value:
                case AluOp.ADD:
                    self.result.next = self.op_a.value + self.op_b.value
                case AluOp.SUB:
                    self.result.next = self.op_a.value - self.op_b.value
                case AluOp.SLL:
                    self.result.next = self.op_a.value << self.op_b.value[0:5]
                case AluOp.SRL:
                    self.result.next = self.op_a.value >> self.op_b.value[0:5]
                case AluOp.SRA:
                    self.result.next, _ = self.op_a.value.arsh(self.op_b.value[0:5])
                case AluOp.SEQ:
                    self.result.next = self.op_a.value.eq(self.op_b.value).zext(32 - 1)
                case AluOp.SLT:
                    self.result.next = self.op_a.value.lt(self.op_b.value).zext(32 - 1)
                case AluOp.SLTU:
                    self.result.next = self.op_a.value.ltu(self.op_b.value).zext(32 - 1)
                case AluOp.XOR:
                    self.result.next = self.op_a.value ^ self.op_b.value
                case AluOp.OR:
                    self.result.next = self.op_a.value | self.op_b.value
                case AluOp.AND:
                    self.result.next = self.op_a.value & self.op_b.value
                case _:
                    self.result.next = zeros(32)

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.result)
            self.result_equal_zero.next = self.result.value.eq(zeros(32))
