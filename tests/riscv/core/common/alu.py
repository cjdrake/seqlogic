"""Arithmetic Logic Unit (ALU)."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec, zeros
from seqlogic.sim import reactive

from .. import AluOp


class Alu(Module):
    """Arithmetic Logic Unit (ALU)."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self._build()

    def _build(self):
        # Ports
        self.result = Bits(name="result", parent=self, dtype=Vec[32])
        self.result_equal_zero = Bit(name="result_equal_zero", parent=self)
        self.alu_function = Bits(name="alu_function", parent=self, dtype=AluOp)
        self.op_a = Bits(name="op_a", parent=self, dtype=Vec[32])
        self.op_b = Bits(name="op_b", parent=self, dtype=Vec[32])

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.alu_function, self.op_a, self.op_b)
            sel = self.alu_function.value
            match sel:
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
                    y = self.op_a.value.eq(self.op_b.value)
                    self.result.next = y.zext(32 - 1)
                case AluOp.SLT:
                    y = self.op_a.value.lt(self.op_b.value)
                    self.result.next = y.zext(32 - 1)
                case AluOp.SLTU:
                    y = self.op_a.value.ltu(self.op_b.value)
                    self.result.next = y.zext(32 - 1)
                case AluOp.XOR:
                    self.result.next = self.op_a.value ^ self.op_b.value
                case AluOp.OR:
                    self.result.next = self.op_a.value | self.op_b.value
                case AluOp.AND:
                    self.result.next = self.op_a.value & self.op_b.value
                case _:
                    self.result.xprop(sel)

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.result)
            self.result_equal_zero.next = self.result.value.eq(zeros(32))
