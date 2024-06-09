"""Arithmetic Logic Unit (ALU)."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Module
from seqlogic.vec import Vec

from . import AluOp


def alu_result(op: AluOp, a: Vec[32], b: Vec[32]) -> Vec[32]:
    match op:
        case AluOp.ADD:
            return a + b
        case AluOp.SUB:
            return a - b
        case AluOp.SLL:
            return a << b[0:5]
        case AluOp.SRL:
            return a >> b[0:5]
        case AluOp.SRA:
            y, _ = a.arsh(b[0:5])
            return y
        case AluOp.SEQ:
            return a.eq(b).zext(32 - 1)
        case AluOp.SLT:
            return a.lt(b).zext(32 - 1)
        case AluOp.SLTU:
            return a.ltu(b).zext(32 - 1)
        case AluOp.XOR:
            return a ^ b
        case AluOp.OR:
            return a | b
        case AluOp.AND:
            return a & b
        case _:
            return Vec[32].xprop(op)


class Alu(Module):
    """Arithmetic Logic Unit (ALU)."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.bits(name="result", dtype=Vec[32], port=True)
        self.bit(name="result_eq_zero", port=True)
        self.bits(name="alu_func", dtype=AluOp, port=True)
        self.bits(name="op_a", dtype=Vec[32], port=True)
        self.bits(name="op_b", dtype=Vec[32], port=True)

        # Combinational Logic
        self.combi(self._result, alu_result, self._alu_func, self._op_a, self._op_b)
        self.combi(self._result_eq_zero, lambda x: x.eq("32h0000_0000"), self._result)
