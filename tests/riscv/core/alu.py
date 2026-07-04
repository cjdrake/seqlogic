"""Arithmetic Logic Unit (ALU)."""

from bvwx import Array, add, eq, lt, slt, srsh, sub, xt

from seqlogic import Module

from . import AluOp


def f(op: AluOp, a: Array[32], b: Array[32]) -> Array[32]:
    match op:
        case AluOp.ADD:
            return add(a, b)
        case AluOp.SUB:
            return sub(a, b)
        case AluOp.SLL:
            return a << b[0:5]
        case AluOp.SRL:
            return a >> b[0:5]
        case AluOp.SRA:
            return srsh(a, b[0:5])
        case AluOp.SEQ:
            return xt(eq(a, b), 32 - 1)
        case AluOp.SLT:
            return xt(slt(a, b), 32 - 1)
        case AluOp.SLTU:
            return xt(lt(a, b), 32 - 1)
        case AluOp.XOR:
            return a ^ b
        case AluOp.OR:
            return a | b
        case AluOp.AND:
            return a & b
        case _:
            return Array[32].xprop(op)


class Alu(Module):
    """Arithmetic Logic Unit (ALU)."""

    def build(self):
        # Ports
        y = self.output(name="y", dtype=Array[32])
        op = self.input(name="op", dtype=AluOp)
        a = self.input(name="a", dtype=Array[32])
        b = self.input(name="b", dtype=Array[32])

        # Combinational Logic
        self.combi(y, f, op, a, b)
