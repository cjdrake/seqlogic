"""Control Path."""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

from seqlogic import Module
from seqlogic.vec import Vec

from . import (
    AluOp,
    CtlAlu,
    CtlAluA,
    CtlAluB,
    CtlPc,
    CtlWriteBack,
    Funct3,
    Funct3AluLogic,
    Funct3Branch,
    Opcode,
)


def f_take_branch(funct3: Funct3, alu_result_eq_zero: Vec[1]) -> Vec[1]:
    match funct3.branch:
        case Funct3Branch.EQ:
            return ~alu_result_eq_zero
        case Funct3Branch.NE:
            return alu_result_eq_zero
        case Funct3Branch.LT:
            return ~alu_result_eq_zero
        case Funct3Branch.GE:
            return alu_result_eq_zero
        case Funct3Branch.LTU:
            return ~alu_result_eq_zero
        case Funct3Branch.GEU:
            return alu_result_eq_zero
        case _:
            return Vec[1].xprop(funct3.branch)


def f_func(funct3: Funct3):
    match funct3.alu_logic:
        case Funct3AluLogic.ADD_SUB:
            default_func = AluOp.ADD
        case Funct3AluLogic.SLL:
            default_func = AluOp.SLL
        case Funct3AluLogic.SLT:
            default_func = AluOp.SLT
        case Funct3AluLogic.SLTU:
            default_func = AluOp.SLTU
        case Funct3AluLogic.XOR:
            default_func = AluOp.XOR
        case Funct3AluLogic.SHIFTR:
            default_func = AluOp.SRL
        case Funct3AluLogic.OR:
            default_func = AluOp.OR
        case Funct3AluLogic.AND:
            default_func = AluOp.AND
        case _:
            default_func = AluOp.xprop(funct3.alu_logic)

    match funct3.alu_logic:
        case Funct3AluLogic.ADD_SUB:
            secondary_func = AluOp.SUB
        case Funct3AluLogic.SHIFTR:
            secondary_func = AluOp.SRA
        case _:
            secondary_func = AluOp.xprop(funct3.alu_logic)

    match funct3.branch:
        case Funct3Branch.EQ | Funct3Branch.NE:
            branch_func = AluOp.SEQ
        case Funct3Branch.LT | Funct3Branch.GE:
            branch_func = AluOp.SLT
        case Funct3Branch.LTU | Funct3Branch.GEU:
            branch_func = AluOp.SLTU
        case _:
            branch_func = AluOp.xprop(funct3.branch)

    return (default_func, secondary_func, branch_func)


def f_alu_op(
    alu_op_type: CtlAlu,
    funct3: Funct3,
    funct7: Vec[7],
    default_func: AluOp,
    secondary_func: AluOp,
    branch_func: AluOp,
):
    match alu_op_type:
        case CtlAlu.ADD:
            return AluOp.ADD
        case CtlAlu.BRANCH:
            return branch_func
        case CtlAlu.OP:
            match funct7[5]:
                case "1b0":
                    return default_func
                case "1b1":
                    return secondary_func
                case _:
                    return AluOp.xprop(funct7[5])
        case CtlAlu.OP_IMM:
            a = funct7[5]
            b = funct3.alu_logic.eq(Funct3AluLogic.SLL)
            c = funct3.alu_logic.eq(Funct3AluLogic.SHIFTR)
            sel = a & (b | c)
            match sel:
                case "1b0":
                    return default_func
                case "1b1":
                    return secondary_func
                case _:
                    return AluOp.xprop(sel)
        case _:
            return AluOp.xprop(alu_op_type)


def f_next_pc_sel(opcode: Opcode, take_branch: Vec[1]):
    match opcode:
        case (
            Opcode.LOAD
            | Opcode.MISC_MEM  # noqa
            | Opcode.OP_IMM  # noqa
            | Opcode.AUIPC  # noqa
            | Opcode.STORE  # noqa
            | Opcode.OP  # noqa
            | Opcode.LUI  # noqa
        ):
            return CtlPc.PC4
        case Opcode.BRANCH:
            match take_branch:
                case "1b0":
                    return CtlPc.PC4
                case "1b1":
                    return CtlPc.PC_IMM
                case _:
                    return CtlPc.xprop(take_branch)
        case Opcode.JALR:
            return CtlPc.RS1_IMM
        case Opcode.JAL:
            return CtlPc.PC_IMM
        case _:
            return CtlPc.xprop(opcode)


def f_ctl(opcode: Opcode):
    match opcode:
        case Opcode.LOAD:
            pc_wr_en = "1b1"
            reg_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b1"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.DATA
            alu_op_type = CtlAlu.ADD
        case Opcode.MISC_MEM:
            pc_wr_en = "1b1"
            reg_wr_en = "1b0"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.RS2
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.ADD
        case Opcode.OP_IMM:
            pc_wr_en = "1b1"
            reg_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.OP_IMM
        case Opcode.AUIPC:
            pc_wr_en = "1b1"
            reg_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.PC
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.ADD
        case Opcode.STORE:
            pc_wr_en = "1b1"
            reg_wr_en = "1b0"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b1"
            reg_wr_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.ADD
        case Opcode.OP:
            pc_wr_en = "1b1"
            reg_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.RS2
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.OP
        case Opcode.LUI:
            pc_wr_en = "1b1"
            reg_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.RS2
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.IMM
            alu_op_type = CtlAlu.ADD
        case Opcode.BRANCH:
            pc_wr_en = "1b1"
            reg_wr_en = "1b0"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.RS2
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.BRANCH
        case Opcode.JALR:
            pc_wr_en = "1b1"
            reg_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.PC4
            alu_op_type = CtlAlu.ADD
        case Opcode.JAL:
            pc_wr_en = "1b1"
            reg_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.PC
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_wr_sel = CtlWriteBack.PC4
            alu_op_type = CtlAlu.ADD
        case _:
            pc_wr_en = Vec[1].xprop(opcode)
            reg_wr_en = Vec[1].xprop(opcode)
            alu_op_a_sel = CtlAluA.xprop(opcode)
            alu_op_b_sel = CtlAluB.xprop(opcode)
            data_mem_rd_en = Vec[1].xprop(opcode)
            data_mem_wr_en = Vec[1].xprop(opcode)
            reg_wr_sel = CtlWriteBack.xprop(opcode)
            alu_op_type = CtlAlu.xprop(opcode)
    return (
        pc_wr_en,
        reg_wr_en,
        alu_op_a_sel,
        alu_op_b_sel,
        data_mem_rd_en,
        data_mem_wr_en,
        reg_wr_sel,
        alu_op_type,
    )


class CtlPath(Module):
    """Control Path Module."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        opcode = self.input(name="opcode", dtype=Opcode)
        funct3 = self.input(name="funct3", dtype=Funct3)
        funct7 = self.input(name="funct7", dtype=Vec[7])
        alu_result_eq_zero = self.input(name="alu_result_eq_zero", dtype=Vec[1])

        pc_wr_en = self.output(name="pc_wr_en", dtype=Vec[1])
        reg_wr_en = self.output(name="reg_wr_en", dtype=Vec[1])
        alu_op_a_sel = self.output(name="alu_op_a_sel", dtype=CtlAluA)
        alu_op_b_sel = self.output(name="alu_op_b_sel", dtype=CtlAluB)
        data_mem_rd_en = self.output(name="data_mem_rd_en", dtype=Vec[1])
        data_mem_wr_en = self.output(name="data_mem_wr_en", dtype=Vec[1])
        reg_wr_sel = self.output(name="reg_wr_sel", dtype=CtlWriteBack)
        alu_op = self.output(name="alu_op", dtype=AluOp)
        next_pc_sel = self.output(name="next_pc_sel", dtype=CtlPc)

        # State
        take_branch = self.bit(name="take_branch")
        alu_op_type = self.bits(name="alu_op_type", dtype=CtlAlu)
        default_func = self.bits(name="default_func", dtype=AluOp)
        secondary_func = self.bits(name="secondary_func", dtype=AluOp)
        branch_func = self.bits(name="branch_func", dtype=AluOp)

        # Combinational Logic
        self.combi(take_branch, f_take_branch, funct3, alu_result_eq_zero)

        ys = (default_func, secondary_func, branch_func)
        self.combi(ys, f_func, funct3)

        xs = (alu_op_type, funct3, funct7, default_func, secondary_func, branch_func)
        self.combi(alu_op, f_alu_op, *xs)

        self.combi(next_pc_sel, f_next_pc_sel, opcode, take_branch)

        ys = (
            pc_wr_en,
            reg_wr_en,
            alu_op_a_sel,
            alu_op_b_sel,
            data_mem_rd_en,
            data_mem_wr_en,
            reg_wr_sel,
            alu_op_type,
        )
        self.combi(ys, f_ctl, opcode)
