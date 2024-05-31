"""Control Path."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Module
from seqlogic.lbool import Vec

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


def f_take_branch(inst_funct3: Funct3, alu_result_eq_zero: Vec[1]) -> Vec[1]:
    sel = inst_funct3.branch
    match sel:
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
            return Vec[1].xprop(sel)


def f_func(inst_funct3: Funct3):
    sel = inst_funct3.alu_logic
    match sel:
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
            default_func = AluOp.xprop(sel)

    match sel:
        case Funct3AluLogic.ADD_SUB:
            secondary_func = AluOp.SUB
        case Funct3AluLogic.SHIFTR:
            secondary_func = AluOp.SRA
        case _:
            secondary_func = AluOp.xprop(sel)

    sel = inst_funct3.branch
    match sel:
        case Funct3Branch.EQ | Funct3Branch.NE:
            branch_func = AluOp.SEQ
        case Funct3Branch.LT | Funct3Branch.GE:
            branch_func = AluOp.SLT
        case Funct3Branch.LTU | Funct3Branch.GEU:
            branch_func = AluOp.SLTU
        case _:
            branch_func = AluOp.xprop(sel)

    return (default_func, secondary_func, branch_func)


def f_alu_func(
    alu_op_type: CtlAlu,
    inst_funct3: Funct3,
    inst_funct7: Vec[7],
    default_func: AluOp,
    secondary_func: AluOp,
    branch_func: AluOp,
):
    sel = alu_op_type
    match sel:
        case CtlAlu.ADD:
            return AluOp.ADD
        case CtlAlu.BRANCH:
            return branch_func
        case CtlAlu.OP:
            sel = inst_funct7[5]
            match sel:
                case "1b0":
                    return default_func
                case "1b1":
                    return secondary_func
                case _:
                    return AluOp.xprop(sel)
        case CtlAlu.OP_IMM:
            a = inst_funct7[5]
            b = inst_funct3.alu_logic.eq(Funct3AluLogic.SLL)
            c = inst_funct3.alu_logic.eq(Funct3AluLogic.SHIFTR)
            sel = a & (b | c)
            match sel:
                case "1b0":
                    return default_func
                case "1b1":
                    return secondary_func
                case _:
                    return AluOp.xprop(sel)
        case _:
            return AluOp.xprop(sel)


def f_next_pc_sel(inst_opcode: Opcode, take_branch: Vec[1]):
    sel = inst_opcode
    match sel:
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
            sel = take_branch
            match sel:
                case "1b0":
                    return CtlPc.PC4
                case "1b1":
                    return CtlPc.PC_IMM
                case _:
                    return CtlPc.xprop(sel)
        case Opcode.JALR:
            return CtlPc.RS1_IMM
        case Opcode.JAL:
            return CtlPc.PC_IMM
        case _:
            return CtlPc.xprop(sel)


def f_ctl(inst_opcode: Opcode):
    sel = inst_opcode
    match sel:
        case Opcode.LOAD:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b1"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.DATA
            alu_op_type = CtlAlu.ADD
        case Opcode.MISC_MEM:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b0"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.RS2
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.ADD
        case Opcode.OP_IMM:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.OP_IMM
        case Opcode.AUIPC:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.PC
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.ADD
        case Opcode.STORE:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b0"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b1"
            reg_writeback_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.ADD
        case Opcode.OP:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.RS2
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.OP
        case Opcode.LUI:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.RS2
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.IMM
            alu_op_type = CtlAlu.ADD
        case Opcode.BRANCH:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b0"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.RS2
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.ALU
            alu_op_type = CtlAlu.BRANCH
        case Opcode.JALR:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.RS1
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.PC4
            alu_op_type = CtlAlu.ADD
        case Opcode.JAL:
            pc_wr_en = "1b1"
            regfile_wr_en = "1b1"
            alu_op_a_sel = CtlAluA.PC
            alu_op_b_sel = CtlAluB.IMM
            data_mem_rd_en = "1b0"
            data_mem_wr_en = "1b0"
            reg_writeback_sel = CtlWriteBack.PC4
            alu_op_type = CtlAlu.ADD
        case _:
            pc_wr_en = Vec[1].xprop(sel)
            regfile_wr_en = Vec[1].xprop(sel)
            alu_op_a_sel = CtlAluA.xprop(sel)
            alu_op_b_sel = CtlAluB.xprop(sel)
            data_mem_rd_en = Vec[1].xprop(sel)
            data_mem_wr_en = Vec[1].xprop(sel)
            reg_writeback_sel = CtlWriteBack.xprop(sel)
            alu_op_type = CtlAlu.xprop(sel)
    return (
        pc_wr_en,
        regfile_wr_en,
        alu_op_a_sel,
        alu_op_b_sel,
        data_mem_rd_en,
        data_mem_wr_en,
        reg_writeback_sel,
        alu_op_type,
    )


class CtlPath(Module):
    """Control Path Module."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        inst_opcode = self.bits(name="inst_opcode", dtype=Opcode, port=True)
        inst_funct3 = self.bits(name="inst_funct3", dtype=Funct3, port=True)
        inst_funct7 = self.bits(name="inst_funct7", dtype=Vec[7], port=True)
        alu_result_eq_zero = self.bit(name="alu_result_eq_zero", port=True)
        pc_wr_en = self.bit(name="pc_wr_en", port=True)
        regfile_wr_en = self.bit(name="regfile_wr_en", port=True)
        alu_op_a_sel = self.bits(name="alu_op_a_sel", dtype=CtlAluA, port=True)
        alu_op_b_sel = self.bits(name="alu_op_b_sel", dtype=CtlAluB, port=True)
        data_mem_rd_en = self.bit(name="data_mem_rd_en", port=True)
        data_mem_wr_en = self.bit(name="data_mem_wr_en", port=True)
        reg_writeback_sel = self.bits(name="reg_writeback_sel", dtype=CtlWriteBack, port=True)
        alu_func = self.bits(name="alu_func", dtype=AluOp, port=True)
        next_pc_sel = self.bits(name="next_pc_sel", dtype=CtlPc, port=True)

        # State
        take_branch = self.bit(name="take_branch")
        alu_op_type = self.bits(name="alu_op_type", dtype=CtlAlu)
        default_func = self.bits(name="default_func", dtype=AluOp)
        secondary_func = self.bits(name="secondary_func", dtype=AluOp)
        branch_func = self.bits(name="branch_func", dtype=AluOp)

        # Combinational Logic
        self.combi(take_branch, f_take_branch, inst_funct3, alu_result_eq_zero)

        ys = (default_func, secondary_func, branch_func)
        self.combi(ys, f_func, inst_funct3)

        xs = (alu_op_type, inst_funct3, inst_funct7, default_func, secondary_func, branch_func)
        self.combi(alu_func, f_alu_func, *xs)

        self.combi(next_pc_sel, f_next_pc_sel, inst_opcode, take_branch)

        ys = (
            pc_wr_en,
            regfile_wr_en,
            alu_op_a_sel,
            alu_op_b_sel,
            data_mem_rd_en,
            data_mem_wr_en,
            reg_writeback_sel,
            alu_op_type,
        )
        self.combi(ys, f_ctl, inst_opcode)
