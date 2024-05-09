"""TODO(cjdrake): Write docstring."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import dcs, ones, zeros
from seqlogic.sim import always_comb

from .. import CtlAlu, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Opcode


class Control(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self.build()

    def build(self):
        # Ports
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bits(name="alu_op_a_sel", parent=self, dtype=CtlAluA)
        self.alu_op_b_sel = Bits(name="alu_op_b_sel", parent=self, dtype=CtlAluB)
        self.alu_op_type = Bits(name="alu_op_type", parent=self, dtype=CtlAlu)
        self.data_mem_rd_en = Bit(name="data_mem_rd_en", parent=self)
        self.data_mem_wr_en = Bit(name="data_mem_wr_en", parent=self)
        self.reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, dtype=CtlWriteBack)
        self.next_pc_sel = Bits(name="next_pc_sel", parent=self, dtype=CtlPc)
        self.inst_opcode = Bits(name="inst_opcode", parent=self, dtype=Opcode)
        self.take_branch = Bit(name="take_branch", parent=self)

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.inst_opcode, self.take_branch)
            match self.inst_opcode.value:
                case (
                    Opcode.LOAD  # noqa
                    | Opcode.MISC_MEM  # noqa
                    | Opcode.OP_IMM  # noqa
                    | Opcode.AUIPC  # noqa
                    | Opcode.STORE  # noqa
                    | Opcode.OP  # noqa
                    | Opcode.LUI  # noqa
                ):
                    self.next_pc_sel.next = CtlPc.PC4
                case Opcode.BRANCH:
                    s = self.take_branch.value
                    self.next_pc_sel.next = s.ite(CtlPc.PC_IMM, CtlPc.PC4)
                case Opcode.JALR:
                    self.next_pc_sel.next = CtlPc.RS1_IMM
                case Opcode.JAL:
                    self.next_pc_sel.next = CtlPc.PC_IMM
                case _:
                    self.next_pc_sel.next = CtlPc.DC

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.inst_opcode)
            match self.inst_opcode.value:
                case Opcode.LOAD:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = ones(1)
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = ones(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.DATA
                case Opcode.MISC_MEM:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = zeros(1)
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.OP_IMM:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = ones(1)
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.OP_IMM
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.AUIPC:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = ones(1)
                    self.alu_op_a_sel.next = CtlAluA.PC
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.STORE:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = zeros(1)
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = ones(1)
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.OP:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = ones(1)
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.OP
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.LUI:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = ones(1)
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.IMM
                case Opcode.BRANCH:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = zeros(1)
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.BRANCH
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.JALR:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = ones(1)
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.PC4
                case Opcode.JAL:
                    self.pc_wr_en.next = ones(1)
                    self.regfile_wr_en.next = ones(1)
                    self.alu_op_a_sel.next = CtlAluA.PC
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = zeros(1)
                    self.data_mem_wr_en.next = zeros(1)
                    self.reg_writeback_sel.next = CtlWriteBack.PC4
                case _:
                    self.pc_wr_en.next = dcs(1)
                    self.regfile_wr_en.next = dcs(1)
                    self.alu_op_a_sel.next = CtlAluA.DC
                    self.alu_op_b_sel.next = CtlAluB.DC
                    self.alu_op_type.next = CtlAlu.DC
                    self.data_mem_rd_en.next = dcs(1)
                    self.data_mem_wr_en.next = dcs(1)
                    self.reg_writeback_sel.next = CtlWriteBack.DC
