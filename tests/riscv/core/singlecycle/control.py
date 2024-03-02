"""TODO(cjdrake): Write docstring."""

from seqlogic import Module
from seqlogic.logicvec import F, T, vec
from seqlogic.sim import notify
from seqlogic.var import Bit, Bits

from ..common.constants import CtlAlu, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Opcode
from ..misc import COMBI


class Control(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bit(name="alu_op_a_sel", parent=self)
        self.alu_op_b_sel = Bit(name="alu_op_b_sel", parent=self)
        self.alu_op_type = Bits(name="alu_op_type", parent=self, shape=(2,))
        self.data_mem_rd_en = Bit(name="data_mem_rd_en", parent=self)
        self.data_mem_wr_en = Bit(name="data_mem_wr_en", parent=self)
        self.reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, shape=(3,))
        self.next_pc_sel = Bits(name="next_pc_sel", parent=self, shape=(2,))
        self.inst_opcode = Bits(name="inst_opcode", parent=self, shape=(7,))
        self.take_branch = Bit(name="take_branch", parent=self)

        # Processes
        self._procs.add((self.proc_next_pc_sel, COMBI))
        self._procs.add((self.proc_others, COMBI))

    async def proc_next_pc_sel(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.inst_opcode.changed, self.take_branch.changed)
            match self.inst_opcode.next:
                case Opcode.BRANCH:
                    if self.take_branch.next == T:
                        self.next_pc_sel.next = CtlPc.PC_IMM
                    else:
                        self.next_pc_sel.next = CtlPc.PC4
                case Opcode.JALR:
                    self.next_pc_sel.next = CtlPc.RS1_IMM
                case Opcode.JAL:
                    self.next_pc_sel.next = CtlPc.PC_IMM
                case _:
                    self.next_pc_sel.next = CtlPc.PC4

    async def proc_others(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.inst_opcode.changed)

            self.pc_wr_en.next = T
            self.regfile_wr_en.next = F
            self.alu_op_a_sel.next = F
            self.alu_op_b_sel.next = F
            self.alu_op_type.next = vec("2b00")
            self.data_mem_rd_en.next = F
            self.data_mem_wr_en.next = F
            self.reg_writeback_sel.next = vec("3b000")

            match self.inst_opcode.next:
                case Opcode.LOAD:
                    self.regfile_wr_en.next = T
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = T
                    self.reg_writeback_sel.next = CtlWriteBack.DATA
                case Opcode.MISC_MEM:
                    pass
                case Opcode.OP_IMM:
                    self.regfile_wr_en.next = T
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.OP_IMM
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.AUIPC:
                    self.regfile_wr_en.next = T
                    self.alu_op_a_sel.next = CtlAluA.PC
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.STORE:
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_wr_en.next = T
                case Opcode.OP:
                    self.regfile_wr_en.next = T
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                    self.alu_op_type.next = CtlAlu.OP
                case Opcode.LUI:
                    self.regfile_wr_en.next = T
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.reg_writeback_sel.next = CtlWriteBack.IMM
                case Opcode.BRANCH:
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.BRANCH
                case Opcode.JALR:
                    self.regfile_wr_en.next = T
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.reg_writeback_sel.next = CtlWriteBack.PC4
                case Opcode.JAL:
                    self.regfile_wr_en.next = T
                    self.alu_op_a_sel.next = CtlAluA.PC
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.reg_writeback_sel.next = CtlWriteBack.PC4
                case _:
                    self.pc_wr_en.next = F
                    self.regfile_wr_en.next = F
                    self.data_mem_rd_en.next = F
                    self.data_mem_wr_en.next = F
