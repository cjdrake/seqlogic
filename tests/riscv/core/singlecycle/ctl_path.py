"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module

from ..common.alu_control import AluControl
from ..common.control_transfer import ControlTransfer
from .control import Control


class CtlPath(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.inst_opcode = Bits(name="inst_opcode", parent=self, shape=(7,))
        self.inst_funct3 = Bits(name="inst_funct3", parent=self, shape=(3,))
        self.inst_funct7 = Bits(name="inst_funct7", parent=self, shape=(7,))
        self.alu_result_equal_zero = Bit(name="alu_result_equal_zero", parent=self)
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bit(name="alu_op_a_sel", parent=self)
        self.alu_op_b_sel = Bit(name="alu_op_b_sel", parent=self)
        self.data_mem_rd_en = Bit(name="data_mem_rd_en", parent=self)
        self.data_mem_wr_en = Bit(name="data_mem_wr_en", parent=self)
        self.reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, shape=(3,))
        self.alu_function = Bits(name="alu_function", parent=self, shape=(5,))
        self.next_pc_sel = Bits(name="next_pc_sel", parent=self, shape=(2,))

        # State
        self.take_branch = Bit(name="take_branch", parent=self)
        self.alu_op_type = Bits(name="alu_op_type", parent=self, shape=(2,))

        # Submodules
        self.control = Control(name="control", parent=self)
        self.control_transfer = ControlTransfer(name="control_transfer", parent=self)
        self.alu_control = AluControl(name="alu_control", parent=self)

    def connect(self):
        """TODO(cjdrake): Write docstring."""
        self.pc_wr_en.connect(self.control.pc_wr_en)
        self.regfile_wr_en.connect(self.control.regfile_wr_en)
        self.alu_op_a_sel.connect(self.control.alu_op_a_sel)
        self.alu_op_b_sel.connect(self.control.alu_op_b_sel)
        self.alu_op_type.connect(self.control.alu_op_type)
        self.data_mem_rd_en.connect(self.control.data_mem_rd_en)
        self.data_mem_wr_en.connect(self.control.data_mem_wr_en)
        self.reg_writeback_sel.connect(self.control.reg_writeback_sel)
        self.next_pc_sel.connect(self.control.next_pc_sel)
        self.control.inst_opcode.connect(self.inst_opcode)
        self.control.take_branch.connect(self.take_branch)

        self.take_branch.connect(self.control_transfer.take_branch)
        self.control_transfer.inst_funct3.connect(self.inst_funct3)
        self.control_transfer.result_equal_zero.connect(self.alu_result_equal_zero)

        self.alu_function.connect(self.alu_control.alu_function)
        self.alu_control.alu_op_type.connect(self.alu_op_type)
        self.alu_control.inst_funct3.connect(self.inst_funct3)
        self.alu_control.inst_funct7.connect(self.inst_funct7)
