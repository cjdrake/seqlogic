"""TODO(cjdrake): Write docstring."""

from seqlogic import Module
from seqlogic.var import Bit, LogicVec

from ..common.data_memory_interface import DataMemoryInterface
from .ctl_path import CtlPath
from .data_path import DataPath


class Core(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.bus_addr = LogicVec(name="bus_addr", parent=self, shape=(32,))
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = LogicVec(name="bus_wr_be", parent=self, shape=(4,))
        self.bus_wr_data = LogicVec(name="bus_wr_data", parent=self, shape=(32,))
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = LogicVec(name="bus_rd_data", parent=self, shape=(32,))

        self.pc = LogicVec(name="pc", parent=self, shape=(32,))
        self.inst = LogicVec(name="inst", parent=self, shape=(32,))

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # State
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bit(name="alu_op_a_sel", parent=self)
        self.alu_op_b_sel = Bit(name="alu_op_b_sel", parent=self)
        self.reg_writeback_sel = LogicVec(name="reg_writeback_sel", parent=self, shape=(3,))
        self.inst_opcode = LogicVec(name="inst_opcode", parent=self, shape=(7,))
        self.inst_funct3 = LogicVec(name="inst_funct3", parent=self, shape=(3,))
        self.inst_funct7 = LogicVec(name="inst_funct7", parent=self, shape=(7,))
        self.next_pc_sel = LogicVec(name="next_pc_sel", parent=self, shape=(2,))
        self.alu_function = LogicVec(name="alu_function", parent=self, shape=(5,))
        self.alu_result_equal_zero = Bit(name="alu_result_equal_zero", parent=self)
        self.addr = LogicVec(name="addr", parent=self, shape=(32,))
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_data = LogicVec(name="wr_data", parent=self, shape=(32,))
        self.rd_en = Bit(name="rd_en", parent=self)
        self.rd_data = LogicVec(name="rd_data", parent=self, shape=(32,))

        # Submodules
        self.ctlpath = CtlPath(name="ctlpath", parent=self)
        self.connect(self.ctlpath.inst_opcode, self.inst_opcode)
        self.connect(self.ctlpath.inst_funct3, self.inst_funct3)
        self.connect(self.ctlpath.inst_funct7, self.inst_funct7)
        self.connect(self.ctlpath.alu_result_equal_zero, self.alu_result_equal_zero)
        self.connect(self.pc_wr_en, self.ctlpath.pc_wr_en)
        self.connect(self.regfile_wr_en, self.ctlpath.regfile_wr_en)
        self.connect(self.alu_op_a_sel, self.ctlpath.alu_op_a_sel)
        self.connect(self.alu_op_b_sel, self.ctlpath.alu_op_b_sel)
        self.connect(self.rd_en, self.ctlpath.data_mem_rd_en)
        self.connect(self.wr_en, self.ctlpath.data_mem_wr_en)
        self.connect(self.reg_writeback_sel, self.ctlpath.reg_writeback_sel)
        self.connect(self.alu_function, self.ctlpath.alu_function)
        self.connect(self.next_pc_sel, self.ctlpath.next_pc_sel)

        self.datapath = DataPath(name="datapath", parent=self)
        self.connect(self.addr, self.datapath.data_mem_addr)
        self.connect(self.wr_data, self.datapath.data_mem_wr_data)
        self.connect(self.datapath.data_mem_rd_data, self.rd_data)
        self.connect(self.datapath.inst, self.inst)
        self.connect(self.pc, self.datapath.pc)
        self.connect(self.inst_opcode, self.datapath.inst_opcode)
        self.connect(self.inst_funct3, self.datapath.inst_funct3)
        self.connect(self.inst_funct7, self.datapath.inst_funct7)
        self.connect(self.alu_result_equal_zero, self.datapath.alu_result_equal_zero)
        self.connect(self.datapath.pc_wr_en, self.pc_wr_en)
        self.connect(self.datapath.regfile_wr_en, self.regfile_wr_en)
        self.connect(self.datapath.alu_op_a_sel, self.alu_op_a_sel)
        self.connect(self.datapath.alu_op_b_sel, self.alu_op_b_sel)
        self.connect(self.datapath.reg_writeback_sel, self.reg_writeback_sel)
        self.connect(self.datapath.next_pc_sel, self.next_pc_sel)
        self.connect(self.datapath.alu_function, self.alu_function)
        self.connect(self.datapath.clock, self.clock)
        self.connect(self.datapath.reset, self.reset)

        self.data_memory_interface = DataMemoryInterface(name="data_memory_interface", parent=self)
        self.connect(self.data_memory_interface.data_format, self.inst_funct3)
        self.connect(self.data_memory_interface.addr, self.addr)
        self.connect(self.data_memory_interface.wr_en, self.wr_en)
        self.connect(self.data_memory_interface.wr_data, self.wr_data)
        self.connect(self.data_memory_interface.rd_en, self.rd_en)
        self.connect(self.rd_data, self.data_memory_interface.rd_data)
        self.connect(self.bus_addr, self.data_memory_interface.bus_addr)
        self.connect(self.bus_wr_en, self.data_memory_interface.bus_wr_en)
        self.connect(self.bus_wr_be, self.data_memory_interface.bus_wr_be)
        self.connect(self.bus_wr_data, self.data_memory_interface.bus_wr_data)
        self.connect(self.bus_rd_en, self.data_memory_interface.bus_rd_en)
        self.connect(self.data_memory_interface.bus_rd_data, self.bus_rd_data)
