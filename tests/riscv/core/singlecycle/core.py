"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module

from ..common.data_memory_interface import DataMemoryInterface
from .ctl_path import CtlPath
from .data_path import DataPath


class Core(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.bus_addr = Bits(name="bus_addr", parent=self, shape=(32,))
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = Bits(name="bus_wr_be", parent=self, shape=(4,))
        self.bus_wr_data = Bits(name="bus_wr_data", parent=self, shape=(32,))
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = Bits(name="bus_rd_data", parent=self, shape=(32,))

        self.pc = Bits(name="pc", parent=self, shape=(32,))
        self.inst = Bits(name="inst", parent=self, shape=(32,))

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # State
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bit(name="alu_op_a_sel", parent=self)
        self.alu_op_b_sel = Bit(name="alu_op_b_sel", parent=self)
        self.reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, shape=(3,))
        self.inst_opcode = Bits(name="inst_opcode", parent=self, shape=(7,))
        self.inst_funct3 = Bits(name="inst_funct3", parent=self, shape=(3,))
        self.inst_funct7 = Bits(name="inst_funct7", parent=self, shape=(7,))
        self.next_pc_sel = Bits(name="next_pc_sel", parent=self, shape=(2,))
        self.alu_function = Bits(name="alu_function", parent=self, shape=(5,))
        self.alu_result_equal_zero = Bit(name="alu_result_equal_zero", parent=self)
        self.addr = Bits(name="addr", parent=self, shape=(32,))
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_data = Bits(name="wr_data", parent=self, shape=(32,))
        self.rd_en = Bit(name="rd_en", parent=self)
        self.rd_data = Bits(name="rd_data", parent=self, shape=(32,))

        # Submodules
        self.ctlpath = CtlPath(name="ctlpath", parent=self)
        self.datapath = DataPath(name="datapath", parent=self)
        self.data_memory_interface = DataMemoryInterface(name="data_memory_interface", parent=self)

    def connect(self):
        """TODO(cjdrake): Write docstring."""
        self.ctlpath.inst_opcode.connect(self.inst_opcode)
        self.ctlpath.inst_funct3.connect(self.inst_funct3)
        self.ctlpath.inst_funct7.connect(self.inst_funct7)
        self.ctlpath.alu_result_equal_zero.connect(self.alu_result_equal_zero)
        self.pc_wr_en.connect(self.ctlpath.pc_wr_en)
        self.regfile_wr_en.connect(self.ctlpath.regfile_wr_en)
        self.alu_op_a_sel.connect(self.ctlpath.alu_op_a_sel)
        self.alu_op_b_sel.connect(self.ctlpath.alu_op_b_sel)
        self.rd_en.connect(self.ctlpath.data_mem_rd_en)
        self.wr_en.connect(self.ctlpath.data_mem_wr_en)
        self.reg_writeback_sel.connect(self.ctlpath.reg_writeback_sel)
        self.alu_function.connect(self.ctlpath.alu_function)
        self.next_pc_sel.connect(self.ctlpath.next_pc_sel)

        self.addr.connect(self.datapath.data_mem_addr)
        self.wr_data.connect(self.datapath.data_mem_wr_data)
        self.datapath.data_mem_rd_data.connect(self.rd_data)
        self.datapath.inst.connect(self.inst)
        self.pc.connect(self.datapath.pc)
        self.inst_opcode.connect(self.datapath.inst_opcode)
        self.inst_funct3.connect(self.datapath.inst_funct3)
        self.inst_funct7.connect(self.datapath.inst_funct7)
        self.alu_result_equal_zero.connect(self.datapath.alu_result_equal_zero)
        self.datapath.pc_wr_en.connect(self.pc_wr_en)
        self.datapath.regfile_wr_en.connect(self.regfile_wr_en)
        self.datapath.alu_op_a_sel.connect(self.alu_op_a_sel)
        self.datapath.alu_op_b_sel.connect(self.alu_op_b_sel)
        self.datapath.reg_writeback_sel.connect(self.reg_writeback_sel)
        self.datapath.next_pc_sel.connect(self.next_pc_sel)
        self.datapath.alu_function.connect(self.alu_function)
        self.datapath.clock.connect(self.clock)
        self.datapath.reset.connect(self.reset)

        self.data_memory_interface.data_format.connect(self.inst_funct3)
        self.data_memory_interface.addr.connect(self.addr)
        self.data_memory_interface.wr_en.connect(self.wr_en)
        self.data_memory_interface.wr_data.connect(self.wr_data)
        self.data_memory_interface.rd_en.connect(self.rd_en)
        self.rd_data.connect(self.data_memory_interface.rd_data)
        self.bus_addr.connect(self.data_memory_interface.bus_addr)
        self.bus_wr_en.connect(self.data_memory_interface.bus_wr_en)
        self.bus_wr_be.connect(self.data_memory_interface.bus_wr_be)
        self.bus_wr_data.connect(self.data_memory_interface.bus_wr_data)
        self.bus_rd_en.connect(self.data_memory_interface.bus_rd_en)
        self.data_memory_interface.bus_rd_data.connect(self.bus_rd_data)
