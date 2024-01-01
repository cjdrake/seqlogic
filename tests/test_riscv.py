"""
TODO(cjdrake): Write docstring.
"""

from collections import defaultdict

from seqlogic.enum import Enum
from seqlogic.hier import Dict, Module
from seqlogic.logicvec import F, T, X, cat, logicvec, rep, uint2vec, vec, xes
from seqlogic.sim import Region, get_loop, notify, sleep
from seqlogic.var import Logic

loop = get_loop()

waves = defaultdict(dict)

TEXT_BASE = 0x0040_0000
TEXT_BITS = 16
TEXT_SIZE = 2**TEXT_BITS

DATA_BASE = 0x8000_0000
DATA_BITS = 17
DATA_SIZE = 2**DATA_BITS

HW, TASK = 0, 1


class Opcode(Enum):
    """TODO(cjdrake): Write docstring."""

    LOAD = "7b000_0011"
    LOAD_FP = "7b000_0111"
    MISC_MEM = "7b000_1111"
    OP_IMM = "7b001_0011"
    AUIPC = "7b001_0111"
    STORE = "7b010_0011"
    STORE_FP = "7b010_0111"
    OP = "7b011_0011"
    LUI = "7b011_0111"
    OP_FP = "7b101_0011"
    BRANCH = "7b110_0011"
    JALR = "7b110_0111"
    JAL = "7b110_1111"
    SYSTEM = "7b111_0011"


class TraceLogic(Logic):
    """Variable that supports dumping to memory."""

    def __init__(self, name: str, parent: Module, shape: tuple[int, ...]):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, shape)
        waves[self._sim.time()][self] = self._value

    def update(self):
        """TODO(cjdrake): Write docstring."""
        if self.dirty():
            waves[self._sim.time()][self] = self._next_value
        super().update()


class Top(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str):
        super().__init__(name, parent=None)

        # Ports
        self.bus_addr = Logic(name="bus_addr", parent=self, shape=(32,))
        self.bus_wr_en = Logic(name="bus_wr_en", parent=self, shape=(1,))
        self.bus_wr_be = Logic(name="bus_wr_be", parent=self, shape=(4,))
        self.bus_wr_data = Logic(name="bus_wr_data", parent=self, shape=(32,))
        self.bus_rd_en = Logic(name="bus_rd_en", parent=self, shape=(1,))
        self.bus_rd_data = Logic(name="bus_rd_data", parent=self, shape=(32,))

        self.pc = TraceLogic(name="pc", parent=self, shape=(32,))
        self.inst = TraceLogic(name="inst", parent=self, shape=(32,))

        self.clock = TraceLogic(name="clock", parent=self, shape=(1,))
        self.reset = TraceLogic(name="reset", parent=self, shape=(1,))

        # Submodules
        self.text_memory_bus = ExampleTextMemoryBus(name="text_memory_bus", parent=self)
        self.data_memory_bus = ExampleDataMemoryBus(name="data_memory_bus", parent=self)
        self.riscv_core = RiscvCore(name="riscv_core", parent=self)

        # Processes
        self.connect(self.text_memory_bus.rd_addr, self.pc)
        self.connect(self.inst, self.text_memory_bus.rd_data)

        self.connect(self.bus_addr, self.riscv_core.bus_addr)
        self.connect(self.bus_wr_en, self.riscv_core.bus_wr_en)
        self.connect(self.bus_wr_be, self.riscv_core.bus_wr_be)
        self.connect(self.bus_wr_data, self.riscv_core.bus_wr_data)
        self.connect(self.bus_rd_en, self.riscv_core.bus_rd_en)
        self.connect(self.riscv_core.bus_rd_data, self.bus_rd_data)
        self.connect(self.pc, self.riscv_core.pc)
        self.connect(self.riscv_core.inst, self.inst)
        self.connect(self.riscv_core.clock, self.clock)
        self.connect(self.riscv_core.reset, self.reset)

        self._procs.add((self.proc_clock, TASK))
        self._procs.add((self.proc_reset, TASK))

    # input logic clock
    async def proc_clock(self):
        self.clock.next = vec("1b0")
        await sleep(1)
        while True:
            self.clock.next = ~(self.clock.value)
            await sleep(1)
            self.clock.next = ~(self.clock.value)
            await sleep(1)

    # input logic reset
    async def proc_reset(self):
        self.reset.next = vec("1b0")
        await sleep(5)
        self.reset.next = ~(self.reset.value)
        await sleep(5)
        self.reset.next = ~(self.reset.value)


class ExampleTextMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.rd_addr = Logic(name="rd_addr", parent=self, shape=(32,))
        self.rd_data = Logic(name="rd_data", parent=self, shape=(32,))

        # State
        self.text = Logic(name="text", parent=self, shape=(32,))

        # Submodules
        self.text_memory = ExampleTextMemory("text_memory", parent=self)

        # Processes
        self.connect(self.text, self.text_memory.rd_data)

        self._procs.add((self.proc_rd_data, HW))
        self._procs.add((self.proc_text_memory_rd_addr, HW))

    # output logic [31:0] rd_data
    async def proc_rd_data(self):
        while True:
            await notify(self.rd_addr.changed, self.text.changed)
            addr = self.rd_addr.next.to_uint()
            is_text = TEXT_BASE <= addr < (TEXT_BASE + TEXT_SIZE)
            if is_text:
                self.rd_data.next = self.text.next
            else:
                self.rd_data.next = xes((32,))

    # text_memory.rd_addr(rd_addr[...])
    async def proc_text_memory_rd_addr(self):
        while True:
            await notify(self.rd_addr.changed)
            self.text_memory.rd_addr.next = self.rd_addr.next[2:TEXT_BITS]


class ExampleTextMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.rd_addr = Logic("rd_addr", parent=self, shape=(14,))
        self.rd_data = Logic("rd_data", parent=self, shape=(32,))

        # State
        self.mem = Dict("mem", parent=self)
        for i in range(320):
            self.mem[i] = Logic(name=str(i), parent=self.mem, shape=(32,))

        # Processes
        self._procs.add((self.proc_rd_data, HW))

    # output logic [31:0] rd_data
    async def proc_rd_data(self):
        while True:
            await notify(self.rd_addr.changed)
            self.rd_data.next = self.mem[self.rd_addr.next.to_uint()].value


class ExampleDataMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports

        # State

        # Submodules
        self.data_memory = ExampleDataMemory("data_memory", parent=self)


class ExampleDataMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports

        # State
        self.mem = Dict("mem", parent=self)


class RiscvCore(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.bus_addr = Logic(name="bus_addr", parent=self, shape=(32,))
        self.bus_wr_en = Logic(name="bus_wr_en", parent=self, shape=(1,))
        self.bus_wr_be = Logic(name="bus_wr_be", parent=self, shape=(4,))
        self.bus_wr_data = Logic(name="bus_wr_data", parent=self, shape=(32,))
        self.bus_rd_en = Logic(name="bus_rd_en", parent=self, shape=(1,))
        self.bus_rd_data = Logic(name="bus_rd_data", parent=self, shape=(32,))

        self.pc = Logic(name="pc", parent=self, shape=(32,))
        self.inst = Logic(name="inst", parent=self, shape=(32,))

        self.clock = Logic(name="clock", parent=self, shape=(1,))
        self.reset = Logic(name="reset", parent=self, shape=(1,))

        # State
        self.pc_wr_en = Logic(name="pc_wr_en", parent=self, shape=(1,))
        self.regfile_wr_en = Logic(name="regfile_wr_en", parent=self, shape=(1,))
        self.alu_op_a_sel = Logic(name="alu_op_a_sel", parent=self, shape=(1,))
        self.alu_op_b_sel = Logic(name="alu_op_b_sel", parent=self, shape=(1,))
        self.reg_writeback_sel = Logic(name="reg_writeback_sel", parent=self, shape=(3,))
        self.inst_opcode = Logic(name="inst_opcode", parent=self, shape=(7,))
        self.inst_funct3 = Logic(name="inst_funct3", parent=self, shape=(3,))
        self.inst_funct7 = Logic(name="inst_funct7", parent=self, shape=(7,))
        self.next_pc_sel = Logic(name="next_pc_sel", parent=self, shape=(2,))
        self.alu_function = Logic(name="alu_function", parent=self, shape=(5,))
        self.alu_result_equal_zero = Logic(name="alu_result_equal_zero", parent=self, shape=(1,))
        self.addr = Logic(name="addr", parent=self, shape=(32,))
        self.wr_en = Logic(name="wr_en", parent=self, shape=(1,))
        self.wr_data = Logic(name="wr_data", parent=self, shape=(32,))
        self.rd_en = Logic(name="rd_en", parent=self, shape=(1,))
        self.rd_data = Logic(name="rd_data", parent=self, shape=(32,))

        # Submodules
        self.singlecycle_ctlpath = SingleCycleCtlPath(name="singlecycle_ctlpath", parent=self)
        self.singlecycle_datapath = SingleCycleDataPath(name="singlecycle_datapath", parent=self)
        self.data_memory_interface = DataMemoryInterface(name="data_memory_interface", parent=self)

        # Processes
        self.connect(self.singlecycle_ctlpath.inst_opcode, self.inst_opcode)
        self.connect(self.singlecycle_ctlpath.inst_funct3, self.inst_funct3)
        self.connect(self.singlecycle_ctlpath.inst_funct7, self.inst_funct7)
        self.connect(self.singlecycle_ctlpath.alu_result_equal_zero, self.alu_result_equal_zero)
        self.connect(self.pc_wr_en, self.singlecycle_ctlpath.pc_wr_en)
        self.connect(self.regfile_wr_en, self.singlecycle_ctlpath.regfile_wr_en)
        self.connect(self.alu_op_a_sel, self.singlecycle_ctlpath.alu_op_a_sel)
        self.connect(self.alu_op_b_sel, self.singlecycle_ctlpath.alu_op_b_sel)
        self.connect(self.rd_en, self.singlecycle_ctlpath.data_mem_rd_en)
        self.connect(self.wr_en, self.singlecycle_ctlpath.data_mem_wr_en)
        self.connect(self.reg_writeback_sel, self.singlecycle_ctlpath.reg_writeback_sel)
        self.connect(self.alu_function, self.singlecycle_ctlpath.alu_function)
        self.connect(self.next_pc_sel, self.singlecycle_ctlpath.next_pc_sel)

        self.connect(self.addr, self.singlecycle_datapath.data_mem_addr)
        self.connect(self.wr_data, self.singlecycle_datapath.data_mem_wr_data)
        self.connect(self.singlecycle_datapath.data_mem_rd_data, self.rd_data)
        self.connect(self.singlecycle_datapath.inst, self.inst)
        self.connect(self.pc, self.singlecycle_datapath.pc)
        self.connect(self.inst_opcode, self.singlecycle_datapath.inst_opcode)
        self.connect(self.inst_funct3, self.singlecycle_datapath.inst_funct3)
        self.connect(self.inst_funct7, self.singlecycle_datapath.inst_funct7)
        self.connect(self.alu_result_equal_zero, self.singlecycle_datapath.alu_result_equal_zero)
        self.connect(self.singlecycle_datapath.pc_wr_en, self.pc_wr_en)
        self.connect(self.singlecycle_datapath.regfile_wr_en, self.regfile_wr_en)
        self.connect(self.singlecycle_datapath.alu_op_a_sel, self.alu_op_a_sel)
        self.connect(self.singlecycle_datapath.alu_op_b_sel, self.alu_op_b_sel)
        self.connect(self.singlecycle_datapath.reg_writeback_sel, self.reg_writeback_sel)
        self.connect(self.singlecycle_datapath.next_pc_sel, self.next_pc_sel)
        self.connect(self.singlecycle_datapath.alu_function, self.alu_function)
        self.connect(self.singlecycle_datapath.clock, self.clock)
        self.connect(self.singlecycle_datapath.reset, self.reset)

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


class SingleCycleCtlPath(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.inst_opcode = Logic(name="inst_opcode", parent=self, shape=(7,))
        self.inst_funct3 = Logic(name="inst_funct3", parent=self, shape=(3,))
        self.inst_funct7 = Logic(name="inst_funct7", parent=self, shape=(7,))
        self.alu_result_equal_zero = Logic(name="alu_result_equal_zero", parent=self, shape=(1,))
        self.pc_wr_en = Logic(name="pc_wr_en", parent=self, shape=(1,))
        self.regfile_wr_en = Logic(name="regfile_wr_en", parent=self, shape=(1,))
        self.alu_op_a_sel = Logic(name="alu_op_a_sel", parent=self, shape=(1,))
        self.alu_op_b_sel = Logic(name="alu_op_b_sel", parent=self, shape=(1,))
        self.data_mem_rd_en = Logic(name="data_mem_rd_en", parent=self, shape=(1,))
        self.data_mem_wr_en = Logic(name="data_mem_wr_en", parent=self, shape=(1,))
        self.reg_writeback_sel = Logic(name="reg_writeback_sel", parent=self, shape=(3,))
        self.alu_function = Logic(name="alu_function", parent=self, shape=(5,))
        self.next_pc_sel = Logic(name="next_pc_sel", parent=self, shape=(2,))

        # State
        self.take_branch = Logic(name="take_branch", parent=self, shape=(1,))
        self.alu_op_type = Logic(name="alu_op_type", parent=self, shape=(2,))

        # Submodules
        self.singlecycle_control = SingleCycleControl(name="singlecycle_control", parent=self)
        self.control_transfer = ControlTransfer(name="control_transfer", parent=self)
        self.alu_control = AluControl(name="alu_control", parent=self)

        # Processes
        self.connect(self.singlecycle_control.pc_wr_en, self.pc_wr_en)
        self.connect(self.singlecycle_control.regfile_wr_en, self.regfile_wr_en)
        self.connect(self.singlecycle_control.alu_op_a_sel, self.alu_op_a_sel)
        self.connect(self.singlecycle_control.alu_op_b_sel, self.alu_op_b_sel)
        self.connect(self.singlecycle_control.alu_op_type, self.alu_op_type)
        self.connect(self.singlecycle_control.data_mem_rd_en, self.data_mem_rd_en)
        self.connect(self.singlecycle_control.data_mem_wr_en, self.data_mem_wr_en)
        self.connect(self.singlecycle_control.reg_writeback_sel, self.reg_writeback_sel)
        self.connect(self.singlecycle_control.next_pc_sel, self.next_pc_sel)
        self.connect(self.inst_opcode, self.singlecycle_control.inst_opcode)
        self.connect(self.take_branch, self.singlecycle_control.take_branch)

        self.connect(self.take_branch, self.control_transfer.take_branch)
        self.connect(self.control_transfer.inst_funct3, self.inst_funct3)
        self.connect(self.control_transfer.result_equal_zero, self.alu_result_equal_zero)

        self.connect(self.alu_function, self.alu_control.alu_function)
        self.connect(self.alu_control.alu_op_type, self.alu_op_type)
        self.connect(self.alu_control.inst_funct3, self.inst_funct3)
        self.connect(self.alu_control.inst_funct7, self.inst_funct7)


class SingleCycleControl(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.pc_wr_en = Logic(name="pc_wr_en", parent=self, shape=(1,))
        self.regfile_wr_en = Logic(name="regfile_wr_en", parent=self, shape=(1,))
        self.alu_op_a_sel = Logic(name="alu_op_a_sel", parent=self, shape=(1,))
        self.alu_op_b_sel = Logic(name="alu_op_b_sel", parent=self, shape=(1,))
        self.alu_op_type = Logic(name="alu_op_type", parent=self, shape=(2,))
        self.data_mem_rd_en = Logic(name="data_mem_rd_en", parent=self, shape=(1,))
        self.data_mem_wr_en = Logic(name="data_mem_wr_en", parent=self, shape=(1,))
        self.reg_writeback_sel = Logic(name="reg_writeback_sel", parent=self, shape=(3,))
        self.next_pc_sel = Logic(name="next_pc_sel", parent=self, shape=(2,))
        self.inst_opcode = Logic(name="inst_opcode", parent=self, shape=(7,))
        self.take_branch = Logic(name="take_branch", parent=self, shape=(1,))


class ControlTransfer(Module):
    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.take_branch = Logic(name="take_branch", parent=self, shape=(1,))
        self.inst_funct3 = Logic(name="inst_funct3", parent=self, shape=(3,))
        self.result_equal_zero = Logic(name="result_equal_zero", parent=self, shape=(1,))


class AluControl(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.alu_function = Logic(name="alu_function", parent=self, shape=(5,))
        self.alu_op_type = Logic(name="alu_op_type", parent=self, shape=(2,))
        self.inst_funct3 = Logic(name="inst_funct3", parent=self, shape=(3,))
        self.inst_funct7 = Logic(name="inst_funct7", parent=self, shape=(7,))

        # State
        self.default_funct = Logic(name="default_funct", parent=self, shape=(5,))
        self.secondary_funct = Logic(name="seconary_funct", parent=self, shape=(5,))
        self.op_funct = Logic(name="op_funct", parent=self, shape=(5,))
        self.op_imm_funct = Logic(name="op_imm_funct", parent=self, shape=(5,))
        self.branch_funct = Logic(name="branch_funct", parent=self, shape=(5,))


class SingleCycleDataPath(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.data_mem_addr = Logic(name="data_mem_addr", parent=self, shape=(32,))
        self.data_mem_wr_data = Logic(name="data_mem_wr_data", parent=self, shape=(32,))
        self.data_mem_rd_data = Logic(name="data_mem_rd_data", parent=self, shape=(32,))

        self.inst = Logic(name="inst", parent=self, shape=(32,))
        self.pc = Logic(name="pc", parent=self, shape=(32,))

        self.inst_opcode = Logic(name="inst_opcode", parent=self, shape=(7,))
        self.inst_funct3 = Logic(name="inst_funct3", parent=self, shape=(3,))
        self.inst_funct7 = Logic(name="inst_funct7", parent=self, shape=(7,))
        self.alu_result_equal_zero = Logic(name="alu_result_equal_zero", parent=self, shape=(1,))

        self.pc_wr_en = Logic(name="pc_wr_en", parent=self, shape=(1,))
        self.regfile_wr_en = Logic(name="regfile_wr_en", parent=self, shape=(1,))
        self.alu_op_a_sel = Logic(name="alu_op_a_sel", parent=self, shape=(1,))
        self.alu_op_b_sel = Logic(name="alu_op_b_sel", parent=self, shape=(1,))
        self.reg_writeback_sel = Logic(name="reg_writeback_sel", parent=self, shape=(3,))
        self.next_pc_sel = Logic(name="next_pc_sel", parent=self, shape=(2,))
        self.alu_function = Logic(name="alu_function", parent=self, shape=(5,))

        self.clock = Logic(name="clock", parent=self, shape=(1,))
        self.reset = Logic(name="reset", parent=self, shape=(1,))

        # State
        self.rd_data = Logic(name="rd_data", parent=self, shape=(32,))
        self.rs1_data = Logic(name="rs1_data", parent=self, shape=(32,))
        self.rs2_data = Logic(name="rs2_data", parent=self, shape=(32,))
        self.inst_rs2 = Logic(name="inst_rs2", parent=self, shape=(5,))
        self.inst_rs1 = Logic(name="inst_rs1", parent=self, shape=(5,))
        self.inst_rd = Logic(name="inst_rd", parent=self, shape=(5,))

        self.pc_plus_4 = TraceLogic(name="pc_plus_4", parent=self, shape=(32,))
        self.pc_plus_immediate = TraceLogic(name="pc_plus_immediate", parent=self, shape=(32,))
        self.pc_next = Logic(name="pc_next", parent=self, shape=(32,))

        self.alu_op_a = Logic(name="alu_op_a", parent=self, shape=(32,))
        self.alu_op_b = Logic(name="alu_op_b", parent=self, shape=(32,))
        self.alu_result = Logic(name="alu_result", parent=self, shape=(32,))

        self.immediate = TraceLogic(name="immediate", parent=self, shape=(32,))

        # Submodules
        self.adder_pc_plus_4 = Adder(name="adder_pc_plus_4", parent=self, width=32)
        self.adder_pc_plus_immediate = Adder(name="adder_pc_plus_immediate", parent=self, width=32)
        self.instruction_decoder = InstructionDecoder(name="instruction_decoder", parent=self)
        self.immediate_generator = ImmedateGenerator(name="immediate_generator", parent=self)
        self.program_counter = Register(
            name="program_counter", parent=self, width=32, init=vec("32h0040_0000")
        )
        self.regfile = Regfile(name="regfile", parent=self)

        self.connect(self.pc_plus_4, self.adder_pc_plus_4.result)
        self.connect(self.adder_pc_plus_4.op_b, self.pc)

        self.connect(self.pc_plus_immediate, self.adder_pc_plus_immediate.result)
        self.connect(self.adder_pc_plus_immediate.op_a, self.pc)
        self.connect(self.adder_pc_plus_immediate.op_b, self.immediate)

        self.connect(self.instruction_decoder.inst, self.inst)
        self.connect(self.inst_opcode, self.instruction_decoder.inst_opcode)
        self.connect(self.inst_funct3, self.instruction_decoder.inst_funct3)
        self.connect(self.inst_funct7, self.instruction_decoder.inst_funct7)
        self.connect(self.inst_rs2, self.instruction_decoder.inst_rs2)
        self.connect(self.inst_rs1, self.instruction_decoder.inst_rs1)
        self.connect(self.inst_rd, self.instruction_decoder.inst_rd)

        self.connect(self.immediate, self.immediate_generator.immediate)
        self.connect(self.immediate_generator.inst, self.inst)

        self.connect(self.pc, self.program_counter.q)
        self.connect(self.program_counter.clock, self.clock)
        self.connect(self.program_counter.reset, self.reset)

        self.connect(self.rs1_data, self.regfile.rs1_data)
        self.connect(self.rs2_data, self.regfile.rs2_data)
        self.connect(self.regfile.wr_en, self.regfile_wr_en)
        self.connect(self.regfile.rd_addr, self.inst_rd)
        self.connect(self.regfile.rs1_addr, self.inst_rs1)
        self.connect(self.regfile.rs2_addr, self.inst_rs2)
        self.connect(self.regfile.rd_data, self.rd_data)
        self.connect(self.regfile.clock, self.clock)

        self._procs.add((self.proc_lit1, HW))

    async def proc_lit1(self):
        self.adder_pc_plus_4.op_a.next = vec("32h0000_0004")


class DataMemoryInterface(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.data_format = Logic(name="data_format", parent=self, shape=(3,))

        self.addr = Logic(name="addr", parent=self, shape=(32,))
        self.wr_en = Logic(name="wr_en", parent=self, shape=(1,))
        self.wr_data = Logic(name="wr_data", parent=self, shape=(32,))
        self.rd_en = Logic(name="rd_en", parent=self, shape=(1,))
        self.rd_data = Logic(name="rd_data", parent=self, shape=(32,))

        self.bus_addr = Logic(name="bus_addr", parent=self, shape=(32,))
        self.bus_wr_en = Logic(name="bus_wr_en", parent=self, shape=(1,))
        self.bus_wr_be = Logic(name="bus_wr_be", parent=self, shape=(4,))
        self.bus_wr_data = Logic(name="bus_wr_data", parent=self, shape=(32,))
        self.bus_rd_en = Logic(name="bus_rd_en", parent=self, shape=(1,))
        self.bus_rd_data = Logic(name="bus_rd_data", parent=self, shape=(32,))

        # State
        self.position_fix = Logic(name="position_fix", parent=self, shape=(32,))
        self.sign_fix = Logic(name="sign_fix", parent=self, shape=(32,))

        # Processes


class InstructionDecoder(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.inst_funct7 = Logic(name="inst_funct7", parent=self, shape=(7,))
        self.inst_rs2 = Logic(name="inst_rs2", parent=self, shape=(5,))
        self.inst_rs1 = Logic(name="inst_rs1", parent=self, shape=(5,))
        self.inst_funct3 = Logic(name="inst_funct3", parent=self, shape=(3,))
        self.inst_rd = Logic(name="inst_rd", parent=self, shape=(5,))
        self.inst_opcode = TraceLogic(name="inst_opcode", parent=self, shape=(7,))
        self.inst = Logic(name="inst", parent=self, shape=(32,))

        # Processes
        self._procs.add((self.proc_funct7, HW))
        self._procs.add((self.proc_rs2, HW))
        self._procs.add((self.proc_rs1, HW))
        self._procs.add((self.proc_funct3, HW))
        self._procs.add((self.proc_rd, HW))
        self._procs.add((self.proc_opcode, HW))

    async def proc_funct7(self):
        while True:
            await notify(self.inst.changed)
            self.inst_funct7.next = self.inst.next[25:32]

    async def proc_rs2(self):
        while True:
            await notify(self.inst.changed)
            self.inst_rs2.next = self.inst.next[20:25]

    async def proc_rs1(self):
        while True:
            await notify(self.inst.changed)
            self.inst_rs1.next = self.inst.next[15:20]

    async def proc_funct3(self):
        while True:
            await notify(self.inst.changed)
            self.inst_funct3.next = self.inst.next[12:15]

    async def proc_rd(self):
        while True:
            await notify(self.inst.changed)
            self.inst_rd.next = self.inst.next[7:12]

    async def proc_opcode(self):
        while True:
            await notify(self.inst.changed)
            self.inst_opcode.next = self.inst.next[0:7]


class ImmedateGenerator(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.immediate = Logic(name="immediate", parent=self, shape=(32,))
        self.inst = Logic(name="inst", parent=self, shape=(32,))

        # Processes
        self._procs.add((self.proc_immediate, HW))

    async def proc_immediate(self):
        while True:
            await notify(self.inst.changed)
            if self.inst.next[0:7] in [Opcode.LOAD, Opcode.LOAD_FP, Opcode.OP_IMM, Opcode.JALR]:
                a = self.inst.next[20:25]
                b = self.inst.next[25:31]
                c = rep(self.inst.next[31], 21)
                self.immediate.next = cat([a, b, c])
            elif self.inst.next[0:7] in [Opcode.STORE_FP, Opcode.STORE]:
                self.immediate.next = cat(
                    [
                        self.inst.next[7:12],
                        self.inst.next[25:31],
                        rep(self.inst.next[31], 21),
                    ]
                )
            elif self.inst.next[0:7] == Opcode.BRANCH:
                self.immediate.next = cat(
                    [
                        F,
                        self.inst.next[8:12],
                        self.inst.next[25:31],
                        self.inst.next[7],
                        rep(self.inst.next[31], 20),
                    ]
                )
            elif self.inst.next[0:7] in [Opcode.AUIPC, Opcode.LUI]:
                self.immediate.next = cat(
                    [
                        vec("12b0000_0000_0000"),
                        self.inst.next[12:20],
                        self.inst.next[20:31],
                        self.inst.next[31],
                    ]
                )
            elif self.inst.next[0:7] == Opcode.JAL:
                self.immediate.next = cat(
                    [
                        F,
                        self.inst.next[21:25],
                        self.inst.next[25:31],
                        self.inst.next[20],
                        self.inst.next[12:20],
                        rep(self.inst.next[31], 12),
                    ]
                )
            else:
                self.immediate.next = vec("32h0000_0000")


class Register(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int, init: logicvec):
        super().__init__(name, parent)

        # Parameters
        self.init = init

        # Ports
        self.q = Logic(name="q", parent=self, shape=(width,))
        self.en = Logic(name="en", parent=self, shape=(1,))
        self.d = Logic(name="d", parent=self, shape=(width,))
        self.clock = Logic(name="clock", parent=self, shape=(1,))
        self.reset = Logic(name="reset", parent=self, shape=(1,))

        self._procs.add((self.proc_q, TASK))

    async def proc_q(self):
        await notify(self.reset.posedge)
        self.q.next = vec("32h0040_0000")
        await notify(self.reset.negedge)
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_0004")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_0008")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_000C")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_0010")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_0014")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_0018")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_001C")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_0020")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_0024")
        await notify(self.clock.posedge)
        self.q.next = vec("32h0040_0028")


class Regfile(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.rs1_data = Logic(name="rs1_data", parent=self, shape=(32,))
        self.rs2_data = Logic(name="rs2_data", parent=self, shape=(32,))
        self.wr_en = Logic(name="wr_en", parent=self, shape=(1,))
        self.rd_addr = Logic(name="rd_addr", parent=self, shape=(5,))
        self.rs1_addr = Logic(name="rs1_addr", parent=self, shape=(5,))
        self.rs2_addr = Logic(name="rs2_addr", parent=self, shape=(5,))
        self.rd_data = Logic(name="rd_data", parent=self, shape=(32,))
        self.clock = Logic(name="clock", parent=self, shape=(1,))


class Adder(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, width: int):
        super().__init__(name, parent)

        # Parameters
        self.width = width

        # Ports
        self.result = Logic(name="result", parent=self, shape=(width,))
        self.op_a = Logic(name="op_a", parent=self, shape=(width,))
        self.op_b = Logic(name="op_b", parent=self, shape=(width,))

        # Processes
        self._procs.add((self.proc_result, HW))

    async def proc_result(self):
        while True:
            await notify(self.op_a.changed, self.op_b.changed)
            self.result.next = self.op_a.next + self.op_b.next


def test_singlecycle1():
    top = Top(name="top")
    exp = (
        [
            top,
            top.bus_addr,
            top.bus_wr_en,
            top.bus_wr_be,
            top.bus_wr_data,
            top.bus_rd_en,
            top.bus_rd_data,
            top.pc,
            top.inst,
            top.clock,
            top.reset,
            top.text_memory_bus,
            top.text_memory_bus.rd_addr,
            top.text_memory_bus.rd_data,
            top.text_memory_bus.text,
            top.text_memory_bus.text_memory,
            top.text_memory_bus.text_memory.rd_addr,
            top.text_memory_bus.text_memory.rd_data,
            top.text_memory_bus.text_memory.mem,
        ]
        + [top.text_memory_bus.text_memory.mem[i] for i in range(320)]
        + [
            top.data_memory_bus,
            top.data_memory_bus.data_memory,
            top.data_memory_bus.data_memory.mem,
            top.riscv_core,
            top.riscv_core.bus_addr,
            top.riscv_core.bus_wr_en,
            top.riscv_core.bus_wr_be,
            top.riscv_core.bus_wr_data,
            top.riscv_core.bus_rd_en,
            top.riscv_core.bus_rd_data,
            top.riscv_core.pc,
            top.riscv_core.inst,
            top.riscv_core.clock,
            top.riscv_core.reset,
            top.riscv_core.pc_wr_en,
            top.riscv_core.regfile_wr_en,
            top.riscv_core.alu_op_a_sel,
            top.riscv_core.alu_op_b_sel,
            top.riscv_core.reg_writeback_sel,
            top.riscv_core.inst_opcode,
            top.riscv_core.inst_funct3,
            top.riscv_core.inst_funct7,
            top.riscv_core.next_pc_sel,
            top.riscv_core.alu_function,
            top.riscv_core.alu_result_equal_zero,
            top.riscv_core.addr,
            top.riscv_core.wr_en,
            top.riscv_core.wr_data,
            top.riscv_core.rd_en,
            top.riscv_core.rd_data,
            top.riscv_core.singlecycle_ctlpath,
            top.riscv_core.singlecycle_ctlpath.inst_opcode,
            top.riscv_core.singlecycle_ctlpath.inst_funct3,
            top.riscv_core.singlecycle_ctlpath.inst_funct7,
            top.riscv_core.singlecycle_ctlpath.alu_result_equal_zero,
            top.riscv_core.singlecycle_ctlpath.pc_wr_en,
            top.riscv_core.singlecycle_ctlpath.regfile_wr_en,
            top.riscv_core.singlecycle_ctlpath.alu_op_a_sel,
            top.riscv_core.singlecycle_ctlpath.alu_op_b_sel,
            top.riscv_core.singlecycle_ctlpath.data_mem_rd_en,
            top.riscv_core.singlecycle_ctlpath.data_mem_wr_en,
            top.riscv_core.singlecycle_ctlpath.reg_writeback_sel,
            top.riscv_core.singlecycle_ctlpath.alu_function,
            top.riscv_core.singlecycle_ctlpath.next_pc_sel,
            top.riscv_core.singlecycle_ctlpath.take_branch,
            top.riscv_core.singlecycle_ctlpath.alu_op_type,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.pc_wr_en,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.regfile_wr_en,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.alu_op_a_sel,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.alu_op_b_sel,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.alu_op_type,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.data_mem_rd_en,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.data_mem_wr_en,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.reg_writeback_sel,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.next_pc_sel,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.inst_opcode,
            top.riscv_core.singlecycle_ctlpath.singlecycle_control.take_branch,
            top.riscv_core.singlecycle_ctlpath.control_transfer,
            top.riscv_core.singlecycle_ctlpath.control_transfer.take_branch,
            top.riscv_core.singlecycle_ctlpath.control_transfer.inst_funct3,
            top.riscv_core.singlecycle_ctlpath.control_transfer.result_equal_zero,
            top.riscv_core.singlecycle_ctlpath.alu_control,
            top.riscv_core.singlecycle_ctlpath.alu_control.alu_function,
            top.riscv_core.singlecycle_ctlpath.alu_control.alu_op_type,
            top.riscv_core.singlecycle_ctlpath.alu_control.inst_funct3,
            top.riscv_core.singlecycle_ctlpath.alu_control.inst_funct7,
            top.riscv_core.singlecycle_ctlpath.alu_control.default_funct,
            top.riscv_core.singlecycle_ctlpath.alu_control.secondary_funct,
            top.riscv_core.singlecycle_ctlpath.alu_control.op_funct,
            top.riscv_core.singlecycle_ctlpath.alu_control.op_imm_funct,
            top.riscv_core.singlecycle_ctlpath.alu_control.branch_funct,
            top.riscv_core.singlecycle_datapath,
            top.riscv_core.singlecycle_datapath.data_mem_addr,
            top.riscv_core.singlecycle_datapath.data_mem_wr_data,
            top.riscv_core.singlecycle_datapath.data_mem_rd_data,
            top.riscv_core.singlecycle_datapath.inst,
            top.riscv_core.singlecycle_datapath.pc,
            top.riscv_core.singlecycle_datapath.inst_opcode,
            top.riscv_core.singlecycle_datapath.inst_funct3,
            top.riscv_core.singlecycle_datapath.inst_funct7,
            top.riscv_core.singlecycle_datapath.alu_result_equal_zero,
            top.riscv_core.singlecycle_datapath.pc_wr_en,
            top.riscv_core.singlecycle_datapath.regfile_wr_en,
            top.riscv_core.singlecycle_datapath.alu_op_a_sel,
            top.riscv_core.singlecycle_datapath.alu_op_b_sel,
            top.riscv_core.singlecycle_datapath.reg_writeback_sel,
            top.riscv_core.singlecycle_datapath.next_pc_sel,
            top.riscv_core.singlecycle_datapath.alu_function,
            top.riscv_core.singlecycle_datapath.clock,
            top.riscv_core.singlecycle_datapath.reset,
            top.riscv_core.singlecycle_datapath.rd_data,
            top.riscv_core.singlecycle_datapath.rs1_data,
            top.riscv_core.singlecycle_datapath.rs2_data,
            top.riscv_core.singlecycle_datapath.inst_rs2,
            top.riscv_core.singlecycle_datapath.inst_rs1,
            top.riscv_core.singlecycle_datapath.inst_rd,
            top.riscv_core.singlecycle_datapath.pc_plus_4,
            top.riscv_core.singlecycle_datapath.pc_plus_immediate,
            top.riscv_core.singlecycle_datapath.pc_next,
            top.riscv_core.singlecycle_datapath.alu_op_a,
            top.riscv_core.singlecycle_datapath.alu_op_b,
            top.riscv_core.singlecycle_datapath.alu_result,
            top.riscv_core.singlecycle_datapath.immediate,
            top.riscv_core.singlecycle_datapath.adder_pc_plus_4,
            top.riscv_core.singlecycle_datapath.adder_pc_plus_4.result,
            top.riscv_core.singlecycle_datapath.adder_pc_plus_4.op_a,
            top.riscv_core.singlecycle_datapath.adder_pc_plus_4.op_b,
            top.riscv_core.singlecycle_datapath.adder_pc_plus_immediate,
            top.riscv_core.singlecycle_datapath.adder_pc_plus_immediate.result,
            top.riscv_core.singlecycle_datapath.adder_pc_plus_immediate.op_a,
            top.riscv_core.singlecycle_datapath.adder_pc_plus_immediate.op_b,
            top.riscv_core.singlecycle_datapath.instruction_decoder,
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_funct7,
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_rs2,
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_rs1,
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_funct3,
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_rd,
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode,
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst,
            top.riscv_core.singlecycle_datapath.immediate_generator,
            top.riscv_core.singlecycle_datapath.immediate_generator.immediate,
            top.riscv_core.singlecycle_datapath.immediate_generator.inst,
            top.riscv_core.singlecycle_datapath.program_counter,
            top.riscv_core.singlecycle_datapath.program_counter.q,
            top.riscv_core.singlecycle_datapath.program_counter.en,
            top.riscv_core.singlecycle_datapath.program_counter.d,
            top.riscv_core.singlecycle_datapath.program_counter.clock,
            top.riscv_core.singlecycle_datapath.program_counter.reset,
            top.riscv_core.singlecycle_datapath.regfile,
            top.riscv_core.singlecycle_datapath.regfile.rs1_data,
            top.riscv_core.singlecycle_datapath.regfile.rs2_data,
            top.riscv_core.singlecycle_datapath.regfile.wr_en,
            top.riscv_core.singlecycle_datapath.regfile.rd_addr,
            top.riscv_core.singlecycle_datapath.regfile.rs1_addr,
            top.riscv_core.singlecycle_datapath.regfile.rs2_addr,
            top.riscv_core.singlecycle_datapath.regfile.rd_data,
            top.riscv_core.singlecycle_datapath.regfile.clock,
            top.riscv_core.data_memory_interface,
            top.riscv_core.data_memory_interface.data_format,
            top.riscv_core.data_memory_interface.addr,
            top.riscv_core.data_memory_interface.wr_en,
            top.riscv_core.data_memory_interface.wr_data,
            top.riscv_core.data_memory_interface.rd_en,
            top.riscv_core.data_memory_interface.rd_data,
            top.riscv_core.data_memory_interface.bus_addr,
            top.riscv_core.data_memory_interface.bus_wr_en,
            top.riscv_core.data_memory_interface.bus_wr_be,
            top.riscv_core.data_memory_interface.bus_wr_data,
            top.riscv_core.data_memory_interface.bus_rd_en,
            top.riscv_core.data_memory_interface.bus_rd_data,
            top.riscv_core.data_memory_interface.position_fix,
            top.riscv_core.data_memory_interface.sign_fix,
        ]
    )
    assert list(top.iter_bfs()) == exp


TEXT = []
with open("tests/add.text", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        for part in parts[1:]:
            TEXT.append(int(part, base=16))


def test_singlecycle2():
    loop.reset()
    waves.clear()

    # Create module hierarchy
    top = Top(name="top")

    for node in top.iter_bfs():
        # TODO(cjdrake): Get rid of isinstance
        if isinstance(node, Module):
            for proc, r in node.procs:
                loop.add_proc(proc, Region(r))

    # Initialize instruction memory
    for i, d in enumerate(TEXT):
        top.text_memory_bus.text_memory.mem[i].value = uint2vec(d, 32)

    loop.run(until=30)

    exp = {
        # Initialize everything to X'es
        -1: {
            top.clock: X,
            top.reset: X,
            top.pc: xes((32,)),
            top.inst: xes((32,)),
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode: xes((7,)),
            top.riscv_core.singlecycle_datapath.pc_plus_4: xes((32,)),
            top.riscv_core.singlecycle_datapath.immediate: xes((32,)),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: xes((32,)),
        },
        0: {
            top.clock: F,
            top.reset: F,
        },
        1: {
            top.clock: T,
        },
        2: {
            top.clock: F,
        },
        3: {
            top.clock: T,
        },
        4: {
            top.clock: F,
        },
        # @(posedge reset)
        5: {
            top.clock: T,
            top.reset: T,
            top.pc: vec("32h0040_0000"),
            top.inst: vec("32h0000_0093"),
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode: Opcode.OP_IMM,
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_0004"),
            top.riscv_core.singlecycle_datapath.immediate: vec("32h0000_0000"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_0000"),
        },
        6: {
            top.clock: F,
        },
        7: {
            top.clock: T,
        },
        8: {
            top.clock: F,
        },
        9: {
            top.clock: T,
        },
        # @(negedge reset)
        10: {
            top.clock: F,
            top.reset: F,
        },
        # @(posedge clock)
        11: {
            top.clock: T,
            top.pc: vec("32h0040_0004"),
            top.inst: vec("32h0000_0113"),
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_0008"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_0004"),
        },
        12: {
            top.clock: F,
        },
        # @(posedge clock)
        13: {
            top.clock: T,
            top.pc: vec("32h0040_0008"),
            top.inst: vec("32h0020_81B3"),
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode: Opcode.OP,
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_000C"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_0008"),
        },
        14: {
            top.clock: F,
        },
        # @(posedge clock)
        15: {
            top.clock: T,
            top.pc: vec("32h0040_000C"),
            top.inst: vec("32h0000_0E93"),
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode: Opcode.OP_IMM,
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_0010"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_000C"),
        },
        16: {
            top.clock: F,
        },
        # @(posedge clock)
        17: {
            top.clock: T,
            top.pc: vec("32h0040_0010"),
            top.inst: vec("32h0020_0E13"),
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_0014"),
            top.riscv_core.singlecycle_datapath.immediate: vec("32h0000_0002"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_0012"),
        },
        18: {
            top.clock: F,
        },
        # @(posedge clock)
        19: {
            top.clock: T,
            top.pc: vec("32h0040_0014"),
            top.inst: vec("32h4DD1_9663"),
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode: Opcode.BRANCH,
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_0018"),
            top.riscv_core.singlecycle_datapath.immediate: vec("32h0000_04CC"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_04E0"),
        },
        20: {
            top.clock: F,
        },
        # @(posedge clock)
        21: {
            top.clock: T,
            top.pc: vec("32h0040_0018"),
            top.inst: vec("32h0010_0093"),
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode: Opcode.OP_IMM,
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_001C"),
            top.riscv_core.singlecycle_datapath.immediate: vec("32h0000_0001"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_0019"),
        },
        22: {
            top.clock: F,
        },
        # @(posedge clock)
        23: {
            top.clock: T,
            top.pc: vec("32h0040_001C"),
            top.inst: vec("32h0010_0113"),
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_0020"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_001D"),
        },
        # @(posedge clock)
        24: {
            top.clock: F,
        },
        # @(posedge clock)
        25: {
            top.clock: T,
            top.pc: vec("32h0040_0020"),
            top.inst: vec("32h0020_81B3"),
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode: Opcode.OP,
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_0024"),
            top.riscv_core.singlecycle_datapath.immediate: vec("32h0000_0000"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_0020"),
        },
        26: {
            top.clock: F,
        },
        # @(posedge clock)
        27: {
            top.clock: T,
            top.pc: vec("32h0040_0024"),
            top.inst: vec("32h0020_0E93"),
            top.riscv_core.singlecycle_datapath.instruction_decoder.inst_opcode: Opcode.OP_IMM,
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_0028"),
            top.riscv_core.singlecycle_datapath.immediate: vec("32h0000_0002"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_0026"),
        },
        28: {
            top.clock: F,
        },
        # @(posedge clock)
        29: {
            top.clock: T,
            top.pc: vec("32h0040_0028"),
            top.inst: vec("32h0030_0E13"),
            top.riscv_core.singlecycle_datapath.pc_plus_4: vec("32h0040_002C"),
            top.riscv_core.singlecycle_datapath.immediate: vec("32h0000_0003"),
            top.riscv_core.singlecycle_datapath.pc_plus_immediate: vec("32h0040_002B"),
        },
    }
    assert waves == exp
