"""Data Memory Bus."""

# pyright: reportAttributeAccessIssue=false

import operator

from seqlogic import Module, clog2
from seqlogic.lbool import Vec, uint2vec

from . import DATA_BASE, DATA_SIZE
from .data_mem import DataMem


def f_rd_data(rd_en: Vec[1], is_data: Vec[1], data: Vec[32]) -> Vec[32]:
    sel = rd_en & is_data
    match sel:
        case "1b1":
            return data
        case _:
            return Vec[32].xprop(sel)


class DataMemBus(Module):
    """Data Memory Bus."""

    def __init__(self, name: str, parent: Module | None, depth: int = 1024):
        super().__init__(name, parent)

        # Parameters
        word_addr_bits = clog2(depth)
        data_start = DATA_BASE
        data_stop = DATA_BASE + DATA_SIZE

        # Ports
        addr = self.bits(name="addr", dtype=Vec[32], port=True)
        wr_en = self.bit(name="wr_en", port=True)
        wr_be = self.bits(name="wr_be", dtype=Vec[4], port=True)
        wr_data = self.bits(name="wr_data", dtype=Vec[32], port=True)
        rd_en = self.bit(name="rd_en", port=True)
        rd_data = self.bits(name="rd_data", dtype=Vec[32], port=True)
        clock = self.bit(name="clock", port=True)

        # State
        is_data = self.bit(name="is_data")
        data = self.bits(name="data", dtype=Vec[32])

        # Submodules
        data_mem = self.submod(name="data_mem", mod=DataMem, word_addr_bits=word_addr_bits)
        self.connect(data_mem.wr_be, wr_be)
        self.connect(data_mem.wr_data, wr_data)
        self.connect(data, data_mem.rd_data)
        self.connect(data_mem.clock, clock)

        # Combinational Logic
        def f_is_data(addr: Vec[32]) -> Vec[1]:
            start = uint2vec(data_start, 32)
            stop = uint2vec(data_stop, 32)
            return start.lteu(addr) & addr.ltu(stop)

        self.combi(is_data, f_is_data, addr)
        self.combi(data_mem.wr_en, operator.and_, wr_en, is_data)
        m, n = 2, 2 + word_addr_bits
        self.combi(data_mem.addr, lambda a: a[m:n], addr)
        self.combi(rd_data, f_rd_data, rd_en, is_data, data)
