"""Data Memory Interface."""

from seqlogic import Module
from seqlogic.vec import Vec, cat, rep


def f_bus_wr_be(data_format: Vec[3], byte_addr: Vec[2]):
    sel = data_format[:2]
    match sel:
        case "2b00":
            return "4b0001" << byte_addr
        case "2b01":
            return "4b0011" << byte_addr
        case "2b10":
            return "4b1111" << byte_addr
        case "2b11":
            return "4b0000"
        case _:
            Vec[4].xprop(sel)


def f_bus_wr_data(wr_data: Vec[32], byte_addr: Vec[2]) -> Vec[32]:
    return wr_data << cat("3b000", byte_addr)


def f_rd_data(data_format: Vec[3], bus_rd_data: Vec[32], byte_addr: Vec[2]) -> Vec[32]:
    data = bus_rd_data >> cat("3b000", byte_addr)
    sel = data_format[:2]
    match sel:
        case "2b00":
            pad = ~data_format[2] & data[8 - 1]
            return cat(data[:8], rep(pad, 24))
        case "2b01":
            pad = ~data_format[2] & data[16 - 1]
            return cat(data[:16], rep(pad, 16))
        case "2b10":
            return data
        case _:
            return Vec[32].xprop(sel)


class DataMemIf(Module):
    """Data Memory Interface."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        data_format = self.bits(name="data_format", dtype=Vec[3], port=True)

        addr = self.bits(name="addr", dtype=Vec[32], port=True)
        wr_en = self.bit(name="wr_en", port=True)
        wr_data = self.bits(name="wr_data", dtype=Vec[32], port=True)
        rd_en = self.bit(name="rd_en", port=True)
        rd_data = self.bits(name="rd_data", dtype=Vec[32], port=True)

        bus_addr = self.bits(name="bus_addr", dtype=Vec[32], port=True)
        bus_wr_en = self.bit(name="bus_wr_en", port=True)
        bus_wr_be = self.bits(name="bus_wr_be", dtype=Vec[4], port=True)
        bus_wr_data = self.bits(name="bus_wr_data", dtype=Vec[32], port=True)
        bus_rd_en = self.bit(name="bus_rd_en", port=True)
        bus_rd_data = self.bits(name="bus_rd_data", dtype=Vec[32], port=True)

        byte_addr = self.bits(name="byte_addr", dtype=Vec[2])

        self.assign(bus_addr, addr)
        self.assign(bus_wr_en, wr_en)
        self.assign(bus_rd_en, rd_en)

        # Combinational Logic
        self.combi(bus_wr_be, f_bus_wr_be, data_format, byte_addr)
        self.combi(bus_wr_data, f_bus_wr_data, wr_data, byte_addr)
        self.combi(rd_data, f_rd_data, data_format, bus_rd_data, byte_addr)
        self.combi(byte_addr, lambda a: a[:2], addr)
