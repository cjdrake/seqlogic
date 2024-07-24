"""Data Memory Interface."""

from seqlogic import Module, Vec, cat, rep

from . import Addr


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
        data_format = self.input(name="data_format", dtype=Vec[3])

        addr = self.input(name="addr", dtype=Addr)
        wr_en = self.input(name="wr_en", dtype=Vec[1])
        wr_data = self.input(name="wr_data", dtype=Vec[32])
        rd_en = self.input(name="rd_en", dtype=Vec[1])
        rd_data = self.output(name="rd_data", dtype=Vec[32])

        bus_addr = self.output(name="bus_addr", dtype=Addr)
        bus_wr_en = self.output(name="bus_wr_en", dtype=Vec[1])
        bus_wr_be = self.output(name="bus_wr_be", dtype=Vec[4])
        bus_wr_data = self.output(name="bus_wr_data", dtype=Vec[32])
        bus_rd_en = self.output(name="bus_rd_en", dtype=Vec[1])
        bus_rd_data = self.input(name="bus_rd_data", dtype=Vec[32])

        byte_addr = self.logic(name="byte_addr", dtype=Vec[2])

        self.assign(bus_addr, addr)
        self.assign(bus_wr_en, wr_en)
        self.assign(bus_rd_en, rd_en)

        # Combinational Logic
        self.combi(bus_wr_be, f_bus_wr_be, data_format, byte_addr)
        self.combi(bus_wr_data, f_bus_wr_data, wr_data, byte_addr)
        self.combi(rd_data, f_rd_data, data_format, bus_rd_data, byte_addr)
        self.combi(byte_addr, lambda a: a[:2], addr)
