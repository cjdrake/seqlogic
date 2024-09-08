"""Globals: clocks and resets."""

from ..design import Packed
from ..sim import sleep


async def drv_reset(
    y: Packed,
    shiftticks: int = 0,
    onticks: int = 1,
    neg: bool = False,
):
    r"""
    Drive a reset signal.

    Active Positive:

    ________/‾‾‾‾‾‾‾‾\\________

    Active Negative:

    ‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾

    Args:
        y
        shiftticks: ticks before reset asserts
        onticks: ticks before reset deasserts
        neg: reset is active negative

    Raises:
        ValueError: Invalid parameter values
    """
    if shiftticks < 0:
        raise ValueError(f"Expected shiftticks ≥ 0, got {shiftticks}")
    if onticks < 0:
        raise ValueError(f"Expected onticks ≥ 0, got {onticks}")

    # T = 0
    y.next = "1b1" if neg else "1b0"
    await sleep(shiftticks)

    y.next = ~y.value
    await sleep(onticks)

    y.next = ~y.value


async def drv_clock(
    y: Packed,
    shiftticks: int = 0,
    onticks: int = 1,
    offticks: int = 1,
    neg: bool = False,
):
    r"""
    Drive a clock signal.

    Active Positive:

    ________/‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\________

    Active Negative:

    ‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\

    Args:
        y
        shiftticks: ticks before clock asserts
        onticks: ticks clock is asserted during period
        offticks: ticks clock is deasserted during period
        neg: clock is active negative

    The period is onticks + offticks
    The duty cycle is onticks / (onticks + offticks)

    Raises:
        ValueError: Invalid parameter values
    """
    if shiftticks < 0:
        raise ValueError(f"Expected shiftticks ≥ 0, got {shiftticks}")
    if onticks < 0:
        raise ValueError(f"Expected onticks ≥ 0, got {onticks}")
    if offticks < 0:
        raise ValueError(f"Expected offticks ≥ 0, got {offticks}")

    # T = 0
    y.next = "1b1" if neg else "1b0"
    await sleep(shiftticks)

    while True:
        y.next = ~y.value
        await sleep(onticks)

        y.next = ~y.value
        await sleep(offticks)
