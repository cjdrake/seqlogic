"""Globals: clocks and resets."""

from ..design import Active, Packed
from ..sim import sleep


async def drv_reset(
    y: Packed,
    offticks: int = 0,
    onticks: int = 1,
    active: Active = Active.POS,
):
    r"""
    Drive a reset signal.

    Active Positive:

    ________/‾‾‾‾‾‾‾‾\\________

    Active Negative:

    ‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾

    Args:

    Raises:
        ValueError: Invalid parameter values
    """
    if offticks < 0:
        raise ValueError(f"Expected offticks ≥ 0, got {offticks}")
    if onticks < 0:
        raise ValueError(f"Expected onticks ≥ 0, got {onticks}")

    # T = 0
    match active:
        case Active.NEG:
            y.next = "1b1"
        case Active.POS:
            y.next = "1b0"
        case _:
            assert False

    await sleep(offticks)

    y.next = ~y.value
    await sleep(onticks)

    y.next = ~y.value


async def drv_clock(
    y: Packed,
    shiftticks: int = 0,
    onticks: int = 1,
    offticks: int = 1,
    active: Active = Active.POS,
):
    r"""
    Drive a clock signal.

    Active Positive:

    ________/‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\________

    Active Negative:

    ‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\

    Args:

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
    match active:
        case Active.NEG:
            y.next = "1b1"
        case Active.POS:
            y.next = "1b0"
        case _:
            assert False

    await sleep(shiftticks)

    while True:
        y.next = ~y.value
        await sleep(onticks)

        y.next = ~y.value
        await sleep(offticks)
