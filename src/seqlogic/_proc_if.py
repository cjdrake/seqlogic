"""Process Interface"""

from collections.abc import Callable
from typing import Any

from deltacycle import TaskCoro

type ProcItem = tuple[Callable[..., TaskCoro], tuple[Any, ...], dict[str, Any]]


class ProcIf:
    """Process interface.

    Implemented by components that contain local simulator processes.
    """

    def __init__(self):
        self._reactive: list[ProcItem] = []
        self._active: list[ProcItem] = []
        self._inactive: list[ProcItem] = []

    @property
    def reactive(self) -> list[ProcItem]:
        return self._reactive

    @property
    def active(self) -> list[ProcItem]:
        return self._active

    @property
    def inactive(self) -> list[ProcItem]:
        return self._inactive
