"""Process Interface"""


class ProcIf:
    """Process interface.

    Implemented by components that contain local simulator processes.
    """

    def __init__(self):
        self._reactive: list = []
        self._active: list = []
        self._inactive: list = []

    @property
    def reactive(self) -> list:
        return self._reactive

    @property
    def active(self) -> list:
        return self._active

    @property
    def inactive(self) -> list:
        return self._inactive
