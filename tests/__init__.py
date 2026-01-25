"""Prepend workspace root to system path."""

import logging
import sys
from pathlib import Path

from deltacycle import get_running_kernel

WORKSPACE = Path(__file__).parents[1]

sys.path.insert(0, str(WORKSPACE / "src"))

# Customize logging
logger = logging.getLogger("deltacycle")


class DeltaCycleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            kernel = get_running_kernel()
        except RuntimeError:
            record.time = -1
            record.taskName = None
        else:
            record.time = kernel.time()
            record.taskName = kernel.task().name
        return True


logger.addFilter(DeltaCycleFilter())
