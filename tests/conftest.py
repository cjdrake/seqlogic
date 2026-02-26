"""PyTest Configuration"""

import pytest
from deltacycle import get_running_kernel

msgs = set[tuple[int, str, str]]()


def trace(msg: str):
    try:
        kernel = get_running_kernel()
    except RuntimeError:
        time = -1
        task_name = ""
    else:
        time = kernel.time()
        task = kernel.task()
        task_name = task.name
    msgs.add((time, task_name, msg))


@pytest.fixture
def captrace():
    yield msgs
    msgs.clear()
