"""Simulation using async/await.

We are intentionally imitating the style of Python's Event Loop API:
https://docs.python.org/3/library/asyncio-eventloop.html

Credit to David Beazley's "Build Your Own Async" tutorial for inspiration:
https://www.youtube.com/watch?v=Y4Gt3Xjd7G8
"""

from __future__ import annotations

import heapq
from abc import ABC
from collections import defaultdict, deque, namedtuple
from collections.abc import Awaitable, Callable, Coroutine, Generator, Hashable
from enum import IntEnum, auto

INIT_TIME = -1
START_TIME = 0

type Predicate = Callable[[], bool]


class CancelledError(Exception):
    """Task has been cancelled."""


class FinishError(Exception):
    """Force the simulation to stop."""


class InvalidStateError(Exception):
    """Task has an invalid state."""


class Region(IntEnum):
    # Coroutines that react to changes from Active region.
    # Used by combinational logic.
    REACTIVE = auto()

    # Coroutines that drive changes to model state.
    # Used by 1) testbench, and 2) sequential logic.
    ACTIVE = auto()

    # Coroutines that monitor model state.
    INACTIVE = auto()


class State(Awaitable):
    """Model component."""

    def __await__(self) -> Generator[None, State, State]:
        _loop.state_wait(self, self.changed)

        # Suspend
        state = yield
        assert state is self

        # Resume
        return state

    state = property(fget=NotImplemented)

    def changed(self) -> bool:
        raise NotImplementedError()  # pragma: no cover

    def update(self):
        raise NotImplementedError()  # pragma: no cover


class Value(ABC):
    """State value."""

    def _get_value(self):
        raise NotImplementedError()  # pragma: no cover

    value = property(fget=_get_value)

    def _set_next(self, value):
        raise NotImplementedError()  # pragma: no cover

    next = property(fset=_set_next)


class Singular(State, Value):
    """Model state organized as a single unit."""

    def __init__(self, value):
        self._value = value
        self._next_value = value
        self._changed = False

    def _get_value(self):
        task = _loop.task()
        if task is not None and task.region == Region.REACTIVE:
            return self._next_value
        return self._value

    value = property(fget=_get_value)
    state = property(fget=_get_value)

    def _set_next(self, value):
        self._changed = value != self._next_value
        self._next_value = value

        # Notify the event loop
        _loop.state_touch(self)

    next = property(fset=_set_next)

    def changed(self) -> bool:
        return self._changed

    def update(self):
        self._value = self._next_value
        self._changed = False

    def dirty(self) -> bool:
        return self._next_value != self._value


class Aggregate(State):
    """Model state organized as multiple units."""

    def __init__(self, value):
        super().__init__()
        self._values = defaultdict(lambda: value)
        self._next_values = defaultdict(lambda: value)
        self._changed: set[Hashable] = set()

    def __getitem__(self, key: Hashable) -> AggrValue:
        def fget():
            values = self._get_values()
            return values[key]

        def fset(value):
            self._set_next(key, value)

        return AggrValue(fget, fset)

    def _get_values(self):
        task = _loop.task()
        if task is not None and task.region == Region.REACTIVE:
            return self._next_values.copy()
        return self._values.copy()

    state = property(fget=_get_values)

    def _set_next(self, key: Hashable, value):
        if value != self._next_values[key]:
            self._changed.add(key)
        self._next_values[key] = value

        # Notify the event loop
        _loop.state_touch(self)

    def changed(self) -> bool:
        return bool(self._changed)

    def update(self):
        for key in self._changed:
            self._values[key] = self._next_values[key]
        self._changed.clear()


class AggrValue(Value):
    """Wrap Aggregate value getter/setter."""

    def __init__(self, fget, fset):
        self._fget = fget
        self._fset = fset

    def _get_value(self):
        return self._fget()

    value = property(fget=_get_value)

    def _set_next(self, value):
        self._fset(value)

    next = property(fset=_set_next)


class TaskState(IntEnum):
    """Task State.

               +--------------------------+
               |                          |
               v                          |
    CREATED -> PENDING -> RUNNING -> WAIT_*
                                  -> CANCELLED
                                  -> EXCEPTED
                                  -> RETURNED
    """

    CREATED = auto()

    WAIT_STATE = auto()
    WAIT_TASK = auto()
    WAIT_EVENT = auto()
    WAIT_SEM = auto()

    PENDING = auto()
    RUNNING = auto()

    RETURNED = auto()
    EXCEPTED = auto()
    CANCELLED = auto()


class Task(Awaitable):
    """Coroutine wrapper."""

    def __init__(self, coro: Coroutine, region: Region = Region.ACTIVE):
        self._coro = coro
        self._region = region

        self._state = TaskState.CREATED
        self._wkey: State | Task | Event | Semaphore | None = None

        self._result = None

        # Set flag to throw exception
        self._exc_flag = False
        self._exception = None

    def __await__(self) -> Generator[None, None, None]:
        if not self.done():
            _loop.task_await(self)
            # Suspend
            yield
        # Resume
        return

    @property
    def region(self):
        return self._region

    def set_state(self, state: TaskState, wkey=None):
        self._state = state
        self._wkey = wkey

    def run(self, value=None):
        self._state = TaskState.RUNNING
        if self._exc_flag:
            self._exc_flag = False
            self._coro.throw(self._exception)
        else:
            self._coro.send(value)

    def done(self) -> bool:
        return self._state in {
            TaskState.CANCELLED,
            TaskState.EXCEPTED,
            TaskState.RETURNED,
        }

    def cancelled(self) -> bool:
        return self._state == TaskState.CANCELLED

    def set_result(self, result):
        self._result = result

    def result(self):
        match self._state:
            case TaskState.CANCELLED:
                assert self._exception is not None and isinstance(self._exception, CancelledError)
                raise self._exception
            case TaskState.EXCEPTED:
                assert self._exception is not None
                raise self._exception  # Re-raise exception
            case TaskState.RETURNED:
                assert self._exception is None
                return self._result
            case _:
                raise InvalidStateError("Task is not done")

    def set_exception(self, exc):
        self._exc_flag = True
        self._exception = exc

    def exception(self):
        match self._state:
            case TaskState.CANCELLED:
                assert self._exception is not None and isinstance(self._exception, CancelledError)
                raise self._exception
            case TaskState.EXCEPTED:
                assert self._exception is not None
                return self._exception  # Return exception
            case TaskState.RETURNED:
                assert self._exception is None
                return self._exception
            case _:
                raise InvalidStateError("Task is not done")

    def get_coro(self) -> Coroutine:
        return self._coro

    def cancel(self, msg: str | None = None):
        match self._state:
            case TaskState.WAIT_STATE:
                _loop.state_drop(self._wkey, self)
            case TaskState.WAIT_TASK:
                _loop.task_drop(self._wkey, self)
            case TaskState.WAIT_EVENT:
                _loop.event_drop(self._wkey, self)
            case TaskState.WAIT_SEM:
                _loop.sem_drop(self._wkey, self)
            case TaskState.PENDING:
                _loop.drop(self)
            case _:
                raise ValueError("Task is not WAITING or PENDING")

        args = () if msg is None else (msg,)
        exc = CancelledError(*args)
        self.set_exception(exc)
        _loop.call_soon(self)


def create_task(coro: Coroutine, region: Region = Region.ACTIVE) -> Task:
    return _loop.task_create(coro, region)


class TaskGroup:
    """Group of tasks."""

    def __init__(self):
        self._tasks = deque()

    def create_task(self, coro: Coroutine, region: Region = Region.ACTIVE) -> Task:
        task = _loop.task_create(coro, region)
        self._tasks.append(task)
        return task

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        while self._tasks:
            task = self._tasks.popleft()
            await task


class Event:
    """Notify multiple tasks that some event has happened."""

    def __init__(self):
        self._flag = False

    async def wait(self):
        if not self._flag:
            _loop.event_wait(self)
            await _TaskAwaitable()

    def set(self):
        _loop.event_set(self)
        self._flag = True

    def clear(self):
        self._flag = False

    def is_set(self) -> bool:
        return self._flag


class Semaphore:
    """Semaphore to synchronize tasks."""

    def __init__(self, value: int = 1):
        if value < 1:
            raise ValueError(f"Expected value >= 1, got {value}")
        self._value = value
        self._cnt = value

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        self.release()

    async def acquire(self):
        assert self._cnt >= 0
        if self._cnt == 0:
            _loop.sem_acquire(self)
            await _TaskAwaitable()
        else:
            self._cnt -= 1

    def try_acquire(self) -> bool:
        assert self._cnt >= 0
        if self._cnt == 0:
            return False
        self._cnt -= 1
        return True

    def release(self):
        assert self._cnt >= 0
        increment = _loop.sem_release(self)
        if increment:
            if self._cnt == self._value:
                raise RuntimeError("Cannot release")
            self._cnt += 1

    def locked(self) -> bool:
        return self._cnt == 0


class Lock(Semaphore):
    """Mutex lock to synchronize tasks."""

    def __init__(self):
        super().__init__(value=1)


_TaskQueueItem = namedtuple("_TaskQueueItem", ["time", "region", "index", "task", "value"])


class _TaskQueue:
    """Priority queue for ordering task execution."""

    def __init__(self):
        # time, region, index, task, value
        self._items: list[_TaskQueueItem] = []

        # Monotonically increasing integer to break ties in the heapq
        self._index: int = 0

    def __bool__(self) -> bool:
        return bool(self._items)

    def clear(self):
        self._items.clear()
        self._index = 0

    def push(self, time: int, task: Task, value: State | None = None):
        item = _TaskQueueItem(time, task.region, self._index, task, value)
        heapq.heappush(self._items, item)
        self._index += 1

    def peek(self) -> tuple[int, Task, State | None]:
        time, _, _, task, value = self._items[0]
        return (time, task, value)

    def pop(self) -> tuple[int, Task, State | None]:
        time, _, _, task, value = heapq.heappop(self._items)
        return (time, task, value)

    def pop_time(self) -> Generator[tuple[int, Task, State | None], None, None]:
        time, _, _, task, value = heapq.heappop(self._items)
        yield (time, task, value)
        while self._items:
            t, _, _, task, value = self._items[0]
            if t == time:
                heapq.heappop(self._items)
                yield (time, task, value)
            else:
                break

    def drop(self, task: Task):
        for i, item in enumerate(self._items):
            if item.task is task:
                index = i
                break
        else:
            raise ValueError("Task is not scheduled")
        self._items.pop(index)


class _TaskAwaitable(Awaitable):
    """Suspend execution of the current task."""

    def __await__(self) -> Generator[None, None, None]:
        # Suspend
        yield
        # Resume
        return


class _StateAwaitable(Awaitable):
    """Suspend execution of the current task."""

    def __await__(self) -> Generator[None, State, State]:
        # Suspend
        state = yield
        # Resume
        return state


class EventLoop:
    """Simulation event loop."""

    def __init__(self):
        """Initialize simulation."""
        # Simulation time
        self._time: int = INIT_TIME
        # Task queue
        self._queue = _TaskQueue()
        # Currently executing task
        self._task: Task | None = None
        # State waiting set
        self._waiting: dict[State, set[Task]] = defaultdict(set)
        self._predicates: dict[State, dict[Task, Predicate]] = defaultdict(dict)
        self._touched: set[State] = set()
        # Task waiting list
        self._task_waiting: dict[Task, deque[Task]] = defaultdict(deque)
        # Event waiting list
        self._event_waiting: dict[Event, deque[Task]] = defaultdict(deque)
        # Semaphore/Lock waiting list
        self._sem_waiting: dict[Semaphore, deque[Task]] = defaultdict(deque)

    def clear(self):
        """Clear all task collections."""
        self._queue.clear()
        self._waiting.clear()
        self._predicates.clear()
        self._touched.clear()
        self._task_waiting.clear()
        self._event_waiting.clear()
        self._sem_waiting.clear()

    def restart(self):
        """Restart current simulation."""
        self._time = INIT_TIME
        self._task = None
        self.clear()

    def time(self) -> int:
        return self._time

    def task(self) -> Task | None:
        return self._task

    # Scheduling methods
    def _schedule(self, time: int, task: Task, value):
        task.set_state(TaskState.PENDING)
        self._queue.push(time, task, value)

    def call_soon(self, task: Task, value=None):
        self._schedule(self._time, task, value)

    def call_later(self, delay: int, task: Task, value=None):
        self._schedule(self._time + delay, task, value)

    def call_at(self, when: int, task: Task, value=None):
        self._schedule(when, task, value)

    def drop(self, task: Task):
        self._queue.drop(task)

    # State suspend / resume callbacks
    def state_wait(self, state: State, predicate: Predicate):
        """Schedule current coroutine after a state update trigger."""
        task = self._task
        task.set_state(TaskState.WAIT_STATE, state)
        self._waiting[state].add(task)
        self._predicates[state][task] = predicate

    def state_touch(self, state: State):
        """Schedule coroutines triggered by touching model state."""
        waiting = self._waiting[state]
        predicates = self._predicates[state]
        pending = [task for task in waiting if predicates[task]()]
        for task in pending:
            self.state_drop(state, task)
            self.call_soon(task, state)
        # Add state to update set
        self._touched.add(state)

    def state_drop(self, state: State, task: Task):
        self._waiting[state].remove(task)
        del self._predicates[state][task]

    def _state_update(self):
        while self._touched:
            state = self._touched.pop()
            state.update()

    # Task await / done callbacks
    def task_create(self, coro: Coroutine, region: Region = Region.ACTIVE) -> Task:
        # Cannot call task_create before the simulation starts
        assert self._time >= 0
        task = Task(coro, region)
        self.call_soon(task)
        return task

    def task_await(self, ptask: Task):
        ctask = self._task
        ctask.set_state(TaskState.WAIT_TASK, ptask)
        self._task_waiting[ptask].append(ctask)

    def task_drop(self, ptask: Task, ctask: Task):
        self._task_waiting[ptask].remove(ctask)

    def _task_return(self, ptask: Task, result):
        waiting = self._task_waiting[ptask]
        while waiting:
            ctask = waiting.popleft()
            self.call_soon(ctask)
        ptask.set_state(TaskState.RETURNED)
        ptask.set_result(result)

    def _task_cancel(self, ptask: Task, e: CancelledError):
        waiting = self._task_waiting[ptask]
        while waiting:
            ctask = waiting.popleft()
            ctask.set_exception(e)
            self.call_soon(ctask)
        ptask.set_state(TaskState.CANCELLED)

    def _task_except(self, ptask: Task, e: Exception):
        waiting = self._task_waiting[ptask]
        while waiting:
            ctask = waiting.popleft()
            ctask.set_exception(e)
            self.call_soon(ctask)
        ptask.set_state(TaskState.EXCEPTED)

    # Event wait / set callbacks
    def event_wait(self, event: Event):
        task = self._task
        task.set_state(TaskState.WAIT_EVENT, event)
        self._event_waiting[event].append(task)

    def event_set(self, event: Event):
        waiting = self._event_waiting[event]
        while waiting:
            task = waiting.popleft()
            self.call_soon(task)

    def event_drop(self, event: Event, task: Task):
        self._event_waiting[event].remove(task)

    # Semaphore acquire / release callbacks
    def sem_acquire(self, sem: Semaphore):
        task = self._task
        task.set_state(TaskState.WAIT_SEM, sem)
        self._sem_waiting[sem].append(task)

    def sem_release(self, sem: Semaphore) -> bool:
        waiting = self._sem_waiting[sem]
        if waiting:
            task = waiting.popleft()
            self.call_soon(task)
            # Do NOT increment semaphore counter
            return False
        # Increment semaphore counter
        return True

    def sem_drop(self, sem: Semaphore, task: Task):
        self._sem_waiting[sem].remove(task)

    def _limit(self, ticks: int | None, until: int | None) -> int | None:
        """Determine the run limit."""
        match ticks, until:
            # Run until no tasks left
            case None, None:
                return None
            # Run until an absolute time
            case None, int():
                return until
            # Run until a number of ticks in the future
            case int(), None:
                return max(START_TIME, self._time) + ticks
            case _:
                s = "Expected either ticks or until to be int | None"
                raise TypeError(s)

    def _run_kernel(self, limit: int | None):
        while self._queue:
            # Peek when next event is scheduled
            time, _, _ = self._queue.peek()

            # Protect against time traveling tasks
            assert time > self._time

            # Exit if we hit the run limit
            if limit is not None and time >= limit:
                break

            # Otherwise, advance to new timeslot
            self._time = time

            # Execute time slot
            for _, task, value in self._queue.pop_time():
                self._task = task
                try:
                    task.run(value)
                except StopIteration as e:
                    self._task_return(task, e.value)
                except CancelledError as e:
                    self._task_cancel(task, e)
                except Exception as e:
                    self._task_except(task, e)
                    raise

            # Update simulation state
            self._state_update()

    def run(self, ticks: int | None = None, until: int | None = None):
        """Run the simulation.

        Until:
        1. We hit the runlimit, OR
        2. There are no tasks left in the queue
        """
        limit = self._limit(ticks, until)

        # Run until either 1) all tasks complete, or 2) finish()
        try:
            self._run_kernel(limit)
        except FinishError:
            self.clear()

    def _iter_kernel(self, limit: int | None) -> Generator[int, None, None]:
        while self._queue:
            # Peek when next event is scheduled
            time, _, _ = self._queue.peek()

            # Protect against time traveling tasks
            assert time > self._time

            # Exit if we hit the run limit
            if limit is not None and time >= limit:
                return

            # Otherwise, advance to new timeslot
            self._time = time

            # Execute time slot
            for _, task, value in self._queue.pop_time():
                self._task = task
                try:
                    task.run(value)
                except StopIteration as e:
                    self._task_return(task, e.value)
                except CancelledError as e:
                    self._task_cancel(task, e)
                except Exception as e:
                    self._task_except(task, e)
                    raise

            # Update simulation state
            self._state_update()
            yield self._time

    def irun(
        self, ticks: int | None = None, until: int | None = None
    ) -> Generator[int, None, None]:
        """Iterate the simulation.

        Until:
        1. We hit the runlimit, OR
        2. There are no tasks left in the queue
        """
        limit = self._limit(ticks, until)

        try:
            yield from self._iter_kernel(limit)
        except FinishError:
            self.clear()


_loop: EventLoop | None = None


def get_running_loop() -> EventLoop:
    if _loop is None:
        raise RuntimeError("No running loop")
    return _loop


def get_event_loop() -> EventLoop | None:
    """Get the current event loop."""
    return _loop


def set_event_loop(loop: EventLoop):
    """Set the current event loop."""
    global _loop
    _loop = loop


def new_event_loop() -> EventLoop:
    """Create and return a new event loop."""
    return EventLoop()


def del_event_loop():
    """Delete the current event loop."""
    global _loop
    _loop = None


def now() -> int:
    if _loop is None:
        raise RuntimeError("No running loop")
    return _loop.time()


def run(
    coro: Coroutine | None = None,
    region: Region = Region.ACTIVE,
    loop: EventLoop | None = None,
    ticks: int | None = None,
    until: int | None = None,
):
    """Run a simulation."""
    global _loop

    task = None
    if loop is not None:
        _loop = loop
    else:
        _loop = EventLoop()
        task = Task(coro, region)
        _loop.call_at(START_TIME, task)

    _loop.run(ticks, until)


def irun(
    coro: Coroutine | None = None,
    region: Region = Region.ACTIVE,
    loop: EventLoop | None = None,
    ticks: int | None = None,
    until: int | None = None,
) -> Generator[int, None, None]:
    """Iterate a simulation."""
    global _loop

    task = None
    if loop is not None:
        _loop = loop
    else:
        _loop = EventLoop()
        task = Task(coro, region)
        _loop.call_at(START_TIME, task)

    yield from _loop.irun(ticks, until)


async def sleep(delay: int):
    """Suspend the task, and wake up after a delay."""
    _loop.call_later(delay, _loop.task())
    await _TaskAwaitable()


async def changed(*states: State) -> State:
    """Resume execution upon state change."""
    for state in states:
        _loop.state_wait(state, state.changed)
    state = await _StateAwaitable()
    return state


async def resume(*triggers: tuple[State, Predicate]) -> State:
    """Resume execution upon event."""
    for state, predicate in triggers:
        _loop.state_wait(state, predicate)
    state = await _StateAwaitable()
    return state


def finish():
    raise FinishError()
