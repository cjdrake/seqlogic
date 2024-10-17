"""Simulation using async/await.

We are intentionally imitating the style of Python's Event Loop API:
https://docs.python.org/3/library/asyncio-eventloop.html

Credit to David Beazley's "Build Your Own Async" tutorial for inspiration:
https://www.youtube.com/watch?v=Y4Gt3Xjd7G8
"""

from __future__ import annotations

import heapq
from abc import ABC
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Coroutine, Generator, Hashable
from enum import IntEnum, auto

INIT_TIME = -1
START_TIME = 0

type Predicate = Callable[[], bool]


class Region(IntEnum):
    # Coroutines that react to changes from Active region.
    # Used by combinational logic.
    REACTIVE = auto()

    # Coroutines that drive changes to model state.
    # Used by 1) testbench, and 2) sequential logic.
    ACTIVE = auto()

    # Coroutines that monitor model state.
    INACTIVE = auto()


class State(ABC):
    """Model component."""

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
        _loop.touch(self)

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
        _loop.touch(self)

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


class Task(Awaitable):
    """Coroutine wrapper."""

    def __init__(self, coro: Coroutine, region: Region = Region.ACTIVE):
        self._region = region
        self._coro = coro
        self._result = None
        self._done = False

    def __await__(self) -> Generator[None, None, None]:
        if not self._done:
            _loop.task_await(self)
            # Suspend
            yield
        # Resume
        return

    @property
    def region(self):
        return self._region

    @property
    def coro(self):
        return self._coro

    def _get_result(self):
        return self._result

    def _set_result(self, value):
        self._result = value

    result = property(fget=_get_result, fset=_set_result)

    def set_done(self):
        self._done = True

    def done(self) -> bool:
        return self._done


def create_task(coro: Coroutine, region: Region = Region.ACTIVE) -> Task:
    return _loop.task_create(coro, region)


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


type _TaskQueueItem = tuple[int, Task, State | None]


class _TaskQueue:
    """Priority queue for ordering task execution."""

    def __init__(self):
        # time, region, index, task, state
        self._items: list[tuple[int, Region, int, Task, State | None]] = []

        # Monotonically increasing integer to break ties in the heapq
        self._index: int = 0

    def __bool__(self) -> bool:
        return bool(self._items)

    def clear(self):
        self._items.clear()
        self._index = 0

    def push(self, time: int, task: Task, state: State | None = None):
        item = (time, task.region, self._index, task, state)
        heapq.heappush(self._items, item)
        self._index += 1

    def peek(self) -> _TaskQueueItem:
        time, _, _, task, state = self._items[0]
        return (time, task, state)

    def pop(self) -> _TaskQueueItem:
        time, _, _, task, state = heapq.heappop(self._items)
        return (time, task, state)

    def pop_region(self) -> Generator[_TaskQueueItem, None, None]:
        time, region, _, task, state = heapq.heappop(self._items)
        yield (time, task, state)
        while self._items:
            t, r, _, task, state = self._items[0]
            if t == time and r == region:
                heapq.heappop(self._items)
                yield (time, task, state)
            else:
                break


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


class Finish(Exception):
    """Force the simulation to stop."""


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

    def _pending(self):
        return [
            self._queue,
            self._waiting,
            self._predicates,
            self._touched,
            self._task_waiting,
            self._event_waiting,
            self._sem_waiting,
        ]

    def clear(self):
        for tasks in self._pending():
            tasks.clear()

    def restart(self):
        """Restart current simulation."""
        self._time = INIT_TIME
        self._task = None
        self.clear()

    def time(self) -> int:
        return self._time

    def task(self) -> Task | None:
        return self._task

    def start(self, task: Task):
        self._queue.push(START_TIME, task)

    def set_timeout(self, delay: int):
        """Schedule current coroutine after delay."""
        self._queue.push(self._time + delay, self._task)

    # State suspend / resume callbacks
    def set_trigger(self, state: State, predicate: Predicate):
        """Schedule current coroutine after a state update trigger."""
        self._waiting[state].add(self._task)
        self._predicates[state][self._task] = predicate

    def touch(self, state: State):
        """Schedule coroutines triggered by touching model state."""
        waiting = self._waiting[state]
        predicates = self._predicates[state]
        pending = [task for task in waiting if predicates[task]()]
        for task in pending:
            self._queue.push(self._time, task, state)
            self._waiting[state].remove(task)
            del self._predicates[state][task]
        # Add state to update set
        self._touched.add(state)

    def _update(self):
        while self._touched:
            state = self._touched.pop()
            state.update()

    # Task await / done callbacks
    def task_create(self, coro: Coroutine, region: Region = Region.ACTIVE) -> Task:
        # Cannot call task_create before the simulation starts
        assert self._time >= 0
        task = Task(coro, region)
        self._queue.push(self._time, task)
        return task

    def task_await(self, task: Task):
        self._task_waiting[task].append(self._task)

    def task_done(self, task: Task):
        waiting = self._task_waiting[task]
        while waiting:
            self._queue.push(self._time, waiting.popleft())
        task.set_done()

    # Event wait / set callbacks
    def event_wait(self, event: Event):
        self._event_waiting[event].append(self._task)

    def event_set(self, event: Event):
        waiting = self._event_waiting[event]
        while waiting:
            self._queue.push(self._time, waiting.popleft())

    # Semaphore acquire / release callbacks
    def sem_acquire(self, sem: Semaphore):
        self._sem_waiting[sem].append(self._task)

    def sem_release(self, sem: Semaphore) -> bool:
        waiting = self._sem_waiting[sem]
        if waiting:
            self._queue.push(self._time, waiting.popleft())
            # Do NOT increment semaphore counter
            return False
        # Increment semaphore counter
        return True

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
            # Peek at when next event is scheduled
            time, _, _ = self._queue.peek()

            # Protect against time traveling tasks
            assert time >= self._time

            # Next task scheduled: future time slot
            if time > self._time:
                # Update simulation state
                self._update()

                # Exit if we hit the run limit
                if limit is not None and time >= limit:
                    break
                # Otherwise, advance to new timeslot
                self._time = time

            # Resume execution
            for _, task, state in self._queue.pop_region():
                self._task = task
                try:
                    task.coro.send(state)
                except StopIteration as e:
                    task.result = e.value
                    self.task_done(task)

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
        except Finish:
            self.clear()

    def _iter_kernel(self, limit: int | None) -> Generator[int, None, None]:
        while self._queue:
            # Peek at when next event is scheduled
            time, _, _ = self._queue.peek()

            # Protect against time traveling tasks
            assert time >= self._time

            # Next task scheduled: future time slot
            if time > self._time:
                # Update simulation state
                self._update()
                yield self._time

                # Exit if we hit the run limit
                if limit is not None and time >= limit:
                    return
                # Otherwise, advance to new timeslot
                self._time = time

            # Resume execution
            for _, task, state in self._queue.pop_region():
                self._task = task
                try:
                    task.coro.send(state)
                except StopIteration as e:
                    task.result = e.value
                    self.task_done(task)

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
        except Finish:
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
        _loop.start(task)

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
        _loop.start(task)

    yield from _loop.irun(ticks, until)


async def sleep(delay: int):
    """Suspend the task, and wake up after a delay."""
    _loop.set_timeout(delay)
    await _TaskAwaitable()


async def changed(*states: State) -> State:
    """Resume execution upon state change."""
    for state in states:
        _loop.set_trigger(state, state.changed)
    state = await _StateAwaitable()
    return state


async def resume(*triggers: tuple[State, Predicate]) -> State:
    """Resume execution upon event."""
    for state, predicate in triggers:
        _loop.set_trigger(state, predicate)
    state = await _StateAwaitable()
    return state


def finish():
    raise Finish()
