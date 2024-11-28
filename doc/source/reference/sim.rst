************************
    Event Simulation
************************

.. autoexception:: seqlogic.CancelledError
.. autoexception:: seqlogic.FinishError
.. autoexception:: seqlogic.InvalidStateError

.. autoclass:: seqlogic.Region

.. TODO(cjdrake): Should seqlogic.sim.State be public?
.. TODO(cjdrake): Should seqlogic.sim.Value be public?

.. autoclass:: seqlogic.Singular
.. autoclass:: seqlogic.Aggregate
.. autoclass:: seqlogic.AggrValue

.. autoclass:: seqlogic.TaskState

    .. autoattribute:: seqlogic.TaskState.CREATED
    .. autoattribute:: seqlogic.TaskState.WAIT_FIFO
    .. autoattribute:: seqlogic.TaskState.WAIT_STATE
    .. autoattribute:: seqlogic.TaskState.PENDING
    .. autoattribute:: seqlogic.TaskState.RUNNING
    .. autoattribute:: seqlogic.TaskState.RETURNED
    .. autoattribute:: seqlogic.TaskState.EXCEPTED
    .. autoattribute:: seqlogic.TaskState.CANCELLED

.. autoclass:: seqlogic.Task

    .. autoproperty:: seqlogic.Task.region
    .. automethod:: seqlogic.Task.set_state
    .. automethod:: seqlogic.Task.run
    .. automethod:: seqlogic.Task.done
    .. automethod:: seqlogic.Task.cancelled
    .. automethod:: seqlogic.Task.set_result
    .. automethod:: seqlogic.Task.result
    .. automethod:: seqlogic.Task.set_exception
    .. automethod:: seqlogic.Task.exception
    .. automethod:: seqlogic.Task.get_coro
    .. automethod:: seqlogic.Task.cancel

.. autoclass:: seqlogic.TaskGroup

    .. automethod:: seqlogic.TaskGroup.create_task

.. autoclass:: seqlogic.Event

    .. automethod:: seqlogic.Event.wait
    .. automethod:: seqlogic.Event.set
    .. automethod:: seqlogic.Event.clear
    .. automethod:: seqlogic.Event.is_set

.. autoclass:: seqlogic.Semaphore

    .. automethod:: seqlogic.Semaphore.acquire
    .. automethod:: seqlogic.Semaphore.try_acquire
    .. automethod:: seqlogic.Semaphore.release
    .. automethod:: seqlogic.Semaphore.locked

.. autoclass:: seqlogic.BoundedSemaphore
    :show-inheritance:

    .. automethod:: seqlogic.Semaphore.release

.. autoclass:: seqlogic.Lock
    :show-inheritance:

.. autoclass:: seqlogic.EventLoop

    .. automethod:: seqlogic.EventLoop.clear
    .. automethod:: seqlogic.EventLoop.restart
    .. automethod:: seqlogic.EventLoop.time
    .. automethod:: seqlogic.EventLoop.task

    .. TODO(cjdrake): Advertise usage of low-level API?

    .. automethod:: seqlogic.EventLoop.run
    .. automethod:: seqlogic.EventLoop.irun

.. autofunction:: seqlogic.get_running_loop
.. autofunction:: seqlogic.get_event_loop
.. autofunction:: seqlogic.set_event_loop
.. autofunction:: seqlogic.new_event_loop
.. autofunction:: seqlogic.del_event_loop

.. autofunction:: seqlogic.now

.. autofunction:: seqlogic.run
.. autofunction:: seqlogic.irun

.. autofunction:: seqlogic.sleep
.. autofunction:: seqlogic.changed
.. autofunction:: seqlogic.resume

.. autofunction:: seqlogic.wait

.. autofunction:: seqlogic.finish
