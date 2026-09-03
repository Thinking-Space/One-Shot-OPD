# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Time limits for the math graders that survive being called off the main thread.

The graders guard their sympy calls with ``signal.SIGALRM``, which was fine when
scoring ran synchronously on the main thread. verl 0.9.0's ``RewardLoopWorker``
dispatches through ``loop.run_in_executor``, so they now run on a pool thread,
and ``signal.signal`` raises ``ValueError: signal only works in main thread``
there — killing the whole training step, not just the one grade.

The code graders are unaffected: they isolate execution in a
``multiprocessing.Process``, whose main thread can still take SIGALRM.
"""

import functools
import signal
import threading

__all__ = ["run_with_timeout", "timeout_ours", "timeout"]


def run_with_timeout(fn, seconds: int, *args, **kwargs):
    """Run ``fn`` under a time limit, raising ``TimeoutError`` if it overruns.

    On the main thread this is SIGALRM, which genuinely interrupts the call. Off
    it, the work goes to a helper thread and is *abandoned* on timeout: Python
    cannot interrupt a thread stuck in sympy, so the caller returns on time and
    the helper runs on until its call finishes. The thread is a daemon and is
    not pooled, so an abandoned one delays neither process exit nor later calls.

    Abandoning a thread is worth it over simply dropping the limit off the main
    thread. Pathological latex can wedge sympy for a long time, and a reward
    worker stuck on one row stalls the training step — which is the exact
    failure these limits were added to prevent.
    """
    if threading.current_thread() is threading.main_thread():

        def handler(signum, frame):
            raise TimeoutError("Operation timed out!")

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    box: dict = {}

    def target():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = e

    worker = threading.Thread(target=target, daemon=True, name="verl-reward-timeout")
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError("Operation timed out!")
    if "error" in box:
        raise box["error"]
    return box["value"]


def timeout_ours(timeout_seconds: int = 8):
    """Decorator form: ``@timeout_ours(timeout_seconds=5)``."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return run_with_timeout(func, timeout_seconds, *args, **kwargs)

        return wrapper

    return decorator


class timeout:
    """Context-manager form, for ``with timeout(1): ...``.

    Only enforces the limit on the main thread. A block, unlike a callable,
    cannot be handed to another thread, so off the main thread this is a no-op
    and the caller is responsible for keeping the block bounded. Prefer
    ``run_with_timeout`` on a function wherever the block can be extracted.
    """

    def __init__(self, seconds: int = 1, error_message: str = "Timeout"):
        self.seconds = seconds
        self.error_message = error_message
        self._armed = False
        self._old_handler = None

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            return self
        self._old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
        self._armed = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._armed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old_handler)
            self._armed = False
        return False
