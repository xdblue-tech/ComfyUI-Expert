"""Bound provider calls at the model boundary, where the cost is actually incurred.

The executor and step callback bound generated *code*; they cannot stop a model
call that never returns, and counting agent runs does not equal counting billed
provider requests (one run makes one request per step plus the final-answer
request). :class:`BudgetedModel` wraps any SmolAgents model with a shared
request counter and the remaining wall-clock budget.
"""

import threading

from smolagents.models import Model


class ModelBudgetExceeded(RuntimeError):
    """A provider request would exceed the run's request or wall-clock budget."""


class ModelBudget:
    """A request counter shared by every model call in one run or one matrix."""

    def __init__(self, max_requests=None):
        if max_requests is not None and max_requests < 1:
            raise ValueError("max_requests must be at least 1 when set")
        self.max_requests = max_requests
        self.requests = 0

    def reserve(self) -> int:
        if self.max_requests is not None and self.requests >= self.max_requests:
            raise ModelBudgetExceeded(
                f"model request budget of {self.max_requests} is exhausted; no further "
                "provider call was made. Raise the budget or narrow the task"
            )
        self.requests += 1
        return self.requests


class BudgetedModel(Model):
    """A SmolAgents model proxy that counts requests and enforces the deadline.

    ``deadline`` is a callable returning the remaining wall-clock seconds, or
    ``None``. A call that outlives the remaining budget raises instead of
    blocking the run: the provider thread is abandoned as a daemon, so the host
    process can still exit.
    """

    def __init__(self, model: Model, budget: ModelBudget, deadline=None):
        self._inner = model
        self._budget = budget
        self._deadline = deadline
        self.requests = 0

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        self._budget.reserve()
        self.requests += 1
        timeout = self._remaining_seconds()
        arguments = {
            "stop_sequences": stop_sequences,
            "response_format": response_format,
            "tools_to_call_from": tools_to_call_from,
            **kwargs,
        }
        call = lambda: self._inner.generate(messages, **arguments)  # noqa: E731
        if timeout is None:
            return call()
        return _call_with_timeout(call, timeout)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _remaining_seconds(self):
        if self._deadline is None:
            return None
        remaining = self._deadline()
        if remaining <= 0:
            raise ModelBudgetExceeded(
                "wall-clock budget is exhausted; no further provider call was made"
            )
        return remaining


def _call_with_timeout(call, timeout: float):
    outcome = {}

    def target():
        try:
            outcome["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc

    thread = threading.Thread(target=target, name="model-call", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise ModelBudgetExceeded(
            f"model call exceeded the remaining {timeout:.1f}s wall-clock budget; the "
            "provider request was abandoned"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
