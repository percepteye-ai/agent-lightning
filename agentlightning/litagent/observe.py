# Copyright (c) Microsoft. All rights reserved.

"""Decorator to run an agent function inside a trace context so spans are captured and sent to the store."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
from typing import Any, Callable, Dict, Optional, Union, overload

from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import Tracer

logger = logging.getLogger(__name__)


def _task_input_from_args(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: Dict[str, Any]
) -> Any:
    """Build task input for start_rollout from the first argument or a default."""
    if args:
        first = args[0]
        if isinstance(first, dict):
            return first  # type: ignore[return-value]
        if hasattr(first, "model_dump"):
            return getattr(first, "model_dump")()
        if hasattr(first, "input"):
            return getattr(first, "input")
        return {"input": first}
    if "task" in kwargs:
        t = kwargs["task"]
        if isinstance(t, dict):
            return t  # type: ignore[return-value]
        if hasattr(t, "model_dump"):
            return getattr(t, "model_dump")()
        return {"input": t}
    return {"observe": func.__name__}


async def _run_observed(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    store: LightningStore,
    tracer: Tracer,
    is_async_func: bool,
) -> tuple[Any, str, str]:
    """Run the function in a trace context and send spans to the store.

    Returns:
        Tuple of (function result, rollout_id, attempt_id).
    """
    task_input = _task_input_from_args(func, args, kwargs)
    with tracer.lifespan(store):
        attempted = await store.start_rollout(input=task_input)
        rollout_id = attempted.rollout_id
        attempt_id = attempted.attempt.attempt_id
        logger.debug("observe: rollout_id=%s attempt_id=%s", rollout_id, attempt_id)
        async with tracer.trace_context(
            name=func.__name__,
            store=store,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
        ):
            if is_async_func:
                result = await func(*args, **kwargs)
            else:
                ctx = contextvars.copy_context()

                def run_in_context() -> Any:
                    return ctx.run(func, *args, **kwargs)

                result = await asyncio.to_thread(run_in_context)
        return (result, rollout_id, attempt_id)


@overload
def observe(func: Callable[..., Any]) -> Callable[..., Any]: ...
@overload
def observe(
    *,
    store: Optional[LightningStore] = None,
    tracer: Optional[Tracer] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
def observe(
    func: Optional[Callable[..., Any]] = None,
    *,
    store: Optional[LightningStore] = None,
    tracer: Optional[Tracer] = None,
) -> Union[Callable[..., Any], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Run the decorated agent function in a trace context and send spans to the store.

    Use ``@observe`` for agent functions that run without a Runner. Each call starts a
    new rollout on the store, enters a trace context with that rollout_id/attempt_id,
    runs the function, and sends all captured spans (from AgentOps auto-instrumentation, etc.) to the store.

    The decorator provides trace_context and captures spans for that rollout_id/attempt_id.
    You do not need to create a rollout or enforce a specific method signature, unlike
    Agent Lightning's Runner/trainer flows.

    If ``store`` and ``tracer`` are omitted, defaults are used: `InMemoryLightningStore`
    and `AgentOpsTracer`. The decorated callable is always async and must be called
    with ``await``.

    After a run, you can deduce the agent design (nodes, tools, sub-agents) from the
    stored trace using [`infer_agent_design`][agentlightning.litagent.infer_agent_design].
    Pass ``return_rollout_ids=True`` and the same ``store`` to get ``(result, rollout_id, attempt_id)``
    so you can call that helper.

    Args:
        func: Function to observe. Can be sync or async; sync is run in a thread with
            context propagated so instrumentation spans attach to the same trace.
        store: Optional store. Spans are written here for the created rollout.
        tracer: Optional tracer. If None, `AgentOpsTracer` is used.
        return_rollout_ids: If True, the wrapper returns ``(result, rollout_id, attempt_id)``
            instead of just the result, so you can query the store for spans and infer design.

    Returns:
        An async wrapper that runs the function inside a trace context.

    Example:
        .. code-block:: python

            @observe(store=store, tracer=tracer)
            async def my_agent(query: str) -> str:
                ...

            result = await my_agent("get weather in Delhi")
    """
    store_arg = store
    tracer_arg = tracer
    return_rollout_ids_arg = return_rollout_ids

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        is_async = asyncio.iscoroutinefunction(f)

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            s: LightningStore = store_arg if store_arg is not None else InMemoryLightningStore()
            t: Tracer = tracer_arg if tracer_arg is not None else AgentOpsTracer()
            result, rollout_id, attempt_id = await _run_observed(f, args, kwargs, s, t, is_async)
            if return_rollout_ids_arg:
                return (result, rollout_id, attempt_id)
            return result

        functools.update_wrapper(wrapper, f)
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
