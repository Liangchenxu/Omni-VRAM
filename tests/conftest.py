"""
Shared pytest configuration.

Async test support
------------------
parts of the test-suite (e.g. ``tests/test_websocket.py``) are written as
``async def`` tests marked with ``@pytest.mark.asyncio``. When
``pytest-asyncio`` is installed it takes over completely; when it is *not*
installed (plain ``pytest`` + ``anyio`` as pulled in by FastAPI) those tests
would error out with "async def functions are not natively supported" instead
of exercising the code.

The hook below adds the missing runner: it executes coroutine test functions
through ``asyncio.run()`` and stays completely inert whenever
``pytest-asyncio`` (or another coroutine-aware plugin) is available.
"""

import asyncio
import inspect

import pytest

try:  # pragma: no cover - depends on the environment
    import pytest_asyncio  # noqa: F401

    _ASYNCIO_PLUGIN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    _ASYNCIO_PLUGIN_AVAILABLE = False


def pytest_configure(config):
    """Register the ``asyncio`` marker so that pytest does not warn about it."""
    config.addinivalue_line(
        "markers", "asyncio: run the test in an asyncio event loop (native shim)"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """
    Run ``async def`` tests on a private event loop when no async plugin exists.

    Returns True only when the test was executed here, so plugins such as
    ``pytest-asyncio`` keep full control in environments where they are present.
    """
    if _ASYNCIO_PLUGIN_AVAILABLE:
        return None

    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    argnames = getattr(pyfuncitem._fixtureinfo, "argnames", ())
    kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
    asyncio.run(test_function(**kwargs))
    return True
