from .handlers import EngineCommandRouter
from .storage import InMemoryEngineStorage


def __getattr__(name: str):
    if name in {"EngineProtocolServer", "run_stdio_server"}:
        from .server import EngineProtocolServer, run_stdio_server

        exports = {
            "EngineProtocolServer": EngineProtocolServer,
            "run_stdio_server": run_stdio_server,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EngineCommandRouter",
    "EngineProtocolServer",
    "InMemoryEngineStorage",
    "run_stdio_server",
]
