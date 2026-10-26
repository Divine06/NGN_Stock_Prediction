"""Minimal stand-ins used only when a library is missing from the test environment,
so the code that calls it can still be exercised. With the real libraries installed
(your machine), the tests use the real packages instead."""
import sys
import types


def install_fastapi_standin() -> bool:
    """Returns False (and does nothing) when the real fastapi is installed."""
    try:
        import fastapi  # noqa: F401
        if not getattr(fastapi, "_is_standin", False):
            return False
    except ImportError:
        pass

    class APIRouter:
        def __init__(self, prefix="", tags=None, **kw):
            self.prefix, self.routes = prefix, {}

        def _reg(self, method, path):
            def deco(fn):
                self.routes[(method, self.prefix + path)] = fn
                return fn
            return deco

        def get(self, path, **kw):
            return self._reg("GET", path)

        def post(self, path, **kw):
            return self._reg("POST", path)

    class FastAPI(APIRouter):
        def __init__(self, title="", description="", version="", lifespan=None, **kw):
            super().__init__()
            self.lifespan, self.middleware, self.routers = lifespan, [], []

        def add_middleware(self, cls, **kw):
            self.middleware.append((cls, kw))

        def include_router(self, router):
            self.routers.append(router)

    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            super().__init__(detail)
            self.status_code, self.detail = status_code, detail

    async def run_in_threadpool(fn, *a, **k):
        return fn(*a, **k)

    fake = types.ModuleType("fastapi")
    fake._is_standin = True
    fake.FastAPI, fake.APIRouter, fake.HTTPException = FastAPI, APIRouter, HTTPException
    fake.Query = lambda default=None, **kw: default
    conc = types.ModuleType("fastapi.concurrency")
    conc.run_in_threadpool = run_in_threadpool
    mw = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = type("CORSMiddleware", (), {})
    sys.modules.update({"fastapi": fake, "fastapi.concurrency": conc, "fastapi.middleware": mw,
                        "fastapi.middleware.cors": cors})
    return True
