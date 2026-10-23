import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.routers import auth, tasks
from app.routers.account import router as account_router
from app.routers.admin import router as admin_router
from app.workspace_cleanup import run_task_deletion_worker


@asynccontextmanager
async def lifespan(_app: FastAPI):
    stop = asyncio.Event()
    worker = asyncio.create_task(run_task_deletion_worker(stop))
    try:
        yield
    finally:
        stop.set()
        await worker


app = FastAPI(
    title="Agent Orchestrator Core", version="0.1.0", lifespan=lifespan
)
app.include_router(auth.router)
app.include_router(account_router)
app.include_router(tasks.router)
app.include_router(admin_router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=422, content={"detail": "Request validation failed"}
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}
