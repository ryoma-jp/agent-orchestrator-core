from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.routers import auth, tasks
from app.routers.account import router as account_router
from app.routers.admin import router as admin_router

app = FastAPI(title="Agent Orchestrator Core", version="0.1.0")
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
