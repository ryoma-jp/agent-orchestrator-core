import asyncio
import logging
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select

from app.config import get_settings
from app.db import SessionFactory
from app.models import Event, Job, Task
from app.orchestrator import InstructionChanged, execute_task
from app.openai_client import AgentExecutionError

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("orchestrator.worker")
worker_id = f"{socket.gethostname()}-{uuid.uuid4()}"


async def claim_job():
    settings = get_settings()
    now = datetime.now(timezone.utc)
    async with SessionFactory() as session:
        job = await session.scalar(
            select(Job)
            .where(
                Job.available_at <= now,
                or_(
                    Job.status == "pending",
                    and_(Job.status == "leased", Job.lease_expires_at < now),
                ),
            )
            .order_by(Job.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            return None
        task = await session.scalar(
            select(Task).where(Task.id == job.task_id).with_for_update()
        )
        if task is None or task.status in {
            "succeeded",
            "partially_completed",
            "failed",
            "cancelled",
        }:
            job.status = "completed"
            await session.commit()
            return None
        job.status = "leased"
        job.lease_owner = worker_id
        job.lease_expires_at = now + timedelta(
            seconds=settings.job_lease_seconds
        )
        job.attempt += 1
        task.status = "planning"
        task.updated_at = now
        session.add(
            Event(
                task_id=task.id,
                type="task.status_changed",
                payload_json={"status": "planning"},
            )
        )
        await session.commit()
        return job.id, job.task_id, job.attempt


async def renew_lease(
    job_id: uuid.UUID, stop: asyncio.Event, lost: asyncio.Event
) -> None:
    settings = get_settings()
    interval = max(1, settings.job_lease_seconds // 3)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        async with SessionFactory() as session:
            job = await session.scalar(
                select(Job).where(
                    Job.id == job_id,
                    Job.lease_owner == worker_id,
                    Job.status == "leased",
                    Job.lease_expires_at > datetime.now(timezone.utc),
                )
            )
            if job is None:
                lost.set()
                return
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=settings.job_lease_seconds
            )
            await session.commit()


async def mark_failure(
    job_id: uuid.UUID, task_id: uuid.UUID, attempt: int, error: Exception
) -> None:
    settings = get_settings()
    async with SessionFactory() as session:
        job = await session.scalar(
            select(Job).where(
                Job.id == job_id,
                Job.lease_owner == worker_id,
                Job.status == "leased",
            )
        )
        if job is None:
            return
        task = await session.get(Task, task_id)
        if task is None:
            job.status = "failed"
        elif attempt < min(
            max(
                task.config_snapshot.get(
                    "max_agent_attempts", settings.max_agent_attempts
                ),
                1,
            ),
            3,
        ):
            job.status = "pending"
            job.available_at = datetime.now(timezone.utc) + timedelta(
                seconds=min(2**attempt, 30)
            )
            job.lease_owner = None
            job.lease_expires_at = None
            task.status = "queued"
            task.error = {
                "code": "agent_attempt_failed",
                "message": "Task execution failed; retry scheduled.",
            }
            task.updated_at = datetime.now(timezone.utc)
            session.add(
                Event(
                    task_id=task_id,
                    type="task.status_changed",
                    payload_json={
                        "status": "queued",
                        "error": "agent_attempt_failed",
                    },
                )
            )
        else:
            job.status = "failed"
            job.lease_owner = None
            job.lease_expires_at = None
            task.status = "failed"
            task.error = {
                "code": (
                    str(error)
                    if isinstance(error, AgentExecutionError)
                    else "task_execution_failed"
                ),
                "message": "Task execution failed after the retry limit.",
            }
            task.updated_at = datetime.now(timezone.utc)
            session.add(
                Event(
                    task_id=task_id,
                    type="task.failed",
                    payload_json=task.error,
                )
            )
            session.add(
                Event(
                    task_id=task_id,
                    type="task.status_changed",
                    payload_json={"status": "failed"},
                )
            )
        await session.commit()


async def run_job(job_id: uuid.UUID, task_id: uuid.UUID, attempt: int) -> None:
    stop = asyncio.Event()
    lost = asyncio.Event()
    heartbeat = asyncio.create_task(renew_lease(job_id, stop, lost))
    execution = asyncio.create_task(execute_task(task_id, job_id, worker_id))
    try:
        done, _ = await asyncio.wait(
            {heartbeat, execution}, return_when=asyncio.FIRST_COMPLETED
        )
        if heartbeat in done and lost.is_set():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            return
        await execution
    except InstructionChanged:
        return
    except Exception as exc:
        logger.warning(
            "Task execution failed for task %s (%s)",
            task_id,
            type(exc).__name__,
        )
        await mark_failure(job_id, task_id, attempt, exc)
    finally:
        stop.set()
        await asyncio.gather(heartbeat, return_exceptions=True)


async def serve() -> None:
    logger.info("Worker %s started", worker_id)
    while True:
        claimed = await claim_job()
        if claimed is None:
            await asyncio.sleep(1)
            continue
        await run_job(*claimed)


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Worker stopped")
