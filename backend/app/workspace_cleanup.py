"""Retryable cleanup for task workspaces and task-owned database state."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select

from app.config import get_settings
from app.db import SessionFactory
from app.models import Task, TaskDeletionJob
from app.workspaces import delete_task_workspace

logger = logging.getLogger(__name__)


async def run_task_deletion_worker(stop: asyncio.Event) -> None:
    worker_id = str(uuid.uuid4())
    while not stop.is_set():
        try:
            task_id = await _lease_deletion(worker_id)
        except Exception as exc:
            logger.warning(
                "Task cleanup lease failed error=%s", type(exc).__name__
            )
            await _wait_or_stop(stop, 2.0)
            continue
        if task_id is None:
            await _wait_or_stop(stop, 1.0)
            continue
        try:
            await asyncio.to_thread(delete_task_workspace, task_id)
            await _complete_deletion(task_id, worker_id)
        except Exception as exc:
            logger.warning(
                "Task cleanup failed task_id=%s error=%s",
                task_id,
                type(exc).__name__,
            )
            try:
                await _retry_deletion(task_id, worker_id)
            except Exception as retry_exc:
                logger.warning(
                    "Task cleanup retry update failed task_id=%s error=%s",
                    task_id,
                    type(retry_exc).__name__,
                )


async def _wait_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def _lease_deletion(worker_id: str) -> uuid.UUID | None:
    now = datetime.now(timezone.utc)
    async with SessionFactory() as session:
        job = await session.scalar(
            select(TaskDeletionJob)
            .where(
                TaskDeletionJob.available_at <= now,
                or_(
                    TaskDeletionJob.status == "pending",
                    TaskDeletionJob.lease_expires_at <= now,
                ),
            )
            .order_by(TaskDeletionJob.available_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if job is None:
            return None
        job.status = "leased"
        job.lease_owner = worker_id
        job.lease_expires_at = now + timedelta(
            seconds=get_settings().job_lease_seconds
        )
        job.attempt += 1
        task_id = job.task_id
        await session.commit()

        return task_id


async def _complete_deletion(task_id: uuid.UUID, worker_id: str) -> None:
    async with SessionFactory() as session:
        job = await session.scalar(
            select(TaskDeletionJob)
            .where(
                TaskDeletionJob.task_id == task_id,
                TaskDeletionJob.lease_owner == worker_id,
            )
            .with_for_update()
        )
        if job is None:
            return
        task = await session.scalar(
            select(Task)
            .where(
                Task.id == task_id,
                Task.deletion_requested_at.is_not(None),
            )
            .with_for_update()
        )
        if task is not None:
            await session.delete(task)
        await session.delete(job)
        await session.commit()


async def _retry_deletion(task_id: uuid.UUID, worker_id: str) -> None:
    async with SessionFactory() as session:
        job = await session.scalar(
            select(TaskDeletionJob)
            .where(
                TaskDeletionJob.task_id == task_id,
                TaskDeletionJob.lease_owner == worker_id,
            )
            .with_for_update()
        )
        if job is None:
            return
        delay = min(2 ** min(job.attempt, 8), 300)
        job.status = "pending"
        job.available_at = datetime.now(timezone.utc) + timedelta(
            seconds=delay
        )
        job.lease_owner = None
        job.lease_expires_at = None
        await session.commit()
