FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /srv/app

RUN addgroup --system app && adduser --system --ingroup app app
COPY pyproject.toml ./
COPY alembic.ini ./
COPY backend ./backend
RUN pip install --no-cache-dir .
RUN mkdir -p /var/lib/agent-orchestrator/workspaces \
    && chown app:app /var/lib/agent-orchestrator/workspaces

USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
