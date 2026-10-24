FROM node:22-slim AS tavily-skill-installer

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
ENV HOME=/tmp/tavily-skill-home
RUN mkdir -p "$HOME" \
    && npx --yes skills add tavily-ai/skills \
        --skill tavily-search \
        --global \
        --agent codex \
        --copy \
        --yes

FROM python:3.12-slim AS backend

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

FROM backend AS worker

USER root
RUN python -m venv /opt/tavily-cli \
    && /opt/tavily-cli/bin/pip install --no-cache-dir tavily-cli \
    && mkdir -p /opt/agent-skills
COPY --from=tavily-skill-installer \
    /tmp/tavily-skill-home/.agents/skills/tavily-search \
    /opt/agent-skills/tavily-search
RUN chown -R app:app /opt/tavily-cli /opt/agent-skills
ENV HOME=/home/app \
    PATH="/opt/tavily-cli/bin:${PATH}" \
    TAVILY_SEARCH_SKILL_PATH=/opt/agent-skills/tavily-search/SKILL.md

USER app

FROM backend AS default
