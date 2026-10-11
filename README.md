# Agent Orchestrator Core

PostgreSQL-backed task coordination service with separate API, worker, user UI, and administrator UI processes.

## Local setup

1. Copy `.env.example` to `.env`. Replace the database password with a long random value in both `POSTGRES_PASSWORD` and `DATABASE_URL`, set real model IDs available to your OpenAI project, and choose a unique administrator password of at least 12 characters. Keep `.env` and Vault credentials out of version control.
2. Start PostgreSQL: `docker compose up -d db`.
3. Apply the initial schema: `docker compose --profile setup run --rm migrate`.
4. Create the first administrator: `docker compose --profile setup run --rm bootstrap`.
5. Start the API and both UIs: `docker compose up --build -d`.
6. Open `http://localhost:8080/` for the user UI or `http://localhost:8080/admin/` for the administrator UI.

For task execution, provision a Vault KV v2 mount and configure two AppRoles restricted to the configured path: an API role that can write but not read, and a Worker role that can read but not write. Place each role's `role_id` and `secret_id` in the four ignored files under `secrets/`. The administrator UI can then store an OpenAI key in Vault. Start the independent Worker with `docker compose --profile agent up --build -d worker`. The Worker does not receive the OpenAI key in its environment and reads the current key from Vault before each model request. Set `SESSION_COOKIE_SECURE=true` when serving through TLS.

The user UI supports login, task creation, conversation continuation, cancellation, terminal-task deletion, durable SSE replay, password changes, and one-time API-key rotation. The administrator UI supports account provisioning, runtime model/concurrency settings, Vault key updates, and privacy-limited task operations. Agent execution supports structured planning, dependency-aware parallel read-only subtasks, replanning on new user messages, and structured response integration. No shell, file mutation, or external write tools are exposed by this initial runtime.

## Remaining scope

Human approval and `needs_attention` resolution flows, the isolated Sandbox service, dynamic private-skill loading, the RAG adapter, artifact storage/download, context summarization, usage/cost records, and database-shared RPM/TPM/concurrency slots are not implemented yet. Until those execution boundaries exist, the Worker only performs model-based informational tasks and exposes no side-effecting tools. Run a single Worker replica for this initial implementation.

## Development checks

- Backend syntax: `/usr/bin/python3 -m compileall -q backend/app backend/migrations`
- User UI: `cd ui/user && npm install && npm run build`
- Admin UI: `cd ui/admin && npm install && npm run build`
- Compose structure: `docker-compose --env-file .env.example config`

### Python tests

Run these commands from the repository root. They create and activate a virtual environment, install the development dependencies declared in `pyproject.toml`, and run the tests under `backend/tests`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

The current tests use isolated/mocked dependencies and do not require PostgreSQL or Docker to be running.
