# Agent Orchestrator Core

PostgreSQL-backed task coordination service with separate API, worker, user UI, and administrator UI processes.

## Initial setup

1. Copy `.env.example` to `.env`. Replace the database password with a long random value in both `POSTGRES_PASSWORD` and `DATABASE_URL`, set real model IDs available to your OpenAI project, and choose a unique administrator password of at least 12 characters. Set `VAULT_CREDENTIAL_GID` to the host's `id -g` value so the non-root API and Worker can read the group-restricted credential files. Keep `.env` and Vault credentials out of version control.
2. Start PostgreSQL and the development Vault: `docker compose up -d db` and `docker compose --profile local-vault up -d --wait vault`.
3. Provision the local KV v2 path and API/Worker AppRoles: `sh scripts/provision-local-vault.sh`.
4. Apply the initial schema: `docker compose --profile setup run --rm migrate`.
5. Create the first administrator: `docker compose --profile setup run --rm bootstrap`.

## Start services

After initial setup, start or restart the API and both UIs with `docker compose --profile local-vault up -d`. Open `http://localhost:8080/` for the user UI or `http://localhost:8080/admin/` for the administrator UI. Add `--build` when you need to rebuild images after code changes.

```sh
docker compose --profile local-vault up -d --build
sh scripts/provision-local-vault.sh
docker compose --profile local-vault --profile agent up --build -d worker
```

### Generate local passwords

Use OpenSSL to generate separate 256-bit random values for PostgreSQL and the initial administrator. The hexadecimal database password is URL-safe; use the same value in both `POSTGRES_PASSWORD` and `DATABASE_URL` in `.env`. Copy the printed lines into `.env` and do not commit them.

```sh
db_password=$(openssl rand -hex 32)
admin_password=$(openssl rand -hex 32)
printf 'POSTGRES_PASSWORD=%s\n' "$db_password"
printf 'DATABASE_URL=postgresql+asyncpg://orchestrator:%s@db:5432/orchestrator\n' "$db_password"
printf 'INITIAL_ADMIN_PASSWORD=%s\n' "$admin_password"
unset db_password admin_password
```

The optional `local-vault` profile runs HashiCorp Vault in dev mode, reachable by the application at `http://vault:8200` and from the host at `http://localhost:8201`. Its root token is for local development only; dev-mode data is in memory and is lost when the Vault container is recreated. Rerun the provisioning script after recreating it. For an external Vault, set `VAULT_ADDR` and the four `VAULT_*_SOURCE` paths in `.env` to the external server and its AppRole credentials instead. Existing `.env` files should set `VAULT_ADDR=http://vault:8200` and use the four `.local-vault/*` source paths shown in the updated `.env.example` when using the local profile.

The provisioning script configures KV v2 paths and two AppRoles: the API role can write the OpenAI and Tavily paths but cannot read secrets, and the Worker role can read both paths but cannot write. It writes their credentials to four ignored files under `.local-vault/` with group-only read permissions. Configure or replace both provider keys from **Runtime settings** in the administrator UI. The values are stored in Vault and are never returned to the UI. For an external Vault, grant the API AppRole write access to both configured secret paths and the Worker AppRole read access to them.

Start the independent Worker with `docker compose --profile local-vault --profile agent up --build -d worker`. The Worker does not receive either provider key in its environment; it reads each key from Vault only when making the corresponding request. For an external Vault, grant the Worker AppRole read access to both configured secret paths. Set `SESSION_COOKIE_SECURE=true` when serving through TLS.

## Agents SDK runtime

Manager planning, checklist reviews, synthesis, and subtask generation/revision run through OpenAI Agents SDK `Agent`/`Runner` behind the shared model adapter. The application still owns PostgreSQL tasks, plans, dependency scheduling, quality budgets, leases, user-message checkpoints, and SSE events. SDK completion is not task acceptance. Handoffs, automatic SDK Sessions, and persisted SDK RunState are not enabled. SDK tracing and sensitive-data capture are explicitly disabled; no OpenAI key environment variable is used.

The tested Python 3.10 dependency set is `openai-agents==0.24.0`, `openai==3.35.1`, and `pydantic==2.14.1`. OpenAI 3.x uses HTTPX2; the application retains HTTPX for its other integrations and tests use HTTPX2 for the SDK transport. Each actual model request, including tool continuations and retries, reads the latest Vault key. API and format retries share one bounded retry budget, and HTTP-client retries are disabled.

Set `MAX_AGENT_TURNS` (default 24, range 1-64) and `MAX_TOOL_CALLS_PER_SUBTASK` (default 20, range 0-100) in the deployment environment. The API snapshots them when creating a task; they are not administrator UI settings. Existing task snapshots without these fields use fixed compatibility defaults of 24 and 20. Tool usage is consumed before execution and stored on each subtask, including failed/interrupted calls, so format correction, quality revision, and Worker reclaim do not reset the budget. New plan versions have new subtask budgets; earlier usage remains recorded. These limits are separate from model retries and quality revision limits.

For an existing deployment, stop the Worker, rebuild the backend image, apply migration `0004_agent_sdk_limits` with `docker compose --profile setup run --build --rm migrate`, and then rebuild/restart the API and Worker. Apply migrations before starting the new Worker; no existing application database is modified automatically at startup. Keep a single Worker replica. SDK integration does not add task-level leases, shared rate limits, side-effect recovery, or a new Sandbox.

## Cleanup

These commands affect this Compose project only. Run them from the repository root.

### Stop services

Stop all project services, including services in optional profiles:

```sh
docker compose --profile local-vault --profile agent --profile setup stop
```

This keeps containers, images, the PostgreSQL database, and local credentials so the services can be started again.

### Delete data and database

Remove project containers and networks as needed to release the database volume, then delete that volume and the generated local Vault credentials:

```sh
docker compose --profile local-vault --profile agent --profile setup down --volumes --remove-orphans
rm -rf .local-vault
```

This permanently deletes the PostgreSQL database and the in-memory dev Vault data; downloaded and built images are retained. Remove `.env` separately with `rm -f .env` only if its local configuration and passwords are no longer needed. If `secrets/` contains credentials for an external Vault, delete those files only when you explicitly intend to revoke and remove them.

### Remove containers and images

For a full cleanup of this project's Docker resources, including all images used by its services, run:

```sh
docker compose --profile local-vault --profile agent --profile setup down --rmi all --volumes --remove-orphans
rm -rf .local-vault
```

This also deletes the database volume and dev Vault data. Images shared with other running containers are not removed. It does not prune Docker resources belonging to other projects.

The user UI supports login, task creation, conversation continuation, cancellation, terminal-task deletion, durable SSE replay, password changes, and one-time API-key rotation. The administrator UI supports account provisioning, runtime model/concurrency settings, OpenAI Vault key updates, and privacy-limited task operations. Agent execution supports structured planning, dependency-aware parallel read-only subtasks, Tavily Web search with source retention/citations, replanning on new user messages, and structured response integration. No shell, file mutation, or external write tools are exposed by this runtime.

## Remaining scope

Additional tool providers and side-effecting tools, human approval and `needs_attention` resolution flows, the isolated Sandbox service, dynamic private-skill loading, the RAG adapter, artifact storage/download, context summarization, usage/cost records, and database-shared RPM/TPM/concurrency slots are not implemented yet. The current Tavily tool is read-only and uses a per-Worker rate limit; the Worker exposes no shell, file mutation, or external write tools. Run a single Worker replica for this initial implementation.

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

The default tests use isolated/mocked dependencies and make no real OpenAI or Tavily requests. SDK tests exercise the actual Runner and OpenAI client with an HTTP mock. PostgreSQL integration tests are skipped unless `TEST_DATABASE_URL` is set to a dedicated PostgreSQL database whose name ends in `_test`:

```sh
docker run --name agent-orchestrator-test-db \
  -e POSTGRES_USER=test_user \
  -e POSTGRES_PASSWORD=test_password \
  -e POSTGRES_DB=orchestrator_test \
  -p 127.0.0.1:5432:5432 \
  -d postgres:16-alpine
TEST_DATABASE_URL=postgresql+asyncpg://test_user:test_password@localhost:5432/orchestrator_test python -m pytest -q
docker rm -f agent-orchestrator-test-db
```

Integration tests create and remove randomly named schemas only in that dedicated database. They cover atomic tool budgets, stale leases/cancellation, snapshot settings, migration upgrade/downgrade, instruction requeue races, and the SDK planning/subtask/review/synthesis path. Real-model quality, cost, latency, and multi-Worker operation are not certified by these tests.
