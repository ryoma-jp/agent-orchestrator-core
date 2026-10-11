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

The provisioning script configures a KV v2 secret path and two AppRoles restricted to that path: an API role that can write but not read, and a Worker role that can read but not write. It writes their credentials to four ignored files under `.local-vault/` with group-only read permissions. The administrator UI can then store an OpenAI key in Vault. Start the independent Worker with `docker compose --profile local-vault --profile agent up --build -d worker`. The Worker does not receive the OpenAI key in its environment and reads the current key from Vault before each model request. Set `SESSION_COOKIE_SECURE=true` when serving through TLS.

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
