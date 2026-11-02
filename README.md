# Agent Orchestrator Core

PostgreSQL-backed task coordination service with separate API, worker, user UI, and administrator UI processes.

## Initial setup

1. Copy `.env.example` to `.env`. Replace the database password in both `POSTGRES_PASSWORD` and `DATABASE_URL`, choose a unique administrator password of at least 12 characters, and set `WORKSPACE_INTERNAL_TOKEN` to a separate long random value. Runtime settings select the Manager, Worker conversational, and Worker image models; image-specialist reviews use the Worker conversational model. Set `VAULT_CREDENTIAL_GID` to the host's `id -g` value so the non-root API and Worker can read the group-restricted credential files. Keep `.env` and Vault credentials out of version control.
2. Check out the private Agent plugin repository locally. Replace `<PRIVATE_AGENT_PLUGIN_GIT_REPOSITORY_URL>` with its Git URL:
  ```sh
  git clone <PRIVATE_AGENT_PLUGIN_GIT_REPOSITORY_URL> ./.ai_agent_base
  ```
  The repository contains `agents/` and `skills/`. The core repository ignores this checkout; manage its changes with the plugin repository's own Git history. Compose mounts the complete plugin read-only into the Worker. Agent and Skill edits are loaded during planning or execution without rebuilding the image. Set `AGENT_PLUGIN_SOURCE` to use another host directory; `PRIVATE_SKILLS_SOURCE` remains a migration alias for older Skill-only checkouts.
3. Start PostgreSQL and the development Vault: `docker compose up -d db` and `docker compose --profile local-vault up -d --wait vault`.
4. If you plan to use the local LLM, ensure the host has an NVIDIA GPU and NVIDIA Container Toolkit configured for Docker, then start Ollama and download the models:
   ```sh
   docker compose --profile local-llm up -d ollama
   docker compose --profile local-llm exec ollama ollama pull qwen3.5:9b
   docker compose --profile local-llm exec ollama ollama pull gemma4:e4b
   ```
   The downloaded model IDs will appear in the administrator UI. Skip this step when using OpenAI only.
5. Provision the local KV v2 path and API/Worker AppRoles: `sh scripts/provision-local-vault.sh`.
6. Apply the initial schema: `docker compose --profile setup run --rm migrate`.
7. Create the first administrator: `docker compose --profile setup run --rm bootstrap`.

## Start services

After initial setup, start or restart the API and both UIs with `docker compose --profile local-vault up -d`. Open `http://localhost:8080/` for the user UI or `http://localhost:8080/admin/` for the administrator UI. Add `--build` when you need to rebuild images after code changes.

```sh
# DBと、ローカルVaultを使う場合はVaultだけ先に起動
docker compose --profile local-vault up -d --wait db vault
sh scripts/provision-local-vault.sh

# DBスキーマを更新してからアプリを起動
docker compose --profile setup run --build --rm migrate
docker compose --profile local-vault up -d --build
docker compose --profile image-generation up -d comfyui
docker compose --profile local-vault --profile agent up --build -d --force-recreate worker
```

### Generate local passwords

Use OpenSSL to generate separate 256-bit random values for PostgreSQL and the initial administrator. The hexadecimal database password is URL-safe; use the same value in both `POSTGRES_PASSWORD` and `DATABASE_URL` in `.env`. Copy the printed lines into `.env` and do not commit them.

```sh
db_password=$(openssl rand -hex 32)
admin_password=$(openssl rand -hex 32)
workspace_token=$(openssl rand -hex 32)
printf 'POSTGRES_PASSWORD=%s\n' "$db_password"
printf 'DATABASE_URL=postgresql+asyncpg://orchestrator:%s@db:5432/orchestrator\n' "$db_password"
printf 'INITIAL_ADMIN_PASSWORD=%s\n' "$admin_password"
printf 'WORKSPACE_INTERNAL_TOKEN=%s\n' "$workspace_token"
unset db_password admin_password workspace_token
```

The optional `local-vault` profile runs HashiCorp Vault in dev mode, reachable by the application at `http://vault:8200` and from the host at `http://localhost:8201`. Its root token is for local development only; dev-mode data is in memory and is lost when the Vault container is recreated. Rerun the provisioning script after recreating it. For an external Vault, set `VAULT_ADDR` and all six `VAULT_*_SOURCE` paths in `.env` to the external server and its AppRole credentials instead. Existing `.env` files should set `VAULT_ADDR=http://vault:8200` and use the six `.local-vault/*` source paths shown in the updated `.env.example` when using the local profile.

The provisioning script configures KV v2 paths and three AppRoles: the API role can write the OpenAI and Tavily paths but cannot read secrets, the Worker role can read both paths but cannot write, and the model-catalog role can read only the OpenAI key path. Their credentials are written to six ignored files under `.local-vault/` with group-only read permissions. Configure or replace provider keys from **Runtime settings** in the administrator UI. The values are stored in Vault and are never returned to the UI. For an external Vault, grant the API AppRole write access to the configured secret paths, the Worker AppRole read access to the OpenAI and Tavily paths, and the catalog AppRole read access only to the OpenAI path.

The optional `local-llm` profile runs Ollama in Compose, reserves all host NVIDIA GPUs, and persists model data in the `ollama_data` volume. GPU access requires an NVIDIA driver and NVIDIA Container Toolkit installed on the host and configured for Docker. Start it with `docker compose --profile local-llm up -d ollama`, then download `qwen3.5:9b` and `gemma4:e4b` as shown in Initial setup. API and Worker always connect to the Compose service at `http://ollama:11434`; host-installed or external Ollama endpoints are not supported. The model selector reads Ollama model IDs from `/api/tags`. AMD ROCm configuration is not included.

ComfyUI image generation is an optional GPU-backed Worker tool. Install the host NVIDIA driver and NVIDIA Container Toolkit, create `./.comfyui/models/checkpoints`, and place a ComfyUI-compatible checkpoint there. Set `COMFYUI_ENABLED=true` and `COMFYUI_MODELS_SOURCE=./.comfyui/models` in `.env`, then select **ComfyUI** under the separate **Worker image endpoint** in administrator Runtime settings. The image model list is fetched from ComfyUI and contains its available checkpoint names; the Worker conversational endpoint does not affect this list. `WORKER_IMAGE_PROVIDER` and `WORKER_IMAGE_MODEL` supply initial settings only; saved Runtime settings take precedence. Set `COMFYUI_USER_ID` and `COMFYUI_GROUP_ID` to a host identity that can read the model files. Start the private service and Worker with:

```sh
docker compose --profile image-generation up -d comfyui
docker compose --profile local-vault --profile agent up --build -d worker
```

Open the ComfyUI WebUI at `http://localhost:8188` from a browser on the Docker host. Compose binds the port to host loopback only, so it is not exposed to the LAN. For a remote VS Code workspace, forward port 8188 to your local browser. The service mounts models read-only and keeps its output in temporary storage; generated PNGs are copied into the task Workspace. The initial fixed workflow uses the standard checkpoint loader and supports compatible checkpoint files, not arbitrary ComfyUI workflows or custom nodes. The administrator's Worker image model field accepts the checkpoint filename when ComfyUI is enabled.

Start the independent Worker with `docker compose --profile local-vault --profile agent up --build -d worker`. The Worker does not receive either provider key in its environment; it reads each key from Vault only when making the corresponding request. For an external Vault, grant the Worker AppRole read access to both configured secret paths. Set `SESSION_COOKIE_SECURE=true` when serving through TLS.

## Agents SDK runtime

The Manager selects specialist Agents from the validated `AGENT_PLUGIN_SOURCE` catalog using their descriptions and the task requirements. For image generation it selects a distinct reviewer; new catalog-bound image generation fails closed if no reviewer is available. Unknown IDs are rejected. Selected Agent IDs and definition digests are pinned per subtask; missing or changed selected definitions cause an error. Plugin metadata does not grant core permissions or execute plugin code. Before each image-generation handoff to the Manager, the generation Agent calls the registered reviewer tool using the task's snapshotted Worker conversational model and supplies the request, actual prompt, settings, PNG, and available crops. Each generation cycle allows up to five reviews per image ID, stopping early on pass; internal revisions share the cycle count. Image-targeted Manager feedback starts a new cycle and requires a newly generated PNG plus Reviewer call; answer-only feedback reuses the existing image. Worker retries retain the same cycle ID. Review history and counts are stored per cycle, separately from Manager quality revisions, and included in the result returned to the Manager. Missing reviews, review failures, or unresolved results before the fifth review stop the handoff; a fifth unresolved review is reported as such. An image with completed specialist pre-review skips the duplicate subtask-level Manager check and receives one final Manager acceptance check per cycle. A Manager-requested image revision creates another cycle and another final check. Progress events identify the calling and executing Agents, including delegated review attempts. Migrations `0016_specialist_agents` and `0017_agent_binding_snapshot` store specialist selection and reviewer bindings; `0018_catalog_review_binding` distinguishes catalog selections from legacy bindings.

Manager planning, checklist reviews, synthesis, and subtask generation/revision run through OpenAI Agents SDK `Agent`/`Runner` behind the shared model adapter. The application still owns PostgreSQL tasks, plans, dependency scheduling, quality budgets, leases, user-message checkpoints, and SSE events. SDK completion is not task acceptance. Handoffs, automatic SDK Sessions, and persisted SDK RunState are not enabled. SDK tracing and sensitive-data capture are explicitly disabled; no OpenAI key environment variable is used.

Plugin-specific tool or model metadata is not a core authorization policy, including restrictive tool lists. Configure and enforce tool access through the core's registered Tool Registry and application settings.

The tested Python 3.10 dependency set is `openai-agents==0.24.0`, `openai==3.35.1`, and `pydantic==2.14.1`. OpenAI 3.x uses HTTPX2; the application retains HTTPX for its other integrations and tests use HTTPX2 for the SDK transport. OpenAI model requests, including tool continuations and retries, read the latest Vault key. Ollama requests use the configured OpenAI-compatible Chat Completions endpoint and do not read the OpenAI key. API and format retries share one bounded retry budget, and HTTP-client retries are disabled.

Generated ComfyUI and OpenAI PNGs are read through a token-protected internal Workspace API for subtask and final checklist reviews. Specialist Agents load only their own referenced Skills; the generic legacy image path retains the optional `image-generation` Skill and built-in fallback. Reviews include full images and available in-memory crops, capped together at five image inputs and 20 MiB; OpenAI vision requests use high detail. Image bytes are not persisted in conversation, events, or the database. Successful generations and their settings are recorded in the owner-scoped `image_generation_revisions` timeline; prompts and unmet text are not included in SSE events or application logs. Migrations `0013` through `0015` preserve earlier image-history compatibility. The Manager and Worker conversational models must support image input. Image-read failures, indeterminate reviews, and image-input failures are execution errors rather than speculative quality revisions. Images are sent to the configured model providers, so their normal data-handling and usage costs apply.

The Worker image provider and model are snapshotted per task, independently from the Worker conversational provider and model. With `worker_image_provider=comfyui` and `COMFYUI_ENABLED=true`, the Worker uses the selected ComfyUI checkpoint to generate requested PNG artifacts through a bounded fixed workflow. With `worker_image_provider=openai`, it calls the Images API using the selected image model and saves the PNG into the task Workspace; this works independently of the Worker conversational provider. OpenAI image generation uses the system OpenAI key stored in Vault and incurs provider usage charges. The shared prompt contract keeps positive and negative prompts separate; immediately before the Images API request, the negative prompt is appended to the regular prompt as a natural-language "Do not include" instruction because the API has no dedicated negative-prompt parameter. The model may not follow exclusions exactly. Ollama image generation is not implemented.

Set `MAX_AGENT_TURNS` (default 24, range 1-64) and `MAX_TOOL_CALLS_PER_SUBTASK` (default 20, range 0-100) in the deployment environment. The API snapshots them when creating a task; they are not administrator UI settings. Existing task snapshots without these fields use fixed compatibility defaults of 24 and 20. Tool usage is consumed before execution and stored on each subtask, including failed/interrupted calls, so format correction, quality revision, and Worker reclaim do not reset the budget. New plan versions have new subtask budgets; earlier usage remains recorded. These limits are separate from model retries and quality revision limits.

When a subtask reaches its tool-call limit, the Worker returns collected evidence for manager review instead of failing the task. If more research is required, the task pauses in `waiting_approval`; the owner may approve another batch equal to the task's initial `MAX_TOOL_CALLS_PER_SUBTASK` snapshot or reject and continue with partial results. There is no cumulative application cap; each approval is explicit and the user controls cost. Chat messages do not approve requests, and retries never reset usage. This requires migrations `0007_tool_budget_approval`, `0008_unbounded_tool_budget`, and `0009_tool_budget_snapshot`.

Task titles are generated by the manager from the first user instruction. Existing tasks initially use their instruction as a display fallback; the Worker replaces it with a generated title when it resumes a saved plan.

When a task requests a file, synthesis returns its filename and complete content separately from the chat response. An explicit filename is preserved; otherwise the agent chooses a descriptive name. The Worker writes only requested artifacts under the task Workspace `artifacts/` directory through the internal file API and reports the path only after persistence succeeds. The Explorer refreshes its listing when the task completes. Chat responses remain in the task conversation and are not copied into Workspace files.

For an existing deployment, stop the Worker, rebuild the backend image, apply pending migrations with `docker compose --profile setup run --build --rm migrate`, and then rebuild/restart the API, both UIs, and Worker. Apply migrations before starting services; no existing application database is modified automatically at startup. Keep a single Worker replica. The User UI provides task-scoped file upload, listing, text editing, PNG/JPEG/WebP/GIF preview for images up to 20 MiB, download, explicit tool-budget approval, and an image generation history showing review deficiencies and each successful prompt revision; a restricted interactive terminal and per-task container Sandbox are not yet available.

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

The user UI supports login, task creation, conversation continuation, cancellation, terminal-task deletion, durable SSE replay, password changes, and one-time API-key rotation. The administrator UI supports account provisioning, OpenAI/Ollama conversation endpoint selection, an OpenAI/Ollama/ComfyUI image endpoint and provider-specific model catalog, runtime concurrency settings, Vault key updates, and privacy-limited task operations. Agent execution supports structured planning, dependency-aware parallel read-only subtasks, specialist Agent selection with generic fallback, Tavily Web search through the official `tavily-search` Agent Skill and Tavily CLI, source retention/citations, replanning on new user messages, and structured response integration. The Worker loads validated Agent and Skill instructions into subagent prompts and exposes only registered tools; it does not expose a shell, arbitrary file mutation, or external write tools.

## Remaining scope

Additional tool providers and general side-effecting tools, human approval and `needs_attention` resolution flows, the isolated Sandbox service, the RAG adapter, context summarization, usage/cost records, and database-shared RPM/TPM/concurrency slots are not implemented yet. The Worker image installs the Tavily CLI and official `tavily-search` skill. The private plugin at `.ai_agent_base/` contains `agents/` and `skills/` and is mounted read-only into the Worker. Agent and Skill definitions are validated and loaded at planning/execution time; their text does not expand registered tool permissions. Searches remain read-only and use a per-Worker rate limit. ComfyUI is a separate optional GPU service; its narrowly scoped image-generation tool consumes the existing subtask tool budget and writes only generated PNGs to task artifacts. Run a single Worker replica for this initial implementation.

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
