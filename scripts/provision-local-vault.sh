#!/bin/sh
set -eu

compose() {
    docker compose --profile local-vault "$@"
}

umask 027
mkdir -p .local-vault
chmod 750 .local-vault

if ! compose exec -T vault vault auth list -format=json | grep -q '"approle/"'; then
    compose exec -T vault vault auth enable approle
fi

compose exec -T vault vault policy write agent-orchestrator-api - <<'POLICY'
path "secret/data/agent-orchestrator/openai" {
  capabilities = ["create", "update"]
}
path "auth/token/lookup-self" {
    capabilities = ["read"]
}
POLICY

compose exec -T vault vault policy write agent-orchestrator-worker - <<'POLICY'
path "secret/data/agent-orchestrator/openai" {
  capabilities = ["read"]
}
path "secret/data/agent-orchestrator/tavily" {
    capabilities = ["read"]
}
path "auth/token/lookup-self" {
    capabilities = ["read"]
}
POLICY

compose exec -T vault vault write auth/approle/role/agent-orchestrator-api \
    token_policies=agent-orchestrator-api token_no_default_policy=true
compose exec -T vault vault write auth/approle/role/agent-orchestrator-worker \
    token_policies=agent-orchestrator-worker token_no_default_policy=true

compose exec -T vault vault read -field=role_id \
    auth/approle/role/agent-orchestrator-api/role-id \
    > .local-vault/api-role-id
compose exec -T vault vault write -f -field=secret_id \
    auth/approle/role/agent-orchestrator-api/secret-id \
    > .local-vault/api-secret-id
compose exec -T vault vault read -field=role_id \
    auth/approle/role/agent-orchestrator-worker/role-id \
    > .local-vault/worker-role-id
compose exec -T vault vault write -f -field=secret_id \
    auth/approle/role/agent-orchestrator-worker/secret-id \
    > .local-vault/worker-secret-id
chmod 640 .local-vault/*

printf '%s\n' 'Local Vault AppRoles provisioned.'