#! /bin/bash

docker compose --profile local-vault up -d --wait db vault
sh scripts/provision-local-vault.sh

# DBスキーマを更新してからアプリを起動
docker compose --profile setup run --build --rm migrate
docker compose --profile local-vault up -d --build
docker compose --profile image-generation up -d comfyui
docker compose --profile local-vault --profile agent up --build -d --force-recreate worker
