#!/usr/bin/env bash
# Sauvegarde nocturne de rag_ionis (Contabo) vers le bucket S3 Scaleway du projet.
# Installé dans /usr/local/bin/rag-ionis-backup.sh, lancé par rag-ionis-backup.timer.
# Envoie la base Postgres (pg_dump -Fc) et la base SQLite de Phoenix sous
# backups/rag_ionis/<date>/, puis supprime les sauvegardes de plus de 30 jours.
# Restauration de la base :
#   rclone cat scw:<bucket>/backups/rag_ionis/<date>/postgres.dump | \
#     docker compose ... exec -T postgres sh -c 'pg_restore --clean --if-exists --single-transaction -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
set -euo pipefail

deploy_dir="${RAG_IONIS_DIR:-/home/ubuntu/rag_ionis}"
retention="${BACKUP_RETENTION:-30d}"
cd "${deploy_dir}"

set -a
# shellcheck disable=SC1091
. ./.env.production
set +a

export RCLONE_CONFIG_SCW_TYPE=s3
export RCLONE_CONFIG_SCW_PROVIDER=Scaleway
export RCLONE_CONFIG_SCW_ACCESS_KEY_ID="${S3_ACCESS_KEY_ID}"
export RCLONE_CONFIG_SCW_SECRET_ACCESS_KEY="${S3_SECRET_ACCESS_KEY}"
export RCLONE_CONFIG_SCW_REGION="${S3_REGION}"
export RCLONE_CONFIG_SCW_ENDPOINT="${S3_ENDPOINT_URL}"
export RCLONE_CONFIG_SCW_NO_CHECK_BUCKET=true
export RCLONE_CONFIG=/dev/null

prefix="scw:${S3_BUCKET_NAME}/backups/rag_ionis"
stamp="$(date -u +%Y-%m-%dT%H%MZ)"
compose=(docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production)

# Base Postgres : instantané cohérent, format custom (compressé).
"${compose[@]}" exec -T postgres sh -c 'pg_dump -Fc -U "$POSTGRES_USER" "$POSTGRES_DB"' </dev/null \
  | rclone rcat "${prefix}/${stamp}/postgres.dump"

# Phoenix (traces) : copie SQLite cohérente via l'API backup, le conteneur n'a ni cat ni sh.
"${compose[@]}" exec -T phoenix python -c '
import os, sqlite3, sys, tempfile
path = tempfile.mktemp(suffix=".db")
source = sqlite3.connect("/mnt/data/phoenix.db")
target = sqlite3.connect(path)
source.backup(target)
target.close()
source.close()
with open(path, "rb") as handle:
    sys.stdout.buffer.write(handle.read())
os.remove(path)
' </dev/null | rclone rcat "${prefix}/${stamp}/phoenix.db"

# Vérifie que les deux fichiers sont arrivés et ne sont pas vides.
for name in postgres.dump phoenix.db; do
  size="$(rclone size --json "${prefix}/${stamp}/${name}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["bytes"])')"
  if [[ "${size}" -lt 1024 ]]; then
    printf 'Sauvegarde suspecte : %s fait %s octets.\n' "${name}" "${size}" >&2
    exit 1
  fi
  printf '%s : %s octets\n' "${name}" "${size}"
done

rclone delete --min-age "${retention}" "${prefix}"
rclone rmdirs --leave-root "${prefix}" 2>/dev/null || true
printf 'Sauvegarde %s terminée.\n' "${stamp}"
