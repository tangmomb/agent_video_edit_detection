#!/usr/bin/env bash
# À lancer une fois depuis le PC (Git Bash), ou à nouveau pour remplacer la clé et les jetons :
#   bash deploy/setup_github_actions.sh <adresse IP du VPS>
# Crée la clé SSH de GitHub Actions, limitée sur le VPS à deploy/update_vps.sh, et les secrets du dépôt.
# Avant : créer un jeton dédié sur https://modal.com/settings/tokens ; il est demandé ici, sans affichage.
set -euo pipefail

host="${1:?Adresse IP du VPS manquante}"
vps="ubuntu@$host"
admin_key="$HOME/.ssh/admin_contabo_infomaniak"
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT

read -rsp "MODAL_TOKEN_ID : " modal_id; echo
read -rsp "MODAL_TOKEN_SECRET : " modal_secret; echo

ssh-keygen -q -t ed25519 -N "" -C github-actions-deploy -f "$dir/deploy"

# La clé ne peut lancer que le script : pas de shell, pas de tunnel.
line="command=\"bash /srv/agent_mots_cles/deploy/update_vps.sh\",restrict $(cat "$dir/deploy.pub")"
ssh -i "$admin_key" "$vps" "sed -i '/github-actions-deploy/d' ~/.ssh/authorized_keys && cat >> ~/.ssh/authorized_keys" <<<"$line"

gh secret set VPS_SSH_KEY < "$dir/deploy"
printf '%s' "$vps" | gh secret set VPS_TARGET
# Lue par la connexion déjà de confiance : GitHub Actions refusera un autre serveur à cette adresse.
ssh -i "$admin_key" "$vps" 'cat /etc/ssh/ssh_host_ed25519_key.pub' | awk -v h="$host" '{print h, $1, $2}' \
    | gh secret set VPS_KNOWN_HOSTS
printf '%s' "$modal_id" | gh secret set MODAL_TOKEN_ID
printf '%s' "$modal_secret" | gh secret set MODAL_TOKEN_SECRET

echo "Fait. Test : gh workflow run deploy.yml"
