#!/usr/bin/env bash
# Mise à jour du VPS, seule commande permise à la clé SSH de GitHub Actions (setup_github_actions.sh).
#   check  : affiche "running" si un run tourne, sinon le commit en place
#   update : git pull, dépendances et Caddy s'ils ont changé, redémarrage des interfaces
# Jamais pendant un run : il chargerait du nouveau code à côté de l'ancien.
set -euo pipefail
cd /srv/agent_mots_cles

action="${SSH_ORIGINAL_COMMAND:-${1:-}}"

running() { systemctl is-active --quiet agent-run; }

case "$action" in
check)
    if running; then echo running; else git rev-parse HEAD; fi
    ;;
update)
    if running; then  # lancé depuis le contrôle du workflow
        echo "Un run vient de démarrer (agent-run) : VPS non mis à jour, nouvel essai à la prochaine heure." >&2
        exit 1
    fi
    old=$(git rev-parse HEAD)
    git pull --ff-only --quiet
    changed=$(git diff --name-only "$old" HEAD)

    if grep -qx pyproject.toml <<<"$changed"; then
        .venv/bin/pip install --quiet -e ".[model_training_pipeline]"
    fi
    if grep -qx deploy/Caddyfile <<<"$changed"; then
        sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
        sudo systemctl reload caddy
    fi
    if grep -qx deploy/agent-interface@.service <<<"$changed"; then
        sudo cp deploy/agent-interface@.service /etc/systemd/system/
        sudo systemctl daemon-reload
    fi

    sudo systemctl restart 'agent-interface@*'
    sleep 10  # un serveur qui plante au démarrage n'est plus actif
    failed=0
    for unit in $(systemctl list-units 'agent-interface@*' --all --plain --no-legend | awk '{print $1}'); do
        if ! systemctl is-active --quiet "$unit"; then
            echo "Interface arrêtée : $unit" >&2
            failed=1
        fi
    done
    [ "$failed" -eq 0 ]
    echo "VPS en $(git rev-parse --short HEAD)"
    ;;
*)
    echo "Action inconnue : '$action' (check ou update)" >&2
    exit 2
    ;;
esac
