# Mise en ligne sur le VPS

Le panneau est servi par Caddy sur https://agent-video-edit-detection.tanguym.fr, chaque outil sous
son chemin (`/runs/`, `/modele/`…). Le code est dans `/srv/agent_mots_cles`, cloné depuis GitHub.
`data/`, `model_results/`, `runs/` et `.env` ne sont pas dans git : **le VPS est leur seule source
de vérité**, les runs se lancent là-bas et plus sur le PC.

Publics en lecture : résumé des runs, conversations, parcours, agents. Sous mot de passe : modèle,
dataset, saut d'images. Le banc des annotateurs (appels payants) n'est pas publié.

## Mettre à jour le code

```bash
cd /srv/agent_mots_cles && git pull
sudo systemctl restart 'agent-interface@*'
```

Après un changement de `deploy/Caddyfile` :

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

## Changer le mot de passe

Le nom d'utilisateur et l'empreinte vivent seulement dans `/etc/caddy/agent_passwd` :

```bash
HASH=$(caddy hash-password)
echo "NOM_UTILISATEUR $HASH" | sudo tee /etc/caddy/agent_passwd >/dev/null
sudo systemctl reload caddy
```

## Lancer un run (payant)

Un run à la main, avec un budget explicite et des plafonds pour laisser le site répondre.
L'annotation part en batch (-50 %) : un run qui collecte peut durer jusqu'à 24 h, sans rien écrire
dans son journal pendant l'attente. Le suivre avec `python -m model_training_pipeline batch status`.

```bash
cd /srv/agent_mots_cles
sudo systemd-run --uid=ubuntu --working-directory=$PWD --unit=agent-run -p MemoryMax=5G -p CPUQuota=300% \
  .venv/bin/python -m agents --budget-usd 0.30 --data-budget 0 --annotation-usd 0 --final-tests 2
journalctl -u agent-run -f
```
