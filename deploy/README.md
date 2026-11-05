# Mise en ligne sur le VPS

Le panneau est servi par Caddy sur https://agent-video-edit-detection.tanguym.fr, chaque outil sous
son chemin (`/runs/`, `/modele/`…). Le code est dans `/srv/agent_mots_cles`, cloné depuis GitHub.
`data/`, `model_results/`, `runs/` et `.env` ne sont pas dans git : **le VPS est leur seule source
de vérité**, les runs se lancent là-bas et plus sur le PC.

Publics en lecture : résumé des runs, conversations, parcours, agents. Sous mot de passe : modèle,
dataset, saut d'images. Le banc des annotateurs (appels payants) n'est pas publié.

## Mettre à jour le code

Automatique à chaque push sur `master` (`.github/workflows/deploy.yml`) : `modal deploy` si
`encoders.py` ou `embed_modal.py` a changé, puis `deploy/update_vps.sh update` sur le VPS (`git pull`,
dépendances, Caddy, redémarrage des interfaces). Seuls les changements de `.md` ne déclenchent rien.

**Refusé tant qu'un run tourne**, le workflow apparaît alors en échec. À relancer à la fin du run :

```bash
gh workflow run deploy.yml
```

Mise en place, ou remplacement de la clé et des jetons : `deploy/setup_github_actions.sh`.

## Changer le mot de passe

Le nom d'utilisateur et l'empreinte vivent seulement dans `/etc/caddy/agent_passwd` :

```bash
HASH=$(caddy hash-password)
echo "NOM_UTILISATEUR $HASH" | sudo tee /etc/caddy/agent_passwd >/dev/null
sudo systemctl reload caddy
```

## Lancer un run (payant)

Un run à la main, avec un budget explicite et des plafonds pour laisser le site répondre.
Tout part en batch (-50 %), agents (un batch par tour) et annotation, sauf `gpt-6-luna`, en direct
car son batch est trop lent. Un run dure plus longtemps et peut rester des heures sur une attente,
sans rien écrire dans son journal. Le suivre
avec `python -m model_training_pipeline batch status`. `--direct-agents` et `--direct-annotation`
repassent en appels directs (prix plein). Le coût affiché des agents compte le pire cas (sans tarif
cache) : à comparer à la console Mistral.

```bash
cd /srv/agent_mots_cles
sudo systemd-run --uid=ubuntu --working-directory=$PWD --unit=agent-run -p MemoryMax=5G -p CPUQuota=300% \
  .venv/bin/python -m agents --budget-usd 0.30 --data-budget 0 --annotation-usd 0 --final-tests 2
journalctl -u agent-run -f
```
