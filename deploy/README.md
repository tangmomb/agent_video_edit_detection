# Mise en ligne sur le VPS

Le panneau est servi par Caddy sur https://agent-video-edit-detection.tanguym.fr, chaque outil sous
son chemin (`/runs/`, `/modele/`…). Le code est dans `/srv/agent_mots_cles`, cloné depuis GitHub.
`data/`, `model_results/`, `runs/` et `.env` ne sont pas dans git : **le VPS est leur seule source
de vérité**, les runs se lancent là-bas et plus sur le PC.

Toutes les pages publiées sont publiques, sans mot de passe. Les pages qui reçoivent des vidéos
(modèle, saut d'images) sont bornées par `interface/quota.py` : 500 Mo et 5 min par vidéo, un envoi
traité à la fois, 3 vidéos par IP et 30 par jour en tout pour le modèle (GPU Modal), 5 et 50 pour le
saut d'images (CPU), envois effacés après 24 h. Le banc des annotateurs (`/banc/`, appels payants)
n'offre en ligne que les 3 annotateurs du dataset, sur ses propres vidéos, une fois par modèle, avec
3 vidéos de 3 min au plus par jour pour tout le monde (pas de limite par IP), 0,25 $ par lancement et 3 $ par jour pour tout le monde. Les réglages
sont en tête de chaque `server.py`.
Les adresses de `QUOTA_ILLIMITE` (`.env` du VPS, séparées par des virgules) n'ont aucune limite.

## Mettre à jour le code

Automatique à chaque push sur `master` (`.github/workflows/deploy.yml`) : `modal deploy` si
`encoders.py` ou `embed_modal.py` a changé, puis `deploy/update_vps.sh update` sur le VPS (`git pull`,
dépendances, Caddy, redémarrage des interfaces). Un push qui ne touche que des `.md` attend le
lancement de l'heure.

**Rien n'est déployé tant qu'un run tourne.** Le workflow repasse toutes les heures et rattrape le
retard après le run ; sans retard, il ne fait rien. Pour ne pas attendre l'heure suivante :

```bash
gh workflow run deploy.yml
```

GitHub désactive les lancements programmés après 60 jours sans activité sur un dépôt public : les
réactiver dans l'onglet Actions.

Mise en place, ou remplacement de la clé et des jetons : `deploy/setup_github_actions.sh`.

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

## Les autres projets du VPS

Le VPS Contabo (95.111.224.166) héberge aussi deux autres projets. `deploy/Caddyfile` sert les trois
sites, et `update_vps.sh` le recopie à chaque déploiement de ce dépôt. Les unités systemd ci-dessous,
elles, s'installent à la main (`sudo cp deploy/<unité> /etc/systemd/system/`, puis
`sudo systemctl daemon-reload` et `sudo systemctl enable --now <timer>`) :

- **rag_ionis** (https://ionis.tanguym.fr) : conteneurs Docker dans `~/rag_ionis`, déployés par la CI
  de son propre dépôt (`tangmomb/rag_ionis`) ; API sur 127.0.0.1:8006, préproduction sur 8007.
  Sauvegarde nocturne de sa base et de Phoenix vers S3 Scaleway : `rag-ionis-backup.sh` (dans
  `/usr/local/bin`), `rag-ionis-backup.service` et `.timer`. La restauration est décrite en tête du script.
- **fine-tuning** (https://fine-tuning.tanguym.fr) : le rapport `00_compte_rendu` du dépôt public
  `tangmomb/fine-tuning`, copie partielle dans `/srv/fine-tuning` mise à jour toutes les 10 minutes
  par `fine-tuning-pull.timer`.

## Surveillance du VPS

https://monitor.tanguym.fr : Beszel (`deploy/beszel/docker-compose.yml`) montre la RAM disponible et
ses pics, le CPU, le disque, le réseau et la consommation de chaque conteneur Docker, avec un historique.
Le hub n'écoute que sur 127.0.0.1:8090, derrière Caddy ; ses données sont dans `/srv/beszel`.
Démarrage ou mise à jour après un changement du compose :

```bash
docker compose -f /srv/agent_mots_cles/deploy/beszel/docker-compose.yml up -d
```

La première visite crée le compte administrateur. L'agent se branche ensuite par « Add System » :
hôte `/beszel_socket/beszel.sock`, puis `KEY=…` et `TOKEN=…` affichés par le hub, à écrire dans
`/srv/beszel/agent.env` (hors git) avant de relancer la commande ci-dessus.

## Le VPS Infomaniak

Le VPS Infomaniak (179.237.98.117) sert de relais YouTube à l'agent (service `yt-tunnel` ici, voir
`YT_PROXY` dans le README principal) et héberge les portfolios https://tanguym.fr et
https://portfolio-ia.tanguym.fr. Sa configuration est dans `deploy/infomaniak/` et se recopie à la main :
pas de déploiement automatique sur ce serveur.

Il est surveillé par le même Beszel, système `infomaniak` : agent 0.21.1 en binaire
(`/usr/local/bin/beszel-agent`, téléchargé depuis les releases GitHub et vérifié par somme de
contrôle), lancé par `deploy/infomaniak/beszel-agent.service` sous l'utilisateur système `beszel`.
Il se connecte lui-même au hub (`HUB_URL=https://monitor.tanguym.fr`) : aucun port ouvert. Sa clé
et son jeton sont dans `/etc/beszel-agent.env` (root, 0600), hors git.
