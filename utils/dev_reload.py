"""Relance un serveur à chaque modification d'un fichier .py du projet (développement).

    .venv/Scripts/python.exe utils/dev_reload.py interface/server.py [arguments du serveur]

Le serveur tourne avec DEV_RELOAD=1 : ses pages se rechargent toutes seules quand leur HTML change
ou que le serveur redémarre (voir /__dev dans les serveurs). Ctrl+C arrête tout.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WATCH = ["interface", "utils", "agents", "model_training_pipeline", "00_video_source"]
POLL_S = 0.7
BOOT = time.time()

# Script ajouté aux pages en mode développement : il recharge la page quand le serveur redémarre
# (BOOT change) ou que son HTML est modifié.
LIVE_RELOAD = """<script>
(() => {
  let last = null;
  setInterval(async () => {
    try {
      const s = await (await fetch("/__dev", { cache: "no-store" })).json();
      const key = `${s.boot}|${s.html}`;
      if (last !== null && key !== last) location.reload();
      last = key;
    } catch (e) { /* serveur en train de redémarrer */ }
  }, 1000);
})();
</script>"""


def enabled() -> bool:
    return os.environ.get("DEV_RELOAD") == "1"


def status(html: Path) -> dict:
    """Réponse de /__dev : démarrage du serveur et date de modification de la page."""
    return {"boot": BOOT, "html": html.stat().st_mtime if html.exists() else None}


def page(html: Path) -> bytes:
    """La page, avec le script de rechargement juste avant </body>."""
    return html.read_text(encoding="utf-8").replace("</body>", LIVE_RELOAD + "\n</body>", 1).encode()


def snapshot() -> dict[Path, float]:
    out = {}
    for d in WATCH:
        for p in (ROOT / d).rglob("*.py"):
            if "__pycache__" not in p.parts:
                try:
                    out[p] = p.stat().st_mtime
                except OSError:
                    pass
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = [sys.executable, *sys.argv[1:]]
    env = {**os.environ, "DEV_RELOAD": "1", "PYTHONUNBUFFERED": "1"}
    name = Path(sys.argv[1]).parent.name
    seen = snapshot()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    try:
        while True:
            time.sleep(POLL_S)
            now = snapshot()
            changed = [p for p in now.keys() | seen.keys() if now.get(p) != seen.get(p)]
            seen = now
            if changed:
                print(f"[reload {name}] {', '.join(sorted(p.name for p in changed)[:3])} modifié : redémarrage",
                      flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
