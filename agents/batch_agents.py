"""Appels des agents en batch Mistral (-50 %), un batch d'une requête par tour.

LiteLLM garde tout son travail (traduction de la requête et de la réponse, raisonnement, outils) :
seul change son client HTTP, qui envoie le corps de la requête en batch au lieu de le poster, attend
le résultat, et le rend comme une réponse HTTP. Mesuré le 04/11/2026 : 10 batchs d'un appel en 23 à
26 s, outils et cache fonctionnels. Pas de repli en direct : la seule limite est le délai de 24 h de
l'API, après lequel le batch est renvoyé (le VPS tourne en continu).
"""

import json
import tempfile
import time
from pathlib import Path

import httpx
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from model_training_pipeline import batch
from video_source import annotators

EVERY_S = 5  # entre deux lectures de l'état du batch
# Facture non vérifiée pour les tokens en cache : le coût compte le pire cas, -50 % sur tout mais
# sans le tarif cache, pour que le budget d'un run tienne quoi qu'il arrive.
DISCOUNT = batch.DISCOUNT


class _BatchError(Exception):
    """Batch expiré, annulé, introuvable ou en échec sans résultat : passager, renvoyé par retry.forever."""
    status_code = 503


class BatchClient(HTTPHandler):
    def __init__(self):
        super().__init__()
        self._api = None

    def post(self, url, data=None, json=None, headers=None, **kwargs):  # noqa: A002
        body = _loads(data if data is not None else json)
        return self._run(url, body)

    def _run(self, url: str, body: dict) -> httpx.Response:
        self._api = self._api or annotators.client("mistral")
        model = body.pop("model")
        body.pop("stream", None)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
            fh.write(_dumps({"custom_id": "tour", "body": body}) + "\n")
        path = Path(fh.name)
        try:
            job_id = batch._upload_and_create("mistral", self._api, model, path)
        finally:
            path.unlink(missing_ok=True)
        while True:
            batch.heartbeat()  # le journal du run ne bouge pas pendant l'attente
            try:
                job = batch._job("mistral", self._api, job_id)
            except Exception as e:
                if getattr(e, "status_code", None) == 404:  # toujours introuvable : le tour est renvoyé
                    raise _BatchError(f"batch {job_id} introuvable") from e
                raise
            if str(job.status) in batch.DONE["mistral"]:
                break
            time.sleep(EVERY_S)
        lines = (batch._download("mistral", self._api, job.output_file)
                 + batch._download("mistral", self._api, job.error_file))
        if not lines:
            raise _BatchError(f"batch {job_id} : {job.status}, aucun résultat")
        response = lines[0].get("response") or {}
        result = httpx.Response(response.get("status_code") or 500, json=response.get("body") or lines[0],
                                request=httpx.Request("POST", url))
        result.raise_for_status()  # comme un appel direct : LiteLLM en fait l'erreur du fournisseur
        return result


def _loads(data) -> dict:
    if isinstance(data, dict):
        return dict(data)
    return json.loads(data)


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def completion(litellm_completion):
    """litellm.completion, en batch pour les modèles Mistral (les autres restent en appel direct)."""
    client = BatchClient()

    def call(**kwargs):
        if kwargs.get("model", "").startswith("mistral/"):
            kwargs["client"] = client
        return litellm_completion(**kwargs)
    return call
