"""Relance des appels aux API externes (agents, annotateurs, batchs, descriptions d'images) après une
erreur passagère : surcharge, erreur serveur, coupure réseau. On retente toutes les WAIT_S secondes,
sans limite : une panne du fournisseur met le système en pause au lieu de l'arrêter (run 008 : un
429 « backend_out_of_capacity » de zai-glm-5-3-flash avait coupé une collecte, et plus d'une minute
et demie de relances n'avait pas suffi). Les autres erreurs (requête invalide, clé refusée…)
remontent tout de suite : réessayer n'y changerait rien.
"""

import sys
import time

WAIT_S = 60
# Codes HTTP passagers ; les SDK OpenAI, Anthropic, Mistral et LiteLLM les mettent dans status_code.
STATUS = {408, 425, 429, 500, 502, 503, 504, 529}
# Erreurs sans code HTTP : coupures et délais dépassés (httpx, SDK OpenAI et Anthropic, LiteLLM).
NAMES = {"APIConnectionError", "APITimeoutError", "Timeout", "ConnectError", "ConnectTimeout", "ReadTimeout",
         "ReadError", "WriteTimeout", "PoolTimeout", "RemoteProtocolError", "ServiceUnavailableError",
         "RateLimitError", "InternalServerError", "ConnectionError"}


def transient(e: BaseException) -> bool:
    return getattr(e, "status_code", None) in STATUS or type(e).__name__ in NAMES


def forever(call, what: str = "appel", say=None):
    """call() jusqu'à ce qu'il réussisse ou lève une erreur qui n'est pas passagère ; say : où
    signaler chaque attente (défaut : stderr)."""
    tries = 0
    while True:
        try:
            return call()
        except Exception as e:
            if not transient(e):
                raise
            tries += 1
            msg = f"[{what}] {type(e).__name__} {getattr(e, 'status_code', '') or ''}: essai {tries}, nouvel essai dans {WAIT_S} s"
            (say or (lambda m: print(m, file=sys.stderr, flush=True)))(msg)
            time.sleep(WAIT_S)
