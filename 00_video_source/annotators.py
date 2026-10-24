"""Annotation d'une image par des LLM de vision : « l'image est-elle brute de la caméra ou contient-elle
un ajout de montage ? ». Une requête par image, réponse {"ajout": true|false}.

Le banc utils/llm_bench compare tous les modèles de MODELS ; le dataset est étiqueté par les trois
modèles d'ANNOTATORS (choisis sur le banc) : une image n'a d'étiquette que s'ils sont tous d'accord.
"""

import base64
import io
import json
import os
from pathlib import Path

from PIL import Image

from . import config  # noqa: F401  (charge le .env)

# Modèles comparés (docs.mistral.ai/models, developers.openai.com/api/docs/pricing,
# platform.claude.com, octobre 2026) : identifiant d'API, nom, lecture d'images, tarif en $ par
# million de tokens, fournisseur.
MODELS = [
    ("mistral-large-4-0", "Mistral Large 4", True, {"input": 1.36, "cached_input": 0.14, "output": 4.18}, "mistral"),
    ("mistral-large-2512", "Mistral Large 3", True, {"input": 0.50, "output": 1.50}, "mistral"),
    ("mistral-small-2603", "Mistral Small 4", True, {"input": 0.15, "output": 0.60}, "mistral"),
    ("ministral-14b-2512", "Ministral 3 14B", True, {"input": 0.20, "output": 0.20}, "mistral"),
    ("zai-glm-5-3-flash", "Z.ai GLM 5.3 Flash", True, {"input": 0.15, "cached_input": 0.03, "output": 0.50}, "mistral"),
    ("gpt-6-luna", "GPT-6 Luna", True, {"input": 0.10, "cached_input": 0.01, "output": 0.50}, "openai"),
    ("gpt-5.6-luna", "GPT-5.6 Luna", True, {"input": 0.20, "cached_input": 0.02, "output": 1.20}, "openai"),
    ("claude-haiku-5-5", "Claude Haiku 5.5", True, {"input": 0.10, "output": 0.50}, "anthropic"),
]
PRICES = {m[0]: m[3] for m in MODELS}
PROVIDER = {m[0]: m[4] for m in MODELS}
KEYS = {"mistral": "MISTRAL_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
REASONING = "medium"  # même effort de raisonnement pour tous les modèles…
NO_REASONING = "none"  # …sauf ceux dont l'API refuse « medium » (zai-glm-5-3-flash : none ou high)
MAX_SIDE = 768  # plus grand côté envoyé ; le nombre de tokens d'une image suit son nombre de pixels

# Les trois annotateurs du dataset (meilleurs du banc, octobre 2026).
ANNOTATORS = ("gpt-6-luna", "gpt-5.6-luna", "zai-glm-5-3-flash")

PROMPT = """Dis si cette image est brute de la caméra ou non.
Réponds true dès que quelque chose a été ajouté au montage : habillage, texte, titre, sous-titres,
photo ou image incrustée, logo, cadre, effet graphique ou tout autre élément qui ne vient pas de la caméra.
Réponds false seulement si l'image est telle que la caméra l'a filmée.
Réponds uniquement en JSON : {"ajout": true|false}"""


def client(provider: str):
    key = os.environ.get(KEYS[provider])
    if not key:
        raise ValueError(f"{KEYS[provider]} absente du .env : impossible d'interroger {provider}")
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=key)
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=key)
    from mistralai.client import Mistral
    return Mistral(api_key=key)


def data_url(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def cost(model: str, tokens: dict) -> float:
    p = PRICES[model]
    fresh = tokens["input"] - tokens["cached_input"]
    return (fresh * p["input"] + tokens["cached_input"] * p.get("cached_input", p["input"])
            + tokens["output"] * p["output"]) / 1e6


def _text(content) -> str:
    """Texte de la réponse : avec le raisonnement, content est une liste de morceaux (réflexion puis
    texte) ; on ne garde que le texte."""
    if isinstance(content, str):
        return content
    get = lambda c, k: c.get(k) if isinstance(c, dict) else getattr(c, k, None)  # noqa: E731
    return "".join(get(c, "text") or "" for c in content or [] if get(c, "type") == "text")


def _usage(u) -> dict:
    """Tokens d'une réponse chat completions (objet du SDK ou dict brut d'un résultat de batch)."""
    get = lambda o, k: (o.get(k) if isinstance(o, dict) else getattr(o, k, None)) if o else None  # noqa: E731
    cached = get(get(u, "prompt_tokens_details"), "cached_tokens") or 0
    return {"input": get(u, "prompt_tokens") or 0, "cached_input": cached, "output": get(u, "completion_tokens") or 0}


def chat_body(provider: str, model: str, path: Path, max_side: int, reasoning: str | None) -> dict:
    """Corps d'une requête chat completions (OpenAI ou Mistral) : consigne + une image. Le même
    corps sert à l'appel direct et aux lignes d'un batch."""
    url = data_url(path, max_side)
    if provider == "openai":
        content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"}, "max_completion_tokens": 8000}
    elif provider == "mistral":
        content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": url}]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"}, "max_tokens": 8000}
    else:
        raise ValueError(f"pas de requête chat completions pour {provider}")
    return {**body, **({"reasoning_effort": reasoning} if reasoning else {})}


def read_chat(response) -> tuple[str, dict]:
    """Texte et tokens d'une réponse chat completions (objet du SDK ou dict brut d'un batch)."""
    if isinstance(response, dict):
        return _text(response["choices"][0]["message"]["content"]), _usage(response.get("usage"))
    return _text(response.choices[0].message.content), _usage(response.usage)


def ask(provider: str, api, model: str, path: Path, max_side: int, reasoning: str | None) -> tuple[str, dict]:
    """Une requête (consigne + une image) ; renvoie le texte de la réponse et les tokens consommés.
    reasoning=None : sans effort de raisonnement imposé."""
    if provider == "anthropic":
        url = data_url(path, max_side)
        # Pas de fallbacks : on compare des modèles, la réponse doit venir du modèle demandé.
        r = api.messages.create(
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": url.split(",", 1)[1]}},
                {"type": "text", "text": PROMPT}]}],
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "object", "properties": {"ajout": {"type": "boolean"}},
                "required": ["ajout"], "additionalProperties": False}},
                **({"effort": reasoning} if reasoning else {})})
        u = r.usage
        cached, created = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        text = "" if r.stop_reason == "refusal" else "".join(b.text for b in r.content if b.type == "text")
        return text, {"input": u.input_tokens + cached + created, "cached_input": cached, "output": u.output_tokens}
    body = chat_body(provider, model, path, max_side, reasoning)
    r = api.chat.completions.create(**body) if provider == "openai" else api.chat.complete(**body)
    return read_chat(r)


def refuses_reasoning(e: Exception) -> bool:
    """L'API a refusé l'effort de raisonnement demandé (le modèle ne raisonne pas)."""
    return any(w in str(e).lower() for w in ("reason", "effort"))


def parse(text: str) -> bool | None:
    """Réponse {"ajout": …} -> True/False ; None si elle est illisible."""
    try:
        answer = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
    except json.JSONDecodeError:
        return None
    return bool(answer["ajout"]) if isinstance(answer, dict) and "ajout" in answer else None


_no_reasoning: set[str] = set()  # modèles dont l'API a refusé « medium » : passés en « none »


def vote(path: Path, apis: dict | None = None, models=ANNOTATORS, max_side: int = MAX_SIDE) -> dict:
    """Fait annoter une image par chaque modèle. Renvoie {"votes": {modèle: bool|None}, "cost_usd",
    "label": 1|0|None} : label n'est donné que si tous les modèles ont répondu la même chose.
    apis : {fournisseur: client} (créés à la demande sinon)."""
    apis = apis if apis is not None else {}
    votes, spent = {}, 0.0
    for model in models:
        provider = PROVIDER[model]
        if provider not in apis:
            apis[provider] = client(provider)
        reasoning = NO_REASONING if model in _no_reasoning else REASONING
        try:
            text, used = ask(provider, apis[provider], model, path, max_side, reasoning)
        except Exception as e:
            if reasoning == NO_REASONING or not refuses_reasoning(e):
                raise
            _no_reasoning.add(model)
            text, used = ask(provider, apis[provider], model, path, max_side, NO_REASONING)
        spent += cost(model, used)
        votes[model] = parse(text)
    return {"votes": votes, "cost_usd": spent, "label": consensus(votes)}


def consensus(votes: dict, models=ANNOTATORS) -> int | None:
    """1 / 0 si tous les modèles ont voté la même chose ; None sinon (désaccord, réponse illisible ou
    vote manquant)."""
    values = {votes.get(m) for m in models}
    return int(values.pop()) if len(values) == 1 and None not in values else None
