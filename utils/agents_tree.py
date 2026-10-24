"""Génère utils/agents_tree.html : l'arbre des agents et de leurs outils, lu dans le code de agents/
(descriptions, paramètres, modèle, tours), pour que la page suive le code.

    .venv/Scripts/python.exe utils/agents_tree.py [--from-run runs/<NNN>_…] [--examples exemples.json]

--from-run : chaque outil montre son premier appel réussi dans le journal de ce run (appel et réponse
réels). exemples.json (facultatif) : {nom d'outil: {"call": {...}, "response": ..., "note": "…"}},
pour les outils que le run n'a pas appelés ; --examples-note s'affiche quand un exemple n'a pas de note.
Sans --from-run ni --examples, la page garde les exemples qu'elle contient déjà (embarqués en JSON) :
on la régénère après chaque changement du code des agents sans avoir besoin d'un run.
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agents import orchestrator  # noqa: E402
from model_training_pipeline.lab import Lab  # noqa: E402

CSS = """
:root{--bg:#f7f6f3;--panel:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#d9d7d1;--code:#f0eee9;
--agent:#2f5bd3;--tool:#2e7d4f;--deleg:#b35c00;--req:#c0392b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151517;--panel:#1e1e21;
--fg:#ececee;--muted:#9a9aa0;--line:#34343a;--code:#2a2a2f;--agent:#7d9cff;--tool:#5fc58a;
--deleg:#f0a24a;--req:#ff7a6b}}
:root[data-theme="dark"]{--bg:#151517;--panel:#1e1e21;--fg:#ececee;--muted:#9a9aa0;--line:#34343a;
--code:#2a2a2f;--agent:#7d9cff;--tool:#5fc58a;--deleg:#f0a24a;--req:#ff7a6b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}
.toolbar{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
button{font:inherit;font-size:13px;padding:5px 12px;border:1px solid var(--line);background:var(--panel);
color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--muted)}
ul.tree{list-style:none;margin:0;padding-left:22px;border-left:1px solid var(--line)}
main>ul.tree{border-left:none;padding-left:0}
li.node{margin:6px 0;position:relative}
ul.tree ul.tree>li.node::before,ul.tree>li.node::before{content:"";position:absolute;left:-22px;top:17px;width:16px;border-top:1px solid var(--line)}
main>ul.tree>li.node::before{display:none}
summary{cursor:pointer;display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:6px 10px;
background:var(--panel);border:1px solid var(--line);border-radius:8px;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸";color:var(--muted);font-size:12px;width:10px}
details[open]>summary::before{content:"▾"}
.badge{font-size:11px;text-transform:uppercase;letter-spacing:.04em;padding:1px 7px;border-radius:10px;
border:1px solid currentColor;font-weight:600}
.badge.agent{color:var(--agent)}.badge.tool{color:var(--tool)}.badge.delegate{color:var(--deleg)}
.aname{font-weight:650;font-size:16px}
.tname{font-weight:600}
.meta{color:var(--muted);font-size:13px;margin-left:auto}
.body{padding:6px 12px 2px 30px}
.desc{margin:0 0 6px}
code{font-family:ui-monospace,Consolas,monospace;font-size:13px;background:var(--code);padding:1px 5px;border-radius:4px}
summary code{background:none;padding:0;font-size:14px}
ul.params{margin:0 0 6px;padding-left:18px;font-size:14px}
ul.params li{margin:3px 0}
.ptype{color:var(--muted);font-size:13px}
.penum{color:var(--muted);font-size:13px}
.req{color:var(--req);font-size:11px;margin-left:6px;text-transform:uppercase}
.pdesc{color:var(--muted)}
.noparam{color:var(--muted);font-size:13px;font-style:italic;margin-bottom:6px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-bottom:12px}
details.ex{margin:0 0 8px}
details.ex>summary{background:none;border:none;padding:2px 0;font-size:13px;color:var(--tool);font-weight:600}
.exnote{color:var(--muted);font-size:13px;margin:4px 0}
details.ex pre{margin:0 0 4px;white-space:pre-wrap;word-break:break-word;padding:10px 12px;background:var(--code);border-radius:6px;overflow-x:auto;font:12px/1.45 ui-monospace,Consolas,monospace;max-height:480px}
.exlabel{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin:6px 0 2px}
"""

DESC = {"orchestrator": "Orchestrateur : décide de la prochaine expérience et délègue aux trois autres agents.",
        "dataset_builder": "Dataset Builder : collecte de vraies vidéos YouTube, annotées image par image "
                           "par trois LLM (étiquette seulement s'ils sont d'accord).",
        "model_researcher": "Model Researcher : constitue les splits, choisit les représentations et les "
                            "modèles, entraîne, mène les expériences sur val.",
        "evaluator": "Evaluator : mesure, analyse les erreurs, vérifie les améliorations, seul accès au test."}

e = html.escape


class _Session:
    """Juste ce qu'il faut à orchestrator.make pour lire ses outils (aucun appel n'est fait)."""
    reserve_reached, spent, budget_usd, reserve_usd = False, 0.0, 1.0, 0.15

    def run(self, agent, task):
        raise RuntimeError("non utilisé")


def _param(name: str, spec: dict, required: bool) -> str:
    kind = spec.get("type", "")
    if kind == "array" and "items" in spec:
        kind += f" de {spec['items'].get('type', '')}"
    extra = ""
    enum = spec.get("enum") or spec.get("items", {}).get("enum")
    if enum:
        extra = " · " + " | ".join(enum)
    elif "minimum" in spec or "maximum" in spec:
        extra = f" · {spec.get('minimum', '')}…{spec.get('maximum', '')}"
    out = f'<li><code class="pname">{e(name)}</code>' + ('<span class=req>requis</span>' if required else "")
    out += f' <span class="ptype">{e(kind)}</span>' + (f'<span class="penum">{e(extra)}</span>' if extra else "")
    if spec.get("description"):
        out += f'<div class=pdesc>{e(spec["description"])}</div>'
    return out + "</li>"


def _example(ex: dict | None, note: str) -> str:
    if not ex:
        return ""
    if ex.get("missing"):  # outil non appelé pendant le run dont viennent les exemples
        return f'<p class="exnote">{e(ex["missing"])}</p>'
    out = f'<details class="ex"><summary>Exemple</summary><p class="exnote">{e(ex.get("note") or note)}</p>'
    if ex.get("call") is not None:
        out += f'<div class="exlabel">Appel</div><pre>{e(json.dumps(ex["call"], ensure_ascii=False, indent=2))}</pre>'
    resp = ex["response"] if isinstance(ex["response"], str) else json.dumps(ex["response"], ensure_ascii=False, indent=2)
    return out + f'<div class="exlabel">Réponse</div><pre>{e(resp)}</pre></details>'


def _tool(tool, kind: str, examples: dict, note: str, child: str = "", agent: str = "") -> str:
    props = tool.input_schema.get("properties", {})
    req = set(tool.input_schema.get("required", []))
    params = ('<ul class="params">' + "".join(_param(n, s, n in req) for n, s in props.items()) + "</ul>"
              if props else '<div class="noparam">aucun paramètre</div>')
    badge = "délègue" if kind == "delegate" else "outil"
    return (f'<li class="node {kind}"><details{" open" if kind == "delegate" else ""}><summary>'
            f'<span class="badge {kind}">{badge}</span><code class="tname">{e(tool.name)}</code></summary>'
            f'<div class="body"><p class="desc">{e(tool.description)}</p>{params}'
            f'{_example(examples.get(f"{agent}.{tool.name}") or examples.get(tool.name), note)}'
            f'</div></details>{child}</li>')


def _agent(agent, examples: dict, note: str, subagents: dict, open_: bool = False) -> str:
    items = []
    for t in agent.tools:
        sub = t.name.removeprefix("ask_")
        if sub in subagents:
            child = '<ul class="tree">' + _agent(subagents[sub], examples, note, {}) + "</ul>"
            items.append(_tool(t, "delegate", examples, note, child, agent.name))
        else:
            items.append(_tool(t, "tool", examples, note, agent=agent.name))
    meta = f"{agent.model} · {agent.max_turns} tours max · {len(agent.tools)} outils"
    return (f'<li class="node agent"><details{" open" if open_ else ""}><summary><span class="badge agent">agent</span>'
            f'<span class="aname">{e(agent.name)}</span><span class="meta">{e(meta)}</span></summary>'
            f'<div class="body"><p class="desc">{e(DESC.get(agent.name, ""))}</p></div>'
            f'<ul class="tree">{"".join(items)}</ul></details></li>')


def _agents() -> list:
    """L'orchestrateur puis ses trois sous-agents, construits comme pour un run."""
    lab = Lab()
    return [orchestrator.make(lab, _Session()),
            *(mod.make(lab) for mod in (orchestrator.dataset_builder, orchestrator.model_researcher,
                                        orchestrator.evaluator))]


def render(examples: dict, note: str) -> str:
    root, *sub = _agents()
    subs = {a.name: a for a in sub}
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agents et outils</title><style>{CSS}</style></head>
<body><main>
<h1>Agents et outils</h1>
<p class="sub">Généré depuis le code de <code>agents/</code> par <code>utils/agents_tree.py</code> : l'orchestrateur délègue aux trois sous-agents via ses outils <code>ask_*</code>. Les données sont de vraies vidéos YouTube, chaque image étiquetée par l'accord de trois LLM.</p>
<div class="toolbar"><button id="open">Tout déplier</button><button id="close">Tout replier</button></div>
<div class="legend"><span><span class="badge agent">agent</span> agent LLM (via LiteLLM)</span>
<span><span class="badge tool">outil</span> appel au labo</span>
<span><span class="badge delegate">délègue</span> lance un sous-agent</span></div>
<ul class="tree">{_agent(root, examples, note, subs, open_=True)}</ul>
</main><script>
document.getElementById('open').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=true);
document.getElementById('close').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=false);
</script>
<script type="application/json" id="examples">{_embed(examples)}</script></body></html>
"""


def _embed(examples: dict) -> str:
    # « </ » fermerait la balise <script> : échappé, le JSON reste valide
    return json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")


def examples_from_page(page: Path) -> dict:
    """Exemples embarqués dans une page déjà générée (vide si elle n'en a pas)."""
    if not page.exists():
        return {}
    m = re.search(r'<script type="application/json" id="examples">(.*?)</script>',
                  page.read_text(encoding="utf-8"), re.S)
    return json.loads(m.group(1)) if m else {}


def examples_from_run(run_dir: Path) -> dict:
    """Premier appel réussi de chaque outil, par agent, dans le journal d'un run
    (runs/<NNN>_…/pipeline.json) : appel et réponse réels, tels que l'agent les a reçus. Clés
    « agent.outil » (un même outil, comme dataset_info, ne renvoie pas la même chose à chaque agent)."""
    log = json.loads((run_dir / "pipeline.json").read_text(encoding="utf-8"))
    out = {}
    for s in log["steps"]:
        key = f"{s['agent']}.{s.get('tool')}"
        if s["type"] != "tool" or key in out or s["error"]:
            continue
        result = s["result"]
        if isinstance(result, dict) and "error" in result:
            continue  # refus d'un outil (budget épuisé…) : on attend un appel abouti
        call = json.loads(s["input"]) if isinstance(s["input"], str) else s["input"]
        out[key] = {"call": call or None,
                          "response": "\n".join(result) if isinstance(result, list) else result,
                          "note": f"Appel et réponse réels (run {run_dir.name.split('_')[0]} du "
                                  f"{s['time'][8:10]}/{s['time'][5:7]}/{s['time'][:4]}, agent {s['agent']})."}
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="utils/agents_tree.py")
    p.add_argument("--from-run", type=Path, help="tire les exemples du journal de ce run (runs/<NNN>_…)")
    p.add_argument("--examples", type=Path, help="exemples d'appels et de réponses par outil (JSON)")
    p.add_argument("--examples-note", default="", help="d'où viennent les exemples")
    p.add_argument("--out", type=Path, default=HERE / "agents_tree.html")
    a = p.parse_args()
    if a.examples:
        examples = json.loads(a.examples.read_text(encoding="utf-8"))
    else:  # sans source d'exemples : ceux que la page contient déjà
        examples = {} if a.from_run else examples_from_page(a.out)
    if a.from_run:
        examples.update(examples_from_run(a.from_run))
        # outils sans appel dans ce run : dit-le plutôt que de montrer un exemple d'ailleurs
        run = a.from_run.name.split("_")[0]
        for agent in _agents():
            for t in agent.tools:
                examples.setdefault(f"{agent.name}.{t.name}",
                                    {"missing": f"Pas d'exemple : outil non appelé pendant le run {run}."})
    a.out.write_text(render(examples, a.examples_note), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
