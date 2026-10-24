"""Génère utils/agents_tree.html : l'arbre des agents et de leurs outils, lu dans le code de agents/
(descriptions, paramètres, modèle, tours), pour que la page suive le code.

    .venv/Scripts/python.exe utils/agents_tree.py [--examples exemples.json] [--examples-note "…"]

exemples.json (facultatif) : {nom d'outil: {"call": {...}, "response": ...}} ; chaque exemple est
affiché sous son outil, précédé de --examples-note (d'où viennent les exemples).
"""

import argparse
import html
import json
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
    out = f'<details class="ex"><summary>Exemple</summary><p class="exnote">{e(ex.get("note") or note)}</p>'
    if ex.get("call") is not None:
        out += f'<div class="exlabel">Appel</div><pre>{e(json.dumps(ex["call"], ensure_ascii=False, indent=2))}</pre>'
    resp = ex["response"] if isinstance(ex["response"], str) else json.dumps(ex["response"], ensure_ascii=False, indent=2)
    return out + f'<div class="exlabel">Réponse</div><pre>{e(resp)}</pre></details>'


def _tool(tool, kind: str, examples: dict, note: str, child: str = "") -> str:
    props = tool.input_schema.get("properties", {})
    req = set(tool.input_schema.get("required", []))
    params = ('<ul class="params">' + "".join(_param(n, s, n in req) for n, s in props.items()) + "</ul>"
              if props else '<div class="noparam">aucun paramètre</div>')
    badge = "délègue" if kind == "delegate" else "outil"
    return (f'<li class="node {kind}"><details{" open" if kind == "delegate" else ""}><summary>'
            f'<span class="badge {kind}">{badge}</span><code class="tname">{e(tool.name)}</code></summary>'
            f'<div class="body"><p class="desc">{e(tool.description)}</p>{params}'
            f'{_example(examples.get(tool.name), note)}</div></details>{child}</li>')


def _agent(agent, examples: dict, note: str, subagents: dict, open_: bool = False) -> str:
    items = []
    for t in agent.tools:
        sub = t.name.removeprefix("ask_")
        if sub in subagents:
            child = '<ul class="tree">' + _agent(subagents[sub], examples, note, {}) + "</ul>"
            items.append(_tool(t, "delegate", examples, note, child))
        else:
            items.append(_tool(t, "tool", examples, note))
    meta = f"{agent.model} · {agent.max_turns} tours max · {len(agent.tools)} outils"
    return (f'<li class="node agent"><details{" open" if open_ else ""}><summary><span class="badge agent">agent</span>'
            f'<span class="aname">{e(agent.name)}</span><span class="meta">{e(meta)}</span></summary>'
            f'<div class="body"><p class="desc">{e(DESC.get(agent.name, ""))}</p></div>'
            f'<ul class="tree">{"".join(items)}</ul></details></li>')


def render(examples: dict, note: str) -> str:
    lab = Lab()
    root = orchestrator.make(lab, _Session())
    subs = {name: mod.make(lab) for name, mod in (("dataset_builder", orchestrator.dataset_builder),
                                                  ("model_researcher", orchestrator.model_researcher),
                                                  ("evaluator", orchestrator.evaluator))}
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agents et outils</title><style>{CSS}</style></head>
<body><main>
<h1>Agents et outils</h1>
<p class="sub">Généré depuis le code de <code>agents/</code> par <code>utils/agents_tree.py</code> : l'orchestrateur délègue aux trois sous-agents via ses outils <code>ask_*</code>. Les données sont de vraies vidéos YouTube, chaque image étiquetée par l'accord de trois LLM.</p>
<div class="toolbar"><button id="open">Tout déplier</button><button id="close">Tout replier</button></div>
<div class="legend"><span><span class="badge agent">agent</span> agent Mistral</span>
<span><span class="badge tool">outil</span> appel au labo</span>
<span><span class="badge delegate">délègue</span> lance un sous-agent</span></div>
<ul class="tree">{_agent(root, examples, note, subs, open_=True)}</ul>
</main><script>
document.getElementById('open').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=true);
document.getElementById('close').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=false);
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(prog="utils/agents_tree.py")
    p.add_argument("--examples", type=Path, help="exemples d'appels et de réponses par outil (JSON)")
    p.add_argument("--examples-note", default="", help="d'où viennent les exemples")
    p.add_argument("--out", type=Path, default=HERE / "agents_tree.html")
    a = p.parse_args()
    examples = json.loads(a.examples.read_text(encoding="utf-8")) if a.examples else {}
    a.out.write_text(render(examples, a.examples_note), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
