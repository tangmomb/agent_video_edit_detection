"""Génère interface/agents_tree/agents_tree.html : l'arbre des agents et de leurs outils, lu dans le code de agents/
(modèle, tours, outils), avec pour chaque outil un exemple d'appel et de réponse.

    .venv/Scripts/python.exe interface/agents_tree/agents_tree.py [--from-run runs/<NNN>_…] [--examples exemples.json]

--from-run : chaque outil montre son premier appel réussi dans le journal de ce run (appel et réponse
réels). exemples.json (facultatif) : {nom d'outil: {"call": {...}, "response": ..., "note": "…"}},
pour les outils que le run n'a pas appelés ; --examples-note s'affiche quand un exemple n'a pas de note.
Sans --from-run ni --examples, la page garde les exemples qu'elle contient déjà (embarqués en JSON) :
on la régénère après chaque changement du code des agents sans avoir besoin d'un run. Les exemples
des outils sans effet (READ_ONLY) sont recalculés sur l'état actuel à chaque génération (gratuit,
sauf l'analyse visuelle d'analyze_errors : --summary-usd) ; --no-refresh les laisse tels quels.
"""

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from agents import orchestrator  # noqa: E402
from model_training_pipeline.lab import Lab  # noqa: E402

CSS = """
:root{color-scheme:dark;--bg:#151517;--panel:#1e1e21;--fg:#ececee;--muted:#9a9aa0;--line:#34343a;--code:#2a2a2f;
--agent:#7d9cff;--tool:#5fc58a;--deleg:#f0a24a;--req:#ff7a6b;
--sys-fg:#c4adff;--user-fg:#8fe0b0;--tools-fg:#f2b36b}
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
code{font-family:ui-monospace,Consolas,monospace;font-size:13px;background:var(--code);padding:1px 5px;border-radius:4px}
summary code{background:none;padding:0;font-size:14px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-bottom:12px}
.exnote{color:var(--muted);font-size:13px;margin:4px 0}
.body pre{margin:0 0 4px;white-space:pre-wrap;word-break:break-word;padding:10px 12px;background:var(--code);border-radius:6px;overflow-x:auto;font:12px/1.45 ui-monospace,Consolas,monospace;max-height:480px}
.sys{color:var(--sys-fg)}.user{color:var(--user-fg)}.tools{color:var(--tools-fg)}
.body pre.llm{max-height:640px}
.llm b{display:block;margin:10px 0 2px;font-weight:700}.llm b:first-child{margin-top:0}
.legend2{font-size:12px;margin:6px 0 2px}.legend2 span{font-weight:600}
ul.params{margin:0 0 6px;padding-left:18px;font-size:14px}
ul.params li{margin:3px 0}
.ptype{color:var(--muted);font-size:13px}
.req{color:var(--req);font-size:11px;margin-left:6px;text-transform:uppercase}
.pdesc{color:var(--muted);font-size:13px}
.noparam{color:var(--muted);font-size:13px;font-style:italic;margin:0 0 6px}
.exlabel{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin:6px 0 2px}
"""

e = html.escape


class _Session:
    """Juste ce qu'il faut à orchestrator.make pour lire ses outils (aucun appel n'est fait)."""
    reserve_reached, spent, budget_usd, reserve_usd = False, 0.0, 1.0, 0.15

    def run(self, agent, task):
        raise RuntimeError("non utilisé")


def _example(ex: dict | None, note: str) -> str:
    if not ex:
        return "<p class=\"exnote\">Pas d'exemple.</p>"
    if ex.get("missing"):  # outil non appelé pendant le run dont viennent les exemples
        return f'<p class="exnote">{e(ex["missing"])}</p>'
    out = ""  # d'où vient l'exemple (note) : gardé dans le JSON embarqué, pas affiché
    if ex.get("call") is not None:
        out += f'<div class="exlabel">Appel</div><pre>{e(json.dumps(ex["call"], ensure_ascii=False, indent=2))}</pre>'
    resp = ex["response"] if isinstance(ex["response"], str) else json.dumps(ex["response"], ensure_ascii=False, indent=2)
    return out + f'<div class="exlabel">Réponse</div><pre>{e(resp)}</pre>'


def _llm_input(agent, user: str | None, user_origin: str) -> str:
    """Tout ce que reçoit le modèle de l'agent, dans un seul encart en trois couleurs : prompt
    système (code), prompt utilisateur (user_origin) et outils (paramètre tools de la requête,
    envoyé à chaque tour à part des messages)."""
    tools = json.dumps([t.schema() for t in agent.tools], ensure_ascii=False, indent=2)
    body = f'<b>SYSTÈME</b>{e(agent.system)}'
    if user is not None:
        body += f'</span><span class="user"><b>UTILISATEUR</b>{e(user)}'
    return ('<div class="exlabel">Ce que reçoit le modèle</div><div class="legend2">'
            '<span class="sys">système</span> : fixe, écrit dans le code · '
            f'<span class="user">utilisateur</span> : {e(user_origin)} · '
            '<span class="tools">outils</span> : schémas envoyés à chaque tour, hors des messages</div>'
            f'<pre class="llm"><span class="sys">{body}</span>'
            f'<span class="tools"><b>OUTILS</b>{e(tools)}</span></pre>')


def _delegation(ex: dict | None, sub) -> str:
    """Délégation : ce que reçoit le sous-agent (son prompt système, la consigne de l'orchestrateur,
    son seul message utilisateur, et ses outils) puis le rapport qu'il renvoie."""
    real = ex and not ex.get("missing") and ex.get("call")
    out = _llm_input(sub, ex["call"].get("task", "") if real else None,
                     "consigne rédigée par l’orchestrateur à chaque délégation")
    if not real:
        return out + _example(ex, "")
    resp = ex["response"] if isinstance(ex["response"], str) else json.dumps(ex["response"], ensure_ascii=False, indent=2)
    return out + f'<div class="exlabel">Réponse</div><pre>{e(resp)}</pre>'


def _params(tool) -> str:
    """Les options de l'appel : nom, type, requis, valeurs permises et description (schéma de l'outil)."""
    props = tool.input_schema.get("properties", {})
    if not props:
        return '<div class="exlabel">Options</div><p class="noparam">aucune</p>'
    req = set(tool.input_schema.get("required", []))
    rows = []
    for name, spec in props.items():
        kind = spec.get("type", "")
        if kind == "array" and "items" in spec:
            kind += f" de {spec['items'].get('type', '')}"
        enum = spec.get("enum") or spec.get("items", {}).get("enum")
        values = (" · " + " | ".join(map(str, enum)) if enum else
                  f" · {spec.get('minimum', '')}…{spec.get('maximum', '')}" if "minimum" in spec or "maximum" in spec else "")
        rows.append(f'<li><code>{e(name)}</code>' + ('<span class="req">requis</span>' if name in req else "")
                    + f' <span class="ptype">{e(kind)}{e(values)}</span>'
                    + (f'<div class="pdesc">{e(spec["description"])}</div>' if spec.get("description") else "") + "</li>")
    return f'<div class="exlabel">Options</div><ul class="params">{"".join(rows)}</ul>'


def _tool(tool, kind: str, examples: dict, note: str, child: str = "", agent: str = "",
          sub=None) -> str:
    badge = "délègue" if kind == "delegate" else "outil"
    ex = examples.get(f"{agent}.{tool.name}") or examples.get(tool.name)
    body = _params(tool) + (_delegation(ex, sub) if kind == "delegate" else _example(ex, note))
    return (f'<li class="node {kind}"><details><summary>'
            f'<span class="badge {kind}">{badge}</span><code class="tname">{e(tool.name)}</code></summary>'
            f'<div class="body">{body}</div></details>{child}</li>')


def _agent(agent, examples: dict, note: str, subagents: dict, open_: bool = False) -> str:
    items = []
    for t in agent.tools:
        sub = t.name.removeprefix("ask_")
        if sub in subagents:
            child = '<ul class="tree">' + _agent(subagents[sub], examples, note, {}, open_=True) + "</ul>"
            items.append(_tool(t, "delegate", examples, note, child, agent.name, subagents[sub]))
        else:
            items.append(_tool(t, "tool", examples, note, agent=agent.name))
    meta = f"{agent.model} · {agent.max_turns} tours max · {len(agent.tools)} outils"
    # Racine (orchestrateur) : ses prompts ici, ceux des sous-agents sont dans leur délégation.
    prompts = "" if subagents == {} else f'<div class="body">{_root_prompts(agent, examples)}</div>'
    return (f'<li class="node agent"><details{" open" if open_ else ""}><summary><span class="badge agent">agent</span>'
            f'<span class="aname">{e(agent.name)}</span><span class="meta">{e(meta)}</span></summary>'
            f'{prompts}<ul class="tree">{"".join(items)}</ul></details></li>')


def _root_prompts(agent, examples: dict) -> str:
    """Ce que reçoit l'orchestrateur, replié : système, objectif du run, outils."""
    goal = examples.get(f"{agent.name}.goal")
    return ('<details class="prompts"><summary>Prompts</summary>'
            + _llm_input(agent, goal["response"] if goal else None,
                         "objectif du run, option --goal de python -m agents (sinon sa valeur par défaut)")
            + "</details>")


def _agents() -> list:
    """L'orchestrateur puis ses trois sous-agents, construits comme pour un run."""
    lab, session = Lab(), _Session()
    return [orchestrator.make(lab, session), orchestrator.dataset_builder.make(lab),
            orchestrator.model_researcher.make(lab, session), orchestrator.evaluator.make(lab, session)]


def render(examples: dict, note: str) -> str:
    root, *sub = _agents()
    subs = {a.name: a for a in sub}
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agents et outils</title><style>{CSS}</style></head>
<body><main>
<h1>Agents et outils</h1>
<p class="sub">Exemples d'appel et de réponse de chaque outil, et prompts des agents, générés par <code>interface/agents_tree/agents_tree.py</code>.</p>
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


def _empty(result) -> bool:
    """Réponse sans contenu, comme {"splits": []} avant le premier split : un exemple inutile."""
    return isinstance(result, dict) and bool(result) and all(v in ([], {}, None) for v in result.values())


def examples_from_run(run_dir: Path) -> dict:
    """Premier appel réussi de chaque outil, par agent, dans le journal d'un run
    (runs/<NNN>_…/pipeline.json) : appel et réponse réels, tels que l'agent les a reçus ; une
    réponse vide n'est gardée que faute d'un appel suivant qui renvoie quelque chose, ou d'une
    réponse du même outil reçue par un autre agent (list_experiments, appelé par l'orchestrateur
    avant le premier split). Clés « agent.outil » (un même outil, comme dataset_info, ne renvoie
    pas la même chose à chaque agent)."""
    log = json.loads((run_dir / "pipeline.json").read_text(encoding="utf-8"))
    out = {}
    for s in log["steps"]:
        if s["type"] == "start" and s["parent"] is None:  # objectif donné à l'agent racine
            out.setdefault(f"{s['agent']}.goal", {"call": None, "response": s["task"]})
            continue
        key = f"{s['agent']}.{s.get('tool')}"
        if s["type"] != "tool" or (key in out and not _empty(out[key]["response"])) or s["error"]:
            continue
        if key in out and _empty(s["result"]):
            continue  # déjà un exemple vide : on attend une réponse avec du contenu
        result = s["result"]
        if isinstance(result, dict) and "error" in result:
            continue  # refus d'un outil (budget épuisé…) : on attend un appel abouti
        call = json.loads(s["input"]) if isinstance(s["input"], str) else s["input"]
        out[key] = {"call": call or None,
                          "response": "\n".join(result) if isinstance(result, list) else result,
                          "note": f"Appel et réponse réels (run {run_dir.name.split('_')[0]} du "
                                  f"{s['time'][8:10]}/{s['time'][5:7]}/{s['time'][:4]}, agent {s['agent']})."}
    for key, ex in out.items():
        if not _empty(ex["response"]):
            continue
        agent, tool = key.split(".", 1)
        other = next((o for k, o in out.items() if k.split(".", 1)[1] == tool and not _empty(o["response"])), None)
        if other:
            out[key] = {**other, "note": other["note"].rstrip(".") + f" ; {agent} n'a reçu de cet outil "
                                                                     "qu'une réponse vide."}
    return out


# Outils sans effet (ils lisent l'état du labo) : leur exemple est recalculé à chaque génération,
# avec le même appel, pour que la page suive le format de leurs réponses sans attendre un run.
# Les autres (collecte, split, test final, délégations) gardent leur réponse réelle, sauf ceux de
# REREAD.
READ_ONLY = {"global_status", "list_experiments", "dataset_info", "data_requested_history",
             "analyze_errors", "show_examples"}
# Outils à effet dont la réponse se relit dans ce qu'ils ont écrit : recalculée sans les rejouer.
REREAD = {"train_model": lambda lab, ex: lab.train_summary(ex["response"]["experiment_id"]),
          "final_test": lambda lab, ex: lab.final_test_summary(**ex["call"])}


def refresh(examples: dict, summary_usd: float = 0.0) -> list[str]:
    """Recalcule sur l'état actuel la réponse des exemples d'outils READ_ONLY. analyze_errors :
    son analyse visuelle est payante (résumé par le modèle des agents) ; refaite seulement si
    summary_usd > 0, sinon celle de l'exemple est gardée. Renvoie les exemples recalculés."""
    from agents.common import analyze_errors_tool

    day = time.strftime("%d/%m/%Y")
    lab, session, done = Lab(), None, []
    if summary_usd > 0:
        import tempfile

        from agents.runner import Session
        # journal hors de runs/ : ce n'est pas un run
        session = Session(summary_usd, run_dir=Path(tempfile.mkdtemp(prefix="agents_tree_")),
                          report_reserve_usd=0.0, verbose=False)
    for agent in _agents():
        for t in agent.tools:
            key = f"{agent.name}.{t.name}"
            ex = examples.get(key)
            if (t.name not in READ_ONLY and t.name not in REREAD) or not ex or ex.get("missing"):
                continue
            call = ex.get("call") or {}
            if t.name in REREAD:
                response = REREAD[t.name](lab, ex)
            elif t.name == "analyze_errors":
                response = analyze_errors_tool(lab, session).fn(**call)
                old = ex["response"] if isinstance(ex["response"], dict) else {}
                if session is None and "analyse_visuelle" in old:
                    response = {"experiment_id": response.pop("experiment_id"),
                                "analyse_visuelle": old["analyse_visuelle"],
                                "cout_analyse_visuelle_usd": old.get("cout_analyse_visuelle_usd"), **response}
            else:
                response = t.fn(**call)
            if t.name == "global_status" and isinstance(ex["response"], dict):
                # budgets de session et de collecte : ceux d'un run, qu'un labo neuf n'a pas
                old = ex["response"]
                response["budget_session_usd"] = old.get("budget_session_usd")
                if "budgets" in old.get("dataset_total_disponible", {}):
                    response["dataset_total_disponible"]["budgets"] = old["dataset_total_disponible"]["budgets"]
            if isinstance(response, list):  # blocs texte et image (show_examples)
                response = "\n".join(b["text"] if b["type"] == "text" else "<image>" for b in response)
            origin = ex.get("origin") or ex.get("note", "").rstrip(".")
            examples[key] = {"call": ex.get("call"), "response": response, "origin": origin,
                             "note": f"Réponse recalculée le {day} sur l'état actuel"
                                     + (" (analyse visuelle de l'exemple d'origine)"
                                        if t.name == "analyze_errors" and session is None else "")
                                     + f" ; appel d'origine : {origin[0].lower() + origin[1:]}."}
            done.append(key)
    return done


def main() -> None:
    p = argparse.ArgumentParser(prog="interface/agents_tree/agents_tree.py")
    p.add_argument("--from-run", type=Path, help="tire les exemples du journal de ce run (runs/<NNN>_…)")
    p.add_argument("--examples", type=Path, help="exemples d'appels et de réponses par outil (JSON)")
    p.add_argument("--examples-note", default="", help="d'où viennent les exemples")
    p.add_argument("--out", type=Path, default=HERE / "agents_tree.html")
    p.add_argument("--no-refresh", action="store_true",
                   help="ne recalcule pas les exemples des outils sans effet (READ_ONLY)")
    p.add_argument("--summary-usd", type=float, default=0.0,
                   help="plafond pour refaire l'analyse visuelle des exemples analyze_errors (payant ; 0 : gardée)")
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
    if not a.no_refresh:
        print("recalculés :", ", ".join(refresh(examples, a.summary_usd)))
    a.out.write_text(render(examples, a.examples_note), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
