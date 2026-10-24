"""A local model running its own optogenetics experiments on the fly brain.

The loop is deliberately small: the model picks one action per turn, the lab
executes it against the real connectome, and the observation goes back into the
transcript.  Four actions are enough to do science here -- look up neurons,
look up cell types, run a stimulation/silencing experiment, and conclude.

Every experiment leaves a full 3D report behind, so the model's reasoning can
be checked against what the brain actually did.
"""

from __future__ import annotations

import json
import re
import time
import webbrowser
from pathlib import Path

import numpy as np

from . import data as D, llm, report, sim

def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = [
    _tool(
        "search",
        "Сколько нейронов попадает под запрос-селектор, в каких областях мозга "
        "и каких типов. Делай это перед экспериментом, чтобы не стимулировать "
        "пустое множество.",
        {"query": {"type": "string",
                   "description": "селектор, например class:gustatory+side:left"}},
        ["query"],
    ),
    _tool(
        "list_types",
        "Поиск по именам типов клеток и классов (подстрока или регулярка). "
        "Так ищут нейроны по памяти о названиях: MBON, P9, DNa02, LC10.",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
    _tool(
        "experiment",
        "Прогнать симуляцию: активировать нейроны (оптогенетика) и, если нужно, "
        "заглушить другие. Возвращает, какие области загорелись и в каком "
        "порядке. Дорогая операция.",
        {
            "activate": {
                "type": "array",
                "description": "наборы стимулируемых нейронов, например "
                               "[{\"query\": \"class:gustatory+region:GNG\", "
                               "\"rate\": 200}]",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "строка-селектор целиком"},
                        "rate": {"type": "number",
                                 "description": "частота стимуляции, Гц"},
                    },
                    "required": ["query"],
                },
            },
            "silence": {"type": "string",
                        "description": "селектор заглушаемых нейронов, можно пусто"},
            "t": {"type": "number", "description": "длительность, с (0.3-1.0)"},
            "note": {"type": "string", "description": "зачем этот опыт"},
        },
        ["activate"],
    ),
    _tool(
        "conclude",
        "Итоговый ответ по задаче, по-русски, с конкретными числами и "
        "названиями из проведённых опытов.",
        {"answer": {"type": "string"}},
        ["answer"],
    ),
]

SYSTEM = """Ты ставишь опыты на симуляторе целого мозга дрозофилы (коннектом
FlyWire, 138 639 нейронов, модель leaky integrate-and-fire). Ты можешь
активировать любые нейроны (имитация оптогенетики) и заглушать другие, а потом
смотреть, какие области мозга загорелись и в каком порядке.

На каждом шаге вызывай РОВНО ОДИН инструмент: search, list_types, experiment
или conclude. Когда данных достаточно — вызови conclude.

Язык запросов (селекторы), через запятую — объединение, через + — пересечение:
  type:MBON        имя типа клетки содержит (регулярка)
  type=MBON01      точное имя типа
  class:gustatory  класс клетки (gustatory, olfactory, Kenyon_Cell, ALPN,
                   mechanosensory, visual_projection, MBON, DAN, ...)
  super:descending супер-класс (central, optic, sensory, descending,
                   ascending, motor, endocrine, visual_projection)
  region:AL        область мозга (AL, MB, LH, GNG, SAD, EB, FB, PB, NO, LAL,
                   ME, LO, LOP, LA, AMMC, WED, SMP, SLP, AVLP, PVLP, NERVE,
                   VNC, CRE, ICL, SCL, IB, ATL, VES, EPA, GOR, SPS, IPS, PS,
                   PLP, AOTU, BU, CAN, PRW, FLA, AME, GA). Субэзофагеальная
                   зона (SEZ) — это GNG; вкусовые нейроны и мотонейроны
                   хоботка там.
  superregion:optic группа областей: optic, mushroom body, central complex,
                   olfactory, superior, inferior, ventrolateral, ventromedial,
                   periesophageal, "gnathal (SEZ)", outside brain
  nt:gaba          нейромедиатор (acetylcholine, gaba, glutamate, dopamine,
                   serotonin, octopamine)
  side:left        сторона (left, right, center)
  id:7205759406...  конкретный root_id FlyWire

Примеры: "class:gustatory+side:left", "type:MBON,type:KC",
"region:EB+nt:gaba", "super:descending+type:DNa".

Имена типов клеток во FlyWire часто не совпадают с генетическими: сахарные
вкусовые нейроны называются LB1-LB4 и LgAG (лабеллярные щетинки), а не Gr64f.
Если запрос ничего не нашёл — ищи через list_types по классу.

Сокращения: MBON — выходной нейрон грибовидного тела, KC — клетка Кеньона,
DN — нисходящий нейрон (команда в грудной ганглий), MN — мотонейрон,
GRN — вкусовой рецепторный нейрон, PER — вытягивание хоботка.

Правила.
1. Один вызов инструмента на шаг.
2. Главное — опыты, а не поиск. Больше двух поисков подряд не делай: бюджет
   шагов маленький, а без experiment ты ничего не узнаешь.
3. Не повторяй запрос, который уже делал.
4. Никогда не описывай результат опыта, которого не было. В выводе ссылайся
   только на числа, которые реально вернул experiment.
5. Чтобы доказать, что путь идёт через звено X, поставь опыт со стимулом,
   потом такой же опыт с silence=X и сравни числа."""


SELECTOR_KEYS = ("type", "class", "super", "region", "superregion", "nt",
                 "side", "nerve", "lineage", "id", "idx")


def _coerce_query(spec):
    """Accept both ``{"query": "class:gustatory+region:GNG"}`` and the shape
    models reach for anyway, ``{"class": "gustatory", "region": "GNG"}``."""
    if isinstance(spec, str):
        return spec.strip()
    if not isinstance(spec, dict):
        return ""
    q = str(spec.get("query", "") or "").strip()
    if q:
        return q
    parts = [f"{k}:{spec[k]}" for k in SELECTOR_KEYS
             if spec.get(k) not in (None, "", [])]
    return "+".join(parts)


def _fmt_search(df, query):
    try:
        idx = D.select(df, query)
    except ValueError as exc:
        return {"error": str(exc)}
    if len(idx) == 0:
        hint = "ничего не найдено, попробуй list_types или другой селектор"
        if "region:" in query:
            hint += ". Существующие области: " + ", ".join(
                df["region"].value_counts().index.tolist()
            )
        return {"query": query, "n": 0, "hint": hint}
    sub = df.loc[idx]
    return {
        "query": query,
        "n": int(len(idx)),
        "regions": sub["region"].value_counts().head(6).to_dict(),
        "types": sub["label"].value_counts().head(10).to_dict(),
        "classes": sub["cell_class"].value_counts().head(5).to_dict(),
        "sides": sub["side"].value_counts().to_dict(),
    }


def _fmt_types(df, pattern):
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"плохая регулярка: {exc}"}
    hit = df[df["label"].str.contains(rx, na=False)
             | df["cell_class"].str.contains(rx, na=False)]
    if hit.empty:
        return {"pattern": pattern, "n": 0, "hint": "ничего не нашлось"}
    g = (
        hit.groupby("label")
        .agg(n=("idx", "size"), region=("region", "first"), cls=("cell_class", "first"))
        .sort_values("n", ascending=False)
        .head(25)
    )
    return {
        "pattern": pattern,
        "n_neurons": int(len(hit)),
        "types": [
            {"type": t, "n": int(r.n), "region": r.region, "class": r.cls}
            for t, r in g.iterrows()
        ],
    }


def _trim(summary):
    """Keep the transcript small enough for a local model's context."""
    return {
        "stimuli": [
            {k: v for k, v in s.items() if k in ("name", "n", "rate_hz", "regions")}
            for s in summary["stimuli"]
        ],
        "silenced": summary["silenced"],
        "duration_s": summary["duration_s"],
        "n_active_driven": summary["n_active_driven"],
        "cascade": summary["cascade"],
        "regions": [
            {
                "region": r["region"],
                "peak_mean_hz": r["peak_mean_hz"],
                "onset_ms": r["onset_ms"],
                "n_active": r["n_active"],
                "top_types": r["top_types"][:3],
            }
            for r in summary["regions"][:8]
        ],
        "top_types": summary["top_types"][:10],
        "output_neurons": summary["output_neurons"][:6],
    }


FINAL_SYSTEM = """Ты пишешь вывод по серии опытов на симуляторе мозга
дрозофилы. Тебе дают ПОЛНЫЙ список того, что реально посчитано: какие нейроны
стимулировали, кого заглушали, какие области загорелись, с какими задержками и
частотами.

Пиши только то, что следует из этих чисел. Если опыта с заглушением не было —
так и скажи, что причинность не проверена, и не выдумывай его результат.
По-русски, 150-250 слов, с конкретными числами и названиями. Структура:
что показали опыты, какой путь сигнала виден, чего не хватает для доказательства."""


def _final_answer(goal, experiments, model):
    """Write the conclusion from the measurements alone.

    The model's own transcript is deliberately left out: by this point it has
    been speculating for several turns, and a local model will happily report a
    control experiment it never ran.  Only the recorded runs go in.
    """
    facts = []
    for i, e in enumerate(experiments, 1):
        s = e["summary"]
        facts.append({
            "опыт": i,
            "зачем": e["note"],
            "стимул": [{"кого": st["name"], "сколько": st["n"],
                        "Гц": st["rate_hz"]} for st in s["stimuli"]],
            "заглушено": s["silenced"],
            "длительность_с": s["duration_s"],
            "активных_кроме_стимула": s["n_active_driven"],
            "каскад": s["cascade"],
            "области": [
                {"обл": r["region"], "Гц": r["peak_mean_hz"],
                 "приход_мс": r["onset_ms"], "активных": r["n_active"]}
                for r in s["regions"][:8]
            ],
            "выходные_нейроны": s["output_neurons"][:6],
        })
    msgs = [
        {"role": "system", "content": FINAL_SYSTEM},
        {"role": "user", "content":
            f"Задача была: {goal}\n\nВсё, что реально посчитано:\n```json\n"
            + json.dumps(facts, ensure_ascii=False, indent=1)
            + "\n```\n\nНапиши вывод."},
    ]
    return llm.chat(msgs, model=model, temperature=0.3).strip()


def _emit(on_event, kind, **payload):
    """Одна точка для событий: окну нужен ход диалога, а не вывод в консоль."""
    if on_event is not None:
        on_event(kind, payload)


def _cut_to_actual(res):
    """Привести прогон, оборванный ``should_stop``, к тому, что посчитано.

    ``sim.run`` при раннем выходе оставляет в ``config.t_run_s`` заказанную
    длительность, поэтому ``rates = counts / t_run_s`` занижены, а в хвосте
    ``bins`` висят нулевые бины — в отчёт попал бы мозг, который будто молчал.
    Возвращает True, если прогон правда оборвался.
    """
    steps_done = int(res.timing.get("steps") or 0)
    full = int(round(res.config.t_run_s * 1000.0 / sim.DT))
    if steps_done <= 0 or steps_done >= full:
        return False
    bin_steps = max(1, int(round(res.config.bin_ms / sim.DT)))
    res.bins = res.bins[:int(np.ceil(steps_done / bin_steps))]
    res.config.t_run_s = steps_done * sim.DT / 1000.0
    return True


def run_loop(goal, model=None, max_steps=6, t_run_s=0.5, open_reports=True,
             out_name=None, on_event=None, should_stop=None):
    """Локальная модель сама ставит опыты на коннектоме.

    ``on_event(kind, payload)`` зовётся по ходу диалога; ``kind`` — одно из
    ``"thought"``, ``"tool"``, ``"observation"``, ``"experiment"``,
    ``"answer"``, ``"error"``. У ``"experiment"`` в payload лежат имя папки
    опыта и путь к его отчёту, так что окно может открыть отчёт само.
    ``should_stop()`` проверяется между шагами: шаг — это запрос к модели плюс
    возможная симуляция, прервать его середину нельзя, но опыт внутри шага
    останавливается по тому же признаку.
    """
    import torch  # imported here so --help stays fast

    models = llm.available_models()
    model = model or (llm.DEFAULT_MODEL if llm.DEFAULT_MODEL in models else models[0])
    print(f"модель: {model}   (доступны: {', '.join(models)})")

    df = D.load_neurons()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("загружаю матрицу связей...")
    weights = D.load_weights(device=device)

    run_name = out_name or (time.strftime("%Y%m%d-%H%M%S") + "-agent")
    out_dir = D.OUT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Задача: {goal}\n\n"
                                    f"Бюджет: {max_steps} шагов. Начинай."},
    ]
    journal = [f"# Агент: {goal}\n", f"модель: `{model}`, шагов: {max_steps}\n"]
    experiments = []
    seen_queries = set()
    answer = None
    fails = 0
    stopped = False

    for step in range(1, max_steps + 1):
        if should_stop is not None and should_stop():
            # Между шагами: внутри шага сидит запрос к модели, который не
            # прерывается, поэтому проверяем до него, а не после.
            stopped = True
            print("\nостановлено")
            journal.append("\n*остановлено по запросу*\n")
            break
        print(f"\n{'='*70}\nшаг {step}/{max_steps}")
        # conclude is withheld until at least one experiment has actually run,
        # so the model cannot summarise experiments it never performed.
        tools = TOOLS if experiments else TOOLS[:-1]
        try:
            msg = llm.chat_tools(messages, tools, model=model)
        except llm.OllamaError as exc:
            # A malformed tool call makes Ollama itself return 500.  Hand the
            # error back to the model and carry on rather than losing the run.
            fails += 1
            print(f"  сбой вызова: {str(exc)[:200]}")
            journal.append(f"\n## Шаг {step} — сбой\n\n`{str(exc)[:400]}`\n")
            _emit(on_event, "error", step=step, message=str(exc))
            if fails >= 3:
                print("  три сбоя подряд, останавливаюсь")
                break
            messages.append({"role": "user", "content":
                             f"Твой вызов инструмента не разобрался: {exc}. "
                             "Повтори вызов с корректным JSON."})
            continue
        fails = 0
        calls = msg.get("tool_calls") or []
        thought = (msg.get("thinking") or msg.get("content") or "").strip()
        messages.append(msg)
        if thought:
            _emit(on_event, "thought", step=step, max_steps=max_steps,
                  text=thought)

        if not calls:
            # No tool call means the model is done talking; whatever it said is
            # the answer.
            answer = (msg.get("content") or "").strip()
            if answer:
                journal.append(f"\n### Вывод\n\n{answer}\n")
                print(f"\n{answer}")
                break
            messages.append({"role": "user", "content":
                             "Ты не вызвал ни одного инструмента. "
                             "Вызови search, list_types, experiment или conclude."})
            continue

        call = calls[0]["function"]
        kind = call["name"]
        act = call.get("arguments") or {}
        if isinstance(act, str):
            try:
                act = json.loads(act)
            except json.JSONDecodeError:
                act = {}
        print(f"  мысль: {thought[:300]}")
        print(f"  действие: {kind} {json.dumps(act, ensure_ascii=False)[:160]}")
        journal.append(f"\n## Шаг {step} — {kind}\n\n{thought}\n")
        _emit(on_event, "tool", step=step, name=kind, args=act)

        if kind == "conclude":
            if not experiments:
                obs = {"error": "нельзя подводить итог, не проведя ни одного "
                                "опыта. Вызови experiment."}
            else:
                draft = str(act.get("answer", "")).strip()
                journal.append(f"\n### Черновик вывода (из диалога)\n\n{draft}\n")
                break

        elif kind == "search":
            query = _coerce_query(act)
            if query in seen_queries:
                obs = {"error": f"запрос {query!r} ты уже делал, "
                                "посмотри историю или ставь опыт"}
            else:
                seen_queries.add(query)
                obs = _fmt_search(df, query)
            print(f"  -> {obs.get('n', obs.get('error'))} нейронов")

        elif kind == "list_types":
            obs = _fmt_types(df, str(act.get("pattern", "")))
            print(f"  -> {obs.get('n_neurons', obs.get('error'))} нейронов")

        elif kind == "experiment":
            stimuli = []
            bad = None
            activate = act.get("activate") or []
            if isinstance(activate, (str, dict)):
                activate = [activate]
            for spec in activate:
                q = _coerce_query(spec)
                if not q:
                    bad = ("в activate нужен ключ query со строкой-селектором, "
                           "например {\"query\": \"class:gustatory+region:GNG\"}")
                    break
                try:
                    idx = D.select(df, q)
                except ValueError as exc:
                    bad = str(exc)
                    break
                if len(idx) == 0:
                    bad = f"запрос {q!r} ничего не нашёл"
                    break
                stimuli.append(
                    sim.Stimulus(idx, float(spec.get("rate", 200.0)),
                                 t_on_ms=50.0, name=q)
                )
            sil_q = _coerce_query(act.get("silence", ""))
            silenced = np.array([], np.int64)
            if not bad and sil_q:
                try:
                    silenced = D.select(df, sil_q)
                except ValueError as exc:
                    bad = str(exc)
                else:
                    if len(silenced) == 0:
                        bad = (f"запрос заглушения {sil_q!r} ничего не нашёл — "
                               "проверь его через search")
            if bad or not stimuli:
                obs = {"error": bad or "не задано ни одного стимула"}
                print(f"  -> ошибка: {obs['error']}")
            else:
                t = float(np.clip(act.get("t", t_run_s) or t_run_s, 0.1, 2.0))
                cfg = sim.RunConfig(t_run_s=t, bin_ms=10.0, device=device)
                print(f"  -> симуляция {t} c, "
                      f"{sum(len(s.indices) for s in stimuli)} нейронов стимула, "
                      f"{len(silenced)} заглушено")
                res = sim.run(stimuli, silenced=silenced, config=cfg,
                              neurons=df, weights=weights, progress=True,
                              should_stop=should_stop)
                cut = _cut_to_actual(res)
                note = str(act.get("note", "")).strip()
                if cut:
                    # Прерванный опыт всё равно сохраняем, но так, чтобы ни
                    # модель, ни отчёт не приняли его за полную длительность.
                    print(f"  -> опыт прерван на {res.config.t_run_s:.2f} с")
                    note = f"{note}; прервано" if note else "прервано"
                summary = llm.summarize(res)
                exp_dir = out_dir / f"exp-{len(experiments)+1:02d}"
                sim.save(res, exp_dir)
                html = report.render(
                    res, exp_dir / "report.html",
                    title=f"[{len(experiments)+1}] {note or goal}",
                    home="../../index.html",
                )
                experiments.append({"dir": exp_dir, "html": html, "note": note,
                                    "summary": summary})
                journal.append(
                    f"\n*{note}*\n\n```json\n"
                    + json.dumps(_trim(summary), ensure_ascii=False, indent=1)
                    + f"\n```\n\n[отчёт]({html.relative_to(out_dir).as_posix()})\n"
                )
                obs = _trim(summary)
                print(f"  -> каскад: {' → '.join(summary['cascade'][:8])}")
                _emit(on_event, "experiment", step=step,
                      index=len(experiments), dir=exp_dir.name,
                      report=str(html), note=note, partial=cut,
                      cascade=summary["cascade"][:8],
                      duration_s=summary["duration_s"])
        else:
            obs = {"error": f"неизвестный инструмент {kind!r}"}

        if kind != "experiment":
            journal.append("```json\n" + json.dumps(obs, ensure_ascii=False, indent=1)
                           + "\n```\n")
        _emit(on_event, "observation", step=step, name=kind, obs=obs)
        messages.append({
            "role": "tool",
            "tool_name": kind,
            "content": json.dumps(obs, ensure_ascii=False),
        })
        left = max_steps - step
        no_silence = experiments and not any(
            e["summary"]["silenced"] for e in experiments)
        if not experiments:
            nudge = "Опытов пока ноль. Следующий вызов — experiment."
        elif left <= 1:
            nudge = "Вызови conclude."
        elif no_silence and len(experiments) >= 1:
            nudge = (f"Проведено опытов: {len(experiments)}, но ни в одном ты "
                     "не заглушал нейроны. Повтори тот же стимул с непустым "
                     "silence, иначе причинность не доказана.")
        else:
            nudge = f"Проведено опытов: {len(experiments)}. Следующий инструмент."
        messages.append({
            "role": "user",
            "content": f"Осталось шагов: {left}. {nudge}",
        })

    if experiments and not stopped:
        print("\nпишу вывод по фактическим измерениям...")
        try:
            answer = _final_answer(goal, experiments, model)
        except llm.OllamaError as exc:
            print(f"не удалось: {exc}")
            _emit(on_event, "error", step=None, message=str(exc))
            answer = None
        if answer:
            if not any(e["summary"]["silenced"] for e in experiments):
                answer += ("\n\n[проверка лаборатории: ни в одном опыте "
                           "заглушения не было, причинность не проверена]")
            journal.append(f"\n### Вывод\n\n{answer}\n")
            print(f"\n{answer}")
    elif stopped:
        # Вывод — это ещё один запрос к модели на десятки секунд; после «стоп»
        # его не делаем, но journal и отчёт по уже посчитанному пишем.
        print("\nостановлено, вывод не пишу")
    else:
        print("\nагент не провёл ни одного опыта")

    (out_dir / "journal.md").write_text("\n".join(journal), encoding="utf-8")

    final_html = None
    if experiments:
        # The report carries what was actually computed before it carries what
        # the model concluded, so a claim can always be checked against a run.
        lines = ["Что реально посчитано:"]
        for i, e in enumerate(experiments, 1):
            s = e["summary"]
            stim = ", ".join(f"{st['name']} ({st['n']} нейронов, {st['rate_hz']:g} Гц)"
                             for st in s["stimuli"])
            lines.append(
                f"  {i}. {e['note'] or '—'}\n"
                f"     стимул: {stim}; заглушено: {s['silenced']}; "
                f"{s['duration_s']} с\n"
                f"     активных (кроме стимула): {s['n_active_driven']}; "
                f"каскад: {' → '.join(s['cascade'][:9])}"
            )
        factual = "\n".join(lines)
        narrative = factual + ("\n\nВывод модели:\n" + answer if answer else "")

        last = experiments[-1]
        res = sim.load(last["dir"], neurons=df)
        final_html = report.render(
            res, out_dir / "report.html",
            narrative=narrative,
            title=f"Агент: {goal}",
        )
        print(f"\nжурнал: {out_dir / 'journal.md'}")
        print(f"итоговый отчёт: {final_html}")
        # Единственное место во всём агенте, где открывается браузер; после
        # «стоп» не открываем и его — окно просили закрыть, а не открыть ещё.
        if open_reports and not stopped:
            webbrowser.open(Path(final_html).resolve().as_uri())
    if answer:
        _emit(on_event, "answer", text=answer, dir=str(out_dir),
              report=str(final_html) if final_html else None,
              n_experiments=len(experiments))
    return {"answer": answer, "experiments": experiments, "dir": out_dir,
            "report": final_html}
