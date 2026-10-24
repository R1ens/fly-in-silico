"""Local models (Ollama) reading the simulation.

A whole-brain run produces 139k firing rates, which is far too much to hand a
language model.  :func:`summarize` boils a run down to the few hundred tokens
that actually carry the story: which regions lit up, in what order, how fast,
which cell types led each one, and what the stimulus was.  That summary is what
the explainer and the agent reason over.

Everything here talks to a local Ollama server; nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from . import regions as R

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("FLYBRAIN_MODEL", "gpt-oss:latest")


class OllamaError(RuntimeError):
    pass


def available_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"Ollama не отвечает на {OLLAMA}: {exc}") from exc


NUM_CTX = int(os.environ.get("FLYBRAIN_NUM_CTX", "16384"))


def chat(messages, model=None, schema=None, temperature=0.3, timeout=900,
         num_ctx=None):
    """One round trip to Ollama's /api/chat.  Returns the assistant text."""
    model = model or DEFAULT_MODEL
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx or NUM_CTX},
    }
    if schema is not None:
        body["format"] = schema
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise OllamaError(f"Ollama вернул {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(f"Ollama не отвечает на {OLLAMA}: {exc}") from exc
    return data["message"]["content"]


def chat_tools(messages, tools, model=None, temperature=0.2, timeout=900,
               num_ctx=None):
    """Round trip with Ollama's native tool calling.

    Returns the whole assistant message, so the caller can read both
    ``tool_calls`` and the reasoning that Ollama reports in ``thinking``.
    Reasoning models such as gpt-oss routinely return an empty ``content``
    together with a tool call, which is why this does not go through
    :func:`chat`.
    """
    model = model or DEFAULT_MODEL
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx or NUM_CTX},
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["message"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise OllamaError(f"Ollama вернул {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(f"Ollama не отвечает на {OLLAMA}: {exc}") from exc


def chat_json(messages, model=None, schema=None, temperature=0.2, retries=2):
    """Same, but insists on getting parseable JSON back."""
    last = None
    for attempt in range(retries + 1):
        text = chat(messages, model=model, schema=schema,
                    temperature=temperature + 0.1 * attempt)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            last = (exc, text)
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Ответ не разобрался как JSON. "
                                            "Пришли только JSON-объект, без текста вокруг."},
            ]
    raise OllamaError(f"модель не вернула JSON: {last[0]}\n{last[1][:400]}")


# --------------------------------------------------------------- summarising


def _onsets(timeline, bin_ms, frac=0.2):
    """First bin at which each row reaches ``frac`` of its own peak, in ms."""
    out = {}
    values = timeline.to_numpy()
    for i, name in enumerate(timeline.index):
        row = values[i]
        peak = row.max()
        if peak <= 0:
            continue
        hit = np.nonzero(row >= peak * frac)[0]
        if len(hit):
            out[name] = float(hit[0] * bin_ms)
    return out


def summarize(result, top_regions=12, top_types=18, min_rate=1.0):
    """Compact, model-readable facts about a run."""
    neurons = result.neurons
    rt = result.rate_table(min_rate=min_rate)
    tl = result.region_timeline()
    onset = _onsets(tl, result.config.bin_ms)

    stim_idx = sorted({int(i) for s in result.stimuli for i in s.indices})
    stim_rows = neurons.loc[stim_idx] if stim_idx else neurons.iloc[:0]
    stim_desc = []
    for s in result.stimuli:
        rows = neurons.loc[list(s.indices)]
        stim_desc.append(
            {
                "name": s.name,
                "n": int(len(s.indices)),
                "rate_hz": s.rate_hz,
                "window_ms": [s.t_on_ms, s.t_off_ms],
                "top_types": rows["label"].value_counts().head(6).to_dict(),
                "regions": rows["region"].value_counts().head(4).to_dict(),
                "classes": rows["cell_class"].value_counts().head(4).to_dict(),
            }
        )

    driven = rt.drop(index=[i for i in stim_idx if i in rt.index], errors="ignore")

    regions = []
    peaks = tl.max(axis=1).sort_values(ascending=False)
    for name in peaks.index[:top_regions]:
        if peaks[name] <= 0:
            continue
        sub = driven[driven["region"] == name]
        regions.append(
            {
                "region": name,
                "name_ru": R.NEUROPILS.get(name, ("", ""))[0],
                "peak_mean_hz": round(float(peaks[name]), 2),
                "onset_ms": onset.get(name),
                "n_active": int(len(sub)),
                "n_total": int((neurons["region"] == name).sum()),
                "top_types": sub["label"].value_counts().head(5).index.tolist(),
            }
        )

    types = (
        driven.groupby("label")
        .agg(n=("rate_hz", "size"), mean_hz=("rate_hz", "mean"),
             region=("region", "first"), cls=("cell_class", "first"),
             nt=("top_nt", "first"))
        .sort_values("mean_hz", ascending=False)
        .head(top_types)
    )
    type_rows = [
        {
            "type": t,
            "n": int(r.n),
            "mean_hz": round(float(r.mean_hz), 1),
            "region": r.region,
            "class": r.cls,
            "nt": r.nt,
        }
        for t, r in types.iterrows()
    ]

    out_classes = (
        driven[driven["super_class"].isin(["descending", "motor", "endocrine"])]
        .groupby("label")
        .agg(n=("rate_hz", "size"), mean_hz=("rate_hz", "mean"),
             sup=("super_class", "first"))
        .sort_values("mean_hz", ascending=False)
        .head(10)
    )

    return {
        "stimuli": stim_desc,
        "silenced": int(len(result.silenced)),
        "duration_s": result.config.t_run_s,
        "bin_ms": result.config.bin_ms,
        "n_active": int(len(rt)),
        "n_active_driven": int(len(driven)),
        "total_spikes": int(result.counts.sum()),
        "regions": regions,
        "cascade": [
            r["region"] for r in sorted(
                [r for r in regions if r["onset_ms"] is not None],
                key=lambda r: r["onset_ms"],
            )
        ],
        "top_types": type_rows,
        "output_neurons": [
            {"type": t, "n": int(r.n), "mean_hz": round(float(r.mean_hz), 1),
             "super_class": r.sup}
            for t, r in out_classes.iterrows()
        ],
    }


# ------------------------------------------------------------------ explainer

SYSTEM = """Ты нейробиолог, который разбирает результат симуляции целого мозга
дрозофилы. Модель построена на коннектоме FlyWire (139 тыс. нейронов): это
leaky integrate-and-fire с весами, равными числу синапсов, без нейромодуляции и
без обучения. Активация нейронов имитирует оптогенетику.

Тебе дают сводку прогона в JSON: что стимулировали, какие области мозга
загорелись, в каком порядке (onset_ms — задержка выхода на 20% своего пика),
какие типы клеток были самыми активными.

Сокращения областей — стандартная номенклатура Ito et al. 2014:
SEZ/GNG — субэзофагеальная зона и гнатальные ганглии (вкус, движения хоботка),
AL — антеннальная долька (запах), MB — грибовидное тело (память),
LH — латеральный рог (врождённое поведение), EB/FB/PB/NO — центральный комплекс
(навигация), ME/LO/LOP/LA — оптические доли (зрение), AMMC/WED — слух и
механорецепция, LAL — выход на моторные команды, NERVE — мотонейроны,
VNC — восходящие из грудного ганглия.

Имена нейронов: MN… — мотонейроны, DN… — нисходящие (в грудной ганглий),
AN… — восходящие, MBON — выходы грибовидного тела, KC — клетки Кеньона,
GRN — вкусовые рецепторные нейроны, ORN — обонятельные, LC — лобулярные
колончатые (зрение). Proboscis по-русски — хоботок; его вытягивание (PER) —
классический ответ на сахар.

Отвечай по-русски, коротко и по делу. Никакой воды и никаких общих слов о том,
как интересна нейронаука."""

EXPLAIN_PROMPT = """Разбери прогон. Нужно:

1. Одно предложение: что произошло в мозге.
2. Путь сигнала: цепочка областей с задержками в мс, со стрелками.
   Опирайся на onset_ms и на то, какие типы клеток вели каждую область.
3. Что это значит для поведения мухи: какое действие она, вероятно, совершает.
   Если среди активных есть descending или motor нейроны — скажи, куда они идут.
4. Что в результате выглядит подозрительно или чего модель заведомо не
   показывает (её ограничения).

Не более 250 слов. Без markdown-заголовков, просто абзацы и стрелки.

Сводка прогона:
```json
{summary}
```"""


def explain(result, model=None):
    summary = summarize(result)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXPLAIN_PROMPT.format(
            summary=json.dumps(summary, ensure_ascii=False, indent=1))},
    ]
    return chat(msgs, model=model, temperature=0.4).strip()


def explain_dir(run_dir, model=None):
    from . import sim

    return explain(sim.load(run_dir), model=model)
