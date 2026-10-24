"""Выборка нейронов: та же, что раньше, но быстрее. Независимая сверка.

Правка `data.select` (этап 1) ускорила выборку в десятки раз, и запись
SPEED.md называла числа, которых не было ни в одном файле замеров: сама
проба лежала во временном каталоге и пропала. Это непорядок — в проекте
число либо воспроизводится, либо не называется.

Поэтому здесь не просто замер, а шлюз. В пробе лежит **отдельная,
независимая реализация** того предиката, который был в `data.py` до
правки: построчное сканирование регулярным выражением по всей таблице.
Она написана заново по тексту прежнего кода и ничего не берёт из
`flybrain.data`, кроме самой таблицы. Дальше:

* каждый из двух с лишним сотен селекторов считается обеими дорогами, и
  результаты сравниваются побитово — по числу, по содержимому и по
  порядку индексов;
* обе дороги меряются по времени, и ускорение получается числом, а не
  воспоминанием.

Что это ловит, чего не ловит сравнение с прошлым снимком: ошибку,
одинаково сделанную и в коде, и в снимке. Независимая реализация
предиката такой ошибки не повторит.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\sel_dump.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "расхождения": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


# ------------------------------------------------- как считалось раньше

def old_term_mask(df, term):
    """Маска одного слагаемого — построчным сканированием.

    Это прежний `data._term_mask`, перенесённый сюда дословно по смыслу:
    на каждое слагаемое `df[col].astype(str)` по всей таблице и
    `.str.contains(..., regex=True)` или `.str.fullmatch(...)`. Ни кэша,
    ни различных значений — именно то, что стоило 19 секунд на двух
    сотнях селекторов.
    """
    term = term.strip().strip("\"'").strip()
    if ":" in term:
        key, value = term.split(":", 1)
        exact = False
    elif "=" in term:
        key, value = term.split("=", 1)
        exact = True
    else:
        key, value, exact = "type", term, False
    key = key.strip().lower()
    value = value.strip()

    def contains(col):
        return df[col].astype(str).str.contains(value, case=False,
                                                regex=True, na=False)

    def equals(col):
        return df[col].astype(str).str.fullmatch(value, case=False, na=False)

    hit = equals if exact else contains

    if key in ("type", "cell_type", "t"):
        return hit("cell_type") | hit("hemibrain_type") | hit("label")
    if key in ("class", "cell_class", "c"):
        return hit("cell_class") | hit("cell_sub_class")
    if key in ("super", "super_class", "s"):
        return hit("super_class")
    if key in ("region", "r"):
        return df["region"].astype(str).str.fullmatch(value, case=False,
                                                      na=False)
    if key in ("superregion", "super_region"):
        return hit("super_region")
    if key in ("nt", "transmitter"):
        return hit("top_nt")
    if key == "side":
        return hit("side")
    if key == "nerve":
        return hit("nerve")
    if key in ("lineage", "hemilineage"):
        return hit("ito_lee_hemilineage")
    if key in ("id", "root_id"):
        return df["root_id"].astype(np.int64) == np.int64(value)
    if key == "idx":
        return df["idx"].to_numpy() == int(value)
    if key == "all":
        return np.ones(len(df), dtype=bool)
    raise ValueError("неизвестный ключ селектора %r в слагаемом %r"
                     % (key, term))


def old_select(df, query):
    """Прежний `data.select`: множество питоновских целых и `sorted`."""
    out = set()
    for part in str(query).split(","):
        part = part.strip().strip("\"'").strip()
        if not part:
            continue
        mask = np.ones(len(df), dtype=bool)
        for t in part.split("+"):
            mask &= np.asarray(old_term_mask(df, t.strip()))
        out.update(df.loc[mask, "idx"].tolist())
    return np.array(sorted(out), dtype=np.int64)


# ------------------------------------------------------------- селекторы

def queries(neurons, world):
    qs = []
    for region in sorted(neurons["region"].unique()):
        qs += ["region:%s+side:left" % region,
               "region:%s+side:right" % region,
               "region:%s" % region]
    for sel in world.PROBE_FAMILIES.values():
        qs += [sel, sel + "+side:left", sel + "+side:right"]
    qs += list(world.SENSORS.values())
    qs += list(world.MOTOR.values())
    qs += ["class:gustatory+side:left", "type:Gr64f", "type=MBON01",
           "super:descending", "nt:gaba", "side:center", "all", "idx:12345",
           "region:AL,region:LH", "  'region:AL+side:left'  ", "nerve:AN",
           "lineage:ALl1", "superregion:optic", "type:LC10+side:right"]
    # Дубли убираем: иначе «селекторов 247» против «различных 246» —
    # ровно то несовпадение, которое уже попало в запись.
    seen, out = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def fp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    from flybrain import data as D, world

    neurons = D.load_neurons()
    qs = queries(neurons, world)
    rep["замеры"]["строк таблицы"] = int(len(neurons))
    rep["замеры"]["различных селекторов"] = len(qs)
    print("таблица %d строк, различных селекторов %d" % (len(neurons),
                                                         len(qs)))

    # --- как сейчас ---
    t0 = time.perf_counter()
    now = {q: D.select(neurons, q) for q in qs}
    t_now = time.perf_counter() - t0

    # --- как было ---
    t0 = time.perf_counter()
    was = {}
    for q in qs:
        try:
            was[q] = old_select(neurons, q)
        except Exception as exc:                          # noqa: BLE001
            was[q] = exc
    t_was = time.perf_counter() - t0

    rep["замеры"].update({
        "старой дорогой, с": round(t_was, 3),
        "нынешней дорогой, с": round(t_now, 3),
        "ускорение": round(t_was / max(t_now, 1e-9), 1),
    })
    print("старой дорогой %.2f с, нынешней %.2f с, в %.1f раза"
          % (t_was, t_now, t_was / max(t_now, 1e-9)))

    bad = []
    for q in qs:
        a, b = was[q], now[q]
        if isinstance(a, Exception):
            bad.append((q, "старая дорога отказала: %s" % a, ""))
            continue
        if a.dtype != b.dtype or len(a) != len(b) or fp(a) != fp(b):
            bad.append((q, "старая: %d, %s" % (len(a), fp(a)),
                        "нынешняя: %d, %s" % (len(b), fp(b))))
    rep["расхождения"] = [{"селектор": q, "было": x, "стало": y}
                          for q, x, y in bad]
    check("ни один селектор не разошёлся с прежней реализацией", not bad,
          "сверено %d, расхождений %d" % (len(qs), len(bad)))
    for q, x, y in bad[:8]:
        print("     %s | %s | %s" % (q, x, y))

    check("нынешняя дорога быстрее прежней", t_now < t_was,
          "%.2f с против %.2f с" % (t_now, t_was))
    # Порог не из вкуса: различных значений в `side` пять на 138 639
    # строк, в `region` тридцать шесть; выигрыш обязан быть не разовым
    # процентом, а разами.
    check("ускорение измеряется разами, а не процентами",
          t_was / max(t_now, 1e-9) > 5.0,
          "в %.1f раза" % (t_was / max(t_now, 1e-9)))

    # Список популяций — то, что окно ждёт на старте.
    world._CANDIDATE_CACHE.clear()
    t0 = time.perf_counter()
    cands = world.probe_candidates(neurons)
    t_pop = time.perf_counter() - t0
    rep["замеры"]["список популяций, с"] = round(t_pop, 3)
    rep["замеры"]["популяций"] = len(cands)
    check("список популяций строится быстрее секунды", t_pop < 1.0,
          "%d популяций за %.3f с" % (len(cands), t_pop))

    rep["отпечатки"] = {q: {"n": int(len(now[q])), "sha": fp(now[q])}
                        for q in qs}
    (OUT / "sel_dump.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
