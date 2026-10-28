"""Точный выбор нейронов по столбцу: сверка по НОМЕРАМ, а не по числу.

Селектор проекта оказался ловушкой на двух уровнях, и оба найдены
замером, а не чтением.

**Первый уровень:** ``type:X`` значит «СОДЕРЖИТ X» регулярным
выражением, а не «начинается с X». ``type:T4`` даёт 6400 нейронов, среди
которых ни одного T4 — затягивает ``LHCENT4``, ``LPT42_Nod4``,
``AC neuron``; а ``type=T4`` даёт ноль, потому что настоящие типы —
``T4a``…``T4d``.

**Второй уровень, глубже:** ``type=X`` сравнивает не с одним столбцом, а
с тремя — ``cell_type``, ``hemibrain_type``, ``label`` — и берёт их
ОБЪЕДИНЕНИЕ. У нисходящих нейронов номенклатуры столкнулись, и
объединение молча склеивает две разные популяции.

Отсюда новые ключи ``celltype=``, ``hemibrain=``, ``label=`` — каждый по
одному столбцу. Легаси ``type`` не тронут: по нему считаны все
сохранённые прогоны.

**Почему проба сверяет номера, а не числа.** Число совпадает и у
правильной выборки, и у подменённой: ровно это и произошло бы, если
селектор начал брать другой столбец того же размера. Поэтому здесь
сверяются отпечатки отсортированных ``root_id``, и отдельно ищется
случай, где **число одинаково, а нейроны разные** — без такого случая
проверка на числах выглядела бы пройденной.

**Встречный замер у каждой проверки — независимый пересчёт.** Маски
считаются прямо по столбцам датафрейма, не через ``select``. Если
``select`` начнёт делать что-то другое, расхождение видно сразу; если
проверять ``select`` самим собой, не видно ничего.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\select_exact.py
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
rep = {"проверки": {}, "замеры": {}, "находки": [], "сорвалось": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def ids_fp(root_ids):
    """Отпечаток выборки по её НОМЕРАМ, а не по размеру."""
    a = np.sort(np.asarray(root_ids, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def main():
    t0 = time.perf_counter()
    print("Точный выбор нейронов: сверка по номерам", flush=True)
    from flybrain import data as D

    n = D.load_neurons()
    root = n["root_id"].to_numpy(np.int64)
    cols = {c: n[c].astype(str).to_numpy() for c in
            ("cell_type", "hemibrain_type", "label")}
    rep["замеры"]["нейронов в таблице"] = int(len(n))

    def by_col(col, value):
        """Независимый пересчёт: прямо по столбцу, без `select`."""
        return np.sort(root[cols[col] == value])

    def by_union(value):
        m = np.zeros(len(n), bool)
        for c in cols:
            m |= cols[c] == value
        return np.sort(root[m])

    def sel_ids(query):
        return np.sort(root[D.select(n, query)])

    # ---------------------------------------- 1. легаси не изменилось
    # Проверяется не «работает», а «считает ровно то же, что и раньше»:
    # легаси `type=` обязан давать объединение трёх столбцов, и это
    # сверяется с независимым пересчётом по НОМЕРАМ.
    print("\nЛегаси `type=`: объединение трёх столбцов", flush=True)
    legacy = {}
    bad = []
    names = ("DNp01", "DNa01", "DNa02", "DNp09", "LC4", "LPLC2", "R1-6",
             "R7", "R8", "L1", "L2", "L3", "Mi1", "Tm1", "Tm9", "T4a",
             "T5a", "MBON01", "Gr64f")
    for q in names:
        got = sel_ids("type=%s" % q)
        want = by_union(q)
        same = (len(got) == len(want)
                and ids_fp(got) == ids_fp(want))
        legacy[q] = {"нейронов": int(len(got)),
                     "отпечаток номеров": ids_fp(got),
                     "совпало с независимым пересчётом": bool(same)}
        if not same:
            bad.append(q)
    rep["замеры"]["легаси type="] = legacy
    print("  сверено %d имён, расхождений %d" % (len(names), len(bad)),
          flush=True)
    check("легаси `type=` считает ровно объединение трёх столбцов — "
          "сверено по номерам нейронов, а не по их числу",
          not bad,
          "на %d именах отпечатки отсортированных root_id совпали с "
          "независимым пересчётом прямо по столбцам. Это и есть гарантия "
          "совместимости: по `type` считаны все сохранённые прогоны"
          % len(names) if not bad else "разошлись: " + ", ".join(bad))

    # ---------------------------------- 2. новые ключи берут один столбец
    print("\nНовые ключи: по одному столбцу", flush=True)
    keys = {"celltype": "cell_type", "ct": "cell_type",
            "cell_type": "cell_type",
            "hemibrain": "hemibrain_type", "hb": "hemibrain_type",
            "hemibrain_type": "hemibrain_type", "label": "label"}
    single, bad2 = {}, []
    for key, col in keys.items():
        for q in ("DNp01", "DNa01", "DNp09", "LC4", "R7"):
            got = sel_ids("%s=%s" % (key, q))
            want = by_col(col, q)
            same = (len(got) == len(want) and ids_fp(got) == ids_fp(want))
            single["%s=%s" % (key, q)] = {
                "нейронов": int(len(got)),
                "столбец": col,
                "отпечаток номеров": ids_fp(got),
                "совпало со столбцом": bool(same)}
            if not same:
                bad2.append("%s=%s" % (key, q))
    rep["замеры"]["новые ключи"] = single
    check("каждый новый ключ берёт ровно свой столбец — сверено по "
          "номерам",
          not bad2,
          "проверено %d пар «ключ — имя» по семи псевдонимам трёх "
          "столбцов; все отпечатки совпали с прямым пересчётом по "
          "столбцу" % len(single) if not bad2
          else "разошлись: " + ", ".join(bad2))

    # ------------------- 3. главное: число совпадает, нейроны разные
    print("\nПоиск случая, где число совпадает, а нейроны разные",
          flush=True)
    # Перебираются все имена, встречающиеся в cell_type или
    # hemibrain_type. Ищется то, на чём проверка по числу прошла бы, а по
    # номерам — нет. Без такого случая требование «сверять номера» было
    # бы умозрительным.
    seen = set(cols["cell_type"]) | set(cols["hemibrain_type"]) \
        | set(cols["label"])
    seen.discard("nan")
    # Первая редакция искала расхождение между ОБЪЕДИНЕНИЕМ и столбцом
    # `cell_type` — и не нашла ничего, правильно не нашла: когда
    # объединение отличается от столбца, отличается и размер. Настоящий
    # случай живёт между ДВУМЯ СТОЛБЦАМИ: одно имя, по два нейрона в
    # каждом, и ни одного общего.
    pairs = (("cell_type", "hemibrain_type"), ("cell_type", "label"),
             ("hemibrain_type", "label"))
    traps = []
    for q in sorted(seen):
        if not q or q == "nan":
            continue
        for c1, c2 in pairs:
            a, b = by_col(c1, q), by_col(c2, q)
            if (len(a) and len(a) == len(b)
                    and ids_fp(a) != ids_fp(b)):
                traps.append({
                    "имя": q, "столбец 1": c1, "столбец 2": c2,
                    "нейронов в каждом": int(len(a)),
                    "общих": int(len(np.intersect1d(a, b))),
                    "отпечаток 1": ids_fp(a), "отпечаток 2": ids_fp(b)})
    traps.sort(key=lambda t: (t["общих"], -t["нейронов в каждом"]))
    rep["замеры"]["число совпало, нейроны разные"] = traps[:40]
    rep["замеры"]["таких случаев всего"] = len(traps)
    for t in traps[:5]:
        print("  %-14s %s против %s: по %d нейрона, общих %d — "
              "отпечатки разные"
              % (t["имя"], t["столбец 1"], t["столбец 2"],
                 t["нейронов в каждом"], t["общих"]), flush=True)
    worst = traps[0] if traps else None
    check("найден случай, где проверка по числу прошла бы, а по номерам "
          "нет — значит сверять номера необходимо, а не желательно",
          bool(traps) and worst["общих"] == 0,
          "таких случаев %d. Крайний — %s: по %d нейрона и в столбце "
          "`%s`, и в столбце `%s`, а общих из них **%d**. То есть "
          "выборки совпадают по размеру и не пересекаются вовсе. "
          "Проверка на числах прошла бы и пропустила подмену целиком; "
          "именно поэтому вся проба сверяет отпечатки номеров"
          % (len(traps), worst["имя"], worst["нейронов в каждом"],
             worst["столбец 1"], worst["столбец 2"], worst["общих"])
          if traps else "ни одного такого случая не нашлось")

    # ------------------------------ 4. аудит нисходящих на столкновения
    print("\nАудит нисходящих на столкновения столбцов", flush=True)
    desc = D.select(n, "super:descending")
    dm = np.zeros(len(n), bool)
    dm[desc] = True
    dnames = sorted(
        set(cols["cell_type"][dm]) | set(cols["hemibrain_type"][dm]))
    coll = []
    for q in dnames:
        if not q or q == "nan":
            continue
        c = by_col("cell_type", q)
        h = by_col("hemibrain_type", q)
        u = by_union(q)
        if len(c) and len(h) and ids_fp(c) != ids_fp(h):
            coll.append({
                "имя": q,
                "по cell_type": int(len(c)),
                "по hemibrain_type": int(len(h)),
                "объединение": int(len(u)),
                "общих": int(len(np.intersect1d(c, h))),
                "столбцы указывают на разные нейроны": True,
            })
    rep["замеры"]["столкновения у нисходящих"] = coll
    print("  имён среди нисходящих %d, столкновений %d"
          % (len(dnames), len(coll)), flush=True)
    for c in coll[:12]:
        print("    %-12s cell_type %d, hemibrain_type %d, объединение "
              "%d, общих %d" % (c["имя"], c["по cell_type"],
                                c["по hemibrain_type"], c["объединение"],
                                c["общих"]), flush=True)
    check("нисходящие проверены на столкновения столбцов, и список "
          "столкновений получен числом",
          True,
          "среди %d имён, встречающихся у нисходящих, столкнулись %d: "
          "%s. Для них `type=` склеивает разные популяции, и научную "
          "выборку надо делать ключом по столбцу"
          % (len(dnames), len(coll),
             ", ".join(c["имя"] for c in coll[:8]) or "ни одного"))

    if coll:
        note("у нисходящих нейронов %d имён, где `cell_type` и "
             "`hemibrain_type` указывают на РАЗНЫЕ нейроны: %s. Легаси "
             "`type=` берёт объединение и склеивает их в одну выборку. "
             "Моторный съём живёт именно здесь, поэтому для него "
             "обязателен ключ по столбцу. Легаси при этом не тронут: по "
             "нему считаны все сохранённые прогоны, и проверка выше "
             "сверяет его по номерам нейронов."
             % (len(coll), ", ".join(c["имя"] for c in coll[:8])))

    # ----------------------------------- 5. пересечение с частями запроса
    # Ключи должны работать и в составном запросе, иначе польза нулевая.
    print("\nСоставные запросы", flush=True)
    comp = {}
    okc = True
    for q, col, val, side in (("celltype=DNp01+side:left", "cell_type",
                               "DNp01", "left"),
                              ("celltype=LC4+side:right", "cell_type",
                               "LC4", "right"),
                              ("hemibrain=DNa01+side:left",
                               "hemibrain_type", "DNa01", "left")):
        got = sel_ids(q)
        sides = n["side"].astype(str).to_numpy()
        m = (cols[col] == val) & (sides == side)
        want = np.sort(root[m])
        same = len(got) == len(want) and ids_fp(got) == ids_fp(want)
        comp[q] = {"нейронов": int(len(got)), "совпало": bool(same)}
        okc &= same
    rep["замеры"]["составные запросы"] = comp
    check("новые ключи работают в составном запросе через `+`", okc,
          "; ".join("%s -> %d нейронов, %s" % (k, v["нейронов"],
                                               "совпало" if v["совпало"]
                                               else "РАЗОШЛОСЬ")
                    for k, v in comp.items()))

    # ------------------------------------------- 6. кэш не перепутал
    # Кэш слагаемых ключуется строкой запроса. Если ключ кэша неполон,
    # разные ключи с одним именем вернут одно и то же — это и проверяется.
    a = sel_ids("type=DNa01")
    b = sel_ids("celltype=DNa01")
    c = sel_ids("hemibrain=DNa01")
    a2 = sel_ids("type=DNa01")
    rep["замеры"]["кэш"] = {
        "type=DNa01": int(len(a)), "celltype=DNa01": int(len(b)),
        "hemibrain=DNa01": int(len(c)),
        "повтор type= дал то же": bool(ids_fp(a) == ids_fp(a2)),
    }
    check("кэш слагаемых не путает ключи с одинаковым именем",
          ids_fp(a) != ids_fp(b) and ids_fp(b) != ids_fp(c)
          and ids_fp(a) == ids_fp(a2),
          "`type=DNa01` даёт %d нейронов, `celltype=DNa01` — %d, "
          "`hemibrain=DNa01` — %d, и все три отпечатка разные; повтор "
          "`type=DNa01` дал тот же отпечаток. Будь ключ кэша неполон, "
          "три запроса вернули бы одно и то же"
          % (len(a), len(b), len(c)))

    rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
    (OUT / "select_exact.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d, находок %d, за %.0f с"
          % (len(rep["проверки"]), len(fails), len(rep["находки"]),
             time.perf_counter() - t0), flush=True)
    if fails:
        print("ПРОВАЛЫ:", flush=True)
        for f in fails:
            print("  " + f, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
