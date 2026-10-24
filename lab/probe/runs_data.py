"""Слой данных «Анализа»: каталог, графики, статистика, сопоставимость.

Окна здесь нет — проверяется то, на чём оно стоит. Числа сверяются с
самими файлами: статистика пересчитывается из сырых поэпизодных значений
и сравнивается с тем, что записал `loop.bench`, а не берётся у него на
веру.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\runs_data.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():
    from flybrain import compare as CP
    from flybrain import plots as PL
    from flybrain import runs as RN
    from flybrain import stats as ST

    t0 = time.perf_counter()
    rs = RN.scan()
    scan_ms = (time.perf_counter() - t0) * 1000.0
    by = {r.name: r for r in rs}
    rep["замеры"]["обзор каталога, мс"] = round(scan_ms, 1)
    rep["замеры"]["прогонов"] = len(rs)
    check("каталог читается быстро", scan_ms < 500.0,
          "%d прогонов за %.0f мс" % (len(rs), scan_ms))

    # --- ни одна папка не выброшена ---
    from flybrain import data as D

    dirs = sum(1 for p in Path(D.OUT).iterdir() if p.is_dir())
    check("ни одна папка не потеряна", len(rs) == dirs,
          "в out/ каталогов %d, в списке %d" % (dirs, len(rs)))

    kinds = {}
    states = {}
    for r in rs:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
        states[r.state] = states.get(r.state, 0) + 1
    rep["виды"], rep["состояния"] = kinds, states
    check("распознаются все виды, которые есть на диске",
          set(kinds) <= {"stim", "arena", "body", "lateral", "verify",
                         "bench", "agent", "unknown"}, kinds)
    check("успех не назначается по существованию папки",
          states.get("unknown", 0) > 0, states)

    # --- обрыв виден по числам, а не по догадке ---
    cut = [r for r in rs if r.state == "cancelled"]
    check("оборванный прогон узнан", bool(cut),
          "; ".join("%s: %s" % (r.name, r.state_why[:60]) for r in cut))
    for r in cut:
        if r.spec and r.spec.get("t") and r.t_actual is not None:
            check("у «%s» факт меньше заказа" % r.name,
                  float(r.t_actual) < float(r.spec["t"]),
                  "заказано %g с, в файле %g с" % (r.spec["t"], r.t_actual))

    # --- графики: ничего не падает, пустых нет ---
    worst = ("", 0.0)
    empty = []
    total = 0
    t0 = time.perf_counter()
    for r in rs:
        for p in PL.available(r):
            total += 1
            t1 = time.perf_counter()
            try:
                d = PL.load(r, p.key)
            except Exception as exc:                          # noqa: BLE001
                empty.append("%s/%s: %s" % (r.name, p.key, exc))
                continue
            ms = (time.perf_counter() - t1) * 1000.0
            if ms > worst[1]:
                worst = ("%s/%s" % (r.name, p.key), ms)
            if d is None or "plot" not in d:
                empty.append("%s/%s: пусто" % (r.name, p.key))
            if d and p.unit == "" and p.what == "":
                empty.append("%s/%s: без описания" % (r.name, p.key))
    rep["замеры"]["графиков всего"] = total
    rep["замеры"]["самый долгий график"] = "%s, %.0f мс" % worst
    check("все %d графиков строятся из файлов" % total, not empty,
          "; ".join(empty[:3]) if empty
          else "самый долгий: %s, %.0f мс" % worst)
    check("у каждого графика есть единицы и источник",
          all(p.source and (p.unit or p.what)
              for r in rs for p in PL.available(r)))
    check("чтение всех графиков уложилось в разумное",
          (time.perf_counter() - t0) < 60.0,
          "%.1f с на %d графиков" % (time.perf_counter() - t0, total))

    # --- временных рядов нет там, где нет оси времени ---
    bad = []
    for r in rs:
        for p in PL.available(r):
            if p.kind != "line":
                continue
            if p.key == "activity" and not r.has("bins.npz"):
                bad.append(r.name + "/activity")
            if p.key != "activity" and not r.has("track.parquet"):
                bad.append(r.name + "/" + p.key)
    check("временной ряд предлагается только при оси времени", not bad,
          "; ".join(bad[:4]) if bad else
          "ряды только из track.parquet и bins.npz")

    # --- статистика: пересчёт сходится с записанным ---
    off = []
    series = 0
    for r in rs:
        b = (RN.detail(r) or {}).get("bench.json")
        if not isinstance(b, dict) or not b:
            continue
        series += 1
        info = ST.series_stats(b)
        for c in info["conditions"]:
            w, got = c["written"], c["stats"]
            for key in ("sd", "sem"):
                a, g = w[key], got[key]
                if a is None or g is None:
                    continue
                if abs(float(a) - float(g)) > 1e-4:
                    off.append("%s/%s/%s: %s против %s"
                               % (r.name, c["name"], key, a, g))
    check("sd и sem пересчитаны из эпизодов и сошлись с файлом",
          not off, "; ".join(off[:3]) if off
          else "серий %d, расхождений нет" % series)

    # --- мало наблюдений: критерий не считается ---
    tiny = ST.series_stats({"a": {"runs": [0.1, 0.2], "n_runs": 2},
                            "b": {"runs": [0.3, 0.4], "n_runs": 2}})
    check("при двух эпизодах критерий не считается",
          not tiny["comparable"] and tiny["test"] is None, tiny["why"])
    one = ST.series_stats({"норма": {"runs": [0.1, 0.2, 0.3], "n_runs": 3}})
    check("одно условие — сравнивать не с чем",
          not one["comparable"], one["why"][:60])

    # --- объединение серий LAL воспроизводит известное число ---
    if "arena-bench-lal" in by and "arena-bench-lal2" in by:
        merged, notes = ST.pool(
            [(n, (RN.detail(by[n]) or {})["bench.json"])
             for n in ("arena-bench-lal", "arena-bench-lal2")])
        pooled = ST.series_stats(merged)
        p = pooled["test"]["p"]
        rep["LAL объединённые"] = {
            "n": [c["usable"] for c in pooled["conditions"]],
            "p": round(p, 4), "diff": round(pooled["test"]["diff"], 4)}
        check("объединённая серия LAL даёт p около 0,196",
              abs(p - 0.1964) < 0.002,
              "n %s, p = %.4f, разница %+.4f"
              % ([c["usable"] for c in pooled["conditions"]], p,
                 pooled["test"]["diff"]))
        check("объединение помечено как ручное", bool(notes), notes[-1][:70])
        check("значимость не объявляется",
              "не доказательство" in ST.verdict(pooled["test"]),
              ST.verdict(pooled["test"])[:60])

    # --- сопоставимость величин ---
    pair = [by.get("arena-lal"), by.get("20261024-180956-body")]
    pair = [x for x in pair if x is not None]
    if len(pair) == 2:
        t = CP.table(pair)
        nope = [m for m in t["measures"] if not m[3]]
        same = [m for m in t["measures"] if m[3]]
        check("разные имена одной величины сведены",
              any(m[0] == "Смещение к цели" or m[0].startswith("До цели, ")
                  for m in same), [m[0] for m in same][:4])
        check("разные величины под похожими именами помечены",
              any("визит" in m[4].lower() for m in nope),
              next((m[4][:60] for m in nope if "визит" in m[4].lower()), ""))
        check("про разные виды сказано прямо",
              any("виды прогонов разные" in n.lower() for n in t["notes"]),
              t["notes"][0][:60] if t["notes"] else "")
    check("процент при нулевом знаменателе не выводится",
          CP.percent(0.0, 5.0)[0] is None, CP.percent(0.0, 5.0)[1])
    check("процент считается там, где можно",
          abs((CP.percent(10.0, 12.0)[0] or 0) - 20.0) < 1e-9)

    # --- повторяемость ---
    ok_n = sum(1 for r in rs if RN.repeatable(r)[0])
    no_spec = [r for r in rs if r.spec is None]
    check("повтор возможен там, где есть спека", ok_n > 0,
          "%d прогонов из %d повторяются" % (ok_n, len(rs)))
    check("без спеки повтор честно отказан",
          all(not RN.repeatable(r)[0] and "spec.json" in RN.repeatable(r)[1]
              for r in no_spec[:5]),
          "прогонов без spec.json: %d" % len(no_spec))

    (OUT / "runs_data.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
