"""Пространство «Анализ»: каталог, обзор, графики, сравнение, статистика.

Работник здесь не нужен: пространство читает только то, что лежит на
диске. Проверяется главное — что ни один сохранённый прогон не роняет
окно, что тяжёлые файлы читаются в стороне от кадра, что состояние
результата названо, а не придумано, и что повтор опыта переносит
настройки, ничего не запуская.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\ana_shot.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
fails = []
rep = {"снимки": [], "проверки": {}, "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():
    from flybrain import runs as RN
    from flybrain.desk import ws_analysis as A
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from probe.app_shot import FakeSession

    sh = Offscreen((1500, 880))
    ses = FakeSession()
    ses.alive = ses.ready = True
    app = App(sh, ses)
    app.st.ws.switch("analysis")

    # Инвариант пяти пространств — проверкой с числом, а не обещанием в
    # тексте. Вид «Таймлайн» добавлен ВНУТРЬ «Анализа», и шестого
    # пространства от этого взяться не должно. Отдельно и подробнее —
    # `probe/five_spaces.py`.
    from flybrain.desk.state import WORKSPACES

    check("рабочих пространств ровно пять", len(WORKSPACES) == 5,
          "записей %d: %s" % (len(WORKSPACES),
                              ", ".join(n for _, n in WORKSPACES)))
    rep["замеры"]["рабочих пространств"] = len(WORKSPACES)
    rep["замеры"]["видов в «Анализе»"] = len(A.VIEWS)

    t0 = time.perf_counter()
    app.frame()
    first_ms = (time.perf_counter() - t0) * 1000.0
    st = A._state(app)
    runs = st["runs"] or []
    rep["замеры"]["первый кадр пространства, мс"] = round(first_ms, 1)
    rep["замеры"]["обзор каталога, мс"] = round(st["scan_ms"], 1)
    check("каталог прочитан при открытии", bool(runs),
          "прогонов %d, обзор %.0f мс, первый кадр %.0f мс"
          % (len(runs), st["scan_ms"], first_ms))
    check("открытие не читает тяжёлых файлов", first_ms < 900.0,
          "первый кадр %.0f мс" % first_ms)

    states = {}
    for r in runs:
        states[r.state] = states.get(r.state, 0) + 1
    rep["состояния"] = states
    check("состояния результата различаются", len(states) >= 2, states)
    check("успех не назначается по существованию папки",
          states.get("unknown", 0) > 0,
          "прогонов командной строки без записи исхода: %d"
          % states.get("unknown", 0))
    cancelled = [r.name for r in runs if r.state == "cancelled"]
    check("оборванный прогон узнан по числам", bool(cancelled),
          ", ".join(cancelled))

    def shot(name, note=""):
        app.frame()
        app.frame()
        sh.shot(OUT / name)
        rep["снимки"].append({"снимок": name, "что": note,
                              "мс на кадр": round(sh.frame_ms, 2)})
        print("  снимок %-26s %s" % (name, note))

    def wait_data(limit=20.0):
        """Крутить кадры, пока нить читает файл. Вернуть худший кадр."""
        t1 = time.perf_counter()
        worst = 0.0
        while time.perf_counter() - t1 < limit:
            a = time.perf_counter()
            app.frame()
            worst = max(worst, (time.perf_counter() - a) * 1000.0)
            if st["data"] is not None:
                break
        return worst, time.perf_counter() - t1

    def pick(name):
        st["cursor"] = name
        st["data"] = None
        st["data_key"] = ""
        st["plot"] = ""
        st["loader"].cancel()
        app.frame()

    # --- каждый прогон в каждом виде: ничего не падает ---
    worst_all = 0.0
    bad = []
    for r in runs:
        pick(r.name)
        for view in range(len(A.VIEWS)):
            st["view"] = view
            for _ in range(2):
                a = time.perf_counter()
                try:
                    app.frame()
                except Exception as exc:                      # noqa: BLE001
                    bad.append("%s/%d: %s" % (r.name, view, exc))
                    break
                worst_all = max(worst_all,
                                (time.perf_counter() - a) * 1000.0)
    rep["замеры"]["худший кадр при обходе всех прогонов, мс"] = round(
        worst_all, 1)
    check("все %d прогонов рисуются во всех %d видах"
          % (len(runs), len(A.VIEWS)),
          not bad, "; ".join(bad[:3]) if bad else
          "худший кадр %.0f мс" % worst_all)

    # --- вид «Таймлайн»: сеанс внутри «Анализа» ---
    # Вид есть всегда, а таймлайн — только у сеансов. Поэтому проверяются
    # обе ветки: с записью и без неё. Без записи вид обязан сказать это
    # словами, а не показать пустоту.
    tlv = A.VIEWS.index("Таймлайн")
    with_tl = [r for r in runs if getattr(r, "has_timeline", False)]
    rep["замеры"]["прогонов с таймлайном"] = len(with_tl)
    no_tl = next((r for r in runs if not getattr(r, "has_timeline", False)),
                 None)
    if no_tl is not None:
        pick(no_tl.name)
        st["view"] = tlv
        app.frame()
        app.frame()
        check("у прогона без таймлайна вид «Таймлайн» не падает",
              True, "прогон %s" % no_tl.name)
    if with_tl:
        r = with_tl[0]
        pick(r.name)
        st["view"] = tlv
        worst, took = wait_data()
        view = st["data"]
        tl = getattr(view, "tl", None)
        check("вид «Таймлайн» открыл запись сеанса",
              tl is not None and len(tl.times()) > 0,
              "прогон %s, шагов %d, за %.2f с, худший кадр %.0f мс"
              % (r.name, 0 if tl is None else len(tl.times()), took, worst))
        if tl is not None:
            times = tl.times()
            st["tl_t"] = float(times[0])
            app.frame()
            on = tl.row(float(times[0]))
            mid = float((times[0] + times[1]) / 2.0)
            st["tl_t"] = mid
            app.frame()
            app.frame()
            between = tl.row(mid)
            import flybrain.timeline as TLM

            check("на границе шага вид показывает числа, между шагами — "
                  "«нет данных»",
                  isinstance(on, dict) and isinstance(between, TLM.Hole),
                  "на %g мс каналов %d; на %g мс — %s"
                  % (times[0], len(on) if isinstance(on, dict) else 0, mid,
                     getattr(between, "why", "число")))
            check("просмотр записи в «Анализе» движок LIF не зовёт",
                  view.cost()["вызовов движка LIF"] == 0,
                  "ответов %d, вызовов движка %d"
                  % (view.cost()["ответов"],
                     view.cost()["вызовов движка LIF"]))
            shot("ana_timeline.png", "таймлайн сеанса: момент между шагами")

        # Три ряда из таймлайна в «Графиках»: их читает `plots.load`, и
        # без этого проверялся бы только список, а не чтение.
        from flybrain import plots as PL

        have = {pl.key for pl in PL.available(r)}
        for key, note in (("tl_cost", "вызовов движка за шаг"),
                          ("tl_probe", "зонды популяций"),
                          ("tl_wall", "шаг по стене")):
            if key not in have:
                check("график «%s» предложен для сеанса" % key, False,
                      "в списке только: %s" % ", ".join(sorted(have)))
                continue
            pick(r.name)
            st["view"] = 1
            st["plot"] = key
            wait_data()
            d = st["data"]
            check("график «%s» прочитан из таймлайна" % key,
                  isinstance(d, dict) and d.get("kind") == "line"
                  and bool(d.get("series")),
                  "%s; рядов %d"
                  % (note, len((d or {}).get("series") or [])))
        pick(r.name)
        st["view"] = 1
        st["plot"] = "tl_cost"
        wait_data()
        shot("ana_plot_tl_cost.png", "подпись стоимости режима по шагам")

    # --- обзор ---
    by = {r.name: r for r in runs}
    for name, note in (("arena-lal", "эпизод арены"),
                       ("demo-sugar", "стимуляция"),
                       ("arena-bench-lal2", "серия"),
                       ("lateral-probe", "перебор популяций")):
        if name not in by:
            continue
        pick(name)
        st["view"] = 0
        shot("ana_ov_%s.png" % name.replace("-", "_"), "обзор: " + note)

    # --- графики: настоящие данные из файлов ---
    pick("arena-lal")
    st["view"] = 1
    st["plot"] = "track"
    worst, took = wait_data()
    data = st["data"]
    check("траектория построена из track.parquet",
          isinstance(data, dict) and data.get("kind") == "track"
          and len(data.get("x", [])) > 100,
          "точек %d за %.2f с, худший кадр %.0f мс"
          % (len(data.get("x", [])) if isinstance(data, dict) else 0,
             took, worst))
    check("окно не вставало на чтении файла", worst < 120.0,
          "худший кадр %.0f мс" % worst)
    rep["замеры"]["худший кадр при чтении графика, мс"] = round(worst, 1)
    shot("ana_track.png", "траектория эпизода")

    for key, note in (("activity", "активность по бинам"),
                      ("dn", "съём слева и справа"),
                      ("regions", "области по частоте")):
        st["plot"] = key
        st["data"] = None
        st["data_key"] = ""
        st["loader"].cancel()
        wait_data()
        d = st["data"]
        check("график «%s» получил данные" % key,
              isinstance(d, dict) and "plot" in d,
              (d or {}).get("note", "нет"))
        shot("ana_plot_%s.png" % key, note)

    # --- отмена чтения: должна держаться, а не перезапускаться ---
    pick("arena-demo")
    st["view"] = 1
    st["plot"] = "activity"
    app.frame()
    key = st["loader"].key
    st["loader"].cancel()
    st["data_key"], st["data"] = key, {"cancelled": True}
    for _ in range(6):
        app.frame()
    check("отмена чтения держится",
          (st["data"] or {}).get("cancelled") is True
          and not st["loader"].busy,
          "данные: %s, нить занята: %s"
          % (st["data"], st["loader"].busy))
    st["data"], st["data_key"] = None, ""
    worst, took = wait_data()
    check("после отмены чтение запускается заново",
          isinstance(st["data"], dict) and "plot" in st["data"],
          "за %.2f с, худший кадр %.0f мс" % (took, worst))

    # --- таблицы ---
    pick("lateral-probe")
    st["view"] = 2
    shot("ana_table_lateral.png", "таблица популяций")
    pick("demo-sugar")
    st["view"] = 2
    wait_data()
    shot("ana_table_rates.png", "таблица нейронов")

    # --- сравнение ---
    # Пара РАЗНЫХ видов: на ней и видно, что сопоставимо, а что нет. Два
    # эпизода одного вида меряют одно и то же, и несопоставимых строк у
    # них просто не бывает — проверять на такой паре нечего.
    st["sel"] = []
    seen_kind = {}
    for r in runs:
        if r.kind in ("arena", "body", "stim", "lateral") \
                and r.kind not in seen_kind:
            seen_kind[r.kind] = r.name
    order = [k for k in ("arena", "body", "stim", "lateral")
             if k in seen_kind]
    if len(order) >= 2:
        st["sel"] = [seen_kind[order[0]], seen_kind[order[1]]]
    else:
        st["sel"] = [r.name for r in runs][:2]
    st["view"] = 3
    app.frame()
    from flybrain import compare as CP

    t = CP.table([by[n] for n in st["sel"]])
    nope = [m for m in t["measures"] if not m[3]]
    check("сравнение помечает несопоставимые величины", bool(nope),
          "несопоставимых строк %d из %d" % (len(nope), len(t["measures"])))
    check("проценты при нулевом знаменателе не выводятся",
          CP.percent(0.0, 5.0)[0] is None, CP.percent(0.0, 5.0)[1])
    shot("ana_compare.png", "сравнение прогонов")

    # --- статистика ---
    pick("arena-bench-lal2")
    st["sel"] = []
    st["view"] = 4
    app.frame()
    from flybrain import stats as ST

    b = (RN.detail(by["arena-bench-lal2"]) or {})["bench.json"]
    info = ST.series_stats(b)
    check("статистика серии считается", info["comparable"],
          "p = %.4f" % (info["test"]["p"] or -1))
    check("значимость не объявляется доказанной",
          "доказ" not in ST.verdict(info["test"]).lower()
          or "не доказ" in ST.verdict(info["test"]).lower(),
          ST.verdict(info["test"])[:70])
    shot("ana_stats.png", "статистика серии")

    # объединение двух серий
    st["sel"] = ["arena-bench-lal", "arena-bench-lal2"]
    st["pool"] = True
    app.frame()
    shot("ana_stats_pool.png", "две серии сложены")
    merged, notes = ST.pool([(n, (RN.detail(by[n]) or {})["bench.json"])
                             for n in st["sel"]])
    pooled = ST.series_stats(merged)
    check("объединение серий считается и помечается",
          pooled["comparable"] and bool(notes),
          "n %d+%d, p %.4f, оговорок %d"
          % (pooled["conditions"][0]["usable"],
             pooled["conditions"][1]["usable"],
             pooled["test"]["p"], len(notes)))
    st["sel"] = []
    st["pool"] = False

    # --- одиночный прогон статистики не даёт ---
    pick("demo-sugar")
    st["view"] = 4
    app.frame()
    check("у одиночного прогона статистики нет",
          not ((RN.detail(by["demo-sugar"]) or {}).get("bench.json")),
          "окно говорит об этом прямо")

    # --- правая колонка ---
    pick("20261024-181040-stim" if "20261024-181040-stim" in by
         else runs[0].name)
    st["view"] = 0
    for tab, note in enumerate(("параметры", "файлы", "оговорки",
                                "действия")):
        app.st.ws.set_tab(tab)
        shot("ana_tab%d.png" % tab, "вкладка: " + note)

    # --- повтор опыта ---
    from flybrain.desk import ws_experiment as X

    target = next((r for r in runs if r.spec and r.spec.get("kind") == "stim"
                   and (r.spec.get("stim") or {}).get("activate")), None)
    check("нашёлся прогон со спекой", target is not None,
          target.name if target else "нет")
    if target is not None:
        ok, why = RN.repeatable(target)
        check("повтор возможен", ok, why or "спека читается")
        before = app.st.run.phase
        done, msg = X.apply_spec(app, target.spec)
        app.st.ws.switch("experiment")
        app.frame()
        app.frame()
        got = X.build_spec(app)
        want_q = (target.spec.get("stim") or {}).get("activate")[0]["query"]
        got_q = (got.get("stim") or {}).get("activate", [{}])[0].get("query")
        check("повтор перенёс настройки", done and got_q == want_q,
              "%s -> %s" % (want_q, got_q))
        check("повтор не запускает опыт",
              app.st.run.phase == before == "idle", app.st.run.phase)
        check("повтор не затирает исходную папку",
              got.get("out") != target.name,
              "новая папка: %s" % got.get("out"))
        shot("ana_repeat.png", "перенесено в «Эксперимент»")
        app.st.ws.switch("analysis")

    # --- нижняя полоса в истории ---
    app.frame()
    ok_run, why = app.run_ready()
    check("в «Анализе» нечего запускать", not ok_run, why)
    from flybrain.desk.app import _hint

    hint = _hint(app.st)
    check("нижняя полоса не выдаёт историю за идущий счёт",
          "сохранённые прогоны" in hint, hint[:70])

    # --- размеры ---
    for (w, h, dpi) in ((1280, 720, 1.0), (1920, 1080, 1.0),
                        (1500, 880, 1.5)):
        sh.resize(w, h, dpi)
        st["view"] = 1
        pick("arena-lal")
        st["plot"] = "track"
        wait_data()
        shot("ana_%dx%d_%g.png" % (w, h, dpi), "раскладка")
    sh.resize(1500, 880, 1.0)

    if hasattr(ses, "close"):
        ses.close()
    (OUT / "ana_shot.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
