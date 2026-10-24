"""Стабилизация v1: сквозные сценарии на настоящих файлах.

Ничего не изображается объектами в памяти: прогоны считаются работником,
файлы ложатся на диск, целостность проверяется хешами, а каталог и
«Анализ» читают то же самое, что лежит в `out/`.

Проверяются десять сценариев из брифа:

  1  пуск -> конец -> сохранение -> просмотр в «Анализе»;
  2  пуск -> отмена -> частичный результат виден и помечен;
  3  отказ работника -> повторный запуск;
  4  повтор опыта из «Анализа» -> новая папка;
  5  переключение всех четырёх пространств во время счёта;
  6  пустой каталог результатов;
  7  повреждённый результат;
  8  построение отчёта не меняет ни одного файла прогона;
  9  закрытие приложения во время задачи и после неё;
 10  обновление индекса ничего не теряет и не портит.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\stable_v1.py
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401  прогоны — во временный каталог

OUT = Path(__file__).parent
fails = []
rep = {"сценарии": {}, "проверки": {}, "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def digest(path):
    """Хеши всех файлов папки: по ним видно любое изменение байта."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    for f in sorted(path.iterdir()):
        if f.is_file():
            out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def wait_run(app, limit=600.0, cancel_after=None, on_frame=None):
    t0 = time.perf_counter()
    worst = 0.0
    asked = False
    while time.perf_counter() - t0 < limit:
        a = time.perf_counter()
        app.frame()
        worst = max(worst, (time.perf_counter() - a) * 1000.0)
        if on_frame is not None:
            on_frame()
        if (cancel_after is not None and not asked
                and time.perf_counter() - t0 > cancel_after
                and app.st.run.phase == "run"):
            asked = True
            app.cancel_run()
        if not app.st.run.busy and app.st.last is not None:
            break
    return app.st.last, worst, time.perf_counter() - t0


def main():                                                   # noqa: C901
    from flybrain import data as D
    from flybrain import index, report, runs as RN, sim
    from flybrain import stats as ST
    from flybrain.desk import ws_analysis as A, ws_experiment as X
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen((1500, 880))
    ses = Session()
    app = App(sh, ses)
    app.st.ws.switch("experiment")
    t0 = time.perf_counter()
    while not app.st.worker.ready and time.perf_counter() - t0 < 180:
        app.frame()
    print("работник готов за %.1f с" % (time.perf_counter() - t0))
    xs = X._state(app)

    def spec_of(kind, **vals):
        X._reset_values(xs, kind)
        xs["values"].update(vals)
        app.frame()
        return X.build_spec(app)

    def catalogue():
        """Перечитать каталог так, как это делает окно."""
        ast = A._state(app)
        ast["runs"] = None
        ast["detail_for"] = ""
        A._ensure(app)
        return {r.name: r for r in (ast["runs"] or [])}

    # ================================================ 1. пуск -> «Анализ»
    print("сценарий 1: пуск, конец, сохранение, просмотр")
    app.st.last = None
    spec = spec_of("stim", t=0.3)
    app.start_run(spec, title="Стимуляция")
    last, worst, took = wait_run(app)
    folder = Path(last.out_dir) if last and last.out_dir else None
    by = catalogue()
    name = folder.name if folder else ""
    r1 = by.get(name)
    check("1 прогон сохранён и виден в каталоге",
          r1 is not None and r1.state == "done",
          "%s: %s" % (name, r1.state_word if r1 else "нет в каталоге"))
    check("1 окно не вставало", worst < 400.0, "худший кадр %.0f мс" % worst)
    rep["сценарии"]["1"] = {"папка": name, "по стене, с": round(took, 1),
                            "файлы": sorted(r1.files) if r1 else []}

    # ============================================ 8. отчёт ничего не портит
    print("сценарий 8: отчёт не меняет файлы прогона")
    before = digest(folder)
    html = report.render(sim.load(folder), folder / "report.html")
    after = digest(folder)
    changed = [k for k in before if before[k] != after.get(k)]
    added = [k for k in after if k not in before]
    check("8 ни один файл прогона не изменился", not changed,
          "изменились: %s" % changed if changed else
          "проверено %d файлов по sha256" % len(before))
    check("8 добавился только сам отчёт", added == ["report.html"], added)
    rep["сценарии"]["8"] = {"файлов сверено": len(before), "добавлено": added,
                            "отчёт": str(html.name)}

    # повтор рендера тоже ничего не трогает
    before2 = digest(folder)
    report.render(sim.load(folder), folder / "report.html",
                  title="другой заголовок")
    after2 = digest(folder)
    ch2 = [k for k in before2 if before2[k] != after2.get(k)
           and k != "report.html"]
    check("8 повторный отчёт с другим заголовком тоже ничего не трогает",
          not ch2, "изменились: %s" % ch2 if ch2 else "только report.html")
    check("8 заголовок читается из самого отчёта",
          report.title_of(folder) == "другой заголовок",
          report.title_of(folder))

    # ============================================= 2. отмена -> частичный
    print("сценарий 2: отмена и частичный результат")
    app.st.last = None
    app.start_run(spec_of("stim", t=15.0), title="Стимуляция")
    last, _w, _t = wait_run(app, cancel_after=2.0, limit=300.0)
    cut = Path(last.out_dir) if last and last.out_dir else None
    by = catalogue()
    r2 = by.get(cut.name) if cut else None
    check("2 отменённый прогон сохранён и помечен",
          r2 is not None and r2.state == "cancelled",
          "%s: %s" % (cut.name if cut else "—",
                      r2.state_why[:60] if r2 else "нет"))
    check("2 окно отличает отмену от завершения",
          last is not None and last.outcome == "cancelled",
          last.outcome if last else None)
    meta = json.loads((cut / "meta.json").read_text(encoding="utf-8"))
    check("2 длительность в файле — фактическая",
          float(meta["config"]["t_run_s"]) < 15.0,
          "t_run_s = %s при заказанных 15" % meta["config"]["t_run_s"])
    from flybrain import plots as PL

    d2 = PL.load(r2, "activity") if r2 else None
    check("2 частичный результат читается и рисуется",
          isinstance(d2, dict) and len(d2.get("x", [])) > 0,
          (d2 or {}).get("note", "нет"))
    rep["сценарии"]["2"] = {"папка": cut.name if cut else "",
                            "t_run_s": meta["config"]["t_run_s"]}

    # ===================================== 5. переключение пространств
    print("сценарий 5: переключение пространств во время счёта")
    seen = {"spaces": set()}
    order = ["brain", "body", "analysis", "experiment"]

    def switch():
        app.st.ws.switch(order[int(time.perf_counter() * 2) % len(order)])
        seen["spaces"].add(app.st.ws.current)

    app.st.last = None
    app.start_run(spec_of("stim", t=1.0), title="Стимуляция")
    last, worst, _t = wait_run(app, on_frame=switch, limit=300.0)
    app.st.ws.switch("experiment")
    check("5 все четыре пространства рисуются во время счёта",
          len(seen["spaces"]) == 4 and last is not None
          and last.phase == "done", "побывали в %s, прогон %s"
          % (sorted(seen["spaces"]), last.phase if last else None))
    check("5 переключение не роняет кадр", worst < 400.0,
          "худший кадр %.0f мс" % worst)
    rep["замеры"]["худший кадр при переключении, мс"] = round(worst, 1)

    # ============================================ 3. отказ и новый пуск
    print("сценарий 3: отказ работника и повторный запуск")
    app.st.last = None
    bad = spec_of("stim", t=0.3)
    bad["stim"] = {"activate": [{"query": "НЕТ-ТАКОЙ-ПОПУЛЯЦИИ",
                                 "rate_hz": 200.0}]}
    app.start_run(bad, title="Стимуляция")
    last, _w, _t = wait_run(app, limit=180.0)
    alive = ses.alive
    check("3 работник пережил отказ", alive, "alive=%s" % alive)
    check("3 окно увидело отказ",
          last is not None and last.phase == "fail",
          (last.message if last else "")[:70])
    app.st.last = None
    app.start_run(spec_of("stim", t=0.3), title="Стимуляция")
    last, _w, _t = wait_run(app, limit=180.0)
    check("3 после отказа прогон снова считается",
          last is not None and last.phase == "done" and bool(last.out_dir),
          last.out_dir if last else "")

    # ====================================== 4. повтор опыта из «Анализа»
    print("сценарий 4: повтор опыта создаёт новую папку")
    by = catalogue()
    src = by.get(Path(last.out_dir).name)
    ok_rep, why = RN.repeatable(src)
    check("4 повтор возможен", ok_rep, why or "спека читается")
    done, msg = X.apply_spec(app, src.spec)
    app.st.ws.switch("experiment")
    app.frame()
    new_spec = X.build_spec(app)
    check("4 папка повтора отличается от исходной",
          done and new_spec.get("out") != src.name,
          "%s -> %s" % (src.name, new_spec.get("out")))
    src_before = digest(src.path)
    app.st.last = None
    app.start_run(new_spec, title="Стимуляция")
    last, _w, _t = wait_run(app, limit=300.0)
    check("4 повтор лёг в свою папку",
          last is not None and Path(last.out_dir).name != src.name,
          last.out_dir if last else "")
    check("4 исходный результат не тронут",
          digest(src.path) == src_before,
          "сверено %d файлов" % len(src_before))

    # ======================= совпадение имён и запуск в одну секунду
    print("перезапись: одинаковые имена и одна секунда")
    names = []
    for i in (1, 2, 3):
        sp = spec_of("stim", t=0.2)
        sp["out"] = "probe-stable-same-name"
        app.st.last = None
        app.start_run(sp, title="Стимуляция")
        last, _w, _t = wait_run(app, limit=300.0)
        names.append(Path(last.out_dir).name if last and last.out_dir else "")
    check("три задачи с одним именем легли в три папки",
          len(set(names)) == 3, ", ".join(names))
    rep["сценарии"]["имена"] = names

    # ================================================= пять видов опыта
    print("пять видов: настоящие файлы каждого")
    kinds = {}
    for kind, vals, want in (
            ("arena", {"t": 0.4}, ["episode.json", "track.parquet"]),
            ("lateral", {"t": 0.6}, ["lateral.json"]),
            ("body", {"t": 0.2}, ["embodied.json", "track.parquet"]),
            ("verify", {"t": 0.1}, ["verify.json", "meta.json"])):
        app.st.last = None
        app.start_run(spec_of(kind, **vals), title=kind)
        last, _w, _t = wait_run(app, limit=900.0)
        got = sorted(x.name for x in Path(last.out_dir).iterdir()) \
            if last and last.out_dir else []
        miss = [w for w in want if w not in got]
        kinds[kind] = {"папка": last.out_dir if last else "", "файлы": got}
        check("вид «%s»: файлы на месте" % kind,
              last is not None and last.phase == "done" and not miss,
              ("нет: " + ", ".join(miss)) if miss else ", ".join(got))
    rep["виды"] = kinds

    # сверка: паспорт на диске и исход
    vdir = Path(kinds["verify"]["папка"])
    if (vdir / "verify.json").exists():
        v = json.loads((vdir / "verify.json").read_text(encoding="utf-8"))
        rep["сверка"] = v
        check("сверка записала исход и версию данных",
              v.get("outcome") == "ok"
              and (v.get("dataset") or {}).get("release")
              and v.get("jaccard") is not None,
              "исход %s, Jaccard %.3f, корреляция %.4f, активных %s/%s"
              % (v.get("outcome"), v.get("jaccard", 0.0), v.get("corr", 0.0),
                 v.get("n_active_mine"), v.get("n_active_upstream")))
        check("сверка записала ограничения", bool(v.get("limits")),
              "%d строк" % len(v.get("limits") or []))
        byv = catalogue()
        rv = byv.get(vdir.name)
        check("сверка узнаётся в каталоге по verify.json",
              rv is not None and rv.kind == "verify",
              rv.kind_title if rv else "нет")

    # отменённая сверка не считается пройденной
    app.st.last = None
    app.start_run(spec_of("verify", t=1.0), title="Сверка")
    last, _w, _t = wait_run(app, cancel_after=2.0, limit=600.0)
    vd = Path(last.out_dir) if last and last.out_dir else None
    if vd and (vd / "verify.json").exists():
        v = json.loads((vd / "verify.json").read_text(encoding="utf-8"))
        check("отменённая сверка помечена отменённой",
              v.get("outcome") == "cancelled" and v.get("compared") is False,
              "исход %s, сравнение %s" % (v.get("outcome"),
                                          v.get("compared")))
    else:
        check("отменённая сверка помечена отменённой", False,
              "verify.json не записан")

    # ========================================= серия: семена и парность
    print("серия: семена, парность, видео")
    # Три эпизода на условие — минимум, на котором критерий вообще
    # считается: на двух окно его честно не считает.
    # Управление должно действительно рулить, иначе перестановка антенн
    # ничего не меняет и оба условия дают одни и те же числа: пары есть,
    # а разности нулевые, и критерий по ним не определён.
    sp = spec_of("arena", t=0.3)
    sp["repeat"] = 3
    sp["repeat_t"] = 0.3
    sp["arena"] = {"probes": False}
    app.st.last = None
    app.start_run(sp, title="Серия")
    last, _w, _t = wait_run(app, limit=1200.0)
    sdir = Path(last.out_dir) if last and last.out_dir else None
    bench = json.loads((sdir / "bench.json").read_text(encoding="utf-8")) \
        if sdir and (sdir / "bench.json").exists() else {}
    rep["серия"] = bench
    seeds = {k: v.get("seeds") for k, v in bench.items()}
    check("серия записала семена эпизодов",
          all(bool(v) for v in seeds.values()) and len(seeds) == 2, seeds)
    info = ST.series_stats(bench)
    check("парность доказана семенами", info["pairing"] == "proven",
          info["pairing_why"][:70])
    check("при доказанной парности критерий парный",
          info.get("test") is not None
          and "парный" in info["test"]["name"]
          and "непарный" not in info["test"]["name"],
          (info.get("test") or {}).get("name", "критерий не считался: "
                                               + info.get("why", "")))
    check("непарный критерий остался рядом как справочный",
          info.get("unpaired") is not None
          and info["unpaired"] is not info["test"],
          info["unpaired"]["name"] if info.get("unpaired") else "нет")
    check("разности пар не нулевые: условия различимы",
          bench["норма"]["runs"] != bench["перестановка антенн"]["runs"],
          "норма %s против контроля %s"
          % (bench["норма"]["runs"],
             bench["перестановка антенн"]["runs"]))
    check("серия: эпизоды пронумерованы",
          all(len(v.get("episodes") or []) == v.get("n_runs")
              for v in bench.values()),
          {k: len(v.get("episodes") or []) for k, v in bench.items()})
    merged, notes = ST.pool([("a", bench), ("b", bench)])
    check("повторное сложение одной серии замечено по семенам",
          any("уже встречались" in n for n in notes),
          next((n[:80] for n in notes if "уже встречались" in n), "нет"))

    # ================================================ 7. повреждённый
    print("сценарий 7: повреждённый результат")
    broken = D.OUT / "probe-stable-broken"
    if broken.exists():
        shutil.rmtree(broken)
    shutil.copytree(folder, broken)
    (broken / "meta.json").write_text("{ это не json", encoding="utf-8")
    by = catalogue()
    rb = by.get(broken.name)
    check("7 повреждённый прогон остался в каталоге", rb is not None,
          rb.name if rb else "исчез")
    check("7 повреждённый не выдаётся за успешный",
          rb is not None and rb.state == "broken",
          rb.state_word if rb else "")
    check("7 названы файл и ошибка",
          rb is not None and "meta.json" in rb.errors
          and "JSON" in rb.errors["meta.json"],
          (rb.errors.get("meta.json") if rb else "")[:70])
    check("7 вид прогона не подменился",
          rb is not None and rb.kind == "stim", rb.kind if rb else "")
    ents = index.scan()
    eb = next((e for e in ents if e["name"] == broken.name), None)
    check("7 указатель тоже не прячет повреждённый",
          eb is not None and eb["kind"] == "broken",
          eb["kind"] if eb else "нет в указателе")
    # рисование повреждённого не роняет окно
    A._state(app)["cursor"] = broken.name
    app.st.ws.switch("analysis")
    worst = 0.0
    for view in range(5):
        A._state(app)["view"] = view
        for _ in range(2):
            a = time.perf_counter()
            app.frame()
            worst = max(worst, (time.perf_counter() - a) * 1000.0)
    check("7 «Анализ» рисует повреждённый прогон без падения", True,
          "худший кадр %.0f мс" % worst)
    sh.shot(OUT / "stable_broken.png")

    # ============================================ 6. пустой каталог
    print("сценарий 6: пустой каталог результатов")
    empty = OUT / "_empty_out"
    if empty.exists():
        shutil.rmtree(empty)
    empty.mkdir(parents=True, exist_ok=True)
    check("6 обзор пустого каталога — пустой список",
          RN.scan(empty) == [], "прогонов: %d" % len(RN.scan(empty)))
    check("6 указатель по пустому каталогу строится",
          bool(index.render(empty, path=empty / "index.html").exists()),
          "index.html записан")
    missing = empty / "нет-такой-папки"
    check("6 несуществующий каталог не роняет обзор",
          RN.scan(missing) == [], "пусто")
    shutil.rmtree(empty)

    # ====================================== 10. обновление индекса
    print("сценарий 10: обновление индекса ничего не теряет")
    dirs = sum(1 for p in D.OUT.iterdir() if p.is_dir())
    before_all = {p.name: digest(p) for p in D.OUT.iterdir() if p.is_dir()}
    t1 = time.perf_counter()
    path = index.render()
    idx_s = time.perf_counter() - t1
    after_all = {p.name: digest(p) for p in D.OUT.iterdir() if p.is_dir()}
    touched = []
    for nm, before_f in before_all.items():
        for f, h in before_f.items():
            if after_all.get(nm, {}).get(f) != h:
                touched.append("%s/%s" % (nm, f))
    check("10 обновление индекса не изменило ни одного файла прогона",
          not touched, "изменились: %s" % touched[:4] if touched
          else "сверено %d папок" % len(before_all))
    ents = index.scan()
    check("10 в указателе все папки", len(ents) == dirs,
          "каталогов %d, записей %d" % (dirs, len(ents)))
    rep["замеры"]["index.render, с"] = round(idx_s, 2)
    rep["замеры"]["папок в out"] = dirs

    # ============================= 9. закрытие во время и после задачи
    print("сценарий 9: закрытие приложения")
    app.start_run(spec_of("stim", t=20.0), title="Стимуляция")
    t1 = time.perf_counter()
    while app.st.run.phase == "prep" and time.perf_counter() - t1 < 60:
        app.frame()
    ses.close()
    t1 = time.perf_counter()
    while ses.alive and time.perf_counter() - t1 < 30:
        time.sleep(0.2)
    check("9 работник закрылся во время задачи", not ses.alive,
          "за %.1f с" % (time.perf_counter() - t1))
    ses2 = Session()
    app2 = App(Offscreen((1280, 720)), ses2)
    t1 = time.perf_counter()
    while not app2.st.worker.ready and time.perf_counter() - t1 < 180:
        app2.frame()
    check("9 после закрытия поднимается новый работник",
          app2.st.worker.ready, "за %.1f с" % (time.perf_counter() - t1))
    ses2.close()
    t1 = time.perf_counter()
    while ses2.alive and time.perf_counter() - t1 < 30:
        time.sleep(0.2)
    check("9 закрытие в простое проходит чисто", not ses2.alive,
          "за %.1f с" % (time.perf_counter() - t1))

    (OUT / "stable_v1.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
