"""Целостность файлов: чтение и разбор не меняют ни байта.

Работник здесь не нужен. Проверяется то, что легко испортить незаметно:
просмотр сохранённого прогона в «Анализе», построение отчёта и
перестроение указателя не должны трогать измерения.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\integrity.py
"""

import hashlib
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


def snapshot(root):
    """sha256 каждого файла каждой папки прогона, кроме служебных."""
    out = {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file():
                out[str(f.relative_to(root))] = hashlib.sha256(
                    f.read_bytes()).hexdigest()
    return out


def diff(before, after):
    changed = [k for k in before if k in after and before[k] != after[k]]
    gone = [k for k in before if k not in after]
    new = [k for k in after if k not in before]
    return changed, gone, new


def main():
    from flybrain import data as D
    from flybrain import index, plots as PL, report, runs as RN, sim
    from flybrain.desk import ws_analysis as A
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from probe.app_shot import FakeSession

    base = snapshot(D.OUT)
    rep["замеры"]["файлов под наблюдением"] = len(base)
    print("под наблюдением %d файлов" % len(base))

    # --- 1. обзор каталога и чтение всех графиков ---
    t0 = time.perf_counter()
    rs = RN.scan()
    n_plots = 0
    for r in rs:
        RN.detail(r)
        for p in PL.available(r):
            PL.load(r, p.key)
            n_plots += 1
    read_s = time.perf_counter() - t0
    ch, gone, new = diff(base, snapshot(D.OUT))
    check("чтение каталога и всех графиков ничего не меняет",
          not ch and not gone and not new,
          "изменилось %s, пропало %s, появилось %s" % (ch[:3], gone[:3],
                                                       new[:3])
          if (ch or gone or new) else
          "%d прогонов, %d графиков за %.1f с" % (len(rs), n_plots, read_s))
    rep["замеры"]["графиков прочитано"] = n_plots

    # --- 2. полный обход «Анализа» в окне ---
    sh = Offscreen((1500, 880))
    ses = FakeSession()
    ses.alive = ses.ready = True
    app = App(sh, ses)
    app.st.ws.switch("analysis")
    app.frame()
    st = A._state(app)
    worst = 0.0
    before = snapshot(D.OUT)
    for r in (st["runs"] or []):
        st["cursor"] = r.name
        st["data"], st["data_key"], st["plot"] = None, "", ""
        st["loader"].cancel()
        for view in range(len(A.VIEWS)):
            st["view"] = view
            for tab in range(4):
                app.st.ws.set_tab(tab)
                a = time.perf_counter()
                app.frame()
                worst = max(worst, (time.perf_counter() - a) * 1000.0)
    ch, gone, new = diff(before, snapshot(D.OUT))
    check("просмотр всех прогонов в «Анализе» ничего не меняет",
          not ch and not gone and not new,
          "изменилось %s" % ch[:3] if ch else
          "%d прогонов × %d видов × 4 вкладки, худший кадр %.0f мс"
          % (len(st["runs"] or []), len(A.VIEWS), worst))
    rep["замеры"]["худший кадр обхода, мс"] = round(worst, 1)

    # --- 2a. пустой каталог в самом окне ---
    st["runs"] = []
    st["cursor"] = ""
    st["sel"] = []
    st["data"], st["data_key"] = None, ""
    ok_empty = True
    for view in range(len(A.VIEWS)):
        st["view"] = view
        for tab in range(4):
            app.st.ws.set_tab(tab)
            try:
                app.frame()
            except Exception as exc:                          # noqa: BLE001
                ok_empty = False
                print("    упало на виде %d вкладке %d: %s"
                      % (view, tab, exc))
    check("окно «Анализа» живёт при пустом каталоге", ok_empty,
          "%d видов × 4 вкладки без прогонов" % len(A.VIEWS))
    sh.shot(OUT / "integrity_empty.png")
    st["runs"] = None
    A._ensure(app)
    app.frame()

    # --- 3. построение отчёта ---
    # Отчёт строится НА КОПИИ во временном каталоге: сама проверка не
    # должна дописывать `report.html` в научный результат, даже если это
    # единственное, что она там меняет.
    target = next((r for r in rs if r.kind in ("stim", "verify")
                   and r.has("meta.json") and r.has("rates.parquet")), None)
    check("нашёлся прогон для отчёта", target is not None,
          target.name if target else "нет")
    if target is not None:
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="flybrain-report-"))
        copy = tmp / target.path.name
        shutil.copytree(target.path, copy)
        before = snapshot(D.OUT)
        meta_sha = hashlib.sha256(
            (copy / "meta.json").read_bytes()).hexdigest()
        report.render(sim.load(copy), copy / "report.html")
        after = snapshot(D.OUT)
        ch, gone, new = diff(before, after)
        check("построение отчёта не трогает научный каталог",
              not ch and not gone and not new,
              "изменилось: %s" % (ch + new) if (ch + new) else
              "отчёт собран на копии в %s" % tmp)
        check("meta.json после отчёта побайтно тот же",
              hashlib.sha256((copy / "meta.json").read_bytes()).hexdigest()
              == meta_sha, "sha256 совпал")
        check("отчёт действительно записан",
              (copy / "report.html").exists(),
              "%.1f МБ" % ((copy / "report.html").stat().st_size / 1e6))
        shutil.rmtree(tmp, ignore_errors=True)

    # --- 4. отчёт для прогона в теле ---
    body = next((r for r in rs if r.kind == "body"
                 and r.has("embodied.json")), None)
    if body is not None:
        import shutil
        import tempfile

        from flybrain import body as B

        tmp = Path(tempfile.mkdtemp(prefix="flybrain-body-"))
        copy = tmp / body.path.name
        shutil.copytree(body.path, copy)
        run = B.load(copy)
        before = snapshot(D.OUT)
        report.render(run.as_result(), copy / "report.html")
        ch, gone, new = diff(before, snapshot(D.OUT))
        check("прогон в теле читается своим загрузчиком",
              len(run.records) > 0 and bool(run.metrics),
              "шагов %d, метрик %d, калибровка %s"
              % (len(run.records), len(run.metrics), bool(run.calib)))
        check("отчёт тела не трогает научный каталог",
              not ch and not gone and not new,
              "изменилось: %s" % (ch + new))
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        # Отсутствие — свойство архива, а не поломка кода: одиночные
        # прогоны в теле до сегодняшнего дня на диск не ложились вовсе.
        # `body.load` проверяется там, где прогон сначала считают: в
        # `cli_check` и в `stable_v1`.
        print("  пропуск  прогона в теле на диске нет — `body.load` "
              "проверяется в cli_check и stable_v1")
        rep["проверки"]["прогон в теле"] = {
            "пройдено": None,
            "замер": "на диске нет ни одного одиночного прогона в теле"}

    # --- 5. перестроение указателя ---
    before = snapshot(D.OUT)
    t0 = time.perf_counter()
    path = index.render()
    idx_s = time.perf_counter() - t0
    ch, gone, new = diff(before, snapshot(D.OUT))
    check("перестроение указателя не трогает прогоны",
          not ch and not gone and not new,
          "изменилось %s" % ch[:3] if ch else
          "%s за %.1f с" % (path.name, idx_s))
    rep["замеры"]["index.render, с"] = round(idx_s, 2)

    # --- 6. сверка итогов ---
    final = snapshot(D.OUT)
    ch, gone, new = diff(base, final)
    check("за весь прогон проверки каталог не изменился",
          not ch and not new and not gone,
          "изменилось %d, появилось %d, пропало %d"
          % (len(ch), len(new), len(gone)))
    rep["итог"] = {"изменено": ch, "появилось": new, "пропало": gone}

    (OUT / "integrity.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    skipped = sum(1 for v in rep["проверки"].values()
                  if v["пройдено"] is None)
    print("\nпроверок %d, провалов %d, пропущено %d"
          % (len(rep["проверки"]) - skipped, len(fails), skipped))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
