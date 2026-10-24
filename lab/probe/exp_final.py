"""Заключительная проверка «Эксперимента»: сохранение, отмена, серия, сверка.

Работник настоящий, прогоны считаются по-настоящему, файлы проверяются на
диске. Проверяется ровно то, что было названо дефектом:

  Э1  исходы серии не пересекаются: запланировано = выполнено + отменено
      + с ошибкой, и чужой одиночный прогон в запись серии не попадает;
  Э4  «Сверка с апстримом» либо считает и сравнивает, либо честно
      говорит, почему не может;
  Э5  результат сохраняется у всех пяти видов, в том числе при отмене и
      при серии, и повторный запуск не ложится в чужую папку;
  Э7  сквозной путь: сборка RunSpec -> пуск -> ход -> конец -> файлы ->
      открытие результата.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\exp_final.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401  прогоны — во временный каталог

OUT = Path(__file__).parent
fails = []
rep = {"прогоны": {}, "проверки": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def wait(app, limit=600.0, cancel_after=None):
    """Крутить кадры до конца прогона. Вернуть снимок последнего."""
    t0 = time.perf_counter()
    worst = 0.0
    asked = False
    while time.perf_counter() - t0 < limit:
        a = time.perf_counter()
        app.frame()
        worst = max(worst, (time.perf_counter() - a) * 1000.0)
        if (cancel_after is not None and not asked
                and time.perf_counter() - t0 > cancel_after
                and app.st.run.phase == "run"):
            asked = True
            app.cancel_run()
        if not app.st.run.busy and app.st.last is not None:
            break
    return app.st.last, worst, time.perf_counter() - t0


def files_of(path):
    p = Path(path) if path else None
    if p is None or not p.exists():
        return []
    return sorted(x.name for x in p.iterdir())


def run_one(app, name, spec, title, want, cancel_after=None, limit=600.0):
    """Посчитать один прогон и проверить, что он лёг на диск как надо."""
    app.st.last = None
    job = app.start_run(spec, title=title)
    last, worst, took = wait(app, limit=limit, cancel_after=cancel_after)
    got = files_of(last.out_dir if last else "")
    runjson = {}
    if last and last.out_dir and (Path(last.out_dir) / "run.json").exists():
        runjson = json.loads((Path(last.out_dir) / "run.json")
                             .read_text(encoding="utf-8"))
    rep["прогоны"][name] = {
        "задача": job, "фаза": last.phase if last else None,
        "исход": last.outcome if last else None,
        "папка": last.out_dir if last else "",
        "файлы": got, "run.json": runjson,
        "по стене, с": round(took, 1),
        "худший кадр окна, мс": round(worst, 1),
    }
    miss = [w for w in want if w not in got]
    check("%s: файлы на месте" % name, last is not None and not miss,
          ("нет: " + ", ".join(miss)) if miss else ", ".join(got))
    check("%s: окно не вставало" % name, worst < 400.0,
          "худший кадр %.0f мс" % worst)
    return last, runjson


def main():
    from flybrain import catalog as K, data as D
    from flybrain.desk import ws_experiment as X
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

    # ---------------------------------------------------- Э5: пять видов
    last, rj = run_one(app, "стимуляция", spec_of("stim", t=0.3),
                       "Стимуляция",
                       ["bins.npz", "meta.json", "rates.parquet",
                        "run.json", "spec.json"])
    check("стимуляция: run.json не считает прогон отменённым",
          rj.get("cancelled") is False, rj)
    check("стимуляция: исход «пройден целиком»",
          last is not None and last.outcome == "done",
          last.outcome if last else None)

    run_one(app, "арена", spec_of("arena", t=0.4), "Арена",
            ["episode.json", "run.json", "spec.json", "track.parquet"])

    run_one(app, "перебор популяций", spec_of("lateral", t=0.6),
            "Перебор популяций",
            ["lateral.json", "run.json", "spec.json"])

    # Тело дороже всех: сборка MuJoCo и разогрев нисходящих идут до
    # первого шага, поэтому и длительность здесь самая короткая.
    run_one(app, "в теле", spec_of("body", t=0.2), "В теле",
            ["embodied.json", "run.json", "spec.json", "track.parquet"],
            limit=900.0)

    # ------------------------------------------------- Э1: серия целиком
    sp = spec_of("arena", t=0.3)
    sp["repeat"] = 1
    sp["repeat_t"] = 0.3
    last, rj = run_one(app, "серия арены", sp, "Арена (серия)",
                       ["bench.json", "run.json", "spec.json"], limit=900.0)
    ser = app.series
    rep["серия"] = dict(ser or {})
    ok_sum = bool(ser) and (ser["done"] + ser["cancelled"] + ser["failed"]
                            == ser["planned"])
    check("Э1 серия: сумма исходов равна запланированному", ok_sum,
          "%s" % ({k: ser.get(k) for k in
                   ("planned", "done", "cancelled", "failed")} if ser
                  else None))
    check("Э1 серия прошла целиком",
          bool(ser) and ser["done"] == ser["planned"]
          and not ser["cancelled"] and not ser["failed"],
          "выполнено %s из %s" % (ser.get("done") if ser else None,
                                  ser.get("planned") if ser else None))
    check("Э1 серия: bench.json читается",
          "bench.json" in rep["прогоны"]["серия арены"]["файлы"]
          and bool(json.loads(
              (Path(last.out_dir) / "bench.json").read_text(
                  encoding="utf-8"))) if last and last.out_dir else False)
    check("Э5 run.json помечает серию серией", rj.get("series") is True, rj)

    # -------------------------------------- Э1: одиночный прогон не портит
    run_one(app, "одиночный после серии", spec_of("stim", t=0.2),
            "Стимуляция", ["run.json", "spec.json"])
    check("Э1 одиночный прогон не попал в запись серии",
          app.series is None, "запись серии: %s" % (app.series,))

    # ---------------------------------------------------- Э5: отмена
    sp = spec_of("stim", t=12.0)
    last, rj = run_one(app, "отменённая стимуляция", sp, "Стимуляция",
                       ["bins.npz", "meta.json", "run.json", "spec.json"],
                       cancel_after=2.0, limit=300.0)
    check("Э5 отменённый прогон сохранён и помечен отменённым",
          rj.get("cancelled") is True, rj)
    check("Э5 окно различает отмену и завершение",
          last is not None and last.outcome == "cancelled",
          last.outcome if last else None)
    meta = {}
    if last and last.out_dir:
        f = Path(last.out_dir) / "meta.json"
        if f.exists():
            meta = json.loads(f.read_text(encoding="utf-8"))
    t_saved = ((meta.get("config") or {}).get("t_run_s")
               if isinstance(meta.get("config"), dict)
               else meta.get("t_run_s"))
    check("Э5 длительность в файле подрезана до посчитанного",
          t_saved is not None and float(t_saved) < 12.0,
          "config.t_run_s = %s при заказанных 12" % t_saved)

    # -------------------------------- Э5: повторный запуск в то же имя
    name = "probe-overwrite-test"
    for i in (1, 2):
        sp = spec_of("stim", t=0.2)
        sp["out"] = name
        last, _rj = run_one(app, "повтор в ту же папку %d" % i, sp,
                            "Стимуляция", ["run.json", "spec.json"])
        rep["прогоны"]["повтор в ту же папку %d" % i]["имя"] = (
            Path(last.out_dir).name if last and last.out_dir else "")
    a = rep["прогоны"]["повтор в ту же папку 1"].get("имя")
    b = rep["прогоны"]["повтор в ту же папку 2"].get("имя")
    check("Э5 второй прогон не лёг в папку первого", a and b and a != b,
          "%s и %s" % (a, b))

    # ---------------------------------------------------- Э4: сверка
    sp = spec_of("verify", t=0.1)
    last, rj = run_one(app, "сверка с апстримом", sp, "Сверка с апстримом",
                       ["run.json", "spec.json"], limit=900.0)
    summ = {}
    if last is not None:
        summ = dict(getattr(ses, "summary", None) or {})
    rep["сверка: сводка"] = summ
    check("Э4 сверка не падает", last is not None and last.phase == "done",
          (last.phase if last else None) + " | " + (last.message if last
                                                    else ""))
    check("Э4 сверка сказала, сравнила ли она что-нибудь",
          "compared" in summ, summ.get("why") or summ.get("compared"))
    if summ.get("compared"):
        check("Э4 корреляция с апстримом не ниже 0,9",
              float(summ.get("corr") or 0.0) >= 0.9,
              "corr %.4f, Jaccard %.3f, мой %s / апстрим %s активных"
              % (summ.get("corr", 0.0), summ.get("jaccard", 0.0),
                 summ.get("n_active_mine"), summ.get("n_active_upstream")))

    # ------------------------------------------- Э7: сквозной путь целиком
    sp = spec_of("stim", t=0.3)
    bad = K.validate(sp)
    check("Э7 собранная спека проходит проверку", not bad, bad)
    from flybrain import spec as SP
    parsed = SP.from_dict(sp)
    check("Э7 спека разбирается в RunSpec", parsed.kind == "stim")
    last, rj = run_one(app, "сквозной путь", sp, "Стимуляция",
                       ["bins.npz", "meta.json", "rates.parquet",
                        "run.json", "spec.json"])
    p = Path(last.out_dir) if last and last.out_dir else None
    again = SP.from_dict(json.loads((p / "spec.json").read_text(
        encoding="utf-8"))) if p else None
    check("Э7 сохранённая спека повторяется",
          again is not None and again.kind == "stim" and again.t == 0.3,
          "t = %s" % (again.t if again else None))

    ses.close()
    (OUT / "exp_final.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
