"""Пространство «Эксперимент»: все виды опыта, запуск, серия, результат.

Работник настоящий: опыт действительно считается, ход и отмена идут по
тому же пути, что у командной строки. Окна нет — размер задаётся.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\exp_shot.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401  прогоны — во временный каталог

OUT = Path(__file__).parent
fails = []
rep = {"виды": {}, "снимки": [], "проверки": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():
    from flybrain import catalog as K
    from flybrain.desk import ws_experiment as X
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen((1500, 880))
    ses = Session()
    app = App(sh, ses)
    app.st.ws.switch("experiment")
    t0 = time.perf_counter()
    while not app.st.worker.ready and time.perf_counter() - t0 < 120:
        app.frame()
    print("работник готов за %.1f с" % (time.perf_counter() - t0))
    xs = X._state(app)

    def shot(name, note=""):
        app.frame()
        app.frame()
        sh.shot(OUT / name)
        rep["снимки"].append({"снимок": name, "что": note,
                              "мс на кадр": round(sh.frame_ms, 2)})
        print("  снимок %-34s %s" % (name, note))

    # --- все виды опыта рисуются и дают разборную спеку ---
    for kind in X.KIND_ORDER:
        X._reset_values(xs, kind)
        app.frame()
        app.frame()
        spec = X.build_spec(app)
        from flybrain import spec as SP
        try:
            parsed = SP.from_dict(spec)
            ok_parse = parsed.kind == kind
        except Exception as exc:                              # noqa: BLE001
            ok_parse = False
            print("    разбор упал:", exc)
        bad = K.validate(spec)
        warn = K.limits(spec)
        rep["виды"][kind] = {"полей": len(K.fields_of(kind)),
                             "спека разобралась": ok_parse,
                             "ошибок проверки": bad,
                             "предупреждений": len(warn)}
        check("вид «%s» собирается и разбирается" % K.KINDS[kind].title,
              ok_parse and not bad,
              "ошибки: %s" % (bad or "нет"))
        shot("exp_%s.png" % kind, K.KINDS[kind].title)

    # --- проверка ловит кривое ---
    X._reset_values(xs, "stim")
    xs["terms"] = [{"query": "", "rate_hz": 200.0, "t_on_ms": 0.0,
                    "t_off_ms": None, "name": ""}]
    sh.ctx.drop_editor("exp/term0")
    app.frame()
    bad = K.validate(X.build_spec(app))
    ok_run, why = app.run_ready()
    check("пустой стимул не даёт запустить", bool(bad) and not ok_run,
          "проверка: %s | кнопка: %s" % (bad, why))
    shot("exp_invalid.png", "проверка не пускает")

    xs["terms"] = [{"query": "class:gustatory+side:left",
                    "rate_hz": 200.0, "t_on_ms": 0.0, "t_off_ms": None,
                    "name": ""}]
    sh.ctx.drop_editor("exp/term0")
    xs["values"]["t"] = 0.5
    app.frame()
    ok_run, why = app.run_ready()
    check("с правильным стимулом пуск доступен", ok_run, why)

    # --- настоящий запуск ---
    spec = X.build_spec(app)
    job = app.start_run(spec, title=K.KINDS["stim"].title)
    check("опыт запустился", bool(job), "задача %s" % job)
    t1 = time.perf_counter()
    worst = 0.0
    shot_run = False
    while time.perf_counter() - t1 < 180:
        a = time.perf_counter()
        app.frame()
        worst = max(worst, (time.perf_counter() - a) * 1000.0)
        if not shot_run and app.st.run.busy and ses.frames_seen > 3:
            shot_run = True
            shot("exp_running.png", "опыт идёт")
        if not app.st.run.busy and app.st.last is not None:
            break
    last = app.st.last
    check("опыт дошёл до конца", last is not None and last.phase == "done",
          "фаза %s, модельное %.2f с, по стене %.1f с"
          % (None if last is None else last.phase,
             0 if last is None else last.t_sim,
             0 if last is None else last.t_wall))
    check("окно снова готово", app.st.run.phase == "idle",
          app.st.run.phase)
    check("окно не вставало при счёте", worst < 300.0,
          "худший кадр %.0f мс" % worst)
    rep["худший кадр при счёте, мс"] = round(worst, 1)
    app.st.ws.set_tab(2)
    shot("exp_result.png", "вкладка «Результат»")

    # --- серия ---
    X._reset_values(xs, "arena")
    xs["values"]["t"] = 0.6
    xs["values"]["repeat"] = 2
    app.frame()
    spec = X.build_spec(app)
    rep["серия: спека"] = spec
    planned = X._planned(spec)
    check("серия планирует норму и контроль", planned == 4,
          "запланировано %d эпизодов при repeat=2" % planned)
    app.st.ws.set_tab(1)
    xs["series"] = {"planned": planned, "done": 0, "cancelled": 0,
                    "failed": 0}
    shot("exp_series.png", "вкладка «Серия» до запуска")

    # --- отмена на длинном опыте ---
    X._reset_values(xs, "stim")
    xs["values"]["t"] = 20.0
    app.frame()
    job = app.start_run(X.build_spec(app), title="Стимуляция")
    t1 = time.perf_counter()
    while app.st.run.phase == "prep" and time.perf_counter() - t1 < 60:
        app.frame()
    app.cancel_run()
    t2 = time.perf_counter()
    while app.st.run.busy and time.perf_counter() - t2 < 120:
        app.frame()
    stopped = time.perf_counter() - t2
    check("отмена останавливает опыт", not app.st.run.busy,
          "остановился за %.1f с" % stopped)
    rep["отмена, с"] = round(stopped, 1)
    shot("exp_cancelled.png", "после отмены")

    for (w, h) in ((1280, 720), (1920, 1080)):
        sh.resize(w, h, 1.0)
        shot("exp_%dx%d.png" % (w, h), "раскладка")
    sh.resize(1500, 880, 1.0)

    ses.close()
    (OUT / "exp_shot.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
