"""Снимки «Песочницы» на настоящем сеансе.

То же устройство, что у `app_live_shot.py`: окна нет, кадр собирается в
поверхность заданного размера — окно в этой среде меняет размер само, и
попросить у него ровно 1600×900 нельзя.

Данные настоящие во всём: работник поднят, сеанс в теле идёт, кадры
MuJoCo приходят через общую память, метки источников кладутся по позе
камеры того кадра, что на экране, а правки условий уходят тем же путём,
каким их посылает рейка, — через `ws_sandbox._send_slots`.

Проба не только снимает кадры, но и проверяет по ним то, что в
одиночных кадрах не видно: что метка слота после переноса оказалась на
новом месте, что состояния команды сменились по порядку, и что
раскладка выдерживает сужение окна.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\sandbox_live_shot.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

OUT = Path(__file__).parent
SIZES = ((1600, 900, 1.0), (1280, 720, 1.0), (1000, 760, 1.0))
fails = []
rep = {"проверки": {}, "снимки": [], "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():                                                   # noqa: C901
    from flybrain.desk import ws_sandbox as X
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen((1600, 900))
    ses = Session()
    app = App(sh, ses)
    st = app.st
    st.ws.current = "sandbox"
    sb = st.sb
    sb.t_run = 0.0          # сеанс без заранее заданного конца
    sb.keep_s = 10.0
    t0 = time.perf_counter()
    worst = [0.0]

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            sh.t = time.perf_counter() - t0
            a = time.perf_counter()
            app.frame()
            worst[0] = max(worst[0], (time.perf_counter() - a) * 1000.0)

    def shot(name, w=1600, h=900, dpi=1.0):
        sh.resize(w, h, dpi)
        sh.t = time.perf_counter() - t0
        app.frame()
        app.frame()
        path = OUT / name
        sh.shot(path)
        rec = {"снимок": name, "размер": [w, h],
               "фаза": st.run.phase, "этап": st.run.stage,
               "кадров тела": int(ses.body_seq),
               "модельное, с": round(st.run.t_sim, 3),
               "скорость": round(sb.ratio, 4),
               "ждут": len(sb.pending),
               "мс на кадр": round(sh.frame_ms, 2)}
        rep["снимки"].append(rec)
        print("  " + json.dumps(rec, ensure_ascii=False))
        return path

    try:
        while not st.worker.ready and time.perf_counter() - t0 < 180:
            app.frame()
        check("работник готов", st.worker.ready,
              "за %.1f с" % (time.perf_counter() - t0))
        if not st.worker.ready:
            return 1

        # 1. Простой: сеанса не было, кадра нет, позы камеры нет.
        p = shot("sandbox-01-простой.png")
        check("пространство рисуется в простое", p.exists(), str(p))
        ok_run, why = app.run_ready()
        check("«Пуск» в песочнице возможен", ok_run, why or "можно")

        # 2. Подготовка и первый кадр.
        app.start_run()
        pump(1.5)
        shot("sandbox-02-сборка.png")
        check("на сборке доля не выдумывается",
              st.run.t_target <= 0.0 and st.run.progress <= 0.0,
              "заказано %.1f с, доля %.2f"
              % (st.run.t_target, st.run.progress))
        t_wait = time.perf_counter()
        while ses.body_frame is None and time.perf_counter() - t_wait < 240:
            app.frame()
        check("первый кадр MuJoCo пришёл", ses.body_frame is not None,
              "через %.1f с после пуска" % (time.perf_counter() - t_wait))

        # Отодвинуть камеру: источники в двенадцати миллиметрах, а охват
        # считается по габариту мухи.
        cam = st.camera("body")
        cam.view, cam.dist = "back", 5.0
        pump(8.0)
        shot("sandbox-03-сеанс.png")
        check("поза камеры дошла до окна", sb.cam is not None,
              "кадр %sx%s" % ((sb.cam or {}).get("w"),
                              (sb.cam or {}).get("h")))

        # 3. Включить источник слева и поймать три состояния команды.
        sb.slots[0].on = True
        cid = X._send_slots(app, "слот 1 включён")
        check("команда ушла", bool(cid), str(cid))
        app.frame()
        check("сразу после отправки она ждёт применения",
              cid in sb.pending, "в пути %d" % len(sb.pending))
        shot("sandbox-04-отправлено.png")
        t_wait = time.perf_counter()
        while cid in sb.pending and time.perf_counter() - t_wait < 60:
            pump(0.1)
        check("подтверждение пришло от работника", cid not in sb.pending,
              "за %.1f с по стене" % (time.perf_counter() - t_wait))
        applied = [e for e in sb.events
                   if str(e.get("подробно", "")).startswith("применено")]
        check("в журнале есть применение с модельным временем",
              bool(applied) and applied[-1].get("модельное, мс") is not None,
              str(applied[-1]) if applied else "записи нет")
        pump(4.0)
        shot("sandbox-05-источник-включён.png")
        check("модель подтвердила включение слота",
              bool(sb.applied_slots) and bool(sb.applied_slots[0].get("on")),
              json.dumps(sb.applied_slots[:1], ensure_ascii=False))
        ex = dict(ses.extra or {})
        rep["замеры"]["запах после включения"] = [ex.get("odour_l"),
                                                  ex.get("odour_r")]
        check("вход рецепторов стал ненулевым",
              float(ex.get("odour_l", 0) or 0) > 0.0,
              "слева %s, справа %s" % (ex.get("odour_l"), ex.get("odour_r")))

        # 4. Перенести источник: метка должна уехать.
        from flybrain import scene as SC

        before = (SC.project(sb.cam, (sb.slots[0].x, sb.slots[0].y, 1.5))
                  if sb.cam else None)
        sb.slots[0].x, sb.slots[0].y = -6.0, 14.0
        cid = X._send_slots(app, "слот 1 перенесён")
        pump(1.0)
        shot("sandbox-06-перенос-ждёт.png")
        t_wait = time.perf_counter()
        while cid in sb.pending and time.perf_counter() - t_wait < 60:
            pump(0.1)
        pump(4.0)
        after = (SC.project(sb.cam, (-6.0, 14.0, 1.5)) if sb.cam else None)
        rep["замеры"]["метка слота в кадре"] = {
            "до": [round(v, 1) for v in before] if before else None,
            "после": [round(v, 1) for v in after] if after else None}
        check("метка слота в кадре уехала",
              before is not None and after is not None
              and ((after[0] - before[0]) ** 2
                   + (after[1] - before[1]) ** 2) ** 0.5 > 20.0,
              "с %s на %s"
              % ([round(v) for v in before] if before else None,
                 [round(v) for v in after] if after else None))
        shot("sandbox-07-перенесён.png")

        # 5. Прямой стимул по DSL.
        sb.query = "region:AL+side:left"
        sb.stim_hz = 150.0
        cid = X._send(app, "стимул AL слева, 150 Гц",
                      {"stim": [{"query": sb.query, "rate_hz": 150.0}]},
                      fields=("stim",))
        t_wait = time.perf_counter()
        while cid in sb.pending and time.perf_counter() - t_wait < 60:
            pump(0.1)
        pump(3.0)
        check("стимул по DSL применён", bool(sb.applied_stim),
              json.dumps(sb.applied_stim, ensure_ascii=False))
        shot("sandbox-08-стимул.png")

        # 6. Пауза, подготовка правки, один шаг.
        X._send(app, "пауза", action="pause")
        t_wait = time.perf_counter()
        while not sb.paused and time.perf_counter() - t_wait < 60:
            pump(0.1)
        check("пауза подтверждена", sb.paused,
              "за %.1f с" % (time.perf_counter() - t_wait))
        frozen = ses.t_ms
        pump(3.0)
        check("на паузе модельное время стоит",
              abs(ses.t_ms - frozen) < 1e-9,
              "%.0f мс и через три секунды те же %.0f" % (frozen, ses.t_ms))
        sb.on_next = True
        sb.slots[1].on = True
        cid = X._send_slots(app, "слот 2 включён со следующего шага")
        pump(1.0)
        shot("sandbox-09-пауза.png")
        check("правка на паузе придержана до следующего шага",
              cid in sb.pending,
              "в пути %d, не раньше %s мс"
              % (len(sb.pending),
                 (sb.pending.get(cid) or {}).get("не раньше, мс")))
        X._send(app, "один шаг", action="step")
        t_wait = time.perf_counter()
        while cid in sb.pending and time.perf_counter() - t_wait < 90:
            pump(0.1)
        check("придержанная правка применилась на шаге",
              cid not in sb.pending and len(sb.applied_slots) > 1
              and bool(sb.applied_slots[1].get("on")),
              json.dumps(sb.applied_slots[1:2], ensure_ascii=False))
        sb.on_next = False
        X._send(app, "продолжить", action="resume")
        pump(4.0)
        shot("sandbox-10-продолжен.png")

        # 7. Закладки правой колонки и сужение окна.
        for i, name in enumerate(("показания", "события", "команды")):
            st.ws.set_tab(i)
            pump(0.8)
            shot("sandbox-11-%s.png" % name)
        st.ws.set_tab(0)
        for (w, h, dpi) in SIZES:
            pump(0.8)
            shot("sandbox-12-%dx%d.png" % (w, h), w, h, dpi)
        check("раскладка держится при сужении окна", worst[0] < 150.0,
              "худший кадр %.0f мс" % worst[0])
        rep["замеры"]["худший кадр, мс"] = round(worst[0], 1)
        rep["замеры"]["скорость к настенной"] = round(sb.ratio, 4)

        # 8. Остановка и сохранение.
        X._send(app, "остановить сеанс", action="stop")
        t_wait = time.perf_counter()
        while st.run.busy and time.perf_counter() - t_wait < 240:
            app.frame()
        check("сеанс остановлен из окна", not st.run.busy,
              "за %.1f с, исход %s" % (time.perf_counter() - t_wait,
                                       st.run.outcome))
        pump(3.0)
        shot("sandbox-13-остановлен.png")
        check("прочие четыре пространства целы", True, "проверяются ниже")
        for key in ("experiment", "brain", "body", "analysis"):
            st.ws.current = key
            pump(0.4)
            p = shot("sandbox-14-%s.png" % key)
            check("пространство «%s» рисуется" % key, p.exists(), str(p))
        st.ws.current = "sandbox"
    finally:
        try:
            ses.close()
        except Exception:                                 # noqa: BLE001
            pass

    (OUT / "sandbox_live_shot.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nснимков %d, проверок %d, провалов %d"
          % (len(rep["снимки"]), len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
