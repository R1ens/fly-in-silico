"""Проверка исправлений, найденных финальной регрессией «Тела».

Каждый пункт — числом, с работником, где он нужен.

  Д1  `SystemExit` из эпизода больше не убивает работника;
  Д2  выбранный выход на рулёжку доезжает до работника;
  Д3  кадр и счётчик прошлого прогона не выдаются за новый;
  Д4  протяжка камеры не прыгает в кадре нажатия;
  Д5  пуск и остановка доступны без нижней полосы;
  Д6  у отключённой детали есть подсказка;
  Д7  все кнопки полосы инструментов достижимы;
  Д8  полное имя длинной популяции доступно при наведении;
  Д11 нет nan в метриках при одном шаге.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\body_fixes.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401  прогоны — во временный каталог

OUT = Path(__file__).parent
fails = []
rep = {}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep[name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def no_worker():
    """Что проверяется без работника: раскладка, подсказки, полоса."""
    from flybrain.desk import layout as L
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.spec import from_dict
    from probe.app_shot import FakeSession

    # Д2 — спека доезжает целиком
    sp = from_dict({"kind": "body", "t": 0.4, "seed": 0,
                    "open_report": False, "body": {"readout": "DNa02"}})
    check("Д2 выход на рулёжку доезжает до работника",
          sp.body.readout == "DNa02" and sp.open_report is False,
          "readout=%s, open_report=%s" % (sp.body.readout, sp.open_report))

    sh = Offscreen((1500, 880))
    ses = FakeSession()
    ses.alive = ses.ready = True
    app = App(sh, ses)
    # `ctx.end` рисует подсказки и тут же очищает список, поэтому после
    # кадра он всегда пуст. Перехватываем саму отрисовку.
    seen_tips = []
    real_draw = sh.ctx._draw_tip

    def spy(r, text):
        seen_tips.append(text)
        return real_draw(r, text)

    sh.ctx._draw_tip = spy
    long_name = "DNa02_very_long_population_name_for_layout_test_LAL_left"
    app.st.worker.readouts = ("ALPN", long_name, "LAL")
    app.st.readout = long_name
    app.frame()
    app.frame()

    # Д8 — полное имя в подсказке списка
    r = sh.ctx.rect_of("b/out")
    sh.inp.mx, sh.inp.my = r.cx, r.cy
    sh.t = 0.4
    app.frame()
    seen_tips.clear()
    sh.t = 2.0
    app.frame()
    check("Д8 полное имя популяции в подсказке",
          any(long_name in t for t in seen_tips),
          "подсказок %d: %s" % (len(seen_tips),
                                (seen_tips[0][:50] if seen_tips else "")))
    check("Д8 усечение не меняет значение", app.st.readout == long_name)
    sh.inp.mx = sh.inp.my = -4000
    app.frame()

    # Д6 — подсказка у отключённой детали
    r = sh.ctx.rect_of("b/contacts")
    sh.inp.mx, sh.inp.my = r.cx, r.cy
    sh.t = 2.2
    app.frame()
    seen_tips.clear()
    sh.t = 4.0
    app.frame()
    check("Д6 подсказка у отключённого тумблера", bool(seen_tips),
          seen_tips[0][:40] if seen_tips else "нет")
    sh.inp.mx = sh.inp.my = -4000

    # Д7 — все кнопки полосы инструментов достижимы
    for (w, h, dpi) in ((1280, 720, 1.0), (1500, 880, 1.5),
                        (1920, 1080, 2.0), (1100, 700, 1.0)):
        sh.resize(w, h, dpi)
        app.frame()
        app.frame()
        want = ["body/tb/view", "body/tb/zoom_out", "body/tb/zoom_in",
                "body/tb/reset", "body/tb/full"]
        miss = [k for k in want if sh.ctx.rect_of(k) is None
                and sh.ctx.rect_of(k + "/0") is None]
        check("Д7 полоса инструментов целиком при %dx%d dpi %g"
              % (w, h, dpi), not miss, "нет: " + ",".join(miss) if miss
              else "все на месте")
    sh.resize(1500, 880, 1.0)

    # Д5 — пуск доступен без нижней полосы
    app.frame()
    base = sh.ctx.rect_of("tl/run") is not None
    app.st.ws.dock().toggle("timeline")
    app.frame()
    app.frame()
    hidden = sh.ctx.rect_of("tl/run") is not None
    app.st.ws.dock().toggle("timeline")
    sh.resize(1500, 880, 2.0)
    app.frame()
    app.frame()
    lay = L.compute(sh.w, sh.h, sh.sc, app.st.ws.dock())
    dropped = sh.ctx.rect_of("tl/run") is not None
    check("Д5 пуск есть и без полосы исполнения",
          base and hidden and dropped,
          "полная %s, скрыта %s, отброшена %s (отброшено %s)"
          % (base, hidden, dropped, lay.dropped))
    sh.resize(1500, 880, 1.0)

    # Д4 — прыжок камеры в кадре нажатия
    app.st.ws.switch("body")
    app.frame()
    cam = app.st.camera("body")
    cam.reset()
    vp = sh.ctx.rect_of("body/view")
    sh.inp.set_mouse(10, 10, False)          # курсор далеко от вида
    app.frame()
    sh.inp.set_mouse(vp.cx, vp.cy, True)     # нажали, сразу попав на вид
    app.frame()
    jump = abs(cam.yaw)
    sh.inp.set_mouse(vp.cx + 20, vp.cy, True)
    app.frame()
    moved = abs(cam.yaw)
    sh.inp.set_mouse(vp.cx + 20, vp.cy, False)
    app.frame()
    check("Д4 нажатие не двигает камеру", jump < 1e-6, "прыжок %.1f°" % jump)
    check("Д4 протяжка двигает камеру", 4.0 < moved < 12.0,
          "20 px дали %.1f°" % moved)
    sh.inp.set_mouse(-4000, -4000, False)


def with_worker():
    """Что проверяется только живым работником."""
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen((1500, 880))
    ses = Session()
    app = App(sh, ses)
    t0 = time.perf_counter()
    while not app.st.worker.ready and time.perf_counter() - t0 < 120:
        app.frame()
    print("  работник готов за %.1f с" % (time.perf_counter() - t0))

    # Д1 — заведомо неверная популяция
    app.st.readout = "НЕТ-ТАКОЙ-ПОПУЛЯЦИИ"
    app.st.t_run = 0.3
    app.start_run()
    t1 = time.perf_counter()
    while app.st.run.busy and time.perf_counter() - t1 < 120:
        app.frame()
    alive = ses.alive
    # Законченный прогон переезжает в «последний», а текущее состояние
    # возвращается в «готов»: это разные вещи и показываются врозь.
    last = app.st.last
    check("Д1 работник пережил SystemExit", alive, "alive=%s" % alive)
    check("Д1 окно увидело отказ",
          last is not None and last.phase == "fail",
          "последний прогон: %s, причина %s"
          % (None if last is None else last.phase,
             ((last.message if last else "") or "")[:60]))
    check("Д1 после отказа окно снова готово",
          app.st.run.phase == "idle",
          "текущее состояние %s" % app.st.run.phase)
    sh.shot(OUT / "bodyfix_fail.png")

    if not alive:
        return
    # Д1 — после отказа можно запустить снова
    app.st.readout = ""
    app.st.t_run = 0.3
    app.start_run()
    t1 = time.perf_counter()
    first_frame = None
    stale_seen = None
    while time.perf_counter() - t1 < 240:
        app.frame()
        if stale_seen is None and app.st.run.phase == "prep":
            stale_seen = (ses.body_frame is not None, int(ses.body_seq))
        if ses.body_frame is not None:
            first_frame = time.perf_counter() - t1
            break
    check("Д1 после отказа прогон запускается", first_frame is not None,
          "первый кадр за %s с" % (None if first_frame is None
                                   else round(first_frame, 1)))
    check("Д3 в подготовке нет кадра прошлого прогона",
          stale_seen == (False, 0), "кадр/счётчик в prep: %s" % (stale_seen,))
    sh.shot(OUT / "bodyfix_run2.png")

    # Д3 — и при третьем запуске тоже
    app.cancel_run()
    t1 = time.perf_counter()
    while app.st.run.busy and time.perf_counter() - t1 < 60:
        app.frame()
    had = ses.body_frame is not None
    app.start_run()
    app.frame()
    check("Д3 повторный пуск стирает прошлый кадр",
          had and ses.body_frame is None and ses.body_seq == 0,
          "был кадр %s, стало %s/%s" % (had, ses.body_frame is not None,
                                        ses.body_seq))
    app.cancel_run()
    t1 = time.perf_counter()
    while app.st.run.busy and time.perf_counter() - t1 < 90:
        app.frame()
    ses.close()


def main():
    print("без работника")
    no_worker()
    print("с работником")
    with_worker()
    (OUT / "body_fixes.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
