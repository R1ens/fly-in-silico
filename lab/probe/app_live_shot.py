"""Снимки рабочего пространства «Тело» с настоящим прогоном.

То же, что `app_live.py`, но без окна: кадр собирается в поверхность
заданного размера. Нужно по простой причине — окно в этой среде меняет
размер само (в прошлом прогоне оно стало 346×880), и попросить у него
1920×1080 нельзя, а показывать раскладку надо в тех размерах, которые
названы в брифе.

Данные настоящие: работник поднят, эпизод в теле идёт, кадры MuJoCo
приходят через общую память, числа в колонке — из записей шага.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\app_live_shot.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
SIZES = ((1920, 1080, 1.0), (1500, 880, 1.0), (1280, 720, 1.0),
         (1500, 880, 1.5))


def main():
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen((1500, 880))
    ses = Session()
    app = App(sh, ses)
    app.st.t_run = 8.0
    rep = {"снимки": []}
    t0 = time.perf_counter()

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            sh.t = time.perf_counter() - t0
            app.frame()

    def shot(name, w, h, dpi=1.0):
        sh.resize(w, h, dpi)
        sh.t = time.perf_counter() - t0
        app.frame()
        app.frame()
        path = OUT / name
        sh.shot(path)
        rec = {"снимок": name, "размер": [w, h], "dpi": dpi,
               "фаза": app.st.run.phase, "этап": app.st.run.stage,
               "кадров тела": int(ses.body_seq),
               "t_sim": round(app.st.run.t_sim, 3),
               "мс на кадр": round(sh.frame_ms, 2),
               "запусков ядра": sh.launches}
        rep["снимки"].append(rec)
        print("  " + json.dumps(rec, ensure_ascii=False))

    try:
        while not app.st.worker.ready and time.perf_counter() - t0 < 120:
            app.frame()
        print("работник готов за %.1f с" % (time.perf_counter() - t0))
        shot("ws_body_ready_1500x880.png", 1500, 880)

        app.start_run()
        pump(1.0)
        shot("ws_body_prep_1500x880.png", 1500, 880)

        t_wait = time.perf_counter()
        while ses.body_frame is None and time.perf_counter() - t_wait < 120:
            app.frame()
        print("первый кадр тела через %.1f с после пуска"
              % (time.perf_counter() - t_wait))

        from PIL import Image
        import numpy as np
        raw = np.ascontiguousarray(ses.body_frame)
        Image.fromarray(raw).save(OUT / "live_raw.png")
        np.save(OUT / "live_raw.npy", raw)
        g = np.abs(raw.astype(np.int16)
                   - raw.astype(np.int16).mean(axis=2, keepdims=True)).mean()
        rep["разброс каналов в сыром кадре"] = round(float(g), 2)
        print("сырой кадр сохранён, разброс каналов %.2f" % g)

        for (w, h, dpi) in SIZES:
            pump(1.2)
            shot("ws_body_run_%dx%d_dpi%d.png" % (w, h, round(dpi * 100)),
                 w, h, dpi)

        app.st.ws.set_tab(1)
        pump(1.0)
        shot("ws_body_track_1500x880.png", 1500, 880)
        app.st.ws.set_tab(2)
        shot("ws_body_vision_1500x880.png", 1500, 880)
        app.st.ws.set_tab(0)

        cam = app.st.camera("body")
        for view in ("side", "top", "front"):
            cam.view = view
            pump(1.6)
            shot("ws_body_view_%s.png" % view, 1500, 880)
        cam.view, cam.dist = "back", 0.6
        pump(1.6)
        shot("ws_body_zoom.png", 1500, 880)
        cam.dist = 1.0

        app.st.ws.dock().maximized = True
        pump(0.6)
        shot("ws_body_max_1920x1080.png", 1920, 1080)
        app.st.ws.dock().maximized = False

        app.cancel_run()
        pump(2.0)
        shot("ws_body_cancel_1500x880.png", 1500, 880)
        pump(2.0)
        shot("ws_body_done_1500x880.png", 1500, 880)

        rep["итог"] = {"кадров тела": int(ses.body_seq),
                       "фаза": app.st.run.phase,
                       "t_sim": round(app.st.run.t_sim, 3),
                       "t_wall": round(app.st.run.t_wall, 1)}
        print("итог: " + json.dumps(rep["итог"], ensure_ascii=False))
    finally:
        ses.close()
        (OUT / "app_live_shot.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("отчёт: probe/app_live_shot.json")


if __name__ == "__main__":
    main()
