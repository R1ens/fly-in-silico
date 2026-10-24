"""Живой прогон нового окна с настоящим работником.

Проверяет то, чего не проверить ни без окна, ни без счёта:

* окно открывается, кадры идут, клавиатура цепляется нативно;
* работник поднимается, и фазы задачи видны по отдельности —
  подготовка, счёт, остановка, конец;
* в теле MuJoCo появляются настоящие кадры и попадают во вьюпорт;
* смена вида камеры доходит до работника и меняет картинку;
* «Стоп» останавливает прогон и процесс не остаётся висеть.

Окно ведётся само: никому не надо ничего нажимать. Снимки пишутся по
ходу, числа — в `probe/app_live.json`.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\app_live.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
PLAN = (
    # (когда, что сделать, имя снимка)
    (1.0, None, "live_01_boot.png"),
    (0.0, "wait_ready", "live_02_ready.png"),
    (0.5, "run", "live_03_prep.png"),
    (0.0, "wait_frame", "live_04_first_frame.png"),
    (2.0, "view_side", "live_05_side.png"),
    (2.0, "view_top", "live_06_top.png"),
    (2.0, "view_front_zoom", "live_07_front_zoom.png"),
    (1.5, "tab_vision", "live_08_vision.png"),
    (1.0, "tab_track", "live_09_track.png"),
    (0.5, "cancel", "live_10_cancel.png"),
    (3.0, "tab_meters", "live_11_after.png"),
    (1.0, "maximize", "live_12_max.png"),
)


def main():
    from flybrain.desk.app import App
    from flybrain.desk.shell import Shell
    from flybrain.ui.session import Session

    sh = Shell("FlyBrain — тело", (1500, 880))
    ses = Session()
    app = App(sh, ses)
    app.st.t_run = 6.0

    rep = {"клавиатура": "нативная" if sh.keys.native else str(sh.keys.failure),
           "dpi": sh.dpi_info, "шаги": [], "размеры": []}
    step = 0
    wait_until = time.perf_counter()
    t0 = time.perf_counter()
    first_frame_at = None
    ready_at = None
    frames_at_cancel = None
    deadline = t0 + 240.0

    def note(what, extra=None):
        rec = {"шаг": what, "t": round(time.perf_counter() - t0, 2),
               "фаза": app.st.run.phase, "этап": app.st.run.stage,
               "кадров тела": int(getattr(ses, "body_seq", 0)),
               "t_sim": round(app.st.run.t_sim, 3),
               "кадр/с окна": round(sh.fps, 1),
               "мс на кадр": round(sh.frame_ms, 2)}
        if extra:
            rec.update(extra)
        rep["шаги"].append(rec)
        print("  " + json.dumps(rec, ensure_ascii=False))

    try:
        while sh.running and time.perf_counter() < deadline:
            app.frame()
            if (sh.w, sh.h) not in rep["размеры"]:
                rep["размеры"].append((sh.w, sh.h))
            now = time.perf_counter()
            if ready_at is None and app.st.worker.ready:
                ready_at = now - t0
            if first_frame_at is None and ses.body_frame is not None:
                first_frame_at = now - t0
            if step >= len(PLAN) or now < wait_until:
                if step >= len(PLAN):
                    break
                continue

            delay, what, shot = PLAN[step]
            if what == "wait_ready" and not app.st.worker.ready:
                continue
            if what == "wait_frame" and ses.body_frame is None:
                continue

            cam = app.st.camera("body")
            if what == "run":
                app.start_run()
            elif what == "view_side":
                cam.view = "side"
            elif what == "view_top":
                cam.view = "top"
            elif what == "view_front_zoom":
                cam.view, cam.dist = "front", 0.55
            elif what == "tab_vision":
                app.st.ws.set_tab(2)
            elif what == "tab_track":
                app.st.ws.set_tab(1)
            elif what == "tab_meters":
                app.st.ws.set_tab(0)
            elif what == "cancel":
                frames_at_cancel = int(ses.body_seq)
                app.cancel_run()
            elif what == "maximize":
                app.st.ws.dock().maximized = True

            app.frame()                       # показать последствие
            sh.shot(OUT / shot)
            if ses.body_frame is not None and not (OUT / "live_raw.png").exists():
                # Сырой кадр работника, как он пришёл: с ним сравнивается
                # то, что видно во вьюпорте.
                from PIL import Image
                import numpy as np
                arr = np.ascontiguousarray(ses.body_frame)
                Image.fromarray(arr).save(OUT / "live_raw.png")
                np.save(OUT / "live_raw.npy", arr)
            note(what or "старт", {"снимок": shot})
            step += 1
            wait_until = now + delay
    finally:
        rep["итог"] = {
            "работник готов за, с": None if ready_at is None else round(ready_at, 1),
            "первый кадр тела за, с": None if first_frame_at is None
            else round(first_frame_at, 1),
            "кадров тела всего": int(getattr(ses, "body_seq", 0)),
            "кадров на момент остановки": frames_at_cancel,
            "фаза в конце": app.st.run.phase,
            "кадр/с окна": round(sh.fps, 1),
            "мс на кадр": round(sh.frame_ms, 2),
            "перестроений окна": len(sh.resizes),
        }
        print("\nитог: " + json.dumps(rep["итог"], ensure_ascii=False,
                                      indent=1))
        ses.close()
        time.sleep(0.4)
        rep["итог"]["работник жив после закрытия"] = bool(
            ses.proc is not None and ses.proc.is_alive())
        sh.close()
        (OUT / "app_live.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("отчёт: probe/app_live.json")


if __name__ == "__main__":
    main()
