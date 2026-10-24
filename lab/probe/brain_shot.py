"""Пространство «Мозг» с настоящим коннектомом: снимки и замеры.

Работник поднимается по-настоящему, геометрия приходит от него, карту
рисует `ui/brainview.py`. Окна нет нарочно: окно в этой среде меняет
размер само, а показать раскладку надо в заданных размерах.

Меряется ровно то, что просит бриф: время кадра, наведения, смены
режима, выделения области, память и задержка обновления активности.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\brain_shot.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
SIZES = ((1920, 1080, 1.0), (1500, 880, 1.0), (1280, 720, 1.0),
         (1500, 880, 1.5))


def main():
    from PIL import Image

    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.desk import ws_brain
    from flybrain.ui.session import Session

    sh = Offscreen((1500, 880))
    ses = Session()
    app = App(sh, ses)
    app.st.ws.switch("brain")
    rep = {"замеры": {}, "снимки": []}
    t0 = time.perf_counter()

    def pump(seconds=0.0, frames=0):
        end = time.perf_counter() + seconds
        k = 0
        while time.perf_counter() < end or k < frames:
            sh.t = time.perf_counter() - t0
            app.frame()
            k += 1
        return k

    def shot(name, w=1500, h=880, dpi=1.0, note=""):
        sh.resize(w, h, dpi)
        app.frame()
        app.frame()
        sh.shot(OUT / name)
        rec = {"снимок": name, "размер": [w, h], "dpi": dpi,
               "мс на кадр": round(sh.frame_ms, 2),
               "запусков ядра": sh.launches, "что": note}
        rep["снимки"].append(rec)
        print("  " + json.dumps(rec, ensure_ascii=False))

    def timeit(key, fn, runs=12):
        fn()
        app.frame()
        ts = []
        for _ in range(runs):
            a = time.perf_counter()
            fn()
            app.frame()
            ts.append((time.perf_counter() - a) * 1000.0)
        ts.sort()
        rep["замеры"][key] = {"медиана, мс": round(ts[len(ts) // 2], 2),
                              "худшее, мс": round(ts[-1], 2)}
        print("  %-34s медиана %6.2f мс, худшее %6.2f мс"
              % (key, ts[len(ts) // 2], ts[-1]))

    try:
        # --- загрузка ---
        shot("brain_loading.png", note="коннектом загружается")
        t_wait = time.perf_counter()
        while app.brain is None and time.perf_counter() - t_wait < 180:
            app.frame()
        boot = time.perf_counter() - t_wait
        rep["замеры"]["геометрия пришла за, с"] = round(boot, 1)
        print("карта готова за %.1f с" % boot)
        bv = app.brain
        rep["карта"] = {"нейронов": int(bv.n),
                        "нейропилей": len(bv.region_names)}

        pump(frames=4)
        shot("brain_anatomy_1500x880.png", note="анатомия, ничего не выбрано")

        # --- охват полного коннектома на разных размерах ---
        def fit_report(w, h, dpi=1.0, tag=""):
            sh.resize(w, h, dpi)
            bv.fit_to(None)
            bv.zoom = 1.0
            app.frame()
            app.frame()
            vp = sh.ctx.rect_of("brain/view")
            a = np.clip(sh.surf.acc.to_numpy(), 0.0, 1.0)
            img = a.transpose(1, 0, 2)[::-1]
            cut = img[vp.y:vp.y1, vp.x:vp.x1]
            lum = cut.mean(axis=2)
            # Порог берём чуть выше фона панели, чтобы считать облако, а
            # не подложку.
            mask = lum > 0.045
            ys, xs = np.nonzero(mask)
            if not len(xs):
                print("  охват %dx%d: облака не видно" % (w, h))
                return
            rec = {"размер": [w, h], "dpi": dpi,
                   "вьюпорт": [vp.w, vp.h],
                   "облако": [int(xs.max() - xs.min()),
                              int(ys.max() - ys.min())],
                   "поля": [int(xs.min()), int(vp.w - xs.max()),
                            int(ys.min()), int(vp.h - ys.max())]}
            rec["заполнение по ширине"] = round(
                (xs.max() - xs.min()) / max(vp.w, 1), 3)
            rec["заполнение по высоте"] = round(
                (ys.max() - ys.min()) / max(vp.h, 1), 3)
            rep.setdefault("охват", []).append(rec)
            print("  охват %-11s вьюпорт %dx%d, облако %dx%d, "
                  "поля л%d п%d в%d н%d, заполнение %.2f x %.2f"
                  % (tag or ("%dx%d" % (w, h)), vp.w, vp.h,
                     rec["облако"][0], rec["облако"][1],
                     rec["поля"][0], rec["поля"][1], rec["поля"][2],
                     rec["поля"][3], rec["заполнение по ширине"],
                     rec["заполнение по высоте"]))

        for (w, h, dpi) in ((1280, 720, 1.0), (1500, 880, 1.0),
                            (1920, 1080, 1.0), (1500, 880, 1.5),
                            (2560, 1440, 1.0)):
            fit_report(w, h, dpi)
        app.st.ws.dock().toggle("inspector")
        app.st.ws.dock().toggle("secondary")
        fit_report(1500, 880, 1.0, tag="без панелей")
        app.st.ws.dock().toggle("inspector")
        app.st.ws.dock().toggle("secondary")
        app.st.ws.dock().maximized = True
        fit_report(1920, 1080, 1.0, tag="развёрнут")
        app.st.ws.dock().maximized = False
        sh.resize(1500, 880, 1.0)
        app.frame()

        # --- замеры кадра и взаимодействия ---
        timeit("кадр целиком (анатомия)", lambda: None, runs=20)

        vp_centre = (sh.w // 2, sh.h // 2)

        def hover():
            sh.inp.mx, sh.inp.my = vp_centre[0] + np.random.randint(-60, 60), \
                vp_centre[1] + np.random.randint(-60, 60)
        timeit("наведение на область", hover)
        sh.inp.mx = sh.inp.my = -4000

        def flip_mode():
            b = app.st.brain
            b.mode = "activity" if b.mode == "anatomy" else "anatomy"
        timeit("смена режима цвета", flip_mode)
        app.st.brain.mode = "anatomy"

        reg_i = [i for i, nm in enumerate(bv.region_names)
                 if nm == "AL"]
        reg_i = reg_i[0] if reg_i else 0

        def pick():
            ws_brain._pick_region(app, reg_i, select=True)
        timeit("выделение области (AL)", pick)
        shot("brain_region_AL.png", note="выделена область AL, камера наведена")

        def rotate():
            bv.yaw += 0.05
            bv._fit_key = None
        timeit("поворот карты", rotate)

        # --- поиск ---
        ws_brain._search(app, "type:LC10")
        got = time.perf_counter()
        while app.st.brain.links_state != "x" and \
                app.st.brain.query_note.startswith("ищу") and \
                time.perf_counter() - got < 30:
            app.frame()
        rep["замеры"]["поиск type:LC10, мс"] = round(
            (time.perf_counter() - got) * 1000.0, 1)
        print("  поиск type:LC10 за %.0f мс, %s"
              % ((time.perf_counter() - got) * 1000.0,
                 app.st.brain.query_note))
        pump(frames=3)
        shot("brain_search_LC10.png", note="найдено по селектору type:LC10")

        # --- связи ---
        app.st.ws.set_tab(2)
        ws_brain._ask_links(app)
        got = time.perf_counter()
        blocked = 0.0
        while app.st.brain.links_state == "wait" and \
                time.perf_counter() - got < 120:
            a = time.perf_counter()
            app.frame()
            blocked = max(blocked, (time.perf_counter() - a) * 1000.0)
        rep["замеры"]["связи: ответ, с"] = round(time.perf_counter() - got, 1)
        rep["замеры"]["связи: худший кадр окна, мс"] = round(blocked, 1)
        print("  связи: ответ за %.1f с, худший кадр окна %.1f мс, "
              "состояние %s" % (time.perf_counter() - got, blocked,
                                app.st.brain.links_state))
        if app.st.brain.links_state == "error":
            print("  связи: " + app.st.brain.links_note)
        pump(frames=3)
        shot("brain_links.png", note="вкладка «Связи»")

        app.st.brain.links_dir = "in"
        ws_brain._ask_links(app)
        got = time.perf_counter()
        while app.st.brain.links_state == "wait" and \
                time.perf_counter() - got < 60:
            app.frame()
        rep["замеры"]["связи: второй запрос, с"] = round(
            time.perf_counter() - got, 2)
        pump(frames=3)
        shot("brain_links_in.png", note="входящие связи")

        # --- режим активности на настоящем прогоне ---
        spec = {"kind": "stim", "t": 0.5, "bin_ms": 10.0,
                "open_report": False,
                "stim": {"activate": [{"query": "class:gustatory+side:left",
                                       "rate_hz": 200.0}]}}
        job = ses.submit(spec)
        print("  прогон стимуляции:", job)
        app.st.brain.mode = "activity"
        app.st.ws.set_tab(1)
        got = time.perf_counter()
        first = None
        worst = 0.0
        while time.perf_counter() - got < 180:
            a0 = time.perf_counter()
            app.frame()
            worst = max(worst, (time.perf_counter() - a0) * 1000.0)
            if first is None and int(getattr(ses, "frames_seen", 0) or 0):
                first = time.perf_counter() - got
            if not ses.busy and first is not None:
                break
        rep["замеры"]["первый кадр активности за, с"] = (
            None if first is None else round(first, 1))
        rep["замеры"]["худший кадр окна при счёте, мс"] = round(worst, 1)
        rep["замеры"]["кадров активности"] = int(
            getattr(ses, "frames_seen", 0) or 0)
        print("  активность: первый кадр за %s с, кадров %d, "
              "худший кадр окна %.1f мс"
              % (first, int(getattr(ses, "frames_seen", 0) or 0), worst))
        lev = app.region_levels()
        if lev is None:
            print("  уровни по областям: НЕТ (act отсутствует)")
        else:
            import numpy as _np
            o = _np.argsort(-lev)[:5]
            print("  уровни по областям: " + ", ".join(
                "%s %.2f" % (app.brain.region_names[i], lev[i]) for i in o))
        pump(frames=3)
        shot("brain_activity.png", note="режим активности, настоящие кадры LIF")
        app.st.brain.mode = "anatomy"
        pump(frames=2)

        # --- щелчок по карте и переход к партнёру ---
        app.st.brain.clear_selection()
        bv.clear_selection()
        bv.fit_to(None)
        pump(frames=3)
        vp = sh.ctx.rect_of("brain/view")
        hit_at = None
        for dx in range(-200, 201, 25):
            for dy in range(-150, 151, 25):
                sh.inp.set_mouse(vp.cx + dx, vp.cy + dy, False)
                app.frame()
                if app.st.brain.hover >= 0:
                    hit_at = (vp.cx + dx, vp.cy + dy)
                    break
            if hit_at:
                break
        if hit_at:
            sh.inp.set_mouse(hit_at[0], hit_at[1], True)
            app.frame()
            sh.inp.set_mouse(hit_at[0], hit_at[1], False)
            app.frame()
        picked = app.st.brain.region
        rep["щелчок по карте выбрал нейропиль"] = (
            bv.region_names[picked] if picked >= 0 else None)
        print("  щелчок по карте: %s"
              % (bv.region_names[picked] if picked >= 0 else "НЕ СРАБОТАЛ"))
        sh.inp.set_mouse(-4000, -4000, False)
        pump(frames=2)
        shot("brain_click_region.png", note="нейропиль выбран щелчком")

        if picked >= 0:
            ws_brain._pick_region(app, picked, select=True)
            app.st.ws.set_tab(2)
            ws_brain._ask_links(app)
            got = time.perf_counter()
            while app.st.brain.links_state == "wait" and                     time.perf_counter() - got < 60:
                app.frame()
            rows = list(app.st.brain.links_rows)
            if rows:
                before = app.st.brain.sel_label
                ws_brain._goto_partner(app, rows[0])
                got = time.perf_counter()
                while app.st.brain.query_note.startswith("ищу") and                         time.perf_counter() - got < 30:
                    app.frame()
                rep["переход к партнёру"] = {
                    "из": before, "в": app.st.brain.sel_label,
                    "нейронов": app.st.brain.sel_n}
                print("  переход к партнёру: %s -> %s (%d нейронов)"
                      % (before, app.st.brain.sel_label,
                         app.st.brain.sel_n))
                pump(frames=2)
                shot("brain_goto_partner.png", note="переход к партнёру")

        # --- остальные вкладки ---
        app.st.ws.set_tab(0)
        shot("brain_tab_region.png", note="вкладка «Область»")
        app.st.ws.set_tab(1)
        shot("brain_tab_activity.png", note="вкладка «Активность» без прогона")
        app.st.ws.set_tab(3)
        shot("brain_tab_about.png", note="вкладка «Сведения»")
        app.st.ws.set_tab(0)

        # --- размеры ---
        for (w, h, dpi) in SIZES:
            shot("brain_%dx%d_dpi%d.png" % (w, h, round(dpi * 100)),
                 w, h, dpi, note="раскладка")
        sh.resize(1500, 880, 1.0)

        # --- развёрнутый вид и скрытые панели ---
        app.st.ws.dock().maximized = True
        shot("brain_max_1920x1080.png", 1920, 1080, note="развёрнутый вид")
        app.st.ws.dock().maximized = False
        sh.resize(1500, 880, 1.0)
        for z in ("inspector", "secondary"):
            app.st.ws.dock().toggle(z)
        shot("brain_no_panels.png", note="боковые панели скрыты")
        for z in ("inspector", "secondary"):
            app.st.ws.dock().toggle(z)

        # --- переключение пространств сохраняет состояние ---
        before = (round(bv.yaw, 4), round(bv.pitch, 4), round(bv.zoom, 4),
                  app.st.brain.sel_n, app.st.ws.tab("brain", 0))
        app.st.ws.switch("body")
        pump(frames=3)
        app.st.ws.switch("brain")
        pump(frames=3)
        after = (round(bv.yaw, 4), round(bv.pitch, 4), round(bv.zoom, 4),
                 app.st.brain.sel_n, app.st.ws.tab("brain", 0))
        rep["состояние переживает переключение"] = (before == after)
        print("  состояние после возврата совпало: %s" % (before == after))

        # --- память ---
        try:
            import psutil
            rep["замеры"]["память окна, МБ"] = round(
                psutil.Process().memory_info().rss / 2 ** 20, 1)
        except ImportError:
            pass
        print("итог: " + json.dumps(rep["замеры"], ensure_ascii=False))
    finally:
        ses.close()
        (OUT / "brain_shot.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("отчёт: probe/brain_shot.json")


if __name__ == "__main__":
    main()
