"""Охват полного коннектома: помещается ли облако в центральную область.

Считается по кадру, а не по формуле: берётся отличие от цвета фона, так
что подложка панели в измерение не попадает. Нужны два числа — поля по
краям (облако не должно упираться в край) и заполнение (облако не должно
болтаться маленьким пятном).

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\fit_check.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
CASES = ((1280, 720, 1.0, ""), (1500, 880, 1.0, ""),
         (1920, 1080, 1.0, ""), (1500, 880, 1.5, "dpi 1,5"),
         (2560, 1440, 1.0, ""), (1100, 700, 1.0, "узкое"))


def main():
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.desk.tokens import C
    from flybrain.ui.session import Session

    sh = Offscreen((1500, 880))
    ses = Session()
    app = App(sh, ses)
    app.st.ws.switch("brain")
    t0 = time.perf_counter()
    while app.brain is None and time.perf_counter() - t0 < 120:
        app.frame()
    bv = app.brain
    print("карта готова за %.1f с" % (time.perf_counter() - t0))

    bg = np.asarray(C.bg, dtype=np.float32)
    rows = []
    bad = []

    def measure(w, h, dpi, tag):
        sh.resize(w, h, dpi)
        bv.fit_to(None)
        bv.zoom = 1.0
        app.frame()
        app.frame()
        vp = sh.ctx.rect_of("brain/view")
        # Меряем саму геометрию проекции, а не пиксели: в кадре есть и
        # полоса инструментов, и подписи, и они сбивали замер на край.
        mvp = bv._mvp
        pts = bv.pos[::7]
        wq = pts @ mvp[3, :3] + mvp[3, 3]
        ok = wq > 1e-6
        pts, wq = pts[ok], wq[ok]
        sx = vp.x + ((pts @ mvp[0, :3] + mvp[0, 3]) / wq * 0.5 + 0.5) * vp.w
        sy = vp.y + (1.0 - ((pts @ mvp[1, :3] + mvp[1, 3]) / wq * 0.5
                            + 0.5)) * vp.h
        # Края по процентилю: десяток выбросов не должен решать, что
        # «облако упирается в край».
        lo_x, hi_x = np.percentile(sx, (0.2, 99.8))
        lo_y, hi_y = np.percentile(sy, (0.2, 99.8))
        rec = {
            "случай": (tag + " " if tag else "") + "%dx%d" % (w, h),
            "вьюпорт": [vp.w, vp.h],
            "облако": [int(hi_x - lo_x), int(hi_y - lo_y)],
            "поля": [int(lo_x - vp.x), int(vp.x1 - hi_x),
                     int(lo_y - vp.y), int(vp.y1 - hi_y)],
            "заполнение": [round(float(hi_x - lo_x) / vp.w, 3),
                           round(float(hi_y - lo_y) / vp.h, 3)],
            "совсем за краем, %": round(
                100.0 * float(((sx < vp.x) | (sx > vp.x1) | (sy < vp.y)
                               | (sy > vp.y1)).mean()), 3),
        }
        rows.append(rec)
        print("  %-20s вьюпорт %4dx%-4d облако %4dx%-4d  поля л%3d п%3d "
              "в%3d н%3d  заполнение %.2f x %.2f  за краем %.2f %%"
              % (rec["случай"], vp.w, vp.h, rec["облако"][0],
                 rec["облако"][1], *rec["поля"], *rec["заполнение"],
                 rec["совсем за краем, %"]))
        m_ = min(rec["поля"])
        if m_ < max(6, int(0.015 * min(vp.w, vp.h))):
            bad.append((rec["случай"], "мало поля: %d px" % m_))
        if max(rec["заполнение"]) < 0.5:
            bad.append((rec["случай"], "облако мелкое: %.2f"
                        % max(rec["заполнение"])))
        if rec["совсем за краем, %"] > 0.5:
            bad.append((rec["случай"], "за краем %.2f %% точек"
                        % rec["совсем за краем, %"]))

    for (w, h, dpi, tag) in CASES:
        measure(w, h, dpi, tag)

    dock = app.st.ws.dock()
    dock.toggle("inspector")
    dock.toggle("secondary")
    measure(1500, 880, 1.0, "без панелей")
    dock.toggle("inspector")
    dock.toggle("secondary")
    dock.maximized = True
    measure(1920, 1080, 1.0, "развёрнут")
    dock.maximized = False

    (OUT / "fit_check.json").write_text(
        json.dumps({"случаи": rows, "замечания": bad}, ensure_ascii=False,
                   indent=1), encoding="utf-8")
    print("\nслучаев %d, замечаний %d" % (len(rows), len(bad)))
    for t, why in bad:
        print("  - %s: %s" % (t, why))
    ses.close()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
