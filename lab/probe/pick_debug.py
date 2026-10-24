"""Почему наведение на карту мозга не находит нейропиль.

Проверка щелчка по карте не сработала: либо не находится область под
курсором, либо не доходит сам щелчок. Здесь разбирается по шагам.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\pick_debug.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen((1500, 880))
    ses = Session()
    app = App(sh, ses)
    app.st.ws.switch("brain")
    t0 = time.perf_counter()
    while app.brain is None and time.perf_counter() - t0 < 120:
        app.frame()
    bv = app.brain
    print("карта готова, нейронов %d" % bv.n)
    for _ in range(4):
        app.frame()

    vp = sh.ctx.rect_of("brain/view")
    print("вьюпорт детали:", vp, "| bv._vp:", bv._vp)
    print("mvp есть:", bv._mvp is not None, "| шаг прорежения:",
          bv._pick_step)

    hits = 0
    first = None
    for dx in range(-240, 241, 40):
        for dy in range(-180, 181, 40):
            cur = (vp.cx + dx, vp.cy + dy)
            j = bv.pick(cur, radius=18.0, active_only=False)
            r = bv.region_at(cur)
            if r is not None:
                hits += 1
                if first is None:
                    first = (cur, j, r, bv.region_names[r])
    print("из %d точек сетки область нашлась в %d"
          % (13 * 10, hits))
    print("первая:", first)

    # То же, но через кадр окна: ставим курсор и смотрим состояние.
    if first:
        cur = first[0]
        sh.inp.set_mouse(cur[0], cur[1], False)
        app.frame()
        print("после кадра: b.hover =", app.st.brain.hover,
              "| hover_neuron =", app.st.brain.hover_neuron)
        sh.inp.set_mouse(cur[0], cur[1], True)
        app.frame()
        sh.inp.set_mouse(cur[0], cur[1], False)
        app.frame()
        print("после щелчка: b.region =", app.st.brain.region,
              "" if app.st.brain.region < 0
              else bv.region_names[app.st.brain.region])
        print("перетаскивание зафиксировано:",
              getattr(app, "_brain_dragged", None))
    ses.close()


if __name__ == "__main__":
    main()
