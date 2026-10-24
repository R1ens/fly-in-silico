"""Почему «Верхние области» пустые: что на самом деле в следе активности.

Вкладка активности печатала нули по всем областям при 678 нейронах со
следом и пике 127 Гц. Здесь считается то же самое, но с числами на
каждом шаге, чтобы стало видно, где теряется.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\levels_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from flybrain.ui.session import Session

    ses = Session()
    t0 = time.perf_counter()
    while not ses.ready and time.perf_counter() - t0 < 120:
        ses.pump(dt=0.01)
    print("работник готов за %.1f с, нейронов %d"
          % (time.perf_counter() - t0, ses.n))

    g = ses.geometry()
    pos, code, names, rgb, _c = g
    counts = np.bincount(code, minlength=len(names))
    print("нейропилей %d, сумма по областям %d" % (len(names), counts.sum()))

    job = ses.submit({"kind": "stim", "t": 0.5, "bin_ms": 10.0,
                      "open_report": False,
                      "stim": {"activate": [
                          {"query": "class:gustatory+side:left",
                           "rate_hz": 200.0}]}})
    print("прогон", job)
    t1 = time.perf_counter()
    snaps = []
    while time.perf_counter() - t1 < 180:
        ses.pump(dt=1 / 60.0)
        if ses.act is not None and ses.frames_seen and len(snaps) < 3 \
                and ses.busy:
            snaps.append((ses.frames_seen, float(ses.act.sum()),
                          float(ses.act.max()), int((ses.act > 0.5).sum())))
        if not ses.busy and ses.frames_seen:
            break
        time.sleep(0.002)
    print("кадров %d, состояние %s" % (ses.frames_seen, ses.state))
    for k, s, mx, na in snaps:
        print("  во время счёта: кадр %d, сумма следа %.0f, пик %.0f, "
              "со следом %d" % (k, s, mx, na))

    a = ses.act
    print("после конца: сумма следа %.1f, пик %.1f, со следом %d"
          % (float(a.sum()), float(a.max()), int((a > 0.5).sum())))
    lev = ses.region_levels(code, len(names), counts=counts)
    order = np.argsort(-lev)[:8]
    print("верхние области по среднему следу:")
    for i in order:
        s = float(np.asarray(a)[code == i].sum())
        print("  %-6s среднее %8.4f Гц   сумма %9.1f   нейронов %6d   "
              "со следом %5d" % (names[i], lev[i], s, counts[i],
                                 int((np.asarray(a)[code == i] > 0.5).sum())))
    # Та же величина, но по тем, кто вообще сработал: средняя по всей
    # области топится в тысячах молчащих нейронов.
    print("то же, но среднее по сработавшим:")
    for i in order:
        m = (code == i) & (np.asarray(a) > 0.5)
        if m.any():
            print("  %-6s %.1f Гц по %d нейронам"
                  % (names[i], float(np.asarray(a)[m].mean()), int(m.sum())))
    ses.close()


if __name__ == "__main__":
    main()
