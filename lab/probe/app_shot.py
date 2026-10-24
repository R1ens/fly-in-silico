"""Снимки нового окна без окна: все требуемые размеры и масштабы.

Проверяет то, что видно глазом и числом сразу: раскладку зон, поведение
при изменении размера, отключение зон на узком окне, разворот центра,
масштаб DPI. Работник не поднимается — здесь проверяется оболочка, а не
счёт; живой прогон проверяет `probe/app_live.py`.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\app_shot.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.desk import layout as L                        # noqa: E402
from flybrain.desk.app import App                            # noqa: E402
from flybrain.desk.shell import Offscreen                    # noqa: E402

OUT = Path(__file__).parent
CASES = (
    (1280, 720, 1.0, "body", {}),
    (1500, 880, 1.0, "body", {}),
    (1920, 1080, 1.0, "body", {}),
    (1920, 1080, 1.5, "body", {}),
    (2560, 1440, 2.0, "body", {}),
    (1500, 880, 1.0, "body", {"max": True}),
    (1500, 880, 1.0, "body", {"tab": 2}),
    (1500, 880, 1.0, "body", {"tab": 1}),
    (1100, 700, 1.0, "body", {"narrow": True}),
    (1500, 880, 1.0, "brain", {}),
    (1500, 880, 1.0, "experiment", {}),
)


class FakeSession:
    """Работник, которого нет: оболочка должна рисоваться и без него."""

    alive = False
    ready = False
    hello = None
    error = None
    job_id = None
    state = "idle"
    phase = ""
    progress = 0.0
    t_ms = 0.0
    boot_phase = "коннектом"
    boot_progress = 0.35
    summary = None
    body_frame = None
    body_seq = 0
    extra = {}
    series = {}
    log = []

    def pump(self, dt=None):
        return []

    def ask(self, what, arg=None):
        return ""

    def submit(self, spec):
        return None

    def cancel(self):
        pass

    def n_active(self):
        return 0


def main():
    from PIL import Image

    sh = Offscreen((1500, 880))
    app = App(sh, FakeSession())
    app.st.worker.readouts = ("DNa02", "DNg13", "LAL")
    app.st.readout = "DNa02"
    app.st.note("окно открыто")
    app.st.note("работник поднимается: коннектом")
    rows = []
    for (w, h, dpi, space, opt) in CASES:
        sh.resize(w, h, dpi)
        app.st.ws.switch(space)
        dock = app.st.ws.dock()
        dock.maximized = bool(opt.get("max"))
        if "tab" in opt:
            app.st.ws.set_tab(opt["tab"])
        t0 = time.perf_counter()
        app.frame()
        ms = (time.perf_counter() - t0) * 1000.0

        lay = L.compute(sh.w, sh.h, sh.sc, dock)
        zones = [(n, getattr(lay, n)) for n in
                 ("topbar", "inspector", "centre", "secondary", "timeline",
                  "footer") if lay.has(n)]
        bad = [(a, b) for i, (a, ra) in enumerate(zones)
               for (b, rb) in zones[i + 1:] if ra.overlaps(rb)]
        tag = "%dx%d_dpi%d_%s%s" % (w, h, round(dpi * 100), space,
                                    ("_" + "_".join(map(str, opt))
                                     if opt else ""))
        name = "app_" + tag + ".png"
        a = np.clip(sh.surf.acc.to_numpy(), 0.0, 1.0)
        Image.fromarray((a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
                        ).save(OUT / name)
        rows.append((name, ms, sh.launches, [n for n, _ in zones], bad))
        print("  %-42s %5.1f мс  %2d запусков  зоны: %s%s"
              % (name, ms, sh.launches, ",".join(n for n, _ in zones),
                 "  НАЛОЖЕНИЕ " + str(bad) if bad else ""))
    fails = [r for r in rows if r[4]]
    print("\nснимков: %d, наложений: %d" % (len(rows), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
