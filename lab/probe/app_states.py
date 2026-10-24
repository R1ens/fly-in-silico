"""Состояния оболочки, которые не проверить живым прогоном.

Что здесь: скрытие и показ боковых панелей и нижней полосы, очень
длинные строки, наведение и подсказки, те же раскладки при DPI 150 и
200. Работник не поднимается — проверяется оболочка.

Подсказка показывается с задержкой, поэтому кадр рисуется дважды:
первый ставит курсор и запускает отсчёт, второй — уже с просроченной
задержкой.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\app_states.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.desk import layout as L                        # noqa: E402
from flybrain.desk.app import App                            # noqa: E402
from flybrain.desk.shell import Offscreen                    # noqa: E402
from probe.app_shot import FakeSession                       # noqa: E402

OUT = Path(__file__).parent

LONG_READOUT = ("DNa02_very_long_population_name_for_layout_test_"
                "LAL_descending_left_right")
LONG_LOG = ("работник: предупреждение — выбранная популяция для рулёжки "
            "содержит нейроны, у которых нет ни одной входящей связи "
            "внутри выбранного полушария, поэтому разница сторон может "
            "оказаться вырожденной и рулёжка не состоится")


def main():
    from PIL import Image

    sh = Offscreen((1500, 880))
    ses = FakeSession()
    # Работник «жив и готов»: иначе половина кнопок отключена, и
    # проверять наведение не на чем.
    ses.alive = True
    ses.ready = True
    app = App(sh, ses)
    app.st.worker.device = "cuda"
    app.st.worker.n_neurons = 138639
    app.st.worker.readouts = ("DNa02", LONG_READOUT, "LAL")
    app.st.readout = LONG_READOUT
    app.st.note("окно открыто")
    app.st.note(LONG_LOG, "warn")
    app.st.note("работник поднимается: коннектом")
    # путь, чтобы вкладка «Путь» была не пустой
    app.st.track = [(i * 20.0, 4.0 * np.cos(i / 9.0) + i * 0.09,
                     3.0 * np.sin(i / 7.0)) for i in range(140)]
    app.st.run.t_target = 8.0

    rows = []

    def shot(name, w=1500, h=880, dpi=1.0, hover=None, tab=None,
             twice=True):
        sh.resize(w, h, dpi)
        if tab is not None:
            app.st.ws.set_tab(tab)
        sh.t = 0.3
        app.frame()
        if hover:
            r = sh.ctx.rect_of(hover)
            if r is None:
                print("  !! нет детали " + hover)
            else:
                sh.inp.mx, sh.inp.my = r.cx, r.cy
                sh.t = 0.35
                app.frame()
                sh.t = 2.0          # задержка подсказки прошла
                app.frame()
        elif twice:
            app.frame()
        a = np.clip(sh.surf.acc.to_numpy(), 0.0, 1.0)
        Image.fromarray((a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
                        ).save(OUT / name)
        dock = app.st.ws.dock()
        lay = L.compute(sh.w, sh.h, sh.sc, dock)
        zones = [(n, getattr(lay, n)) for n in
                 ("topbar", "inspector", "centre", "secondary", "timeline",
                  "footer") if lay.has(n)]
        bad = [(x, y) for i, (x, ra) in enumerate(zones)
               for (y, rb) in zones[i + 1:] if ra.overlaps(rb)]
        rows.append((name, [n for n, _ in zones], bad))
        print("  %-38s %-52s %s" % (name, ",".join(n for n, _ in zones),
                                    "НАЛОЖЕНИЕ " + str(bad) if bad else ""))
        sh.inp.mx = sh.inp.my = -4000

    dock = app.st.ws.dock()

    print("скрытие и показ зон")
    shot("state_all_zones.png")
    dock.toggle("inspector")
    shot("state_no_inspector.png")
    dock.toggle("inspector")
    dock.toggle("secondary")
    shot("state_no_secondary.png")
    dock.toggle("secondary")
    dock.toggle("timeline")
    shot("state_no_timeline.png")
    dock.toggle("timeline")
    dock.toggle("inspector")
    dock.toggle("secondary")
    dock.toggle("timeline")
    shot("state_only_centre.png")
    for z in ("inspector", "secondary", "timeline"):
        dock.toggle(z)

    print("ширина панелей на пределах")
    dock.inspector_w = 420.0
    dock.secondary_w = 460.0
    shot("state_wide_panels.png")
    dock.inspector_w = 220.0
    dock.secondary_w = 240.0
    shot("state_narrow_panels.png")
    dock.inspector_w, dock.secondary_w = 280.0, 300.0

    print("длинные строки")
    shot("state_long_readout.png", tab=3)
    shot("state_long_log_1280.png", 1280, 720, tab=3)

    print("наведение и подсказки")
    shot("state_tip_camera.png", hover="b/cam/info", tab=0)
    shot("state_tip_zone.png", hover="zone/secondary")
    shot("state_tip_toolbar.png", hover="body/tb/reset")
    shot("state_hover_run.png", hover="tl/run")

    print("вкладки правой колонки")
    for i, nm in enumerate(("meters", "track", "vision", "log")):
        shot("state_tab_%s.png" % nm, tab=i)

    print("масштаб")
    shot("state_dpi150.png", 1500, 880, 1.5, tab=2)
    shot("state_dpi200.png", 1920, 1080, 2.0, tab=2)
    shot("state_720p.png", 1280, 720, 1.0, tab=0)

    bad = [r for r in rows if r[2]]
    print("\nкадров: %d, наложений зон: %d" % (len(rows), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
