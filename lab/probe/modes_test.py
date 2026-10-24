"""Раскладка рейки во всех режимах, без счёта.

Прогонять зонд или серию ради картинки незачем — это минуты. Здесь
поднимается настоящий работник (списки съёмов и популяций нужны настоящие),
а сам прогон не запускается: смотрим только на то, как лежат органы
управления в каждом режиме и в каждом размере окна.
"""

import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.ui import app, gpu, kit, theme                    # noqa: E402
from flybrain.ui.session import Session                         # noqa: E402

OUT = Path(__file__).parent
SIZES = ((1945, 984), (803, 984))
MODES = ("stim", "arena", "body", "lateral", "demos", "agent", "runs")


def shot(surf, path):
    from PIL import Image

    a = np.clip(surf.acc.to_numpy(), 0.0, 1.0)
    Image.fromarray((a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)).save(path)


def main():
    ti.init(arch=ti.vulkan, offline_cache=True)
    surf = gpu.Surface()
    ui = kit.Ui()
    sess = Session()
    t0 = last = time.perf_counter()
    while time.perf_counter() - t0 < 180 and not (sess.ready or sess.error):
        now = time.perf_counter()
        sess.pump(now - last)
        last = now
        time.sleep(1.0 / 120.0)
    print("работник: " + ("готов" if sess.ready else "НЕ ПОДНЯЛСЯ"))

    scene = app.Scene(sess)
    scene.ensure_map()

    for w, h in SIZES:
        surf.ensure(w, h)
        for mode in MODES:
            scene.st.mode = mode
            scene.st.want_run = False
            ui.input.feed(-50, -50, False)          # курсор вне окна
            ui.begin(w, h)
            surf.clear(theme.BG)
            sess.pump(1.0 / 60.0)
            lay = scene.draw(ui, surf, 60.0)
            scene.st.want_run = False               # ничего не запускаем
            ui.flush(surf)
            ti.sync()
            name = f"mode_{mode}_{w}x{h}.png"
            shot(surf, OUT / name)
            print("  " + name.ljust(28) + " команд " + str(ui.commands())
                  + (("  [" + "; ".join(lay.notes) + "]") if lay.notes else ""))

    sess.close()


if __name__ == "__main__":
    main()
