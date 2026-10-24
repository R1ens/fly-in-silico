"""Кадр приложения без окна, в заданных размерах.

Окно в этой среде самовольно меняет размер — за один сеанс оно побывало
1500×880 и 257×1824, — поэтому судить о виде по снимку живого окна нельзя.
Здесь кадр собирается в тот размер, который задан, и пишется картинкой.

Работник поднимается настоящий, и задача гоняется настоящая: нарисовать
пустую панель мало, смотреть надо на ту, где идут числа.
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
SIZES = ((1945, 984), (1500, 880), (803, 984))


def shot(surf, path):
    from PIL import Image

    a = np.clip(surf.acc.to_numpy(), 0.0, 1.0)
    Image.fromarray((a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)).save(path)


def pump_until(sess, cond, limit, fps=60.0):
    t0 = last = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        now = time.perf_counter()
        sess.pump(now - last)
        last = now
        if cond(sess):
            return True
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - now)))
    return False


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stim"
    ti.init(arch=ti.vulkan, offline_cache=True)
    surf = gpu.Surface()
    ui = kit.Ui()

    sess = Session()
    print("жду работника…")
    pump_until(sess, lambda s: s.ready or s.error, 180)
    print("готов: " + str(sess.n) + " нейронов, " + str(sess.device))

    scene = app.Scene(sess)
    scene.st.mode = mode
    if mode == "body":
        scene.st.t_body = 3.0
    scene.ensure_map()

    spec = app._spec(scene.st)
    sess.submit(spec)
    print("задача: " + str(spec))

    # Копим, пока не наберётся активность, и рисуем на ходу.
    need = 6 if mode == "body" else 30
    t0 = last = time.perf_counter()
    while time.perf_counter() - t0 < 240 and sess.frames_seen <= need:
        now = time.perf_counter()
        sess.pump(now - last)
        last = now
        # Путь копится в состоянии панели, поэтому панель надо звать по ходу,
        # а не только в конце.
        if mode == "body" and sess.extra:
            from flybrain.ui import panels as _p
            st = scene.st
            if sess.extra.get("t_ms") != st.track_t:
                st.track_t = sess.extra.get("t_ms")
                st.track.append((sess.extra.get("x", 0.0),
                                 sess.extra.get("y", 0.0)))
        time.sleep(1.0 / 120.0)
    print("кадров получено: " + str(sess.frames_seen)
          + ", активных " + str(sess.n_active))

    for w, h in SIZES:
        surf.ensure(w, h)
        ui.input.feed(w * 0.62, h * 0.45, False)
        ui.begin(w, h)
        surf.clear(theme.BG)
        sess.pump(1.0 / 60.0)
        lay = scene.draw(ui, surf, 60.0)
        ui.flush(surf)
        ti.sync()
        name = "app_" + mode + "_" + str(w) + "x" + str(h) + ".png"
        shot(surf, OUT / name)
        print("  " + name + "  команд " + str(ui.commands())
              + ", масштаб " + format(ui.scale, ".2f")
              + (("  [" + "; ".join(lay.notes) + "]") if lay.notes else ""))

    sess.close()


if __name__ == "__main__":
    main()
