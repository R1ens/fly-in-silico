"""Попиксельная проверка примитивов gpu.py.

Рисуем прямоугольники с заранее известными границами и сверяем поле. Любая
ошибка в нарезке на плитки или в перевороте координат видна как расхождение,
а не как «что-то не так выглядит».
"""

import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.ui import font as F                               # noqa: E402
from flybrain.ui import gpu, theme                              # noqa: E402

W, H = 400, 300
bad = 0


def check(name, cond, detail=""):
    global bad
    if not cond:
        bad += 1
    print(("[OK  ] " if cond else "[FAIL] ") + name + ((": " + detail) if detail else ""))


def main():
    global bad
    ti.init(arch=ti.vulkan, offline_cache=True)
    surf = gpu.Surface()
    surf.ensure(W, H)
    atlas = F.load(14)

    # 1. один прямоугольник: границы ровно те, что заданы
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.rect(10, 20, 110, 70, (1.0, 0.0, 0.0))
    surf.draw(dl, atlas)
    ti.sync()
    a = surf.acc.to_numpy()                      # (W, H, 3), y снизу вверх
    scr = a.transpose(1, 0, 2)[::-1]             # (H, W, 3), y сверху вниз
    red = scr[:, :, 0] > 0.5
    ys, xs = np.nonzero(red)
    check("прямоугольник: границы",
          xs.min() == 10 and xs.max() == 109 and ys.min() == 20 and ys.max() == 69,
          "x " + str((int(xs.min()), int(xs.max()))) + " y "
          + str((int(ys.min()), int(ys.max()))) + ", ждали x (10, 109) y (20, 69)")
    check("прямоугольник: ничего лишнего", int(red.sum()) == 100 * 50,
          str(int(red.sum())) + " пикселей, ждали 5000")

    # 2. прямоугольник шире плитки: нарезка не теряет и не добавляет
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.rect(5, 7, 5 + 213, 7 + 151, (0.0, 1.0, 0.0))
    surf.draw(dl, atlas)
    ti.sync()
    scr = surf.acc.to_numpy().transpose(1, 0, 2)[::-1]
    g = scr[:, :, 1] > 0.5
    ys, xs = np.nonzero(g)
    check("нарезка на плитки",
          int(g.sum()) == 213 * 151 and xs.min() == 5 and xs.max() == 5 + 212
          and ys.min() == 7 and ys.max() == 7 + 150,
          str(int(g.sum())) + " пикселей, ждали " + str(213 * 151)
          + "; x " + str((int(xs.min()), int(xs.max())))
          + " y " + str((int(ys.min()), int(ys.max()))))

    # 3. прямоугольник, вылезающий за край: обрезается, не заворачивается
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.rect(W - 30, H - 20, W + 90, H + 60, (0.0, 0.0, 1.0))
    surf.draw(dl, atlas)
    ti.sync()
    scr = surf.acc.to_numpy().transpose(1, 0, 2)[::-1]
    b = scr[:, :, 2] > 0.5
    ys, xs = np.nonzero(b)
    check("обрезка по краю",
          int(b.sum()) == 30 * 20 and xs.min() == W - 30 and ys.min() == H - 20,
          str(int(b.sum())) + " пикселей, ждали 600; начало "
          + str((int(xs.min()), int(ys.min()))))

    # 4. альфа смешивает, а не заменяет
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.rect(0, 0, 50, 50, (1.0, 1.0, 1.0), alpha=0.25)
    surf.draw(dl, atlas)
    ti.sync()
    v = float(surf.acc.to_numpy()[10, H - 1 - 10, 0])
    check("альфа", abs(v - 0.25) < 1e-4, "получили " + format(v, ".4f") + ", ждали 0.2500")

    # 5. рамка: четыре стороны по одному пикселю, середина пустая
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.frame(20, 20, 120, 80, (1.0, 1.0, 1.0))
    surf.draw(dl, atlas)
    ti.sync()
    scr = surf.acc.to_numpy().transpose(1, 0, 2)[::-1]
    f = scr[:, :, 0] > 0.5
    perimeter = 100 * 2 + 60 * 2 - 4
    check("рамка", int(f.sum()) == perimeter and not f[50, 70],
          str(int(f.sum())) + " пикселей, ждали " + str(perimeter)
          + "; середина " + ("пустая" if not f[50, 70] else "ЗАЛИТА"))

    # 6. текст не выходит за свою строку и стоит правильной стороной
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.text(atlas, "Ag", 40, 60, (1.0, 1.0, 1.0))
    surf.draw(dl, atlas)
    ti.sync()
    scr = surf.acc.to_numpy().transpose(1, 0, 2)[::-1]
    t = scr[:, :, 0] > 0.2
    ys, xs = np.nonzero(t)
    width = atlas.width("Ag")
    check("текст: внутри своей коробки",
          ys.min() >= 60 and ys.max() < 60 + atlas.gh
          and xs.min() >= 40 and xs.max() < 40 + width,
          "x " + str((int(xs.min()), int(xs.max()))) + " в [40, "
          + str(40 + width) + "), y " + str((int(ys.min()), int(ys.max())))
          + " в [60, " + str(60 + atlas.gh) + ")")
    # у прописной A перекладина ниже середины, у строчной g хвост уходит вниз:
    # если строка перевёрнута, центр тяжести уедет вверх
    rows = t.sum(axis=1)
    com = float((np.arange(H) * rows).sum() / max(rows.sum(), 1)) - 60
    check("текст: не перевёрнут", com > atlas.gh * 0.30,
          "центр тяжести на " + format(com, ".1f") + " из " + str(atlas.gh))

    # 7. картинка ложится в заданный прямоугольник
    surf.clear((0.0, 0.0, 0.0))
    src = np.zeros((40, 60, 3), dtype=np.uint8)
    src[:, :, 0] = 255
    src[0, :, :] = (0, 255, 0)          # верхняя строка источника — зелёная
    dl = gpu.DrawList()
    dl.image(src, 100, 50, 120, 80)
    surf.draw(dl, atlas)
    ti.sync()
    scr = surf.acc.to_numpy().transpose(1, 0, 2)[::-1]
    top_green = scr[50, 160, 1] > 0.5
    bottom_red = scr[50 + 79, 160, 0] > 0.5
    check("картинка: верх источника наверху", top_green and bottom_red,
          "верх зелёный " + str(bool(top_green)) + ", низ красный " + str(bool(bottom_red)))

    # 8. сплаты попадают во вьюпорт и не вылезают
    surf.clear((0.0, 0.0, 0.0))
    rng = np.random.default_rng(0)
    pos = np.ascontiguousarray((rng.random((20000, 3)).astype(np.float32) - 0.5) * 1.6)
    act = np.ascontiguousarray(rng.random(20000).astype(np.float32))
    col = np.ascontiguousarray(np.ones((20000, 3), dtype=np.float32))
    mvp, _ = gpu.orbit_mvp(0.3, 0.2, 2.2, 200 / 150)
    surf.splat(pos, act, col, mvp, (150, 100, 200, 150))
    ti.sync()
    scr = surf.acc.to_numpy().transpose(1, 0, 2)[::-1]
    lit = scr[:, :, 0] > 0.01
    ys, xs = np.nonzero(lit)
    check("сплаты во вьюпорте",
          xs.min() >= 150 and xs.max() < 350 and ys.min() >= 100 and ys.max() < 250,
          "x " + str((int(xs.min()), int(xs.max()))) + " в [150, 350), y "
          + str((int(ys.min()), int(ys.max()))) + " в [100, 250)")

    # 9. цена кадра: та же сцена, но без счёта на CPU
    surf.ensure(1500, 880)
    big = gpu.DrawList()
    big.rect(0, 0, 300, 880, theme.PANEL)
    big.rect(1220, 0, 1500, 880, theme.PANEL)
    for i in range(40):
        big.rect(16, 100 + i * 18, 284, 114 + i * 18, theme.PANEL_HI)
        big.text(atlas, "область " + str(i) + " · 123,4 Гц", 22, 100 + i * 18,
                 theme.INK)
    for i in range(30):
        big.text(atlas, "строка показаний " + str(i), 1236, 40 + i * 20, theme.DIM)
    nglyph = len(big.gi)
    nrect = len(big.rects)

    pos2 = np.ascontiguousarray((rng.random((138625, 3)).astype(np.float32) - .5) * 1.6)
    act2 = np.ascontiguousarray((rng.random(138625) ** 3).astype(np.float32))
    col2 = np.ascontiguousarray(rng.random((138625, 3)).astype(np.float32))
    mvp2, _ = gpu.orbit_mvp(0.5, 0.2, 2.9, 904 / 848)

    def one():
        surf.clear(theme.BG)
        surf.splat(pos2, act2, col2, mvp2, (316, 16, 904, 848))
        surf.draw(big, atlas)

    for _ in range(8):
        one()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(50):
        one()
    ti.sync()
    ms = (time.perf_counter() - t0) * 1000 / 50
    print("\nкадр 1500×880: " + format(ms, ".2f") + " мс ("
          + format(1000 / ms, ".0f") + " кадр/с) при " + str(nrect)
          + " прямоугольниках, " + str(nglyph) + " глифах, 138 625 сплатах")

    print("\nпровалов: " + str(bad))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
