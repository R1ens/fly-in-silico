"""Проверка порядка наложения, обрезки и отрезков.

Прежний растеризатор параллелил по командам, и пересекающиеся прямоугольники
ложились в неопределённом порядке. Попиксельные проверки этого не поймали,
потому что рисовали непересекающееся. Здесь — ровно тот случай.
"""

import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.ui import font as F                               # noqa: E402
from flybrain.ui import gpu, theme                              # noqa: E402

W, H = 600, 400
bad = 0


def check(name, cond, detail=""):
    global bad
    if not cond:
        bad += 1
    print(("[OK  ] " if cond else "[ПРОВАЛ] ") + name
          + ((": " + detail) if detail else ""))


def screen(surf):
    return np.clip(surf.acc.to_numpy(), 0, 1).transpose(1, 0, 2)[::-1]


def main():
    global bad
    ti.init(arch=ti.vulkan, offline_cache=True)
    surf = gpu.Surface()
    surf.ensure(W, H)
    atlas = F.load(13)

    # 1. Двадцать пересекающихся команд: должен остаться цвет последней.
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    for i in range(20):
        v = (i + 1) / 20.0
        dl.rect(20 + i * 4, 20 + i * 4, 320, 320, (v, 0.0, 0.0))
    surf.draw(dl, atlas)
    ti.sync()
    scr = screen(surf)
    patch = scr[300:318, 300:318, 0]
    want = 1.0
    frac = float((np.abs(patch - want) < 0.01).mean())
    check("порядок: верхняя команда побеждает",
          frac > 0.999,
          "совпало " + format(frac * 100, ".1f") + "% площади (раньше ~50%)")

    # Вдоль строки видна лестница из двадцати ступеней — каждая следующая
    # команда начинается правее. Признак правильного порядка не в том, что
    # значение одно, а в том, что оно не убывает: убывание означало бы, что
    # ранняя команда легла поверх поздней.
    row = np.round(scr[310, 24:316, 0], 3)
    drops = int((np.diff(row) < -0.001).sum())
    check("порядок: лестница не убывает", drops == 0,
          "ступеней " + str(len(np.unique(row))) + ", падений "
          + str(drops) + " (любое означало бы перепутанный порядок)")

    # 2. Обратный порядок: первая команда должна быть закрыта
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.rect(50, 50, 250, 250, (1.0, 0.0, 0.0))
    dl.rect(100, 100, 200, 200, (0.0, 1.0, 0.0))
    surf.draw(dl, atlas)
    ti.sync()
    scr = screen(surf)
    check("перекрытие: верхний цвет внутри",
          scr[150, 150, 1] > 0.9 and scr[150, 150, 0] < 0.1,
          "в середине " + str(np.round(scr[150, 150], 2).tolist()))
    check("перекрытие: нижний цвет по краю",
          scr[60, 60, 0] > 0.9 and scr[60, 60, 1] < 0.1,
          "в углу " + str(np.round(scr[60, 60], 2).tolist()))

    # 3. Полупрозрачность складывается по порядку
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.rect(0, 0, 100, 100, (1.0, 1.0, 1.0), alpha=0.5)
    dl.rect(0, 0, 100, 100, (1.0, 1.0, 1.0), alpha=0.5)
    surf.draw(dl, atlas)
    ti.sync()
    v = float(screen(surf)[50, 50, 0])
    check("две полупрозрачные ложатся как 0,75", abs(v - 0.75) < 0.01,
          "получили " + format(v, ".3f"))

    # 4. Обрезка
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    prev = dl.push_clip(100, 100, 200, 200)
    dl.rect(50, 50, 300, 300, (1.0, 1.0, 1.0))
    dl.text(atlas, "обрезано", 60, 150, (1.0, 1.0, 1.0))
    dl.pop_clip(prev)
    dl.rect(400, 50, 450, 100, (0.0, 0.0, 1.0))
    surf.draw(dl, atlas)
    ti.sync()
    scr = screen(surf)
    white = scr[:, :, 0] > 0.5
    ys, xs = np.nonzero(white)
    check("обрезка: заливка не вышла за рамку",
          xs.min() >= 100 and xs.max() < 200
          and ys.min() >= 100 and ys.max() < 200,
          "x " + str((int(xs.min()), int(xs.max()))) + " y "
          + str((int(ys.min()), int(ys.max()))) + ", ждали [100, 200)")
    check("обрезка снята для следующей команды", scr[70, 420, 2] > 0.5,
          "синий прямоугольник " + ("на месте" if scr[70, 420, 2] > 0.5 else "пропал"))

    # 5. Отрезок
    surf.clear((0.0, 0.0, 0.0))
    dl = gpu.DrawList()
    dl.seg(100, 100, 300, 200, (1.0, 1.0, 1.0), width=3.0)
    surf.draw(dl, atlas)
    ti.sync()
    scr = screen(surf)
    lit = scr[:, :, 0] > 0.3
    ys, xs = np.nonzero(lit)
    on_line = np.abs((xs - 100) * 100 - (ys - 100) * 200) / (200 * 200 + 100 * 100) ** .5
    check("отрезок идёт по прямой", float(on_line.max()) < 3.0,
          "наибольшее отклонение " + format(float(on_line.max()), ".2f") + " px")
    check("отрезок в своих границах",
          xs.min() >= 97 and xs.max() <= 303 and ys.min() >= 97 and ys.max() <= 203,
          "x " + str((int(xs.min()), int(xs.max()))) + " y "
          + str((int(ys.min()), int(ys.max()))))
    edge = scr[:, :, 0]
    soft = int(((edge > 0.05) & (edge < 0.95)).sum())
    check("край отрезка мягкий", soft > 200,
          str(soft) + " полупрозрачных пикселей по краю")

    # 6. Ломаная: по команде на звено
    dl = gpu.DrawList()
    pts = np.stack([np.linspace(20, 580, 300),
                    200 + 80 * np.sin(np.linspace(0, 8, 300))], axis=1)
    dl.polyline(pts, theme.ACCENT, width=2.0)
    check("ломаная: команд по звеньям", len(dl.cmds) == 299,
          str(len(dl.cmds)) + " команд на 300 точек "
          + "(раньше путь выкладывался точками)")
    surf.clear(theme.BG)
    surf.draw(dl, atlas)
    ti.sync()
    from PIL import Image
    Image.fromarray((screen(surf) * 255).astype(np.uint8)).save(
        Path(__file__).parent / "order_line.png")

    # 7. Цена кадра на нагрузке, похожей на настоящую панель
    surf.ensure(1500, 880)
    big = gpu.DrawList()
    big.rect(0, 0, 330, 880, theme.PANEL)
    big.rect(1180, 0, 1500, 880, theme.PANEL)
    for i in range(40):
        big.rect(16, 80 + i * 19, 314, 97 + i * 19, theme.PANEL_HI)
        big.frame(16, 80 + i * 19, 314, 97 + i * 19, theme.HAIRLINE)
        big.text(atlas, "область " + str(i) + " · 123,4 Гц", 22, 80 + i * 19,
                 theme.INK)
    for i in range(6):
        pts = np.stack([np.linspace(1190, 1490, 200),
                        120 + i * 120 + 40 * np.sin(np.linspace(0, 9, 200) + i)],
                       axis=1)
        big.polyline(pts, theme.ACCENT, width=1.6)
    nshape, nglyph = len(big.cmds), len(big.gi)

    for _ in range(6):
        surf.clear(theme.BG)
        surf.draw(big, atlas)
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(40):
        surf.clear(theme.BG)
        surf.draw(big, atlas)
    ti.sync()
    ms = (time.perf_counter() - t0) * 1000 / 40
    print("\nпанель 1500×880: " + format(ms, ".2f") + " мс ("
          + format(1000 / ms, ".0f") + " кадр/с) при " + str(nshape)
          + " фигурах и " + str(nglyph) + " глифах, запусков ядра 2")

    t0 = time.perf_counter()
    for _ in range(40):
        big._bin(1500, 880)
    bin_ms = (time.perf_counter() - t0) * 1000 / 40
    print("разбор по плиткам: " + format(bin_ms, ".2f")
          + " мс (раньше кройка стоила 1,9 мс)")

    print("\nпровалов: " + str(bad))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
