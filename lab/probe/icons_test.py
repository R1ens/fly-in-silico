"""Проверка набора иконок без окна.

Иконку нельзя принять по числу команд и по тому, что вызов не упал: она либо
читается при 16 px, либо нет. Поэтому здесь два снимка — таблица всех иконок
в трёх размерах на фоне панели и увеличение каждой при 16 px в четыре раза
без сглаживания, — а в консоль печатаются цена иконки в командах и доля
полупрозрачных пикселей.

Увеличение делается самим растеризатором: `k_image` берёт ближайший пиксель,
то есть ровно то, что нужно, — PIL с интерполяцией замазал бы как раз те
полутона, которые мы и ищем.

Мера мыла считается не по всем полутонам: при толщине 1,5 у любого честного
штриха есть два края по 0,25, и считать их браком значит мерить не то.
Браком считается середина — от 0,35 до 0,85: столько набирает пиксель,
когда штрих разложился поровну на две строки, то есть когда ось уехала с
нужной фазы. У прямых штрихов на своей фазе таких пикселей нет вовсе, у
дуг и диагоналей они неизбежны.
"""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.desk import icons                                   # noqa: E402
from flybrain.ui import font as F                                 # noqa: E402
from flybrain.ui import gpu, theme                                # noqa: E402

W, H = 980, 660
OUT = Path(__file__).parent

SIZES = (16, 20, 24)
CELL = 24                 # клетка бесцветной сетки для замера и увеличения
PAD = 2                   # поля клетки: в них видно, не вылезла ли иконка
COLS = 7                  # столбцов в обоих снимках
ZOOM = 4


def screen(surf):
    return np.clip(surf.acc.to_numpy(), 0, 1).transpose(1, 0, 2)[::-1]


def save(img, name):
    from PIL import Image

    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT / name)
    print("записан " + str(OUT / name))


def main():
    ti.init(arch=ti.vulkan, offline_cache=True)
    body = F.load(13)
    small = F.load(11)
    surf = gpu.Surface()
    surf.ensure(W, H)

    names = icons.NAMES
    rows = (len(names) + COLS - 1) // COLS

    # ---- снимок для замера и увеличения: иконки в бесцветной сетке -------
    gw, gh = COLS * CELL + 2 * PAD, rows * CELL + 2 * PAD
    surf.clear(theme.BG)
    dl = gpu.DrawList()
    dl.rect(0, 0, gw, gh, theme.PANEL)
    for i, name in enumerate(names):
        cx = PAD + (i % COLS) * CELL + (CELL - 16) // 2
        cy = PAD + (i // COLS) * CELL + (CELL - 16) // 2
        icons.draw(dl, name, cx, cy, 16, theme.INK)
    surf.draw(dl, body)
    ti.sync()
    grid = screen(surf)[:gh, :gw]

    # ---- цена и чёткость --------------------------------------------------
    panel = np.array(theme.PANEL, dtype=np.float32)
    ink = np.array(theme.INK, dtype=np.float32)
    print("иконка           команд 16/20/24   пикселей  середины  вылезло")
    tiles, soft_all, out_all = [], [], 0
    for i, name in enumerate(names):
        counts = []
        for s in SIZES:
            probe = gpu.DrawList()
            icons.draw(probe, name, 0, 0, s, theme.INK)
            counts.append(len(probe.cmds))
        x0 = PAD + (i % COLS) * CELL + (CELL - 16) // 2 - PAD
        y0 = PAD + (i // COLS) * CELL + (CELL - 16) // 2 - PAD
        tile = grid[y0:y0 + 16 + 2 * PAD, x0:x0 + 16 + 2 * PAD]
        tiles.append(tile)

        t = ((tile - panel) / (ink - panel)).mean(axis=2)
        lit = t > 0.04
        mid = lit & (t > 0.35) & (t < 0.85)
        share = float(mid.sum()) / max(1, int(lit.sum()))
        soft_all.append(share)
        # Поля клетки: иконка обязана уместиться в свой квадрат, иначе в
        # ряду кнопок она сдвинет соседей.
        edge = lit.copy()
        edge[PAD:-PAD, PAD:-PAD] = False
        out_all += int(edge.sum())
        print(name.ljust(16) + " "
              + "/".join(str(c).rjust(3) for c in counts) + "   "
              + str(int(lit.sum())).rjust(8) + "  "
              + format(share * 100, ".0f").rjust(7) + " %"
              + str(int(edge.sum())).rjust(8))
    print("середины в среднем: "
          + format(float(np.mean(soft_all)) * 100, ".0f") + " %, хуже всех — "
          + names[int(np.argmax(soft_all))] + " ("
          + format(max(soft_all) * 100, ".0f") + " %)")
    print("за квадрат 16×16 вылезло пикселей: " + str(out_all))

    # ---- снимок 1: таблица трёх размеров на панели ------------------------
    ew, rh, tcols = 228, 50, 4
    trows = (len(names) + tcols - 1) // tcols
    tw, th = tcols * ew + 24, trows * rh + 56
    surf.clear(theme.BG)
    dl = gpu.DrawList()
    dl.rect(0, 0, tw, th, theme.PANEL)
    dl.text(body, "иконки · 16 / 20 / 24 логических пикселя · штрих "
            + ", ".join(format(icons.stroke_for(s), ".2f").replace(".", ",")
                        for s in SIZES), 12, 12, theme.DIM)
    dl.hline(12, tw - 12, 34, theme.HAIRLINE)
    for i, name in enumerate(names):
        col, row = i // trows, i % trows
        x = 12 + col * ew
        y = 44 + row * rh
        dl.text(small, name, x, y + 14, theme.DIM)
        bx = x + 92
        for s in SIZES:
            dl.rect(bx, y + 2, bx + s + 12, y + 2 + s + 12, theme.PANEL_HI)
            icons.draw(dl, name, bx + 6, y + 8, s,
                       theme.ACCENT if name == "record" else theme.INK)
            bx += s + 16
    surf.draw(dl, body)
    ti.sync()
    save(screen(surf)[:th, :tw], "icons_table.png")

    # ---- снимок 2: увеличение в четыре раза без сглаживания --------------
    side = (16 + 2 * PAD) * ZOOM
    bw, bh = side + 14, side + 26
    zw, zh = COLS * bw + 24, rows * bh + 46
    surf.clear(theme.BG)
    dl = gpu.DrawList()
    dl.rect(0, 0, zw, zh, theme.PANEL)
    dl.text(body, "16 px, увеличено в 4 раза без сглаживания: видно фазу "
            "штриха и мыло", 12, 12, theme.DIM)
    for i, name in enumerate(names):
        x = 12 + (i % COLS) * bw
        y = 36 + (i // COLS) * bh
        dl.rect(x - 1, y - 1, x + side + 1, y + side + 1, theme.HAIRLINE)
        dl.image(np.ascontiguousarray(
            (np.clip(tiles[i], 0, 1) * 255).astype(np.uint8)),
            x, y, side, side)
        dl.text(small, name, x, y + side + 4, theme.DIM)
    surf.draw(dl, body)
    ti.sync()
    save(screen(surf)[:zh, :zw], "icons_zoom4.png")


if __name__ == "__main__":
    main()
