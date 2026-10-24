"""Проверка gpu.py без окна.

Снимок swapchain в этой среде выходит чёрным, но поле-накопитель читается
напрямую, поэтому кадр можно собрать headless и посмотреть глазами как PNG.
Заодно это замер: сколько стоит кадр со сплатами, панелями и текстом вместе.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import data as D                                  # noqa: E402
from flybrain import regions as R                               # noqa: E402
from flybrain.ui import font as F                               # noqa: E402
from flybrain.ui import gpu, theme                              # noqa: E402

W, H = 1500, 880
OUT = Path(__file__).parent


def main():
    t0 = time.perf_counter()
    df = D.load_neurons()
    xyz = df[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    p = xyz[keep]
    p -= 0.5 * (p.min(0) + p.max(0))
    p /= float(np.abs(p).max())
    p[:, 1] *= -1.0
    p[:, 2] *= -1.0
    pos = np.ascontiguousarray(p)

    full = np.zeros(len(df), dtype=np.float32)
    z = np.load(D.OUT / "demo-sugar" / "bins.npz")
    full[z["neuron_idx"].astype(np.int64)] = z["bins"].max(axis=0).astype(np.float32)
    act = full[keep]
    hi = float(np.percentile(act[act > 0], 99))
    act = np.ascontiguousarray(np.clip(act / hi, 0, 1).astype(np.float32))

    reg = df["region"].to_numpy()[keep]
    names = sorted({str(r) for r in reg})
    r2s = {n: R.super_region(n) for n in names}
    cols = theme.region_colors(r2s)
    reg_rgb = np.array([cols[n] for n in names], dtype=np.float32)
    code = np.searchsorted(np.array(names), reg.astype(str)).astype(np.int32)

    mg = np.array(theme.MAGMA, dtype=np.float32)
    t = np.clip(act, 0, 1) * (len(mg) - 1)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(mg) - 1)
    w = (t - i0.astype(np.float32))[:, None]
    col_act = np.ascontiguousarray((mg[i0] * (1 - w) + mg[i1] * w).astype(np.float32))
    col_reg = np.ascontiguousarray(
        (reg_rgb[code] * (0.22 + 0.78 * act[:, None])).astype(np.float32))
    print("данные: " + str(len(pos)) + " точек, "
          + str(int((act > 0).sum())) + " активных, "
          + str(len(names)) + " областей, "
          + format(time.perf_counter() - t0, ".2f") + " с")

    ti.init(arch=ti.vulkan, offline_cache=True)
    atlas = F.load(14)
    small = F.load(12)
    surf = gpu.Surface()
    surf.ensure(W, H)

    rail, meter, pad = theme.RAIL_W, theme.METER_W, theme.PAD
    foot = theme.FOOT_H
    vx = rail + pad
    vw = W - rail - meter - pad * 2
    vy = pad
    vh = H - pad * 2 - foot

    mvp, _ = gpu.orbit_mvp(0.55, 0.22, 2.9, vw / vh)

    def frame(by_region=False):
        surf.clear(theme.BG)
        surf.splat(pos, act, col_reg if by_region else col_act, mvp,
                   (vx, vy, vw, vh), base=0.13)

        dl = gpu.DrawList()
        # левая рейка
        dl.rect(0, 0, rail, H, theme.PANEL)
        dl.vline(rail, 0, H, theme.HAIRLINE)
        y = pad
        dl.text(atlas, "Мозг мухи", pad, y, theme.INK)
        y += atlas.gh + 6
        dl.text(small, "FlyWire v783 · 138 625 нейронов", pad, y, theme.DIM)
        y += small.gh + 14
        dl.hline(pad, rail - pad, y, theme.HAIRLINE)
        y += 14
        for i, name in enumerate(("Стимуляция", "Арена", "Тело", "Зонд стороны",
                                  "Серия", "Агент", "Демо", "Прогоны")):
            on = i == 2
            dl.rect(pad, y, rail - pad, y + theme.ROW_H,
                    theme.PANEL_HI if on else theme.PANEL, 1.0 if on else 0.0)
            if on:
                dl.rect(pad, y, pad + 3, y + theme.ROW_H, theme.ACCENT)
            dl.text(atlas, name, pad + 12, y + 5,
                    theme.INK if on else theme.DIM)
            y += theme.ROW_H + 2

        y += 18
        dl.text(small, "ЗАПРОС", pad, y, theme.FAINT)
        y += small.gh + 8
        dl.rect(pad, y, rail - pad, y + 30, theme.BG)
        dl.frame(pad, y, rail - pad, y + 30, theme.HAIRLINE)
        dl.text(atlas, "class:olfactory+side:left", pad + 10, y + 7, theme.ACCENT)
        y += 44
        for lbl, val in (("длительность", "0,8 с"), ("частота", "150 Гц"),
                         ("бин", "10 мс"), ("съём", "ALPN")):
            dl.text(small, lbl, pad, y, theme.DIM)
            dl.text_right(small, val, rail - pad, y, theme.INK)
            y += small.gh + 7

        y += 16
        dl.rect(pad, y, rail - pad, y + 34, theme.ACCENT)
        dl.text_centre(atlas, "Пуск", pad, rail - pad, y + 9, (0.06, 0.07, 0.08))

        # правая колонка
        mx = W - meter
        dl.rect(mx, 0, W, H, theme.PANEL)
        dl.vline(mx, 0, H, theme.HAIRLINE)
        y = pad
        dl.text(small, "ПОКАЗАНИЯ", mx + pad, y, theme.FAINT)
        y += small.gh + 10
        for lbl, val in (("время", "640 мс"), ("активных", "9110"),
                         ("пик", "312 Гц"), ("шкала", "128"),
                         ("ведёт", "AL")):
            dl.text(small, lbl, mx + pad, y, theme.DIM)
            dl.text_right(atlas, val, W - pad, y - 2, theme.INK)
            y += atlas.gh + 6
        y += 10
        dl.hline(mx + pad, W - pad, y, theme.HAIRLINE)
        y += 12
        dl.text(small, "ОБЛАСТИ", mx + pad, y, theme.FAINT)
        y += small.gh + 8

        sums = np.zeros(len(names), dtype=np.float64)
        np.add.at(sums, code, act)
        cnt = np.bincount(code, minlength=len(names)).astype(np.float64)
        mean = sums / np.maximum(cnt, 1)
        order = np.argsort(-mean)[:14]
        top = float(mean[order[0]]) if len(order) else 1.0
        bw = meter - pad * 2
        for i in order:
            lv = float(mean[i]) / max(top, 1e-9)
            dl.rect(mx + pad, y, mx + pad + bw, y + 14, theme.BG)
            dl.rect(mx + pad, y, mx + pad + int(bw * lv), y + 14,
                    cols[names[i]], 0.95)
            dl.text(small, names[i], mx + pad + 6, y, theme.INK)
            dl.text_right(small, format(mean[i] * 100, ".0f"), W - pad - 6, y,
                          theme.DIM)
            y += 18

        # подвал
        dl.hline(0, W, H - foot, theme.HAIRLINE)
        dl.text(small, "cuda · 138 625 нейронов · 15 091 983 связи",
                pad, H - foot + 5, theme.DIM)
        dl.text_right(small, str(W) + "×" + str(H) + " · 60 кадр/с",
                      W - pad, H - foot + 5, theme.FAINT)

        # рамка вьюпорта мозга
        dl.frame(vx - 1, vy - 1, vx + vw + 1, vy + vh + 1, theme.HAIRLINE)
        dl.text(small, "АКТИВНОСТЬ" if not by_region else "ОБЛАСТИ",
                vx + 10, vy + 8, theme.FAINT)
        surf.draw(dl, atlas)
        return dl

    for _ in range(6):
        frame()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(40):
        frame()
    ti.sync()
    ms = (time.perf_counter() - t0) * 1000 / 40
    print("кадр целиком: " + format(ms, ".2f") + " мс ("
          + format(1000 / ms, ".0f") + " кадр/с), окно " + str(W) + "×" + str(H))

    from PIL import Image

    for name, by_reg in (("render_activity.png", False), ("render_regions.png", True)):
        frame(by_reg)
        ti.sync()
        a = surf.acc.to_numpy()
        img = np.clip(a, 0, 1).transpose(1, 0, 2)[::-1]
        Image.fromarray((img * 255).astype(np.uint8)).save(OUT / name)
        print("записан " + str(OUT / name))


if __name__ == "__main__":
    main()
