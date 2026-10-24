"""Сборка оболочки и карты в живом окне.

Проверяет то, что headless не проверишь: что окно показывает собранный кадр,
что курсор считается в ту же сторону, что рисование, что вращение и колесо
работают, что при изменении размера окна поля пересоздаются без падения.

Перекрестие рисуется ровно там, где оболочка считает курсор. Если оно ходит
не за мышью, а зеркально по вертикали — значит начало отсчёта курсора в GGUI
обратное, и это надо поправить в `shell.Input`.

Запуск:
    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\shell_test.py
Выход — Esc. Переменная LOOK_SECONDS=10 закрывает окно сама.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import data as D                                  # noqa: E402
from flybrain import regions as R                               # noqa: E402
from flybrain.ui import brainview, shell, theme                 # noqa: E402


def load():
    df = D.load_neurons()
    xyz = df[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    p = xyz[keep]
    p -= 0.5 * (p.min(0) + p.max(0))
    p /= float(np.abs(p).max())
    p[:, 1] *= -1.0
    p[:, 2] *= -1.0

    reg = df["region"].to_numpy()[keep].astype(str)
    names = sorted(set(reg))
    cols = theme.region_colors({n: R.super_region(n) for n in names})
    rgb = np.array([cols[n] for n in names], dtype=np.float32)
    code = np.searchsorted(np.array(names), reg).astype(np.int32)

    full = np.zeros(len(df), dtype=np.float32)
    z = np.load(D.OUT / "demo-sugar" / "bins.npz")
    full[z["neuron_idx"].astype(np.int64)] = z["bins"].max(axis=0).astype(np.float32)
    act = full[keep]
    hi = float(np.percentile(act[act > 0], 99))
    return p, code, names, rgb, np.clip(act / hi, 0, 1).astype(np.float32), df, keep


def main():
    pos, code, names, rgb, act, df, keep = load()
    labels = df["label"].to_numpy()[keep].astype(str)

    sh = shell.Shell("Мозг мухи — оболочка", (1500, 880))
    bv = brainview.BrainView(pos, code, names, rgb)
    bv.set_activity(act)

    limit = float(os.environ.get("LOOK_SECONDS", "0"))
    started = time.perf_counter()
    shot_done = False

    while sh.running:
        if limit and time.perf_counter() - started > limit:
            break
        dl = sh.begin()
        inp = sh.input

        for k in inp.keys:
            if k == "Escape":
                sh.stop()
            elif k == " ":
                bv.next_view()
            elif k in ("r", "R"):
                bv.set_mode("regions" if bv.mode == "activity" else "activity")
            elif k in ("o", "O"):
                bv.only_active = not bv.only_active
            elif k in ("l", "L"):
                bv.show_labels = not bv.show_labels
            elif k == "Up":
                bv.zoom = max(0.25, bv.zoom * 0.9)
            elif k == "Down":
                bv.zoom = min(4.0, bv.zoom * 1.1)

        pad = sh.px(theme.PAD)
        rail = sh.px(theme.RAIL_W)
        vp = (rail + pad, pad, sh.w - rail - pad * 2, sh.h - pad * 2 - sh.px(28))

        bv.interact(vp, inp.cursor, inp.down, inp.wheel)
        bv.draw(sh.surf, dl, sh.small, vp)
        dl.frame(vp[0] - 1, vp[1] - 1, vp[0] + vp[2] + 1, vp[1] + vp[3] + 1,
                 theme.HAIRLINE)

        # левая рейка — пока только чтобы видеть масштаб и раскладку
        dl.rect(0, 0, rail, sh.h, theme.PANEL)
        dl.vline(rail, 0, sh.h, theme.HAIRLINE)
        y = pad
        dl.text(sh.title_atlas, "Мозг мухи", pad, y, theme.INK)
        y += sh.title_atlas.gh + sh.px(6)
        for line in ("FlyWire v783 · 138 625 нейронов",
                     "demo-sugar · " + str(int((act > 0).sum())) + " активных"):
            dl.text(sh.small, line, pad, y, theme.DIM)
            y += sh.small.gh + sh.px(3)
        y += sh.px(12)
        dl.hline(pad, rail - pad, y, theme.HAIRLINE)
        y += sh.px(12)
        for lbl, val in (("вид", bv.view_name if bv.view_i >= 0 else "свой"),
                         ("режим", "области" if bv.mode == "regions" else "активность"),
                         ("подписи", "да" if bv.show_labels else "нет"),
                         ("только активные", "да" if bv.only_active else "нет"),
                         ("приближение", format(1.0 / bv.zoom, ".2f") + "×"),
                         ("масштаб", format(sh.scale, ".2f")),
                         ("кегль", str(sh.atlas.size)),
                         ("окно", str(sh.w) + "×" + str(sh.h)),
                         ("кадр/с", format(sh.fps, ".0f"))):
            dl.text(sh.small, lbl, pad, y, theme.DIM)
            dl.text_right(sh.small, str(val), rail - pad, y, theme.INK)
            y += sh.small.gh + sh.px(4)
        y += sh.px(10)
        dl.text(sh.small, "пробел вид · R режим · O активные", pad, y, theme.FAINT)
        y += sh.small.gh + sh.px(3)
        dl.text(sh.small, "L подписи · стрелки приближение · Esc", pad, y, theme.FAINT)

        # наведение: подпись под курсором
        hit = bv.pick(inp.cursor)
        if hit is not None:
            name = labels[hit]
            reg = names[int(code[hit])]
            txt = name + " · " + reg + " · " + format(act[hit] * 100, ".0f") + "%"
            tw = sh.small.width(txt) + sh.px(14)
            tx = min(inp.x + sh.px(14), sh.w - tw - 2)
            ty = max(inp.y - sh.px(22), 2)
            dl.rect(tx, ty, tx + tw, ty + sh.small.gh + sh.px(8), theme.PANEL, 0.96)
            dl.frame(tx, ty, tx + tw, ty + sh.small.gh + sh.px(8), theme.HAIRLINE)
            dl.text(sh.small, txt, tx + sh.px(7), ty + sh.px(4), theme.INK)

        # перекрестие там, где оболочка считает курсор
        dl.hline(inp.x - 9, inp.x + 10, inp.y, theme.ACCENT, 1, 0.8)
        dl.vline(inp.x, inp.y - 9, inp.y + 10, theme.ACCENT, 1, 0.8)

        dl.hline(0, sh.w, sh.h - sh.px(28), theme.HAIRLINE)
        dl.text(sh.small, "курсор " + str(inp.x) + ", " + str(inp.y)
                + ("  нажата" if inp.down else ""),
                pad, sh.h - sh.px(22), theme.DIM)

        sh.end(dl)

        if not shot_done and time.perf_counter() - started > 2.0:
            sh.shot(str(Path(__file__).parent / "shell_frame.png"))
            shot_done = True

    print("размеры окна за сеанс: " + str(sh.resizes))
    print("кадр/с на выходе: " + format(sh.fps, ".0f"))
    sh.close()


if __name__ == "__main__":
    main()
