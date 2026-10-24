"""Референсы по телу: смена камеры, разворот симуляции, зрение мухи.

Всё на настоящих данных: кадры сняты с тех камер, что есть в NeuroMechFly
(их семнадцать в config.yaml), зрение — настоящая мозаика из 721 омматидия
на глаз с двумя типами фоторецепторов. Нарисованного здесь нет ничего.

Три состояния одного экрана, а не три разных экрана:

  D «обычное»   — мозг ведёт, тело сбоку, камера выбирается чипами;
  E «развёрнуто» — симуляция во весь экран, мозг ушёл в угол;
  F «зрение»     — оба глаза крупно, тело и мозг отступают.
"""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import data as D                                  # noqa: E402
from flybrain import regions as R                               # noqa: E402
from flybrain.ui import brainview, font as F, gpu, theme         # noqa: E402

OUT = Path(__file__).parent
W, H = 1920, 1080

CAMS = [("сзади", "camera_back_track_game"), ("сбоку", "camera_right"),
        ("сверху", "camera_top"), ("в лицо", "camera_front_zoomin")]

BODY = {"readout": "ALPN", "dn_l": "100 Гц", "dn_r": "92 Гц",
        "amp_l": "0,74", "amp_r": "1,06", "dist": "18,1 мм",
        "speed": "14,6 мм/с", "grad": "+0,53", "t": "1,18 с"}


class Faces:
    def __init__(self, scale=1.0):
        body = int(round(15 * scale))
        small = int(round(13 * scale))
        self.ui = F.load(body)
        self.small = F.load(small)
        self.strong = F.load(body, role="strong")
        self.head = F.load(int(round(17 * scale)), role="strong")
        self.title = F.load(int(round(22 * scale)), role="strong")
        self.num = F.load(F.match_size(body, "num"), role="num")
        self.num_s = F.load(F.match_size(small, "num"), role="num")
        self.num_big = F.load(F.match_size(int(round(26 * scale)), "num"),
                              role="num")
        self.mono = F.load(F.match_size(small, "mono"), role="mono")


def load_png(name):
    from PIL import Image

    p = OUT / name
    if not p.exists():
        return None
    return np.ascontiguousarray(np.asarray(Image.open(p).convert("RGB")))


def eye_image(side):
    """Спокойная сборка глаза из двух типов фоторецепторов.

    Бледные и жёлтые омматидии показываются не как «синее и оранжевое», а
    как общая светлота с лёгким тёплым сдвигом там, где жёлтых больше: так
    видно, что видит муха, а не то, что мы раскрасили.
    """
    from PIL import Image

    pale = OUT / f"eye_{side}_pale.png"
    yellow = OUT / f"eye_{side}_yellow.png"
    if not (pale.exists() and yellow.exists()):
        return None
    p = np.asarray(Image.open(pale).convert("L"), dtype=np.float32) / 255.0
    y = np.asarray(Image.open(yellow).convert("L"), dtype=np.float32) / 255.0
    lum = 0.5 * (p + y)
    warm = np.clip(y - p, -1, 1) * 0.25
    rgb = np.stack([
        np.clip(lum * 1.00 + warm, 0, 1),
        np.clip(lum * 0.96 + warm * 0.35, 0, 1),
        np.clip(lum * 0.88 - warm * 0.30, 0, 1)], axis=-1)
    # Снаружи мозаики сетчатка кладёт белое, а не пустоту, и без маски
    # глаз выходит на кремовом квадрате. Маска снимается один раз: пиксель
    # внутри мозаики — тот, что меняется между нулём и единицей.
    mp = OUT / "eye_mask.npy"
    if mp.exists():
        mask = np.load(mp)[..., None]
        rgb = np.where(mask, rgb, np.array(theme.BG, dtype=np.float32))
    return np.ascontiguousarray((rgb * 255).astype(np.uint8))


def load_brain():
    df = D.load_neurons()
    xyz = df[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float32)
    good = np.isfinite(xyz).all(axis=1)
    centre = np.nanmean(xyz[good], axis=0)
    p = np.where(good[:, None], xyz, centre[None, :]).astype(np.float32)
    p -= 0.5 * (p.min(0) + p.max(0))
    p /= float(np.abs(p).max())
    p[:, 1] *= -1.0
    p[:, 2] *= -1.0
    reg = df["region"].to_numpy().astype(str)
    names = sorted(set(reg))
    cols = theme.region_colors({n: R.super_region(n) for n in names})
    rgb = np.array([cols[n] for n in names], dtype=np.float32)
    code = np.searchsorted(np.array(names), reg).astype(np.int32)
    act = np.zeros(len(df), dtype=np.float32)
    z = np.load(D.OUT / "demo-odour" / "bins.npz")
    act[z["neuron_idx"].astype(np.int64)] = z["bins"].max(axis=0).astype(np.float32)
    hi = float(np.percentile(act[act > 0], 99))
    bv = brainview.BrainView(np.ascontiguousarray(p), code, names, rgb)
    bv.set_activity(np.clip(act / hi, 0, 1).astype(np.float32))
    return bv, cols


# ---------------------------------------------------------------- приёмы

def chips(dl, f, x, y, h, items, current, gap=6):
    """Ряд чипов: выбранный держит акцент, остальные — только рамку."""
    for i, name in enumerate(items):
        w = f.small.width(name) + 24
        on = i == current
        dl.rect(x, y, x + w, y + h, theme.PANEL_HI if on else theme.PANEL)
        dl.frame(x, y, x + w, y + h, theme.ACCENT if on else theme.HAIRLINE)
        dl.text_centre(f.small, name, x, x + w, y + (h - f.small.gh) // 2,
                       theme.INK if on else theme.DIM)
        x += w + gap
    return x


def readings(dl, f, x, y, w, rows, gap=8):
    for lbl, val in rows:
        dl.text(f.small, lbl, x, y + 2, theme.DIM)
        dl.text_right(f.num, val, x + w, y, theme.INK)
        y += f.num.gh + gap
    return y


def panel(dl, x0, y0, x1, y1, title=None, f=None, fill=theme.PANEL):
    dl.rect(x0, y0, x1, y1, fill)
    dl.frame(x0, y0, x1, y1, theme.HAIRLINE)
    if title and f is not None:
        dl.text(f.small, title, x0 + 12, y0 + 9, theme.DIM)
        return y0 + 9 + f.small.gh + 9
    return y0


def expand_mark(dl, f, x1, y0, label="развернуть"):
    """Знак, что панель можно развернуть: подпись и уголок."""
    w = f.small.width(label) + 34
    x0 = x1 - w
    dl.text(f.small, label, x0, y0 + 9, theme.DIM)
    bx = x1 - 20
    by = y0 + 9 + f.small.gh // 2
    for dx, dy in ((0, -5), (0, 5)):
        dl.hline(bx, bx + 11, by + dy, theme.DIM)
    dl.vline(bx, by - 5, by + 6, theme.DIM)
    dl.vline(bx + 10, by - 5, by + 6, theme.DIM)
    return x0


# ---------------------------------------------------------------- D

def ref_d(surf, f, bv, cols, cam, eyes):
    """Обычное состояние: мозг ведёт, тело сбоку, камера выбирается чипами."""
    dl = gpu.DrawList()
    surf.clear(theme.BG)
    rail, meters, foot = 300, 288, 30
    sim_w = 640

    bx0, bx1 = rail, W - meters - sim_w
    bv.set_mode("activity")
    bv.draw(surf, dl, f.small, (bx0 + 14, 14, bx1 - bx0 - 28, H - foot - 28))

    dl.rect(0, 0, rail, H - foot, theme.PANEL)
    dl.vline(rail - 1, 0, H - foot, theme.HAIRLINE)
    dl.text(f.title, "Мозг мухи", 18, 18, theme.INK)
    dl.text(f.small, "тело · NeuroMechFly v2", 18, 18 + f.title.gh + 2,
            theme.FAINT)
    y = 18 + f.title.gh + f.small.gh + 16
    dl.hline(18, rail - 18, y, theme.HAIRLINE)
    y += 14
    dl.text(f.strong, "съём на рулёжку", 18, y, theme.DIM)
    y += f.strong.gh + 8
    dl.rect(18, y, rail - 18, y + 30, theme.BG)
    dl.frame(18, y, rail - 18, y + 30, theme.HAIRLINE)
    dl.text(f.mono, BODY["readout"], 28, y + (30 - f.mono.gh) // 2, theme.ACCENT)
    y += 44
    dl.text(f.strong, "рулёжка", 18, y, theme.DIM)
    y += f.strong.gh + 8
    y = readings(dl, f, 18, y, rail - 36,
                 [("слева", BODY["dn_l"]), ("справа", BODY["dn_r"]),
                  ("амплитуда L", BODY["amp_l"]), ("амплитуда R", BODY["amp_r"])])
    dl.rect(18, H - foot - 54, rail - 18, H - foot - 22, theme.ACCENT)
    dl.text_centre(f.strong, "Пуск", 18, rail - 18,
                   H - foot - 54 + (32 - f.strong.gh) // 2, (0.06, 0.07, 0.08))

    # панель симуляции
    sx0 = W - meters - sim_w
    dl.rect(sx0, 0, W - meters, H - foot, theme.PANEL)
    dl.vline(sx0, 0, H - foot, theme.HAIRLINE)
    hy = 14
    dl.text(f.head, "симуляция", sx0 + 16, hy, theme.INK)
    expand_mark(dl, f, W - meters - 16, hy - 2)
    hy += f.head.gh + 12
    chips(dl, f, sx0 + 16, hy, 28, [n for n, _ in CAMS], 0)
    hy += 38
    if cam is not None:
        ch = int((sim_w - 32) * cam.shape[0] / cam.shape[1])
        dl.image(cam, sx0 + 16, hy, sim_w - 32, ch)
        dl.frame(sx0 + 15, hy - 1, W - meters - 15, hy + ch + 1, theme.HAIRLINE)
        hy += ch + 18

    dl.text(f.strong, "зрение", sx0 + 16, hy, theme.DIM)
    dl.text_right(f.small, "721 омматидий на глаз", W - meters - 16, hy + 2,
                  theme.FAINT)
    hy += f.strong.gh + 10
    ew = (sim_w - 32 - 14) // 2
    for i, (side, img) in enumerate(eyes):
        ex = sx0 + 16 + i * (ew + 14)
        eh = min(int(ew * img.shape[0] / img.shape[1]), H - foot - hy - 16)
        dl.image(img, ex, hy, ew, eh)
        dl.frame(ex - 1, hy - 1, ex + ew + 1, hy + eh + 1, theme.HAIRLINE)
        dl.text(f.small, side, ex + 8, hy + 6, theme.DIM)

    # показания
    mx = W - meters
    dl.rect(mx, 0, W, H - foot, theme.PANEL)
    dl.vline(mx, 0, H - foot, theme.HAIRLINE)
    y = 18
    dl.text(f.strong, "показания", mx + 18, y, theme.DIM)
    y += f.strong.gh + 10
    y = readings(dl, f, mx + 18, y, meters - 36,
                 [("время", BODY["t"]), ("до источника", BODY["dist"]),
                  ("скорость", BODY["speed"]), ("градиент", BODY["grad"])])
    y += 8
    dl.hline(mx + 18, W - 18, y, theme.HAIRLINE)

    dl.hline(0, W, H - foot, theme.HAIRLINE)
    dl.text(f.small, "cuda · 138 639 нейронов · тело 0,051× реального времени",
            18, H - foot + (foot - f.small.gh) // 2, theme.DIM)
    surf.draw(dl, f.small)


# ---------------------------------------------------------------- E

def ref_e(surf, f, bv, cols, cam, eyes):
    """Развёрнуто: симуляция во весь экран, мозг ушёл в угол."""
    dl = gpu.DrawList()
    surf.clear(theme.BG)
    top, foot = 56, 30
    side = 360

    dl.rect(0, 0, W, top, theme.PANEL)
    dl.hline(0, W, top - 1, theme.HAIRLINE)
    dl.text(f.head, "симуляция", 24, (top - f.head.gh) // 2, theme.INK)
    x = 24 + f.head.width("симуляция") + 28
    chips(dl, f, x, 12, top - 24, [n for n, _ in CAMS], 0)
    expand_mark(dl, f, W - 24, (top - f.small.gh) // 2 - 9, "свернуть")

    # кадр во весь доступный прямоугольник
    vx0, vx1 = 0, W - side
    vy0, vy1 = top, H - foot
    if cam is not None:
        ar = cam.shape[1] / cam.shape[0]
        vw = vx1 - vx0
        vh = int(vw / ar)
        if vh > vy1 - vy0:
            vh = vy1 - vy0
            vw = int(vh * ar)
        ox = vx0 + ((vx1 - vx0) - vw) // 2
        oy = vy0 + ((vy1 - vy0) - vh) // 2
        dl.image(cam, ox, oy, vw, vh)

    # правая полоса: глаза и показания
    dl.rect(W - side, top, W, H - foot, theme.PANEL, 0.96)
    dl.vline(W - side, top, H - foot, theme.HAIRLINE)
    sx = W - side + 18
    sw = side - 36
    y = top + 16
    dl.text(f.strong, "зрение мухи", sx, y, theme.DIM)
    dl.text_right(f.small, "721 × 2", W - 18, y + 2, theme.FAINT)
    y += f.strong.gh + 10
    for name, img in eyes:
        eh = int(sw * img.shape[0] / img.shape[1])
        dl.image(img, sx, y, sw, eh)
        dl.frame(sx - 1, y - 1, sx + sw + 1, y + eh + 1, theme.HAIRLINE)
        dl.text(f.small, name, sx + 8, y + 6, theme.DIM)
        y += eh + 14
    y += 6
    dl.hline(sx, sx + sw, y, theme.HAIRLINE)
    y += 14
    dl.text(f.strong, "рулёжка", sx, y, theme.DIM)
    y += f.strong.gh + 10
    y = readings(dl, f, sx, y, sw,
                 [("съём", BODY["readout"]), ("слева", BODY["dn_l"]),
                  ("справа", BODY["dn_r"]), ("до источника", BODY["dist"]),
                  ("скорость", BODY["speed"])])

    dl.hline(0, W, H - foot, theme.HAIRLINE)
    dl.text(f.small, "камера сзади · следит за грудью · 0,2× реального времени",
            24, H - foot + (foot - f.small.gh) // 2, theme.DIM)
    surf.draw(dl, f.small)

    # Врезка с мозгом — последней: сплаты ложатся прямо на поверхность, а
    # картинка тела проходит буфером команд и легла бы поверх них.
    iw, ih = 300, 210
    ix, iy = 18, H - foot - ih - 18
    over = gpu.DrawList()
    over.rect(ix, iy, ix + iw, iy + ih, theme.BG)
    surf.draw(over, f.small)
    bv.show_labels = False
    bv.set_mode("regions")
    chrome = gpu.DrawList()
    bv.draw(surf, chrome, f.small, (ix + 1, iy + 1, iw - 2, ih - 2))
    chrome.frame(ix, iy, ix + iw, iy + ih, theme.HAIRLINE)
    chrome.text(f.small, "мозг · области", ix + 10, iy + 8, theme.DIM)
    surf.draw(chrome, f.small)
    bv.show_labels = True


# ---------------------------------------------------------------- F

def ref_f(surf, f, bv, cols, cam, eyes, raw):
    """Зрение: оба глаза крупно, тело и мозг отступают."""
    dl = gpu.DrawList()
    surf.clear(theme.BG)
    top, foot = 56, 30
    rail = 300

    dl.rect(0, 0, W, top, theme.PANEL)
    dl.hline(0, W, top - 1, theme.HAIRLINE)
    dl.text(f.head, "зрение мухи", 24, (top - f.head.gh) // 2, theme.INK)
    x = 24 + f.head.width("зрение мухи") + 28
    chips(dl, f, x, 12, top - 24,
          ["омматидии", "сырой кадр", "оба канала"], 0)
    dl.text_right(f.small, "721 омматидий на глаз · бледные и жёлтые",
                  W - 24, (top - f.small.gh) // 2, theme.FAINT)

    # два глаза крупно
    gap = 28
    ew = (W - rail - gap * 3) // 2
    ey = top + 28
    for i, (name, img) in enumerate(eyes):
        ex = rail + gap + i * (ew + gap)
        eh = int(ew * img.shape[0] / img.shape[1])
        eh = min(eh, H - foot - ey - 190)
        ew2 = int(eh * img.shape[1] / img.shape[0])
        ox = ex + (ew - ew2) // 2
        dl.image(img, ox, ey, ew2, eh)
        dl.frame(ox - 1, ey - 1, ox + ew2 + 1, ey + eh + 1, theme.HAIRLINE)
        dl.text_centre(f.strong, name, ex, ex + ew, ey + eh + 12, theme.DIM)

    # под ними: тело и сырой кадр
    by = ey + eh + 12 + f.strong.gh + 20
    bh = H - foot - by - 18
    if cam is not None:
        bw = int(bh * cam.shape[1] / cam.shape[0])
        dl.image(cam, rail + gap, by, bw, bh)
        dl.frame(rail + gap - 1, by - 1, rail + gap + bw + 1, by + bh + 1,
                 theme.HAIRLINE)
        dl.text(f.small, "тело · камера сзади", rail + gap + 8, by + 6,
                theme.DIM)
        rx = rail + gap + bw + gap
    else:
        rx = rail + gap
    if raw is not None:
        rw = int(bh * raw.shape[1] / raw.shape[0])
        dl.image(raw, rx, by, rw, bh)
        dl.frame(rx - 1, by - 1, rx + rw + 1, by + bh + 1, theme.HAIRLINE)
        dl.text(f.small, "до омматидий · что попадает на глаз", rx + 8, by + 6,
                theme.DIM)

    # рейка слева
    dl.rect(0, top, rail, H - foot, theme.PANEL)
    dl.vline(rail, top, H - foot, theme.HAIRLINE)
    y = top + 18
    dl.text(f.strong, "оптические доли", 18, y, theme.DIM)
    y += f.strong.gh + 10
    y = readings(dl, f, 18, y, rail - 36,
                 [("нейронов", "96 495"), ("медулла", "66 915"),
                  ("ламина", "12 153"), ("лобула", "10 456")])
    y += 10
    dl.hline(18, rail - 18, y, theme.HAIRLINE)
    y += 14
    for line in ("Сейчас оптические доли не получают",
                 "входа: 70 % мозга молчит.",
                 "",
                 "Сетчатка NeuroMechFly отдаёт 721",
                 "омматидий на глаз — это и есть вход",
                 "для R1-6, R7 и R8."):
        dl.text(f.small, line, 18, y, theme.DIM if line else theme.FAINT,
                max_px=rail - 36)
        y += f.small.gh + 4

    dl.hline(0, W, H - foot, theme.HAIRLINE)
    dl.text(f.small, "зрение включается параметром enable_vision тела",
            24, H - foot + (foot - f.small.gh) // 2, theme.FAINT)
    surf.draw(dl, f.small)


def main():
    ti.init(arch=ti.vulkan, offline_cache=True)
    bv, cols = load_brain()
    f = Faces(1.0)
    surf = gpu.Surface()
    surf.ensure(W, H)

    cam = load_png("cam_camera_back_track_game.png")
    raw = load_png("eye_raw_left.png")
    eyes = [(n, eye_image(s)) for n, s in (("левый глаз", "left"),
                                           ("правый глаз", "right"))]
    eyes = [(n, i) for n, i in eyes if i is not None]
    if not eyes:
        print("нет снимков глаз — сперва probe/eyes.py")
        return

    from PIL import Image

    for name, fn in (("D_обычное", lambda: ref_d(surf, f, bv, cols, cam, eyes)),
                     ("E_развёрнуто", lambda: ref_e(surf, f, bv, cols, cam, eyes)),
                     ("F_зрение", lambda: ref_f(surf, f, bv, cols, cam, eyes, raw))):
        bv.view_i = 3
        _, bv.yaw, bv.pitch = brainview.VIEWS[3]
        bv.refit()
        fn()
        ti.sync()
        a = np.clip(surf.acc.to_numpy(), 0, 1).transpose(1, 0, 2)[::-1]
        p = OUT / ("ref_" + name + ".png")
        Image.fromarray((a * 255).astype(np.uint8)).save(p)
        print("записан " + p.name)


if __name__ == "__main__":
    main()
