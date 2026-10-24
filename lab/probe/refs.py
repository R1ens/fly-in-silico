"""Три макета одного экрана — на утверждение, до единой строки кода окна.

Рисуются тем же растеризатором, тем же атласом и на тех же данных, что и
настоящее окно: утверждённое будет выглядеть ровно так, а не «примерно так».
Виджетов здесь нет намеренно — координаты расставлены руками, потому что
набор виджетов как раз и пересобирается.

Три направления различаются устройством, а не оттенком:

  A «стенд»    — закреплённые колонки по краям, плотная сетка, много чисел;
  B «разворот» — карта во весь экран, управление сверху, показания карточкой
                 поверх карты;
  C «полосы»   — горизонтальные полосы: команда, карта, показания плиткой.

Запуск: cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\refs.py
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

# Данные показаний взяты с настоящего прогона: вкус слева, 200 Гц, 0,8 с.
RUN = {
    "query": "class:gustatory+side:left",
    "rate": "200 Гц", "dur": "0,8 с", "bin": "10 мс",
    "t": "0,32 с", "active": "986", "peak": "146 Гц", "scale": "146 Гц",
    "frames": "32", "device": "cuda", "neurons": "138 639",
}
REGIONS = [("GNG", 4.9, 1.00), ("NERVE", 1.9, 0.39), ("ATL", 1.9, 0.38),
           ("CRE", 1.0, 0.21), ("SAD", 0.8, 0.16), ("LAL", 0.7, 0.15),
           ("VES", 0.6, 0.12), ("SIP", 0.3, 0.07), ("AL", 0.3, 0.06),
           ("VNC", 0.3, 0.05), ("SMP", 0.2, 0.05), ("WED", 0.2, 0.04)]


class Faces:
    """Начертания, собранные один раз: текст, полужирный, заголовок, числа."""

    def __init__(self, scale=1.0):
        body = int(round(15 * scale))
        small = int(round(13 * scale))
        self.ui = F.load(body)
        self.small = F.load(small)
        self.strong = F.load(body, role="strong")
        self.title = F.load(int(round(22 * scale)), role="strong")
        self.head = F.load(int(round(17 * scale)), role="strong")
        self.num = F.load(F.match_size(body, "num"), role="num")
        self.num_s = F.load(F.match_size(small, "num"), role="num")
        self.num_big = F.load(
            F.match_size(int(round(26 * scale)), "num"), role="num")
        # Селектор и имена типов остаются моноширинными: там
        # одинаковая ширина знака несёт смысл.
        self.mono = F.load(F.match_size(small, "mono"), role="mono")


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
    z = np.load(D.OUT / "demo-sugar" / "bins.npz")
    act[z["neuron_idx"].astype(np.int64)] = z["bins"].max(axis=0).astype(np.float32)
    hi = float(np.percentile(act[act > 0], 99))
    act = np.clip(act / hi, 0, 1).astype(np.float32)

    bv = brainview.BrainView(np.ascontiguousarray(p), code, names, rgb)
    bv.set_activity(act)
    return bv, cols


def body_frame():
    from PIL import Image

    p = OUT / "ipc_body_frame.png"
    if p.exists():
        return np.ascontiguousarray(np.asarray(Image.open(p).convert("RGB")))
    return None


# ---------------------------------------------------------------- приёмы

def tile(dl, x0, y0, x1, y1, fill=None, line=True):
    if fill is not None:
        dl.rect(x0, y0, x1, y1, fill)
    if line:
        dl.frame(x0, y0, x1, y1, theme.HAIRLINE)


def bars(dl, f, x0, y0, w, rows, cols, row_h, gap, label_w, num_w):
    """Полосы по областям: подпись, дорожка, число — в три закреплённые колонки."""
    y = y0
    for name, val, frac in rows:
        dl.text(f.small, name, x0, y + (row_h - f.small.gh) // 2, theme.INK,
                max_px=label_w)
        tx0 = x0 + label_w
        tx1 = x0 + w - num_w
        dl.rect(tx0, y + row_h // 2 - 3, tx1, y + row_h // 2 + 3, theme.BG)
        dl.rect(tx0, y + row_h // 2 - 3,
                tx0 + int((tx1 - tx0) * frac), y + row_h // 2 + 3,
                cols.get(name, theme.ACCENT))
        dl.text_right(f.num_s, f"{val:.1f}", x0 + w,
                      y + (row_h - f.num_s.gh) // 2, theme.DIM)
        y += row_h + gap
    return y


def spark(dl, x0, y0, x1, y1, series, color):
    n = len(series)
    lo, hi = float(min(series)), float(max(series))
    rng = max(hi - lo, 1e-6)
    pts = np.stack([
        np.linspace(x0, x1, n),
        y1 - (np.asarray(series, np.float32) - lo) / rng * (y1 - y0)], axis=1)
    dl.polyline(pts, color, width=1.6)


DEMO_SERIES = [0.2, 0.26, 0.31, 0.44, 0.52, 0.49, 0.58, 0.71, 0.66, 0.74,
               0.83, 0.79, 0.88, 0.95, 0.9, 0.97, 0.92, 0.85, 0.8, 0.76]


# ---------------------------------------------------------------- макет A

def ref_a(surf, f, bv, cols, frame):
    """«Стенд»: закреплённые колонки, плотная сетка, числа ведут."""
    dl = gpu.DrawList()
    rail, meters, foot = 304, 288, 30
    strip = 236
    surf.clear(theme.BG)

    # карта: единственная область, которая тянется
    mx0, mx1 = rail, W - meters
    my0, my1 = 0, H - foot - strip
    bv.show_labels = True
    bv.set_mode("activity")
    bv.draw(surf, dl, f.small, (mx0 + 1, my0 + 1, mx1 - mx0 - 2, my1 - my0 - 2))

    # левая рейка
    dl.rect(0, 0, rail, H - foot, theme.PANEL)
    dl.vline(rail - 1, 0, H - foot, theme.HAIRLINE)
    x, w = 18, rail - 36
    dl.text(f.title, "Мозг мухи", x, 18, theme.INK)
    dl.text(f.small, "FlyWire v783", x, 18 + f.title.gh + 2, theme.FAINT)
    y = 18 + f.title.gh + f.small.gh + 14
    dl.hline(x, x + w, y, theme.HAIRLINE)
    y += 12

    for row in (("Стимуляция", "Арена", "Тело"), ("Зонд", "Демо", "Прогоны")):
        cw = w // len(row)
        for i, name in enumerate(row):
            on = name == "Стимуляция"
            bx = x + i * cw
            dl.rect(bx, y, bx + cw, y + 28,
                    theme.PANEL_HI if on else theme.PANEL)
            dl.frame(bx, y, bx + cw, y + 28, theme.HAIRLINE)
            if on:
                dl.rect(bx, y + 26, bx + cw, y + 28, theme.ACCENT)
            dl.text_centre(f.small, name, bx, bx + cw,
                           y + (28 - f.small.gh) // 2,
                           theme.INK if on else theme.DIM)
        y += 30
    y += 14

    dl.text(f.strong, "запрос", x, y, theme.DIM)
    y += f.strong.gh + 6
    dl.rect(x, y, x + w, y + 30, theme.BG)
    dl.frame(x, y, x + w, y + 30, theme.HAIRLINE)
    dl.text(f.mono, RUN["query"], x + 9, y + (30 - f.mono.gh) // 2,
            theme.ACCENT, max_px=w - 18)
    y += 40

    dl.text(f.strong, "параметры", x, y, theme.DIM)
    y += f.strong.gh + 6
    for lbl, val in (("частота", RUN["rate"]), ("длительность", RUN["dur"]),
                     ("бин", RUN["bin"]), ("повторов", "1")):
        dl.text(f.small, lbl, x, y, theme.DIM)
        dl.text_right(f.num_s, val, x + w, y, theme.INK)
        y += f.small.gh + 7
    y += 10

    dl.rect(x, H - foot - 54, x + w, H - foot - 22, theme.ACCENT)
    dl.text_centre(f.strong, "Пуск", x, x + w,
                   H - foot - 54 + (32 - f.strong.gh) // 2, (0.06, 0.07, 0.08))
    dl.text(f.small, "идёт счёт · 0,32 с", x, H - foot - 76, theme.ACCENT)

    # правая колонка
    dl.rect(W - meters, 0, W, H - foot, theme.PANEL)
    dl.vline(W - meters, 0, H - foot, theme.HAIRLINE)
    rx, rw = W - meters + 18, meters - 36
    y = 18
    dl.text(f.strong, "показания", rx, y, theme.DIM)
    y += f.strong.gh + 8
    for lbl, val in (("время", RUN["t"]), ("активных", RUN["active"]),
                     ("пик", RUN["peak"]), ("шкала", RUN["scale"]),
                     ("кадров", RUN["frames"])):
        dl.text(f.small, lbl, rx, y + 2, theme.DIM)
        dl.text_right(f.num, val, rx + rw, y, theme.INK)
        y += f.num.gh + 6
    y += 8
    dl.hline(rx, rx + rw, y, theme.HAIRLINE)
    y += 12
    dl.text(f.strong, "области", rx, y, theme.DIM)
    y += f.strong.gh + 8
    bars(dl, f, rx, y, rw, REGIONS, cols, 22, 2, 58, 40)

    # нижняя полоса
    sy = H - foot - strip
    dl.hline(rail, W - meters, sy, theme.HAIRLINE)
    half = (W - meters - rail) // 2
    if frame is not None:
        dl.image(frame, rail + 16, sy + 16, half - 32, strip - 32)
        tile(dl, rail + 16, sy + 16, rail + half - 16, sy + strip - 16, None)
    dl.text(f.small, "вид сверху", rail + 24, sy + 24, theme.DIM)
    bx0 = rail + half + 16
    tile(dl, bx0, sy + 16, W - meters - 16, sy + strip - 16, theme.BG)
    dl.text(f.small, "ход активности", bx0 + 8, sy + 24, theme.DIM)
    spark(dl, bx0 + 12, sy + 48, W - meters - 28, sy + strip - 28,
          DEMO_SERIES, theme.ACCENT)

    # подвал
    dl.hline(0, W, H - foot, theme.HAIRLINE)
    dl.text(f.small, f"{RUN['device']} · {RUN['neurons']} нейронов", 18,
            H - foot + (foot - f.small.gh) // 2, theme.DIM)
    dl.text_right(f.num_s, f"{W}×{H} · 60 кадр/с", W - 18,
                  H - foot + (foot - f.num_s.gh) // 2, theme.FAINT)
    surf.draw(dl, f.small)
    return dl


# ---------------------------------------------------------------- макет B

def ref_b(surf, f, bv, cols, frame):
    """«Разворот»: карта во весь экран, управление сверху, показания карточкой."""
    dl = gpu.DrawList()
    bar = 60
    surf.clear(theme.BG)

    bv.show_labels = True
    bv.set_mode("activity")
    bv.draw(surf, dl, f.small, (0, bar, W, H - bar))

    # верхняя полоса управления
    dl.rect(0, 0, W, bar, theme.PANEL)
    dl.hline(0, W, bar - 1, theme.HAIRLINE)
    dl.text(f.head, "Мозг мухи", 24, (bar - f.head.gh) // 2, theme.INK)
    x = 24 + f.head.width("Мозг мухи") + 28
    for name in ("Стимуляция", "Арена", "Тело", "Зонд", "Демо", "Прогоны"):
        on = name == "Стимуляция"
        tw = f.small.width(name) + 26
        dl.text_centre(f.small, name, x, x + tw, (bar - f.small.gh) // 2,
                       theme.INK if on else theme.DIM)
        if on:
            dl.rect(x + 10, bar - 4, x + tw - 10, bar - 1, theme.ACCENT)
        x += tw
    # строка запроса и пуск прижаты вправо
    go_w = 112
    q1 = W - 24 - go_w - 16
    q0 = q1 - 420
    dl.rect(q0, 13, q1, bar - 13, theme.BG)
    dl.frame(q0, 13, q1, bar - 13, theme.HAIRLINE)
    dl.text(f.mono, RUN["query"], q0 + 12, (bar - f.num_s.gh) // 2,
            theme.ACCENT, max_px=q1 - q0 - 24)
    dl.rect(W - 24 - go_w, 13, W - 24, bar - 13, theme.ACCENT)
    dl.text_centre(f.strong, "Пуск", W - 24 - go_w, W - 24,
                   (bar - f.strong.gh) // 2, (0.06, 0.07, 0.08))

    # карточка показаний поверх карты
    cw, ch = 360, 330
    cx, cy = 28, H - ch - 28
    dl.rect(cx, cy, cx + cw, cy + ch, theme.PANEL, 0.94)
    dl.frame(cx, cy, cx + cw, cy + ch, theme.HAIRLINE)
    px = cx + 20
    dl.text(f.small, "активных нейронов", px, cy + 18, theme.DIM)
    dl.text(f.num_big, RUN["active"], px, cy + 18 + f.small.gh + 4, theme.INK)
    vx = px + 150
    dl.text(f.small, "пик", vx, cy + 18, theme.DIM)
    dl.text(f.num, RUN["peak"], vx, cy + 18 + f.small.gh + 10, theme.INK)
    dl.text(f.small, "время", vx + 96, cy + 18, theme.DIM)
    dl.text(f.num, RUN["t"], vx + 96, cy + 18 + f.small.gh + 10, theme.INK)
    y = cy + 18 + f.small.gh + f.num_big.gh + 16
    dl.hline(px, cx + cw - 20, y, theme.HAIRLINE)
    y += 14
    dl.text(f.strong, "области", px, y, theme.DIM)
    y += f.strong.gh + 8
    bars(dl, f, px, y, cw - 40, REGIONS[:7], cols, 22, 2, 58, 38)

    # кадр тела карточкой справа внизу
    if frame is not None:
        bw, bh = 340, 255
        bx, by = W - bw - 28, H - bh - 28
        dl.rect(bx - 1, by - 1, bx + bw + 1, by + bh + 1, theme.PANEL, 0.94)
        dl.image(frame, bx, by + 26, bw, bh - 26)
        dl.frame(bx - 1, by - 1, bx + bw + 1, by + bh + 1, theme.HAIRLINE)
        dl.text(f.small, "тело · вид сверху", bx + 10, by + 5, theme.DIM)
    surf.draw(dl, f.small)
    return dl


# ---------------------------------------------------------------- макет C

def ref_c(surf, f, bv, cols, frame):
    """«Полосы»: команда, карта, показания плиткой — без боковых колонок."""
    dl = gpu.DrawList()
    top, cmd, bottom, foot = 56, 52, 232, 28
    surf.clear(theme.BG)

    my0 = top + cmd
    my1 = H - foot - bottom
    bv.show_labels = True
    bv.set_mode("regions")
    bv.draw(surf, dl, f.small, (0, my0, W, my1 - my0))

    # полоса заголовка
    dl.rect(0, 0, W, top, theme.PANEL)
    dl.text(f.head, "Мозг мухи", 24, (top - f.head.gh) // 2, theme.INK)
    x = W // 2 - 320
    for name in ("Стимуляция", "Арена", "Тело", "Зонд", "Демо", "Прогоны"):
        on = name == "Стимуляция"
        tw = f.small.width(name) + 30
        if on:
            dl.rect(x, 10, x + tw, top - 10, theme.PANEL_HI)
            dl.frame(x, 10, x + tw, top - 10, theme.HAIRLINE)
        dl.text_centre(f.small, name, x, x + tw, (top - f.small.gh) // 2,
                       theme.INK if on else theme.DIM)
        x += tw + 4
    dl.text_right(f.small, "идёт счёт · 0,32 с", W - 24,
                  (top - f.small.gh) // 2, theme.ACCENT)
    dl.hline(0, W, top - 1, theme.HAIRLINE)

    # полоса команды: весь запуск одной строкой
    dl.rect(0, top, W, top + cmd, theme.BG)
    dl.hline(0, W, top + cmd - 1, theme.HAIRLINE)
    y = top + (cmd - f.num_s.gh) // 2
    dl.text(f.small, "зажечь", 24, y, theme.DIM)
    qx = 24 + f.small.width("зажечь") + 14
    dl.rect(qx, top + 9, qx + 430, top + cmd - 9, theme.PANEL)
    dl.frame(qx, top + 9, qx + 430, top + cmd - 9, theme.HAIRLINE)
    dl.text(f.mono, RUN["query"], qx + 12, y, theme.ACCENT, max_px=406)
    px = qx + 460
    for lbl, val in (("частота", RUN["rate"]), ("длительность", RUN["dur"]),
                     ("бин", RUN["bin"])):
        dl.text(f.small, lbl, px, y, theme.DIM)
        px += f.small.width(lbl) + 10
        dl.text(f.num_s, val, px, y, theme.INK)
        px += f.num_s.width(val) + 28
    dl.rect(W - 24 - 112, top + 9, W - 24, top + cmd - 9, theme.ACCENT)
    dl.text_centre(f.strong, "Пуск", W - 24 - 112, W - 24,
                   top + (cmd - f.strong.gh) // 2, (0.06, 0.07, 0.08))

    # нижняя полоса: плитки показаний, тело, области
    by0 = H - foot - bottom
    dl.rect(0, by0, W, H - foot, theme.PANEL)
    dl.hline(0, W, by0, theme.HAIRLINE)
    tiles = (("время", RUN["t"]), ("активных", RUN["active"]),
             ("пик", RUN["peak"]), ("кадров", RUN["frames"]))
    tw = 190
    for i, (lbl, val) in enumerate(tiles):
        tx = 24 + i * tw
        dl.text(f.small, lbl, tx, by0 + 20, theme.DIM)
        dl.text(f.num_big, val, tx, by0 + 20 + f.small.gh + 6, theme.INK)
        if i:
            dl.vline(tx - 18, by0 + 16, H - foot - 16, theme.HAIRLINE)
    gx = 24 + len(tiles) * tw + 6
    dl.vline(gx - 18, by0 + 16, H - foot - 16, theme.HAIRLINE)
    dl.text(f.small, "ход активности", gx, by0 + 20, theme.DIM)
    spark(dl, gx, by0 + 46, gx + 330, H - foot - 24, DEMO_SERIES, theme.ACCENT)

    rx = gx + 370
    dl.vline(rx - 18, by0 + 16, H - foot - 16, theme.HAIRLINE)
    dl.text(f.small, "области", rx, by0 + 20, theme.DIM)
    bars(dl, f, rx, by0 + 44, 300, REGIONS[:6], cols, 20, 2, 56, 38)

    if frame is not None:
        bw = 300
        bx = W - bw - 24
        dl.vline(bx - 18, by0 + 16, H - foot - 16, theme.HAIRLINE)
        dl.image(frame, bx, by0 + 40, bw, bottom - 56)
        dl.text(f.small, "тело", bx, by0 + 20, theme.DIM)

    dl.hline(0, W, H - foot, theme.HAIRLINE)
    dl.text(f.small, f"{RUN['device']} · {RUN['neurons']} нейронов", 24,
            H - foot + (foot - f.small.gh) // 2, theme.DIM)
    dl.text_right(f.num_s, f"{W}×{H} · 60 кадр/с", W - 24,
                  H - foot + (foot - f.num_s.gh) // 2, theme.FAINT)
    surf.draw(dl, f.small)
    return dl


def main():
    ti.init(arch=ti.vulkan, offline_cache=True)
    bv, cols = load_brain()
    frame = body_frame()
    f = Faces(1.0)
    surf = gpu.Surface()
    surf.ensure(W, H)

    from PIL import Image

    for name, fn, view in (("A_стенд", ref_a, 3), ("B_разворот", ref_b, 0),
                           ("C_полосы", ref_c, 3)):
        bv.view_i = view
        _, bv.yaw, bv.pitch = brainview.VIEWS[view]
        bv.refit()
        fn(surf, f, bv, cols, frame)
        ti.sync()
        a = np.clip(surf.acc.to_numpy(), 0, 1).transpose(1, 0, 2)[::-1]
        p = OUT / ("ref_" + name + ".png")
        Image.fromarray((a * 255).astype(np.uint8)).save(p)
        print("записан " + p.name)


if __name__ == "__main__":
    main()
