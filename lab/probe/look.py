"""Первый взгляд: карта мозга в новом движке.

Это не игрушка, а проверка всего кадрового пути разом, вынесенная в окно,
потому что из-под автоматики снимок экрана выходит чёрным, а размер
поверхности меняется сам по себе.

Что проверяется:
  - сплаты 138 625 нейронов своим ядром, аддитивно, палитрой magma;
  - свой атлас глифов (кириллица) поверх;
  - пересоздание полей, когда окно меняет размер;
  - мышь: вращение перетаскиванием, приближение колесом;
  - какие строки реально приходят в event.key.

Управление: тащить мышью — повернуть, колесо — приблизить,
пробел — сменить вид, R — раскрасить по областям, Esc — выход.
Нажми по очереди буквы, цифры и знаки : + = , — они попадут в отчёт.

Запуск:
    D:\\fly-brain\\.venv\\Scripts\\python.exe -I D:\\fly-brain\\lab\\probe\\look.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import taichi as ti

HERE = Path(__file__).parent
CACHE = Path(r"D:\fly-brain\cache\neurons.parquet")
BINS = Path(r"D:\fly-brain\out\demo-sugar\bins.npz")

INK = (0.93, 0.94, 0.95)
DIM = (0.66, 0.69, 0.72)
ACCENT = (0.949, 0.651, 0.231)
BG = (0.055, 0.063, 0.071)

GLYPHS = ("".join(chr(c) for c in range(32, 127))
          + "".join(chr(c) for c in range(0x410, 0x450)) + "\u0401\u0451"
          + "\u00b0\u00b1\u00d7\u00b7\u2192\u2026\u2265\u2264")


# ------------------------------------------------------------------ данные

def load():
    df = pd.read_parquet(CACHE, columns=["x_um", "y_um", "z_um", "region"])
    xyz = df[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    p = xyz[keep]
    p -= 0.5 * (p.min(0) + p.max(0))
    p /= float(np.abs(p).max())
    p[:, 1] *= -1.0
    p[:, 2] *= -1.0

    full = np.zeros(len(df), dtype=np.float32)
    try:
        z = np.load(BINS)
        idx = z["neuron_idx"].astype(np.int64)
        full[idx] = z["bins"].max(axis=0).astype(np.float32)
    except Exception:                                          # noqa: BLE001
        pass
    act = full[keep]
    pos_act = act[act > 0]
    hi = float(np.percentile(pos_act, 99)) if pos_act.size else 1.0
    act = np.clip(act / max(hi, 1e-6), 0.0, 1.0).astype(np.float32)

    reg = df["region"].to_numpy()[keep]
    names = sorted(set(str(r) for r in reg))
    code = np.array([names.index(str(r)) for r in reg], dtype=np.int32)
    return (np.ascontiguousarray(p), np.ascontiguousarray(act),
            np.ascontiguousarray(code), names)


def magma(t):
    stops = np.array([
        [0.001, 0.000, 0.014], [0.099, 0.068, 0.233], [0.232, 0.059, 0.437],
        [0.390, 0.100, 0.502], [0.550, 0.161, 0.506], [0.716, 0.215, 0.475],
        [0.868, 0.288, 0.409], [0.967, 0.440, 0.360], [0.994, 0.624, 0.427],
        [0.996, 0.793, 0.557], [0.987, 0.991, 0.750]], dtype=np.float32)
    t = np.clip(np.asarray(t, dtype=np.float32), 0, 1) * (len(stops) - 1)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(stops) - 1)
    w = (t - i0.astype(np.float32))[:, None]
    return (stops[i0] * (1 - w) + stops[i1] * w).astype(np.float32)


def region_palette(k):
    """Качественная шкала: ровная светлота, развёрнутый тон, без фиолетового."""
    out = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        h = (i * 0.618033988749895) % 1.0
        s = 0.42 + 0.18 * ((i % 3) / 2.0)
        v = 0.62 + 0.22 * ((i % 2))
        c = v * s
        x = c * (1 - abs(((h * 6) % 2) - 1))
        m = v - c
        seg = int(h * 6) % 6
        rgb = [(c, x, 0), (x, c, 0), (0, c, x),
               (0, x, c), (x, 0, c), (c, 0, x)][seg]
        out[i] = (rgb[0] + m, rgb[1] + m, rgb[2] + m)
    return out


def build_atlas(size):
    from PIL import Image, ImageDraw, ImageFont

    path = None
    for cand in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(cand).exists():
            path = cand
            break
    font = ImageFont.truetype(path, size)
    chars = list(dict.fromkeys(GLYPHS))
    widths, asc = [], 0
    for ch in chars:
        box = font.getbbox(ch)
        widths.append(max(2, box[2] - box[0] + 2))
        asc = max(asc, box[3])
    gh = asc + 3
    img = Image.new("L", (int(sum(widths)), gh), 0)
    draw = ImageDraw.Draw(img)
    gx, x = [], 0
    for ch, wd in zip(chars, widths):
        gx.append(x)
        draw.text((x, 0), ch, fill=255, font=font)
        x += wd
    atlas = np.ascontiguousarray(np.asarray(img, dtype=np.uint8).T)
    return (atlas, np.array(gx, np.int32), np.array(widths, np.int32), gh,
            {c: i for i, c in enumerate(chars)}, Path(path).name)


# ------------------------------------------------------------------ камера

VIEWS = [("фронт", 0.0, 0.0), ("сверху", 0.0, 1.35),
         ("сбоку", 1.57, 0.0), ("три четверти", 0.9, 0.55)]


def look_at(eye, target, up):
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    s = np.cross(fwd, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -fwd
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, fwd @ eye
    return m


def mvp(yaw, pitch, dist, aspect):
    e = np.array([dist * math.cos(pitch) * math.sin(yaw),
                  dist * math.sin(pitch),
                  dist * math.cos(pitch) * math.cos(yaw)], dtype=np.float32)
    t = 1.0 / math.tan(math.radians(38.0) * 0.5)
    pr = np.zeros((4, 4), np.float32)
    pr[0, 0] = t / aspect
    pr[1, 1] = t
    pr[2, 2] = 40.05 / (0.05 - 40.0)
    pr[2, 3] = 2 * 40.0 * 0.05 / (0.05 - 40.0)
    pr[3, 2] = -1.0
    return np.ascontiguousarray(pr @ look_at(e, np.zeros(3, np.float32),
                                             np.array([0.0, 1.0, 0.0], np.float32)))


# ------------------------------------------------------------------ окно

class Surface:
    """Поля кадра. Пересоздаются, когда окно меняет размер."""

    def __init__(self):
        self.w = self.h = 0
        self.tree = None

    def ensure(self, w, h):
        if w == self.w and h == self.h:
            return False
        if self.tree is not None:
            self.tree.destroy()
        self.acc = ti.Vector.field(3, ti.f32)
        self.out = ti.Vector.field(3, ti.u8)
        fb = ti.FieldsBuilder()
        fb.dense(ti.ij, (w, h)).place(self.acc, self.out)
        self.tree = fb.finalize()
        self.w, self.h = w, h
        self._compile()
        return True

    def _compile(self):
        acc, out = self.acc, self.out

        @ti.kernel
        def clear(r: ti.f32, g: ti.f32, b: ti.f32):
            for x, y in acc:
                acc[x, y] = ti.Vector([r, g, b])

        @ti.kernel
        def tone():
            for x, y in acc:
                v = ti.min(acc[x, y], ti.Vector([1.0, 1.0, 1.0]))
                out[x, y] = ti.Vector([ti.cast(v[0] * 255, ti.u8),
                                       ti.cast(v[1] * 255, ti.u8),
                                       ti.cast(v[2] * 255, ti.u8)])

        @ti.kernel
        def splat(p: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  a: ti.types.ndarray(dtype=ti.f32, ndim=1),
                  c: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  m: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  w: ti.i32, h: ti.i32, base: ti.f32):
            for i in range(p.shape[0]):
                x, y, z = p[i, 0], p[i, 1], p[i, 2]
                cw = m[3, 0] * x + m[3, 1] * y + m[3, 2] * z + m[3, 3]
                if cw > 1e-6:
                    inv = 1.0 / cw
                    sx = ((m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3])
                          * inv * .5 + .5) * w
                    # у поля начало координат внизу слева, как и у NDC
                    sy = ((m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3])
                          * inv * .5 + .5) * h
                    av = a[i]
                    rad = 1.25 + 2.6 * av
                    r2 = rad * rad
                    ir = ti.cast(ti.ceil(rad), ti.i32)
                    x0, y0 = ti.cast(sx, ti.i32), ti.cast(sy, ti.i32)
                    amp = base + (1.0 - base) * av
                    for dx in range(-ir, ir + 1):
                        for dy in range(-ir, ir + 1):
                            px, py = x0 + dx, y0 + dy
                            if 0 <= px < w and 0 <= py < h:
                                d2 = ti.cast(dx * dx + dy * dy, ti.f32)
                                if d2 <= r2:
                                    wt = amp * ti.exp(-2.5 * d2 / r2)
                                    ti.atomic_add(acc[px, py][0], c[i, 0] * wt)
                                    ti.atomic_add(acc[px, py][1], c[i, 1] * wt)
                                    ti.atomic_add(acc[px, py][2], c[i, 2] * wt)

        @ti.kernel
        def rects(cmd: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  cnt: ti.i32, tw: ti.i32, th: ti.i32, w: ti.i32, h: ti.i32):
            for k, j in ti.ndrange(cnt, tw * th):
                x0 = ti.cast(cmd[k, 0], ti.i32)
                y0 = ti.cast(cmd[k, 1], ti.i32)
                x1 = ti.cast(cmd[k, 2], ti.i32)
                y1 = ti.cast(cmd[k, 3], ti.i32)
                x, y = x0 + j % tw, y0 + j // tw
                fy = h - 1 - y          # координаты панелей сверху вниз
                if x < x1 and y < y1 and 0 <= x < w and 0 <= fy < h:
                    al = cmd[k, 7]
                    o = acc[x, fy]
                    acc[x, fy] = ti.Vector([o[0] * (1 - al) + cmd[k, 4] * al,
                                            o[1] * (1 - al) + cmd[k, 5] * al,
                                            o[2] * (1 - al) + cmd[k, 6] * al])

        @ti.kernel
        def text(atlas: ti.types.ndarray(dtype=ti.u8, ndim=2),
                 gx: ti.types.ndarray(dtype=ti.i32, ndim=1),
                 gw: ti.types.ndarray(dtype=ti.i32, ndim=1),
                 ig: ti.types.ndarray(dtype=ti.i32, ndim=1),
                 ix: ti.types.ndarray(dtype=ti.i32, ndim=1),
                 iy: ti.types.ndarray(dtype=ti.i32, ndim=1),
                 ic: ti.types.ndarray(dtype=ti.f32, ndim=2),
                 gh: ti.i32, mw: ti.i32, w: ti.i32, h: ti.i32):
            for k, j in ti.ndrange(ig.shape[0], mw * gh):
                u, v = j % mw, j // mw
                if u < gw[ig[k]]:
                    px = ix[k] + u
                    py = h - 1 - (iy[k] + v)     # строка глифа идёт сверху вниз
                    if 0 <= px < w and 0 <= py < h:
                        al = ti.cast(atlas[gx[ig[k]] + u, v], ti.f32) / 255.0
                        if al > 0.004:
                            o = acc[px, py]
                            acc[px, py] = ti.Vector([o[0] * (1 - al) + ic[k, 0] * al,
                                                     o[1] * (1 - al) + ic[k, 1] * al,
                                                     o[2] * (1 - al) + ic[k, 2] * al])

        self.clear, self.tone, self.splat, self.rects, self.text = (
            clear, tone, splat, rects, text)


class TextBatch:
    """Накопитель команд: все подписи кадра — один запуск ядра."""

    def __init__(self, atlas, gx, gw, gh, index, maxw):
        self.atlas, self.gx, self.gw, self.gh = atlas, gx, gw, gh
        self.index, self.maxw = index, maxw
        self.reset()

    def reset(self):
        self.g, self.x, self.y, self.c = [], [], [], []

    def add(self, s, x, y, color=INK):
        for ch in s:
            gi = self.index.get(ch, self.index["?"])
            self.g.append(gi)
            self.x.append(int(x))
            self.y.append(int(y))
            self.c.append(color)
            x += int(self.gw[gi])
        return x

    def width(self, s):
        return sum(int(self.gw[self.index.get(ch, self.index["?"])]) for ch in s)

    def flush(self, surf):
        if not self.g:
            return
        surf.text(self.atlas, self.gx, self.gw,
                  np.array(self.g, np.int32), np.array(self.x, np.int32),
                  np.array(self.y, np.int32),
                  np.ascontiguousarray(np.array(self.c, np.float32)),
                  self.gh, self.maxw, surf.w, surf.h)
        self.reset()


def main():
    t0 = time.perf_counter()
    pos, act, code, names = load()
    n = pos.shape[0]
    pal = region_palette(len(names))
    col_act = np.ascontiguousarray(magma(act))
    col_reg = np.ascontiguousarray(pal[code] * (0.25 + 0.75 * act[:, None]))
    load_s = time.perf_counter() - t0

    ti.init(arch=ti.vulkan, offline_cache=True)

    window = ti.ui.Window("Мозг мухи — первый взгляд", (1500, 880),
                          vsync=True, pos=(80, 60))
    canvas = window.get_canvas()
    surf = Surface()

    # Кегль и отступы считаются от площади окна, а не прибиты к пикселям:
    # окно здесь само меняет размер и бывает портретным.
    REF = (1500.0 * 880.0) ** 0.5
    atlas = batch = None
    fontname = ""
    size_now = -1

    def fit(w, h):
        nonlocal atlas, batch, fontname, size_now
        s = max(0.78, min(2.0, (w * h) ** 0.5 / REF))
        size = max(11, min(30, int(round(14 * s))))
        if size != size_now:
            at, gx, gw, gh, index, fontname = build_atlas(size)
            atlas = at
            batch = TextBatch(at, gx, gw, gh, index, int(gw.max()))
            size_now = size
        return s

    yaw, pitch, dist = 0.0, 0.0, 3.0
    view_i, by_region = 0, False
    drag = None
    keys_seen, key_log = [], []
    fps, last, frames, tmark = 0.0, time.perf_counter(), 0, time.perf_counter()
    resizes = []
    import os

    limit = float(os.environ.get("LOOK_SECONDS", "0"))
    started = time.perf_counter()

    while window.running:
        if limit and time.perf_counter() - started > limit:
            break
        w, h = window.get_window_shape()
        if surf.ensure(int(w), int(h)):
            resizes.append((int(w), int(h)))

        for e in window.get_events(ti.ui.PRESS):
            k = str(e.key)
            if k not in keys_seen:
                keys_seen.append(k)
            key_log.append(k)
            if k == ti.ui.ESCAPE:
                window.running = False
            elif k == ti.ui.SPACE:
                view_i = (view_i + 1) % len(VIEWS)
                _, yaw, pitch = VIEWS[view_i]
            elif k in ("r", "R", "к", "К"):
                by_region = not by_region
            elif k == ti.ui.UP:
                dist = max(1.05, dist * 0.9)
            elif k == ti.ui.DOWN:
                dist = min(9.0, dist * 1.1)

        cx, cy = window.get_cursor_pos()
        if window.is_pressed(ti.ui.LMB):
            if drag is not None:
                yaw -= (cx - drag[0]) * 4.0
                pitch = max(-1.45, min(1.45, pitch + (cy - drag[1]) * 4.0))
            drag = (cx, cy)
        else:
            drag = None

        s = fit(surf.w, surf.h)
        m = mvp(yaw, pitch, dist, surf.w / max(surf.h, 1))
        surf.clear(*BG)
        surf.splat(pos, act, col_reg if by_region else col_act,
                   m, surf.w, surf.h, 0.14)

        pad = int(round(14 * s))
        lh = batch.gh + int(round(4 * s))
        lines = [
            ("Мозг мухи", INK),
            ("FlyWire v783 · " + str(n) + " нейронов", DIM),
            ("активных " + str(int((act > 0).sum())) + " · demo-sugar", DIM),
            ("вид: " + VIEWS[view_i][0] + " · "
             + ("области" if by_region else "активность"), ACCENT),
            (str(surf.w) + "×" + str(surf.h) + " · " + format(fps, ".0f")
             + " кадр/с · кегль " + str(size_now), DIM),
            ("тащить · колесо · пробел · R · Esc", DIM),
        ]
        inner = int(round(12 * s))
        pw = max(batch.width(t) for t, _ in lines) + inner * 2
        pw = min(pw, surf.w - pad * 2)
        ph = lh * len(lines) + inner * 2
        panel = np.array([[pad, pad, pad + pw, pad + ph,
                           0.082, 0.094, 0.106, 0.92]], dtype=np.float32)
        surf.rects(np.ascontiguousarray(panel), 1, pw, ph, surf.w, surf.h)
        for i, (txt, cl) in enumerate(lines):
            batch.add(txt, pad + inner, pad + inner + i * lh, cl)

        foot = [
            ("нажми буквы, цифры и знаки : + = , — проверяю клавиатуру", DIM),
            ("поймано " + str(len(keys_seen)) + ": "
             + " ".join(keys_seen[-18:]), INK),
        ]
        fw = max(batch.width(t) for t, _ in foot) + inner * 2
        fw = min(fw, surf.w - pad * 2)
        fh = lh * len(foot) + inner * 2
        fy = surf.h - pad - fh
        fpanel = np.array([[pad, fy, pad + fw, fy + fh,
                            0.082, 0.094, 0.106, 0.92]], dtype=np.float32)
        surf.rects(np.ascontiguousarray(fpanel), 1, fw, fh, surf.w, surf.h)
        for i, (txt, cl) in enumerate(foot):
            batch.add(txt, pad + inner, fy + inner + i * lh, cl)
        batch.flush(surf)

        surf.tone()
        canvas.set_image(surf.out)
        window.show()

        frames += 1
        now = time.perf_counter()
        if now - tmark > 0.4:
            fps = frames / (now - tmark)
            frames, tmark = 0, now

    report = {
        "neurons": int(n),
        "active": int((act > 0).sum()),
        "load_s": round(load_s, 2),
        "font": fontname,
        "atlas": list(atlas.shape),
        "surfaces": resizes,
        "keys_distinct": keys_seen,
        "keys_count": len(key_log),
    }
    (HERE / "look_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nотчёт: " + str(HERE / "look_report.json"))


if __name__ == "__main__":
    main()
