"""Shag 0: probe Taichi GGUI before rewriting the UI of D:\\fly-brain.

Gates (a failure here changes the design, so they run first):
  A  arch + atomics: does arch=ti.vulkan open a window, does ti.atomic_add
     compile for f32 fields, and does it compile for i32 fields (the
     fixed-point fallback).
  B  set_image contract: axis order and origin, proven with save_image + PIL
     so the answer is not circular; cost of u8 field vs f32 field vs numpy.
  C  own splat rasteriser on the real 138k soma coordinates, at three camera
     distances -- the close one is the worst case for atomic contention.
  D  scene.particles on the same points, for comparison.
  E  do set_image and scene coexist in one frame.
  F  glyph atlas from a TTF, and the decisive measurement: 400 separate
     kernel launches vs one kernel over a command buffer.
  G  readback cost.
  H  does mujoco.Renderer work in the same process as a taichi window, and
     what does it cost to blit a body frame from shared memory.
  I  optional: which strings arrive in event.key  (--keys SECONDS)

ASCII only on purpose.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

W, H = 1600, 900
FIX = 4096.0                      # fixed-point scale for the i32 accumulator
CACHE = Path(r"D:\fly-brain\cache\neurons.parquet")
BINS = Path(r"D:\fly-brain\out\demo-sugar\bins.npz")
RESULTS = {}


def say(tag, ok, detail=""):
    mark = "[OK  ]" if ok else "[FAIL]"
    print(mark + " " + tag + ((": " + detail) if detail else ""), flush=True)
    RESULTS[tag] = {"ok": bool(ok), "detail": detail}


def f(x, nd=2):
    return format(x, "." + str(nd) + "f")


# ------------------------------------------------------------------ data

def load_positions():
    import pandas as pd

    df = pd.read_parquet(CACHE, columns=["x_um", "y_um", "z_um"])
    p = df.to_numpy(dtype=np.float32)
    keep = np.isfinite(p).all(axis=1)
    p = p[keep]
    lo, hi = p.min(0), p.max(0)
    p = p - 0.5 * (lo + hi)
    p /= float(np.abs(p).max())
    p[:, 1] *= -1.0               # viewer.js: y grows ventrally in FlyWire
    p[:, 2] *= -1.0
    return np.ascontiguousarray(p), int(len(keep)), int(keep.sum())


def load_activity(n, n_all, keep_mask_len):
    """Real activity from a saved run, so the density is not uniform noise."""
    act = np.zeros(n, dtype=np.float32)
    note = "uniform fallback"
    try:
        z = np.load(BINS)
        idx = z["neuron_idx"].astype(np.int64)
        bins = z["bins"]
        peak = bins.max(axis=0).astype(np.float32)
        full = np.zeros(n_all, dtype=np.float32)
        ok = idx < n_all
        full[idx[ok]] = peak[ok]
        # positions were filtered by finite soma coords; re-apply the same mask
        import pandas as pd

        df = pd.read_parquet(CACHE, columns=["x_um", "y_um", "z_um"])
        m = np.isfinite(df.to_numpy(dtype=np.float32)).all(axis=1)
        act = full[m].astype(np.float32)
        hi = float(np.percentile(act[act > 0], 99)) if (act > 0).any() else 1.0
        act = np.clip(act / max(hi, 1e-6), 0.0, 1.0)
        note = str(int((act > 0).sum())) + " active of " + str(n) + " (from demo-sugar)"
    except Exception as exc:                                   # noqa: BLE001
        rng = np.random.default_rng(0)
        act = (rng.random(n).astype(np.float32) ** 3)
        note = "uniform fallback: " + repr(exc)[:70]
    return np.ascontiguousarray(act), note


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


def perspective(fovy_deg, aspect, zn, zf):
    t = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = t / aspect
    m[1, 1] = t
    m[2, 2] = (zf + zn) / (zn - zf)
    m[2, 3] = 2.0 * zf * zn / (zn - zf)
    m[3, 2] = -1.0
    return m


def mvp_at(eye_z):
    return np.ascontiguousarray(
        perspective(38.0, W / H, 0.05, 40.0) @ look_at(
            np.array([0.0, 0.0, eye_z], np.float32),
            np.zeros(3, np.float32),
            np.array([0.0, 1.0, 0.0], np.float32)))


def magma(t):
    stops = np.array([
        [0.001, 0.000, 0.014], [0.099, 0.068, 0.233], [0.232, 0.059, 0.437],
        [0.390, 0.100, 0.502], [0.550, 0.161, 0.506], [0.716, 0.215, 0.475],
        [0.868, 0.288, 0.409], [0.967, 0.440, 0.360], [0.994, 0.624, 0.427],
        [0.996, 0.793, 0.557], [0.987, 0.991, 0.750],
    ], dtype=np.float32)
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(stops) - 1)
    w = (t - i0)[:, None]
    return (stops[i0] * (1 - w) + stops[i1] * w).astype(np.float32)


GLYPHS = ("".join(chr(c) for c in range(32, 127))
          + "".join(chr(c) for c in range(0x410, 0x450)) + "\u0401\u0451"
          + "\u00b0\u00b1\u00d7\u00b7\u2192\u2026\u2265\u2264\u2500\u2502\u25b2\u25bc\u25a0")


def build_atlas(size):
    from PIL import Image, ImageDraw, ImageFont

    path = None
    for cand in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\segoeui.ttf",
                 r"C:\Windows\Fonts\arial.ttf"):
        if Path(cand).exists():
            path = cand
            break
    if path is None:
        raise RuntimeError("no TTF found")
    font = ImageFont.truetype(path, size)

    chars = list(dict.fromkeys(GLYPHS))
    widths, asc = [], 0
    for ch in chars:
        box = font.getbbox(ch)
        widths.append(max(2, box[2] - box[0] + 2))
        asc = max(asc, box[3])
    gh = asc + 3
    total = int(sum(widths))
    img = Image.new("L", (total, gh), 0)
    draw = ImageDraw.Draw(img)
    gx, x = [], 0
    for ch, wd in zip(chars, widths):
        gx.append(x)
        draw.text((x, 0), ch, fill=255, font=font)
        x += wd
    a = np.asarray(img, dtype=np.uint8)                 # (gh, total), top-down
    atlas = np.ascontiguousarray(a.T)                   # (total, gh), x-major
    missing = [c for c, g, wd in zip(chars, gx, widths)
               if atlas[g:g + wd, :].max() == 0 and c != " "]
    return (atlas, np.array(gx, np.int32), np.array(widths, np.int32), gh,
            chars, Path(path).name, missing)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--keys", type=float, default=0.0)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    pos, n_all, n_fin = load_positions()
    n = pos.shape[0]
    act, act_note = load_activity(n, n_all, n_all)
    col = np.ascontiguousarray(magma(act))
    say("data", n > 100000,
        str(n) + " of " + str(n_all) + " have finite soma coords; " + act_note
        + "; loaded in " + f(time.perf_counter() - t_load) + "s")

    # ---- A. arch + atomics gate
    import taichi as ti

    t0 = time.perf_counter()
    arch = "vulkan"
    try:
        ti.init(arch=ti.vulkan, offline_cache=True)
    except Exception as exc:                                   # noqa: BLE001
        say("A arch vulkan", False, repr(exc)[:200])
        ti.init(arch=ti.gpu, offline_cache=True)
        arch = "gpu-fallback"
    say("A ti.init", True, arch + ", " + f(time.perf_counter() - t0) + "s, taichi "
        + str(ti.__version__))

    probe_f32 = ti.field(ti.f32, shape=(64,))
    probe_i32 = ti.field(ti.i32, shape=(64,))
    atomic_f32 = atomic_i32 = False
    try:
        @ti.kernel
        def k_at_f32():
            for i in range(1024):
                ti.atomic_add(probe_f32[i % 64], 1.0)

        k_at_f32()
        ti.sync()
        atomic_f32 = abs(float(probe_f32.to_numpy().sum()) - 1024.0) < 1.0
    except Exception as exc:                                   # noqa: BLE001
        say("A atomic f32", False, repr(exc)[:200])
    try:
        @ti.kernel
        def k_at_i32():
            for i in range(1024):
                ti.atomic_add(probe_i32[i % 64], 1)

        k_at_i32()
        ti.sync()
        atomic_i32 = int(probe_i32.to_numpy().sum()) == 1024
    except Exception as exc:                                   # noqa: BLE001
        say("A atomic i32", False, repr(exc)[:200])
    say("A atomics", atomic_f32 or atomic_i32,
        "f32 " + ("works" if atomic_f32 else "NOT available")
        + ", i32 " + ("works" if atomic_i32 else "NOT available")
        + " -> accumulator: " + ("f32" if atomic_f32 else "i32 fixed-point"))

    # ---- fields
    acc_f = ti.Vector.field(3, ti.f32, shape=(W, H))
    acc_i = ti.Vector.field(3, ti.i32, shape=(W, H))
    rgb8 = ti.Vector.field(3, ti.u8, shape=(W, H))
    rgbf = ti.Vector.field(3, ti.f32, shape=(W, H))

    @ti.kernel
    def k_clear_f(r: ti.f32, g: ti.f32, b: ti.f32):
        for x, y in acc_f:
            acc_f[x, y] = ti.Vector([r, g, b])

    @ti.kernel
    def k_clear_i(r: ti.f32, g: ti.f32, b: ti.f32):
        for x, y in acc_i:
            acc_i[x, y] = ti.Vector([ti.cast(r * FIX, ti.i32),
                                     ti.cast(g * FIX, ti.i32),
                                     ti.cast(b * FIX, ti.i32)])

    @ti.kernel
    def k_tone_f():
        for x, y in acc_f:
            v = ti.min(acc_f[x, y], ti.Vector([1.0, 1.0, 1.0]))
            rgb8[x, y] = ti.Vector([ti.cast(v[0] * 255, ti.u8),
                                    ti.cast(v[1] * 255, ti.u8),
                                    ti.cast(v[2] * 255, ti.u8)])

    @ti.kernel
    def k_tone_i():
        for x, y in acc_i:
            for c in ti.static(range(3)):
                v = ti.min(ti.cast(acc_i[x, y][c], ti.f32) / FIX, 1.0)
                rgb8[x, y][c] = ti.cast(v * 255, ti.u8)

    @ti.kernel
    def k_splat_f(p: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  a: ti.types.ndarray(dtype=ti.f32, ndim=1),
                  c: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  m: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  w: ti.i32, h: ti.i32, gain: ti.f32):
        for i in range(p.shape[0]):
            x, y, z = p[i, 0], p[i, 1], p[i, 2]
            cw = m[3, 0] * x + m[3, 1] * y + m[3, 2] * z + m[3, 3]
            if cw > 1e-6:
                inv = 1.0 / cw
                sx = ((m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3]) * inv * .5 + .5) * w
                sy = ((m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3]) * inv * .5 + .5) * h
                av = a[i]
                rad = 1.3 + 2.6 * av
                r2 = rad * rad
                ir = ti.cast(ti.ceil(rad), ti.i32)
                x0, y0 = ti.cast(sx, ti.i32), ti.cast(sy, ti.i32)
                amp = gain * (0.16 + 0.84 * av)
                for dx in range(-ir, ir + 1):
                    for dy in range(-ir, ir + 1):
                        px, py = x0 + dx, y0 + dy
                        if 0 <= px < w and 0 <= py < h:
                            d2 = ti.cast(dx * dx + dy * dy, ti.f32)
                            if d2 <= r2:
                                wt = amp * ti.exp(-2.5 * d2 / r2)
                                ti.atomic_add(acc_f[px, py][0], c[i, 0] * wt)
                                ti.atomic_add(acc_f[px, py][1], c[i, 1] * wt)
                                ti.atomic_add(acc_f[px, py][2], c[i, 2] * wt)

    @ti.kernel
    def k_splat_i(p: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  a: ti.types.ndarray(dtype=ti.f32, ndim=1),
                  c: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  m: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  w: ti.i32, h: ti.i32, gain: ti.f32):
        for i in range(p.shape[0]):
            x, y, z = p[i, 0], p[i, 1], p[i, 2]
            cw = m[3, 0] * x + m[3, 1] * y + m[3, 2] * z + m[3, 3]
            if cw > 1e-6:
                inv = 1.0 / cw
                sx = ((m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3]) * inv * .5 + .5) * w
                sy = ((m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3]) * inv * .5 + .5) * h
                av = a[i]
                rad = 1.3 + 2.6 * av
                r2 = rad * rad
                ir = ti.cast(ti.ceil(rad), ti.i32)
                x0, y0 = ti.cast(sx, ti.i32), ti.cast(sy, ti.i32)
                amp = gain * (0.16 + 0.84 * av) * FIX
                for dx in range(-ir, ir + 1):
                    for dy in range(-ir, ir + 1):
                        px, py = x0 + dx, y0 + dy
                        if 0 <= px < w and 0 <= py < h:
                            d2 = ti.cast(dx * dx + dy * dy, ti.f32)
                            if d2 <= r2:
                                wt = amp * ti.exp(-2.5 * d2 / r2)
                                ti.atomic_add(acc_i[px, py][0], ti.cast(c[i, 0] * wt, ti.i32))
                                ti.atomic_add(acc_i[px, py][1], ti.cast(c[i, 1] * wt, ti.i32))
                                ti.atomic_add(acc_i[px, py][2], ti.cast(c[i, 2] * wt, ti.i32))

    @ti.kernel
    def k_one_rect(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32,
                   r: ti.f32, g: ti.f32, b: ti.f32):
        for x, y in ti.ndrange((x0, x1), (y0, y1)):
            rgbf[x, y] = ti.Vector([r, g, b])

    @ti.kernel
    def k_many_rects(cmd: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for k in range(cmd.shape[0]):
            x0 = ti.cast(cmd[k, 0], ti.i32)
            y0 = ti.cast(cmd[k, 1], ti.i32)
            x1 = ti.cast(cmd[k, 2], ti.i32)
            y1 = ti.cast(cmd[k, 3], ti.i32)
            for x in range(x0, x1):
                for y in range(y0, y1):
                    rgbf[x, y] = ti.Vector([cmd[k, 4], cmd[k, 5], cmd[k, 6]])

    @ti.kernel
    def k_text(atlas: ti.types.ndarray(dtype=ti.u8, ndim=2),
               gx: ti.types.ndarray(dtype=ti.i32, ndim=1),
               gw: ti.types.ndarray(dtype=ti.i32, ndim=1),
               ig: ti.types.ndarray(dtype=ti.i32, ndim=1),
               ix: ti.types.ndarray(dtype=ti.i32, ndim=1),
               iy: ti.types.ndarray(dtype=ti.i32, ndim=1),
               gh: ti.i32, w: ti.i32, h: ti.i32,
               r: ti.f32, g: ti.f32, b: ti.f32):
        for k in range(ig.shape[0]):
            gi = ig[k]
            ax, aw = gx[gi], gw[gi]
            bx, by = ix[k], iy[k]
            for u in range(aw):
                for v in range(gh):
                    px, py = bx + u, by + v
                    if 0 <= px < w and 0 <= py < h:
                        al = ti.cast(atlas[ax + u, gh - 1 - v], ti.f32) / 255.0
                        if al > 0.004:
                            o = rgbf[px, py]
                            rgbf[px, py] = ti.Vector([o[0] * (1 - al) + r * al,
                                                      o[1] * (1 - al) + g * al,
                                                      o[2] * (1 - al) + b * al])

    @ti.kernel
    def k_blit(src: ti.types.ndarray(dtype=ti.u8, ndim=3),
               x0: ti.i32, y0: ti.i32, dw: ti.i32, dh: ti.i32):
        sh, sw = src.shape[0], src.shape[1]
        for i, j in ti.ndrange(dw, dh):
            su = ti.cast(ti.cast(i, ti.f32) / dw * sw, ti.i32)
            sv = ti.cast(ti.cast(dh - 1 - j, ti.f32) / dh * sh, ti.i32)
            px, py = x0 + i, y0 + j
            if 0 <= px < W and 0 <= py < H:
                rgbf[px, py] = ti.Vector([ti.cast(src[sv, su, 0], ti.f32) / 255.0,
                                          ti.cast(src[sv, su, 1], ti.f32) / 255.0,
                                          ti.cast(src[sv, su, 2], ti.f32) / 255.0])

    window = ti.ui.Window("fly-brain probe", (W, H), vsync=False, fps_limit=10000)
    canvas = window.get_canvas()
    say("A window", True, "shape " + str(window.get_window_shape()))

    # ---- B. set_image contract
    marker = np.zeros((W, H, 3), dtype=np.float32)
    marker[0:40, 0:40, :] = 1.0                 # block at index [0, 0]
    marker[200:204, :, 0] = 1.0                 # red column at first index 200
    marker[:, 100:104, 1] = 1.0                 # green row at second index 100
    canvas.set_image(np.ascontiguousarray(marker))
    window.show()
    axis_png = out / "probe_axes.png"
    window.save_image(str(axis_png))
    try:
        from PIL import Image

        im = np.asarray(Image.open(axis_png).convert("RGB"), dtype=np.float32) / 255.0
        ih, iw, _ = im.shape
        corners = {
            "top-left": im[0:30, 0:30].mean(),
            "top-right": im[0:30, iw - 30:].mean(),
            "bottom-left": im[ih - 30:, 0:30].mean(),
            "bottom-right": im[ih - 30:, iw - 30:].mean(),
        }
        where = max(corners, key=corners.get)
        red_col = int(np.argmax(im[:, :, 0].mean(axis=0) * (im[:, :, 1].mean(axis=0) < 0.5)))
        grn_row = int(np.argmax(im[:, :, 1].mean(axis=1) * (im[:, :, 0].mean(axis=1) < 0.5)))
        say("B axes", where != "", "png " + str(iw) + "x" + str(ih)
            + "; index [0,0] lands " + where
            + "; first index 200 -> column " + str(red_col)
            + "; second index 100 -> row " + str(grn_row))
    except Exception as exc:                                   # noqa: BLE001
        say("B axes", False, repr(exc)[:200])

    variants = {}
    np_u8 = np.ascontiguousarray((marker * 255).astype(np.uint8))
    np_f32 = np.ascontiguousarray(marker)
    rgb8.from_numpy(np_u8)
    rgbf.from_numpy(np_f32)
    for name, payload in (("field u8", rgb8), ("field f32", rgbf),
                          ("numpy u8", np_u8), ("numpy f32", np_f32)):
        try:
            for _ in range(10):
                canvas.set_image(payload)
                window.show()
            t0 = time.perf_counter()
            for _ in range(150):
                canvas.set_image(payload)
                window.show()
            variants[name] = (time.perf_counter() - t0) * 1000 / 150
        except Exception as exc:                               # noqa: BLE001
            variants[name] = None
            say("B set_image " + name, False, repr(exc)[:140])
    ok_v = {k: v for k, v in variants.items() if v is not None}
    say("B set_image cost", bool(ok_v),
        ", ".join(k + " " + f(v) + " ms" for k, v in ok_v.items())
        + ("; cheapest " + min(ok_v, key=ok_v.get) if ok_v else ""))

    # ---- C. own rasteriser, three camera distances
    use_i32 = not atomic_f32
    k_clear = k_clear_i if use_i32 else k_clear_f
    k_splat = k_splat_i if use_i32 else k_splat_f
    k_tone = k_tone_i if use_i32 else k_tone_f
    rast = {}
    for label, eye_z in (("far 40%", 7.0), ("fit 100%", 3.0), ("close 300%", 1.15)):
        m = mvp_at(eye_z)
        for _ in range(10):
            k_clear(0.055, 0.063, 0.071)
            k_splat(pos, act, col, m, W, H, 1.0)
            k_tone()
        ti.sync()
        t0 = time.perf_counter()
        for _ in range(args.frames):
            k_clear(0.055, 0.063, 0.071)
            k_splat(pos, act, col, m, W, H, 1.0)
            k_tone()
        ti.sync()
        kern = (time.perf_counter() - t0) * 1000 / args.frames
        t0 = time.perf_counter()
        for _ in range(args.frames):
            k_clear(0.055, 0.063, 0.071)
            k_splat(pos, act, col, m, W, H, 1.0)
            k_tone()
            canvas.set_image(rgb8)
            window.show()
        full = (time.perf_counter() - t0) * 1000 / args.frames
        rast[label] = (kern, full)
        window.save_image(str(out / ("probe_splat_" + label.split()[0] + ".png")))
    worst = max(v[1] for v in rast.values())
    say("C own rasteriser", worst <= 16.0,
        ("i32 fixed-point" if use_i32 else "f32") + " accumulator, " + str(n) + " points; "
        + "; ".join(k + ": kernels " + f(v[0]) + " / frame " + f(v[1]) + " ms ("
                    + f(1000 / v[1], 0) + " fps)" for k, v in rast.items())
        + "; worst " + f(worst) + " ms, budget 16")

    # ---- D. scene.particles
    scene = camera = pos_f = col_f = None
    try:
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(0.0, 0.0, 3.0)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(38.0)
        pos_f = ti.Vector.field(3, ti.f32, shape=n)
        col_f = ti.Vector.field(3, ti.f32, shape=n)
        pos_f.from_numpy(pos)
        col_f.from_numpy(col)

        def draw_scene():
            scene.set_camera(camera)
            scene.ambient_light((0.45, 0.45, 0.45))
            scene.point_light(pos=(0.0, 0.0, 3.0), color=(1.0, 1.0, 1.0))
            scene.particles(pos_f, radius=0.0022, per_vertex_color=col_f)
            canvas.scene(scene)
            window.show()

        for _ in range(10):
            draw_scene()
        t0 = time.perf_counter()
        for _ in range(args.frames):
            draw_scene()
        part_ms = (time.perf_counter() - t0) * 1000 / args.frames
        window.save_image(str(out / "probe_particles.png"))
        say("D scene.particles", True,
            f(part_ms) + " ms/frame (" + f(1000 / part_ms, 0) + " fps) with ti fields; "
            + "own rasteriser at fit is " + f(rast["fit 100%"][1]) + " ms")
    except Exception as exc:                                   # noqa: BLE001
        say("D scene.particles", False, repr(exc)[:220])

    # ---- E. set_image together with scene
    try:
        bg = np.zeros((W, H, 3), dtype=np.float32)
        bg[:, :, 2] = 0.35
        bg[W - W // 10:, H - H // 10:, 0] = 1.0
        canvas.set_image(np.ascontiguousarray(bg))
        scene.set_camera(camera)
        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(0.0, 0.0, 3.0), color=(1.0, 1.0, 1.0))
        scene.particles(pos_f, radius=0.0022, per_vertex_color=col_f)
        canvas.scene(scene)
        window.show()
        both = out / "probe_coexist.png"
        window.save_image(str(both))
        from PIL import Image

        im = np.asarray(Image.open(both).convert("RGB"), dtype=np.float32) / 255.0
        ih, iw, _ = im.shape
        blue = float(im[:, :, 2].mean())
        mid = float(im[ih // 2 - 80:ih // 2 + 80, iw // 2 - 80:iw // 2 + 80].max())
        say("E set_image + scene", True,
            "mean blue " + f(blue, 3) + " (background survives if > 0.08), brightest centre "
            + f(mid, 3) + " (particles drawn if > 0.5) -> "
            + ("both layers" if blue > 0.08 and mid > 0.5 else "only one layer"))
    except Exception as exc:                                   # noqa: BLE001
        say("E set_image + scene", False, repr(exc)[:220])

    # ---- F. glyph atlas and kernel launch overhead
    try:
        t0 = time.perf_counter()
        atlas, gx, gw, gh, chars, fname, missing = build_atlas(14)
        bake_s = time.perf_counter() - t0
        index = {c: i for i, c in enumerate(chars)}
        line = ("AVLP perednij ventrolateralnyj protocerebrum 1234.56 Gc "
                "\u0430\u043a\u0442\u0438\u0432\u043d\u044b\u0445 9110 \u043f\u0438\u043a 312 \u0413\u0446 ")
        ig, ix, iy, cx, cy = [], [], [], 16, 40
        total_lines = 0
        while total_lines < 400:
            for ch in line:
                gi = index.get(ch, index["?"])
                ig.append(gi)
                ix.append(cx)
                iy.append(cy)
                cx += int(gw[gi])
            total_lines += 1
            cx, cy = 16, cy + gh + 2
            if cy > H - gh - 4:
                cy = 40
        ig = np.array(ig, np.int32)
        ix = np.array(ix, np.int32)
        iy = np.array(iy, np.int32)
        for _ in range(6):
            k_text(atlas, gx, gw, ig, ix, iy, gh, W, H, 0.93, 0.94, 0.95)
        ti.sync()
        t0 = time.perf_counter()
        for _ in range(40):
            k_text(atlas, gx, gw, ig, ix, iy, gh, W, H, 0.93, 0.94, 0.95)
        ti.sync()
        text_ms = (time.perf_counter() - t0) * 1000 / 40
        canvas.set_image(rgbf)
        window.show()
        window.save_image(str(out / "probe_text.png"))
        say("F glyph atlas", text_ms <= 3.0 and not missing,
            fname + ", atlas " + str(atlas.shape) + " = "
            + f(atlas.nbytes / 1e6) + " MB, " + str(len(chars)) + " glyphs, gh=" + str(gh)
            + ", baked in " + f(bake_s) + "s; " + str(len(ig)) + " glyphs ("
            + str(total_lines) + " lines) in " + f(text_ms, 3) + " ms/frame, budget 3; "
            + ("missing: " + " ".join(missing) if missing else "no missing glyphs"))
    except Exception as exc:                                   # noqa: BLE001
        say("F glyph atlas", False, repr(exc)[:220])

    try:
        rects = np.zeros((400, 7), dtype=np.float32)
        rng = np.random.default_rng(1)
        rects[:, 0] = rng.integers(0, W - 60, 400)
        rects[:, 1] = rng.integers(0, H - 40, 400)
        rects[:, 2] = rects[:, 0] + 50
        rects[:, 3] = rects[:, 1] + 30
        rects[:, 4:] = rng.random((400, 3))
        rects = np.ascontiguousarray(rects)
        for _ in range(3):
            k_many_rects(rects)
            for r in rects[:20]:
                k_one_rect(int(r[0]), int(r[1]), int(r[2]), int(r[3]),
                           float(r[4]), float(r[5]), float(r[6]))
        ti.sync()
        t0 = time.perf_counter()
        for _ in range(20):
            k_many_rects(rects)
        ti.sync()
        batched_ms = (time.perf_counter() - t0) * 1000 / 20
        t0 = time.perf_counter()
        for _ in range(5):
            for r in rects:
                k_one_rect(int(r[0]), int(r[1]), int(r[2]), int(r[3]),
                           float(r[4]), float(r[5]), float(r[6]))
        ti.sync()
        perwidget_ms = (time.perf_counter() - t0) * 1000 / 5
        per_launch_us = perwidget_ms * 1000 / 400
        say("F launch overhead", per_launch_us < 30.0,
            "400 rects: one batched kernel " + f(batched_ms, 3) + " ms vs 400 launches "
            + f(perwidget_ms) + " ms (" + f(per_launch_us, 1)
            + " us per launch); command buffer is "
            + ("optional" if per_launch_us < 30 else "MANDATORY"))
    except Exception as exc:                                   # noqa: BLE001
        say("F launch overhead", False, repr(exc)[:220])

    # ---- G. readback
    try:
        for _ in range(3):
            window.get_image_buffer_as_numpy()
        t0 = time.perf_counter()
        for _ in range(20):
            buf = window.get_image_buffer_as_numpy()
        read_ms = (time.perf_counter() - t0) * 1000 / 20
        say("G readback", True, str(buf.shape) + " " + str(buf.dtype) + " in "
            + f(read_ms) + " ms (debug tool if slow, not a frame path)")
    except Exception as exc:                                   # noqa: BLE001
        say("G readback", False, repr(exc)[:200])

    # ---- H. mujoco in the same process + body frame blit
    try:
        import mujoco

        xml = ("<mujoco><worldbody><light pos='0 0 2'/>"
               "<geom type='plane' size='2 2 .1'/>"
               "<body pos='0 0 .3'><freejoint/><geom type='sphere' size='.1'/></body>"
               "</worldbody></mujoco>")
        mm = mujoco.MjModel.from_xml_string(xml)
        md = mujoco.MjData(mm)
        mujoco.mj_forward(mm, md)
        t0 = time.perf_counter()
        ren = mujoco.Renderer(mm, 480, 640)
        first_s = time.perf_counter() - t0
        ren.update_scene(md)
        frame = ren.render()
        t0 = time.perf_counter()
        for _ in range(20):
            ren.update_scene(md)
            frame = ren.render()
        ren_ms = (time.perf_counter() - t0) * 1000 / 20
        say("H mujoco in taichi process", True,
            "Renderer " + str(frame.shape) + " " + str(frame.dtype)
            + "; context " + f(first_s) + "s, then " + f(ren_ms)
            + " ms/frame -> no GL clash with a Vulkan window")
    except Exception as exc:                                   # noqa: BLE001
        frame = np.zeros((480, 640, 3), np.uint8)
        say("H mujoco in taichi process", False, repr(exc)[:260])

    try:
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(create=True, size=frame.nbytes * 4)
        ring = np.ndarray((4,) + frame.shape, dtype=np.uint8, buffer=shm.buf)
        ring[0] = frame
        src = np.ascontiguousarray(ring[0])
        for _ in range(5):
            k_blit(src, 20, 20, 560, 420)
        ti.sync()
        t0 = time.perf_counter()
        for _ in range(60):
            k_blit(src, 20, 20, 560, 420)
        ti.sync()
        blit_ms = (time.perf_counter() - t0) * 1000 / 60
        canvas.set_image(rgbf)
        window.show()
        window.save_image(str(out / "probe_body_blit.png"))
        say("H body blit", blit_ms <= 1.0,
            "shared_memory ring 4x" + str(frame.shape) + " = "
            + f(shm.size / 1e6) + " MB; blit 640x480 -> 560x420 in "
            + f(blit_ms, 3) + " ms, budget 1")
        shm.close()
        shm.unlink()
    except Exception as exc:                                   # noqa: BLE001
        say("H body blit", False, repr(exc)[:220])

    # ---- I. optional: event.key strings
    if args.keys > 0:
        print("\n>>> click the probe window and press keys for " + f(args.keys, 0)
              + "s: a z 1 0 - = : + , . ; arrows Shift+a Ctrl+c Space Tab Enter",
              flush=True)
        seen, deadline = [], time.time() + args.keys
        gui = window.get_gui()
        while time.time() < deadline and window.running:
            for e in window.get_events(ti.ui.PRESS):
                seen.append(repr(e.key))
            canvas.set_image(rgbf)
            with gui.sub_window("keys", 0.05, 0.05, 0.4, 0.2):
                gui.text("press keys: " + str(len(seen)) + " seen")
                gui.text(" ".join(sorted(set(seen))[-12:]))
            window.show()
        uniq = sorted(set(seen))
        say("I event.key", len(uniq) > 0,
            str(len(uniq)) + " distinct: " + " ".join(uniq)
            + " | cursor " + str(window.get_cursor_pos()))

    window.destroy()
    (out / "probe_results.json").write_text(
        json.dumps(RESULTS, indent=2, ensure_ascii=False), encoding="utf-8")
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    print("\nwrote " + str(out / "probe_results.json"))
    print("failed gates: " + (", ".join(bad) if bad else "none"))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
