"""Shag 0, second pass: two things the first probe left unclear.

1. The real framebuffer size. We asked for (1600, 900); get_window_shape() says
   (1600, 900); but get_image_buffer_as_numpy() came back (1905, 984, 4) and
   save_image() wrote a 1824x257 png. Display scaling silently resizing the
   surface is exactly what destroyed the previous GUI attempt, so pin it down:
   what size must our field be, and which index is width.

2. A clean frame time. The first run reported 3.63 / 14.71 / 3.50 ms for the
   same 1.6 ms of kernels at three camera distances, which cannot be the
   rasteriser. Measure again without save_image in between, and separate
   kernels / set_image / show.

Also: how fast does a command-buffer kernel get if it is parallelised over
pixels instead of over commands (the first run gave 3.9 ms for 400 rects,
which is slow for 600k pixels).

ASCII only.
"""

import time
from pathlib import Path

import numpy as np
import taichi as ti

OUT = Path(__file__).parent


def f(x, nd=2):
    return format(x, "." + str(nd) + "f")


ti.init(arch=ti.vulkan, offline_cache=True)
ASK_W, ASK_H = 1600, 900
window = ti.ui.Window("probe2", (ASK_W, ASK_H), vsync=False, fps_limit=10000)
canvas = window.get_canvas()

print("asked for      : " + str((ASK_W, ASK_H)))
print("get_window_shape: " + str(window.get_window_shape()))

# --- 1. what size is the surface really
img = np.zeros((ASK_W, ASK_H, 3), dtype=np.float32)
img[0:60, 0:60, :] = 1.0          # white block at index [0, 0]
img[300:306, :, 0] = 1.0          # red at first index 300
img[:, 150:156, 1] = 1.0          # green at second index 150
canvas.set_image(np.ascontiguousarray(img))
window.show()

buf = window.get_image_buffer_as_numpy()
print("readback shape  : " + str(buf.shape) + " " + str(buf.dtype))
RW, RH = buf.shape[0], buf.shape[1]

white = buf[..., 0] * buf[..., 1] * buf[..., 2]
wi = np.unravel_index(int(np.argmax(white)), white.shape)
red_cols = np.where((buf[..., 0].mean(axis=1) > 0.4) & (buf[..., 1].mean(axis=1) < 0.4))[0]
grn_rows = np.where((buf[..., 1].mean(axis=0) > 0.4) & (buf[..., 0].mean(axis=0) < 0.4))[0]
print("white block at readback index : " + str(wi))
print("first index 300 -> readback[0] : " + str(red_cols[:8]) + " of " + str(RW))
print("second index 150 -> readback[1]: " + str(grn_rows[:8]) + " of " + str(RH))
print("  -> scale x " + f(RW / ASK_W, 3) + ", scale y " + f(RH / ASK_H, 3))

png = OUT / "probe2_axes.png"
window.save_image(str(png))
try:
    from PIL import Image

    im = np.asarray(Image.open(png).convert("RGB"))
    print("save_image png shape (h, w, c): " + str(im.shape))
    wh = (im[..., 0].astype(int) * im[..., 1] * im[..., 2])
    print("  white block at png (row, col): "
          + str(np.unravel_index(int(np.argmax(wh)), wh.shape)))
except Exception as exc:                                       # noqa: BLE001
    print("png read failed: " + repr(exc)[:160])

# --- 2. clean frame time at the real surface size
W, H = RW, RH
acc = ti.Vector.field(3, ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, ti.u8, shape=(W, H))

import pandas as pd

df = pd.read_parquet(r"D:\fly-brain\cache\neurons.parquet",
                     columns=["x_um", "y_um", "z_um"])
p = df.to_numpy(dtype=np.float32)
p = p[np.isfinite(p).all(axis=1)]
p -= 0.5 * (p.min(0) + p.max(0))
p /= float(np.abs(p).max())
p[:, 1] *= -1.0
p[:, 2] *= -1.0
pos = np.ascontiguousarray(p)
n = pos.shape[0]

z = np.load(r"D:\fly-brain\out\demo-sugar\bins.npz")
peak = z["bins"].max(axis=0).astype(np.float32)
full = np.zeros(138639, dtype=np.float32)
idx = z["neuron_idx"].astype(np.int64)
full[idx] = peak
mask = np.isfinite(df.to_numpy(dtype=np.float32)).all(axis=1)
act = full[mask]
hi = float(np.percentile(act[act > 0], 99))
act = np.ascontiguousarray(np.clip(act / hi, 0.0, 1.0))

stops = np.array([
    [0.001, 0.000, 0.014], [0.099, 0.068, 0.233], [0.232, 0.059, 0.437],
    [0.390, 0.100, 0.502], [0.550, 0.161, 0.506], [0.716, 0.215, 0.475],
    [0.868, 0.288, 0.409], [0.967, 0.440, 0.360], [0.994, 0.624, 0.427],
    [0.996, 0.793, 0.557], [0.987, 0.991, 0.750]], dtype=np.float32)
t = np.clip(act, 0, 1) * (len(stops) - 1)
i0 = np.floor(t).astype(np.int32)
i1 = np.minimum(i0 + 1, len(stops) - 1)
wgt = (t - i0)[:, None]
col = np.ascontiguousarray((stops[i0] * (1 - wgt) + stops[i1] * wgt).astype(np.float32))


@ti.kernel
def k_clear(r: ti.f32, g: ti.f32, b: ti.f32):
    for x, y in acc:
        acc[x, y] = ti.Vector([r, g, b])


@ti.kernel
def k_tone():
    for x, y in acc:
        v = ti.min(acc[x, y], ti.Vector([1.0, 1.0, 1.0]))
        rgb8[x, y] = ti.Vector([ti.cast(v[0] * 255, ti.u8),
                                ti.cast(v[1] * 255, ti.u8),
                                ti.cast(v[2] * 255, ti.u8)])


@ti.kernel
def k_splat(pp: ti.types.ndarray(dtype=ti.f32, ndim=2),
            aa: ti.types.ndarray(dtype=ti.f32, ndim=1),
            cc: ti.types.ndarray(dtype=ti.f32, ndim=2),
            m: ti.types.ndarray(dtype=ti.f32, ndim=2),
            w: ti.i32, h: ti.i32):
    for i in range(pp.shape[0]):
        x, y, zz = pp[i, 0], pp[i, 1], pp[i, 2]
        cw = m[3, 0] * x + m[3, 1] * y + m[3, 2] * zz + m[3, 3]
        if cw > 1e-6:
            inv = 1.0 / cw
            sx = ((m[0, 0] * x + m[0, 1] * y + m[0, 2] * zz + m[0, 3]) * inv * .5 + .5) * w
            sy = ((m[1, 0] * x + m[1, 1] * y + m[1, 2] * zz + m[1, 3]) * inv * .5 + .5) * h
            av = aa[i]
            rad = 1.3 + 2.6 * av
            r2 = rad * rad
            ir = ti.cast(ti.ceil(rad), ti.i32)
            x0, y0 = ti.cast(sx, ti.i32), ti.cast(sy, ti.i32)
            amp = 0.16 + 0.84 * av
            for dx in range(-ir, ir + 1):
                for dy in range(-ir, ir + 1):
                    px, py = x0 + dx, y0 + dy
                    if 0 <= px < w and 0 <= py < h:
                        d2 = ti.cast(dx * dx + dy * dy, ti.f32)
                        if d2 <= r2:
                            wt = amp * ti.exp(-2.5 * d2 / r2)
                            ti.atomic_add(acc[px, py][0], cc[i, 0] * wt)
                            ti.atomic_add(acc[px, py][1], cc[i, 1] * wt)
                            ti.atomic_add(acc[px, py][2], cc[i, 2] * wt)


def look_at(eye, target, up):
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    s = np.cross(fwd, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -fwd
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, fwd @ eye
    return m


def mvp(eye_z):
    import math

    tt = 1.0 / math.tan(math.radians(38.0) * 0.5)
    pr = np.zeros((4, 4), np.float32)
    pr[0, 0] = tt / (W / H)
    pr[1, 1] = tt
    pr[2, 2] = (40.05) / (0.05 - 40.0)
    pr[2, 3] = 2 * 40.0 * 0.05 / (0.05 - 40.0)
    pr[3, 2] = -1.0
    return np.ascontiguousarray(pr @ look_at(
        np.array([0.0, 0.0, eye_z], np.float32), np.zeros(3, np.float32),
        np.array([0.0, 1.0, 0.0], np.float32)))


print("\nsurface " + str(W) + "x" + str(H) + ", " + str(n) + " points, "
      + str(int((act > 0).sum())) + " active")
print("dist      kernels   set_image   show    total    fps")
for label, ez in (("far", 7.0), ("fit", 3.0), ("close", 1.15)):
    m = mvp(ez)
    for _ in range(15):
        k_clear(0.055, 0.063, 0.071)
        k_splat(pos, act, col, m, W, H)
        k_tone()
        canvas.set_image(rgb8)
        window.show()
    ti.sync()
    N = 150
    t0 = time.perf_counter()
    for _ in range(N):
        k_clear(0.055, 0.063, 0.071)
        k_splat(pos, act, col, m, W, H)
        k_tone()
    ti.sync()
    t_kern = (time.perf_counter() - t0) * 1000 / N
    t0 = time.perf_counter()
    for _ in range(N):
        canvas.set_image(rgb8)
    t_set = (time.perf_counter() - t0) * 1000 / N
    t0 = time.perf_counter()
    for _ in range(N):
        canvas.set_image(rgb8)
        window.show()
    t_show = (time.perf_counter() - t0) * 1000 / N - t_set
    t0 = time.perf_counter()
    for _ in range(N):
        k_clear(0.055, 0.063, 0.071)
        k_splat(pos, act, col, m, W, H)
        k_tone()
        canvas.set_image(rgb8)
        window.show()
    t_all = (time.perf_counter() - t0) * 1000 / N
    print(label.ljust(9) + f(t_kern).rjust(7) + f(t_set).rjust(11)
          + f(t_show).rjust(8) + f(t_all).rjust(9) + f(1000 / t_all, 0).rjust(7))

window.save_image(str(OUT / "probe2_brain.png"))

# --- 3. command buffer: parallelise over pixels, not over commands
MAXR = 2048
TILE = 64 * 48
rects = ti.field(ti.f32, shape=(MAXR, 7))


@ti.kernel
def k_rects_bycmd(cnt: ti.i32):
    for k in range(cnt):
        x0 = ti.cast(rects[k, 0], ti.i32)
        y0 = ti.cast(rects[k, 1], ti.i32)
        x1 = ti.cast(rects[k, 2], ti.i32)
        y1 = ti.cast(rects[k, 3], ti.i32)
        for x in range(x0, x1):
            for y in range(y0, y1):
                acc[x, y] = ti.Vector([rects[k, 4], rects[k, 5], rects[k, 6]])


@ti.kernel
def k_rects_bypix(cnt: ti.i32, tw: ti.i32, th: ti.i32):
    for k, j in ti.ndrange(cnt, tw * th):
        x0 = ti.cast(rects[k, 0], ti.i32)
        y0 = ti.cast(rects[k, 1], ti.i32)
        x1 = ti.cast(rects[k, 2], ti.i32)
        y1 = ti.cast(rects[k, 3], ti.i32)
        dx, dy = j % tw, j // tw
        x, y = x0 + dx, y0 + dy
        if x < x1 and y < y1 and 0 <= x < W and 0 <= y < H:
            acc[x, y] = ti.Vector([rects[k, 4], rects[k, 5], rects[k, 6]])


rng = np.random.default_rng(1)
data = np.zeros((MAXR, 7), np.float32)
cnt = 400
data[:cnt, 0] = rng.integers(0, W - 70, cnt)
data[:cnt, 1] = rng.integers(0, H - 50, cnt)
data[:cnt, 2] = data[:cnt, 0] + 50
data[:cnt, 3] = data[:cnt, 1] + 30
data[:cnt, 4:] = rng.random((cnt, 3))
rects.from_numpy(data)

for _ in range(5):
    k_rects_bycmd(cnt)
    k_rects_bypix(cnt, 50, 30)
ti.sync()
t0 = time.perf_counter()
for _ in range(30):
    k_rects_bycmd(cnt)
ti.sync()
bycmd = (time.perf_counter() - t0) * 1000 / 30
t0 = time.perf_counter()
for _ in range(30):
    k_rects_bypix(cnt, 50, 30)
ti.sync()
bypix = (time.perf_counter() - t0) * 1000 / 30
print("\n400 rects, one kernel: parallel over commands " + f(bycmd, 3)
      + " ms, parallel over pixels " + f(bypix, 3) + " ms")

window.destroy()
print("\nwrote " + str(OUT / "probe2_brain.png"))
