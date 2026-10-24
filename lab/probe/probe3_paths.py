"""Shag 0, third pass: separate "did the kernel compute it" from "did it reach
the screen".

The saved screenshots came out at three different sizes (1905x984, 1600x900,
257x1824) and the ones taken after set_image(u8 field) or canvas.scene() were
pure black, while the ones after set_image(f32 field / numpy) had content. So
either the u8 path does not present, or save_image/readback is unreliable here.

This checks the kernels by reading OUR OWN fields back to numpy (a plain
device->host copy, nothing to do with the swapchain), and checks presentation
separately.

ASCII only.
"""

import numpy as np
import taichi as ti

W, H = 1200, 700

ti.init(arch=ti.vulkan, offline_cache=True)

acc = ti.Vector.field(3, ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, ti.u8, shape=(W, H))
rgbf = ti.Vector.field(3, ti.f32, shape=(W, H))


@ti.kernel
def k_pattern():
    for x, y in acc:
        acc[x, y] = ti.Vector([x / W, y / H, 0.5])


@ti.kernel
def k_to_u8():
    for x, y in acc:
        v = ti.min(acc[x, y], ti.Vector([1.0, 1.0, 1.0]))
        rgb8[x, y] = ti.Vector([ti.cast(v[0] * 255, ti.u8),
                                ti.cast(v[1] * 255, ti.u8),
                                ti.cast(v[2] * 255, ti.u8)])


@ti.kernel
def k_to_f32():
    for x, y in acc:
        rgbf[x, y] = ti.min(acc[x, y], ti.Vector([1.0, 1.0, 1.0]))


k_pattern()
k_to_u8()
k_to_f32()
ti.sync()

a = acc.to_numpy()
u = rgb8.to_numpy()
g = rgbf.to_numpy()
print("acc   field: shape " + str(a.shape) + " " + str(a.dtype)
      + " mean " + format(float(a.mean()), ".4f") + " max " + format(float(a.max()), ".4f"))
print("rgb8  field: shape " + str(u.shape) + " " + str(u.dtype)
      + " mean " + format(float(u.mean()), ".2f") + " max " + str(int(u.max())))
print("rgbf  field: shape " + str(g.shape) + " " + str(g.dtype)
      + " mean " + format(float(g.mean()), ".4f") + " max " + format(float(g.max()), ".4f"))
print("  -> kernels and u8 conversion are " +
      ("CORRECT" if u.max() > 200 and a.max() > 0.9 else "WRONG"))

window = ti.ui.Window("probe3", (W, H), vsync=False, fps_limit=10000)
canvas = window.get_canvas()
print("\nget_window_shape: " + str(window.get_window_shape()))

cases = [
    ("field u8", rgb8),
    ("field f32", rgbf),
    ("numpy u8", np.ascontiguousarray((g * 255).astype(np.uint8))),
    ("numpy f32", np.ascontiguousarray(g)),
]
for name, payload in cases:
    try:
        for _ in range(3):
            canvas.set_image(payload)
            window.show()
        buf = window.get_image_buffer_as_numpy()
        print(name.ljust(11) + " -> readback " + str(buf.shape)
              + " mean " + format(float(buf[..., :3].mean()), ".4f")
              + " max " + format(float(buf[..., :3].max()), ".4f")
              + ("   PRESENTED" if float(buf[..., :3].max()) > 0.5 else "   BLACK"))
    except Exception as exc:                                   # noqa: BLE001
        print(name.ljust(11) + " -> FAILED " + repr(exc)[:160])

# does canvas.scene() reach the readback at all
try:
    scene = window.get_scene()
    cam = ti.ui.Camera()
    cam.position(0.0, 0.0, 2.0)
    cam.lookat(0.0, 0.0, 0.0)
    cam.up(0.0, 1.0, 0.0)
    cam.fov(45.0)
    pts = ti.Vector.field(3, ti.f32, shape=5000)
    rng = np.random.default_rng(0)
    pts.from_numpy((rng.random((5000, 3)).astype(np.float32) - 0.5) * 0.8)
    for _ in range(3):
        scene.set_camera(cam)
        scene.ambient_light((1.0, 1.0, 1.0))
        scene.point_light(pos=(0.0, 0.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.particles(pts, radius=0.01, color=(1.0, 0.9, 0.2))
        canvas.scene(scene)
        window.show()
    buf = window.get_image_buffer_as_numpy()
    print("scene only  -> readback " + str(buf.shape)
          + " mean " + format(float(buf[..., :3].mean()), ".4f")
          + " max " + format(float(buf[..., :3].max()), ".4f")
          + ("   PRESENTED" if float(buf[..., :3].max()) > 0.5 else "   BLACK"))
except Exception as exc:                                       # noqa: BLE001
    print("scene only  -> FAILED " + repr(exc)[:200])

# how stable is the surface size across repeated reads
sizes = []
for _ in range(6):
    canvas.set_image(rgbf)
    window.show()
    sizes.append(window.get_image_buffer_as_numpy().shape[:2])
print("\nreadback sizes over 6 frames: " + str(sizes))
print("get_window_shape still: " + str(window.get_window_shape()))

window.destroy()
