"""Подбор вида сцены MuJoCo: ракурс, охват, фон, тени, отражения.

Центральная сцена выглядела сырее остального интерфейса: много пустого
серого пола, белая полоса неба сверху, муха мелкая. Здесь перебираются
настройки рендера и камеры, каждая снимается и меряется числом, чтобы
выбор не держался на впечатлении.

Меры:
  доля мухи   сколько процентов кадра занимает не-пол и не-небо;
  небо        доля почти белых пикселей сверху;
  контраст    разброс яркости по кадру (чем ровнее серо, тем «пустее»).

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\look_body.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent


def measure(img):
    a = np.asarray(img, dtype=np.float32)
    lum = a.mean(axis=2)
    # Муха тёплая: у неё красный заметно выше синего. Пол и небо серые.
    warm = (a[:, :, 0] - a[:, :, 2]) > 12.0
    sky = lum > 235.0
    return {
        "доля мухи, %": round(float(warm.mean()) * 100, 2),
        "небо, %": round(float(sky.mean()) * 100, 1),
        "контраст": round(float(lum.std()), 1),
        "медиана яркости": round(float(np.median(lum)), 1),
    }


def main():
    import mujoco
    from PIL import Image

    from probe.cams import build
    from flybrain.worker import _thorax_id, _fly_span

    sim, _ = build(False)
    obs, _ = sim.reset(seed=0)
    for _ in range(400):
        obs, _, term, trunc, _ = sim.step(np.array([1.0, 1.0]))
        if term or trunc:
            break
    model = sim.physics.model.ptr
    data = sim.physics.data.ptr
    tid = _thorax_id(model)
    span = _fly_span(data, tid)
    theta = float(np.arctan2(*np.asarray(obs["fly_orientation"])[[1, 0]]))
    print("габарит мухи %.2f мм, курс %.0f°"
          % (span, np.degrees(theta)))

    rep = {"габарит мухи, мм": round(span, 3), "варианты": {}}
    flags = mujoco.mjtRndFlag

    def shoot(name, w, h, dist_k, elev, rnd=(), vis=()):
        ren = mujoco.Renderer(model, h, w)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = tid
        cam.azimuth = float(np.degrees(theta)) + 180.0
        cam.elevation = elev
        cam.distance = span * dist_k
        opt = None
        if vis:
            opt = mujoco.MjvOption()
            mujoco.mjv_defaultOption(opt)
            for k, v in vis:
                opt.flags[int(k)] = v
        if opt is None:
            ren.update_scene(data, camera=cam)
        else:
            ren.update_scene(data, camera=cam, scene_option=opt)
        for k, v in rnd:
            ren.scene.flags[int(k)] = v
        img = np.asarray(ren.render()).copy()
        ren.close()
        Image.fromarray(img).save(OUT / ("look_" + name + ".png"))
        m = measure(img)
        m["размер"] = [w, h]
        m["охват"] = round(dist_k, 2)
        m["возвышение"] = elev
        rep["варианты"][name] = m
        print("  %-22s %s" % (name, json.dumps(m, ensure_ascii=False)))
        return img

    # как сейчас
    shoot("now", 640, 480, 2.2, -18.0)
    # ближе и ниже горизонта
    for k in (1.6, 1.3, 1.1):
        shoot("dist%.1f" % k, 640, 480, k, -18.0)
    for el in (-25.0, -32.0, -40.0):
        shoot("el%d" % int(-el), 640, 480, 1.3, el)
    # без неба, без отражений, без теней — по одному
    shoot("noskybox", 640, 480, 1.3, -25.0,
          rnd=((flags.mjRND_SKYBOX, False),))
    shoot("norefl", 640, 480, 1.3, -25.0,
          rnd=((flags.mjRND_REFLECTION, False),))
    shoot("noshadow", 640, 480, 1.3, -25.0,
          rnd=((flags.mjRND_SHADOW, False),))
    shoot("haze", 640, 480, 1.3, -25.0,
          rnd=((flags.mjRND_HAZE, True),))
    shoot("noskybox_norefl", 640, 480, 1.3, -25.0,
          rnd=((flags.mjRND_SKYBOX, False),
               (flags.mjRND_REFLECTION, False)))
    # разрешение
    shoot("hi960", 960, 720, 1.3, -25.0,
          rnd=((flags.mjRND_SKYBOX, False),))
    # Больше 1280x720 оффскрин-буфер модели не даёт: MuJoCo отказывает
    # прямо, и поднимать его пришлось бы правкой MJCF.
    shoot("hi1280", 1280, 720, 1.3, -25.0,
          rnd=((flags.mjRND_SKYBOX, False),))

    # Небо при рабочем возвышении: чем его смягчить, не теряя горизонта.
    for name, rnd in (
            ("sky_base", ()),
            ("sky_haze", ((flags.mjRND_HAZE, True),)),
            ("sky_off", ((flags.mjRND_SKYBOX, False),)),
            ("sky_off_haze", ((flags.mjRND_SKYBOX, False),
                              (flags.mjRND_HAZE, True))),
            ("sky_norefl", ((flags.mjRND_REFLECTION, False),)),
            ("sky_noshadow", ((flags.mjRND_SHADOW, False),))):
        shoot(name, 960, 720, 1.3, -18.0, rnd=rnd)

    # Пол: он занимает большую часть кадра и в тёмном окне светится.
    # Красить его надо через материал — у плоскости он есть, и
    # `geom_rgba` при этом не действует.
    import mujoco as _mj

    planes = [i for i in range(model.ngeom)
              if model.geom_type[i] == _mj.mjtGeom.mjGEOM_PLANE]
    mats = sorted({int(model.geom_matid[i]) for i in planes
                   if model.geom_matid[i] >= 0})
    rep["плоскостей"] = len(planes)
    rep["материалов пола"] = mats
    saved = {m: model.mat_rgba[m].copy() for m in mats}
    saved_g = {i: model.geom_rgba[i].copy() for i in planes}
    for k in (0.65, 0.5, 0.38):
        for m in mats:
            model.mat_rgba[m][:3] = saved[m][:3] * k
        for i in planes:
            if model.geom_matid[i] < 0:
                model.geom_rgba[i][:3] = saved_g[i][:3] * k
        shoot("floor%d" % int(k * 100), 960, 720, 1.3, -18.0,
              rnd=((flags.mjRND_SKYBOX, False),))
        shoot("floor%d_norefl" % int(k * 100), 960, 720, 1.3, -18.0,
              rnd=((flags.mjRND_SKYBOX, False),
                   (flags.mjRND_REFLECTION, False)))
    for m in mats:
        model.mat_rgba[m][:] = saved[m]
    for i in planes:
        model.geom_rgba[i][:] = saved_g[i]

    (OUT / "look_body.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("отчёт: probe/look_body.json")
    sim.close()


if __name__ == "__main__":
    main()
