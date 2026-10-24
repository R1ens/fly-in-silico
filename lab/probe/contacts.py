"""Как включить точки касания, не включив заодно всю коллизионную геометрию.

Слой отладки, сделанный «в лоб», залил вьюпорт оранжевым: свежий
`MjvOption` показывает все группы геометрии, а у мухи в невидимых
группах лежат капсулы столкновений, и они закрывают собой модель.

Здесь подбирается набор групп, при котором вид совпадает с обычным, а
точки касания появляются.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\contacts.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent


def main():
    import mujoco
    from PIL import Image

    from flybrain.worker import _thorax_id
    from probe.cams import build

    sim, _ = build(False)
    sim.reset(seed=0)
    for _ in range(300):
        sim.step(np.array([1.0, 1.0]))
    model = sim.physics.model.ptr
    data = sim.physics.data.ptr

    ren = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = _thorax_id(model)
    cam.azimuth, cam.elevation, cam.distance = 180.0, -18.0, 6.0

    def render(opt=None, name="base"):
        if opt is None:
            ren.update_scene(data, camera=cam)
        else:
            ren.update_scene(data, camera=cam, scene_option=opt)
        img = np.asarray(ren.render()).copy()
        Image.fromarray(img).save(OUT / ("contacts_" + name + ".png"))
        return img

    base = render(None, "base")
    rep = {"групп в модели": 6, "варианты": {}}

    def diff(img):
        return float(np.abs(img.astype(np.int16)
                            - base.astype(np.int16)).mean())

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    rep["варианты"]["умолчание MjvOption"] = round(
        diff(render(opt, "default_opt")), 2)

    # По одной группе: какая именно заливает кадр.
    for g in range(6):
        o = mujoco.MjvOption()
        mujoco.mjv_defaultOption(o)
        for k in range(6):
            o.geomgroup[k] = 1 if k == g else 0
        rep["варианты"]["только группа %d" % g] = round(
            diff(render(o, "group%d" % g)), 2)

    # Группы, как их показывает модель по своим настройкам.
    best = None
    for keep in ((0,), (0, 1), (0, 1, 2), (1, 2), (2,)):
        o = mujoco.MjvOption()
        mujoco.mjv_defaultOption(o)
        for k in range(6):
            o.geomgroup[k] = 1 if k in keep else 0
        d = diff(render(o, "keep" + "".join(map(str, keep))))
        rep["варианты"]["группы " + str(keep)] = round(d, 2)
        if best is None or d < best[1]:
            best = (keep, d)
    rep["ближе всего к обычному виду"] = {"группы": list(best[0]),
                                          "расхождение": round(best[1], 2)}

    # И с точками касания поверх выбранного набора.
    o = mujoco.MjvOption()
    mujoco.mjv_defaultOption(o)
    for k in range(6):
        o.geomgroup[k] = 1 if k in best[0] else 0
    o.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)] = True
    img = render(o, "with_points")
    rep["с точками касания"] = {"расхождение с обычным": round(diff(img), 2)}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    (OUT / "contacts_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    ren.close()
    sim.close()


if __name__ == "__main__":
    main()
