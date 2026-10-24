"""Какой сдвиг азимута даёт вид сзади, сбоку, сверху и спереди.

В окне виды подписаны словами, и подпись обязана совпадать с картинкой.
Камера следует за курсом мухи: азимут считается как курс плюс сдвиг.
Какой именно сдвиг соответствует «сзади», из документации MuJoCo прямо
не следует, поэтому проверяется снимком.

Ориентир один и надёжный: голова мухи. Её положение в кадре относительно
груди и говорит, откуда мы смотрим.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\views.py
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

    from probe.cams import build

    sim, _ = build(False)
    obs, _ = sim.reset(seed=0)
    for _ in range(400):
        obs, _, term, trunc, _ = sim.step(np.array([1.0, 1.0]))
        if term or trunc:
            break
    fwd = np.asarray(obs["fly_orientation"])
    theta = float(np.arctan2(fwd[1], fwd[0]))
    print("курс мухи: %.1f°" % np.degrees(theta))

    model = sim.physics.model.ptr
    data = sim.physics.data.ptr

    def body_id(suffix):
        for i in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if nm.endswith(suffix):
                return i
        return -1

    tid, hid = body_id("Thorax"), body_id("Head")
    print("грудь %d, голова %d" % (tid, hid))

    ren = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = tid
    cam.distance = 6.0

    # Куда голова смещена в кадре: вправо, влево, вверх или «на нас».
    head_world = np.asarray(data.xpos[hid])
    thorax_world = np.asarray(data.xpos[tid])
    rep = {"курс, °": round(float(np.degrees(theta)), 1), "сдвиги": {}}
    for off in (0, 90, 180, 270):
        cam.azimuth = float(np.degrees(theta)) + off
        cam.elevation = -18.0
        ren.update_scene(data, camera=cam)
        Image.fromarray(np.asarray(ren.render())).save(
            OUT / ("view_off%d.png" % off))
        # Направление взгляда камеры в плоскости
        a = np.radians(cam.azimuth)
        eye_dir = np.array([-np.cos(a), -np.sin(a)])      # от камеры к цели
        right = np.array([-eye_dir[1], eye_dir[0]])
        v = (head_world - thorax_world)[:2]
        rep["сдвиги"][off] = {
            "голова вперёд по взгляду": round(float(v @ eye_dir), 3),
            "голова вправо в кадре": round(float(v @ right), 3),
        }
        print("  сдвиг %3d°: голова по взгляду %+6.3f, вправо %+6.3f"
              % (off, v @ eye_dir, v @ right))
    # Сзади: голова дальше от камеры, то есть «вперёд по взгляду» > 0 и
    # вбок почти ноль. Спереди — наоборот. Сбоку — вбок по модулю больше.
    best_back = max(rep["сдвиги"], key=lambda k:
                    rep["сдвиги"][k]["голова вперёд по взгляду"])
    best_front = min(rep["сдвиги"], key=lambda k:
                     rep["сдвиги"][k]["голова вперёд по взгляду"])
    best_side = max(rep["сдвиги"], key=lambda k:
                    abs(rep["сдвиги"][k]["голова вправо в кадре"]))
    rep["вывод"] = {"сзади": best_back, "спереди": best_front,
                    "сбоку": best_side}
    print("\nвывод: " + json.dumps(rep["вывод"], ensure_ascii=False))
    (OUT / "views_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    ren.close()
    sim.close()


if __name__ == "__main__":
    main()
