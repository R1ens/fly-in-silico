"""Настоящие кадры тела с разных камер и настоящее зрение мухи.

Нужны для референсов: рисовать выдуманный «глаз мухи» нечестно, а
NeuroMechFly отдаёт и то, и другое. Здесь строится тело со зрением,
делается несколько шагов и снимаются кадры с нескольких камер плюс обе
сетчатки.

Сетчатка: 721 омматидий на глаз, два типа фоторецепторов (бледный и
жёлтый), человекочитаемый вид — шестиугольная мозаика 512×450.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
CAMS = ("camera_back_track_game", "camera_right", "camera_top",
        "camera_front_zoomin")


def main():
    from flygym_gymnasium import Fly, YawOnlyCamera
    from flygym_gymnasium.preprogrammed import (
        default_leg_sensor_placements)
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    t0 = time.perf_counter()
    fly = Fly(enable_vision=True, render_raw_vision=True,
              contact_sensor_placements=default_leg_sensor_placements)
    print("муха собрана за " + format(time.perf_counter() - t0, ".1f") + " с")

    # Камеры из config.yaml попадают в модель только через объект камеры:
    # он их туда и вписывает при создании. Снимать потом можно напрямую.
    cams = [YawOnlyCamera(attachment_point=fly.model.worldbody,
                          camera_name=name, targeted_fly_names=[fly.name],
                          play_speed=0.2, fps=30)
            for name in CAMS]
    sim = HybridTurningController(fly=fly, cameras=cams, timestep=2e-4)
    obs, _ = sim.reset(seed=0)

    # Немного пройтись, иначе глаз смотрит в пустую арену.
    for i in range(1200):
        obs, _, term, trunc, _ = sim.step(np.array([1.0, 1.0]))
        if term or trunc:
            break
    print("шагов сделано, зрение есть:", "vision" in obs)

    # Рендерим напрямую через MuJoCo по имени камеры: обёртка FlyGym ждёт
    # своё наблюдение и отмеряет кадры по модельному времени, а нам нужен
    # кадр здесь и сейчас с любой из камер.
    import mujoco
    from PIL import Image

    model = sim.physics.model.ptr
    data = sim.physics.data.ptr
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
             for i in range(model.ncam)]
    print("камер в модели: " + str(len(names)))
    ren = mujoco.Renderer(model, 540, 720)
    for want in CAMS:
        name = next((n for n in names if n and n.endswith(want)), None)
        if name is None:
            print("  нет камеры " + want)
            continue
        ren.update_scene(data, camera=name)
        img = ren.render()
        Image.fromarray(np.asarray(img)).save(OUT / ("cam_" + want + ".png"))
        print("  кадр " + want + " " + str(np.asarray(img).shape))
    ren.close()

    vis = obs.get("vision")
    if vis is None:
        print("зрения в наблюдении нет")
        return
    vis = np.asarray(vis)
    print("зрение:", vis.shape, vis.dtype, "диапазон",
          round(float(vis.min()), 3), "…", round(float(vis.max()), 3))
    retina = fly.retina
    for eye, label in ((0, "left"), (1, "right")):
        for ch, chname in ((0, "pale"), (1, "yellow")):
            hr = retina.hex_pxls_to_human_readable(vis[eye, :, ch],
                                                   color_8bit=True)
            Image.fromarray(hr.astype(np.uint8)).save(
                OUT / f"eye_{label}_{chname}.png")
        # обе колбочки вместе: бледная в синий канал, жёлтая в красный с зелёным
        p = retina.hex_pxls_to_human_readable(vis[eye, :, 0], color_8bit=True)
        y = retina.hex_pxls_to_human_readable(vis[eye, :, 1], color_8bit=True)
        rgb = np.stack([y, (y.astype(np.float32) * 0.55
                            + p.astype(np.float32) * 0.45).astype(np.uint8), p],
                       axis=-1)
        Image.fromarray(rgb).save(OUT / f"eye_{label}.png")
        print("  глаз " + label + " " + str(rgb.shape))

    raw = obs.get("raw_vision")
    if raw is not None:
        raw = np.asarray(raw)
        Image.fromarray(raw[0].astype(np.uint8)).save(OUT / "eye_raw_left.png")
        print("  сырой кадр глаза " + str(raw[0].shape))

    sim.close()


if __name__ == "__main__":
    main()
