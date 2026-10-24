"""Проверка тела: штатная ходьба NeuroMechFly v2 и сколько она стоит.

Нужно ответить на один вопрос — успеет ли тело за мозгом. Мозг считает
0,14x реального времени, и если тело быстрее, то в замкнутом цикле оно будет
ждать мозг, а не наоборот.

Запуск:  D:\\fly-brain\\.venv\\Scripts\\python.exe lab\\bench_body.py
"""

from __future__ import annotations

import time

import numpy as np

from flygym_gymnasium import Fly, Camera
from flygym_gymnasium.preprogrammed import default_leg_sensor_placements
from flygym_gymnasium.examples.locomotion import HybridTurningController


def build(render=False, timestep=1e-4):
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=False,
        init_pose="stretch",
        control="position",
        # Контроллер следит за спотыканием по контактам голени и лапки,
        # без этих датчиков он не собирается.
        contact_sensor_placements=default_leg_sensor_placements,
    )
    cams = []
    if render:
        cams = [Camera(attachment_point=fly.model.worldbody,
                       camera_name="camera_top", targeted_fly_names=[fly.name])]
    return HybridTurningController(fly=fly, cameras=cams, timestep=timestep)


def walk(sim, seconds, action):
    """Прогнать ходьбу и вернуть время счёта и пройденный путь.

    FlyGym меряет длину в миллиметрах, так что домножать ничего не нужно.
    """
    obs, _ = sim.reset(seed=0)
    start = np.array(obs["fly"][0][:2], dtype=float)
    n = int(round(seconds / sim.timestep))
    t0 = time.perf_counter()
    for _ in range(n):
        obs, _, term, trunc, _ = sim.step(action)
        if term or trunc:
            break
    wall = time.perf_counter() - t0
    end = np.array(obs["fly"][0][:2], dtype=float)
    return wall, n, float(np.linalg.norm(end - start)), end - start


BRAIN_RATIO = 0.14  # мозг на RTX 4090, измерено


def main():
    for timestep in (1e-4, 2e-4):
        print(f"\n{'=' * 64}\nшаг физики {timestep * 1000:g} мс")
        t0 = time.perf_counter()
        sim = build(render=False, timestep=timestep)
        print(f"  сборка тела: {time.perf_counter() - t0:.1f} с")

        for label, action in (("прямо", np.array([1.0, 1.0])),
                              ("поворот направо", np.array([1.0, 0.2]))):
            wall, n, dist, delta = walk(sim, 0.5, action)
            sim_s = n * sim.timestep
            ratio = sim_s / wall
            print(f"\n  {label}: нисходящая команда {action}")
            print(f"    {n} шагов = {sim_s:g} с модельного времени за {wall:.2f} с")
            print(f"    {n / wall:.0f} шагов/с, {ratio:.3f}x реального времени")
            print(f"    прошла {dist:.2f} мм ({dist / sim_s:.1f} мм/с), "
                  f"смещение [{delta[0]:+.2f}, {delta[1]:+.2f}] мм")
            last = ratio
        sim.close()

        combined = 1.0 / (1.0 / BRAIN_RATIO + 1.0 / last)
        print(f"\n  мозг {BRAIN_RATIO:.2f}x + тело {last:.3f}x "
              f"-> вместе {combined:.3f}x")
        print(f"  то есть 1 с жизни мухи = {1 / combined:.0f} с счёта, "
              f"эпизод 5 с = {5 / combined / 60:.1f} мин")


if __name__ == "__main__":
    main()
