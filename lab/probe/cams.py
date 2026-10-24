"""Какие камеры есть у тела и как в них попадает муха.

Нужно решить, можно ли дать рабочему пространству «Тело» смену вида
(сзади, сбоку, сверху, спереди), ничего не ломая. Открытый вопрос один:
камеры из `config.yaml` FlyGym попадают в модель только тогда, когда
создан объект камеры, а `body._build_body` создаёт его лишь при записи
видео. Значит в живом прогоне их может не быть вовсе.

Здесь тело собирается ровно так, как его собирает `fly body` сегодня, и
печатается список камер модели. Затем — то же самое, но с созданными (и
**не** переданными контроллеру) камерами вида: так видно и прибавку к
списку, и то, что физика осталась той же.

Для каждой камеры снимается кадр и считается, какую долю площади
занимает муха: бриф прямо запрещает вид, где тело занимает десятую часть
вьюпорта.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\cams.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent
WANT = ("camera_back_track_game", "camera_right", "camera_top",
        "camera_front_zoomin", "camera_left", "camera_back", "camera_front",
        "camera_bottom", "camera_neck_zoomin", "camera_head_zoomin")
SHOT = (480, 640)


def cam_names(model):
    import mujoco
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or ""
            for i in range(model.ncam)]


def fly_coverage(img):
    """Доля кадра, занятая не-фоном.

    Фон арены ровный, поэтому «муха» — это пиксели, заметно отличающиеся
    от самого частого цвета кадра.
    """
    a = np.asarray(img, dtype=np.int16)
    flat = a.reshape(-1, a.shape[-1])
    # самый частый цвет: берём по решётке, чтобы не считать 300 тыс. строк
    sample = flat[::37]
    colours, counts = np.unique(sample, axis=0, return_counts=True)
    bg = colours[counts.argmax()]
    d = np.abs(a - bg[None, None, :]).sum(axis=-1)
    mask = d > 40
    frac = float(mask.mean())
    if not mask.any():
        return frac, None
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return frac, box


def build(with_views):
    from flygym_gymnasium import Fly, YawOnlyCamera
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.preprogrammed import default_leg_sensor_placements
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    from flybrain.body import DEFAULT_SOURCES

    arena = OdorArena(odor_source=DEFAULT_SOURCES,
                      peak_odor_intensity=np.ones((len(DEFAULT_SOURCES), 1)))
    fly = Fly(enable_adhesion=True, enable_olfaction=True,
              contact_sensor_placements=default_leg_sensor_placements,
              init_pose="stretch", control="position",
              spawn_pos=(0.0, 0.0, 0.2))
    made = []
    if with_views:
        # Объект камеры вписывает её в MJCF. Контроллеру камеры НЕ отдаём:
        # иначе FlyGym копит кадры в `camera._frames` без предела, а нам
        # нужен только сам факт существования камеры в модели.
        for name in ("camera_back_track_game", "camera_right",
                     "camera_top", "camera_front_zoomin"):
            try:
                made.append(YawOnlyCamera(
                    attachment_point=fly.model.worldbody, camera_name=name,
                    targeted_fly_names=[fly.name], play_speed=0.2, fps=30))
            except Exception as exc:                        # noqa: BLE001
                print("  камера " + name + " не создалась: " + repr(exc))
    sim = HybridTurningController(fly=fly, cameras=[], arena=arena,
                                  timestep=1e-4)
    return sim, made


def main():
    import mujoco
    from PIL import Image

    report = {}
    for with_views in (False, True):
        tag = "с камерами вида" if with_views else "как сейчас в fly body"
        print("\n=== " + tag + " ===")
        t0 = time.perf_counter()
        sim, _made = build(with_views)
        obs, _ = sim.reset(seed=0)
        print("  собрано за %.1f с" % (time.perf_counter() - t0))
        model = sim.physics.model.ptr
        data = sim.physics.data.ptr
        names = [n for n in cam_names(model) if n]
        print("  камер в модели: %d" % len(names))
        for n in names:
            print("    " + n)
        # немного пройтись, иначе муха стоит в стартовой позе
        for _ in range(400):
            obs, _, term, trunc, _ = sim.step(np.array([1.0, 1.0]))
            if term or trunc:
                break
        info = {"камеры": names, "кадры": {}}
        if with_views:
            ren = mujoco.Renderer(model, SHOT[0], SHOT[1])
            for want in WANT:
                real = next((n for n in names if n.endswith(want)), None)
                if real is None:
                    continue
                ren.update_scene(data, camera=real)
                img = np.asarray(ren.render())
                frac, box = fly_coverage(img)
                info["кадры"][want] = {"доля кадра": round(frac, 4),
                                       "рамка": box}
                Image.fromarray(img).save(OUT / ("body_" + want + ".png"))
                print("    %-26s муха занимает %5.1f %% кадра  рамка %s"
                      % (want, frac * 100, box))
            # свободная камера: ею и будет вращение, если именованных мало
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = next(
                (i for i in range(model.nbody)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                     or "").endswith("Thorax")), 0)
            for label, (az, el, dist) in {
                    "свободная_сзади": (90.0, -15.0, 8.0),
                    "свободная_сбоку": (0.0, -10.0, 8.0),
                    "свободная_сверху": (90.0, -89.0, 9.0),
                    "свободная_спереди": (270.0, -15.0, 8.0)}.items():
                cam.azimuth, cam.elevation, cam.distance = az, el, dist
                cam.lookat[:] = data.xpos[cam.trackbodyid]
                ren.update_scene(data, camera=cam)
                img = np.asarray(ren.render())
                frac, box = fly_coverage(img)
                info["кадры"][label] = {"доля кадра": round(frac, 4),
                                        "рамка": box}
                Image.fromarray(img).save(OUT / ("body_" + label + ".png"))
                print("    %-26s муха занимает %5.1f %% кадра  рамка %s"
                      % (label, frac * 100, box))
            ren.close()
        report[tag] = info
        sim.close()

    a = set(report["как сейчас в fly body"]["камеры"])
    b = set(report["с камерами вида"]["камеры"])
    print("\nприбавка от объектов камер: %d имён" % len(b - a))
    for n in sorted(b - a):
        print("  + " + n)
    (OUT / "cams_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("отчёт: probe/cams_report.json")


if __name__ == "__main__":
    main()
