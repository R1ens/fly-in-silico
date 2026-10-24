"""Щелчок по сцене — в координаты модели: сверка с настоящим кадром.

Утверждение, которое надо доказать: если взять источник в сцене и
перенести, его новые координаты будут координатами модели. Проверять
такое рассуждением о системах координат OpenGL бессмысленно — знак одной
оси перепутать легче, чем заметить.

Поэтому проверка сквозная и по кадру. Маркеры источников ставятся в
известные места, кадр снимается тем же рендерером и той же камерой
слежения, какими его снимает работник, а потом в кадре ищутся цветные
пиксели маркера — у каждого слота свой цвет. Центр найденного пятна
сравнивается с тем, что предсказал :func:`flybrain.scene.project`, и
обратный перевод этого же пятна — с положением слота в модели.

При неверном знаке оси «вправо» пятно оказывается зеркально отражённым
относительно середины кадра, и это видно по числам, а не на глаз.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\scene_pick.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "слоты": []}

# Слоты поближе к мухе, чем в научных эпизодах: песочнице нужно, чтобы
# источник был виден в кадре вместе с мухой, а камера слежения считает
# охват по габариту мухи — пять миллиметров.
SLOTS = np.array([[12.0, 8.0, 1.5], [12.0, -8.0, 1.5],
                  [-12.0, 8.0, 1.5], [-12.0, -8.0, 1.5]])

# Во сколько раз отдалить камеру от габарита мухи, чтобы слоты попали в
# кадр. При множителе по умолчанию (1,3) камера стоит в шести
# миллиметрах и источников в двенадцати не видно вовсе.
ZOOM_K = 6.0


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():                                                   # noqa: C901
    import mujoco

    from flybrain import body as B, scene as SC
    from flybrain.worker import RING_SHAPE, _fly_span, _thorax_id

    cfg = B.BodyConfig(duration_s=0.1, control_ms=20.0, seed=0, device="cpu")
    sim, _ = B._build_body(cfg, SLOTS.copy())
    sim.reset(seed=0)
    data = sim.physics.data.ptr
    model = sim.physics.model.ptr

    # Рендерер и камера — как у работника, иначе сверять было бы нечего.
    ren = mujoco.Renderer(model, RING_SHAPE[0], RING_SHAPE[1])  # h, w
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = _thorax_id(model)
    span = _fly_span(data, cam.trackbodyid)
    cam.azimuth, cam.elevation = 0.0, -35.0
    cam.distance = span * ZOOM_K
    ren.update_scene(data, camera=cam)
    img = ren.render()
    gc = ren.scene.camera[0]
    pose = {
        "pos": [float(v) for v in gc.pos],
        "forward": [float(v) for v in gc.forward],
        "up": [float(v) for v in gc.up],
        "near": float(gc.frustum_near), "far": float(gc.frustum_far),
        "bottom": float(gc.frustum_bottom), "top": float(gc.frustum_top),
        "centre": float(gc.frustum_center),
        "width": float(getattr(gc, "frustum_width", 0.0) or 0.0),
        "ortho": bool(gc.orthographic),
        # (высота, ширина) — так его принимает `mujoco.Renderer`.
        "w": int(RING_SHAPE[1]), "h": int(RING_SHAPE[0]),
    }
    rep["замеры"]["габарит мухи, мм"] = round(float(span), 3)
    rep["замеры"]["камера"] = pose
    W, H = int(RING_SHAPE[1]), int(RING_SHAPE[0])
    check("кадр снят", img is not None and img.shape[:2] == (H, W),
          "форма %s при ожидаемых (%d, %d)" % (img.shape, H, W))

    # Муха — та точка, за которой камера следит: она обязана быть у
    # середины кадра. Это первая, грубая проверка самой проекции.
    thorax = np.asarray(data.xpos[cam.trackbodyid], dtype=float)
    at = SC.project(pose, thorax)
    off = ((at[0] - W / 2.0) ** 2 + (at[1] - H / 2.0) ** 2) ** 0.5
    check("муха проецируется в середину кадра", off < 40.0,
          "пиксель (%.0f, %.0f) при середине (%.0f, %.0f), промах %.0f px"
          % (at[0], at[1], W / 2, H / 2, off))

    # Теперь по каждому слоту: где его видно и где его предсказали.
    found = 0
    for i in range(int(SLOTS.shape[0])):
        bid = None
        for b in range(int(model.nbody)):
            if model.body(b).name == "odor_source_marker_%d" % i:
                bid = b
                break
        if bid is None:
            check("слот %d: тело маркера найдено" % i, False, "нет такого")
            continue
        g = int(model.body_geomadr[bid])
        rgb = np.asarray(model.geom_rgba[g][:3], dtype=float) * 255.0
        # Пиксели маркера: близкие к его цвету и достаточно насыщенные,
        # чтобы не поймать серый пол и не спутать слоты между собой.
        dist = np.linalg.norm(img.astype(float) - rgb[None, None, :], axis=2)
        mask = dist < 40.0
        n = int(mask.sum())
        want = SC.project(pose, SLOTS[i])
        row = {"слот": i, "в модели": SLOTS[i].tolist(),
               "цвет": rgb.round(1).tolist(), "пикселей": n,
               "предсказано": [round(v, 1) for v in want] if want else None}
        if n < 8:
            row["вывод"] = "в кадр не попал"
            rep["слоты"].append(row)
            continue
        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        row["найдено"] = [round(cx, 1), round(cy, 1)]
        miss = ((cx - want[0]) ** 2 + (cy - want[1]) ** 2) ** 0.5
        row["промах, px"] = round(miss, 1)
        back = SC.unproject(pose, cx, cy, z=float(SLOTS[i][2]))
        row["обратно"] = [round(v, 2) for v in back] if back else None
        err_mm = (float(np.linalg.norm(np.array(back[:2]) - SLOTS[i][:2]))
                  if back else None)
        row["ошибка обратного перевода, мм"] = (round(err_mm, 3)
                                                if err_mm is not None else None)
        rep["слоты"].append(row)
        found += 1
        # Маркер — капсула радиусом 0,25 мм: на этом отдалении это
        # десяток пикселей, и центр пятна не обязан совпасть с центром
        # тела точнее, чем на его размер.
        check("слот %d: предсказание совпало с кадром" % i, miss < 25.0,
              "предсказано (%.0f, %.0f), найдено (%.0f, %.0f), промах "
              "%.1f px по %d пикселям" % (want[0], want[1], cx, cy, miss, n))
        check("слот %d: обратный перевод даёт координаты модели" % i,
              err_mm is not None and err_mm < 1.5,
              "в модели (%.1f, %.1f), обратно (%.2f, %.2f), ошибка %.3f мм"
              % (SLOTS[i][0], SLOTS[i][1], back[0], back[1], err_mm)
              if back else "обратный перевод не вышел")

    check("в кадр попало больше одного слота", found >= 2,
          "нашлось %d из %d" % (found, int(SLOTS.shape[0])))

    # Самосогласованность: туда и обратно по сетке пикселей. Ловит
    # ошибку, которая на маркерах могла бы скрыться за их размером.
    worst = 0.0
    for px in (120, 480, 840):
        for py in (150, 360, 560):
            p = SC.unproject(pose, px, py, z=1.5)
            if p is None:
                continue
            b = SC.project(pose, p)
            worst = max(worst, ((b[0] - px) ** 2 + (b[1] - py) ** 2) ** 0.5)
    rep["замеры"]["худшая невязка туда-обратно, px"] = round(worst, 4)
    check("перевод туда и обратно замкнулся", worst < 0.01,
          "худшая невязка %.5f px" % worst)

    # Зеркальность: если знак оси «вправо» перепутан, точка справа от
    # мухи окажется слева в кадре. Проверяется явно.
    right_of_fly = thorax + np.array([0.0, -6.0, 0.0])   # правее по модели
    left_of_fly = thorax + np.array([0.0, 6.0, 0.0])
    pr, pl = SC.project(pose, right_of_fly), SC.project(pose, left_of_fly)
    rep["замеры"]["проверка зеркальности"] = {
        "правее по модели": [round(v, 1) for v in pr],
        "левее по модели": [round(v, 1) for v in pl],
        "азимут камеры": float(cam.azimuth)}
    check("две точки по разные стороны не слиплись в кадре",
          abs(pr[0] - pl[0]) > 20.0,
          "по горизонтали %.0f и %.0f" % (pr[0], pl[0]))

    out = OUT / "scene_pick.png"
    try:
        from PIL import Image

        shot = Image.fromarray(img)
        shot.save(out)
        rep["замеры"]["кадр"] = str(out)
    except Exception as exc:                              # noqa: BLE001
        rep["замеры"]["кадр"] = "не сохранён: %s" % exc

    ren.close()
    sim.close()
    (OUT / "scene_pick.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
