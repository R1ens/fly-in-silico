"""Весь путь целиком: команда — модель — сцена — кадр — окно.

Остальные пробы режут цепочку на части и каждую проверяют отдельно:
`sandbox_link.py` — что правка меняет состояние модели, `scene_pick.py` —
что проекция совпадает с кадром. Остаётся стык, на котором можно
обмануться: правка применилась, кадр снят, а переехал ли маркер в самом
кадре — никто не смотрел.

Здесь смотрит. Через настоящего работника запускается сеанс в теле,
источник переносится живой правкой, и в кадре, пришедшем в окно, по
цветным пикселям ищется маркер. Его найденное место сравнивается с тем,
что предсказала :func:`flybrain.scene.project` по позе камеры ЭТОГО
кадра, а обратный перевод — с заказанным положением.

Что это ловит и чего не ловят другие пробы:

* правка применилась, но маркер остался на месте (забыли `mocap_pos`);
* маркер переехал, а яркость считается по старому месту (не пересобрали
  кэш) — здесь это видно по сенсорному входу;
* поза камеры отстаёт от кадра на шаг, и щелчок по сцене попадёт мимо;
* ширина и высота кадра перепутаны местами — промах в сотни пикселей.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\sandbox_scene.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "кадры": []}

# Слоты поближе к мухе, чтобы попасть в кадр: камера слежения считает
# охват по габариту мухи, а он около пяти миллиметров.
SLOTS = [[12.0, 8.0, 1.5], [12.0, -8.0, 1.5],
         [-12.0, 8.0, 1.5], [-12.0, -8.0, 1.5]]

# Цвета маркеров задаёт `OdorArena` из своего набора; читать их из модели
# работника нельзя — она в другом процессе. Поэтому маркер ищется не по
# заранее известному цвету, а по тому, что в кадре изменилось между
# снимками: это и надёжнее, и не зависит от набора цветов.
MOVE_TO = (6.0, 13.0)


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def _disc(shape, at, radius):
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    return ((xs - at[0]) ** 2 + (ys - at[1]) ** 2) <= radius ** 2


def changed(img_a, img_b, near, radius=90.0):
    """Сколько пикселей изменилось вокруг точки. Грубая мера, без центра."""
    d = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16)).sum(axis=2)
    return int(((d > 40) & _disc(img_a.shape, near, radius)).sum())


def marker_colour(img, at, radius=9.0):
    """Цвет маркера: самый частый насыщенный цвет в пятачке вокруг точки.

    Цвет маркера задаёт `OdorArena` из своего набора, и прочитать его из
    модели работника нельзя — она в другом процессе. Поэтому цвет
    снимается с самого кадра там, где маркер заведомо стоит.
    """
    m = _disc(img.shape, at, radius)
    px = img[m].astype(np.int16)
    if not len(px):
        return None
    # Пол серый и клетчатый: у него каналы почти равны. Маркер цветной,
    # значит у него разброс каналов большой — по нему и отбираем.
    spread = px.max(axis=1) - px.min(axis=1)
    pick = px[spread >= max(30, int(np.percentile(spread, 85)))]
    if len(pick) < 4:
        return None
    return np.median(pick, axis=0)


def find_colour(img, rgb, near, radius=90.0, tol=48.0):
    """Центр пятна заданного цвета вокруг точки. Возвращает (центр, сколько)."""
    d = np.linalg.norm(img.astype(float) - np.asarray(rgb, float)[None, None],
                       axis=2)
    m = (d < tol) & _disc(img.shape, near, radius)
    n = int(m.sum())
    if n < 5:
        return None, n
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())), n


def main():                                                   # noqa: C901
    from flybrain import scene as SC
    from flybrain.ui.session import Session

    ses = Session()
    t0 = time.perf_counter()
    while not ses.ready and time.perf_counter() - t0 < 300:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("работник поднялся", ses.ready,
          "за %.1f с" % (time.perf_counter() - t0))
    if not ses.ready:
        return 1

    applied, arrived = [], {}

    def pump(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            ses.pump(dt=0.02)
            for rec in ses.take_applied():
                applied.append(rec)
                arrived[rec.cmd_id] = time.perf_counter()
            time.sleep(0.01)

    def wait_frame(prev_seq, seconds=20.0):
        """Дождаться кадра, снятого ПОСЛЕ момента вызова."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pump(0.05)
            if ses.body_seq > prev_seq and ses.body_frame is not None:
                cam = (ses.extra or {}).get("cam")
                if cam:
                    return (ses.body_seq, np.array(ses.body_frame, copy=True),
                            dict(cam), dict(ses.extra or {}))
        return None

    # Слот 0 включён с самого начала: его и будем переносить.
    spec = {"kind": "body", "t": 0.0, "seed": 0, "open_report": False,
            "out": "probe-sandbox-scene", "bin_ms": 10.0,
            "body": {"readout": "ALPN", "control_ms": 20.0, "keep_s": 5.0,
                     "odour_ref": 0.005, "sources": SLOTS,
                     "live_initial": {"sources": [
                         {"x": 12.0, "y": 8.0, "strength": 1.0, "on": True},
                         {"x": 12.0, "y": -8.0, "strength": 1.0,
                          "on": False},
                         {"x": -12.0, "y": 8.0, "strength": 1.0,
                          "on": False},
                         {"x": -12.0, "y": -8.0, "strength": 1.0,
                          "on": False}]}}}
    ses.submit(spec)
    t0 = time.perf_counter()
    while ses.state != "running" and time.perf_counter() - t0 < 420:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("сеанс пошёл", ses.state == "running",
          "%s за %.1f с" % (ses.state, time.perf_counter() - t0))
    if ses.state != "running":
        ses.close()
        return 1

    # Камеру отодвигаем: при охвате по умолчанию источник в двенадцати
    # миллиметрах в кадр не попадает вовсе.
    ses.ask("camera", {"az": 0.0, "el": -40.0, "k": 7.0, "follow": True})
    pump(6.0)

    got = wait_frame(ses.body_seq, 25.0)
    check("кадр с позой камеры пришёл", got is not None,
          "кадров %d" % ses.body_seq)
    if got is None:
        ses.close()
        return 1
    seq0, img0, cam0, ex0 = got
    rep["замеры"]["кадр"] = {"ширина": cam0["w"], "высота": cam0["h"],
                             "номер": seq0}
    check("поза камеры несёт размер кадра, совпадающий с картинкой",
          img0.shape[1] == int(cam0["w"]) and img0.shape[0] == int(cam0["h"]),
          "картинка %s, поза говорит %sx%s"
          % (img0.shape[:2], cam0["w"], cam0["h"]))

    # Где предсказано место слота 0 до переноса.
    at0 = SC.project(cam0, (12.0, 8.0, 1.5))
    inside = at0 is not None and 0 <= at0[0] < cam0["w"] \
        and 0 <= at0[1] < cam0["h"]
    check("слот виден в кадре до переноса", inside,
          "предсказано %s при кадре %sx%s"
          % (None if at0 is None else (round(at0[0]), round(at0[1])),
             cam0["w"], cam0["h"]))

    odour0 = (float(ex0.get("odour_l", 0.0)), float(ex0.get("odour_r", 0.0)))

    # --- перенос ---
    sent_ms, sent_wall = ses.t_ms, time.perf_counter()
    cid = ses.control("set", {"sources": [
        {"x": MOVE_TO[0], "y": MOVE_TO[1], "strength": 1.0, "on": True},
        {"x": 12.0, "y": -8.0, "strength": 1.0, "on": False},
        {"x": -12.0, "y": 8.0, "strength": 1.0, "on": False},
        {"x": -12.0, "y": -8.0, "strength": 1.0, "on": False},
    ]})
    pump(8.0)
    ok_set = [a for a in applied if a.cmd_id == cid
              and a.outcome == "applied"]
    check("перенос подтверждён работником", bool(ok_set),
          "подтверждений %d" % len(ok_set))
    if ok_set:
        rep["замеры"]["задержка переноса"] = {
            "модельного, мс": ok_set[0].t_model_ms - sent_ms,
            "по стене, с": round(arrived.get(cid, time.perf_counter())
                                 - sent_wall, 3)}

    got = wait_frame(ses.body_seq, 25.0)
    check("кадр после переноса пришёл", got is not None)
    if got is None:
        ses.close()
        return 1
    seq1, img1, cam1, ex1 = got

    # --- маркер в кадре переехал туда, куда просили ---
    at1 = SC.project(cam1, (MOVE_TO[0], MOVE_TO[1], 1.5))
    rep["замеры"]["предсказано"] = {
        "до": [round(v, 1) for v in at0],
        "после": [round(v, 1) for v in at1] if at1 else None}
    check("предсказанное место слота в кадре изменилось",
          at1 is not None and ((at1[0] - at0[0]) ** 2
                               + (at1[1] - at0[1]) ** 2) ** 0.5 > 20.0,
          "с (%.0f, %.0f) на %s" % (at0[0], at0[1],
                                    None if at1 is None else
                                    (round(at1[0]), round(at1[1]))))

    # Старое место обязано опустеть, новое — заполниться. Ищем пятно
    # изменений вокруг каждого из двух мест отдельно: муха шевелится в
    # середине кадра и перебила бы поиск по всему кадру.
    # Цвет маркера снимается с ПЕРВОГО кадра там, где он заведомо стоял,
    # и дальше ищется во втором. Так проверка не зависит ни от набора
    # цветов арены, ни от того, что ещё изменилось в кадре: первая
    # версия искала любые изменения, и в радиус попадала шевелящаяся
    # муха — пятнадцать тысяч пикселей на двадцать пять тысяч площади.
    rgb = marker_colour(img0, at0)
    check("цвет маркера снят с кадра", rgb is not None,
          "RGB %s" % (None if rgb is None else [int(v) for v in rgb]))
    n_gone = changed(img0, img1, at0)
    n_came = changed(img0, img1, at1) if at1 else 0
    was_there, n_was = (find_colour(img0, rgb, at0, radius=40.0)
                        if rgb is not None else (None, 0))
    gone, n_still = (find_colour(img1, rgb, at0, radius=40.0)
                     if rgb is not None else (None, 0))
    came, n_now = (find_colour(img1, rgb, at1, radius=60.0)
                   if (rgb is not None and at1) else (None, 0))
    rep["кадры"] = [{"номер": seq0}, {"номер": seq1},
                    {"изменений у старого места": n_gone,
                     "изменений у нового места": n_came,
                     "цвет маркера": None if rgb is None
                     else [int(v) for v in rgb],
                     "пикселей цвета у старого места: было": n_was,
                     "стало": n_still,
                     "пикселей цвета у нового места": n_now}]
    check("маркер был виден на старом месте", was_there is not None,
          "пикселей его цвета %d" % n_was)
    check("с старого места маркер ушёл", n_still * 3 < max(n_was, 1),
          "было %d пикселей его цвета, осталось %d" % (n_was, n_still))
    check("на новом месте маркер появился", came is not None,
          "пикселей его цвета %d (изменений в кадре рядом %d)"
          % (n_now, n_came))
    if came is not None and at1 is not None:
        miss = ((came[0] - at1[0]) ** 2 + (came[1] - at1[1]) ** 2) ** 0.5
        rep["замеры"]["промах в кадре, px"] = round(miss, 1)
        # Порог — не вкус: маркер это капсула радиусом 0,25 мм, на таком
        # отдалении десяток пикселей, и пятно изменений шире его на
        # размытие края. Пятьдесят пикселей на кадре 960×720 — это пять
        # процентов ширины, то есть промах, который виден глазом.
        check("маркер в кадре стоит там, где предсказано", miss < 50.0,
              "предсказано (%.0f, %.0f), пятно (%.0f, %.0f), промах "
              "%.0f px" % (at1[0], at1[1], came[0], came[1], miss))
        back = SC.unproject(cam1, came[0], came[1], z=1.5)
        err = (float(np.linalg.norm(np.array(back[:2])
                                    - np.array(MOVE_TO))) if back else None)
        rep["замеры"]["обратный перевод, мм"] = (round(err, 3)
                                                 if err is not None else None)
        check("обратный перевод пятна даёт заказанные координаты",
              err is not None and err < 3.0,
              "просили (%.1f, %.1f), обратно (%.2f, %.2f), ошибка %.2f мм"
              % (MOVE_TO[0], MOVE_TO[1], back[0], back[1], err)
              if back else "обратный перевод не вышел")

    # --- и чутьё поехало вместе с маркером, а не отдельно ---
    odour1 = (float(ex1.get("odour_l", 0.0)), float(ex1.get("odour_r", 0.0)))
    rep["замеры"]["запах"] = {"до": odour0, "после": odour1}
    check("сенсорный вход изменился вместе с маркером",
          abs(odour1[0] - odour0[0]) > 1e-9
          or abs(odour1[1] - odour0[1]) > 1e-9,
          "слева %.7g -> %.7g, справа %.7g -> %.7g"
          % (odour0[0], odour1[0], odour0[1], odour1[1]))

    try:
        from PIL import Image

        for name, im in (("до", img0), ("после", img1)):
            p = OUT / ("sandbox-scene-%s.png" % name)
            Image.fromarray(im).save(p)
            rep["кадры"].append({"что": name, "файл": str(p)})
    except Exception as exc:                              # noqa: BLE001
        rep["кадры"].append({"ошибка": str(exc)})

    ses.control("stop")
    t1 = time.perf_counter()
    while ses.busy and time.perf_counter() - t1 < 180:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("сеанс остановлен", not ses.busy,
          "за %.1f с" % (time.perf_counter() - t1))
    ses.close()

    (OUT / "sandbox_scene.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
