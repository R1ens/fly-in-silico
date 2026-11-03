"""Чего стоит транспорт кадра глаз: v3 против замороженного v2.

Один прогон на запуск, чтобы сравнение шло ЧЕРЕДУЮЩИМИСЯ процессами:
две сборки кода в одном процессе не уживаются, а мерить их в разное
время нельзя — машина дышит. Поэтому драйвер (`eye_cost_pairs.py`)
запускает этот файл попеременно на двух деревьях и считает отношение
ВНУТРИ пары.

    --lab <путь>   какое дерево брать (его `lab`, там лежит `flybrain`)
    --steps N      шагов управления
    --seed N       зерно

Что считается равной нагрузкой. Зрение включено в обоих случаях и с
одной частотой, то есть камеры рендерятся одинаково и сетчатка
обновляется одинаково. Разница ровно одна: в v3 на каждом обновлении
срабатывает крючок `on_vision`, кадр приводится к uint8 и ложится в
кольцо разделяемой памяти; в v2 крючка нет вовсе. Это и есть
измеряемая величина.

Отметка: в v2 `run_episode` не принимает `on_vision`, и подсунуть его
туда нельзя — поэтому дерево определяет себя само, по наличию
параметра, и печатает, что именно оно мерило.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_eye_cost_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=None)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.lab:
        # Дерево выбирается ДО первого импорта `flybrain`: иначе
        # победит то, что уже лежит на пути.
        sys.path.insert(0, str(Path(args.lab).resolve()))
    import inspect

    from flybrain import body as B, data as D

    root = Path(inspect.getfile(B)).resolve()
    has_hook = "on_vision" in inspect.signature(B.run_episode).parameters

    neurons = D.load_neurons()
    weights = D.load_weights(device="cuda")
    cfg = B.BodyConfig(
        duration_s=args.steps * 0.02, control_ms=20.0, seed=args.seed,
        device="cuda", readout="ALPN", vision=True, vision_hz=100.0)

    kw = {}
    ring = None
    seen = {"кадров": 0, "обновлений": 0}
    if has_hook:
        from flybrain import ipc

        ring = ipc.EyeRing(shape=(512, 450, 3), omma=(721, 2), slots=3)

        def on_vision(step, t_ms, fisheye, omma, updates, renders):
            seen["обновлений"] += int(updates)
            img = np.clip(np.asarray(fisheye), 0.0, 255.0).astype(np.uint8,
                                                                  copy=False)
            seen["кадров"] += 1
            ring.write(float(t_ms), img, np.asarray(omma, dtype=np.float32),
                       2 * seen["обновлений"], seen["обновлений"],
                       seen["кадров"])

        kw["on_vision"] = on_vision

    # Время меряется ПО ШАГАМ, а не по всему вызову. Первая редакция
    # делила всё время `run_episode` на число шагов — а туда входит
    # сборка модели MuJoCo, и она занимает порядка тридцати секунд при
    # одной секунде модельного времени. То есть 95 % измеряемой
    # величины приходилось на то, что в обеих сборках одинаково, и
    # разницу, которую ищем, оно разбавляло в двадцать раз.
    marks = []

    def on_step(step, rec, bins_row):
        marks.append(time.perf_counter())

    kw["on_step"] = on_step
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False, **kw)
    wall = time.perf_counter() - t0
    # Первый шаг отбрасывается: на нём ещё доигрывается разогрев и
    # заводятся графы движка.
    per_step = ((marks[-1] - marks[1]) * 1000.0 / max(len(marks) - 2, 1)
                if len(marks) > 2 else float("nan"))
    build_s = (marks[0] - t0) if marks else wall
    if ring is not None:
        ring.close()
        ring.unlink()

    model_s = args.steps * 0.02
    out = {
        "дерево": str(root.parents[2]),
        "крючок есть": bool(has_hook),
        "шагов": args.steps,
        "зерно": args.seed,
        "по стене, с": round(wall, 3),
        "сборка тела, с": round(build_s, 2),
        "шагов замерено": max(len(marks) - 2, 0),
        "мс на шаг управления": round(per_step, 3),
        "доля реального на шагах": round(20.0 / per_step, 5),
        "доля реального со сборкой": round(model_s / wall, 5),
        "кадров в кольцо": seen["кадров"],
        "обновлений сетчатки": seen["обновлений"],
        "зрение в прогоне": (run.timing.get("зрение") or {}).get(
            "обновлений кадра за сеанс"),
    }
    print("RESULT " + json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
