# -*- coding: utf-8 -*-
r"""Откуда у окна росла память и что её остановило. Без работника.

Долгий сеанс намерил рост памяти окна **541 -> 1408 МБ за 105 секунд**
с открытым видом глаз — около десяти мегабайт в секунду, ровно и без
насыщения. Эта проба отвечает, откуда он брался, тремя замерами, и
каждый из них может провалиться.

1. **Цена идёт за запусками ядер, а не за нашими массивами.**
   Меряется рост памяти на кадр для возрастающего набора работы:
   только очистка, плюс тон, плюс фигуры, плюс глифы, плюс картинка.
   Если бы дело было в том, что мы подаём в ядро новые массивы, рост
   шёл бы за их размером; он идёт за числом запусков.
2. **`ti.sync()` раз в кадр возвращает память.** Один и тот же кадр,
   600 раз, с синхронизацией и без.
3. **Постоянный буфер под команды ничего не даёт.** Это замер
   отвергнутой правки: превращение питоновского списка команд в массив
   стоит одинаково, кладём ли мы его в новый массив или в заранее
   отведённый буфер. Числа взяты чередующимися кругами в одном
   процессе.

Taichi здесь настоящий, окна нет: меряется ровно отрисовка.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\draw_leak.py
"""

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import taichi as ti  # noqa: E402

ti.init(arch=ti.vulkan, offline_cache=True)

from flybrain.desk.tokens import C  # noqa: E402
from flybrain.ui import font, gpu  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def rss():
    gc.collect()
    return psutil.Process().memory_info().rss / (1 << 20)


def main():                                                   # noqa: C901
    surf = gpu.Surface()
    surf.ensure(1280, 720)
    at = font.bake(16)

    dl_rect = gpu.DrawList()
    for i in range(300):
        dl_rect.rect(float(i), float(i), float(i + 40), float(i + 24),
                     C.panel, 1.0, 3.0)
    dl_text = gpu.DrawList()
    for i in range(60):
        dl_text.text(at, "проверка %d" % i, 20.0, 10.0 + i * 11, C.ink)
    img = (np.random.rand(512, 450, 3) * 255).astype(np.uint8)
    dl_img = gpu.DrawList()
    dl_img.image(img, 10, 10, 400, 400)

    def leg(fn, n=600):
        for _ in range(60):
            fn()
        m0 = rss()
        for _ in range(n):
            fn()
        return (rss() - m0) / n

    # --- 1. рост идёт за запусками ---------------------------------
    виды = [
        ("только очистка", lambda: surf.clear(C.bg)),
        ("+ тон", lambda: (surf.clear(C.bg), surf._k_tone(1.0))),
        ("+ фигуры", lambda: (surf.clear(C.bg), surf.draw(dl_rect, None))),
        ("+ глифы", lambda: (surf.clear(C.bg), surf.draw(dl_text, at))),
        ("+ картинка", lambda: (surf.clear(C.bg), surf.draw(dl_img, None))),
    ]
    шаги = {}
    for имя, fn in виды:
        шаги[имя] = round(leg(fn), 4)
        print("  %-18s %+.4f МБ на кадр" % (имя, шаги[имя]), flush=True)
    rep["замеры"]["рост по видам работы, МБ на кадр"] = шаги
    ряд = [шаги[и] for и, _ in виды]
    check("рост растёт вместе с числом запусков",
          all(b >= a - 0.002 for a, b in zip(ряд, ряд[1:])),
          " < ".join("%.4f" % v for v in ряд))
    check("одна очистка течёт заметно меньше полного кадра",
          ряд[0] * 3 < ряд[-1],
          "%.4f против %.4f" % (ряд[0], ряд[-1]))

    # --- 2. синхронизация возвращает память ------------------------
    def кадр(sync):
        def go():
            surf.clear(C.bg)
            surf.draw(dl_rect, None)
            surf.draw(dl_text, at)
            if sync:
                surf.finish()
        return go

    без = round(leg(кадр(False)) * 600, 2)
    с = round(leg(кадр(True)) * 600, 2)
    rep["замеры"]["600 кадров без синхронизации, МБ"] = без
    rep["замеры"]["600 кадров с синхронизацией, МБ"] = с
    check("синхронизация раз в кадр снимает рост", с < без / 10.0,
          "%+.2f МБ против %+.2f за 600 кадров" % (с, без))

    # --- 3. отвергнутая правка: буфер под команды ------------------
    dl = gpu.DrawList()
    rng = np.random.default_rng(0)
    for i in range(1500):
        x, y = float(rng.integers(0, 1200)), float(rng.integers(0, 680))
        dl.rect(x, y, x + 50, y + 30, C.panel, 1.0, 3.0)
    keep = {}

    def в_буфер():
        n, cols = len(dl.cmds), len(dl.cmds[0])
        b = keep.get("cmd")
        if b is None or b.shape[0] < n:
            b = np.zeros((int(n * 1.25) + 64, cols), dtype=np.float32)
            keep["cmd"] = b
        b[:n] = dl.cmds
        return b

    def в_новый():
        return np.ascontiguousarray(np.asarray(dl.cmds, dtype=np.float32))

    def круг(fn, n=200):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / n

    пары = []
    for k in range(3):
        a, b = круг(в_новый), круг(в_буфер)
        пары.append((round(a, 3), round(b, 3)))
        print("  круг %d: новый массив %.3f мс, буфер %.3f мс"
              % (k + 1, a, b), flush=True)
    rep["замеры"]["команд в массив, мс (новый, буфер)"] = пары
    медиана = float(np.median([b / a for a, b in пары]))
    rep["замеры"]["отношение буфер/новый"] = round(медиана, 4)
    check("постоянный буфер под команды выигрыша не даёт",
          0.9 < медиана < 1.1, "отношение %.4f" % медиана)

    (OUT / "draw_leak.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
