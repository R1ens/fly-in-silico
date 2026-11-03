"""Три величины одновременно, в НАСТОЯЩЕМ окне: показ, глаза, счёт.

Прежняя проба (`eye_view.py`) гоняет то же приложение без окна: там
кадр собирается в поверхность, swapchain не участвует, и частоты показа
у неё нет вовсе — она честно показывает прочерк. Здесь открывается
настоящее окно тем же способом, что по двойному щелчку, и меряется то,
чего без окна измерить нельзя.

Четыре величины, и ни одна не выводится из другой:

* **A — частота показа окна.** Законченные выводы кадра на экран за
  настенную секунду. Считает сам `Shell`, после `window.show()`.
  Вертикальная синхронизация включена, поэтому сверху величина
  упирается в частоту монитора; чтобы запас был виден, рядом меряется
  время сборки кадра и его хвосты p95/p99.
* **B — сырые рендеры камер** в секунду по стене, по одному на глаз.
* **C — обновления сетчатки** в секунду по стене: полные парные
  наблюдения, те самые, что получила сеть.
* **D — кадры глаз, дошедшие до окна** в секунду по стене.

И отдельно — **доля реального времени** всего контура.

Замер идёт двумя отрезками по тридцать секунд: с открытым видом глаз и
со сценой. Разность отвечает на вопрос, которого иначе не закрыть, —
сколько стоит сам показ глаз.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\eye_fps.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_eye_fps_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
BOOT_S = 240.0
BUILD_S = 300.0
LEG_S = 30.0

fails = []
rep = {"проверки": {}, "замеры": {}, "снимки": [], "находки": [],
       "сорвалось": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def load():
    """Загрузка машины: процессор, видеокарта, её память.

    Спрашивается раз в секунду, а не каждый кадр: `nvidia-smi` — это
    запуск процесса, и шестьдесят запусков в секунду мерили бы сами
    себя. Если его нет, строка будет пустой, а не выдуманной.
    """
    import subprocess

    import psutil

    gpu = mem = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        a, b = out.stdout.strip().splitlines()[0].split(",")
        gpu, mem = int(a), int(b)
    except Exception:                                      # noqa: BLE001
        pass
    return {"процессор, %": psutil.cpu_percent(None),
            "видеокарта, %": gpu, "видеопамять, МБ": mem,
            "окно, МБ": round(psutil.Process().memory_info().rss
                              / (1 << 20), 1)}


def main():                                                   # noqa: C901
    from flybrain.desk.app import App
    from flybrain.desk.shell import Shell
    from flybrain.ui.session import Session

    sh = Shell("FlyBrain — замер глаз", (1600, 900))
    ses = Session()
    app = App(sh, ses)
    st = app.st
    st.ws.current = "sandbox"
    sb = st.sb
    sb.t_run = 0.0
    sb.keep_s = 10.0
    sb.vision = True
    t0 = time.perf_counter()

    def leg(label, seconds=LEG_S):
        """Отрезок замера. Все величины снимаются ОДНОВРЕМЕННО.

        Именно одновременно: взять показ в одну минуту, а наблюдения в
        другую — и получится пара чисел, которых вместе никогда не было.
        """
        fps, ms, load_pts = [], [], []
        next_load = 0.0
        ren0, upd0, tak0 = ses.eye_renders, ses.eye_updates, ses.eye_taken
        t_ms0 = ses.t_ms
        w0 = time.perf_counter()
        end = w0 + seconds
        while time.perf_counter() < end and sh.running:
            a = time.perf_counter()
            app.frame()
            ms.append((time.perf_counter() - a) * 1000.0)
            if sh.fps > 0:
                fps.append(sh.fps)
            if time.perf_counter() > next_load:
                next_load = time.perf_counter() + 1.0
                load_pts.append(load())
        wall = time.perf_counter() - w0
        arr = np.asarray(ms, dtype=np.float64)
        out = {
            "секунд по стене": round(wall, 2),
            "A: показ окна, кадр/с": (round(float(np.median(fps)), 1)
                                      if fps else None),
            "кадр окна, мс": {
                "медиана": round(float(np.median(arr)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "p99": round(float(np.percentile(arr, 99)), 2),
                "наибольший": round(float(arr.max()), 2)},
            "B: рендеров камер, 1/с": round(
                (ses.eye_renders - ren0) / wall, 2),
            "C: обновлений сетчатки, 1/с": round(
                (ses.eye_updates - upd0) / wall, 2),
            "D: кадров глаз в окно, 1/с": round(
                (ses.eye_taken - tak0) / wall, 2),
            "доля реального": round((ses.t_ms - t_ms0) / 1000.0 / wall, 4),
            "вытеснено внутри шага": int(ses.eye_lost_in_step),
            "вытеснено кольцом": int(ses.eye_missed),
            "задержка доставки, мс": round(ses.eye_lag_s * 1000.0, 1),
        }
        if load_pts:
            def med(key):
                vals = [p[key] for p in load_pts if p[key] is not None]
                return (round(float(np.median(vals)), 1) if vals else None)

            out["загрузка (медиана)"] = {
                "процессор, %": med("процессор, %"),
                "видеокарта, %": med("видеокарта, %"),
                "видеопамять, МБ": med("видеопамять, МБ"),
                "окно, МБ": med("окно, МБ"),
            }
        rep["замеры"][label] = out
        print("  " + label + ": " + json.dumps(out, ensure_ascii=False),
              flush=True)
        return out

    try:
        while not st.worker.ready and time.perf_counter() - t0 < BOOT_S:
            app.frame()
        check("работник готов", st.worker.ready,
              "за %.1f с" % (time.perf_counter() - t0))
        if not st.worker.ready:
            return

        app.start_run()
        t_wait = time.perf_counter()
        while ses.eye_img is None and time.perf_counter() - t_wait < BUILD_S:
            app.frame()
        check("кадр глаз пошёл", ses.eye_img is not None,
              "через %.1f с" % (time.perf_counter() - t_wait))
        if ses.eye_img is None:
            rep["сорвалось"].append("кадра глаз не дождались")
            return

        # Порядок ГЛАЗА / СЦЕНА / СЦЕНА / ГЛАЗА. Первая редакция гнала
        # по одному отрезку на вид подряд, и разность вида смешалась с
        # разностью времени: счёт во втором отрезке шёл медленнее
        # (0,0656 против 0,0746) просто потому, что он был вторым. Тот
        # же урок, что в `eye_view.py`, и цена та же — ложный вывод.
        def show(eye_view, label, shot=None):
            sb.eye_view = bool(eye_view)
            for _ in range(120):
                app.frame()
            if shot:
                sh.shot(OUT / shot)
            return leg(label)

        a1 = show(True, "вид глаз, отрезок 1", "eyefps-01-глаза.png")
        b1 = show(False, "вид сцены, отрезок 1", "eyefps-02-сцена.png")
        b2 = show(False, "вид сцены, отрезок 2")
        a2 = show(True, "вид глаз, отрезок 2")

        def mid(x, y, key):
            return 0.5 * (x[key] + y[key])

        a = {k: (mid(a1, a2, k) if isinstance(a1[k], (int, float)) else a1[k])
             for k in a1}
        b = {k: (mid(b1, b2, k) if isinstance(b1[k], (int, float)) else b1[k])
             for k in b1}
        a["кадр окна, мс"] = {k: 0.5 * (a1["кадр окна, мс"][k]
                                        + a2["кадр окна, мс"][k])
                              for k in a1["кадр окна, мс"]}
        b["кадр окна, мс"] = {k: 0.5 * (b1["кадр окна, мс"][k]
                                        + b2["кадр окна, мс"][k])
                              for k in b1["кадр окна, мс"]}
        rep["замеры"]["вид глаз, среднее по двум отрезкам"] = a
        rep["замеры"]["вид сцены, среднее по двум отрезкам"] = b

        fa = a["A: показ окна, кадр/с"]
        check("показ окна держится не ниже 30 кадр/с с видом глаз",
              fa is not None and fa >= 30.0, "%s кадр/с" % fa)
        check("показ окна держится не ниже 30 кадр/с со сценой",
              b["A: показ окна, кадр/с"] is not None
              and b["A: показ окна, кадр/с"] >= 30.0,
              "%s кадр/с" % b["A: показ окна, кадр/с"])
        check("частота показа не выдаётся за частоту наблюдений",
              fa is not None and a["D: кадров глаз в окно, 1/с"] < fa,
              "показ %s против %s кадров глаз"
              % (fa, a["D: кадров глаз в окно, 1/с"]))
        check("рендеров вдвое больше обновлений и на отрезке",
              abs(a["B: рендеров камер, 1/с"]
                  - 2 * a["C: обновлений сетчатки, 1/с"]) < 0.05,
              "%s против %s" % (a["B: рендеров камер, 1/с"],
                                2 * a["C: обновлений сетчатки, 1/с"]))
        rep["замеры"]["цена показа глаз"] = {
            "кадр окна, мс, медиана": round(
                a["кадр окна, мс"]["медиана"]
                - b["кадр окна, мс"]["медиана"], 3),
            "доля реального": round(a["доля реального"]
                                    - b["доля реального"], 4),
        }
        check("показ глаз не замедляет счёт",
              abs(a["доля реального"] - b["доля реального"])
              <= 0.1 * max(b["доля реального"], 1e-9),
              "%s против %s" % (a["доля реального"], b["доля реального"]))
        app.cancel_run()
        t_wait = time.perf_counter()
        while ses.busy and time.perf_counter() - t_wait < 120:
            app.frame()
    finally:
        ses.close()
        sh.close()

    (OUT / "eye_fps.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if (fails or rep["сорвалось"]) else 0)
