"""Проверка обмена с работником без всякого интерфейса.

Отвечает на пять вопросов, каждый из которых потом невозможно отделить от
окна: доходит ли ``Hello`` и за сколько, с какой частотой идут кадры
активности, попадают ли кадры картинки в кольцо и не течёт ли при этом
память, и за сколько работник встаёт по ``Cancel``.

Запуск:
    set PYTHONIOENCODING=utf-8
    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\ipc_test.py
"""

from __future__ import annotations

import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import ipc, worker  # noqa: E402

OUT = Path(__file__).resolve().parent


def say(ok, title, text=""):
    print(("  [ок] " if ok else "  [!!] ") + title + (f": {text}" if text else ""),
          flush=True)


def collect(ch, until, timeout, on_msg=None):
    """Разбирать очереди, пока ``until(msg)`` не вернёт True или не выйдет срок."""
    msgs = []
    t0 = time.perf_counter()
    stop = None
    while time.perf_counter() - t0 < timeout:
        batch = ch.drain()
        if not batch:
            time.sleep(0.005)
            continue
        for m in batch:
            msgs.append(m)
            if on_msg is not None:
                on_msg(m)
            if until(m):
                stop = m
        if stop is not None:
            return msgs, stop
    return msgs, None


def frame_stats(frames):
    if len(frames) < 2:
        return None
    arrive = [f[0] for f in frames]
    gaps = [b - a for a, b in zip(arrive, arrive[1:])]
    ks = [len(f[1].idx) for f in frames]
    return {
        "n": len(frames),
        "hz": 1.0 / statistics.mean(gaps),
        "gap_ms": statistics.mean(gaps) * 1000.0,
        "k_mean": statistics.mean(ks),
        "k_max": max(ks),
    }


def main():
    mp.set_start_method("spawn", force=True)
    cmd = mp.Queue()
    ch = ipc.Channel()
    proc = mp.Process(target=worker.run, args=(cmd, ch), daemon=True)

    print("=" * 70)
    print("1. поднимаю работника")
    t0 = time.perf_counter()
    proc.start()
    import psutil

    ps = psutil.Process(proc.pid)

    # ---------------------------------------------------------------- Hello
    print("\n2. жду Hello")
    msgs, hello = collect(ch, lambda m: isinstance(m, ipc.Hello), timeout=300,
                          on_msg=lambda m: isinstance(m, ipc.Status)
                          and print(f"     {m.state:7s} {m.phase}", flush=True))
    if hello is None:
        say(False, "Hello не пришёл", str(msgs[-3:]))
        proc.terminate()
        return 1
    boot_s = time.perf_counter() - t0
    say(True, "Hello", f"{boot_s:.1f} с")
    print(f"     нейронов {hello.n_neurons}, устройство {hello.device}")
    print(f"     областей {len(hello.regions)}, популяций {len(hello.populations)},"
          f" выходов {len(hello.readouts)}")
    print(f"     позиции {hello.positions_quantised.shape}"
          f" {hello.positions_quantised.dtype},"
          f" коды {hello.region_code.shape} {hello.region_code.dtype}")
    print(f"     bbox lo {hello.bbox['lo']} hi {hello.bbox['hi']}")
    print(f"     каналы {list(hello.sensors)}")
    print(f"     кольцо {hello.frame_ring}")
    ok_hello = (hello.n_neurons == len(hello.region_code)
                and hello.positions_quantised.shape == (hello.n_neurons, 3)
                and len(hello.populations) > 20)
    say(ok_hello, "состав Hello", "полный" if ok_hello else "чего-то нет")

    ring = ipc.FrameRing.from_descriptor(hello.frame_ring)

    # ----------------------------------------------------------------- stim
    print("\n3. задача stim: class:gustatory+side:left, 150 Гц, 0,4 с")
    frames, series, logs = [], [], []
    t0 = time.perf_counter()
    cmd.put(ipc.Submit("stim-1", {
        "kind": "stim",
        "activate": [{"query": "class:gustatory+side:left", "rate": 150.0}],
        "t": 0.4, "bin": 10.0,
    }))

    def note(m):
        if isinstance(m, ipc.Frame):
            frames.append((time.perf_counter(), m))
        elif isinstance(m, ipc.Series):
            series.append(m)
        elif isinstance(m, ipc.Log):
            logs.append(m)
        elif isinstance(m, ipc.Error):
            print("     ошибка:", m.message, "\n", m.tb, flush=True)

    msgs, done = collect(ch, lambda m: isinstance(m, (ipc.Done, ipc.Error)),
                         timeout=180, on_msg=note)
    wall = time.perf_counter() - t0
    if not isinstance(done, ipc.Done):
        say(False, "stim не доехал", str(done))
        cmd.put(ipc.Shutdown())
        return 1
    st = frame_stats(frames)
    say(st is not None and st["n"] == 40, "кадры stim",
        f"{st['n']} шт (ждали 40) за {wall:.1f} с")
    print(f"     частота {st['hz']:.1f} Гц, шаг {st['gap_ms']:.0f} мс,"
          f" ненулевых в кадре {st['k_mean']:.0f} (макс {st['k_max']})")
    print(f"     Series {len(series)}, Log {len(logs)},"
          f" выброшено кадров {done.summary.get('frames_dropped')}")
    print(f"     итог: {done.summary.get('n_active')} активных,"
          f" {done.summary.get('total_spikes')} спайков,"
          f" счёт {done.summary['timing']['sim_s']} с")
    f0 = frames[0][1]
    say(f0.idx.dtype == np.int32 and f0.val.dtype == np.uint8, "типы кадра",
        f"idx {f0.idx.dtype}, val {f0.val.dtype}")
    stim_hz = st["hz"]

    # ----------------------------------------------------------------- body
    def body_job(job_id, seconds):
        """Эпизод в теле: кадры из кольца и память работника по шагам."""
        frames, series, pics, rss = [], [], [], []
        shapes = set()
        seen_seq = [0]
        t0 = time.perf_counter()
        rss_before = ps.memory_info().rss
        cmd.put(ipc.Submit(job_id, {"kind": "body", "t": seconds,
                                    "readout": "ALPN", "dn_tau": 0.12,
                                    "control_ms": 20.0}))

        def note(m):
            if isinstance(m, ipc.Frame):
                frames.append((time.perf_counter(), m))
            elif isinstance(m, ipc.Series):
                series.append(m)
            elif isinstance(m, ipc.Error):
                print("     ошибка:", m.message, "\n", m.tb, flush=True)
            elif isinstance(m, ipc.Status) and m.state != "running":
                print(f"     {m.state:7s} {m.phase}", flush=True)
            seq, img = ring.latest()
            if img is not None and seq != seen_seq[0]:
                seen_seq[0] = seq
                pics.append((seq, float(img.mean())))
                shapes.add(img.shape)
                rss.append(ps.memory_info().rss)

        _, done = collect(ch, lambda m: isinstance(m, (ipc.Done, ipc.Error)),
                          timeout=900, on_msg=note)
        wall = time.perf_counter() - t0
        return dict(done=done, frames=frames, series=series, pics=pics,
                    shapes=shapes, rss=rss, rss_before=rss_before,
                    rss_after=ps.memory_info().rss, wall=wall)

    print("\n4. задача body: 1,5 с")
    r = body_job("body-1", 1.5)
    done = r["done"]
    if not isinstance(done, ipc.Done):
        say(False, "body не доехал", str(done))
        cmd.put(ipc.Shutdown())
        return 1
    st = frame_stats(r["frames"])
    pics, rss = r["pics"], r["rss"]
    say(st is not None, "кадры активности",
        f"{st['n']} шт, {st['hz']:.1f} Гц, ненулевых {st['k_mean']:.0f}")
    say(len(pics) > 0, "кадры картинки",
        f"{len(pics)} новых из кольца, размер {r['shapes']},"
        f" яркость {statistics.mean(p[1] for p in pics):.1f}"
        if pics else "не пришло ни одного")
    print(f"     работник отрендерил {done.summary.get('picture_frames')} кадров"
          f" по {done.summary.get('render_ms')} мс,"
          f" шагов {done.summary.get('n_steps')}, за {r['wall']:.1f} с")
    # Сборка тела стоит сотни мегабайт одноразово, поэтому утечку ищем
    # внутри эпизода: от первого снятого кадра до последнего.
    if len(rss) > 2:
        grow = (rss[-1] - rss[0]) / 1e6
        print(f"     память по шагам {rss[0] / 1e6:.0f} -> {rss[-1] / 1e6:.0f} МБ"
              f" ({grow:+.1f} за {len(rss)} кадров,"
              f" {grow * 1e3 / len(rss):+.0f} КБ на кадр)")
    print(f"     память вокруг задачи {r['rss_before'] / 1e6:.0f} ->"
          f" {r['rss_after'] / 1e6:.0f} МБ (вместе со сборкой тела)")
    print(f"     метрики: {done.summary.get('metrics')}")
    if r["series"]:
        print(f"     ряды {len(r['series'])}, ключи {sorted(r['series'][0].values)}")
    seq, img = ring.latest()
    if img is not None:
        try:
            import imageio.v3 as iio

            iio.imwrite(OUT / "ipc_body_frame.png", np.asarray(img))
            print(f"     последний кадр записан в {OUT / 'ipc_body_frame.png'}")
        except Exception as exc:  # noqa: BLE001
            print("     png не записан:", repr(exc)[:120])

    print("\n4б. второй эпизод, 0,8 с: не растёт ли память от эпизода к эпизоду")
    r2 = body_job("body-2", 0.8)
    if isinstance(r2["done"], ipc.Done):
        leak = (r2["rss_after"] - r["rss_after"]) / 1e6
        say(abs(leak) < 200, "память между эпизодами",
            f"{r['rss_after'] / 1e6:.0f} -> {r2['rss_after'] / 1e6:.0f} МБ"
            f" ({leak:+.0f})")
        print(f"     кадров картинки {len(r2['pics'])},"
              f" отрендерено {r2['done'].summary.get('picture_frames')},"
              f" за {r2['wall']:.1f} с")
    else:
        say(False, "второй эпизод не доехал", str(r2["done"]))
    rss0, rss1 = r["rss_before"], r2["rss_after"]

    # ------------------------------------------------------- ошибка и запрос
    print("\n5. ошибка задачи и запрос: работник должен выжить")
    cmd.put(ipc.Submit("bad-1", {"kind": "stim",
                                 "activate": [{"query": "type:неттакого"}]}))
    _, bad = collect(ch, lambda m: isinstance(m, (ipc.Error, ipc.Done)),
                     timeout=60)
    say(isinstance(bad, ipc.Error), "ошибка пришла как Error",
        bad.message if isinstance(bad, ipc.Error) else str(bad))
    cmd.put(ipc.Query("q-1", "select", "class:gustatory+side:left"))
    _, ans = collect(ch, lambda m: isinstance(m, ipc.Answer), timeout=60)
    say(isinstance(ans, ipc.Answer) and ans.payload.get("n", 0) > 0,
        "запрос после ошибки", str(ans.payload)[:120] if ans else "нет ответа")

    # ----------------------------------------------------------------- отмена
    print("\n6. отмена: длинная задача и Cancel через секунду")
    cmd.put(ipc.Submit("stim-long", {
        "kind": "stim",
        "activate": [{"query": "class:gustatory+side:left", "rate": 150.0}],
        "t": 10.0, "bin": 10.0,
    }))
    # Ждём, пока счёт действительно пойдёт, иначе отмена придёт в setup.
    collect(ch, lambda m: isinstance(m, ipc.Frame), timeout=120)
    # Заодно проверяем, что запрос посреди счёта не ждёт конца прогона.
    tq = time.perf_counter()
    cmd.put(ipc.Query("q-2", "select", "super:descending+side:left"))
    _, ans = collect(ch, lambda m: isinstance(m, ipc.Answer), timeout=30)
    say(isinstance(ans, ipc.Answer), "ответ посреди счёта",
        f"{(time.perf_counter() - tq) * 1000:.0f} мс,"
        f" n={ans.payload.get('n') if ans else '-'}")
    time.sleep(1.0)
    t0 = time.perf_counter()
    cmd.put(ipc.Cancel("stim-long"))
    msgs, done = collect(ch, lambda m: isinstance(m, (ipc.Done, ipc.Error)),
                         timeout=120)
    stop_s = time.perf_counter() - t0
    ok = isinstance(done, ipc.Done) and stop_s < 1.0
    say(ok, "остановился", f"{stop_s * 1000:.0f} мс после Cancel")
    if isinstance(done, ipc.Done):
        print(f"     отмечено cancelled={done.summary.get('cancelled')},"
              f" шагов {done.summary['timing']['steps']} из 100000")

    print("\n6б. отмена во время разогрева тела")
    cmd.put(ipc.Submit("body-stop", {"kind": "body", "t": 5.0,
                                     "readout": "ALPN", "dn_tau": 0.5}))
    _, warm = collect(ch, lambda m: isinstance(m, ipc.Status)
                      and m.state == "warmup", timeout=300)
    say(warm is not None, "дождался разогрева")
    t0 = time.perf_counter()
    cmd.put(ipc.Cancel("body-stop"))
    _, done = collect(ch, lambda m: isinstance(m, (ipc.Done, ipc.Error)),
                      timeout=300)
    warm_stop = time.perf_counter() - t0
    say(isinstance(done, ipc.Done) and warm_stop < 5.0, "остановился",
        f"{warm_stop:.1f} с после Cancel,"
        f" шагов {done.summary.get('n_steps') if isinstance(done, ipc.Done) else '-'}")

    # --------------------------------------------------------------- выключение
    print("\n7. выключение")
    cmd.put(ipc.Shutdown())
    proc.join(timeout=30)
    say(not proc.is_alive(), "работник вышел", f"код {proc.exitcode}")
    ring.close()

    print("\n" + "=" * 70)
    print("итог")
    print(f"  Hello за {boot_s:.1f} с, кадры stim {stim_hz:.1f} Гц,"
          f" кадры тела {len(pics)} шт, отмена за {stop_s * 1000:.0f} мс")
    print(f"  память работника {rss0 / 1e6:.0f} -> {rss1 / 1e6:.0f} МБ"
          f" за два эпизода со сборкой тела")
    return 0


if __name__ == "__main__":
    sys.exit(main())
