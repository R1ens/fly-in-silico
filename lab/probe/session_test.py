"""Проверка связки окна с работником без окна.

Смотрим на то, ради чего `session.py` и написан: что активность копится с
затуханием по времени окна, а не скачет на каждом редком кадре работника, и
что кадр тела доходит из кольца.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.ui.session import Session                         # noqa: E402


def pump_until(sess, cond, limit=120.0, fps=60.0):
    t0 = time.perf_counter()
    last = t0
    while time.perf_counter() - t0 < limit:
        now = time.perf_counter()
        sess.pump(now - last)
        last = now
        if cond(sess):
            return True
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - now)))
    return False


def main():
    print("поднимаю работника...")
    t0 = time.perf_counter()
    sess = Session()
    if not pump_until(sess, lambda s: s.ready or s.error, limit=180):
        print("[ПРОВАЛ] Hello не пришёл")
        sess.close()
        return 1
    print("[ок] Hello за " + format(time.perf_counter() - t0, ".1f") + " с: "
          + str(sess.n) + " нейронов, " + sess.device + ", "
          + str(len(sess.hello.regions)) + " областей, "
          + str(len(sess.hello.populations)) + " популяций, кольцо "
          + ("есть" if sess.ring is not None else "НЕТ"))

    # --- стимуляция: следим за сглаживанием между кадрами
    spec = {"kind": "stim", "activate": [
        {"query": "class:gustatory+side:left", "rate": 200.0, "on": 0.0}],
        "t": 0.6, "bin_ms": 10.0}
    print("\nстимуляция " + spec["activate"][0]["query"] + ", 0,6 с")
    sess.submit(spec)

    samples = []
    frames_at = []
    t0 = time.perf_counter()
    last = t0
    while time.perf_counter() - t0 < 60:
        now = time.perf_counter()
        got = sess.pump(now - last)
        last = now
        if any(type(m).__name__ == "Frame" for m in got):
            frames_at.append(now - t0)
        if sess.act is not None:
            samples.append((now - t0, float(sess.disp.max()),
                            int((sess.act > 0.5).sum())))
        if sess.state in ("done", "error"):
            break
        time.sleep(1.0 / 60.0)

    if sess.state == "error":
        print("[ПРОВАЛ] " + str(sess.error))
        sess.close()
        return 1

    gaps = np.diff(frames_at) if len(frames_at) > 1 else np.zeros(1)
    print("[ок] кадров " + str(sess.frames_seen) + ", приход "
          + format(1.0 / max(gaps.mean(), 1e-6), ".1f") + " Гц, разборов окна "
          + str(len(samples)))
    print("     активных " + str(sess.n_active) + ", пик "
          + format(sess.peak_hz, ".0f") + " Гц, шкала "
          + format(sess.scale, ".0f"))

    # между приходами кадров картинка должна течь, а не стоять
    arr = np.asarray([s[1] for s in samples], dtype=np.float32)
    steps = np.abs(np.diff(arr))
    big = int((steps > 0.2).sum())
    print("     яркость карты: " + str(len(arr)) + " отсчётов, скачков больше "
          + "0,2 — " + str(big) + " (должно быть мало: кадры редкие, "
          + "а картинка течёт)")
    if sess.summary:
        print("     итог: " + str({k: sess.summary[k] for k in
                                   list(sess.summary)[:5]}))

    # --- тело: кадр из кольца
    print("\nтело, 1,2 с")
    sess.submit({"kind": "body", "t": 1.2, "readout": "ALPN", "control_ms": 20.0})
    seqs = []
    t0 = time.perf_counter()
    last = t0
    while time.perf_counter() - t0 < 180:
        now = time.perf_counter()
        sess.pump(now - last)
        last = now
        if sess.body_frame is not None and (not seqs or seqs[-1] != sess.body_seq):
            seqs.append(sess.body_seq)
        if sess.state in ("done", "error"):
            break
        time.sleep(1.0 / 60.0)

    if sess.state == "error":
        print("[ПРОВАЛ] " + str(sess.error))
    else:
        print("[ок] кадров тела " + str(len(seqs)) + ", последний seq "
              + str(sess.body_seq) + ", размер "
              + str(None if sess.body_frame is None else sess.body_frame.shape))
        print("     ряды: " + str(sorted(sess.series)))
        print("     последняя запись шага: "
              + str({k: sess.extra[k] for k in list(sess.extra)[:6]}))
        if sess.body_frame is not None:
            from PIL import Image
            Image.fromarray(np.asarray(sess.body_frame)).save(
                Path(__file__).parent / "session_body.png")
            print("     кадр записан в session_body.png")

    print("\nжурнал, последние строки:")
    for line in sess.log[-4:]:
        print("   " + line[:110])

    sess.close()
    print("\nработник остановлен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
