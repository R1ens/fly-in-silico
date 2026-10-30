"""Воспроизводим ли эпизод СО ЗРЕНИЕМ: в одном процессе и между процессами.

Почему эта проба появилась
--------------------------

Подтверждающий прогон сквозного опыта дал на том же коде, той же
конфигурации и тех же зёрнах ДРУГИЕ числа в двух условиях из четырёх:
счёт спайков `DNp01` у условия «без предмета» вышел
``[176, 131, 76, 126, 148, 92]`` в одном прогоне и
``[176, 131, 162, 126, 121, 92]`` в другом — расходятся зёрна 2 и 4, а
четыре остальных совпадают точно. Условия «нависание» и «заглушён
DNp01» при этом воспроизвелись полностью.

Вся проверка этого проекта стоит на побитовой воспроизводимости, поэтому
такое расхождение — находка, а не помеха, и его надо либо объяснить,
либо назвать необъяснённым.

Главный подозреваемый назван заранее: **кадр глаза приходит с
видеокарты.** Рендер `Physics.render` идёт через GL-контекст, и его
последние биты не обязаны совпадать между процессами (другой контекст,
другое состояние драйвера). Частоты фоторецепторов считаются из кадра, а
сеть пороговая — значит расхождение в последних битах может перевернуть
одно пересечение порога и дальше разойтись на весь эпизод.

Отсюда и устройство пробы: **два уровня, и они различают причину.**

* **A/A в одном процессе** — два эпизода подряд с одним зерном. Если
  расходятся уже здесь, причина внутри процесса, а не в контексте;
* **A/A между процессами** — тот же эпизод, запущенный отдельным
  процессом. Если в одном процессе сходится, а между процессами нет,
  причина в том, что зависит от процесса: GL-контекст, состояние
  драйвера, порядок захвата.

Встречный замер обязателен и встроен: **без зрения** та же пара обязана
совпасть и в одном процессе, и между процессами. Без него «совпало»
ничего не значило бы — совпасть могло бы потому, что проба сравнивает
две копии одного ответа.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\vision_repro.py
    ..\\.venv\\Scripts\\python.exe probe\\vision_repro.py --длина 1.4 --зёрен 3

Коды: 0 — всё сошлось; 1 — есть расхождение.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent

fails = []
rep = {"проверки": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  — " + text)
    rep.setdefault("оговорки", []).append(text)


def отпечаток(run):
    """Отпечаток эпизода: траектория, растр и счёт спайков вместе."""
    recs = run.records
    поля = ("x", "y", "theta", "odour_l", "odour_r", "dn_l", "dn_r",
            "amp_l", "amp_r", "spikes")
    t = np.array([[r.get(k, np.nan) for k in поля] for r in recs],
                 dtype=np.float64)
    bins = np.asarray(run.bins.cpu().numpy() if hasattr(run.bins, "cpu")
                      else run.bins)
    counts = np.asarray(run.counts.cpu().numpy()
                        if hasattr(run.counts, "cpu") else run.counts)
    h = hashlib.sha256()
    for a in (t, bins, counts):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def эпизод(зрение, зерно, длина, neurons, weights, dev):
    from flybrain import body as B, objects as O

    kw = dict(duration_s=длина, control_ms=20.0, seed=зерно, device=dev,
              readout="ALPN")
    if зрение:
        kw.update(vision=True, vis_objects=1, vision_hz=100.0,
                  vis_track=O.preset("loom_dark"))
    run = B.run_episode(cfg=B.BodyConfig(**kw), neurons=neurons,
                        weights=weights, progress=False)
    return отпечаток(run)


def в_другом_процессе(зрение, зерно, длина):
    """Тот же эпизод отдельным процессом. Печатает только отпечаток."""
    код = (
        "import sys, json; sys.path.insert(0, '.')\n"
        "import probe._isolate\n"
        "from probe.vision_repro import эпизод\n"
        "import torch\n"
        "from flybrain import data as D\n"
        "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "n = D.load_neurons(); W = D.load_weights(device=dev)\n"
        "fp = эпизод(%r, %d, %r, n, W, dev)\n"
        # Метка ASCII нарочно: кодировка стандартного вывода дочернего
        # процесса нам не подчинена, и кириллическая метка приходила
        # искажённой — проба тогда докладывала «нет вывода» при
        # РАБОТАЮЩЕМ дочернем процессе. Поймано прогоном, а потом
        # подтверждено ручным вызовом того же дочернего кода.
        "print('FP=' + fp)\n" % (bool(зрение), int(зерно), float(длина)))
    корень = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-I", "-c", код], cwd=str(корень),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=3600)
    for строка in (r.stdout or "").splitlines():
        if "FP=" in строка:
            return строка.split("FP=", 1)[1].strip()
    # Отказ разбора обязан отличаться от отказа процесса: иначе «нет
    # вывода» читается как «процесс не дошёл», а он дошёл.
    return ("нет метки FP= в выводе; код возврата %s; хвост stderr: %s"
            % (r.returncode, (r.stderr or "")[-200:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--длина", type=float, default=0.6)
    ap.add_argument("--зёрен", type=int, default=2)
    a = ap.parse_args()

    import torch
    from flybrain import data as D

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("— воспроизводимость эпизода со зрением —")
    print("  устройство %s, эпизод %.2f с, зёрен %d"
          % (dev, a.длина, a.зёрен))
    neurons = D.load_neurons()
    weights = D.load_weights(device=dev)
    rep["устройство"] = dev
    rep["длина, с"] = a.длина

    for зрение, имя in ((False, "без зрения"), (True, "со зрением")):
        свои = {}
        for зерно in range(a.зёрен):
            # A/A в одном процессе: два эпизода подряд, одно зерно.
            п1 = эпизод(зрение, зерно, a.длина, neurons, weights, dev)
            п2 = эпизод(зрение, зерно, a.длина, neurons, weights, dev)
            свои[зерно] = (п1, п2)
            check("%s, зерно %d: A/A в ОДНОМ процессе совпало"
                  % (имя, зерно), п1 == п2, "%s / %s" % (п1, п2))
        for зерно in range(a.зёрен):
            # A/A между процессами: тот же эпизод отдельным процессом.
            чужой = в_другом_процессе(зрение, зерно, a.длина)
            check("%s, зерно %d: A/A МЕЖДУ процессами совпало"
                  % (имя, зерно), чужой == свои[зерно][0],
                  "%s / %s" % (свои[зерно][0], чужой))
        rep[имя] = {str(k): {"в процессе": list(v)} for k, v in свои.items()}

    (OUT / "vision_repro.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    if fails:
        print("РАСХОЖДЕНИЯ: %s" % ", ".join(fails))
        return 1
    print("эпизод воспроизводим и в одном процессе, и между процессами")
    return 0


if __name__ == "__main__":
    sys.exit(main())
