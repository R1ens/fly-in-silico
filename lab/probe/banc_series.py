r"""Парная серия: объявленные веса против весов по BANC.

Предсказание и приёмка объявляются ДО прогона, иначе проверка не может
провалиться.

Что уже замерено короткой серией (6 пар по 3 с, `banc_series-3s.json`):

* команда поворота выросла в **1,185** раза при предсказанных 1,169 —
  это прямое следствие знаменателя ``Σ|w|``, который падает с 1,800 до
  1,539, потому что из шести зондов под запахом стреляет один `DNa02`;
* знак команды совпал на 6 парах из 6;
* поворот курса вырос примерно вдвое;
* **наведение на источник не улучшилось** — выросло на 2 парах из 6.

Короткая серия на этом и кончилась: эпизод 3 с, из них 1,5 с съедает
разогрев декодера, и `grad_cos` за остаток скачет от −0,48 до +0,21 в
обоих условиях. У такого сценария нет разрешающей способности.

Длинная серия ставит вопрос так, чтобы на него можно было ответить:

* эпизод **10 с**, разогрев те же 1,5 с, на замер остаётся 8,5;
* **16 пар** на условие — столько же, сколько в опубликованном опыте
  про наведение через ALPN (+0,142 против −0,035, p = 0,006);
* **отрицательный контроль: переставленные антенны** (`swap_antennae`).
  Он проверяет саму метрику: если при переставленных антеннах
  наведение НЕ падает, значит метрика меряет снос, а не рулёжку, и
  тогда разницу между весами по ней обсуждать нельзя.

Приёмка, объявленная заранее:

1. разница по наведению считается настоящей, только если она
   положительна **не менее чем на 12 парах из 16** (биномиальный порог
   при p < 0,05) И медиана разности превышает размах, который даёт
   отрицательный контроль;
2. знак команды поворота не должен перевернуться ни в одной паре;
3. при переставленных антеннах наведение обязано упасть к нулю в обоих
   условиях.

Метрика наведения — `grad_alignment`, среднее `grad_cos` по эпизоду:
ровно та, которой считался опубликованный результат (`loop.py`).
Печатается и по всему эпизоду, и без разогрева; сравнение ведётся по
второй, а первая даётся для сопоставимости с опубликованным числом.

Оговорка о сопоставимости: опубликованные +0,142 сняты в ПЛОСКОЙ арене
(`loop.bench`), здесь же тело в MuJoCo. Это разные сценарии, и числа
сравнивать между собой нельзя — совпадает только протокол.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\banc_series.py --пар 16 --секунд 10 --контроль 4
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from flybrain import data as D  # noqa: E402

КАРТА = (Path(__file__).resolve().parents[1] / "flybrain" / "assets"
         / "motor_map_banc_v888.json")
ОТЧЁТ = Path(__file__).parent / "banc_series.json"
CONTROL_MS = 20.0
TAU_S = 0.5


def эпизод(neurons, weights, device, seed, карта, секунд, swap=False):
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=секунд, control_ms=CONTROL_MS, seed=seed,
                       device=device, readout="ALPN", motor="dn_linear",
                       swap_antennae=bool(swap),
                       motor_map=(str(карта) if карта else None))
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False)
    wall = time.perf_counter() - t0
    recs = run.records
    skip = min(int(round(3 * TAU_S / (CONTROL_MS / 1000.0))), len(recs) // 3)
    r = recs[skip:]
    amp_l = np.array([x["amp_l"] for x in r], float)
    amp_r = np.array([x["amp_r"] for x in r], float)
    команда = amp_l - amp_r
    th = np.unwrap(np.array([x["theta"] for x in recs], float))
    x = np.array([v["x"] for v in recs], float)
    y = np.array([v["y"] for v in recs], float)
    c, s = math.cos(th[0]), math.sin(th[0])
    снос = -(x - x[0]) * s + (y - y[0]) * c
    все_gc = [v.get("grad_cos") for v in recs if v.get("grad_cos") is not None]
    gc = [v.get("grad_cos") for v in r if v.get("grad_cos") is not None]
    ключи = [k for k in recs[0] if "DNa02" in k]
    return {
        "зерно": seed, "карта": bool(карта), "антенны переставлены": bool(swap),
        "команда, медиана |L-R|": float(np.median(np.abs(команда))),
        "команда, знак": float(np.sign(np.median(команда))),
        "поворот, рад": float(th[-1] - th[0]),
        "снос, мм": float(снос[-1]),
        "grad_alignment": float(np.mean(все_gc)) if все_gc else None,
        "grad_alignment без разогрева": float(np.mean(gc)) if gc else None,
        "дистанция, мм": float(recs[-1].get("dist", float("nan"))),
        "зонд DNa02": {k: float(np.mean([v[k] for v in r])) for k in ключи},
        "шагов": len(recs), "стена, с": round(wall, 1),
    }


def пара(neurons, weights, device, seed, секунд, swap, i):
    """Оба условия на одном зерне, порядок внутри пары чередуется."""
    порядок = [None, КАРТА] if i % 2 == 0 else [КАРТА, None]
    снято = {}
    for карта in порядок:
        e = эпизод(neurons, weights, device, seed, карта, секунд, swap)
        снято["BANC" if карта else "объявленные"] = e
        print("  %s зерно %d, %-12s команда %.4f, поворот %+.3f, "
              "наведение %+.4f (%.0f с)"
              % ("контроль" if swap else "        ", seed,
                 "BANC" if карта else "объявленные",
                 e["команда, медиана |L-R|"], e["поворот, рад"],
                 e["grad_alignment без разогрева"] or 0.0, e["стена, с"]),
              flush=True)
    return снято


def свод(пары, имя):
    a = np.array([p["объявленные"]["grad_alignment без разогрева"]
                  for p in пары], float)
    b = np.array([p["BANC"]["grad_alignment без разогрева"] for p in пары],
                 float)
    ка = np.array([p["объявленные"]["команда, медиана |L-R|"] for p in пары])
    кb = np.array([p["BANC"]["команда, медиана |L-R|"] for p in пары])
    знак = sum(1 for p in пары
               if p["объявленные"]["команда, знак"] == p["BANC"]["команда, знак"])
    print("\n--- %s, пар %d ---" % (имя, len(пары)))
    print("  наведение: объявленные медиана %+.4f, BANC медиана %+.4f"
          % (np.median(a), np.median(b)))
    print("  разность по парам: медиана %+.4f, размах %+.4f…%+.4f, "
          "положительна на %d из %d"
          % (np.median(b - a), (b - a).min(), (b - a).max(),
             int((b > a).sum()), len(пары)))
    print("  команда: отношение медиана %.3f, знак совпал %d из %d"
          % (np.median(кb / np.where(ка == 0, np.nan, ка)), знак, len(пары)))
    return {"пар": len(пары),
            "наведение, объявленные медиана": float(np.median(a)),
            "наведение, BANC медиана": float(np.median(b)),
            "разность медиана": float(np.median(b - a)),
            "разность размах": [float((b - a).min()), float((b - a).max())],
            "положительных пар": int((b > a).sum()),
            "знак команды совпал": знак,
            "отношение команды": float(
                np.median(кb / np.where(ка == 0, np.nan, ка)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--пар", type=int, default=16)
    ap.add_argument("--секунд", type=float, default=10.0)
    ap.add_argument("--контроль", type=int, default=4,
                    help="пар с переставленными антеннами")
    args = ap.parse_args()
    import torch

    n = int(getattr(args, "пар"))
    секунд = float(getattr(args, "секунд"))
    nk = int(getattr(args, "контроль"))
    print("приёмка объявлена до прогона: разница по наведению настоящая "
          "только при 12 парах из 16 и медиане выше разброса контроля; "
          "знак команды не переворачивается; при переставленных антеннах "
          "наведение падает к нулю в обоих условиях", flush=True)
    print("эпизодов %d по %.0f с модельного" % (2 * (n + nk), секунд),
          flush=True)

    neurons = D.load_neurons()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = D.load_weights(device=device)

    пары = [пара(neurons, weights, device, 100 + i, секунд, False, i)
            for i in range(n)]
    контроль = [пара(neurons, weights, device, 200 + i, секунд, True, i)
                for i in range(nk)]

    итог = {"основная": свод(пары, "основная серия")}
    if контроль:
        итог["контроль"] = свод(контроль, "контроль: антенны переставлены")
        ка = np.median([abs(p["объявленные"]["grad_alignment без разогрева"])
                        for p in контроль])
        кб = np.median([abs(p["BANC"]["grad_alignment без разогрева"])
                        for p in контроль])
        print("\n  наведение при переставленных антеннах по модулю: "
              "объявленные %.4f, BANC %.4f" % (ка, кб))
    ОТЧЁТ.write_text(json.dumps(
        {"предсказание": "Σ|w| 1,800 -> 1,539, команда +17 %",
         "приёмка": "12 из 16 пар и медиана выше разброса контроля",
         "секунд на эпизод": секунд,
         "итог": итог, "пары": пары, "контроль": контроль},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nотчёт: %s" % ОТЧЁТ.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
