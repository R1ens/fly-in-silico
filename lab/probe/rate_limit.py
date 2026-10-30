"""Предел частоты подачи: где он, кто его проверяет и что без проверки.

Почему эта проба существует
---------------------------

Пуассоновский вход считается ``torch.bernoulli(rates * DT/1000)``. У
`bernoulli` вероятность обязана лежать в [0, 1], и проверяет это **ассерт
на карте** (`ATen/native/cuda/DistributionTemplates.h:624`,
``Assertion '0 <= p4 && p4 <= 1' failed``). При DT = 0,1 мс вероятность
доходит до единицы ровно на 10 000 Гц.

Это не косметика, и вот почему. **Ассерт на карте портит контекст CUDA
необратимо**, и дальше одна и та же причина докладывается РАЗНЫМИ
текстами тому, кто первым спросит у карты ответ. Замерено в одном
процессе: сразу после ассерта `cusparseCreate` сказал ``initialization
error``, а следующая честная сборка движка отказала с ``device-side
assert triggered``, и обе трассы указали на `sim.py`, то есть НЕ на
место ошибки. Это ровно та подмена места, которую замерил разбор отказа
ядра (`KERNEL_FAULT.md` §4): виноватым называется не тот, кто ошибся, а
тот, кто первым синхронизовался.

**Чем это НЕ является.** Доказательством, что тот единичный
`CUDA illegal memory access` пришёл отсюда, — нет. Под поставленными
настройками ни один сенсорный канал предела не достигает (фоторецепторы
150 Гц, обоняние порядка 100), и этого текста здесь не воспроизведено.
Закрыт именно НЕЗАЩИЩЁННЫЙ ПУТЬ: накачка `DnIntervention`, потолок
фоторецепторов и усиление зрительного переходника задаются человеком и
ничем не были ограничены. Прямой стимул из окна был ограничен и до
этого (`live.MAX_STIM_HZ` = 1000 Гц, и там названа та же причина).

Что проверяется
---------------

1. **Порог поштучно**: 9 999 и 10 000 Гц проходят, 10 001 отказывает.
   Это и есть встречный замер к самому пределу: если бы проверка
   отказывала раньше, порог был бы выдуман.
2. **Каждый, кто пишет частоты, отказывает ГРОМКО и называет место** —
   отказ `sim.RateTooHigh` с именем канала, а не ассерт карты.
3. **Отрицательная частота тоже отказ**: `bernoulli` падает на обоих
   концах, и проверка обязана ловить оба.
4. **Без проверки карта действительно падает** — иначе проверка
   охраняла бы то, чего нет. Этот пункт идёт ОТДЕЛЬНЫМ ПРОЦЕССОМ: после
   ассерта контекст CUDA не восстанавливается, и продолжать в том же
   процессе нечем.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\rate_limit.py

Коды: 0 — всё сошлось; 1 — есть провал.
"""

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


def _без_проверки():
    """Отдельный процесс: шаг с частотой выше предела БЕЗ проверки.

    Проверка снимается подменой `sim.check_rates_hz` на пустышку — то
    есть ровно то, что было до неё, и ничего больше. Контекст CUDA после
    ассерта не восстановить, поэтому и отдельный процесс.
    """
    код = (
        "import sys, json; sys.path.insert(0, '.')\n"
        "import probe._isolate\n"
        "import torch\n"
        "from flybrain import data as D, sim\n"
        "sim.check_rates_hz = lambda v, где: v\n"
        "n = D.load_neurons(); W = D.load_weights(device='cuda')\n"
        "eng = sim.build_engine(weights=W, neurons=n, device='cuda',"
        " seed=0, graphs=False)\n"
        "r = torch.zeros(1, len(n), device='cuda'); r[0, :10] = 50000.0\n"
        "итог = {}\n"
        "try:\n"
        "    eng.step(r); torch.cuda.synchronize(); итог['отказ'] = None\n"
        "except BaseException as e:\n"
        "    итог['отказ'] = type(e).__name__\n"
        "    итог['текст'] = str(e).splitlines()[0]\n"
        "print('ИТОГ' + json.dumps(итог, ensure_ascii=False))\n")
    корень = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-I", "-c", код], cwd=str(корень),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    итог = {"код возврата": r.returncode, "stderr хвост":
            (r.stderr or "")[-400:]}
    for строка in (r.stdout or "").splitlines():
        if строка.startswith("ИТОГ"):
            итог.update(json.loads(строка[4:]))
            return итог
    # Процесс может и не дойти до печати: ассерт на карте валит его на
    # месте. Тогда свидетельство — код возврата и текст ассерта в
    # stderr, и так это и записывается, а не выдаётся за пойманное
    # исключение.
    итог["отказ"] = "процесс не дошёл до печати"
    итог["текст"] = ""
    for строка in (r.stderr or "").splitlines():
        if "Assertion" in строка or "device-side assert" in строка:
            итог["текст"] = строка.strip()
            break
    return итог


def main():
    import torch
    from flybrain import body as B, data as D, motor as M, sim, vision as V

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("— предел частоты подачи —")
    print("  DT = %.4f мс, prob_scale = %.6g, предел = %.0f Гц"
          % (sim.DT, sim.DT / 1000.0, sim.MAX_RATE_HZ))
    rep["предел, Гц"] = sim.MAX_RATE_HZ
    check("предел равен 1000/DT, а не числу из головы",
          abs(sim.MAX_RATE_HZ - 1000.0 / sim.DT) < 1e-9, sim.MAX_RATE_HZ)

    # --- 1. порог поштучно -------------------------------------------
    for hz, ждём in ((9999.0, True), (10000.0, True), (10001.0, False),
                     (-1.0, False)):
        прошло = True
        текст = ""
        try:
            sim.check_rates_hz([0.0, hz], "проба порога")
        except sim.RateTooHigh as e:
            прошло = False
            текст = str(e).splitlines()[0][:70]
        check("%s Гц %s" % (hz, "проходит" if ждём else "отказывает"),
              прошло == ждём, текст or "прошло")

    # --- 2. каждый писатель отказывает громко ------------------------
    neurons = D.load_neurons()

    # накачка нисходящих: отказ при СБОРКЕ, до первого шага
    try:
        M.DnIntervention(neurons, dev, drive="DNa02=50000", silence=None)
        check("накачка выше предела отказывает при сборке", False,
              "собралась молча")
    except sim.RateTooHigh as e:
        check("накачка выше предела отказывает при сборке", True,
              str(e).splitlines()[0][:60])
    except Exception as e:
        check("накачка выше предела отказывает при сборке", False,
              "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:60]))

    # фоторецепторы: потолок задаётся человеком
    vis = V.make("retina", mode="whiten", assign="capacity", shuffle=False,
                 seed=0, channel="max", max_hz=50000.0, ref=0.5,
                 eyes="both")
    vis.attach(neurons, dev)
    кадр = np.full((2, 721, 2), 0.9, dtype=np.float32)
    rates = torch.zeros(1, len(neurons), device=dev)
    try:
        vis.feed(rates, кадр)
        check("подача фоторецепторов выше предела отказывает", False,
              "записалась молча")
    except sim.RateTooHigh as e:
        check("подача фоторецепторов выше предела отказывает", True,
              str(e).splitlines()[0][:60])

    # встречный замер: при поставленном потолке 150 Гц та же подача идёт
    vis.max_hz = 150.0
    свод = vis.feed(rates, кадр)
    check("при поставленном потолке та же подача ПРОХОДИТ",
          свод["нейронов"] > 0 and свод["наибольшая, Гц"] <= 150.0 + 1e-6,
          "%d нейронов, наибольшая %.3f Гц"
          % (свод["нейронов"], свод["наибольшая, Гц"]))

    # --- 3. без проверки карта падает --------------------------------
    print("  отдельный процесс: шаг выше предела БЕЗ проверки…")
    без = _без_проверки()
    rep["без проверки"] = без
    check("без проверки шаг НЕ проходит",
          без.get("отказ") is not None,
          "%s (код возврата %s)" % (без.get("отказ"),
                                    без.get("код возврата")))
    check("и причина — именно ассерт `0 <= p4 && p4 <= 1` на карте",
          "0 <= p4" in str(без.get("текст", "")),
          без.get("текст", "")[:90] or "ассерта в stderr не нашлось")

    (OUT / "rate_limit.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    if fails:
        print("ПРОВАЛЫ: %s" % ", ".join(fails))
        return 1
    print("предел замерен, и все, кто пишет частоты, его держат")
    return 0


if __name__ == "__main__":
    sys.exit(main())
