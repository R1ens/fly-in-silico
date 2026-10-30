"""Доставка ядра на машину БЕЗ `nvcc` и MSVC: замером, а не рассуждением.

Вопрос владельца был поставлен так: запуск приложения не должен требовать
`nvcc`/MSVC только для того, чтобы считать, — и при этом не должно
появиться ни тихого отката, ни неконтролируемых численных расхождений.

Прежняя дорога собирала `ev_spmv.cu` вызовом `nvcc` на машине
пользователя. Новая дорога (`evspmv.LOAD_PTX`, теперь по умолчанию)
собирает PTX ОДИН РАЗ здесь, кладёт его в дерево рядом с источником, а на
машине пользователя догружает JIT драйвера через `nvcuda.dll`
(`cuModuleLoadDataEx` плюс `cuLaunchKernel`) — хостового кода на C больше
нет вовсе, значит не нужен и компилятор. Прежняя дорога осталась
эталоном доставки и выбирается явно (`FLYBRAIN_EV_LOAD=nvcc`).

Двадцать шесть проверок в семи разделах, из них двенадцать — встречные
замеры. Что проверяется и почему именно это.

1. **Происхождение артефакта.** PTX в дереве — это не бинарник
   неизвестного происхождения: он пересобирается здесь из `ev_spmv.cu`
   тем же `nvcc` и сверяется с лежащим в дереве ПОБАЙТОВО. Встречный
   замер: в копии артефакта меняется один символ, и сверка обязана это
   увидеть, иначе она ничего не стоит.
2. **Загрузка и цена первого запуска.** JIT драйвера стоит времени, и
   его надо знать отдельно от установившейся скорости. Мерится в трёх
   отдельных процессах: с пустым кэшем драйвера, с уже прогретым, и с
   кэшем, отключённым вовсе (`CUDA_CACHE_DISABLE=1`). Так видно и цену
   JIT, и то, работает ли кэш драйвера на этой машине.
3. **Побитовое равенство.** Шесть сочетаний дорога×графы: `csrmv`,
   событийное через PTX и событийное через DLL, каждое с захватом графов
   и без. Все шесть обязаны дать ОДИН отпечаток состояния. Отдельно —
   число захваченных графов счётчиком, а не по скорости: захват в этом
   проекте отказывает молча.
4. **Скорость попарно.** Перемежающимися кругами в одном процессе, на
   настоящем коннектоме и настоящем шаге движка с графами. Встречный
   замер на оснастку — круг `csrmv` против `csrmv`: его размах и есть
   порог, ниже которого «разница» от фона не отличима.
5. **Машина без `nvcc` и без MSVC.** Подменой путей, а не рассуждением:
   в отдельном процессе `FLYBRAIN_NVCC` и `FLYBRAIN_VCVARS` уводятся в
   никуда, `CUDA_PATH*` убирается, а `PATH` чистится от CUDA и MSVC.
   Проба требует ДВУХ вещей сразу: что `nvcc` в этом окружении и правда
   недостижим (иначе замер ничего не значит), и что движок при этом
   СЧИТАЕТ и даёт тот же отпечаток.
6. **Громкость отказов.** Шесть порч, каждая в своём процессе: нет
   артефакта, нет манифеста, артефакт испорчен, источник правили без
   пересборки артефакта, манифест от другого abi и PTX с версией ISA
   выше той, что понимает драйвер (нижняя граница драйвера — настоящее
   ограничение этой дороги, и отказ по ней обязан НАЗЫВАТЬ причину).
   Каждая обязана дать `EventKernelUnavailable` с причиной — ни одна не
   смеет превратиться в тихую сборку через `nvcc` или в тихий `csrmv`.
7. **Нижняя граница драйвера.** Единственная настоящая цена этой
   дороги: драйвер обязан понимать `.version` артефакта, а nvcc 12.8
   пишет 8.7. Мерится, насколько границу можно опустить, не меняя счёта:
   тот же текст грузится с пониженной `.version`, и результат обязан
   остаться побитово тем же. Встречный замер — версия, заниженная
   слишком сильно: она обязана быть отвергнута ГРОМКО.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\kernel_ship.py

Пересобрать артефакт в дереве (делается на машине разработчика после
любой правки `ev_spmv.cu`):

    ..\\.venv\\Scripts\\python.exe probe\\kernel_ship.py --пересобрать
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
LAB = Path(__file__).resolve().parents[1]

fails = []
rep = {"проверки": {}, "замеры": {}, "встречные замеры": {}, "находки": []}


def check(name, ok, info=""):
    print(("  ok      " if ok else "  ПРОВАЛ  ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def fp(a):
    """Отпечаток от сырых байтов: ловит расхождение в последнем бите."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def state_fp(eng):
    return {"v": fp(eng.v), "проводимость": fp(eng.conductance),
            "кольцо": fp(eng.delay_buffer), "рефрактерность": fp(eng.refrac),
            "указатель": int(eng.ptr)}


def run_child(body, env_extra=None, timeout=900, tag="child"):
    """Выполнить кусок кода в ОТДЕЛЬНОМ процессе и вернуть его вывод.

    Отдельный процесс нужен всюду, где меняется окружение или портится
    артефакт: в своём процессе этого не отменить, а `evspmv` держит
    загруженное ядро на процесс.
    """
    src = textwrap.dedent(body) % {"lab": str(LAB),
                                   "out": os.environ["FLYBRAIN_OUT"]}
    script = Path(os.environ["FLYBRAIN_OUT"]) / ("kernel_ship_%s.py" % tag)
    script.write_text(src, encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    for key in list(env):
        if env[key] == "<убрать>":
            env.pop(key)
    got = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         env=env, timeout=timeout)
    return got


def say_json(got):
    """Последняя строка вывода ребёнка — JSON. Остальное — для разбора."""
    lines = [ln for ln in (got.stdout or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("ребёнок не отдал JSON (код %d):\n%s\n%s"
                       % (got.returncode, (got.stdout or "")[-2000:],
                          (got.stderr or "")[-2000:]))


# --- 1: происхождение артефакта ---------------------------------------


def part_origin():
    """PTX пересобирается здесь и сверяется с лежащим в дереве."""
    from flybrain import evspmv

    man_tree = json.loads(evspmv.PTX_MANIFEST.read_text(encoding="utf-8"))
    tree = evspmv.PTX.read_text(encoding="utf-8")
    for k in ("источник", "цель", "версия ptx", "собран чем", "abi",
              "отпечаток источника", "отпечаток артефакта"):
        print("     %-22s %s" % (k, man_tree.get(k)))

    t0 = time.perf_counter()
    text, man = evspmv.compile_ptx()
    took = time.perf_counter() - t0
    rep["замеры"]["происхождение"] = {
        "в дереве": man_tree, "пересобрано сейчас": man,
        "сборка PTX, с": round(took, 2),
        "размер PTX, байт": len(tree.encode("utf-8")),
    }
    # Сравнение знак в знак, а не байт в байт, и это не поблажка: в
    # репозитории `core.autocrlf=true`, то есть текст выгружается с CRLF
    # и сравнение сырых байтов упало бы на переводе строк, ничего не
    # сказав о счёте. Обе стороны читаются через `read_text`, который
    # приводит перевод строк к `\n` — ровно так же, как читает `evspmv`.
    check("PTX в дереве знак в знак совпадает с пересобранным из "
          "`ev_spmv.cu` тем же nvcc",
          text == tree,
          "%d байт, сборка %.2f с, %s"
          % (len(tree.encode("utf-8")), took, man["отпечаток артефакта"]))
    check("манифест артефакта сходится с самим артефактом и с источником",
          man["отпечаток артефакта"] == man_tree["отпечаток артефакта"]
          and man["отпечаток источника"] == man_tree["отпечаток источника"]
          and man["abi"] == evspmv.ABI,
          "abi %d, цель %s, ptx %s"
          % (man["abi"], man["цель"], man["версия ptx"]))
    check("в артефакте есть обе точки входа, и имена у них простые",
          all((".entry %s" % k.decode()) in tree
              for k in evspmv.KERNELS.values()),
          ", ".join(k.decode() for k in evspmv.KERNELS.values()))

    # Артефакт — ТЕКСТ, а git на Windows с `autocrlf=true` выгружает
    # текст с CRLF. Считай отпечаток по сырым байтам — и движок отказался
    # бы грузить свой же артефакт на машине, которая просто клонировала
    # дерево. Поэтому отпечаток считается через `read_text` (перевод
    # строк приводится к `\n`), и это проверяется, а не подразумевается.
    raw = evspmv.PTX.read_bytes()
    as_git = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    crlf = Path(os.environ["FLYBRAIN_OUT"]) / "ev_spmv_crlf.ptx"
    crlf.write_bytes(as_git)
    # Читается ровно так, как читает `evspmv`: `read_text` приводит
    # перевод строк к `\n`, поэтому отпечаток обязан совпасть.
    same_text = evspmv._sha(crlf.read_text(encoding="utf-8"))
    rep["замеры"]["происхождение"] = dict(
        rep["замеры"].get("происхождение", {}),
        **{"байт в дереве": len(raw), "байт после CRLF": len(as_git)})
    check("отпечаток переживает выгрузку из git с CRLF "
          "(по сырым байтам он бы разошёлся)",
          same_text == man_tree["отпечаток артефакта"]
          and evspmv._sha(as_git) != evspmv._sha(raw),
          "%d байт -> %d байт, отпечаток тот же; по сырым байтам разошёлся"
          % (len(raw), len(as_git)))

    # Встречный замер на саму сверку: один символ в копии.
    spoiled = tree.replace("mul.f32", "mul.ftz.f32", 1)
    rep["встречные замеры"]["сверка ловит правку одного знака"] = {
        "что испорчено": "первое mul.f32 -> mul.ftz.f32",
        "сверка увидела": spoiled != tree,
    }
    check("сверка ловит правку ОДНОЙ инструкции (встречный замер)",
          spoiled != tree and evspmv._sha(spoiled)
          != man_tree["отпечаток артефакта"],
          "mul.f32 -> mul.ftz.f32: отпечаток разошёлся")


# --- 2: загрузка и цена первого запуска -------------------------------


LOAD_CHILD = r'''
    import json, os, sys, time
    sys.path.insert(0, %(lab)r)
    os.environ["FLYBRAIN_OUT"] = %(out)r
    import torch
    torch.zeros(1, device="cuda")          # контекст поднят как у движка
    from flybrain import evspmv
    t0 = time.perf_counter()
    kern, info = evspmv.build(load=evspmv.LOAD_PTX)
    took = time.perf_counter() - t0
    print(json.dumps({"загрузка, с": round(took, 4),
                      "по счётчику модуля, с": info["загрузка (JIT), с"],
                      "версия драйвера": info["версия драйвера"],
                      "цель": info["цель артефакта"],
                      "архитектура": info["архитектура"],
                      "кэш": os.environ.get("CUDA_CACHE_PATH", ""),
                      "кэш отключён": os.environ.get(
                          "CUDA_CACHE_DISABLE", "")},
                     ensure_ascii=False))
'''

BUILD_CHILD = r'''
    import json, os, sys, time
    sys.path.insert(0, %(lab)r)
    os.environ["FLYBRAIN_OUT"] = %(out)r
    import torch
    torch.zeros(1, device="cuda")
    from flybrain import evspmv
    t0 = time.perf_counter()
    kern, info = evspmv.build(load=evspmv.LOAD_NVCC, force=True)
    took = time.perf_counter() - t0
    print(json.dumps({"сборка, с": round(took, 4),
                      "по счётчику модуля, с": info["сборка, с"],
                      "собрано сейчас": info["собрано сейчас"]},
                     ensure_ascii=False))
'''


def part_load():
    """Цена первого запуска: JIT драйвера против сборки `nvcc`."""
    cache = Path(os.environ["FLYBRAIN_OUT"]) / "jitcache"
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)

    cold = say_json(run_child(LOAD_CHILD, {"CUDA_CACHE_PATH": str(cache),
                                           "CUDA_CACHE_DISABLE": "<убрать>"},
                              tag="load_cold"))
    warm = say_json(run_child(LOAD_CHILD, {"CUDA_CACHE_PATH": str(cache),
                                           "CUDA_CACHE_DISABLE": "<убрать>"},
                              tag="load_warm"))
    nocache = say_json(run_child(LOAD_CHILD, {"CUDA_CACHE_DISABLE": "1",
                                              "CUDA_CACHE_PATH": "<убрать>"},
                                 tag="load_nocache"))
    built = say_json(run_child(BUILD_CHILD, tag="build_nvcc"))

    files = sum(1 for _ in cache.rglob("*") if _.is_file())
    rep["замеры"]["цена первого запуска"] = {
        "PTX, кэш драйвера пуст, с": cold["загрузка, с"],
        "PTX, кэш драйвера прогрет, с": warm["загрузка, с"],
        "PTX, кэш драйвера отключён, с": nocache["загрузка, с"],
        "nvcc -> DLL, сборка с нуля, с": built["сборка, с"],
        "файлов в кэше драйвера после": files,
        "версия драйвера": cold["версия драйвера"],
    }
    for k, v in rep["замеры"]["цена первого запуска"].items():
        print("     %-34s %s" % (k, v))
    check("PTX грузится на порядок дешевле, чем nvcc собирает DLL",
          cold["загрузка, с"] < built["сборка, с"] / 5.0,
          "%.3f с против %.3f с — в %.0f раз"
          % (cold["загрузка, с"], built["сборка, с"],
             built["сборка, с"] / max(cold["загрузка, с"], 1e-9)))
    check("кэш JIT драйвера на этой машине работает: прогретая загрузка "
          "дешевле первой",
          warm["загрузка, с"] < cold["загрузка, с"] and files > 0,
          "пусто %.3f с -> прогрето %.3f с, файлов в кэше %d"
          % (cold["загрузка, с"], warm["загрузка, с"], files))
    check("даже с отключённым кэшем JIT загрузка остаётся дешевле сборки",
          nocache["загрузка, с"] < built["сборка, с"],
          "%.3f с против %.3f с" % (nocache["загрузка, с"],
                                    built["сборка, с"]))


# --- 3: побитовое равенство -------------------------------------------


def part_equal(ctx, steps=200):
    """Шесть сочетаний дорога×графы — один отпечаток на все."""
    import torch

    from flybrain import evspmv, sim

    neurons, W, device, n = ctx["neurons"], ctx["W"], ctx["device"], ctx["n"]
    kw = dict(weights=W, neurons=neurons, device=device, seed=11,
              no_refractory=ctx["driven"])
    out, got_graphs = {}, {}
    arms = (("csrmv + графы", sim.SPMV_CSR, None, True),
            ("csrmv без графов", sim.SPMV_CSR, None, False),
            ("PTX + графы", sim.SPMV_EVENT, evspmv.LOAD_PTX, True),
            ("PTX без графов", sim.SPMV_EVENT, evspmv.LOAD_PTX, False),
            ("DLL + графы", sim.SPMV_EVENT, evspmv.LOAD_NVCC, True),
            ("DLL без графов", sim.SPMV_EVENT, evspmv.LOAD_NVCC, False))
    for name, spmv, road, graphs in arms:
        if road:
            evspmv.LOAD_DEFAULT = road
        eng = sim.build_engine(spmv=spmv, graphs=graphs, **kw)
        rates = torch.zeros(1, n, device=device)
        rates[0, ctx["sens"]] = 150.0
        total = 0
        for step in range(steps):
            if step == 100:
                eng.set_masks(hold=ctx["busy"][:400],
                              inc=ctx["busy"][400:800])
            total += int(eng.step(rates).sum())
        out[name] = {"спайков": total, "графов": len(eng._graphs)
                     if eng._graphs else 0, **state_fp(eng)}
        got_graphs[name] = len(eng._graphs) if eng._graphs else 0
        if road:
            out[name]["дорога"] = eng._ev.build_info["дорога доставки"]
        del eng
        torch.cuda.empty_cache()
    evspmv.LOAD_DEFAULT = evspmv.LOAD_PTX
    rep["замеры"]["шесть сочетаний дорога×графы"] = out

    def body(v):
        return {k: x for k, x in v.items() if k not in ("графов", "дорога")}

    base = body(out["csrmv + графы"])
    same = {k: (body(v) == base) for k, v in out.items()}
    for k, v in out.items():
        print("     %-18s спайков %6d, графов %2d, v %s"
              % (k, v["спайков"], v["графов"], v["v"]))
    check("PTX-дорога пишется в CUDA-граф (по графу на слот кольца)",
          got_graphs["PTX + графы"] == ctx["buf_len"],
          "захвачено %d при buf_len %d"
          % (got_graphs["PTX + графы"], ctx["buf_len"]))
    check("все шесть сочетаний дорога×графы дают ОДИН отпечаток состояния",
          all(same.values()),
          "v %s, спайков %d, не совпало: %s"
          % (out["csrmv + графы"]["v"], out["csrmv + графы"]["спайков"],
             ", ".join(k for k, v in same.items() if not v) or "ничего"))
    rep["замеры"]["совпало с csrmv+графы"] = same


# --- 4: скорость попарно ----------------------------------------------


def part_speed(ctx, rounds=9, steps=400):
    """Перемежающимися кругами в одном процессе, на настоящем шаге движка.

    Пять плеч в КРУГЕ, подряд: csrmv, csrmv, DLL, PTX, PTX. Отношения
    считаются ВНУТРИ круга, потому что фон на видеокарте гуляет, и
    плечи, стоящие рядом во времени, видят один фон.

    Почему встречных замеров на оснастку ДВА, а не один. В первом
    прогоне порог брался по паре `csrmv` против `csrmv` — он вышел ±2 %,
    и по нему пара DLL/PTX (1,064) выглядела расхождением. Но это был
    порог не от того плеча: `csrmv` читает все 15 091 983 связи и потому
    ровен, а событийное ядро трогает 0,04 % матрицы, его шаг короткий, и
    разброс по кругам у него ВДЕСЯТЕРО больше (0,127…0,235 мс против
    0,282…0,289 мс). Порог обязан мериться плечом того же рода, поэтому
    здесь есть и пара PTX против PTX: её размах и есть та черта, ниже
    которой дороги доставки между собой не отличимы.
    """
    import torch

    from flybrain import evspmv, sim

    neurons, W, device, n = ctx["neurons"], ctx["W"], ctx["device"], ctx["n"]
    kw = dict(weights=W, neurons=neurons, device=device, seed=11,
              no_refractory=ctx["driven"])
    engines, rates = {}, torch.zeros(1, n, device=device)
    rates[0, ctx["sens"]] = 150.0
    order = ("csrmv", "csrmv2", "DLL", "PTX", "PTX2")
    for name, spmv, road in (("csrmv", sim.SPMV_CSR, None),
                             ("csrmv2", sim.SPMV_CSR, None),
                             ("DLL", sim.SPMV_EVENT, evspmv.LOAD_NVCC),
                             ("PTX", sim.SPMV_EVENT, evspmv.LOAD_PTX),
                             ("PTX2", sim.SPMV_EVENT, evspmv.LOAD_PTX)):
        if road:
            evspmv.LOAD_DEFAULT = road
        engines[name] = sim.build_engine(spmv=spmv, graphs=True, **kw)
        got = len(engines[name]._graphs) if engines[name]._graphs else 0
        if got != engines[name].buf_len:
            check("плечо %s захватило графы" % name, False,
                  "графов %d при buf_len %d" % (got, engines[name].buf_len))
    evspmv.LOAD_DEFAULT = evspmv.LOAD_PTX

    def one(eng):
        for _ in range(20):
            eng.step(rates)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            eng.step(rates)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / steps

    cols = {k: [] for k in engines}
    for r in range(rounds):
        for name in order:
            cols[name].append(one(engines[name]))
        print("     круг %d: csrmv %.4f / %.4f, DLL %.4f, PTX %.4f / %.4f"
              " мс/шаг"
              % (r + 1, cols["csrmv"][-1], cols["csrmv2"][-1],
                 cols["DLL"][-1], cols["PTX"][-1], cols["PTX2"][-1]),
              flush=True)

    def rat(a, b):
        rr = [x / y for x, y in zip(cols[a], cols[b])]
        return {"медиана": round(statistics.median(rr), 4),
                "мин": round(min(rr), 4), "макс": round(max(rr), 4)}

    res = {
        "мс/шаг (медиана по кругам)": {
            k: round(statistics.median(v), 4) for k, v in cols.items()},
        "мс/шаг (размах по кругам)": {
            k: [round(min(v), 4), round(max(v), 4)] for k, v in cols.items()},
        "csrmv / PTX": rat("csrmv", "PTX"),
        "csrmv / DLL": rat("csrmv", "DLL"),
        "DLL / PTX (дорога против дороги)": rat("DLL", "PTX"),
        "csrmv / csrmv (встречный: порог на ровном плече)": rat("csrmv",
                                                                "csrmv2"),
        "PTX / PTX (встречный: порог на СОБЫТИЙНОМ плече)": rat("PTX",
                                                                "PTX2"),
        "кругов": rounds, "шагов в круге": steps,
    }
    rep["замеры"]["скорость попарно"] = res
    for k, v in res.items():
        print("     %-50s %s" % (k, v))
    flat = res["csrmv / csrmv (встречный: порог на ровном плече)"]
    aa = res["PTX / PTX (встречный: порог на СОБЫТИЙНОМ плече)"]
    # Порог берётся по размаху одинаковых плеч, а не по медиане: вопрос
    # не «сошлись ли медианы», а «отличима ли разница от фона вообще».
    noise = max(abs(aa["макс"] - 1.0), abs(aa["мин"] - 1.0))
    dp = res["DLL / PTX (дорога против дороги)"]
    check("встречный замер на ровном плече: csrmv против csrmv около "
          "единицы",
          abs(flat["медиана"] - 1.0) <= 0.05,
          "медиана %.4f, размах %.4f…%.4f"
          % (flat["медиана"], flat["мин"], flat["макс"]))
    check("встречный замер на событийном плече: PTX против PTX около "
          "единицы, и он же даёт порог",
          abs(aa["медиана"] - 1.0) <= 0.15,
          "медиана %.4f, размах %.4f…%.4f -> порог ±%.3f"
          % (aa["медиана"], aa["мин"], aa["макс"], noise))
    check("дорога доставки на установившуюся скорость не влияет: "
          "DLL/PTX внутри порога событийного плеча",
          abs(dp["медиана"] - 1.0) <= noise,
          "DLL/PTX %.4f (%.4f…%.4f) при пороге ±%.3f по PTX/PTX"
          % (dp["медиана"], dp["мин"], dp["макс"], noise))
    check("событийное ядро по-прежнему быстрее csrmv на шаге движка",
          res["csrmv / PTX"]["медиана"] > 1.5,
          "csrmv/PTX %.3f, csrmv/DLL %.3f"
          % (res["csrmv / PTX"]["медиана"], res["csrmv / DLL"]["медиана"]))
    for eng in engines.values():
        del eng
    engines.clear()
    torch.cuda.empty_cache()


# --- 5: машина без nvcc и без MSVC ------------------------------------


BARE_CHILD = r'''
    import json, os, sys
    sys.path.insert(0, %(lab)r)
    os.environ["FLYBRAIN_OUT"] = %(out)r
    import hashlib
    import numpy as np
    import torch
    from flybrain import data as D, evspmv, sim, world

    # Сначала — что средства сборки в этом окружении И ПРАВДА недостижимы.
    # Без этого замер ничего не значит: движок мог бы считать потому, что
    # nvcc всё-таки нашёлся.
    reach = {}
    for name, fn in (("nvcc", evspmv.find_nvcc),
                     ("vcvars64", evspmv.find_vcvars)):
        try:
            reach[name] = str(fn())
        except evspmv.EventKernelUnavailable as exc:
            reach[name] = "НЕДОСТИЖИМ: " + str(exc).splitlines()[0]

    neurons = D.load_neurons()
    W = D.load_weights(device="cuda")
    n = len(neurons)
    sens = np.unique(np.concatenate([
        D.select(neurons, world.SENSORS["odour_left"]),
        D.select(neurons, world.SENSORS["odour_right"]),
        D.select(neurons, world.SENSORS["taste"])]))
    rates = torch.zeros(1, n, device="cuda")
    rates[0, torch.as_tensor(sens, device="cuda")] = 150.0
    eng = sim.build_engine(weights=W, neurons=neurons, device="cuda",
                           seed=11, no_refractory=sens,
                           spmv=sim.SPMV_EVENT, graphs=True)
    total = 0
    for step in range(200):
        if step == 100:
            n_in, n_out = eng.fan()
            busy = torch.argsort(n_in + n_out, descending=True)[:1200]
            busy = busy.cpu().numpy().astype(np.int64)
            eng.set_masks(hold=busy[:400], inc=busy[400:800])
        total += int(eng.step(rates).sum())

    def fp(a):
        raw = np.ascontiguousarray(a.detach().cpu().numpy()).tobytes()
        return hashlib.sha256(raw).hexdigest()[:16]

    print(json.dumps({
        "достижимость средств сборки": reach,
        "умножение связей": eng.spmv,
        "дорога доставки": eng._ev.build_info["дорога доставки"],
        "графов": len(eng._graphs) if eng._graphs else 0,
        "спайков": total, "v": fp(eng.v),
        "проводимость": fp(eng.conductance),
        "кольцо": fp(eng.delay_buffer),
        "рефрактерность": fp(eng.refrac),
        "указатель": int(eng.ptr),
        "which nvcc в PATH": __import__("shutil").which("nvcc"),
        "which cl в PATH": __import__("shutil").which("cl"),
    }, ensure_ascii=False))
'''


def bare_env():
    """Окружение машины, где нет ни CUDA Toolkit, ни MSVC.

    Не «не будем пользоваться», а именно недостижимо: явные пути уведены
    в никуда, `CUDA_PATH*` убраны, а из `PATH` вычищены все каталоги, в
    которых лежат `nvcc.exe` или `cl.exe`.
    """
    env = {"FLYBRAIN_NVCC": r"Z:\нет-такого\nvcc.exe",
           "FLYBRAIN_VCVARS": r"Z:\нет-такого\vcvars64.bat"}
    for key in os.environ:
        if key == "CUDA_PATH" or key.startswith("CUDA_PATH_V"):
            env[key] = "<убрать>"
    keep = []
    for part in (os.environ.get("PATH") or "").split(os.pathsep):
        if not part:
            continue
        p = Path(part)
        try:
            bad = (p / "nvcc.exe").exists() or (p / "cl.exe").exists()
        except OSError:
            bad = False
        if not bad:
            keep.append(part)
    env["PATH"] = os.pathsep.join(keep)
    return env


def part_bare(good):
    """Движок на машине без `nvcc` и MSVC: считает и даёт те же байты."""
    env = bare_env()
    # `FLYBRAIN_NVCC` указан и не существует — по правилам `evspmv` это
    # громкий отказ, а не поиск мимо просьбы. Но дорога PTX его не зовёт
    # вовсе, поэтому движок обязан посчитать.
    got = run_child(BARE_CHILD, env, tag="bare", timeout=1800)
    try:
        res = say_json(got)
    except RuntimeError as exc:
        rep["замеры"]["без nvcc и MSVC"] = {
            "код выхода": got.returncode,
            "stderr": (got.stderr or "")[-1500:]}
        check("на машине без nvcc и без MSVC движок СЧИТАЕТ", False, str(exc))
        return
    rep["замеры"]["без nvcc и MSVC"] = res
    for k in ("достижимость средств сборки", "дорога доставки", "графов",
              "спайков", "v", "which nvcc в PATH", "which cl в PATH"):
        print("     %-28s %s" % (k, res[k]))
    reach = res["достижимость средств сборки"]
    check("в этом окружении nvcc и MSVC и правда недостижимы "
          "(иначе замер ничего не значит)",
          all(str(v).startswith("НЕДОСТИЖИМ") for v in reach.values())
          and not res["which nvcc в PATH"] and not res["which cl в PATH"],
          "nvcc: %s | cl: %s" % (reach["nvcc"][:60], reach["vcvars64"][:60]))
    check("на машине без nvcc и без MSVC движок СЧИТАЕТ событийным ядром",
          res["умножение связей"] == "событийный"
          and res["спайков"] > 0 and "PTX" in res["дорога доставки"],
          "%s, спайков %d, графов %d"
          % (res["дорога доставки"], res["спайков"], res["графов"]))
    keys = ("спайков", "v", "проводимость", "кольцо", "рефрактерность",
            "указатель")
    body = {k: res[k] for k in keys}
    want = {k: good[k] for k in keys}
    rep["замеры"]["без nvcc и MSVC"]["сверено с"] = want
    check("и даёт ТЕ ЖЕ БАЙТЫ, что та же дорога при полном наборе средств",
          body == want,
          "v %s, спайков %d%s"
          % (body["v"], body["спайков"],
             "" if body == want
             else "; разошлось: %s" % [k for k in keys
                                       if body[k] != want[k]]))
    check("графы при этом захвачены все (счётчиком, а не по скорости)",
          res["графов"] == good["графов"] and res["графов"] > 0,
          "графов %d при %d в своём процессе"
          % (res["графов"], good["графов"]))


# --- 6: громкость отказов ---------------------------------------------


SPOIL_CHILD = r'''
    import json, os, shutil, sys
    from pathlib import Path
    sys.path.insert(0, %(lab)r)
    os.environ["FLYBRAIN_OUT"] = %(out)r
    import torch
    from flybrain import evspmv, sim

    # Артефакт и манифест переносятся в СВОЙ каталог и портятся там:
    # дерево проекта не трогается вовсе, иначе порча осталась бы в нём
    # при любом падении пробы.
    where = Path(%(out)r) / ("spoil_" + os.environ["SPOIL"])
    shutil.rmtree(where, ignore_errors=True)
    where.mkdir(parents=True, exist_ok=True)
    ptx, man = where / "ev_spmv.ptx", where / "ev_spmv.ptx.json"
    shutil.copyfile(evspmv.PTX, ptx)
    shutil.copyfile(evspmv.PTX_MANIFEST, man)
    evspmv.PTX, evspmv.PTX_MANIFEST = ptx, man

    how = os.environ["SPOIL"]
    if how == "нет-артефакта":
        ptx.unlink()
    elif how == "нет-манифеста":
        man.unlink()
    elif how == "испорчен-артефакт":
        t = ptx.read_text(encoding="utf-8")
        ptx.write_text(t.replace("mul.f32", "mul.ftz.f32", 1),
                       encoding="utf-8", newline="\n")
    elif how == "источник-правили":
        d = json.loads(man.read_text(encoding="utf-8"))
        d["отпечаток источника"] = "sha256:" + "0" * 64
        man.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    elif how == "чужой-abi":
        d = json.loads(man.read_text(encoding="utf-8"))
        d["abi"] = evspmv.ABI + 7
        man.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    elif how == "ptx-из-будущего":
        # Единственная порча, которая обязана дойти ДО JIT драйвера, а не
        # упасть на отпечатке: версия ISA поднята выше той, что понимает
        # драйвер. Поэтому отпечаток в манифесте пересчитывается — иначе
        # проверка происхождения остановила бы загрузку раньше, и путь
        # отказа самого JIT остался бы непроверенным. Это и есть замер
        # на нижнюю границу драйвера: отказ обязан НАЗЫВАТЬ причину.
        t = ptx.read_text(encoding="utf-8")
        t = t.replace(".version 8.7", ".version 9.9", 1)
        ptx.write_text(t, encoding="utf-8", newline="\n")
        d = json.loads(man.read_text(encoding="utf-8"))
        d["отпечаток артефакта"] = evspmv._sha(t)
        man.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    else:
        raise SystemExit("неизвестная порча " + how)

    W = torch.sparse_coo_tensor(
        torch.zeros(2, 1, dtype=torch.long), torch.ones(1),
        (4, 4), device="cuda").coalesce()
    try:
        eng = sim.Engine(W, 4, device="cuda", spmv=sim.SPMV_EVENT)
    except evspmv.EventKernelUnavailable as exc:
        print(json.dumps({"итог": "ГРОМКО",
                          "первая строка": str(exc).splitlines()[0][:200]},
                         ensure_ascii=False))
        raise SystemExit(7)
    except Exception as exc:
        print(json.dumps({"итог": "ИНАЯ ОШИБКА",
                          "первая строка": "%%s: %%s" %% (type(exc).__name__,
                                                          exc)},
                         ensure_ascii=False))
        raise SystemExit(8)
    print(json.dumps({"итог": "ТИХО: движок собрался",
                      "дорога": eng._ev.build_info["дорога доставки"]},
                     ensure_ascii=False))
    raise SystemExit(9)
'''


def part_loud():
    """Пять порч артефакта — каждая обязана отказать громко."""
    for how in ("нет-артефакта", "нет-манифеста", "испорчен-артефакт",
                "источник-правили", "чужой-abi", "ptx-из-будущего"):
        got = run_child(SPOIL_CHILD, {"SPOIL": how},
                        tag="spoil_" + how.replace("-", "_"))
        try:
            res = say_json(got)
        except RuntimeError:
            res = {"итог": "ребёнок не ответил",
                   "первая строка": (got.stderr or "")[-200:]}
        rep["встречные замеры"]["порча: " + how] = {
            "код выхода": got.returncode, **res}
        check("порча «%s» -> громкий отказ с причиной, а не тихая "
              "подмена дороги" % how,
              got.returncode == 7,
              res.get("первая строка", "")[:150])


# --- 7: нижняя граница драйвера ---------------------------------------


FLOOR_CHILD = r'''
    import hashlib, json, shutil, sys, os
    from pathlib import Path
    sys.path.insert(0, %(lab)r)
    os.environ["FLYBRAIN_OUT"] = %(out)r
    import numpy as np
    import torch
    torch.zeros(1, device="cuda")
    from flybrain import evspmv

    where = Path(%(out)r) / "floor"
    orig_ptx, orig_man = evspmv.PTX, evspmv.PTX_MANIFEST
    text0 = orig_ptx.read_text(encoding="utf-8")
    have = [ln.split()[1] for ln in text0.splitlines()
            if ln.startswith(".version")][0]

    # Матрица маленькая, но с теми же свойствами, из которых следует
    # побитовое равенство: значения целые из того же диапазона, спайк
    # 0/1. Коннектом здесь не нужен — проверяется ядро, а не модель.
    n, batch, nnz = 4096, 2, 40000
    g = torch.Generator(device="cpu").manual_seed(5)
    rows = torch.randint(0, n, (nnz,), generator=g)
    cols = torch.randint(0, n, (nnz,), generator=g)
    vals = torch.randint(-2405, 1898, (nnz,), generator=g).float()
    W = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals,
                                (n, n)).coalesce().to("cuda").to_sparse_csr()
    sp = (torch.rand(batch, n, device="cuda") < 0.01).float()
    mo = torch.ones(batch, n, device="cuda")
    ref = torch.sparse.mm(W, (sp * mo).t()).t().contiguous()

    def fpr(a):
        return hashlib.sha256(np.ascontiguousarray(
            a.detach().cpu().numpy()).tobytes()).hexdigest()[:16]

    res = {"в артефакте": have, "эталон csrmv": fpr(ref), "по версиям": {}}
    for ver in ("8.7", "7.0", "6.0", "5.0", "4.3", "3.2"):
        w = where / ver.replace(".", "_")
        shutil.rmtree(w, ignore_errors=True)
        w.mkdir(parents=True, exist_ok=True)
        t = text0.replace(".version " + have, ".version " + ver, 1)
        p, m = w / "ev_spmv.ptx", w / "ev_spmv.ptx.json"
        p.write_text(t, encoding="utf-8", newline="\n")
        d = json.loads(orig_man.read_text(encoding="utf-8"))
        d["отпечаток артефакта"] = evspmv._sha(t)
        m.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        evspmv.PTX, evspmv.PTX_MANIFEST = p, m
        evspmv._LIB = None
        try:
            ev = evspmv.Events(W, batch, "cuda", load=evspmv.LOAD_PTX)
            out = ev.mm(sp, mo)
            res["по версиям"][ver] = {
                "принят": True, "отпечаток": fpr(out),
                "побитово с csrmv": bool(torch.equal(out, ref))}
            del ev
        except evspmv.EventKernelUnavailable as exc:
            why = [ln.strip() for ln in str(exc).splitlines()
                   if "fatal" in ln or "CUresult" in ln]
            res["по версиям"][ver] = {"принят": False,
                                      "причина": (why or ["?"])[0][:160]}
        evspmv._LIB = None
    evspmv.PTX, evspmv.PTX_MANIFEST = orig_ptx, orig_man
    print(json.dumps(res, ensure_ascii=False))
'''


def part_floor():
    """Нижняя граница драйвера — единственная настоящая цена PTX-дороги.

    Драйвер обязан понимать `.version` артефакта. nvcc 12.8 пишет 8.7,
    то есть требует драйвер не старше R570, тогда как torch cu128 по
    совместимости младших версий живёт и на более старых. Вопрос замера:
    можно ли опустить границу, не меняя счёта. В счёте только
    ld/st/mul/atom/setp/bra — всё это есть с PTX 2.x, поэтому пониженная
    `.version` ДОЛЖНА приниматься; «должна» здесь проверяется.
    """
    got = run_child(FLOOR_CHILD, tag="floor", timeout=900)
    try:
        res = say_json(got)
    except RuntimeError as exc:
        check("нижняя граница драйвера замерена", False, str(exc)[:200])
        return
    rep["замеры"]["нижняя граница драйвера"] = res
    ok_vers, bad_vers = [], []
    for ver, v in res["по версиям"].items():
        state = ("принят, побитово %s" % v["побитово с csrmv"]
                 if v["принят"] else "ОТКАЗ: " + v["причина"][:70])
        print("     .version %-5s %s" % (ver, state))
        (ok_vers if v["принят"] else bad_vers).append(ver)
    equal = [v for v in res["по версиям"].values()
             if v["принят"] and not v["побитово с csrmv"]]
    check("артефакт принимается с пониженной версией ISA и считает ТО ЖЕ "
          "(граница драйвера опускается без правки счёта)",
          len(ok_vers) > 1 and not equal,
          "приняты %s, все побитово с csrmv" % ", ".join(ok_vers))
    check("слишком низкая версия ISA отвергается ГРОМКО, а не считает "
          "иначе (встречный замер на сам замер)",
          bool(bad_vers),
          "отвергнуты %s" % (", ".join(bad_vers) or "ни одна"))


# --- окружение ---------------------------------------------------------


def build_ctx(device=None):
    import torch

    from flybrain import data as D, evspmv, sim, world

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    neurons = D.load_neurons()
    W = D.load_weights(device=device)
    n = len(neurons)
    sens = np.unique(np.concatenate([
        D.select(neurons, world.SENSORS["odour_left"]),
        D.select(neurons, world.SENSORS["odour_right"]),
        D.select(neurons, world.SENSORS["taste"]),
    ]))
    evspmv.LOAD_DEFAULT = evspmv.LOAD_PTX
    ev = sim.build_engine(weights=W, neurons=neurons, device=device, seed=0,
                          no_refractory=sens, spmv=sim.SPMV_EVENT)
    n_in, n_out = ev.fan()
    busy = torch.argsort(n_in + n_out, descending=True)[:1200]
    # Нейроны для вмешательств берутся по числу связей: маска на нейрон
    # без связей не вмешивается ни во что, и равенство под такой маской
    # следовало бы из её пустоты, а не из замера.
    ctx = {"device": device, "neurons": neurons, "W": W, "n": n,
           "sens": sens, "driven": sens, "buf_len": int(ev.buf_len),
           "связей": int(ev._ev.val.numel()),
           "busy": busy.cpu().numpy().astype(np.int64)}
    # Движок-разведчик дальше не нужен, а памяти держит столько же,
    # сколько плечи замера: четыре плеча рядом и так на пределе.
    del ev, n_in, n_out, busy
    torch.cuda.empty_cache()
    return ctx


def regenerate():
    """Пересобрать артефакт в дереве. Делается здесь, не у пользователя."""
    from flybrain import evspmv

    text, man = evspmv.compile_ptx(verbose=True)
    evspmv.PTX.write_text(text, encoding="utf-8", newline="\n")
    evspmv.PTX_MANIFEST.write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print("PTX %d байт -> %s" % (len(text.encode("utf-8")), evspmv.PTX))
    print("происхождение -> %s" % evspmv.PTX_MANIFEST)
    print(json.dumps(man, ensure_ascii=False, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--пересобрать", dest="regen", action="store_true",
                    help="собрать PTX из ev_spmv.cu и положить в дерево")
    ap.add_argument("--происхождение", dest="origin_only",
                    action="store_true",
                    help="только сверка артефакта с пересобранным")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--skip-bare", action="store_true",
                    help="не поднимать процессы для замера без nvcc/MSVC")
    ap.add_argument("--части", dest="parts", default="все",
                    help="какие разделы считать через запятую: "
                         "происхождение, загрузка, равенство, скорость, "
                         "без-средств, громкость, граница; по умолчанию все")
    args = ap.parse_args()
    ALL = {"происхождение", "загрузка", "равенство", "скорость",
           "без-средств", "громкость", "граница"}
    want = (set(ALL) if args.parts == "все"
            else {s.strip() for s in args.parts.split(",") if s.strip()})
    bad = want - ALL
    if bad:
        ap.error("неизвестные разделы: %s" % ", ".join(sorted(bad)))

    if args.regen:
        return regenerate()

    import torch

    if not torch.cuda.is_available():
        print("событийное умножение считает только на CUDA, а её нет")
        return 2

    if args.origin_only:
        want = {"происхождение"}
    if args.skip_bare:
        want -= {"без-средств", "громкость", "граница"}

    if "происхождение" in want:
        print("\n— 1: происхождение артефакта —", flush=True)
        part_origin()
    if "загрузка" in want:
        print("\n— 2: цена первого запуска —", flush=True)
        part_load()

    if want & {"равенство", "скорость"}:
        ctx = build_ctx()
        print("\nнейронов %d, связей %d, устройство %s"
              % (ctx["n"], ctx["связей"], ctx["device"]))
        if "равенство" in want:
            print("\n— 3: побитовое равенство, шесть сочетаний —",
                  flush=True)
            part_equal(ctx)
        if "скорость" in want:
            print("\n— 4: скорость попарно, перемежающимися кругами —",
                  flush=True)
            part_speed(ctx, rounds=args.rounds, steps=args.steps)

    if "без-средств" in want:
        # Сверять нечему, если раздел равенства не считался: отпечаток
        # «при полном наборе средств» берётся именно оттуда.
        good = rep["замеры"].get("шесть сочетаний дорога×графы", {}).get(
            "PTX + графы")
        if good is None:
            rep["находки"].append(
                "раздел «без-средств» требует раздела «равенство»: "
                "сверять не с чем")
        else:
            print("\n— 5: машина без nvcc и без MSVC —", flush=True)
            part_bare(good)
    if "громкость" in want:
        print("\n— 6: громкость отказов —", flush=True)
        part_loud()
    if "граница" in want:
        print("\n— 7: нижняя граница драйвера —", flush=True)
        part_floor()

    rep["итог"] = {"проверок": len(rep["проверки"]),
                   "провалов": len(fails), "провалившиеся": fails}
    path = OUT / "kernel_ship.json"
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for name in fails:
        print("  ПРОВАЛ", name)
    print("отчёт:", path)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
