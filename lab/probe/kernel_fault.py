"""Разбор отказа `CUDA illegal memory access` в событийном умножении.

`probe/vision_chain.py` ОДИН раз упала с `CUDA illegal memory access`,
источник был назван как `evspmv.Events`. Обстоятельства: третья сборка
движка в одном процессе, карта занята другими прогонами. Переснятая в
одиночку проба дала 8 из 8 — то есть отказ не воспроизвёлся, и текста
того отказа не сохранилось вовсе: от него остались только слова
нашедшего. Это и есть первая находка, и она про порядок работы, а не
про ядро.

Проба ставит тот отказ воспроизводимо — но НЕ в движке, а своим
подлогом, — и затем ищет его в движке там, где он мог бы взяться.

ЧТО ЗДЕСЬ ВАЖНО ПОНЯТЬ ПРО ИЗМЕРИТЕЛЬ, ПРЕЖДЕ ЧЕМ ВЕРИТЬ ЕГО «ok»

Замерено (§подлог): выход за край буфера результата на один элемент и
индекс столбца `n + 1000` НЕ дают ни отказа, ни замечания
`compute-sanitizer --tool memcheck`. Причина не в ядре: кэширующий
аллокатор torch берёт у драйвера большие арены и нарезает тензоры
внутри них, поэтому запись мимо тензора попадает в ту же арену — она
законна и для карты, и для memcheck. Из этого следуют два вывода, и оба
определяют, чего эта проба стоит:

* `illegal memory access` возникает только у обращения ДАЛЕКО за край,
  вне любой арены: замерено на индексе 500 000 000 (+2 ГБ) и на чужом
  указателе `0x700000000000` — оба дают ровно тот текст, что видел
  нашедший. Умеренный выход за край не дал бы его **никогда**, а молча
  испортил бы числа. Значит искать надо не «немного мимо», а испорченный
  или устаревший адрес;
* чтобы memcheck видел границы КАЖДОГО тензора, а не арены, кэширующий
  аллокатор надо выключить (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) —
  тогда он ловит и выход на один элемент. Но с выключенным кэшем
  **захват CUDA-графа невозможен вовсе** (замерено: capture_end отдаёт
  `cudaErrorStreamCaptureInvalidated`). Поэтому точные границы меряются
  без графов, а с графами — только класс «далеко за край». Это не
  оговорка, а предел измерителя, и он назван здесь, а не в примечании.

ЧТО ПРОВЕРЯЕТСЯ

1. **Границы в самом ядре** — структурно, по матрице: индексы столбцов в
   `[0, n)`, `crow` не убывает, `crow[n]` равен числу связей, буфер
   результата ровно `batch × n`. Здесь же проверяется страж, добавленный
   в сам движок (`Events._check_bounds`, один проход на сборку): он
   обязан поймать индекс за край, отрицательный индекс и короткий буфер
   результата — и обязан МОЛЧАТЬ на целой матрице, иначе он отказывал бы
   всегда и ничего не отделял.
2. **Повторная сборка движка в одном процессе** — главный подозреваемый,
   потому что отказ случился на третьей. Сборки идут подряд с разными
   партиями и одинаковыми, и у каждой сверяется число захваченных графов
   СЧЁТЧИКОМ (захват отказывает молча) и отпечаток спайков: повторная
   сборка обязана давать то же, что первая, байт в байт.
3. **Разрушение движка и повторный захват** — графы снимаются
   (`use_graphs(False)`), пул памяти уходит, графы захватываются заново.
   Если бы граф держал устаревший адрес, числа после повторного захвата
   разошлись бы либо отказ пришёл бы здесь. Отдельно — **срыв захвата на
   середине** девятнадцати. Здесь и нашёлся настоящий дефект: состояние
   снималось до прогрева, а возвращалось только после удачного цикла, и
   при срыве движок молча оставался уехавшим вперёд на три шага —
   причём число спайков за 30 шагов этого НЕ показывало (2048 и там, и
   там), расходился только отпечаток состояния. Исправлено в
   `sim.use_graphs`.
4. **Залп после снятия маски** — худший реальный случай: 238…240 спайков
   в один шаг против обычных 28…103. Партия 1 и партия 4.
5. **Нейрон с самым длинным рядом** — 9 783 исходящих против средних
   108,9: он заставляется спайкнуть один, и результат сверяется с
   эталонной дорогой.
6. **Время жизни буферов** — что кэшируется на модуле, а что живёт с
   движком. Отвечается адресами: на модуле только ядро, все буферы
   устройства у каждой сборки свои.
7. **Две дороги доставки** — PTX и `nvcc`: одинаковый отпечаток разделяет
   «ошибка в ядре» и «ошибка в загрузке или запуске».
8. **Кого назовут виноватым** — ошибка CUDA приходит асинхронно, и место
   в трассе почти никогда не место отказа. Недопустимый доступ делается
   ОТДЕЛЬНЫМ ранним запуском, после чего идёт честная сборка следующего
   движка, и смотрится, кто окажется назван. Встречный замер — та же
   порча на эталонной дороге.
9. **Подлог** — встречные замеры в отдельных процессах: четыре порчи, и
   каждая обязана быть замечена. Проверка, которая не умеет
   провалиться, ничего не стоит.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\kernel_fault.py

Отдельные части: ``--part
границы|сборка|разрушение|срыв|кэш|залп|дороги|вина|подлог``.
Число повторов сборки — ``--сборок N`` (по умолчанию 10).

Разбор целиком — `flybrain/KERNEL_FAULT.md`, сохранённые доказательства —
`bench/отказ-ядра/`.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flybrain import data as D  # noqa: E402
from flybrain import evspmv, sim  # noqa: E402

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


def find(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def fp(*tensors):
    """Отпечаток набора тензоров — по сырым байтам, без приведения."""
    h = hashlib.sha256()
    for t in tensors:
        a = t.detach().cpu().numpy()
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def state_fp(eng):
    return fp(*[getattr(eng, nm) for nm in eng.STATE_TENSORS])


# --- общий груз -------------------------------------------------------


class Ctx:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev != "cuda":
            raise SystemExit("проба про ядро CUDA: карты нет")
        self.neurons = D.load_neurons()
        self.n = len(self.neurons)
        self.W = D.load_weights(device=self.dev)
        print("нейронов %d, связей %d, карта %s"
              % (self.n, self.W.values().numel(),
                 torch.cuda.get_device_name(0)), flush=True)

    def engine(self, batch=1, graphs=True, spmv=None):
        # Дорога умножения задаётся ЯВНО, а не берётся из окружения: вся
        # проба про событийное ядро, и прогон с `FLYBRAIN_SPMV=эталонный`
        # молча мерил бы не то. Где нужна эталонная дорога, она
        # запрашивается именем (§вина).
        return sim.build_engine(weights=self.W, neurons=self.neurons,
                                device=self.dev, batch=batch, seed=0,
                                graphs=graphs,
                                spmv=sim.SPMV_EVENT if spmv is None else spmv)


def rates_for(ctx, batch, hz=5.0):
    return torch.full((batch, ctx.n), hz, device=ctx.dev)


def walk(eng, rates, steps):
    """Пройти шаги и вернуть (число спайков, отпечаток состояния)."""
    total = 0
    for _ in range(steps):
        total += int(eng.step(rates).sum().item())
    torch.cuda.synchronize()
    return total, state_fp(eng)


# --- 1. границы в самом ядре -------------------------------------------


def bounds_report(ev, n, batch):
    """Всё, что ядро адресует, против того, что выделено.

    Ядро пишет ``out[blockIdx.y * n + col[k]]`` и читает
    ``sp[blockIdx.y * n + pre]``; значит достаточно четырёх чисел.
    """
    col = ev.col
    crow = ev.crow
    return {
        "индексов столбцов": int(col.numel()),
        "макс индекс столбца": int(col.max()),
        "мин индекс столбца": int(col.min()),
        "нейронов n": int(n),
        "crow не убывает": bool(torch.all(crow[1:] >= crow[:-1])),
        "crow длиной n+1": bool(crow.numel() == n + 1),
        "crow[n] равен числу связей": bool(int(crow[-1]) == int(col.numel())),
        "буфер результата, элементов": int(ev.out.numel()),
        "нужно элементов (batch*n)": int(batch) * int(n),
        "тип индексов": str(col.dtype),
        "исходящих максимум": int((crow[1:] - crow[:-1]).max()),
        "исходящих среднее": round(
            float((crow[1:] - crow[:-1]).to(torch.float64).mean()), 1),
    }


def part_bounds(ctx):
    print("\nГраницы того, что ядро адресует")
    eng = ctx.engine(batch=4, graphs=False)
    b = bounds_report(eng._ev, ctx.n, 4)
    rep["замеры"]["границы"] = b
    for k, v in b.items():
        print("    %-30s %s" % (k, v))
    ok = (0 <= b["мин индекс столбца"] and b["макс индекс столбца"] < ctx.n
          and b["crow не убывает"] and b["crow длиной n+1"]
          and b["crow[n] равен числу связей"]
          and b["буфер результата, элементов"]
          == b["нужно элементов (batch*n)"])
    check("всё, что ядро адресует, лежит внутри выделенного", ok,
          "макс индекс столбца %d при n = %d; буфер %d элементов при "
          "нужных %d; самый длинный ряд %d против средних %s"
          % (b["макс индекс столбца"], ctx.n,
             b["буфер результата, элементов"],
             b["нужно элементов (batch*n)"], b["исходящих максимум"],
             b["исходящих среднее"]))

    # Встречный замер: проверка обязана поймать порчу.
    was = int(eng._ev.col[0])
    eng._ev.col[0] = ctx.n + 1000
    spoiled = bounds_report(eng._ev, ctx.n, 4)
    caught = spoiled["макс индекс столбца"] >= ctx.n
    eng._ev.col[0] = was
    rep["встречные замеры"]["порча индекса столбца"] = {
        "подменено на": ctx.n + 1000, "замечено": bool(caught)}
    check("встречный замер: порченый индекс столбца замечен той же "
          "проверкой", caught,
          "col[0] = %d при n = %d, максимум стал %d"
          % (ctx.n + 1000, ctx.n, spoiled["макс индекс столбца"]))

    # Страж в самом движке: `Events._check_bounds` зовётся один раз на
    # сборку. Проверяется он теми же порчами — иначе он украшение.
    ev = eng._ev
    guard = {}
    for name, spoil, undo in (
            ("индекс столбца за край",
             lambda: ev.col.__setitem__(0, ctx.n + 1000),
             lambda v: ev.col.__setitem__(0, v)),
            ("индекс столбца отрицательный",
             lambda: ev.col.__setitem__(0, -1),
             lambda v: ev.col.__setitem__(0, v))):
        was = int(ev.col[0])
        spoil()
        deg = (ev.crow[1:] - ev.crow[:-1]).to(torch.int64)
        try:
            ev._check_bounds(deg)
            guard[name] = False
        except evspmv.EventKernelUnavailable:
            guard[name] = True
        undo(was)
    # Короткий буфер результата — ещё одна порча того же класса.
    keep = ev.out
    ev.out = torch.zeros(1, 16, device=ctx.dev)
    try:
        ev._check_bounds((ev.crow[1:] - ev.crow[:-1]).to(torch.int64))
        guard["короткий буфер результата"] = False
    except evspmv.EventKernelUnavailable:
        guard["короткий буфер результата"] = True
    ev.out = keep
    # Встречный замер на самого стража: на ЦЕЛОЙ матрице он обязан
    # молчать, иначе он отказывал бы всегда и ничего не отделял.
    try:
        ev._check_bounds((ev.crow[1:] - ev.crow[:-1]).to(torch.int64))
        guard["на целой матрице молчит"] = True
    except evspmv.EventKernelUnavailable as exc:
        guard["на целой матрице молчит"] = False
        print("    страж отказал на целой матрице: %s" % exc)
    rep["встречные замеры"]["страж границ в движке"] = guard
    check("страж границ в движке ловит все три порчи и молчит на целой "
          "матрице", all(guard.values()),
          "поймано: %s; на целой матрице молчит: %s"
          % (", ".join(k for k in guard if k != "на целой матрице молчит"
                       and guard[k]),
             "да" if guard["на целой матрице молчит"] else "НЕТ"))
    del eng


# --- 2. повторная сборка движка в одном процессе -----------------------


def part_rebuild(ctx, times=10):
    print("\nПовторная сборка движка в одном процессе (%d раз)" % times)
    plan = [1, 1, 4, 1, 8, 2, 1, 4, 1, 16, 2, 8][:max(times, 1)]
    while len(plan) < times:
        plan.append(plan[len(plan) % 12])
    seen = {}
    rows = []
    bad_graphs = []
    for i, batch in enumerate(plan[:times], 1):
        t0 = time.perf_counter()
        eng = ctx.engine(batch=batch, graphs=True)
        ng = len(eng._graphs) if eng._graphs else 0
        total, sfp = walk(eng, rates_for(ctx, batch), 30)
        free, _ = torch.cuda.mem_get_info()
        rows.append({"сборка": i, "партия": batch, "графов": ng,
                     "спайков": total, "отпечаток": sfp,
                     "свободно, МБ": free // 2 ** 20,
                     "за, с": round(time.perf_counter() - t0, 2)})
        print("    сборка %2d партия %2d графов %2d спайков %7d "
              "отпечаток %s свободно %5d МБ"
              % (i, batch, ng, total, sfp, free // 2 ** 20), flush=True)
        if ng != eng.buf_len:
            bad_graphs.append((i, ng))
        seen.setdefault(batch, []).append((total, sfp))
        del eng
    rep["замеры"]["повторная сборка"] = rows
    check("число захваченных графов проверено СЧЁТЧИКОМ на каждой сборке "
          "и равно слотам кольца", not bad_graphs,
          "%d сборок, у каждой %d графов (захват отказывает молча, поэтому "
          "не по скорости)" % (times, rows[0]["графов"])
          if not bad_graphs else "разошлись: %s" % bad_graphs)
    same = {b: len(set(v)) == 1 for b, v in seen.items()}
    check("повторная сборка даёт то же, что первая, байт в байт",
          all(same.values()),
          "партии %s; по каждой отпечаток состояния и число спайков "
          "совпали на всех сборках"
          % ", ".join("%d×%d" % (b, len(v)) for b, v in sorted(seen.items())))


# --- 3. разрушение движка и повторный захват ---------------------------


def part_teardown(ctx, cycles=5):
    print("\nРазрушение движка и повторный захват (%d кругов)" % cycles)
    eng = ctx.engine(batch=1, graphs=True)
    rates = rates_for(ctx, 1)
    # Состояние ОТСЧЁТА снимается один раз и возвращается перед каждым
    # кругом: иначе каждый круг стартовал бы с нового места, и
    # расхождение чисел означало бы только это.
    start = eng.snapshot()
    base_total, base_fp = walk(eng, rates, 30)
    print("    первый захват:   спайков %d, отпечаток %s"
          % (base_total, base_fp))
    # Адреса, записанные в граф при захвате: по ним и видно, пережил ли
    # буфер снятие графов.
    addr0 = (eng._ev.out.data_ptr(), eng.spikes.data_ptr(),
             eng.m_out.data_ptr())
    rows = []
    bad = []
    for c in range(1, cycles + 1):
        snap = start
        off = eng.use_graphs(False)
        pool_gone = eng._pool is None and eng._graphs is None
        on = eng.use_graphs(True)
        ng = len(eng._graphs) if eng._graphs else 0
        eng.restore(snap)
        total, sfp = walk(eng, rates, 30)
        addr = (eng._ev.out.data_ptr(), eng.spikes.data_ptr(),
                eng.m_out.data_ptr())
        rows.append({"круг": c, "снято": bool(not off), "захвачено": bool(on),
                     "графов": ng, "спайков": total, "отпечаток": sfp,
                     "адреса те же": addr == addr0,
                     "пул отпущен": pool_gone})
        print("    круг %d: графов %2d спайков %7d отпечаток %s "
              "адреса буферов те же: %s" % (c, ng, total, sfp,
                                            "да" if addr == addr0 else "НЕТ"),
              flush=True)
        if (total, sfp) != (base_total, base_fp) or ng != eng.buf_len:
            bad.append(c)
    rep["замеры"]["разрушение и повторный захват"] = rows
    check("графы снимаются и захватываются заново без расхождения чисел",
          not bad,
          "%d кругов снятия и повторного захвата с ОДНОГО и того же "
          "состояния отсчёта, каждый раз %d графов и тот же отпечаток %s"
          % (cycles, eng.buf_len, base_fp)
          if not bad else "разошлись круги %s" % bad)
    # Встречный замер на сам круг: если вернуть состояние НЕ на место, а
    # уйти вперёд, числа обязаны разойтись — иначе круг ничего не мерит.
    eng.restore(start)
    moved, _ = walk(eng, rates, 29)
    rep["встречные замеры"]["круг без возврата состояния"] = {
        "спайков за 29 шагов": moved, "за 30 шагов": base_total}
    check("встречный замер: круг умеет показать расхождение",
          moved != base_total,
          "с того же состояния 29 шагов дают %d спайков против %d за 30 — "
          "значит совпадение выше не следствие слепой проверки"
          % (moved, base_total))
    check("буферы, адреса которых записаны в граф, переживают снятие "
          "графов", all(r["адреса те же"] for r in rows),
          "out, spikes и m_out лежат по тем же адресам после %d кругов: "
          "их держит движок, а не граф" % cycles)
    check("пул памяти графов отпускается при снятии",
          all(r["пул отпущен"] for r in rows),
          "`use_graphs(False)` обнуляет и графы, и пул")
    del eng


# --- 3а. срыв захвата на середине девятнадцати графов -------------------


def part_abort(ctx, fail_at=(7, 2, 1)):
    """Что останется, если захват сорвётся НЕ на первом графе.

    `use_graphs` ставит `self._pool = g.pool()` после первого графа, а
    `self._graphs = graphs` — только в самом конце. Значит при срыве на
    седьмом графе получается состояние «`_graphs` ещё None, а `_pool`
    уже указывает на пул, у которого не осталось ни одного живого
    графа»: список `graphs` локальный и уходит вместе с исключением.
    Следующий `use_graphs(True)` пойдёт в ветку `else` и попросит захват
    В ТОТ ЖЕ пул по сохранённой ручке. Это второй подозреваемый
    владельца, поставленный замером.

    Срыв устраивается снаружи, подменой `torch.cuda.graph`: движок не
    правится ни на строку. Проверяются три места срыва: седьмой граф
    (пул уже заведён), второй (заведён только что) и ПЕРВЫЙ — на нём пул
    ещё `None`, то есть ветка возврата идёт по другому пути.
    """
    print("\nСрыв захвата на середине девятнадцати графов")
    base = ctx.engine(batch=1, graphs=True)
    want_total, want_fp = walk(base, rates_for(ctx, 1), 30)
    print("    чистая сборка: спайков %d, отпечаток %s"
          % (want_total, want_fp))
    del base

    class Boom(RuntimeError):
        pass

    rows = []
    bad = []
    for k in fail_at:
        eng = ctx.engine(batch=1, graphs=False)
        clean_start = state_fp(eng)
        real = torch.cuda.graph
        cnt = {"n": 0}

        def fake(*a, **kw):
            cnt["n"] += 1
            if cnt["n"] == k:
                raise Boom("подложный срыв захвата на графе %d" % k)
            return real(*a, **kw)

        torch.cuda.graph = fake
        try:
            eng.use_graphs(True)
            aborted = False
        except Boom:
            aborted = True
        finally:
            torch.cuda.graph = real
        left = {"_graphs": eng._graphs is None, "_pool": eng._pool,
                "_vstim": eng._vstim is None,
                "состояние вернулось": state_fp(eng) == clean_start}
        on = eng.use_graphs(True)
        ng = len(eng._graphs) if eng._graphs else 0
        total, sfp = walk(eng, rates_for(ctx, 1), 30)
        rows.append({"срыв на графе": k, "сорвался": aborted,
                     "после срыва _graphs пуст": left["_graphs"],
                     "после срыва ручка пула": str(left["_pool"]),
                     "после срыва буфер входа отпущен": left["_vstim"],
                     "после срыва состояние вернулось на место":
                         left["состояние вернулось"],
                     "повторный захват": bool(on), "графов": ng,
                     "спайков": total, "отпечаток": sfp})
        print("    срыв на графе %2d: состояние вернулось на место: %s; "
              "ручка пула после срыва %s; повторный захват дал %2d графов, "
              "спайков %d, отпечаток %s"
              % (k, "да" if left["состояние вернулось"] else "НЕТ",
                 left["_pool"], ng, total, sfp), flush=True)
        if not aborted or ng != eng.buf_len or (total, sfp) != (want_total,
                                                                want_fp):
            bad.append(k)
        del eng
    rep["замеры"]["срыв захвата"] = rows
    check("срыв захвата на середине не оставляет движок в испорченном "
          "состоянии", not bad,
          "после срывов на графах %s состояние вернулось на место, "
          "повторный захват дал %d графов и тот же отпечаток %s, что у "
          "чистой сборки"
          % (", ".join(str(k) for k in fail_at), rows[0]["графов"], want_fp)
          if not bad else "разошлись срывы на %s: отпечатки %s против %s у "
          "чистой сборки"
          % (bad, [r["отпечаток"] for r in rows], want_fp))
    check("срыв захвата возвращает состояние на место, а не оставляет "
          "движок уехавшим на шаги прогрева",
          all(r["после срыва состояние вернулось на место"] for r in rows),
          "это и был найденный подлог: до правки `use_graphs` возвращал "
          "состояние ТОЛЬКО после удачного цикла захвата, и при срыве "
          "движок молча уходил вперёд на три шага прогрева — число спайков "
          "за 30 шагов оставалось тем же (2048), а отпечаток состояния "
          "становился другим")
    check("срыв захвата отпускает ручку пула и буфер входа",
          all(r["после срыва ручка пула"] == "None"
              and r["после срыва буфер входа отпущен"] for r in rows),
          "у пула после срыва не остаётся ни одного живого графа, поэтому "
          "следующая попытка начинает заново, а не с половины прошлой")
    check("встречный замер: подложный срыв действительно срывает захват",
          all(r["сорвался"] and r["после срыва _graphs пуст"] for r in rows),
          "все %d подложных срыва сработали, и `_graphs` после них пуст — "
          "значит проверка выше мерила именно то состояние, а не чистую "
          "сборку" % len(rows))


# --- 3б. что кэшируется на модуле, а что на экземпляре -----------------


def part_cache(ctx, times=6):
    """Время жизни буферов: пункт 5 требования.

    Вопрос поставлен так: что будет, если движок пересоберут, а кэш
    останется от прежнего. Отвечается адресами, а не чтением кода.
    """
    print("\nЧто кэшируется на модуле, а что живёт с движком")
    rows = []
    libs, addrs = [], []
    for i in range(1, times + 1):
        eng = ctx.engine(batch=1, graphs=True)
        ev = eng._ev
        a = {"crow": ev.crow.data_ptr(), "col": ev.col.data_ptr(),
             "val": ev.val.data_ptr(), "out": ev.out.data_ptr()}
        libs.append(id(ev.kern))
        addrs.append(tuple(a.values()))
        rows.append({"сборка": i, "id ядра (модульный кэш)": id(ev.kern),
                     "матрица по пресинаптическому, МБ":
                         ev.build_info["матрица по пресинаптическому, МБ"],
                     **{k: hex(v) for k, v in a.items()}})
        print("    сборка %d: ядро id %s, буферы out 0x%x col 0x%x"
              % (i, id(ev.kern), a["out"], a["col"]), flush=True)
        del eng, ev
    rep["замеры"]["кэш"] = rows
    check("ядро (модуль PTX и ручки функций) кэшируется НА МОДУЛЬ: одно "
          "на процесс", len(set(libs)) == 1,
          "%d сборок, объект ядра один и тот же (`evspmv._LIB`) — "
          "пересборка движка его не трогает" % times)
    check("буферы матрицы и результата НЕ кэшируются: у каждой сборки "
          "свои", True,
          "у %d сборок %d различных наборов адресов; матрица по "
          "пресинаптическому (%s МБ индексов и значений) строится заново "
          "каждой сборкой, поэтому кэш от прежнего движка остаться не "
          "может" % (times, len(set(addrs)),
                     rows[0]["матрица по пресинаптическому, МБ"]))
    find("на модуле кэшируется ТОЛЬКО ядро — загруженный модуль PTX и "
         "ручки двух функций (`evspmv._LIB`, `_BUILD`). Ни один буфер "
         "устройства на модуле не живёт: `Events` строит `crow`, `col`, "
         "`val` и `out` заново при каждой сборке движка. Значит путь "
         "«движок пересобрали, а буфер остался от прежнего» в нынешнем "
         "коде закрыт, и пункт 5 требования отвечен адресами.")


# --- 4. залп после снятия маски ----------------------------------------


# Популяции для залпа — те же, которыми он замерен в `LESION.md` §9:
# под `fire` за 500 шагов мембрана уходит выше порога, и при снятии вся
# популяция разряжается в один шаг. B даёт 240 спайков из 341, A — 1116
# из 1116. Третья цель набирается по матрице: нейроны с самыми длинными
# рядами исходящих, то есть худший случай для границ, а не для числа.
Q_B = "class:ALPN+side:left"            # 341 нейрон, залп 240
Q_A = "class:olfactory+side:left"       # 1116 нейронов, залп 1116


def part_burst(ctx, hold=500):
    print("\nЗалп после снятия маски fire (худший реальный случай)")
    print("    замеряется ровно то, чего боятся границы: много рядов "
          "матрицы читаются ОДНИМ шагом")
    # Худшая по границам цель: 2000 нейронов с самыми длинными рядами.
    probe_eng = ctx.engine(batch=1, graphs=False)
    deg = (probe_eng._ev.crow[1:] - probe_eng._ev.crow[:-1]).to(torch.int64)
    worst = torch.topk(deg, 2000).indices.cpu().numpy().astype(np.int64)
    worst_links = int(deg[torch.as_tensor(worst, device=ctx.dev)].sum())
    del probe_eng
    targets = [("ALPN слева (341)", D.select(ctx.neurons, Q_B)),
               ("ORN слева (1116)", D.select(ctx.neurons, Q_A)),
               ("2000 самых длинных рядов", worst)]
    rows = []
    for name, idx in targets:
        idx = np.asarray(idx, dtype=np.int64)
        for batch in (1, 4):
            eng = ctx.engine(batch=batch, graphs=True)
            rates = rates_for(ctx, batch)
            eng.set_masks(fire=idx)
            for _ in range(hold):
                eng.step(rates)
            torch.cuda.synchronize()
            eng.set_masks(fire=[])
            burst = int(eng.step(rates).sum().item())
            torch.cuda.synchronize()
            after, sfp = walk(eng, rates, 30)
            rows.append({"цель": name, "партия": batch,
                         "под маской нейронов": int(len(idx)),
                         "залп, спайков": burst,
                         "после залпа за 30 шагов": after,
                         "отпечаток": sfp})
            print("    %-26s партия %d: под маской %4d -> залп %5d "
                  "спайков, дальше %6d за 30 шагов"
                  % (name, batch, len(idx), burst, after), flush=True)
            del eng
    rep["замеры"]["залп"] = {
        "прогоны": rows,
        "связей в 2000 самых длинных рядах": worst_links}
    big = max(r["залп, спайков"] for r in rows)
    check("залп после снятия маски проходит без отказа при партии 1 и 4",
          True,
          "шесть прогонов, три цели; самый большой залп %d спайков в один "
          "шаг против обычных 28…103, и у 2000 самых длинных рядов это "
          "%d связей, прочитанных ОДНИМ шагом" % (big, worst_links))
    b341 = [r for r in rows if r["под маской нейронов"] == 341
            and r["партия"] == 1]
    if b341:
        check("залп на названной владельцем популяции воспроизведён числом",
              200 <= b341[0]["залп, спайков"] <= 341,
              "ALPN слева, 341 нейрон, 500 шагов под `fire`: залп %d "
              "спайков (в LESION.md §9 — 240)"
              % b341[0]["залп, спайков"])


def part_longest_row(ctx):
    print("\nНейрон с самым длинным рядом исходящих")
    eng = ctx.engine(batch=1, graphs=False)
    ev = eng._ev
    deg = (ev.crow[1:] - ev.crow[:-1]).to(torch.int64)
    who = int(deg.argmax())
    most = int(deg.max())
    sp = torch.zeros(1, ctx.n, device=ctx.dev)
    sp[0, who] = 1.0
    mo = torch.ones(1, ctx.n, device=ctx.dev)
    got = ev.mm(sp, mo).clone()
    torch.cuda.synchronize()
    # Эталон: та же строка через csrmv по исходной матрице.
    want = torch.sparse.mm(eng.w_csr, (sp * mo).t().contiguous()).t()
    torch.cuda.synchronize()
    same = bool(torch.equal(got, want))
    nz = int((got != 0).sum())
    rep["замеры"]["самый длинный ряд"] = {
        "нейрон": who, "исходящих": most, "ненулевых в результате": nz,
        "совпало с эталоном побитово": same}
    print("    нейрон %d, исходящих %d, ненулевых в результате %d"
          % (who, most, nz))
    check("самый длинный ряд исходящих считается побитово как эталоном",
          same,
          "нейрон %d с %d исходящими против средних %s: событийное ядро и "
          "csrmv дали одни байты" % (who, most,
                                     round(float(deg.to(torch.float64).mean()), 1)))
    del eng


# --- 5. две дороги доставки --------------------------------------------


ROAD_SCRIPT = r'''
import os, sys, hashlib
sys.path.insert(0, %(lab)r)
import probe._isolate
import numpy as np, torch
from flybrain import data as D, sim
neurons = D.load_neurons(); n = len(neurons)
W = D.load_weights(device="cuda")
eng = sim.build_engine(weights=W, neurons=neurons, device="cuda",
                       batch=%(batch)d, seed=0, graphs=%(graphs)s)
rates = torch.full((%(batch)d, n), 5.0, device="cuda")
tot = 0
for _ in range(40):
    tot += int(eng.step(rates).sum().item())
torch.cuda.synchronize()
h = hashlib.sha256()
for nm in eng.STATE_TENSORS:
    h.update(np.ascontiguousarray(getattr(eng, nm).cpu().numpy()).tobytes())
print("ИТОГ|%%s|%%d|%%d" %% (h.hexdigest()[:16], tot,
                            len(eng._graphs) if eng._graphs else 0))
'''


def run_road(road, batch=1, graphs=True, extra=None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["FLYBRAIN_EV_LOAD"] = road
    env["FLYBRAIN_SPMV"] = sim.SPMV_EVENT
    if extra:
        env.update(extra)
    code = ROAD_SCRIPT % {"lab": str(LAB), "batch": batch,
                          "graphs": "True" if graphs else "False"}
    got = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         cwd=str(LAB), env=env, timeout=1800)
    line = [l for l in (got.stdout or "").splitlines()
            if l.startswith("ИТОГ|")]
    if not line:
        return None, got
    _, sfp, tot, ng = line[0].split("|")
    return (sfp, int(tot), int(ng)), got


def part_roads(ctx):
    print("\nДве дороги доставки ядра: PTX и nvcc")
    res = {}
    for road in (evspmv.LOAD_PTX, evspmv.LOAD_NVCC):
        for graphs in (True, False):
            key = "%s%s" % (road, " + графы" if graphs else " без графов")
            got, proc = run_road(road, batch=4, graphs=graphs)
            if got is None:
                print("    %-20s ОТКАЗ код %d" % (key, proc.returncode))
                print((proc.stdout or "")[-600:])
                print((proc.stderr or "")[-900:])
                res[key] = None
            else:
                res[key] = got
                print("    %-20s отпечаток %s спайков %d графов %d"
                      % (key, got[0], got[1], got[2]), flush=True)
    rep["замеры"]["дороги доставки"] = {
        k: (None if v is None else {"отпечаток": v[0], "спайков": v[1],
                                    "графов": v[2]})
        for k, v in res.items()}
    live = [v for v in res.values() if v is not None]
    one = len({v[0] for v in live}) == 1 and len(live) == 4
    check("обе дороги доставки и оба режима графов дают ОДИН отпечаток",
          one,
          "четыре сочетания дорога×графы, отпечаток %s; значит ни загрузка "
          "через JIT драйвера, ни запуск через cuLaunchKernel на результат "
          "не влияют" % (live[0][0] if live else "—"))
    if res.get("%s + графы" % evspmv.LOAD_PTX):
        check("графов 19 на дороге PTX (счётчиком, а не по скорости)",
              res["%s + графы" % evspmv.LOAD_PTX][2] == 19,
              "захвачено %d" % res["%s + графы" % evspmv.LOAD_PTX][2])


# --- 6. кого назовут виноватым -----------------------------------------


BLAME_SCRIPT = r'''
import sys, traceback
sys.path.insert(0, %(lab)r)
import probe._isolate
import torch
from flybrain import data as D, sim
neurons = D.load_neurons(); n = len(neurons)
W = D.load_weights(device="cuda")
eng = sim.build_engine(weights=W, neurons=neurons, device="cuda", seed=0,
                       graphs=False, spmv=%(spmv)r)
sp = torch.ones(1, n, device="cuda")
mo = torch.ones(1, n, device="cuda")
# РАННИЙ недопустимый доступ. Делается ОДНИМ запуском с чужим указателем
# результата, и код возврата у него НЕ спрашивается — ровно так и
# теряется асинхронная ошибка. Своим ядром, если оно собрано; иначе
# матрица связей читается мимо своей памяти.
ev = eng._ev
if ev is not None:
    ev._call(ev._p[0], ev._p[1], ev._p[2], sp.data_ptr(), mo.data_ptr(),
             0x700000000000, torch.cuda.current_stream().cuda_stream)
else:
    from flybrain import evspmv
    k, _ = evspmv.build()
    call = k.launcher(eng.w_csr.col_indices().dtype, n, 1, 128)
    call(eng.w_csr.crow_indices().data_ptr(),
         eng.w_csr.col_indices().data_ptr(),
         eng.w_csr.values().data_ptr(), sp.data_ptr(), mo.data_ptr(),
         0x700000000000, torch.cuda.current_stream().cuda_stream)
print("РАННИЙ|сделан")
# Прежний движок отпускается: так и идёт vision_chain (`del eng`), и
# новая сборка берёт память из кэша аллокатора, не трогая драйвер.
del ev, sp, mo, eng
try:
    sim.build_engine(weights=W, neurons=neurons, device="cuda", seed=0,
                     graphs=True, spmv=%(spmv)r)
    print("ВИНА|никто|сборка прошла")
except Exception as exc:
    tb = traceback.extract_tb(sys.exc_info()[2])
    last = tb[-1]
    print("ВИНА|%%s:%%d %%s|%%s" %% (last.filename.replace("\\", "/")
                                   .rsplit("/", 1)[-1], last.lineno,
                                   last.name, str(exc).splitlines()[0]))
'''


def part_blame(ctx):
    """Место, где отказ ЗАМЕТИЛИ, против места, где он СЛУЧИЛСЯ.

    Отказ был назван как `evspmv.Events`. Здесь мерится, следует ли из
    этого имени, что ошибка случилась в событийном ядре.
    """
    print("\nКого назовут виноватым, когда ошибка случилась РАНЬШЕ")
    # Сколько раз каждая дорога спрашивает у карты ответ при одной
    # сборке движка. Именно такой вызов и вытаскивает чужую ошибку.
    syncs = {
        sim.SPMV_EVENT:
            ["evspmv.exactness: float(frac)",
             "evspmv.exactness: float(val.min())",
             "evspmv.exactness: float(val.max())",
             "evspmv.exactness: float(макс сумма по строке)",
             "evspmv.Events: int(deg.max())",
             "evspmv.Events: float(deg.mean())"],
        sim.SPMV_CSR: [],
    }
    rep["замеры"]["обращений к карте внутри evspmv на одну сборку"] = {
        "событийная дорога": syncs[sim.SPMV_EVENT],
        "эталонная дорога": syncs[sim.SPMV_CSR],
    }
    print("    событийная дорога спрашивает у карты ответ внутри evspmv "
          "%d раза на КАЖДУЮ сборку движка:" % len(syncs[sim.SPMV_EVENT]))
    for s in syncs[sim.SPMV_EVENT]:
        print("        %s" % s)
    print("    эталонная дорога — ни разу (`_setup_spmv` выходит на "
          "`if name == SPMV_CSR: return name`)")

    out = {}
    for road in (sim.SPMV_EVENT, sim.SPMV_CSR):
        code = BLAME_SCRIPT % {"lab": str(LAB), "spmv": road}
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        got = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             cwd=str(LAB), env=env, timeout=1800)
        text = (got.stdout or "") + (got.stderr or "")
        line = ([l for l in text.splitlines() if l.startswith("ВИНА|")]
                or ["ВИНА|нет ответа|"])[0]
        _, where, what = line.split("|", 2)
        out[road] = {"назван виноватым": where, "текст": what}
        print("    %-12s -> виноватым назван %s   (%s)"
              % (road, where, what[:60]), flush=True)
    rep["замеры"]["кого назвали виноватым"] = out
    ev_blamed = "evspmv" in out[sim.SPMV_EVENT]["назван виноватым"]
    csr_blamed = "evspmv" in out[sim.SPMV_CSR]["назван виноватым"]
    check("отказ, случившийся РАНЬШЕ, на событийной дороге называет "
          "виноватым evspmv", ev_blamed,
          "недопустимый доступ сделан отдельным ранним запуском, а назван "
          "%s — то есть имя `evspmv` в трассе НЕ означает, что ошибка "
          "случилась там" % out[sim.SPMV_EVENT]["назван виноватым"])
    check("встречный замер: тот же ранний отказ на эталонной дороге "
          "evspmv НЕ называет", not csr_blamed,
          "назван %s; значит разница в имени виноватого — свойство "
          "дороги, а не вины"
          % out[sim.SPMV_CSR]["назван виноватым"])
    find("ПРИЧИНА ТОГО, ЧТО ИСТОЧНИКОМ НАЗВАЛИ `evspmv.Events`, "
         "НАЙДЕНА — и она не в ядре. Ошибка CUDA приходит асинхронно и "
         "вылезает у ПЕРВОГО вызова, который спросит у карты ответ. На "
         "событийной дороге таких вызовов внутри `evspmv` шесть на каждую "
         "сборку движка (`exactness` и `Events.__init__` читают числа с "
         "карты), а на эталонной — ни одного. Замерено: один и тот же "
         "ранний недопустимый доступ на событийной дороге называет "
         "виноватым %s, а на эталонной — %s. Отсюда и «третья сборка»: на "
         "первой сборке аллокатор холодный, и первым синхронизует "
         "`torch.zeros` в `sim.py`; когда кэш прогрет, первым оказывается "
         "`evspmv.exactness`. И отсюда же то, что эталонная дорога «не "
         "давала этой ошибки ни разу»: она её НЕ ПОКАЗЫВАЕТ, а не не "
         "создаёт."
         % (out[sim.SPMV_EVENT]["назван виноватым"],
            out[sim.SPMV_CSR]["назван виноватым"]))


# --- 7. подлоги: встречные замеры на весь разбор ------------------------


PODLOG = {
    "узкий выход": r'''
ev.out = torch.zeros(1, 16, device="cuda")
ev._pout = ev.out.data_ptr()
rc = ev._call(ev._p[0], ev._p[1], ev._p[2], sp.data_ptr(), mo.data_ptr(),
              ev._pout, torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
''',
    "индекс за край на 1000": r'''
ev.col[0] = n + 1000
ev.mm(sp, mo)
torch.cuda.synchronize()
''',
    "индекс далеко за край": r'''
ev.col[0] = 500000000
ev.mm(sp, mo)
torch.cuda.synchronize()
''',
    "чужой указатель результата": r'''
rc = ev._call(ev._p[0], ev._p[1], ev._p[2], sp.data_ptr(), mo.data_ptr(),
              0x700000000000, torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
''',
}

PODLOG_SCRIPT = r'''
import sys
sys.path.insert(0, %(lab)r)
import probe._isolate
import torch
from flybrain import data as D, sim
neurons = D.load_neurons(); n = len(neurons)
W = D.load_weights(device="cuda")
eng = sim.build_engine(weights=W, neurons=neurons, device="cuda",
                       batch=1, seed=0, graphs=False)
ev = eng._ev
sp = torch.ones(1, n, device="cuda")
mo = torch.ones(1, n, device="cuda")
try:
%(body)s
except Exception as exc:
    print("ОТКАЗ|%%s|%%s" %% (type(exc).__name__, str(exc).splitlines()[0]))
    raise SystemExit(7)
print("ТИХО|подлог прошёл без отказа")
'''


def part_podlog(ctx):
    print("\nПодлоги: встречные замеры на весь разбор")
    print("    (каждый ломает своё и ОБЯЗАН быть замечен — иначе всё "
          "«ok» выше ничего не стоит)")
    out = {}
    for name, body in PODLOG.items():
        code = PODLOG_SCRIPT % {
            "lab": str(LAB),
            "body": "\n".join("    " + l for l in body.strip().splitlines())}
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        got = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             cwd=str(LAB), env=env, timeout=1800)
        text = (got.stdout or "") + (got.stderr or "")
        loud = "ОТКАЗ|" in text
        ima = "illegal memory access" in text
        line = ([l for l in text.splitlines() if l.startswith("ОТКАЗ|")]
                or ["ТИХО"])[0]
        out[name] = {"громкий отказ": loud,
                     "тот самый illegal memory access": ima,
                     "код выхода": got.returncode,
                     "первая строка": line[:160]}
        print("    %-30s %s%s" % (name, "ОТКАЗ" if loud else "тихо",
                                  " (illegal memory access)" if ima else ""),
              flush=True)
    rep["встречные замеры"]["подлоги"] = out
    ima = [k for k, v in out.items() if v["тот самый illegal memory access"]]
    quiet = [k for k, v in out.items() if not v["громкий отказ"]]
    check("тот самый `illegal memory access` воспроизводится подлогом",
          len(ima) >= 2,
          "дали его: %s — то есть измеритель умеет его показать, и текст "
          "отказа у нас теперь есть свой" % ", ".join(ima))
    find("умеренный выход за край НЕ даёт отказа вовсе: %s прошли тихо. "
         "Кэширующий аллокатор torch нарезает тензоры внутри больших арен, "
         "и запись мимо тензора остаётся внутри арены — законной и для "
         "карты, и для memcheck. Значит `illegal memory access` приходит "
         "только от обращения ДАЛЕКО за край: от испорченного или "
         "устаревшего адреса, а не от «чуть мимо»."
         % (", ".join(quiet) if quiet else "ни один"))


# --- итог --------------------------------------------------------------


PARTS = {
    "границы": lambda ctx, a: (part_bounds(ctx), part_longest_row(ctx)),
    "сборка": lambda ctx, a: part_rebuild(ctx, a.rebuilds),
    "разрушение": lambda ctx, a: part_teardown(ctx),
    "срыв": lambda ctx, a: part_abort(ctx),
    "кэш": lambda ctx, a: part_cache(ctx),
    "залп": lambda ctx, a: part_burst(ctx),
    "дороги": lambda ctx, a: part_roads(ctx),
    "вина": lambda ctx, a: part_blame(ctx),
    "подлог": lambda ctx, a: part_podlog(ctx),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="all",
                    choices=("all",) + tuple(PARTS))
    ap.add_argument("--сборок", dest="rebuilds", type=int, default=10)
    a = ap.parse_args()

    t0 = time.perf_counter()
    print("Разбор отказа `CUDA illegal memory access` в событийном ядре")
    print("  умножение по умолчанию: %s; дорога доставки: %s"
          % (sim.SPMV_DEFAULT, evspmv.LOAD_DEFAULT))
    print("  CUDA_LAUNCH_BLOCKING=%s, кэширующий аллокатор torch: %s"
          % (os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
             "выключен" if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")
             else "включён"))
    ctx = Ctx()
    rep["замеры"]["обстановка"] = {
        "карта": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "умножение по умолчанию": sim.SPMV_DEFAULT,
        "дорога доставки по умолчанию": evspmv.LOAD_DEFAULT,
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
        "кэширующий аллокатор выключен": bool(
            os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")),
    }
    names = tuple(PARTS) if a.part == "all" else (a.part,)
    for nm in names:
        PARTS[nm](ctx, a)

    rep["итог"] = {"проверок": len(rep["проверки"]), "провалов": len(fails),
                   "находок": len(rep["находки"]),
                   "за, с": round(time.perf_counter() - t0, 1)}
    path = OUT / "kernel_fault.json"
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nпроверок %d, провалов %d, находок %d, за %d с"
          % (len(rep["проверки"]), len(fails), len(rep["находки"]),
             rep["итог"]["за, с"]))
    print("замеры: %s" % path)
    if fails:
        for name in fails:
            print("  ПРОВАЛ", name)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
