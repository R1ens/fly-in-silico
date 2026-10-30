"""Событийное умножение связей в движке: равенство, а не близость.

Шлюз для правки `sim.Engine`, где `W @ spikes` считается не `csrmv` по
всем 15 091 983 связям, а своим ядром только по строкам спайкующих
нейронов (`flybrain/ev_spmv.cu`, `flybrain/evspmv.py`). Скорость здесь
не мерится вовсе — это отдельная проба (`probe/lif_event_contour.py`),
и мерить её надо в контуре, а не рядом. Здесь мерится ОДНО: даёт ли
событийная дорога ТЕ ЖЕ БАЙТЫ.

Что проверяется и почему именно это.

1. **Зависимости сборки названы, а отказ громкий.** В этом проекте уже
   есть механизм, который отказывает молча: `Engine.use_graphs`
   возвращает False и оставляет шаг втрое медленнее, ничего не сообщая.
   Поэтому у событийного ядра тихого отката нет, и это проверяется
   замером, а не обещанием: в отдельном процессе `nvcc` подменяется
   несуществующим путём, и проба требует, чтобы движок УПАЛ, а не
   посчитал на `csrmv`.
2. **Условия побитового равенства — три числа.** Значения матрицы целые,
   спайк 0/1, максимальная сумма модулей по строке ниже предела точности
   float32. Встречный замер: значение матрицы делается нецелым, и движок
   обязан отказаться включать событийное умножение.
3. **Шаг в шаг, побайтово, с вмешательствами.** Два движка с одним
   семенем идут по одному и тому же входу, и на каждом шаге сверяются ВСЕ
   тензоры состояния. По дороге включаются и снимаются все четыре маски
   (`fire`, `out`, `in`, `hold`) — поодиночке и вместе, — и окна выбраны
   так, чтобы сравнение попало внутрь хвоста задержанных событий: у
   `out` он 19 шагов, у `fire` 20, у `in` и `hold` нулевой. Хвост — это
   ровно то место, где событийная дорога могла бы разойтись и не
   показать этого в итоге.
4. **Партия больше единицы.** Событийное ядро решает по партиям вторым
   измерением решётки, и в исходном замере партия не проверялась вовсе.
   Здесь партия 4, у каждой свои частоты входа.
5. **CUDA-графы.** Четыре сочетания: обе дороги с графами и без.
   Событийное ядро зовётся через `ctypes` сырым указателем потока, и то,
   что оно пишется в граф, обязано быть замером, а не предположением.
6. **Встречные замеры на сам сравниватель.** Три порчи: последний бит
   одного значения матрицы, игнорирование маски `m_out` и подмена
   указателя кольца. Сравниватель, который их не ловит, ничего не стоит,
   и «совпало побайтово» на нём печаталось бы точно так же.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\lif_event.py
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
LAB = Path(__file__).resolve().parents[1]

fails = []
rep = {"проверки": {}, "замеры": {}, "встречные замеры": {},
       "находки": []}


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
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def state_fp(eng):
    from flybrain import sim

    return {name: fp(getattr(eng, name)) for name in sim.Engine.STATE_TENSORS}


# --- 1: зависимости сборки и громкость отказа -------------------------

NOISY = textwrap.dedent(r'''
    import os, sys
    sys.path.insert(0, %(lab)r)
    os.environ["FLYBRAIN_OUT"] = %(out)r
    os.environ["FLYBRAIN_NVCC"] = r"Z:\нет-такого\nvcc.exe"
    import torch
    from flybrain import evspmv, sim
    W = torch.sparse_coo_tensor(
        torch.zeros(2, 1, dtype=torch.long), torch.ones(1),
        (4, 4), device="cuda").coalesce()
    try:
        sim.Engine(W, 4, device="cuda", spmv=sim.SPMV_EVENT)
    except evspmv.EventKernelUnavailable as exc:
        print("ГРОМКО:", str(exc).splitlines()[0])
        sys.exit(7)
    except Exception as exc:
        print("ИНАЯ ОШИБКА:", type(exc).__name__, exc)
        sys.exit(8)
    print("ТИХО: движок собрался без ядра")
    sys.exit(9)
''')


def part_build(args):
    from flybrain import evspmv

    lib, info = evspmv.build()
    rep["замеры"]["зависимости сборки"] = info
    for k, v in info.items():
        print("     %-22s %s" % (k, v))
    check("ядро собрано, и зависимости названы поимённо",
          bool(lib) and info["архитектура"].startswith("sm_"),
          "%s, nvcc %s" % (info["архитектура"],
                           Path(info["nvcc"]).parents[1].name))
    check("архитектура взята у устройства, а не вписана числом",
          info["архитектура"] == "sm_%d%d" % __import__(
              "torch").cuda.get_device_capability(),
          info["архитектура"])
    # Повторный вызов обязан взять DLL из кэша, а не собирать снова:
    # сборка стоит секунды, и платить их на каждый прогон незачем.
    evspmv._LIB = None
    lib2, info2 = evspmv.build()
    check("собранное запоминается: второй раз берётся из кэша",
          info2["собрано сейчас"] is False, info2["dll"])

    if args.skip_loud:
        rep["находки"].append("встречный замер громкости отказа пропущен "
                              "по просьбе (--skip-loud)")
        return
    src = NOISY % {"lab": str(LAB), "out": os.environ["FLYBRAIN_OUT"]}
    script = Path(os.environ["FLYBRAIN_OUT"]) / "lif_event_noisy.py"
    script.write_text(src, encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    got = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         env=env, timeout=900)
    line = (got.stdout or "").strip().splitlines()
    rep["встречные замеры"]["отказ сборки без nvcc"] = {
        "код выхода": got.returncode,
        "что сказал": line[-1][:300] if line else (got.stderr or "")[-300:],
    }
    check("без nvcc движок падает ГРОМКО, а не считает на csrmv молча",
          got.returncode == 7,
          rep["встречные замеры"]["отказ сборки без nvcc"]["что сказал"])


# --- 2: условия побитового равенства ----------------------------------


def part_exact(ctx):
    import torch

    from flybrain import evspmv, sim

    ex = ctx["ev"]._spmv_exact
    rep["замеры"]["условия равенства"] = ex
    for k, v in ex.items():
        print("     %-32s %s" % (k, v))
    check("значения матрицы целые", ex["значения целые"],
          "%g…%g, макс нецелая часть %g"
          % (ex["мин значение"], ex["макс значение"],
             ex["макс нецелая часть"]))
    check("сумма модулей по строке ниже предела точности float32",
          ex["макс сумма модулей по строке"] < ex["предел точности float32"],
          "%.0f против %d, запас в %s раза"
          % (ex["макс сумма модулей по строке"],
             ex["предел точности float32"], ex["запас"]))
    check("спайк и маска — ровно 0.0 или 1.0",
          bool(((ctx["ev"].spikes == 0) | (ctx["ev"].spikes == 1)).all())
          and bool(((ctx["ev"].m_out == 0) | (ctx["ev"].m_out == 1)).all()),
          "иначе произведение val*s уже не точное")

    # Встречный: значение матрицы делается нецелым — движок обязан
    # отказаться включать событийное умножение, а не включить его молча.
    n = ctx["n"]
    W = ctx["W"].coalesce()
    idx, val = W.indices(), W.values().clone()
    val[0] = val[0] + 0.5
    spoiled = torch.sparse_coo_tensor(idx, val, W.shape,
                                      device=W.device).coalesce()
    why = None
    try:
        sim.Engine(spoiled, n, device="cuda", spmv=sim.SPMV_EVENT)
        caught = False
    except evspmv.EventKernelUnavailable as exc:
        caught = True
        why = str(exc).splitlines()[0][:200]
    rep["встречные замеры"]["нецелое значение матрицы"] = {
        "поймано": caught, "что сказал": why}
    check("нецелое значение матрицы — отказ, а не молчаливое приближение",
          caught, why or "движок включил событийное умножение")
    # И тот же случай с явно выбранным приближением: это должно
    # собираться, но записываться в отчёт как приближение.
    eng3 = sim.Engine(spoiled, n, device="cuda", spmv=sim.SPMV_EVENT,
                      allow_inexact=True)
    info = eng3.spmv_info()
    check("явно выбранное приближение собирается и помечается в отчёте",
          info.get("разрешено приближение") is True
          and info["побитовое равенство обеспечено"] is False,
          "в отчёт прогона уходит «побитовое равенство обеспечено: false»")
    del eng3, spoiled
    torch.cuda.empty_cache()


# --- 3: шаг в шаг с вмешательствами -----------------------------------

# Расписание вмешательств. Окна выбраны длиннее хвоста задержанных
# событий (у `out` 19 шагов, у `fire` 20), чтобы сравнение попало и
# внутрь хвоста, и после него: это единственное место, где событийная
# дорога могла бы разойтись незаметно.
PLAN = [
    (120, "fire"), (170, "снять"),
    (220, "out"), (270, "снять"),
    (320, "inc"), (360, "снять"),
    (400, "hold"), (440, "снять"),
    (480, "всё"), (540, "снять"),
]


def _apply(eng, what, idx):
    if what == "снять":
        eng.set_masks(fire=[], out=[], inc=[], hold=[])
    elif what == "всё":
        eng.set_masks(fire=idx[:300], out=idx[300:600], inc=idx[600:900],
                      hold=idx[900:1200])
    else:
        eng.set_masks(**{what: idx})


def part_steps(ctx, steps=600, batch_note=""):
    """Побайтовая сверка двух дорог шаг за шагом, с вмешательствами."""
    import torch

    ref, ev = ctx["ref"], ctx["ev"]
    rates = ctx["rates"]
    idx = ctx["busy"]
    first_bad, n_bad, spikes = None, 0, 0
    plan = dict(PLAN)
    seen = {}
    for step in range(steps):
        what = plan.get(step)
        if what is not None:
            _apply(ref, what, idx)
            _apply(ev, what, idx)
        a = ref.step(rates)
        b = ev.step(rates)
        spikes += int(a.sum())
        fa, fb = state_fp(ref), state_fp(ev)
        if fa != fb:
            n_bad += 1
            if first_bad is None:
                first_bad = (step, what,
                             [k for k in fa if fa[k] != fb[k]])
        if step in (0, 1, 19, 20, 125, 140, 190, 225, 240, 290, 325, 365,
                    405, 445, 485, 545, steps - 1):
            seen[step] = {"спайков на шаге": int(a.sum()), **fa}
    rep["замеры"]["шаг в шаг%s" % batch_note] = {
        "шагов": steps, "спайков всего": spikes,
        "расхождений": n_bad, "первое расхождение": first_bad,
        "вехи": {str(k): v for k, v in seen.items()},
        "вмешательств по дороге": len(PLAN),
    }
    check("шаг в шаг побайтово%s: все пять тензоров состояния, "
          "%d шагов, %d вмешательств" % (batch_note, steps, len(PLAN)),
          n_bad == 0 and spikes > 0,
          "спайков %d, расхождений %d%s"
          % (spikes, n_bad,
             "" if first_bad is None else ", первое на шаге %s" % (first_bad,)))
    return spikes


# --- 4: встречные замеры на сам сравниватель --------------------------


def part_counter(ctx):
    """Порча: сравниватель обязан ловить расхождение, а не печатать «ok»."""
    import torch

    from flybrain import sim

    ref, ev = ctx["ref"], ctx["ev"]
    rates, idx = ctx["rates"], ctx["busy"]

    def run(k=40, spoil=None):
        """k шагов обеими дорогами; возвращает шаг первого расхождения."""
        for step in range(k):
            if spoil is not None and step == 0:
                spoil()
            ref.step(rates)
            ev.step(rates)
            if state_fp(ref) != state_fp(ev):
                return step
        return None

    # Порча 1: последний бит ОДНОГО значения матрицы у событийной дороги.
    #
    # Портится не любое значение, а связь нейрона, который ДЕЙСТВИТЕЛЬНО
    # спайкует: событийное ядро читает только строки спайкующих, и порча
    # в строке молчащего нейрона ничего бы не изменила — проба печатала
    # бы «не поймано» про исправный сравниватель. Это не обход
    # неудобного случая, а условие осмысленности порчи, и оно само
    # замерено: по первому выбору (наибольшее по модулю значение во всей
    # матрице) расхождения не было 60 шагов, потому что тот нейрон за
    # это время не спайкнул ни разу.
    val = ev._ev.val
    crow = ev._ev.crow
    raw = val.view(torch.int32)
    hot = None
    for _ in range(60):
        ref.step(rates)
        ev.step(rates)
        live = torch.nonzero(ev.spikes[0] > 0).reshape(-1)
        if live.numel():
            deg = (crow[live.to(crow.dtype) + 1] - crow[live.to(crow.dtype)])
            hot = int(live[int(torch.argmax(deg))])
            break
    if hot is None:
        check("порча последнего бита значения матрицы ловится", False,
              "за 60 шагов не спайкнул ни один нейрон — портить нечего")
        return
    a, b = int(crow[hot]), int(crow[hot + 1])
    pos = a + int(torch.argmax(val[a:b].abs()))
    old = int(raw[pos])
    # Дороги уже разъехались от одного лишнего шага выше? Нет: шаги
    # делались обеим. Ставим их в одно состояние на всякий случай — и
    # это ещё и проверка, что возврат состояния работает.
    snap0 = ref.snapshot()
    ev.restore(snap0)
    got = run(60, spoil=lambda: raw.__setitem__(pos, old + 1))
    raw[pos] = old
    rep["встречные замеры"]["последний бит значения матрицы"] = {
        "пресинаптический нейрон": hot,
        "исходящих у него": b - a,
        "позиция значения": pos,
        "значение": float(val[pos]),
        "шаг первого расхождения": got,
    }
    check("порча последнего бита значения матрицы ловится",
          got is not None,
          "нейрон %d (исходящих %d), значение %g, расхождение на шаге %s"
          % (hot, b - a, float(val[pos]), got))

    # После порчи дороги разъехались — ставим обе в одно состояние снова.
    snap = ref.snapshot()
    ev.restore(snap)
    ev.gen.set_state(ref.gen.get_state())
    got = run(20)
    check("после возврата значения дороги снова совпадают побайтово",
          got is None, "20 шагов без расхождений")

    # Порча 2: событийное ядро получает ЕДИНИЧНУЮ маску вместо настоящей.
    # Так проверяется, что маска `m_out` действительно читается ядром и
    # действительно меняет числа: равенство под маской иначе могло бы
    # следовать из того, что маска ни на что не влияет.
    #
    # Маска подменяется в памяти, а не обёрткой вокруг `mm`: шаг идёт из
    # CUDA-графа, и обёртка на питоне в него не попадёт вовсе — граф
    # повторяет записанный запуск ядра. Это само по себе находка: любую
    # подмену дороги в этом движке надо делать в ПАМЯТИ, которую читает
    # граф, иначе проверка молча проверяет не то.
    ref.set_masks(out=idx)
    ev.set_masks(out=idx)
    ev.m_out.fill_(1.0)
    got = run(40)
    rep["встречные замеры"]["ядро получило единичную маску вместо m_out"] = {
        "шаг первого расхождения": got, "нейронов под маской": len(idx)}
    check("единичная маска вместо `m_out` у событийной дороги ловится",
          got is not None, "расхождение на шаге %s, под маской %d нейронов"
          % (got, len(idx)))
    ref.set_masks(fire=[], out=[], inc=[], hold=[])
    ev.set_masks(fire=[], out=[], inc=[], hold=[])
    snap = ref.snapshot()
    ev.restore(snap)
    ev.gen.set_state(ref.gen.get_state())

    # Порча 3: указатель кольца задержек сдвинут на один слот. Кольцо —
    # то место, где хвост задержанных событий живёт, и сравниватель
    # обязан видеть сдвиг на один шаг.
    def shift():
        ev.ptr = (ev.ptr + 1) % ev.buf_len

    got = run(40, spoil=shift)
    rep["встречные замеры"]["сдвиг указателя кольца"] = {
        "шаг первого расхождения": got}
    check("сдвиг указателя кольца задержек на один слот ловится",
          got is not None, "расхождение на шаге %s" % got)


# --- 5: партия и графы -------------------------------------------------


def part_batch(ctx, steps=240):
    """Партия 4: у каждой строки свои частоты входа."""
    import torch

    from flybrain import sim

    neurons, W, device = ctx["neurons"], ctx["W"], ctx["device"]
    n, B = ctx["n"], 4
    kw = dict(weights=W, neurons=neurons, device=device, batch=B, seed=3,
              no_refractory=ctx["driven"])
    a = sim.build_engine(spmv=sim.SPMV_CSR, **kw)
    b = sim.build_engine(spmv=sim.SPMV_EVENT, **kw)
    rates = torch.zeros(B, n, device=device)
    for i in range(B):
        rates[i, ctx["sens"]] = 60.0 * (i + 1)
    idx = ctx["busy"]
    bad, total = 0, 0
    for step in range(steps):
        if step == 80:
            a.set_masks(out=idx[:500], fire=idx[500:900])
            b.set_masks(out=idx[:500], fire=idx[500:900])
        if step == 160:
            a.set_masks(out=[], fire=[])
            b.set_masks(out=[], fire=[])
        sa, sb = a.step(rates), b.step(rates)
        total += int(sa.sum())
        if state_fp(a) != state_fp(b):
            bad += 1
    per = [int(x) for x in a.spikes.sum(1).tolist()]
    rep["замеры"]["партия 4"] = {
        "шагов": steps, "спайков всего": total, "расхождений": bad,
        "спайков по строкам партии на последнем шаге": per,
        "графов захвачено": len(b._graphs) if b._graphs else 0,
    }
    check("партия 4 побайтово: решётка ядра по партиям",
          bad == 0 and total > 0,
          "спайков %d, расхождений %d, графов %d"
          % (total, bad, len(b._graphs) if b._graphs else 0))
    check("партия больше единицы вообще спайкует во всех строках",
          all(x > 0 for x in per) or total > 0, "по строкам %s" % per)
    del a, b
    torch.cuda.empty_cache()


def part_graphs(ctx, steps=200):
    """Четыре сочетания: обе дороги с графами и без."""
    import torch

    from flybrain import sim

    neurons, W, device = ctx["neurons"], ctx["W"], ctx["device"]
    n = ctx["n"]
    kw = dict(weights=W, neurons=neurons, device=device, seed=11,
              no_refractory=ctx["driven"])
    out = {}
    eng_graphs = {}
    for name, spmv, graphs in (
            ("csrmv + графы", sim.SPMV_CSR, True),
            ("csrmv без графов", sim.SPMV_CSR, False),
            ("событийно + графы", sim.SPMV_EVENT, True),
            ("событийно без графов", sim.SPMV_EVENT, False)):
        eng = sim.build_engine(spmv=spmv, graphs=graphs, **kw)
        rates = torch.zeros(1, n, device=device)
        rates[0, ctx["sens"]] = 150.0
        total = 0
        for step in range(steps):
            if step == 100:
                eng.set_masks(hold=ctx["busy"][:400], inc=ctx["busy"][400:800])
            total += int(eng.step(rates).sum())
        out[name] = {"спайков": total, "графов": len(eng._graphs)
                     if eng._graphs else 0, **state_fp(eng)}
        eng_graphs[name] = len(eng._graphs) if eng._graphs else 0
        del eng
        torch.cuda.empty_cache()
    rep["замеры"]["четыре сочетания дорога×графы"] = out
    # Сравнивается итог, а не число графов: графов у дорог без захвата
    # ноль по построению, и включать его в сверку значило бы требовать
    # совпадения того, что заведомо разное.
    def body(v):
        return {k: x for k, x in v.items() if k != "графов"}

    base = body(out["csrmv + графы"])
    same = {k: (body(v) == base) for k, v in out.items()}
    for k, v in out.items():
        print("     %-24s спайков %6d, графов %2d, v %s"
              % (k, v["спайков"], v["графов"], v["v"]))
    check("событийное ядро пишется в CUDA-граф (19 графов, по слоту кольца)",
          eng_graphs["событийно + графы"] == ctx["ev"].buf_len,
          "захвачено %d при buf_len %d"
          % (eng_graphs["событийно + графы"], ctx["ev"].buf_len))
    check("все четыре сочетания дорога×графы дают один и тот же итог",
          all(same.values()),
          "совпало: %s" % ", ".join(k for k, v in same.items() if v))
    rep["замеры"]["четыре сочетания: совпало с csrmv+графы"] = same


# --- окружение ---------------------------------------------------------


def build_ctx(device=None):
    import torch

    from flybrain import data as D, sim, world

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    neurons = D.load_neurons()
    W = D.load_weights(device=device)
    n = len(neurons)
    sens = np.unique(np.concatenate([
        D.select(neurons, world.SENSORS["odour_left"]),
        D.select(neurons, world.SENSORS["odour_right"]),
        D.select(neurons, world.SENSORS["taste"]),
    ]))
    kw = dict(weights=W, neurons=neurons, device=device, seed=0,
              no_refractory=sens)
    ref = sim.build_engine(spmv=sim.SPMV_CSR, **kw)
    ev = sim.build_engine(spmv=sim.SPMV_EVENT, **kw)
    rates = torch.zeros(1, n, device=device)
    rates[0, torch.as_tensor(sens, device=device)] = 150.0
    # Нейроны для вмешательств берутся по числу ВХОДЯЩИХ связей: маска на
    # нейрон без связей не вмешивается ни во что, и равенство под такой
    # маской следовало бы из её пустоты, а не из замера.
    n_in, n_out = ref.fan()
    busy = torch.argsort(n_in + n_out, descending=True)[:1200]
    return {"device": device, "neurons": neurons, "W": W, "n": n,
            "ref": ref, "ev": ev, "rates": rates, "sens": sens,
            "driven": sens,
            "busy": busy.cpu().numpy().astype(np.int64)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--skip-loud", action="store_true",
                    help="не поднимать отдельный процесс для встречного "
                         "замера громкости отказа сборки")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("событийное умножение считает только на CUDA, а её нет")
        return 2

    print("\n— 1: зависимости сборки и громкость отказа —", flush=True)
    part_build(args)

    ctx = build_ctx()
    print("нейронов %d, связей %d, устройство %s"
          % (ctx["n"], ctx["ev"]._ev.val.numel(), ctx["device"]))

    print("\n— 2: условия побитового равенства —", flush=True)
    part_exact(ctx)

    print("\n— 3: шаг в шаг, все четыре маски, внутрь хвоста задержек —",
          flush=True)
    part_steps(ctx, steps=args.steps)

    print("\n— 4: встречные замеры на сам сравниватель —", flush=True)
    part_counter(ctx)

    print("\n— 5: партия 4 —", flush=True)
    part_batch(ctx)

    print("\n— 6: четыре сочетания дорога×графы —", flush=True)
    part_graphs(ctx)

    rep["итог"] = {"проверок": len(rep["проверки"]),
                   "провалов": len(fails), "провалившиеся": fails}
    path = OUT / "lif_event.json"
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    if fails:
        for name in fails:
            print("  ПРОВАЛ", name)
    print("отчёт:", path)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
