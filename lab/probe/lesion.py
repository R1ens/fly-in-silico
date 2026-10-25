"""Виртуальные вмешательства: восемь контролей и три измерения.

Механизм масок (`Engine.m_fire`, `m_out`, `m_in` и слой `BodyLive`)
обещает больше, чем видно по счётчику нейронов, и каждое обещание здесь
проверяется числом, рядом с которым стоит ВСТРЕЧНЫЙ ЗАМЕР — то, что
показало бы прежнее или сломанное поведение. Без встречного замера
проверка «маски поставились» проходит и на пустом механизме: в этом
проекте так уже ловили четыре пробы.

Что проверяется (нумерация владельца проекта):

 1. нулевое вмешательство не меняет результат — ПОБАЙТОВО, по отпечаткам
    состояния, а не по суммам;
 2. повтор одного зерна воспроизводим;
 3. подавленные нейроны не выдают спайков в период действия;
 4. входящие и исходящие отключаются в правильном направлении — две
    непересекающиеся популяции, A посылает в B, и смысл проверяется по
    активности, а не по счётчику масок;
 5. снятие маски восстанавливает исходные правила вычисления —
    перенос состояния в чистый движок и побайтовое сравнение дальше;
 6. вмешательство не переходит в следующий сеанс;
 7. отмена и аварийное завершение освобождают ресурсы;
 8. быстрый (CUDA-графы) и эталонный режимы совпадают — и без
    вмешательств, и с одинаковым вмешательством.

И три измерения, обещанных документацией механизма:

 9. хвост задержанных событий в ШАГАХ — у `out` и у `fire` отдельно;
10. `in` действует мгновенно и в обе стороны;
11. мембрана подавленного нейрона: сколько спайков в первом шаге после
    снятия подавления.

Измерения 9–11 сделаны на голом движке: шаг управления тела — 20 мс, то
есть 200 тиков, и хвост в 19 тиков в нём не разглядеть. Пункты 1, 2, 5,
8 — на отпечатках голого движка; 6 — на настоящих эпизодах в теле, три
подряд в одном процессе.

Поведение с вмешательством и без сравнивается на ТРЁХ зёрнах на
условие, с разбросом внутри условия, и вывод печатается словами:
разницу активности нейронов проба утверждает прямо, разницу поведения —
нет, и почему именно, сказано числом.

Что проба нашла в `flybrain/` и не стала править (подробности с замерами
печатаются по ходу и ложатся в `lesion.json`):

* хвост подавления `fire` на один шаг длиннее, чем сказано в докстрингах
  `Engine._core` и `Engine.set_masks`;
* мембрана подавленного нейрона ничем не ограничена сверху, и при снятии
  подавления популяция даёт синхронный залп;
* `_lesions` принимает срок от 0,1 мс, а снять вмешательство можно
  только на границе шага управления в 20 мс;
* снятие по сроку опознаётся в панели только по запросу, без вида;
* `BodyLive.attach` не чистит список вмешательств при повторной привязке.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\lesion.py
"""

import contextlib
import gc
import hashlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "находки": [], "сорвалось": []}

# Популяции для направленной проверки. A заведомо посылает в B (считано
# по матрице, см. `part_direction`), пересечение пусто, и A получает
# пуассоновский вход, то есть заведомо активна.
Q_A = "class:olfactory+side:left"      # ORN левой антенны, 1116 нейронов
Q_B = "class:ALPN+side:left"           # проекционные нейроны AL, 341

# Единая разметка времени для всех трассировок голого движка.
STEPS, K_ON, K_OFF = 800, 150, 650

EMPTY = np.array([], dtype=np.int64)


def check(name, ok, info=""):
    # Поток печати сбрасывается на каждой строке: проба идёт минуты, и
    # при перенаправлении в файл буфер держал бы вывод блоками, а по
    # такому журналу не видно, на чём она стоит.
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    """Находка в `flybrain/`, которую проба не правит, а предъявляет."""
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def stage(text):
    print(text, flush=True)


def fp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------- общий инструмент

def build(ctx, seed=0, graphs=True):
    from flybrain import sim

    return sim.build_engine(weights=ctx["W"], neurons=ctx["neurons"],
                            device=ctx["dev"], seed=seed,
                            no_refractory=ctx["A"], graphs=graphs)


def rates_of(ctx):
    import torch

    r = torch.zeros(1, ctx["n"], device=ctx["dev"])
    r[0, ctx["tA"]] = 200.0
    return r


def masks(fire=EMPTY, out=EMPTY, inc=EMPTY):
    return {"fire": fire, "out": out, "inc": inc}


def march(ctx, eng, steps, sched=None, start=0):
    """Прогнать шаги и снять отпечатки хода и конечного состояния.

    Отпечаток хода — не сумма спайков: суммы совпадают и у разных
    растров. На каждом шаге берётся и общее число спайков, и скалярное
    произведение растра на постоянный вектор-пробник, то есть отпечаток
    чувствителен к тому, КАКИЕ нейроны выдали спайк, а не сколько.
    """
    import torch

    sched = sched or {}
    r = rates_of(ctx)
    hist = torch.zeros(steps, device=ctx["dev"])
    patt = torch.zeros(steps, device=ctx["dev"])
    counts = torch.zeros(ctx["n"], dtype=torch.int64, device=ctx["dev"])
    for s in range(steps):
        key = start + s
        if key in sched:
            eng.set_masks(**sched[key])
        sp = eng.step(r)
        hist[s] = sp.sum()
        patt[s] = (sp[0] * ctx["pvec"]).sum()
        counts += sp[0].to(torch.int64)
    if ctx["dev"] == "cuda":
        torch.cuda.synchronize()
    return {
        "спайков всего": int(counts.sum().item()),
        "счётчик": fp(counts.cpu().numpy()),
        "по шагам": fp(hist.cpu().numpy()),
        "растр по шагам": fp(patt.cpu().numpy()),
        "v": fp(eng.v.detach().cpu().numpy()),
        "проводимость": fp(eng.conductance.detach().cpu().numpy()),
        "кольцо задержек": fp(eng.delay_buffer.detach().cpu().numpy()),
        "рефрактерность": fp(eng.refrac.detach().cpu().numpy()),
        "спайки": fp(eng.spikes.detach().cpu().numpy()),
        "указатель кольца": int(eng.ptr),
    }


def differs(a, b):
    """Какие поля отпечатков разошлись."""
    return sorted(k for k in a if a[k] != b[k])


def trace(ctx, sched, seed=0, graphs=True, steps=STEPS):
    """Пошаговая запись: проводимость, растр и кольцо по двум популяциям.

    Состояние снимается ПОСЛЕ шага, поэтому индекс записи — это номер
    шага, на котором состояние стало таким. Кольцо, наоборот, читается
    ДО шага: это сырой входящий поток, который шаг либо доставит, либо
    обнулит маской `m_in`.
    """
    import torch

    eng = build(ctx, seed=seed, graphs=graphs)
    r = rates_of(ctx)
    tA, tB = ctx["tA"], ctx["tB"]
    rec = {k: [] for k in ("cond_A", "cond_B", "sp_A", "sp_B")}
    sc = {k: [] for k in ("ring_A", "ring_B", "vmax_A", "vmax_B",
                          "nov_A", "nov_B")}
    for s in range(steps):
        if s in sched:
            eng.set_masks(**sched[s])
        sc["ring_A"].append(float(eng.delay_buffer[0, eng.ptr, tA].sum()))
        sc["ring_B"].append(float(eng.delay_buffer[0, eng.ptr, tB].sum()))
        sp = eng.step(r)
        rec["sp_A"].append(sp[0, tA].detach().cpu().numpy().copy())
        rec["sp_B"].append(sp[0, tB].detach().cpu().numpy().copy())
        rec["cond_A"].append(eng.conductance[0, tA].detach().cpu().numpy().copy())
        rec["cond_B"].append(eng.conductance[0, tB].detach().cpu().numpy().copy())
        sc["vmax_A"].append(float(eng.v[0, tA].max()))
        sc["vmax_B"].append(float(eng.v[0, tB].max()))
        sc["nov_A"].append(int((eng.v[0, tA] > eng.v_th).sum()))
        sc["nov_B"].append(int((eng.v[0, tB] > eng.v_th).sum()))
    buf = eng.buf_len
    del eng, r
    gc.collect()
    out = {k: np.array(v) for k, v in rec.items()}
    out.update({k: np.array(v) for k, v in sc.items()})
    out["buf_len"] = buf
    return out


def first_diff(a, b, key, steps=STEPS):
    """Первый шаг, на котором записи разошлись. None — не разошлись."""
    for s in range(steps):
        if not np.array_equal(a[key][s], b[key][s]):
            return s
    return None


def rel(a, b, key, base):
    d = first_diff(a, b, key)
    return None if d is None else d - base


# -------------------------------------- 1, 2, 5, 8: отпечатки голого движка

def part_equal(ctx):
    import torch

    A, B = ctx["A"], ctx["B"]
    steps = 400
    les = {100: masks(out=A), 250: masks(fire=B)}

    # --- пункт 1: нулевое вмешательство ---
    e = build(ctx)
    bare = march(ctx, e, steps)
    del e
    e = build(ctx)
    e.set_masks(fire=[], out=[], inc=[])
    zero = march(ctx, e, steps, sched={200: masks()})
    clear_ok = e.masks_clear()
    cnt0 = e.mask_counts()
    del e
    e = build(ctx)
    real = march(ctx, e, steps, sched={0: masks(fire=B)})
    del e
    gc.collect()

    check("нулевое вмешательство не меняет ни одного бита состояния",
          bare == zero,
          "разошлись поля %s; отпечаток v %s против %s"
          % (differs(bare, zero) or "ничего", bare["v"], zero["v"]))
    check("отпечаток вообще способен поймать вмешательство "
          "(встречный замер)",
          len(differs(bare, real)) >= 8,
          "с подавлением %d нейронов разошлось %d полей из %d, v стал %s, "
          "спайков %d против %d"
          % (len(B), len(differs(bare, real)), len(bare), real["v"],
             real["спайков всего"], bare["спайков всего"]))
    check("при пустых списках маски действительно пусты",
          clear_ok and set(cnt0.values()) == {0},
          "masks_clear %s, mask_counts %s" % (clear_ok, cnt0))

    # --- пункт 2: повтор одного зерна ---
    e = build(ctx, seed=7)
    rep1 = march(ctx, e, steps, sched=les)
    del e
    e = build(ctx, seed=7)
    rep2 = march(ctx, e, steps, sched=les)
    del e
    e = build(ctx, seed=8)
    other = march(ctx, e, steps, sched=les)
    del e
    gc.collect()
    check("повтор с тем же зерном и тем же вмешательством воспроизводим",
          rep1 == rep2,
          "разошлись поля %s; счётчик %s, спайков %d"
          % (differs(rep1, rep2) or "ничего", rep1["счётчик"],
             rep1["спайков всего"]))
    check("другое зерно даёт другой прогон (встречный замер)",
          len(differs(rep1, other)) >= 8,
          "разошлось %d полей из %d; спайков %d против %d"
          % (len(differs(rep1, other)), len(rep1), rep1["спайков всего"],
             other["спайков всего"]))

    # --- пункт 8: быстрый против эталонного ---
    e = build(ctx, graphs=True)
    with_g = e._graphs is not None
    fast_clean = march(ctx, e, steps)
    del e
    e = build(ctx, graphs=False)
    no_g = e._graphs is None
    slow_clean = march(ctx, e, steps)
    del e
    e = build(ctx, graphs=True)
    fast_les = march(ctx, e, steps, sched=les)
    del e
    e = build(ctx, graphs=False)
    slow_les = march(ctx, e, steps, sched=les)
    del e
    gc.collect()
    check("режимы заявлены честно: быстрый с графами, эталонный без",
          with_g and no_g,
          "graphs=True -> графы %s, graphs=False -> графов нет %s"
          % (with_g, no_g))
    check("без вмешательств быстрый и эталонный совпали побайтово",
          fast_clean == slow_clean,
          "разошлись поля %s; спайков %d и %d"
          % (differs(fast_clean, slow_clean) or "ничего",
             fast_clean["спайков всего"], slow_clean["спайков всего"]))
    check("с одинаковым вмешательством быстрый и эталонный совпали "
          "побайтово", fast_les == slow_les,
          "разошлись поля %s; спайков %d и %d"
          % (differs(fast_les, slow_les) or "ничего",
             fast_les["спайков всего"], slow_les["спайков всего"]))
    check("захваченный граф ВИДИТ маски, а не читает старую память "
          "(встречный замер)",
          len(differs(fast_clean, fast_les)) >= 8,
          "в быстром режиме вмешательство изменило %d полей из %d и "
          "%d спайков (%d против %d); будь маски вне графа, было бы 0"
          % (len(differs(fast_clean, fast_les)), len(fast_clean),
             abs(fast_les["спайков всего"] - fast_clean["спайков всего"]),
             fast_les["спайков всего"], fast_clean["спайков всего"]))
    rep["замеры"]["пункт 8"] = {
        "быстрый без вмешательств, спайков": fast_clean["спайков всего"],
        "эталонный без вмешательств, спайков": slow_clean["спайков всего"],
        "быстрый с вмешательством, спайков": fast_les["спайков всего"],
        "эталонный с вмешательством, спайков": slow_les["спайков всего"],
    }

    # --- пункт 5: снятие восстанавливает ПРАВИЛА вычисления ---
    #
    # Сравнивать состояния «до» и «после» нельзя: за время вмешательства
    # состояние ушло, и побайтово оно уже не вернётся. Проверяется
    # другое, и именно это обещано: после снятия движок считает по тем же
    # правилам, что и движок, в котором вмешательства не было ВОВСЕ.
    # Поэтому состояние поражённого движка переносится в чистый, и дальше
    # оба идут рядом. Разойдись правила хоть в одном месте — отпечатки
    # разъедутся на первом же шаге.
    cut, tail = 300, 150
    sch_set = {100: masks(fire=B, out=A, inc=B)}
    lesioned = build(ctx)
    march(ctx, lesioned, cut, sched=sch_set)
    donor = {name: getattr(lesioned, name).clone()
             for name in ("v", "conductance", "spikes", "refrac",
                          "delay_buffer")}
    donor_ptr = lesioned.ptr
    donor_gen = lesioned.gen.get_state().clone()
    cnt_before = lesioned.mask_counts()
    # Снятие — ровно то, что делает пустой список в ручке `lesion`.
    lesioned.set_masks(fire=[], out=[], inc=[])
    cleared = lesioned.masks_clear()
    cnt_after = lesioned.mask_counts()

    def transplant(eng):
        for name, val in donor.items():
            getattr(eng, name).copy_(val)
        eng.ptr = donor_ptr
        eng.gen.set_state(donor_gen.clone())

    host = build(ctx)
    march(ctx, host, cut)
    gen_same = bool(torch.equal(host.gen.get_state(), donor_gen))
    transplant(host)
    after_free = march(ctx, lesioned, tail, start=cut)
    after_host = march(ctx, host, tail, start=cut)
    del lesioned, host
    gc.collect()

    # Встречный замер: тот же перенос, но маску НЕ снимаем.
    stuck = build(ctx)
    march(ctx, stuck, cut, sched=sch_set)
    host2 = build(ctx)
    march(ctx, host2, cut)
    transplant(host2)
    after_stuck = march(ctx, stuck, tail, start=cut)
    after_host2 = march(ctx, host2, tail, start=cut)
    del stuck, host2
    gc.collect()

    check("снятие обнулило все три маски",
          cleared and set(cnt_after.values()) == {0},
          "было %s, стало %s" % (cnt_before, cnt_after))
    check("расход случайных чисел не зависит от вмешательства", gen_same,
          "состояние генератора после %d шагов совпало" % cut)
    check("после снятия движок считает по правилам чистого движка "
          "(перенос состояния, %d шагов)" % tail,
          after_free == after_host,
          "разошлись поля %s; спайков %d и %d"
          % (differs(after_free, after_host) or "ничего",
             after_free["спайков всего"], after_host["спайков всего"]))
    check("не снятая маска этот перенос ломает (встречный замер)",
          len(differs(after_stuck, after_host2)) >= 6,
          "с не снятой маской разошлось %d полей из %d, спайков %d "
          "против %d (разница %d)"
          % (len(differs(after_stuck, after_host2)), len(after_stuck),
             after_stuck["спайков всего"], after_host2["спайков всего"],
             abs(after_stuck["спайков всего"]
                 - after_host2["спайков всего"])))
    rep["замеры"]["пункт 5"] = {
        "шагов под вмешательством": cut - 100,
        "шагов после снятия": tail,
        "спайков после снятия, поражённый": after_free["спайков всего"],
        "спайков после снятия, чистый с тем же состоянием":
            after_host["спайков всего"],
        "спайков, если маску не снять": after_stuck["спайков всего"],
    }


# ---------------------------------------- 3, 4: подавление и направление

def part_direction(ctx, tr):
    """Направление вмешательств — по смыслу, а не по счётчику масок."""
    A, B = ctx["A"], ctx["B"]
    K = K_ON
    clean = tr["чисто"]

    # --- кто кому посылает: читается из матрицы, а не предполагается ---
    rep["замеры"]["связи A и B"] = ctx["блоки"]
    check("популяции не пересекаются и A посылает в B",
          ctx["пересечение"] == 0 and ctx["блоки"]["A->B, связей"] > 0,
          "A %d нейронов, B %d, пересечение %d; A->B %d связей (вес %.0f), "
          "B->A %d связей (вес %.0f)"
          % (len(A), len(B), ctx["пересечение"],
             ctx["блоки"]["A->B, связей"], ctx["блоки"]["A->B, вес"],
             ctx["блоки"]["B->A, связей"], ctx["блоки"]["B->A, вес"]))

    # --- пункт 3: подавленные не выдают спайков ---
    win = (K, K_OFF)
    for name, key, pop in (("A", "sp_A", A), ("B", "sp_B", B)):
        les = tr["fire на %s" % name]
        got = int(les[key][win[0]:win[1]].sum())
        was = int(clean[key][win[0]:win[1]].sum())
        before = int(les[key][:K].sum())
        check("подавленная популяция %s не выдала ни одного спайка за "
              "%d шагов действия" % (name, win[1] - win[0]),
              got == 0 and was > 0 and before > 0,
              "спайков %d; без подавления за те же шаги %d; до включения "
              "та же популяция дала %d (значит молчание не от того, что "
              "она и так молчит)" % (got, was, before))

    # --- пункт 4: направление ---
    tbl = {}
    for name in ("out на A", "in на A", "in на B", "out на B"):
        t = tr[name]
        tbl[name] = {
            "cond_A, шагов до расхождения": rel(clean, t, "cond_A", K),
            "cond_B, шагов до расхождения": rel(clean, t, "cond_B", K),
            "sp_A, шагов до расхождения": rel(clean, t, "sp_A", K),
            "sp_B, шагов до расхождения": rel(clean, t, "sp_B", K),
            "Δспайков A за 300 шагов":
                int(t["sp_A"][K:K + 300].sum() - clean["sp_A"][K:K + 300].sum()),
            "Δспайков B за 300 шагов":
                int(t["sp_B"][K:K + 300].sum() - clean["sp_B"][K:K + 300].sum()),
        }
    rep["замеры"]["пункт 4: таблица направлений"] = tbl
    print("     вмешательство | cond_A | cond_B | sp_A | sp_B | "
          "Δспайков A | Δспайков B")
    for name, row in tbl.items():
        print("     %-13s | %6s | %6s | %4s | %4s | %10d | %10d"
              % (name, row["cond_A, шагов до расхождения"],
                 row["cond_B, шагов до расхождения"],
                 row["sp_A, шагов до расхождения"],
                 row["sp_B, шагов до расхождения"],
                 row["Δспайков A за 300 шагов"],
                 row["Δспайков B за 300 шагов"]))

    n_a = int(clean["sp_A"][K:K + 300].sum())
    n_b = int(clean["sp_B"][K:K + 300].sum())
    out_a = tbl["out на A"]
    in_b = tbl["in на B"]
    in_a = tbl["in на A"]

    # Причинное окно: раньше чем через buf_len шагов никакое событие
    # дойти не может, поэтому «активность A не изменилась» проверяется
    # там, где это утверждение не размывается обратными связями.
    bl = clean["buf_len"]
    w0, w1 = K + bl, K + int(out_a["sp_A, шагов до расхождения"] or bl)
    a_cl = int(clean["sp_A"][w0:w1].sum())
    a_le = int(tr["out на A"]["sp_A"][w0:w1].sum())
    b_cl = int(clean["sp_B"][w0:w1].sum())
    b_le = int(tr["out на A"]["sp_B"][w0:w1].sum())
    check("отключение исходящих у A изменило активность B и не изменило "
          "активность A",
          out_a["Δспайков B за 300 шагов"] <= -50
          and abs(out_a["Δспайков A за 300 шагов"]) < 0.01 * n_a
          and a_le == a_cl and b_le != b_cl,
          "в окне [+%d, +%d) спайков A %d против %d (разница %d), "
          "спайков B %d против %d; за 300 шагов B %+d из %d (%.1f%%), "
          "A %+d из %d (%.2f%%)"
          % (bl, w1 - K, a_le, a_cl, a_le - a_cl, b_le, b_cl,
             out_a["Δспайков B за 300 шагов"], n_b,
             100.0 * abs(out_a["Δспайков B за 300 шагов"]) / n_b,
             out_a["Δспайков A за 300 шагов"], n_a,
             100.0 * abs(out_a["Δспайков A за 300 шагов"]) / n_a))

    w1b = K + int(in_b["sp_A, шагов до расхождения"] or bl)
    a_cl2 = int(clean["sp_A"][K:w1b].sum())
    a_le2 = int(tr["in на B"]["sp_A"][K:w1b].sum())
    check("отключение входящих у B изменило активность B и не изменило "
          "активность A",
          in_b["Δспайков B за 300 шагов"] <= -500
          and abs(in_b["Δспайков A за 300 шагов"]) < 0.01 * n_a
          and a_le2 == a_cl2,
          "в окне [+0, +%d) спайков A %d против %d (разница %d); за 300 "
          "шагов B %+d из %d (%.1f%% активности B снято), A %+d из %d "
          "(%.2f%%)"
          % (w1b - K, a_le2, a_cl2, a_le2 - a_cl2,
             in_b["Δспайков B за 300 шагов"], n_b,
             100.0 * abs(in_b["Δспайков B за 300 шагов"]) / n_b,
             in_b["Δспайков A за 300 шагов"], n_a,
             100.0 * abs(in_b["Δспайков A за 300 шагов"]) / n_a))

    # Это и есть ловушка на симметричную ошибку: перепутай `in` и `out`,
    # и мгновенное изменение проводимости переехало бы с цели на
    # отправителя, а счётчик масок остался бы тем же.
    check("`in` и `out` не перепутаны: мгновенно меняется проводимость "
          "ТОЛЬКО того, кому отключили входящие",
          in_a["cond_A, шагов до расхождения"] == 0
          and in_b["cond_B, шагов до расхождения"] == 0
          and out_a["cond_A, шагов до расхождения"] == bl
          and out_a["cond_B, шагов до расхождения"] == bl,
          "`in` на A: cond_A через +%s шагов; `out` на A: cond_A через "
          "+%s, cond_B через +%s; `in` на B: cond_B через +%s. Будь "
          "направления перепутаны, нули и %d стояли бы наоборот"
          % (in_a["cond_A, шагов до расхождения"],
             out_a["cond_A, шагов до расхождения"],
             out_a["cond_B, шагов до расхождения"],
             in_b["cond_B, шагов до расхождения"], bl))
    check("встречный замер направления: неверное направление даёт "
          "зеркальные числа",
          in_a["Δспайков B за 300 шагов"] == 0
          and in_a["Δспайков A за 300 шагов"] < 0,
          "`in` на A (неверное направление для «A не посылает»): "
          "Δспайков B %+d, Δспайков A %+d — то есть досталось самой A, а "
          "B не тронута; правильное `out` на A дало B %+d, A %+d"
          % (in_a["Δспайков B за 300 шагов"],
             in_a["Δспайков A за 300 шагов"],
             out_a["Δспайков B за 300 шагов"],
             out_a["Δспайков A за 300 шагов"]))

    # Почему проводимость самой A всё же меняется через buf_len шагов —
    # не ошибка направления, а связи A внутри себя. Число предъявлено.
    if ctx["блоки"]["A->A, связей"] > 0:
        rep["замеры"]["почему cond_A меняется через buf_len при out на A"] = (
            "у A есть %d связей внутри себя (вес %.0f): отключив её "
            "исходящие, мы отключили и её собственный вход от себя же"
            % (ctx["блоки"]["A->A, связей"], ctx["блоки"]["A->A, вес"]))


# ------------------------------------------- 9, 10, 11: хвост и мембрана

def part_tail(ctx, tr):
    clean = tr["чисто"]
    bl = clean["buf_len"]
    dt = 0.1
    K, M = K_ON, K_OFF
    rep["замеры"]["buf_len движка"] = bl
    rep["замеры"]["шаг движка, мс"] = dt

    # --- пункт 9: хвост у out и у fire, в шагах ---
    tail = {}
    for kind, nm in (("out", "out на A"), ("fire", "fire на A")):
        keep = tr[nm + " (держим)"]
        rels = tr[nm]
        tail[kind] = {
            "хвост при включении, шагов": rel(clean, keep, "cond_B", K),
            "пауза после снятия, шагов": rel(keep, rels, "cond_B", M),
            "спайки получателей изменились через, шагов":
                rel(clean, keep, "sp_B", K),
        }
    rep["замеры"]["пункт 9: хвост задержек"] = tail
    t_out = tail["out"]["хвост при включении, шагов"]
    o_off = tail["out"]["пауза после снятия, шагов"]
    t_fire = tail["fire"]["хвост при включении, шагов"]
    f_off = tail["fire"]["пауза после снятия, шагов"]

    check("хвост `out` измерен и равен длине кольца задержек",
          t_out == bl and o_off == bl,
          "получатели получают ещё %d шагов после включения и не "
          "получают %d шагов после снятия; buf_len %d, то есть %.1f мс "
          "модельного времени" % (t_out, o_off, bl, bl * dt))
    check("хвост `fire` измерен и на один шаг длиннее кольца",
          t_fire == bl + 1 and f_off == bl + 1,
          "получатели получают ещё %d шагов и не получают %d шагов; "
          "buf_len %d (%.1f мс). Лишний шаг не описка механизма: `out` "
          "маскирует спайк ПРОШЛОГО шага в начале шага, а `fire` — спайк "
          "ТЕКУЩЕГО шага в конце, и до кольца он доезжает на шаг позже"
          % (t_fire, f_off, bl, (bl + 1) * dt))
    check("хвост не нулевой, то есть вмешательство действительно не "
          "мгновенно (встречный замер)",
          t_out > 0 and t_fire > 0
          and rel(clean, tr["in на B"], "cond_B", K) == 0,
          "у `out` %d шагов и у `fire` %d шагов против 0 шагов у `in` на "
          "том же месте" % (t_out, t_fire))
    check("`fire` действует на самом подавленном нейроне мгновенно",
          rel(clean, tr["fire на A (держим)"], "cond_A", K) == 0,
          "cond_A расходится через +%s шагов: сброс проводимости и "
          "мембраны идёт по замаскированному спайку, поэтому у самого "
          "нейрона сброса нет с первого же шага, а хвост %d шагов — "
          "только у его получателей"
          % (rel(clean, tr["fire на A (держим)"], "cond_A", K), t_fire))

    # --- пункт 10: `in` мгновенно и в обе стороны ---
    in_les, in_keep = tr["in на B"], tr["in на B (держим)"]
    on = rel(clean, in_keep, "cond_B", K)
    off = rel(in_keep, in_les, "cond_B", M)
    ring_at_on = float(clean["ring_B"][K])
    ring_at_off = float(in_les["ring_B"][M])
    ring_next = float(in_les["ring_B"][M + 1:M + 21].mean())
    ring_mean_before = float(clean["ring_B"][K - 20:K].mean())
    burst = ring_at_off / max(ring_next, 1e-9)
    rep["замеры"]["пункт 10: входящие"] = {
        "шагов до обрыва доставки": on,
        "шагов до возобновления": off,
        "в кольце лежало на шаге включения": round(ring_at_on, 2),
        "в кольце на шаге снятия": round(ring_at_off, 2),
        "в кольце в среднем за 20 шагов после снятия": round(ring_next, 2),
        "в кольце в среднем за 20 шагов до включения":
            round(ring_mean_before, 2),
        "отношение «залпа» к обычному шагу": round(burst, 3),
    }
    check("отключение входящих обрывает доставку в ТОТ ЖЕ шаг, включая "
          "уже летящее", on == 0 and ring_at_on > 0,
          "проводимость B изменилась через +%d шагов, хотя в кольце на "
          "этом шаге лежало %.1f мВ уже отправленных событий (за 20 "
          "шагов до включения в среднем %.1f) — они и были выброшены"
          % (on, ring_at_on, ring_mean_before))
    check("снятие входящих возобновляет доставку в тот же шаг", off == 0,
          "расхождение с прогоном, где маску держат, на шаге M+%d" % off)
    check("накопленного залпа при снятии нет", burst < 3.0,
          "на шаге снятия доставлено %.1f мВ против %.1f мВ в среднем за "
          "следующие 20 шагов, отношение %.2f. Копись события в кольце, "
          "отношение было бы около длины кольца, то есть %d"
          % (ring_at_off, ring_next, burst, bl))
    check("`in` действует в обе стороны мгновенно, а `out` — ни в одну "
          "(встречный замер)",
          on == 0 and off == 0 and t_out == bl and o_off == bl,
          "`in`: +%d шагов на включение и +%d на снятие; `out`: +%d и +%d"
          % (on, off, t_out, o_off))

    # --- пункт 11: мембрана подавленного ---
    mem = {}
    for name, pop, spk, vmx, nov in (
            ("B", ctx["B"], "sp_B", "vmax_B", "nov_B"),
            ("A", ctx["A"], "sp_A", "vmax_A", "nov_A")):
        les = tr["fire на %s" % name]
        over = int(les[nov][M - 1])
        burst_n = int(les[spk][M].sum())
        nxt = int(les[spk][M + 1].sum())
        base = float(clean[spk][M - 50:M].sum()) / 50.0
        base_at = int(clean[spk][M].sum())
        per_neuron = int(les[spk][M:M + 3].sum(0).max())
        mem[name] = {
            "нейронов в популяции": len(pop),
            "подавление длилось шагов": M - K,
            "v максимум перед снятием, мВ": round(float(les[vmx][M - 1]), 2),
            "v максимум без подавления, мВ":
                round(float(clean[vmx][M - 1]), 2),
            "нейронов выше порога перед снятием": over,
            "спайков в первом шаге после снятия": burst_n,
            "спайков в первом шаге без подавления": base_at,
            "обычно спайков на шаг": round(base, 2),
            "спайков во втором шаге после снятия": nxt,
            "максимум спайков на нейрон за три шага после снятия":
                per_neuron,
        }
    rep["замеры"]["пункт 11: мембрана подавленного"] = mem
    m = mem["B"]
    check("подавленный нейрон копит потенциал, а не стоит на сбросе",
          m["v максимум перед снятием, мВ"] > 0.0,
          "у популяции B мембрана дошла до %.1f мВ против %.1f мВ без "
          "подавления и порога %.1f мВ"
          % (m["v максимум перед снятием, мВ"],
             m["v максимум без подавления, мВ"], -45.0))

    def one_spike(name, d, free_refrac):
        """Залп при снятии — это РОВНО ОДИН спайк, а не очередь.

        Проверяется три вещи, и каждая числом. Первое: спайков в первом
        шаге после снятия столько же, сколько нейронов стояло выше
        порога на конец предыдущего шага; разница на единицу-две бывает
        и предъявляется — порог проверяется после обновления мембраны, и
        кто-то переходит его уже на самом шаге снятия. Второе: залп
        кратно больше обычного уровня, иначе измерять было бы нечего.
        Третье: на следующем шаге активность вернулась к обычной, то
        есть накопленный потенциал ушёл в один спайк, а не в череду.

        Сколько спайков успевает дать ОДИН нейрон за три шага, зависит
        не от вмешательства, а от рефрактерности: у стимулируемой
        популяции она снята (так делает и upstream, чтобы пуассоновская
        частота воспроизводилась точно), и такой нейрон может выдать
        спайк и на следующем шаге. Поэтому у популяции с рефрактерностью
        ждём ровно один спайк, у популяции без неё — число предъявляем.
        """
        pop = d["нейронов в популяции"]
        extra = (d["спайков в первом шаге после снятия"]
                 - d["нейронов выше порога перед снятием"])
        base = max(d["обычно спайков на шаг"], 1e-9)
        ratio = d["спайков в первом шаге после снятия"] / base
        back = d["спайков во втором шаге после снятия"]
        per = d["максимум спайков на нейрон за три шага после снятия"]
        ok = (0 <= extra <= max(2, int(0.02 * pop))
              and ratio >= 10.0
              and back <= max(3.0 * base, 5.0)
              and (per == 1 or free_refrac))
        check("при снятии подавления популяция %s выдаёт ровно ОДИН "
              "спайк сразу, и это измерено" % name, ok,
              "в первом шаге после снятия спайков %d при %d нейронах, "
              "стоявших выше порога (из %d), то есть %+d перешло порог на "
              "самом этом шаге; в обычном прогоне на этом шаге %d, в "
              "среднем %.2f на шаг — залп в %.0f раз больше обычного; во "
              "втором шаге уже %d, то есть вернулось к обычному уровню, "
              "а не продолжилось; максимум спайков на один нейрон за три "
              "шага %d (%s)"
              % (d["спайков в первом шаге после снятия"],
                 d["нейронов выше порога перед снятием"], pop, extra,
                 d["спайков в первом шаге без подавления"], base, ratio,
                 back, per,
                 "рефрактерность снята, два подряд возможны и без "
                 "вмешательства" if free_refrac
                 else "рефрактерность на месте, больше одного быть не "
                      "может"))

    one_spike("B", m, free_refrac=False)
    one_spike("A (встречный замер на другой выборке)", mem["A"],
              free_refrac=True)

    note("flybrain/sim.py: в докстрингах `Engine._core` и "
         "`Engine.set_masks` хвост назван «до buf_len шагов (1,9 мс)». "
         "Для `out` это ровно так (%d шагов), а для `fire` измерено %d "
         "шагов (%.1f мс) — на шаг больше, потому что `fire` маскирует "
         "спайк в конце шага, а в кольцо он попадает на следующем. "
         "Описание занижает хвост подавления на один шаг."
         % (t_out, t_fire, t_fire * dt))
    note("flybrain/sim.py: мембрана подавленного нейрона ничем не "
         "ограничена сверху. Измерено: за %d шагов подавления v дошла до "
         "%.0f мВ у популяции B и %.0f мВ у A (порог −45 мВ, покой "
         "−52 мВ), и проводимость тоже не сбрасывается. Последствие "
         "документировано как «один спайк», но величина — нет: при "
         "снятии разом выдают спайк %d нейронов из %d в B и %d из %d в "
         "A, то есть синхронный залп в %.0f раз выше обычного уровня, "
         "которому в сети взяться неоткуда. Чем дольше подавление, тем "
         "выше залп, и предела этому в коде нет."
         % (m["подавление длилось шагов"],
            m["v максимум перед снятием, мВ"],
            mem["A"]["v максимум перед снятием, мВ"],
            m["спайков в первом шаге после снятия"],
            m["нейронов в популяции"],
            mem["A"]["спайков в первом шаге после снятия"],
            mem["A"]["нейронов в популяции"],
            m["спайков в первом шаге после снятия"]
            / max(m["обычно спайков на шаг"], 1e-9)))


# ------------------------------------------- слой вмешательств: BodyLive

def part_layer(ctx):
    """Ручка `lesion`: разбор, сроки, снятие и отказы."""
    from flybrain import body as Bo, live

    cfg = Bo.BodyConfig(duration_s=0.1, control_ms=20.0, seed=0,
                        device=ctx["dev"])
    sources = Bo.DEFAULT_SOURCES.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        sim_body, _ = Bo._build_body(cfg, sources)
        sim_body.reset(seed=0)
    eng = build(ctx)
    st = live.BodyLive()
    st.attach(sim_body, sources, cfg, ctx["neurons"], ctx["dev"], engine=eng)

    n_in, n_out = eng.fan()
    idx_a = ctx["A"]
    want_out = int(n_out[ctx["tA"]].sum())
    want_in = int(n_in[ctx["tA"]].sum())

    now = st.apply({"lesion": [
        {"query": Q_A, "kind": "out"},
        {"query": Q_B, "kind": "fire", "ms": 200.0},
    ]}, cmd_id="c1", t_ms=100.0)
    shown = now["lesion"]
    cnt = eng.mask_counts()
    check("ручка `lesion` поставила обе маски по запросам DSL",
          cnt["исходящие отключены"] == len(ctx["A"])
          and cnt["не выдают спайк"] == len(ctx["B"])
          and cnt["входящие отключены"] == 0,
          "mask_counts %s при |A| %d и |B| %d"
          % (cnt, len(ctx["A"]), len(ctx["B"])))
    check("маска встала на ТЕ ЖЕ индексы, что даёт выборка DSL",
          bool(np.array_equal(
              np.nonzero((eng.m_out[0] == 0).cpu().numpy())[0], idx_a)),
          "обнулённых столбцов %d, в выборке %d, совпали поэлементно"
          % (int((eng.m_out[0] == 0).sum()), len(idx_a)))
    syn = {w["вид"]: w["связей"] for w in shown}
    check("число связей считается по виду вмешательства, а не одним "
          "числом на все виды",
          syn["out"] == want_out and want_out != want_in,
          "для `out` на A показано %d связей — это исходящие (%d), а не "
          "входящие (%d); разница %d связей и была бы видна при путанице"
          % (syn["out"], want_out, want_in, abs(want_out - want_in)))
    check("подтверждение несёт модельное время и срок",
          shown[0]["с, мс"] == 100.0 and shown[0]["до, мс"] is None
          and shown[1]["до, мс"] == 300.0,
          "с %s мс; у бессрочного до %s, у срочного до %s мс"
          % (shown[0]["с, мс"], shown[0]["до, мс"], shown[1]["до, мс"]))

    # --- срок в МОДЕЛЬНОМ времени ---
    done_early = st.tick(250.0)
    cnt_early = eng.mask_counts()
    done_late = st.tick(300.0)
    cnt_late = eng.mask_counts()
    check("срок вмешательства снимается по модельному времени, не раньше",
          not done_early and cnt_early["не выдают спайк"] == len(ctx["B"])
          and len(done_late) == 1
          and cnt_late["не выдают спайк"] == 0,
          "на 250 мс снято %d, подавлено ещё %d нейронов; на 300 мс "
          "снято %d, подавлено %d"
          % (len(done_early), cnt_early["не выдают спайк"],
             len(done_late), cnt_late["не выдают спайк"]))
    check("снятие по сроку не задело бессрочное вмешательство "
          "(встречный замер)",
          cnt_late["исходящие отключены"] == len(ctx["A"]),
          "исходящие отключены у %d нейронов, как и было; снято только "
          "срочное" % cnt_late["исходящие отключены"])
    check("бессрочное не снимается никаким временем",
          not st.tick(1e9) and eng.mask_counts()["исходящие отключены"]
          == len(ctx["A"]),
          "на 1e9 мс модельного времени снято 0, маска на месте")

    # --- отказ не оставляет модель наполовину изменённой ---
    before = eng.mask_counts()
    refused = []
    for bad, why in (
            ({"lesion": [{"query": Q_A, "kind": "нет такого"}]}, "вид"),
            ({"lesion": [{"query": "", "kind": "out"}]}, "пустой запрос"),
            ({"lesion": [{"query": "type:нетакогонейрона", "kind": "out"}]},
             "запрос без нейронов"),
            ({"lesion": [{"query": Q_A, "kind": "out"}] * 9}, "слишком много"),
            ({"lesion": [{"query": Q_A, "kind": "out", "ms": -5.0}]},
             "срок вне предела"),
            ({"lesion": "не список"}, "не список"),
    ):
        try:
            st.apply(bad, cmd_id="bad", t_ms=400.0)
            refused.append((why, None))
        except (ValueError, TypeError) as exc:
            refused.append((why, str(exc)[:60]))
    after = eng.mask_counts()
    all_refused = all(msg is not None for _, msg in refused)
    check("негодная правка отклонена и маски не тронуты",
          all_refused and before == after,
          "отклонено %d из %d (%s); mask_counts до %s, после %s"
          % (sum(1 for _, m in refused if m), len(refused),
             ", ".join(w for w, _ in refused), before, after))

    # --- пустой список снимает всё ---
    st.apply({"lesion": []}, cmd_id="c2", t_ms=500.0)
    check("пустой список снимает все вмешательства",
          eng.masks_clear() and st.lesions_shown() == []
          and set(eng.mask_counts().values()) == {0},
          "masks_clear %s, в показе %d записей, mask_counts %s"
          % (eng.masks_clear(), len(st.lesions_shown()),
             eng.mask_counts()))

    # --- предел срока тоньше, чем граница, на которой он снимается ---
    #
    # Разбор принимает `ms` от 0,1, а снимается вмешательство только на
    # границе шага управления. Проверяется, во что превращается просьба
    # «на одну миллисекунду».
    st.apply({"lesion": [{"query": Q_B, "kind": "fire", "ms": 1.0}]},
             cmd_id="c5", t_ms=1000.0)
    until = st.lesions_shown()[0]["до, мс"]
    gone_at = next((t for t in (1000.0, 1020.0, 1040.0) if st.tick(t)), None)
    lived = None if gone_at is None else gone_at - 1000.0
    rep["замеры"]["срок тоньше шага управления"] = {
        "просили, мс": 1.0, "срок до, мс": until,
        "снято на границе, мс": gone_at, "продлилось, мс": lived,
    }
    if lived and lived > 1.0:
        note("flybrain/live.py: разбор `_lesions` принимает длительность "
             "от 0,1 мс (`_num(..., 0.1, MAX_LESION_MS)`), а снимается "
             "вмешательство только в `tick` на границе шага управления. "
             "Измерено: просили 1,0 мс (срок до %.1f мс) — продлилось "
             "%.0f мс, в %.0f раз дольше. Нижний предел в 200 раз тоньше "
             "достижимой точности, и панель принимает срок, который "
             "выдержать нечем. Сама точность в `tick` описана честно, "
             "предел разбора с ней не согласован."
             % (until, lived, lived / 1.0))

    # --- находка: снятие по сроку опознаётся в окне только по запросу ---
    st.apply({"lesion": [
        {"query": Q_B, "kind": "fire", "ms": 100.0},
        {"query": Q_B, "kind": "out"},
    ]}, cmd_id="c6", t_ms=2000.0)
    done = st.tick(2100.0)
    left = st.lesions_shown()
    if done and left:
        q_done = done[0]["query"]
        would_show = [w for w in left if w["запрос"] != q_done]
        rep["замеры"]["снятие по сроку при одинаковых запросах"] = {
            "снято": [(w["query"], w["kind"]) for w in done],
            "осталось в движке": eng.mask_counts(),
            "осталось бы в панели после фильтра по запросу":
                len(would_show),
        }
        if not would_show:
            note("flybrain/desk/ws_sandbox.py: снятие по сроку убирает из "
                 "панели ВСЕ вмешательства с тем же запросом — фильтр "
                 "сравнивает только «запрос» и игнорирует «вид», хотя "
                 "подтверждение несёт оба поля. Измерено: два "
                 "вмешательства на один запрос (%s: `fire` со сроком и "
                 "`out` бессрочно), срок вышел у одного — в движке "
                 "осталось %s, а в панели осталось бы %d записей. Окно "
                 "показывало бы «вмешательств нет» при действующей маске "
                 "на %d нейронах."
                 % (Q_B, eng.mask_counts(), len(would_show),
                    eng.mask_counts()["исходящие отключены"]))
    st.apply({"lesion": []}, cmd_id="c7", t_ms=2200.0)

    # --- три вида разом, и каждый на своём месте ---
    st.apply({"lesion": [
        {"query": Q_A, "kind": "fire"},
        {"query": Q_B, "kind": "out"},
        {"query": "class:gustatory+side:left", "kind": "in"},
    ]}, cmd_id="c3", t_ms=600.0)
    c3 = eng.mask_counts()
    g = len(ctx["Dsel"]("class:gustatory+side:left"))
    check("три вида встают в три РАЗНЫЕ маски",
          c3["не выдают спайк"] == len(ctx["A"])
          and c3["исходящие отключены"] == len(ctx["B"])
          and c3["входящие отключены"] == g,
          "mask_counts %s при |A| %d, |B| %d, |вкус| %d"
          % (c3, len(ctx["A"]), len(ctx["B"]), g))
    rep["замеры"]["слой вмешательств"] = {
        "показ действующих": st.lesions_shown(),
        "mask_counts": c3,
        "отклонённых правок": sum(1 for _, m in refused if m),
    }

    # --- находка: повторная привязка не чистит список ---
    eng2 = build(ctx)
    st.attach(sim_body, sources, cfg, ctx["neurons"], ctx["dev"], engine=eng2)
    stale = len(st.lesions_shown())
    fresh = eng2.masks_clear()
    if stale and fresh:
        note("flybrain/live.py: `BodyLive.attach` не чистит `self.lesions` "
             "и не ставит маски заново. После повторной привязки к новому "
             "движку показ говорит «действующих вмешательств: %d», а в "
             "движке маски пусты (masks_clear %s) — окно и журнал "
             "показывали бы то, чего модель не считает. Сегодня это "
             "латентно: `jobs.py` создаёт `BodyLive` на каждый сеанс "
             "заново, так что повторная привязка в коде не встречается."
             % (stale, fresh))
    rep["замеры"]["повторная привязка"] = {
        "в показе осталось вмешательств": stale,
        "маски нового движка пусты": bool(fresh),
    }
    st.apply({"lesion": []}, cmd_id="c4", t_ms=700.0)
    del eng, eng2, st, sim_body
    gc.collect()


# -------------------------------------------- 6: сеансы и 7: ресурсы

def episode(ctx, seed, lesion=None, ms=None, duration=0.6, attach=True):
    """Один эпизод в теле. `lesion` — запрос DSL или None.

    Живые условия привязываются и к чистым эпизодам тоже: иначе чистый и
    поражённый эпизоды отличались бы ещё и тем, был ли привязан слой, и
    сравнение их отпечатков ничего не говорило бы о вмешательстве.
    """
    from flybrain import body as Bo, live

    cfg = Bo.BodyConfig(duration_s=duration, control_ms=20.0, seed=seed,
                        device=ctx["dev"], readout="ALPN")
    st = live.BodyLive() if attach else None
    got = {"правка": None, "снято": [], "маски": []}

    def on_step(step, rec, bins_row):
        if lesion is not None and step == 3:
            got["правка"] = st.apply(
                {"lesion": [{"query": lesion, "kind": "fire", "ms": ms}]},
                cmd_id="les-%d" % seed, t_ms=rec["t_ms"])
            got["маски"].append((rec["t_ms"], st.engine.mask_counts()))
        # Срок снимается на границе шага управления — там же, где это
        # делает работник (`jobs.control_point`).
        done = st.tick(rec["t_ms"])
        if done:
            got["снято"].append((rec["t_ms"], len(done),
                                 st.engine.mask_counts()))

    t0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run = Bo.run_episode(cfg=cfg, neurons=ctx["neurons"],
                             weights=ctx["W"], progress=False, live=st,
                             on_step=(on_step if st is not None else None))
    cols = ("x", "y", "theta", "odour_l", "odour_r", "dn_l", "dn_r",
            "amp_l", "amp_r", "dist", "grad_cos", "spikes")
    tab = np.array([[r[c] for c in cols] for r in run.records],
                   dtype=np.float64)
    out = {
        "по стене, с": round(time.perf_counter() - t0, 1),
        "шагов": len(run.records),
        "траектория": fp(tab[:, :3]),
        "моторные": fp(tab[:, 5:9]),
        "всё по шагам": fp(tab),
        "активность": fp(run.counts),
        "растр": fp(run.bins),
        "спайков всего": int(run.counts.sum()),
        "метрики": dict(run.metrics),
        "событие": got,
    }
    del run, st
    gc.collect()
    return out


def part_sessions(ctx):
    """Пункт 6 на настоящих эпизодах тела, с отпечатками."""
    ref = episode(ctx, 0)
    stage("     эпизод 1: чистый, %d шагов, %.0f с по стене"
          % (ref["шагов"], ref["по стене, с"]))
    les = episode(ctx, 0, lesion=Q_B)
    stage("     эпизод 2: с подавлением %s до конца эпизода" % Q_B)
    again = episode(ctx, 0)
    stage("     эпизод 3: снова чистый, тот же процесс и то же зерно")
    expiring = episode(ctx, 0, lesion=Q_B, ms=200.0)
    stage("     эпизод 4: подавление на 200 мс модельного времени")
    bare = episode(ctx, 0, attach=False)
    stage("     эпизод 5: чистый и вовсе без привязки живых условий")

    keys = ("траектория", "моторные", "всё по шагам", "активность", "растр")
    check("привязанный, но не тронутый слой вмешательств не меняет "
          "эпизод ни на бит",
          all(ref[k] == bare[k] for k in keys)
          and ref["спайков всего"] == bare["спайков всего"],
          "все %d отпечатков совпали с эпизодом без слоя; спайков %d и "
          "%d — значит расхождение поражённого эпизода не от самой "
          "привязки" % (len(keys), ref["спайков всего"],
                        bare["спайков всего"]))
    same = all(ref[k] == again[k] for k in keys)
    diff = [k for k in keys if ref[k] != les[k]]
    check("вмешательство из прошлого сеанса не перешло в следующий",
          same and ref["спайков всего"] == again["спайков всего"],
          "все %d отпечатков третьего эпизода совпали с первым "
          "побайтово (траектория %s, активность %s), спайков %d и %d"
          % (len(keys), again["траектория"], again["активность"],
             ref["спайков всего"], again["спайков всего"]))
    check("второй эпизод был действительно поражён (встречный замер)",
          len(diff) == len(keys),
          "разошлись все %d отпечатков: спайков %d против %d (на %d "
          "меньше, %.1f%%), моторный выход %s против %s"
          % (len(diff), les["спайков всего"], ref["спайков всего"],
             ref["спайков всего"] - les["спайков всего"],
             100.0 * (ref["спайков всего"] - les["спайков всего"])
             / ref["спайков всего"], les["моторные"], ref["моторные"]))
    ev = expiring["событие"]
    off = ev["снято"]
    check("срок вмешательства вышел посреди эпизода, и дальше эпизод "
          "шёл чистым",
          bool(off) and off[0][2]["не выдают спайк"] == 0
          and expiring["всё по шагам"] not in (ref["всё по шагам"],
                                              les["всё по шагам"]),
          "подавление с %.0f мс (масок %s), снято на %.0f мс (масок %s); "
          "отпечаток эпизода отличается и от чистого, и от поражённого "
          "до конца: спайков %d против %d чистого и %d поражённого"
          % (ev["маски"][0][0], ev["маски"][0][1]["не выдают спайк"],
             off[0][0], off[0][2]["не выдают спайк"],
             expiring["спайков всего"], ref["спайков всего"],
             les["спайков всего"]))
    rep["замеры"]["пункт 6"] = {
        "эпизод 1 чистый": {k: ref[k] for k in keys},
        "эпизод 2 поражённый": {k: les[k] for k in keys},
        "эпизод 3 чистый после поражённого": {k: again[k] for k in keys},
        "спайков: чисто / поражён / со сроком":
            [ref["спайков всего"], les["спайков всего"],
             expiring["спайков всего"]],
        "длительность эпизода, с": 0.6,
    }
    return ref, les


def part_resources(ctx):
    """Пункт 7: отмена и аварийное завершение освобождают ресурсы."""
    import torch

    from flybrain import body as Bo

    def mem():
        gc.collect()
        if ctx["dev"] == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return int(torch.cuda.memory_allocated())
        return 0

    base = mem()
    held = []
    rest = []
    for i in range(3):
        eng = build(ctx)
        eng.set_masks(fire=ctx["B"], out=ctx["A"], inc=ctx["B"])
        r = rates_of(ctx)
        for _ in range(40):
            eng.step(r)
        held.append(mem() - base)
        del eng, r
        rest.append(mem() - base)
    check("отмена освобождает память движка вместе с масками и графами",
          rest == [0, 0, 0] and held[0] > 0,
          "живой движок с графами держит %.1f МБ, после отмены остаток "
          "%s байт — и он не растёт от сеанса к сеансу"
          % (held[0] / 1e6, rest))

    # --- аварийное завершение: исключение посреди шагов ---
    boom = None
    before = mem()
    try:
        eng = build(ctx)
        eng.set_masks(fire=ctx["B"])
        r = rates_of(ctx)
        for s in range(40):
            eng.step(r)
            if s == 20:
                raise RuntimeError("аварийное завершение для пробы")
    except RuntimeError as exc:
        boom = str(exc)
    finally:
        eng = r = None
    after = mem()
    fresh = build(ctx)
    clear = fresh.masks_clear()
    cnt = fresh.mask_counts()
    del fresh
    gc.collect()
    check("аварийное завершение освобождает память и не оставляет масок",
          boom is not None and after - before == 0 and clear
          and set(cnt.values()) == {0},
          "исключение «%s» поймано на 21-м шаге; остаток памяти %d байт; "
          "у следующего движка masks_clear %s, mask_counts %s"
          % (boom, after - before, clear, cnt))

    # --- отмена эпизода на полпути ---
    # Отмена именно ПОСРЕДИ эпизода, а не на разогреве: `should_stop`
    # опрашивается и в разогреве, и глупая отмена «на шестом опросе»
    # обрывала бы сеанс до того, как он начался, и ничего не проверяла.
    seen = {"n": 0}
    before = mem()
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        cfg = Bo.BodyConfig(duration_s=1.0, control_ms=20.0, seed=0,
                            device=ctx["dev"], readout="ALPN")
        run = Bo.run_episode(
            cfg=cfg, neurons=ctx["neurons"], weights=ctx["W"],
            progress=False,
            on_step=lambda step, rec, row: seen.__setitem__("n", step + 1),
            should_stop=lambda: seen["n"] >= 10)
    wall = time.perf_counter() - t0
    steps = len(run.records)
    warmed = bool(run.calib)
    del run
    after = mem()
    check("отмена посреди эпизода доводит его до конца и всё освобождает",
          10 <= steps < 50 and warmed and after - before == 0,
          "из 50 шагов посчитано %d за %.0f с (разогрев при этом прошёл "
          "целиком: калибровка снята), остаток памяти %d байт"
          % (steps, wall, after - before))
    rep["замеры"]["пункт 7"] = {
        "движок держит, МБ": round(held[0] / 1e6, 2),
        "остатки после трёх отмен, байт": rest,
        "остаток после аварии, байт": 0 if boom else None,
        "шагов до отмены эпизода": steps,
    }


# ----------------------------------------------- научное сравнение поведения

def part_science(ctx, ref0, les0):
    """Поведение с вмешательством и без — на трёх зёрнах, с разбросом."""
    data = {"чисто": {0: ref0["метрики"]}, "подавление": {0: les0["метрики"]}}
    for seed in (1, 2):
        data["чисто"][seed] = episode(ctx, seed)["метрики"]
        data["подавление"][seed] = episode(ctx, seed, lesion=Q_B)["метрики"]
        stage("     зерно %d: оба условия посчитаны" % seed)

    fields = ("dist_end_mm", "path_mm", "grad_alignment", "mean_dn_hz")
    tabl = {}
    verdict = []
    for f in fields:
        a = [float(data["чисто"][s].get(f, np.nan)) for s in (0, 1, 2)]
        b = [float(data["подавление"][s].get(f, np.nan)) for s in (0, 1, 2)]
        spread = max(np.ptp(a), np.ptp(b))
        gap = abs(float(np.mean(a)) - float(np.mean(b)))
        tabl[f] = {
            "чисто": [round(v, 3) for v in a],
            "подавление": [round(v, 3) for v in b],
            "разброс внутри условия": round(float(spread), 3),
            "разница между условиями": round(float(gap), 3),
            "разница больше разброса": bool(gap > spread),
        }
        verdict.append(gap > spread)
        print("     %-15s чисто %s | подавление %s | разброс %.3f, "
              "разница %.3f -> %s"
              % (f, [round(v, 2) for v in a], [round(v, 2) for v in b],
                 spread, gap, "больше разброса" if gap > spread
                 else "В ПРЕДЕЛАХ РАЗБРОСА"))
    rep["замеры"]["научное сравнение"] = tabl
    spikes = rep["замеры"].get("пункт 6", {}).get(
        "спайков: чисто / поражён / со сроком", [0, 0, 0])
    weak = [f for f, v in zip(fields, verdict) if not v]
    words = (
        "Три зерна на условие, эпизод 0,6 с модельного времени, 30 шагов "
        "управления. Что утверждается прямо: подавленная популяция дала "
        "НОЛЬ спайков за всё время действия маски (пункт 3 выше, со "
        "встречным замером того же окна без подавления), а всего по "
        "мозгу спайков стало %d против %d (−%.0f%%). Это измерение, а не "
        "вывод: маска стоит на той самой популяции, и считать нечему "
        "больше. "
        "Что НЕ утверждается: причинный ПОВЕДЕНЧЕСКИЙ эффект. По %d "
        "показателям из %d (%s) разница средних между условиями меньше "
        "разброса внутри условия, то есть на трёх зёрнах эти показатели "
        "не отличают условия вовсе. Три эпизода на условие — три "
        "наблюдения, по ним дисперсию не оценить, и ни один из %d "
        "показателей не стоит объявлять эффектом. Отдельно и прямо: ОДНА "
        "пара эпизодов «с вмешательством и без» не доказывает ничего — "
        "при любом другом зерне траектории расходятся сами по себе, и "
        "здесь это видно числом: разброс чистого условия по конечному "
        "расстоянию %.2f мм при разнице между условиями %.2f мм."
        % (spikes[1], spikes[0],
           100.0 * (spikes[0] - spikes[1]) / max(spikes[0], 1),
           len(weak), len(fields), ", ".join(weak), len(fields),
           float(np.ptp(tabl["dist_end_mm"]["чисто"])),
           tabl["dist_end_mm"]["разница между условиями"]))
    rep["замеры"]["научное сравнение, оговорка"] = words
    print("     " + words.replace(". ", ".\n     "))

    check("поведенческое сравнение сделано не на одной паре эпизодов",
          len(data["чисто"]) == 3 and len(data["подавление"]) == 3,
          "зёрен на условие %d, эпизодов всего %d; разброс внутри условия "
          "посчитан и сравнён с разницей между условиями по %d "
          "показателям" % (len(data["чисто"]),
                           len(data["чисто"]) + len(data["подавление"]),
                           len(fields)))
    check("у каждого показателя поведения разброс и разница названы "
          "числом, а не словом",
          all(np.isfinite(v) for f in fields
              for v in (tabl[f]["разброс внутри условия"],
                        tabl[f]["разница между условиями"]))
          and all(tabl[f]["разброс внутри условия"] > 0.0 for f in fields),
          "по %d показателям разброс внутри условия от %.3f до %.3f, "
          "разница между условиями от %.3f до %.3f; за разброс вышел %d "
          "показатель (%s), остальные %d — нет"
          % (len(fields),
             min(tabl[f]["разброс внутри условия"] for f in fields),
             max(tabl[f]["разброс внутри условия"] for f in fields),
             min(tabl[f]["разница между условиями"] for f in fields),
             max(tabl[f]["разница между условиями"] for f in fields),
             len(fields) - len(weak),
             ", ".join(f for f, v in zip(fields, verdict) if v) or "ни один",
             len(weak)))
    check("разница активности нейронов названа числом и она прямая "
          "(встречный замер к поведению)",
          spikes[0] > 0 and spikes[1] < spikes[0],
          "спайков по мозгу %d без вмешательства и %d с ним, то есть на "
          "%.0f%% меньше — это измерение одного эпизода и ему хватает "
          "одного зерна, в отличие от любого поведенческого показателя "
          "выше" % (spikes[0], spikes[1],
                    100.0 * (spikes[0] - spikes[1]) / max(spikes[0], 1)))


# ----------------------------------------------------------------- сборка

def collect_traces(ctx):
    A, B = ctx["A"], ctx["B"]
    plan = {
        "чисто": {},
        "out на A": {K_ON: masks(out=A), K_OFF: masks()},
        "out на A (держим)": {K_ON: masks(out=A)},
        "out на B": {K_ON: masks(out=B)},
        "in на A": {K_ON: masks(inc=A)},
        "in на B": {K_ON: masks(inc=B), K_OFF: masks()},
        "in на B (держим)": {K_ON: masks(inc=B)},
        "fire на A": {K_ON: masks(fire=A), K_OFF: masks()},
        "fire на A (держим)": {K_ON: masks(fire=A)},
        "fire на B": {K_ON: masks(fire=B), K_OFF: masks()},
    }
    out = {}
    t0 = time.perf_counter()
    for name, sched in plan.items():
        out[name] = trace(ctx, sched)
    stage("     %d трассировок по %d шагов за %.1f с"
          % (len(out), STEPS, time.perf_counter() - t0))
    return out


def context():
    import torch

    from flybrain import data as D

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    W = D.load_weights(device=dev)
    n = len(neurons)
    A = D.select(neurons, Q_A)
    B = D.select(neurons, Q_B)

    # Кто кому посылает — считается по матрице, а не предполагается.
    w = W.coalesce()
    idx = w.indices().cpu().numpy()
    val = w.values().cpu().numpy()
    post, pre = idx[0], idx[1]
    mA = np.zeros(n, bool)
    mA[A] = True
    mB = np.zeros(n, bool)
    mB[B] = True
    blocks = {}
    for nm, src, dst in (("A->A", mA, mA), ("A->B", mA, mB),
                         ("B->A", mB, mA), ("B->B", mB, mB)):
        sel = dst[post] & src[pre]
        blocks["%s, связей" % nm] = int(sel.sum())
        blocks["%s, вес" % nm] = round(float(val[sel].sum()), 1)
    n_syn = int(len(val))
    del w, idx, val, post, pre
    gc.collect()

    # Вектор-пробник отпечатка растра: постоянный и не зависящий от
    # прогона, иначе отпечатки нельзя сравнивать между прогонами.
    pvec = torch.arange(1, n + 1, dtype=torch.float32, device=dev)
    return {
        "dev": dev, "neurons": neurons, "W": W, "n": n,
        "A": A, "B": B,
        "tA": torch.as_tensor(A, device=dev),
        "tB": torch.as_tensor(B, device=dev),
        "pvec": pvec,
        "блоки": blocks,
        "связей": n_syn,
        "пересечение": int(len(np.intersect1d(A, B))),
        "Dsel": lambda q: D.select(neurons, q),
    }


def guarded(name, fn, *args):
    try:
        return fn(*args)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        rep["сорвалось"].append({"часть": name, "след": tb[-1500:]})
        check("часть «%s» дошла до конца" % name, False,
              "сорвалась: %s" % tb.strip().splitlines()[-1][:120])
        return None


def main():
    t_all = time.perf_counter()
    ctx = context()
    print("устройство: %s, нейронов %d, связей %d"
          % (ctx["dev"], ctx["n"], ctx["связей"]))
    rep["замеры"]["устройство"] = ctx["dev"]
    rep["замеры"]["нейронов"] = ctx["n"]
    rep["замеры"]["связей"] = ctx["связей"]

    stage("— отпечатки голого движка: 1, 2, 5, 8 —")
    guarded("отпечатки", part_equal, ctx)

    stage("— пошаговые трассировки —")
    tr = guarded("трассировки", collect_traces, ctx)
    if tr is not None:
        stage("— подавление и направление: 3, 4 —")
        guarded("направление", part_direction, ctx, tr)
        stage("— хвост задержек и мембрана: 9, 10, 11 —")
        guarded("хвост", part_tail, ctx, tr)
        del tr
        gc.collect()

    stage("— слой вмешательств: ручка `lesion` —")
    guarded("слой", part_layer, ctx)

    stage("— ресурсы: 7 —")
    guarded("ресурсы", part_resources, ctx)

    stage("— сеансы в теле: 6 —")
    pair = guarded("сеансы", part_sessions, ctx)

    if pair is not None:
        stage("— научное сравнение поведения: три зерна на условие —")
        guarded("наука", part_science, ctx, pair[0], pair[1])

    rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t_all, 1)
    (OUT / "lesion.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nотчёт: %s" % (OUT / "lesion.json"))
    print("проверок %d, провалов %d, находок %d, за %.0f с"
          % (len(rep["проверки"]), len(fails), len(rep["находки"]),
             rep["замеры"]["проба заняла, с"]))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
