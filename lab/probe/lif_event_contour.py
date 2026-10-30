"""Событийное умножение в НАСТОЯЩЕМ контуре, попарно в одном процессе.

Зачем не микробенчмарк. Та же работа в одиночку и в контуре стоит
разного **в 4,4 раза**: блок движка около 60 мс отдельно против 264,97
мс в шаге управления (`LIF_PERF.md`). Поэтому ядро, которое быстрее
только рядом с контуром, отклоняется, а решать обязан замер внутри
`body.run_episode` — с телом NeuroMechFly в MuJoCo, с физикой, с
измерителем и с записью.

Как мерится. Два пути умножения связей (`sim.SPMV_CSR` и
`sim.SPMV_EVENT`) — это одна и та же сборка пакета, выбор одним полем
`BodyConfig.spmv`. Поэтому эпизоды гоняются ВПЕРЕМЕЖКУ в одном
процессе, и отношение считается ВНУТРИ пары: оба эпизода видят одну и ту
же постороннюю нагрузку, потому что стоят рядом во времени. В итог идёт
медиана отношений, размах и число пар. Абсолютные миллисекунды здесь
печатаются только для разбора — их мерит владелец на спокойной машине.

Три величины, и обвязка у них та же, что у пробы владельца
(`probe/lifperf_contour.py`):

* **мозг (LIF + синхронизация + учёт)** — от конца предыдущего шага
  управления до первого шага физики. Это та часть шага, в которой живёт
  умножение связей, и это ЕДИНСТВЕННАЯ величина, по которой правка
  может выиграть;
* **шаг управления целиком** — официальное число стенда. Выигрыш здесь
  обязан быть меньше: мозг занимает около четверти шага, остальное —
  физика и контроллер походки, которых правка не касается;
* **физика на ОДИН шаг физики** (их 100 на шаг управления при
  `control_ms` 20 и `physics_dt` 2e-4) — ВСТРЕЧНЫЙ замер на фон. Правка её не
  трогает, значит отношение по ней обязано лечь около единицы. Если не
  легло — фон за время пары сдвинулся, и паре верить нельзя.

И встречный замер на саму оснастку: пара `csrmv` против `csrmv` (A/A).
Её медиана обязана лечь около единицы, а размах по парам и есть тот
порог, ниже которого «ускорение» от фона не отличить. Без этого числа
говорить о чьём-то ускорении бессмысленно.

Равнозначность проверяется здесь же, в тех же эпизодах: отпечатки
траектории, спайков и конечной позы сравниваются побайтово. Скорость без
равенства — не результат.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\lif_event_contour.py --pairs 5
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent

# Убирать ли движок прошлого эпизода перед следующим. По умолчанию да —
# см. `--грязно` и §5 отчёта: без уборки замер мерит не дорогу, а
# занятую память.
CLEAN = True


def episode(spmv, neurons, weights, device, t_episode):
    """Один эпизод настоящего контура. Мс мозговой части на шаг управления.

    Обвязка та же, что в `probe/lifperf_contour.py`: `body.step`
    обёрнут, чтобы знать, когда начался первый шаг физики, и мозговая
    часть считается от конца предыдущего шага управления до него.
    """
    from flybrain import body as B

    st = {"prev_end": None, "phys_first": None, "rows": [], "tail": [],
          "phys_tot": 0.0, "phys_n": 0}

    def on_body(body):
        orig = body.step

        def phys(action):
            t0 = time.perf_counter()
            if st["phys_first"] is None:
                st["phys_first"] = t0
            out = orig(action)
            st["phys_tot"] += (time.perf_counter() - t0) * 1000.0
            st["phys_n"] += 1
            return out

        body.step = phys

    def on_step(step, rec, row):
        now = time.perf_counter()
        if st["prev_end"] is not None and st["phys_first"] is not None:
            st["rows"].append((now - st["prev_end"]) * 1000.0)
            st["tail"].append((st["phys_first"] - st["prev_end"]) * 1000.0)
        st["prev_end"] = time.perf_counter()
        st["phys_first"] = None

    cfg = B.BodyConfig(duration_s=t_episode, device=device, seed=0,
                       spmv=spmv)
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False, on_step=on_step, on_body=on_body)
    recs = run.records
    got = {
        "мозг, мс": float(statistics.median(st["tail"])),
        "шаг управления, мс": float(statistics.median(st["rows"])),
        "физика, мс на шаг физики": st["phys_tot"] / max(st["phys_n"], 1),
        "шагов в замере": len(st["rows"]),
        "умножение": (run.timing or {}).get("умножение связей"),
        # Память видеокарты ПОСЛЕ эпизода. Печатается не для порядка:
        # каждый эпизод строит свой движок, и подозрение, что прежний
        # остаётся жив вместе со своей матрицей и девятнадцатью пулами
        # графов, надо было снять замером. Снято: с уборкой занято
        # держится на 288,3 МБ от эпизода к эпизоду, а время при этом
        # всё равно гуляет в 4,4 раза — значит гуляет не память, а
        # посторонняя нагрузка на видеокарту.
        "память занято, МБ": round(
            __import__("torch").cuda.memory_allocated() / 1048576.0, 1),
        "память зарезервировано, МБ": round(
            __import__("torch").cuda.memory_reserved() / 1048576.0, 1),
        "отпечатки": {
            "траектория": [round(float(r["x"]), 6) for r in recs],
            "курс": [round(float(r["theta"]), 6) for r in recs],
            "спайков всего": int(run.counts.sum()),
            "растр": int(run.bins.sum()),
            "спайков на последнем шаге": int(recs[-1]["spikes"]),
            "dn": [round(float(recs[-1]["dn_l"]), 6),
                   round(float(recs[-1]["dn_r"]), 6)],
            "конечная поза": [round(float(recs[-1][k]), 6)
                              for k in ("x", "y", "theta")],
        },
    }
    if CLEAN:
        # Уборка между эпизодами. Нужна не для памяти, а чтобы память
        # вообще перестала быть подозреваемым: без неё движок прошлого
        # эпизода мог бы остаться жив (его держали бы возвращённый
        # объект прогона и ссылки замыканий), и объяснять разброс
        # времени было бы нечем. С уборкой занято одинаково от эпизода
        # к эпизоду, и разброс остался — значит дело не в памяти.
        import gc

        import torch

        del run, recs
        gc.collect()
        torch.cuda.empty_cache()
        got["память после уборки, МБ"] = round(
            torch.cuda.memory_allocated() / 1048576.0, 1)
    return got


def sweep(tag, rounds=9):
    """Где событийное становится ДОРОЖЕ `csrmv`. Это микробенчмарк.

    Названо так прямо, потому что решающим это число быть не может:
    та же работа в контуре стоит в 4,4 раза дороже. Но граница нужна, и
    лучше назвать её числом, чем оставить на удивление: событийное
    умножение тем дороже, чем больше нейронов спайкует, и при полном
    залпе всей сети оно обязано проиграть.

    Замер попарный: в одном круге подряд оба варианта, отношение внутри
    круга, в итог — медиана с размахом.
    """
    import torch

    from flybrain import data as D, sim

    device = "cuda"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    n = len(neurons)
    eng = sim.build_engine(weights=weights, neurons=neurons, device=device,
                           seed=0, spmv=sim.SPMV_EVENT, graphs=False)
    csr, ev = eng.w_csr, eng._ev
    deg = (ev.crow[1:] - ev.crow[:-1]).to(torch.int64)
    nnz = int(deg.sum())
    ones = torch.ones(1, n, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    perm = torch.randperm(n, device=device, generator=gen)

    rep = {"метка": tag, "это": "микробенчмарк, не решающее число",
           "связей всего": nnz, "развёртка": {}}
    print("развёртка по числу спайкующих (МИКРОБЕНЧМАРК, не решающее "
          "число)", flush=True)
    for k in (28, 103, 1000, 10000, 50000, n):
        sp = torch.zeros(1, n, device=device)
        sp[0, perm[:k]] = 1.0
        col = sp.t().contiguous()
        touched = int(deg[perm[:k]].sum())
        a_ref = torch.sparse.mm(csr, col).t().reshape(-1)
        a_got = ev.mm(sp, ones).reshape(-1)
        same = bool(np.array_equal(a_ref.cpu().numpy().view(np.uint8),
                                   a_got.cpu().numpy().view(np.uint8)))
        ra, rb, rr = [], [], []
        for _ in range(2):
            torch.sparse.mm(csr, col)
            ev.mm(sp, ones)
        torch.cuda.synchronize()
        for _ in range(rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                torch.sparse.mm(csr, col)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(20):
                ev.mm(sp, ones)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            ra.append((t1 - t0) * 50.0)
            rb.append((t2 - t1) * 50.0)
            rr.append((t1 - t0) / (t2 - t1))
        got = {"связей задето": touched,
               "доля связей": round(touched / nnz, 5),
               "побитово то же": same,
               "csrmv, мс": round(statistics.median(ra), 4),
               "событийно, мс": round(statistics.median(rb), 4),
               "медиана отношения": round(statistics.median(rr), 4),
               "мин": round(min(rr), 4), "макс": round(max(rr), 4),
               "кругов": rounds}
        rep["развёртка"][k] = got
        print("  спайкует %6d (связей %9d, %6.2f %%): в %.2f раза "
              "(%.2f…%.2f, %d кругов), побитово %s"
              % (k, touched, 100.0 * touched / nnz,
                 got["медиана отношения"], got["мин"], got["макс"],
                 rounds, same), flush=True)
    path = OUT / ("lif_event_sweep_%s.json" % tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("отчёт:", path)
    return 0


def ratios(rows, key):
    """Отношения a/b внутри пар. Медиана, размах, число пар."""
    rr = [a[key] / b[key] for a, b in rows if b[key]]
    if not rr:
        return None
    return {
        "медиана": round(statistics.median(rr), 4),
        "мин": round(min(rr), 4), "макс": round(max(rr), 4),
        "пар": len(rr),
        "a, мс (медиана по парам)": round(
            statistics.median([a[key] for a, b in rows]), 3),
        "b, мс (медиана по парам)": round(
            statistics.median([b[key] for a, b in rows]), 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=5, metavar="КРУГОВ",
                    help="кругов по три эпизода: A(csrmv), B(событийно), "
                         "A(csrmv). Порог доверия снимается в том же "
                         "круге, а не отдельной серией")
    ap.add_argument("--duration", type=float, default=0.4,
                    help="модельного времени на эпизод, с")
    ap.add_argument("--tag", default="сейчас")
    ap.add_argument("--seq", default=None, metavar="AABAB",
                    help="диагностика: последовательность эпизодов буквами "
                         "(A — эталонный, B — событийный). Печатает время и "
                         "память по каждому, без отношений. Нужна, чтобы "
                         "отличить разницу дорог от переноса состояния "
                         "между эпизодами")
    ap.add_argument("--грязно", dest="dirty", action="store_true",
                    help="НЕ убирать движок прошлого эпизода: встречный "
                         "замер на саму уборку")
    ap.add_argument("--sweep", action="store_true",
                    help="только развёртка по числу спайкующих: где "
                         "событийное становится дороже csrmv. Это "
                         "МИКРОБЕНЧМАРК, решающим числом быть не может")
    args = ap.parse_args()

    import torch

    from flybrain import data as D, sim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("событийное умножение считает только на CUDA")
        return 2
    global CLEAN
    CLEAN = not args.dirty
    if args.sweep:
        return sweep(args.tag)

    if args.seq:
        # Диагностика переноса состояния между эпизодами. Без неё
        # отношение внутри пары можно принять за разницу дорог, когда
        # это разница «чистый процесс / процесс с живым прошлым
        # движком».
        print("читаю коннектом...", flush=True)
        neurons = D.load_neurons()
        weights = D.load_weights(device=device)
        letters = {"A": sim.SPMV_CSR, "B": sim.SPMV_EVENT}
        rows = []
        print("уборка между эпизодами: %s"
              % ("да" if CLEAN else "НЕТ (грязно)"), flush=True)
        for i, ch in enumerate(args.seq.upper()):
            if ch not in letters:
                raise SystemExit("буква %r не A и не B" % ch)
            got = episode(letters[ch], neurons, weights, device,
                          args.duration)
            rows.append({"№": i + 1, "дорога": ch, **{
                k: v for k, v in got.items() if k != "отпечатки"}})
            print("  %2d %s(%-10s) мозг %7.2f мс, шаг %7.2f мс, "
                  "занято %7.1f МБ, зарезервировано %7.1f МБ%s"
                  % (i + 1, ch, got["умножение"], got["мозг, мс"],
                     got["шаг управления, мс"], got["память занято, МБ"],
                     got["память зарезервировано, МБ"],
                     ("", " -> после уборки %.1f МБ"
                      % got["память после уборки, МБ"])[CLEAN]),
                  flush=True)
        path = OUT / ("lif_event_seq_%s.json" % args.tag)
        path.write_text(json.dumps(
            {"последовательность": args.seq, "уборка": CLEAN,
             "эпизоды": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print("отчёт:", path)
        return 0

    print("читаю коннектом...", flush=True)
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    rep = {"метка": args.tag, "устройство": device,
           "модельного времени на эпизод, с": args.duration,
           "шагов управления в эпизоде": int(round(
               args.duration * 1000.0 / 20.0))}

    # Холодный эпизод каждой дорогой: сборка тела, прогрев библиотек и
    # захват графов не должны попасть в пары.
    print("холодные эпизоды (в пары не идут)...", flush=True)
    cold = {}
    for spmv in (sim.SPMV_CSR, sim.SPMV_EVENT):
        got = episode(spmv, neurons, weights, device, args.duration)
        cold[spmv] = got
        print("  %-12s мозг %7.2f мс, шаг %7.2f мс, физика %6.2f мс"
              % (spmv, got["мозг, мс"], got["шаг управления, мс"],
                 got["физика, мс на шаг физики"]), flush=True)
    rep["холодные"] = {k: {kk: vv for kk, vv in v.items()
                           if kk != "отпечатки"} for k, v in cold.items()}

    # --- равнозначность: те же эпизоды, побайтово ---
    same = (cold[sim.SPMV_CSR]["отпечатки"]
            == cold[sim.SPMV_EVENT]["отпечатки"])
    diff = [k for k in cold[sim.SPMV_CSR]["отпечатки"]
            if cold[sim.SPMV_CSR]["отпечатки"][k]
            != cold[sim.SPMV_EVENT]["отпечатки"][k]]
    rep["равнозначность"] = {
        "эпизоды совпали побайтово": bool(same),
        "разошлось": diff,
        "отпечатки": cold[sim.SPMV_CSR]["отпечатки"],
    }
    print("\nравнозначность эпизодов: %s%s"
          % ("побайтово то же" if same else "РАЗОШЛИСЬ",
             "" if same else " по полям %s" % diff), flush=True)

    # --- круги по ТРИ эпизода: A, B, A ---
    #
    # Почему тройками, а не парами A/B и отдельно парами A/A. Замер
    # показал, что эти две дороги по-разному чувствительны к
    # посторонней нагрузке на видеокарту: у `csrmv` мозговая часть
    # гуляла в 4,4 раза между эпизодами одной и той же сборки (60,16 и
    # 262,01 мс), у событийной — в 1,13 (42,40 и 47,74). Это объяснимо:
    # `csrmv` читает 182 МБ на каждый тик 0,1 мс, то есть дерётся за
    # полосу памяти со всем, что ещё считает на этой карте, а событийное
    # ядро читает связи только спайкующих и почти ничего не читает.
    #
    # Отсюда следствие, которое ломает обычную оснастку: пара защищает
    # от ОБЩЕЙ нагрузки, а здесь нагрузка действует на два варианта
    # по-разному, и само отношение зависит от окна. Поэтому порог
    # доверия надо снимать В ТОМ ЖЕ круге, а не отдельной серией в
    # конце: прошлый замер снял пары A/A последними, окно к тому
    # времени стало тише, и порог оказался из другого мира, чем
    # отношения.
    #
    # Круг: A1, B, A2. Отношение берётся к среднему геометрическому двух
    # A — так B стоит между ними во времени; порог того же круга — A1/A2.
    t_start = time.perf_counter()
    print("\nкруги A(csrmv), B(событийно), A(csrmv) — %d" % args.pairs,
          flush=True)
    rows, aa, triples = [], [], []
    for i in range(args.pairs):
        a1 = episode(sim.SPMV_CSR, neurons, weights, device, args.duration)
        b = episode(sim.SPMV_EVENT, neurons, weights, device, args.duration)
        a2 = episode(sim.SPMV_CSR, neurons, weights, device, args.duration)
        mid = {k: ((a1[k] * a2[k]) ** 0.5 if isinstance(a1[k], float)
                   else a1[k]) for k in a1 if k != "отпечатки"}
        mid["отпечатки"] = a1["отпечатки"]
        rows.append((mid, b))
        aa.append((a1, a2))
        triples.append({name: {k: v for k, v in got.items()
                               if k != "отпечатки"}
                        for name, got in (("A1", a1), ("B", b), ("A2", a2))})
        print("  круг %d: мозг %7.2f /%7.2f /%7.2f -> в %.3f раза, "
              "порог круга %.3f;  шаг -> в %.3f раза, порог %.3f;  "
              "физика -> %.3f"
              % (i + 1, a1["мозг, мс"], b["мозг, мс"], a2["мозг, мс"],
                 mid["мозг, мс"] / b["мозг, мс"],
                 a1["мозг, мс"] / a2["мозг, мс"],
                 mid["шаг управления, мс"] / b["шаг управления, мс"],
                 a1["шаг управления, мс"] / a2["шаг управления, мс"],
                 mid["физика, мс на шаг физики"]
                 / b["физика, мс на шаг физики"]), flush=True)
        for name, got in (("A1", a1), ("A2", a2)):
            if got["отпечатки"] != b["отпечатки"]:
                rep.setdefault("находки", []).append(
                    "круг %d: эпизоды %s и B разошлись" % (i + 1, name))
    rep["замер шёл, с"] = round(time.perf_counter() - t_start, 1)
    rep["круги"] = triples

    rep["отношения csrmv/событийно"] = {
        k: ratios(rows, k) for k in ("мозг, мс", "шаг управления, мс",
                                     "физика, мс на шаг физики")}
    rep["порог доверия того же круга (A1/A2)"] = {
        k: ratios(aa, k) for k in ("мозг, мс", "шаг управления, мс")}

    print("\n— итог, отношениями внутри кругов —")
    for name, got in rep["отношения csrmv/событийно"].items():
        print("  csrmv/событийно по «%s»: в %.3f раза (%.3f…%.3f, %d кругов)"
              % (name, got["медиана"], got["мин"], got["макс"], got["пар"]))
    for name, got in rep["порог доверия того же круга (A1/A2)"].items():
        print("  csrmv/csrmv   по «%s»: %.3f (%.3f…%.3f, %d кругов) — порог"
              % (name, got["медиана"], got["мин"], got["макс"], got["пар"]))

    path = OUT / ("lif_event_contour_%s.json" % args.tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nотчёт:", path)
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
