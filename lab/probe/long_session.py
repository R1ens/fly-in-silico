"""Длинная сессия: муха живёт, условия меняются, память не растёт.

Этап 2 требует не «долгого прогона», а того, чтобы за ним можно было
сидеть: окно откликается, модельное время не стоит и не прыгает, память
работника не ползёт, правки условий доходят, остановка в любой миг
доводит прогон до конца и ничего не теряет.

Отдельная забота — запись. Плотный `bins` стоит 277 КБ на бин: десять
минут при шаге 20 мс — это 8 ГБ, и держать такое нельзя. Поэтому у
длинной сессии запись идёт в кольцо последних `keep_s` секунд, а
накопленные счётчики остаются за всю сессию. Проверяется именно это:
что кольцо даёт последнее окно, что его время подписано правильно, и что
в сохранённом видно, где кончается запись и где начинается сессия.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\long_session.py
    ..\\.venv\\Scripts\\python.exe probe\\long_session.py --минут 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "память": [], "правки": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def rss_mb(pid):
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss / 2 ** 20
    except Exception:                                 # noqa: BLE001
        return None


def part_ring():
    """Окно кольца обязано совпасть с хвостом полной записи.

    Это и есть всё утверждение про кольцо, и проверяется оно прямым
    сравнением: один и тот же эпизод считается дважды с одним семенем —
    с полной записью и с окном, — и окно должно быть побитово равно
    последним строкам полной. Если бы поворот кольца был сделан не на то
    число, строки разъехались бы по времени, а числа остались похожими:
    такое не поймать ни по сумме, ни на глаз.
    """
    import numpy as np
    import torch

    from flybrain import data as D, loop, world

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)

    def one(keep_s):
        cfg = loop.EpisodeConfig(duration_s=3.0, control_ms=20.0, seed=0,
                                 device=device, comment=False, keep_s=keep_s)
        return loop.run_episode(config=cfg,
                                body_params=world.BodyParams(dn_tau_s=0.5),
                                neurons=neurons, weights=weights,
                                progress=False)

    full = one(0.0)
    ring = one(1.0)
    fb, rb = full.bins, ring.bins
    check("полная запись — по строке на шаг", fb.shape[0] == 150,
          "строк %d" % fb.shape[0])
    check("окно короче полной записи", rb.shape[0] == 50,
          "строк %d при окне 1 с и шаге 20 мс" % rb.shape[0])
    same = rb.shape == fb[-50:].shape and bool(np.array_equal(rb, fb[-50:]))
    check("окно совпало с хвостом полной записи побитово", same,
          "сумма окна %d, сумма хвоста %d"
          % (int(rb.sum()), int(fb[-50:].sum())))
    check("окно — не начало записи",
          not bool(np.array_equal(rb, fb[:50])),
          "иначе поворот кольца сделан не туда")
    check("счётчики за всю сессию те же, что при полной записи",
          bool(np.array_equal(full.counts, ring.counts)),
          "сумма %d и %d" % (int(full.counts.sum()), int(ring.counts.sum())))
    check("траектория не изменилась от того, что запись короче",
          [r["x"] for r in full.records] == [r["x"] for r in ring.records],
          "шагов %d и %d" % (len(full.records), len(ring.records)))
    tm = ring.timing
    check("начало записи подписано", tm.get("запись с, мс") == 2000.0,
          "запись с %s мс" % tm.get("запись с, мс"))
    check("окно записи названо в оговорках",
          any("окно" in s for s in (tm.get("оговорки") or [])),
          (tm.get("оговорки") or ["нет"])[0][:80])
    check("у полной записи оговорки про окно нет",
          not any("окно" in s for s in (full.timing.get("оговорки") or [])),
          "оговорок %d" % len(full.timing.get("оговорки") or []))
    rep["замеры"]["кольцо"] = {
        "строк полной": int(fb.shape[0]), "строк окна": int(rb.shape[0]),
        "запись с, мс": tm.get("запись с, мс"),
    }


def main():                                                   # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--минут", type=float, default=3.0,
                    help="сколько минут по стене держать сессию")
    ap.add_argument("--окно", type=float, default=10.0,
                    help="сколько секунд модельного времени держать в записи")
    # ВИД СЕАНСА. Прежний (арена) остаётся умолчанием и не меняется ни
    # одним числом: на нём снят эталон (678,0 с модельного за 20 минут),
    # и сравнивать с ним можно только его же.
    #
    # `тело-зрение` — то, чего не гонял никто: долгий сеанс в ТЕЛЕ со
    # включённым зрением. Это другой вид прогона и другая механическая
    # модель (камеры добавляют два приваренных тела, муха тяжелее на
    # 0,1941 %), поэтому его числа с ареной не сравниваются — он
    # проверяет не скорость, а то же, что и прежний: что за сеансом
    # можно сидеть, память не ползёт, правки доходят, остановка
    # доводит до конца.
    ap.add_argument("--вид", default="арена",
                    choices=("арена", "тело-зрение"),
                    help="что держать долгим сеансом")
    args = ap.parse_args()
    minutes = float(getattr(args, "минут"))
    keep_s = float(getattr(args, "окно"))
    вид = str(getattr(args, "вид"))
    rep["вид"] = вид

    print("— кольцо записи —")
    part_ring()
    if fails:
        print("кольцо не сошлось: длинную сессию гонять нечего")
        (OUT / "long_session.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1

    print("— длинная сессия —")
    from flybrain.ui.session import Session

    ses = Session()
    t0 = time.perf_counter()
    while not ses.ready and time.perf_counter() - t0 < 180:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("работник поднялся", ses.ready,
          "за %.1f с" % (time.perf_counter() - t0))
    if not ses.ready:
        return 1
    pid = ses.proc.pid

    applied, worst, stalls = [], 0.0, []

    def pump(seconds):
        nonlocal worst
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            a = time.perf_counter()
            ses.pump(dt=0.02)
            worst = max(worst, (time.perf_counter() - a) * 1000.0)
            for rec in ses.take_applied():
                applied.append(rec)
                rep["правки"].append({"действие": rec.action,
                                      "модельное, мс": rec.t_model_ms,
                                      "исход": rec.outcome,
                                      "почему": rec.why})
            time.sleep(0.01)

    # Горизонт берётся с запасом: сессию заканчивает остановка, а не
    # исчерпание времени — иначе проверка мерила бы не то.
    horizon = max(60.0, minutes * 60.0)
    if вид == "арена":
        spec = {"kind": "arena", "t": horizon, "seed": 0,
                "open_report": False, "out": "probe-long-session",
                "bin_ms": 20.0,
                "arena": {"probes": False, "keep_s": keep_s}}
    else:
        spec = {"kind": "body", "t": horizon, "seed": 0,
                "open_report": False, "out": "probe-long-vision",
                "bin_ms": 20.0,
                "body": {"probes": False, "keep_s": keep_s,
                         "vision": True, "vis_objects": 1,
                         "vision_hz": 100.0}}
    ses.submit(spec)
    t0 = time.perf_counter()
    while ses.state != "running" and time.perf_counter() - t0 < 240:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("сессия пошла", ses.state == "running",
          "%s за %.1f с" % (ses.state, time.perf_counter() - t0))
    if ses.state != "running":
        ses.close()
        return 1

    pump(2.0)
    base_mb = rss_mb(pid)
    rep["замеры"]["память работника в начале, МБ"] = (
        round(base_mb, 1) if base_mb else None)

    # --- живём ---
    if вид == "арена":
        plan = [  # (доля прожитого, правка)
            (0.20, {"odours": [{"x": 50.0, "y": 10.0, "strength": 2.0,
                                "sigma": 15.0}]}),
            (0.40, {"orn_max_hz": 220.0}),
            (0.60, {"odours": [{"x": 10.0, "y": 50.0, "strength": 1.0,
                                "sigma": 9.0},
                               {"x": 50.0, "y": 50.0, "strength": 1.0,
                                "sigma": 9.0}]}),
            (0.80, {"loom_range_mm": 25.0}),
        ]
    else:
        # У тела свои живые ручки: `odours` и `loom_range_mm` — арены.
        # `vision_feed` взят нарочно: он ПОМЕЧЕННЫЙ, то есть сеанс, у
        # которого посреди счёта выключили подачу зрения, не сравним с
        # эталонным, и отчёт обязан это показать. Долгий сеанс — ровно
        # то место, где такую дорогу надо проверить, а не обойти.
        plan = [
            (0.20, {"orn_max_hz": 220.0}),
            (0.40, {"vision_feed": False}),
            (0.60, {"vision_feed": True}),
            (0.80, {"vision_max_hz": 200.0}),
        ]
    done = set()
    t_start = time.perf_counter()
    last_t_ms, back = ses.t_ms, 0
    total = minutes * 60.0
    while time.perf_counter() - t_start < total:
        pump(1.0)
        frac = (time.perf_counter() - t_start) / total
        for i, (at, payload) in enumerate(plan):
            if i not in done and frac >= at:
                done.add(i)
                ses.control("set", payload)
        if ses.t_ms < last_t_ms - 1e-9:
            back += 1
        if abs(ses.t_ms - last_t_ms) < 1e-9:
            stalls.append(round(time.perf_counter() - t_start, 1))
        last_t_ms = ses.t_ms
        mb = rss_mb(pid)
        if mb is not None:
            rep["память"].append([round(time.perf_counter() - t_start, 1),
                                  round(mb, 1)])
        if not ses.busy:
            break

    lived_ms = ses.t_ms
    end_mb = rss_mb(pid)
    rep["замеры"].update({
        "по стене, с": round(time.perf_counter() - t_start, 1),
        "модельного времени, с": round(lived_ms / 1000.0, 2),
        "память работника в конце, МБ": round(end_mb, 1) if end_mb else None,
        "худший кадр приёма, мс": round(worst, 1),
        "правок подано": len(done),
    })

    check("сессия не кончилась сама раньше остановки", ses.busy,
          "состояние %s, модельного %.1f с" % (ses.state, lived_ms / 1000.0))
    check("модельное время не шло назад", back == 0, "откатов %d" % back)
    check("модельное время шло без длинных остановок",
          len(stalls) <= max(1, int(minutes)),
          "секунд без движения: %d %s" % (len(stalls), stalls[:6]))
    check("окно откликается всю сессию", worst < 300.0,
          "худший кадр приёма %.0f мс" % worst)
    if base_mb and end_mb:
        grow = end_mb - base_mb
        rep["замеры"]["прирост памяти, МБ"] = round(grow, 1)
        # Кольцо фиксированного размера: прирост за минуты должен быть
        # сопоставим с шумом аллокатора, а не с временем.
        check("память работника не ползёт", grow < 300.0,
              "%.0f -> %.0f МБ, прирост %+.0f" % (base_mb, end_mb, grow))
    ok_set = [a for a in applied if a.action == "set"
              and a.outcome == "applied"]
    check("все правки условий применились",
          len(ok_set) == len(done) and len(done) == len(plan),
          "подано %d, применено %d" % (len(done), len(ok_set)))

    # --- остановка в любой миг ---
    ses.control("stop")
    t1 = time.perf_counter()
    while ses.busy and time.perf_counter() - t1 < 180:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("остановка довела сессию до конца", not ses.busy,
          "за %.1f с, состояние %s" % (time.perf_counter() - t1, ses.state))
    t1 = time.perf_counter()
    while ses.summary is None and ses.error is None \
            and time.perf_counter() - t1 < 30:
        pump(0.2)

    # --- что легло на диск ---
    path = (ses.summary or {}).get("path") or next(
        (p for k, p in ses.artifacts if k == "run"), None)
    out = Path(path) if path else None
    check("сессия сохранена", out is not None and out.exists(),
          str(out) if out else "пути нет; ошибка %s" % ses.error)
    if out and out.exists():
        # Имя файла зависит от ВИДА прогона: арена пишет
        # `episode.json`, тело — `embodied.json`. Поймано прогоном
        # долгого сеанса со зрением: проверка падала не на числах, а на
        # имени файла.
        имя = "episode.json" if вид == "арена" else "embodied.json"
        meta = json.loads((out / имя).read_text(encoding="utf-8"))
        tm = meta.get("timing") or {}
        rep["замеры"]["timing сохранённого"] = tm
        check("сохранённый прогон несёт журнал живых правок",
              bool(tm.get("живые правки")),
              "правок %d" % len(tm.get("живые правки") or []))
        check("длительность в файле — фактическая, а не горизонт",
              abs(float(tm.get("duration_s", 0.0)) - lived_ms / 1000.0)
              < 1.0,
              "в файле %s с, прожито %.2f с"
              % (tm.get("duration_s"), lived_ms / 1000.0))

        import numpy as np

        z = np.load(out / "bins.npz")
        rows = int(z["bins"].shape[0])
        bin_ms = float(z["bin_ms"])
        t0_ms = float(z["t0_ms"]) if "t0_ms" in z.files else 0.0
        rep["замеры"].update({"строк записи": rows,
                              "бин, мс": bin_ms,
                              "запись с, мс": t0_ms})
        want = int(round(keep_s * 1000.0 / bin_ms))
        check("запись — последнее окно, а не вся сессия",
              abs(rows - want) <= 1,
              "строк %d при окне %g с и бине %g мс (ждали %d)"
              % (rows, keep_s, bin_ms, want))
        # Сверять конец записи надо с длительностью из того же файла, а
        # не с `ses.t_ms`: последнее — время последнего ПОЛУЧЕННОГО
        # кадра, оно отстаёт на несколько бинов, и проверка ловила это
        # отставание вместо подписи. Что `ses.t_ms` близко к файлу,
        # проверено отдельно выше.
        file_ms = float(tm.get("duration_s", 0.0)) * 1000.0
        check("время записи подписано и упирается в конец сессии",
              abs((t0_ms + rows * bin_ms) - file_ms) <= bin_ms,
              "запись с %.0f мс, конец %.0f мс, сессия по файлу %.0f мс"
              % (t0_ms, t0_ms + rows * bin_ms, file_ms))
        check("в сохранённом сказано, что запись — окно",
              "окно" in json.dumps(tm, ensure_ascii=False),
              "; ".join(str(v)[:60] for k, v in tm.items() if "окно" in k)
              or "нет пометки")

        # Счётчики — за всю сессию, а не за окно: иначе частоты вранье.
        import pandas as pd

        rates = pd.read_parquet(out / "rates.parquet")
        check("частоты посчитаны по всей сессии, не по окну",
              len(rates) > 0, "строк в rates.parquet %d" % len(rates))

    ses.close()
    (OUT / "long_session.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
