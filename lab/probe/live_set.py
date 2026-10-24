"""Живые условия: правка из окна доходит до идущей модели или честно отказывает.

Две половины, и первая не требует ни видеокарты, ни работника.

**Проверка значений.** Что можно менять — перечислено (`live.KNOBS`), и
проверяется именно отказ: неизвестное поле, значение за пределом, мусор
вместо числа, слишком много источников. Главное здесь — что отказ
ничего не меняет: наполовину применённая правка дала бы состояние,
которого нет ни в одном описании опыта.

**Правка доходит до модели.** Второй половиной запускается настоящий
эпизод на арене через работника, и условия меняются на ходу: источник
запаха переносится, усиление рецепторов меняется, антенны меняются
местами. Проверяется не факт отправки, а то, что модель это увидела —
по записи шагов, пришедшей из прогона, и по подтверждению с модельным
временем.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\live_set.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}, "подтверждения": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


# ------------------------------------------------------- проверка значений

def part_values():
    from flybrain import live, world

    arena = world.Arena(foods=[world.Food(10.0, 10.0)],
                        odours=[world.Odour(20.0, 30.0)])
    params = world.BodyParams()
    st = live.ArenaLive(arena, params)

    before = st.snapshot()
    check("перечень разрешённого не пуст и описан",
          len(live.KNOBS) >= 10
          and all(k.what for k in live.KNOBS.values()),
          "%d свойств" % len(live.KNOBS))

    got = st.apply({"orn_max_hz": 200.0})
    check("обычная правка применяется",
          got["orn_max_hz"] == 200.0 and params.orn_max_hz == 200.0,
          "orn_max_hz %g -> %g" % (before["orn_max_hz"], params.orn_max_hz))

    bad = [
        ("неизвестное свойство", {"цвет_крыльев": 1}, KeyError),
        ("значение за пределом", {"orn_max_hz": 1e6}, ValueError),
        ("мусор вместо числа", {"orn_max_hz": "много"}, ValueError),
        ("бесконечность", {"orn_max_hz": float("inf")}, ValueError),
        ("пустая правка", {}, ValueError),
        ("не словарь", ["orn_max_hz", 1], TypeError),
        ("нулевая сигма источника",
         {"odours": [{"x": 1, "y": 2, "sigma": 0.0}]}, ValueError),
        ("лишнее поле у источника",
         {"odours": [{"x": 1, "y": 2, "яркость": 3}]}, ValueError),
        ("источников больше предела",
         {"odours": [{"x": 0, "y": 0}] * (live.MAX_SOURCES + 1)}, ValueError),
        ("источник не словарь", {"odours": [[1, 2]]}, ValueError),
    ]
    for name, payload, exc in bad:
        was = st.snapshot()
        try:
            st.apply(payload)
            check("отказ: " + name, False, "правка прошла, а не должна была")
        except exc as e:
            check("отказ: " + name, st.snapshot() == was,
                  "%s, состояние не изменилось" % str(e)[:60])
        except Exception as e:                       # noqa: BLE001
            check("отказ: " + name, False,
                  "другое исключение %s: %s" % (type(e).__name__, e))

    # Отказ посреди многополевой правки не должен применить первую половину.
    was = st.snapshot()
    try:
        st.apply({"orn_max_hz": 111.0, "grn_hz": -5.0})
    except ValueError:
        pass
    check("отказ в середине правки не применяет её начало",
          st.snapshot() == was,
          "orn_max_hz остался %g" % params.orn_max_hz)

    got = st.apply({"odours": [{"x": 1.0, "y": 2.0, "strength": 0.5,
                                "sigma": 7.0},
                               {"x": -3.0, "y": 4.0}]})
    check("источники заменяются целиком",
          len(arena.odours) == 2 and arena.odours[0].sigma == 7.0
          and arena.odours[1].sigma == 12.0,
          "стало %d источников, сигмы %s"
          % (len(arena.odours), [o.sigma for o in arena.odours]))
    check("пустой список убирает источники",
          st.apply({"odours": []})["odours"] == []
          and arena.odours == [],
          "источников %d" % len(arena.odours))

    st.apply({"swap_antennae": True})
    marked = st.log[-1]
    check("правка, меняющая смысл, помечена",
          marked.get("помечено") == ["swap_antennae"]
          and "оговорка" in marked,
          marked.get("оговорка", "")[:70])
    check("обычная правка оговорки не несёт",
          "оговорка" not in st.log[0],
          "первая правка: %s" % sorted(st.log[0]))
    check("журнал правок хранит было и стало",
          all({"было", "стало"} <= set(r) for r in st.log),
          "правок в журнале %d" % len(st.log))

    sch = {s["имя"]: s for s in st.schema()}
    check("описание свойств отдаёт текущие значения",
          sch["orn_max_hz"]["сейчас"] == params.orn_max_hz
          and sch["swap_antennae"]["сейчас"] is True,
          "orn_max_hz %g" % sch["orn_max_hz"]["сейчас"])

    # Движок читает это один раз при сборке — менять нельзя вообще.
    for name in ("seed", "device", "control_ms", "readout"):
        try:
            st.apply({name: 1})
            check("нельзя менять «%s» на ходу" % name, False, "прошло")
        except KeyError as e:
            check("нельзя менять «%s» на ходу" % name, True, str(e)[:48])


# ------------------------------------------------- правка доходит до модели

def part_live():                                              # noqa: C901
    from flybrain.ui.session import Session

    ses = Session()
    t0 = time.perf_counter()
    while not ses.ready and time.perf_counter() - t0 < 180:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("работник поднялся", ses.ready,
          "за %.1f с" % (time.perf_counter() - t0))
    if not ses.ready:
        return

    applied = []

    def pump(seconds):
        end = time.perf_counter() + seconds
        worst = 0.0
        while time.perf_counter() < end:
            a = time.perf_counter()
            ses.pump(dt=0.02)
            worst = max(worst, (time.perf_counter() - a) * 1000.0)
            for rec in ses.take_applied():
                applied.append(rec)
                rep["подтверждения"].append(
                    {"действие": rec.action, "модельное, мс": rec.t_model_ms,
                     "исход": rec.outcome, "значения": rec.values,
                     "почему": rec.why})
            time.sleep(0.01)
        return worst

    spec = {"kind": "arena", "t": 12.0, "seed": 0, "open_report": False,
            "out": "probe-live-set", "arena": {"probes": False}}
    job = ses.submit(spec)
    t0 = time.perf_counter()
    while ses.state != "running" and time.perf_counter() - t0 < 180:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("эпизод пошёл", ses.state == "running",
          "%s за %.1f с" % (ses.state, time.perf_counter() - t0))
    if ses.state != "running":
        ses.close()
        return

    pump(1.5)
    control_ms = float(ses.bin_ms)

    # 1. Отказ: модель идёт дальше, подтверждение несёт причину.
    sent = ses.t_ms
    ses.control("set", {"цвет_крыльев": 7})
    pump(1.5)
    bad = [a for a in applied if a.action == "set"
           and a.outcome == "rejected"]
    check("негодная правка отклонена с причиной", bool(bad),
          (bad[0].why[:70] if bad else "подтверждения нет"))
    check("после отказа прогон продолжается",
          ses.state == "running" and ses.t_ms > sent,
          "модельное %.0f -> %.0f мс" % (sent, ses.t_ms))

    # 2. Перенос источника запаха: вход рецепторов обязан измениться.
    before = dict(ses.extra or {})
    sent = ses.t_ms
    ses.control("set", {"odours": [{"x": 5.0, "y": 5.0, "strength": 3.0,
                                    "sigma": 20.0}]})
    pump(2.5)
    ok_set = [a for a in applied if a.action == "set"
              and a.outcome == "applied"]
    check("правка подтверждена модельным временем", bool(ok_set),
          "подана на %.0f мс, применена на %s"
          % (sent, ok_set[0].t_model_ms if ok_set else "—"))
    if ok_set:
        # Чем это НЕ может быть проверено. `ses.t_ms` — время последнего
        # полученного кадра, а кадры идут с отставанием и при переполнении
        # отбрасываются. Значит «подано на» заведомо раньше настоящего
        # модельного времени подачи, и разность подача->применение меньше
        # одного шага быть не обязана. Первая версия этой проверки
        # требовала одного шага и падала на двух — врал замер, а не код.
        #
        # Что проверяемо: применение не раньше наблюдённого времени, не
        # позже нескольких шагов, и строго на границе шага управления —
        # последнее и есть то свойство, из-за которого правку вообще
        # можно применять безопасно.
        lag = ok_set[0].t_model_ms - sent
        rep["замеры"]["задержка правки, мс"] = lag
        rep["замеры"]["шаг управления, мс"] = control_ms
        check("правка применена не раньше наблюдённого времени и близко к нему",
              0.0 <= lag <= 4.0 * control_ms + 1e-6,
              "%.1f мс при шаге %.1f мс" % (lag, control_ms))
        on_edge = abs((ok_set[0].t_model_ms / control_ms)
                      - round(ok_set[0].t_model_ms / control_ms)) < 1e-6
        check("правка применена ровно на границе шага управления", on_edge,
              "%.1f мс кратно %.1f" % (ok_set[0].t_model_ms, control_ms))
        check("подтверждение несёт применённые значения",
              isinstance(ok_set[0].values, dict)
              and "odours" in (ok_set[0].values or {}),
              str(ok_set[0].values)[:80])
    after = dict(ses.extra or {})
    moved = (abs(float(after.get("odour_l", 0))
                 - float(before.get("odour_l", 0))) > 1e-9
             or abs(float(after.get("odour_r", 0))
                    - float(before.get("odour_r", 0))) > 1e-9)
    check("перенос источника изменил вход рецепторов", moved,
          "было odour_l %s, стало %s"
          % (before.get("odour_l"), after.get("odour_l")))

    # 3. Помеченная правка: смысл прогона изменился, и это видно.
    ses.control("set", {"swap_antennae": True})
    pump(2.0)
    sw = [a for a in applied if a.action == "set"
          and (a.values or {}).get("swap_antennae") is True]
    check("помеченная правка применилась", bool(sw),
          "подтверждений %d" % len(sw))

    ses.control("stop")
    t1 = time.perf_counter()
    while ses.busy and time.perf_counter() - t1 < 90:
        ses.pump(dt=0.02)
        time.sleep(0.02)
    check("прогон остановлен и доведён до конца", not ses.busy,
          "за %.1f с" % (time.perf_counter() - t1))
    # Состояние «не занят» и пришедшая сводка — не одно и то же событие.
    # Без этого ожидания проверка сохранения падала через раз: выход из
    # цикла случался раньше, чем разбиралось сообщение со путём.
    t1 = time.perf_counter()
    while ses.summary is None and ses.error is None \
            and time.perf_counter() - t1 < 15:
        pump(0.1)

    # 4. Правка — часть результата: она обязана лежать в сохранённом файле.
    path = (ses.summary or {}).get("path") or next(
        (p for k, p in ses.artifacts if k == "run"), None)
    out = Path(path) if path else None
    meta = None
    if out and (out / "episode.json").exists():
        meta = json.loads((out / "episode.json").read_text(encoding="utf-8"))
    check("прогон сохранён", meta is not None,
          str(out) if out else "пути нет; состояние %s, ошибка %s; журнал: %s"
          % (ses.state, ses.error, " | ".join(ses.log[-4:])))
    if meta:
        edits = (meta.get("timing") or {}).get("живые правки") \
            or meta.get("живые правки")
        check("сохранённый прогон несёт журнал живых правок",
              bool(edits), "правок записано %s"
              % (len(edits) if edits else 0))
        note = json.dumps(meta, ensure_ascii=False)
        check("в сохранённом есть оговорка о помеченной правке",
              "оговорка" in note, "найдена" if "оговорка" in note else "нет")

    ses.close()


def part_same_arena():
    """Арена, собранная снаружи, даёт тот же эпизод.

    `_job_arena` теперь строит арену сам и отдаёт её в `run_episode` —
    иначе живым условиям нечего менять. Раньше её строил сам
    `run_episode` из того же семени. Равенство здесь очевидно по коду
    (одна и та же функция, одно и то же семя), но замороженный эталон
    этой подстановки не проверяет, а «очевидно» уже один раз стоило
    падения всех сохранений. Поэтому — замером.
    """
    import hashlib

    import numpy as np
    import torch

    from flybrain import data as D, loop, world

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)

    def one(arena):
        cfg = loop.EpisodeConfig(duration_s=1.0, control_ms=20.0, seed=0,
                                 device=device, comment=False)
        ep = loop.run_episode(arena=arena, config=cfg,
                              body_params=world.BodyParams(dn_tau_s=0.5),
                              neurons=neurons, weights=weights,
                              progress=False)
        tab = np.array([[r["x"], r["y"], r["theta"], r["dn_l"], r["dn_r"],
                         r["odour_l"], r["odour_r"]] for r in ep.records],
                       dtype=np.float64)
        return (hashlib.sha256(tab.tobytes()).hexdigest()[:16],
                hashlib.sha256(np.ascontiguousarray(ep.counts)
                               .tobytes()).hexdigest()[:16])

    inside = one(None)
    outside = one(world.default_arena(0))
    check("арена снаружи даёт тот же эпизод", inside == outside,
          "%s против %s" % (inside, outside))


def main():
    print("— проверка значений —")
    part_values()
    print("— арена снаружи —")
    part_same_arena()
    print("— правка доходит до модели —")
    part_live()
    (OUT / "live_set.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
