"""Полнота состояния: продолжение побитово и выбрасывание по одному.

Что здесь мерится, двумя числами.

**Первое — точность продолжения.** Один и тот же эпизод считается
дважды: непрерывно (`A`) и с остановкой на сохранённом состоянии и
продолжением с него (`B`). Сравниваются не «похожие кривые», а байты:
запись каждого шага, накопленные счётчики спайков и всё состояние движка
на конце — мембрана, проводимость, спайки, рефрактерный счётчик, кольцо
задержек, указатель кольца и состояние генератора. Требование —
побитовое совпадение на ``TAIL_STEPS`` шагов управления, то есть шесть
постоянных сглаживания при tau 0,5 с.

**Второе — полнота перечня.** Утверждение «в состоянии есть всё нужное»
проверяется не рассуждением, а **выбрасыванием каждого элемента по
одному** (:meth:`flybrain.replay.State.drop`). Элемент, выбрасывание
которого ничего не меняет, в состояние не входит — и это находка, о
которой проба говорит прямо, а не замалчивает.

Выбрасывание здесь — ещё и **встречный замер на сам сравниватель**.
«Совпало побитово» на пустом сравнивателе печаталось бы точно так же;
девять выбрасываний показывают, что он расхождение ловит и на каком
шаге.

Чтобы маски вмешательств не были тривиальными (все единицы — тогда их
потеря ничего не меняет по построению, а не по замеру), эпизод считается
с настоящим вмешательством: мембрана части нейронов удержана.

    cd D:\\fly-brain\\lab
    ..\\.venv\\Scripts\\python.exe probe\\resume_state.py --part arena
    ..\\.venv\\Scripts\\python.exe probe\\resume_state.py --part body
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent

# Где снимается состояние (шаг управления) и сколько шагов продолжения
# сверяется побитово. 150 шагов — это 3 с модельного времени, шесть
# постоянных сглаживания выхода популяции (tau 0,5 с): меньше нельзя,
# фильтр за это время успевает забыть начальное значение.
CAPTURE_STEP = 30
TAIL_STEPS = 150
CONTROL_MS = 20.0

# Сколько нейронов удержать вмешательством. Нужно одному: чтобы маски
# были непустыми и их потеря мерилась, а не следовала из того, что они
# и так все единицы.
HOLD_N = 500

fails = []
rep = {"проверки": {}, "замеры": {}, "выбрасывание": {}, "находки": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def fp(a):
    """Отпечаток от сырых байтов: ловит расхождение в последнем бите."""
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def rec_fp(rec):
    """Отпечаток записи шага. Числа идут repr'ом, то есть без потерь.

    ``t_ms`` из отпечатка исключён нарочно, и это не послабление. В
    записи эпизода он считается от начала ЭТОГО счёта (``step *
    control_ms``), поэтому у продолжения он всегда начинается с нуля: по
    нему разошлись бы все 150 шагов, и сверка измеряла бы нумерацию, а
    не модель. Модельное время продолжения проверяется отдельно и тоже
    побитово — по оси таймлайна, куда его кладёт ``Tape.t_offset_ms``.
    """
    body = json.dumps({k: v for k, v in rec.items() if k != "t_ms"},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def eng_fp(snap):
    """Отпечатки состояния движка по полям. Словарь, чтобы видеть, что чем."""
    out = {}
    for k, v in snap.items():
        if k == "calls":
            continue
        out[k] = (int(v) if np.ndim(v) == 0 and k == "ptr" else fp(v))
    return out


def first_diff(a, b):
    """Номер первого расхождения в двух списках отпечатков или None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ============================================================== арена


def _arena_setup(device):
    from flybrain import data as D, sim, world

    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    sensors = np.unique(np.concatenate(
        [D.select(neurons, q) for q in world.SENSORS.values()]))
    # Кого удержать: берём подряд из середины таблицы, чтобы это были не
    # сенсорные нейроны (им движок отменяет рефрактерность, и маска у них
    # вела бы себя иначе).
    hold = np.arange(60000, 60000 + HOLD_N, dtype=np.int64)
    return neurons, weights, sensors, hold


def _arena_engine(weights, neurons, sensors, hold, device, seed):
    from flybrain import sim

    eng = sim.build_engine(weights=weights, neurons=neurons, device=device,
                           seed=seed, no_refractory=sensors)
    eng.set_masks(hold=hold)
    eng.calls = 0
    return eng


def part_arena(device):
    from flybrain import data as D, loop, replay as RP, sim, world

    neurons, weights, sensors, hold = _arena_setup(device)
    seed = 0
    n_steps = CAPTURE_STEP + 1 + TAIL_STEPS
    cfg = loop.EpisodeConfig(duration_s=n_steps * CONTROL_MS / 1000.0,
                             control_ms=CONTROL_MS, seed=seed, device=device,
                             comment=False)
    # Опора и усиление поворота нарочно не заданы: тогда эпизод мерит их
    # на разогреве, и калибровка становится настоящим элементом
    # состояния — её потеря обязана быть видна.
    params = world.BodyParams(dn_ref_hz=None, turn_gain_rad_s=None)
    arena = world.default_arena(seed)
    probe_idx = {"съём L": np.asarray(D.select(neurons, params.motor_left),
                                      np.int64)}

    print("  эпизод A: непрерывно, %d шагов, снимок на шаге %d"
          % (n_steps, CAPTURE_STEP))
    engA = _arena_engine(weights, neurons, sensors, hold, device, seed)
    tapeA = RP.Tape(kind="arena", control_ms=CONTROL_MS, keep_s=20.0,
                    probe_idx=probe_idx, mode="пересчёт",
                    capture_at_ms=CAPTURE_STEP * CONTROL_MS,
                    meta={"семя": seed})
    t0 = time.perf_counter()
    epA = loop.run_episode(arena=arena, config=cfg, body_params=params,
                           neurons=neurons, weights=weights, progress=False,
                           engine=engA, tape=tapeA)
    wallA = time.perf_counter() - t0
    state = epA.state
    rep["замеры"]["шагов в A"] = len(epA.records)
    rep["замеры"]["A, с"] = round(wallA, 2)
    rep["замеры"]["вызовов движка в A"] = int(engA.calls)
    check("снимок состояния снят на границе шага",
          state is not None and abs(state.t_ms
                                    - CAPTURE_STEP * CONTROL_MS) < 1e-9,
          "модельное время снимка %s мс" % (state and state.t_ms))
    if state is None:
        return
    check("снимок полон: ни одного обязательного элемента не недостаёт",
          state.complete(), "элементов %d: %s; не хватает: %s"
          % (len(state.have), ", ".join(sorted(state.have)),
             ", ".join(state.missing()) or "ничего"))
    rep["замеры"]["элементы снимка"] = sorted(state.have)

    # Снимок через диск: продолжение из файла, а не из живого объекта.
    where = Path(probe._isolate.OUT) / "_resume_arena"
    state.save(where)
    state2 = RP.State.load(where)
    check("снимок пишется и читается без потерь",
          state2.complete() and sorted(state2.have) == sorted(state.have)
          and abs(state2.t_ms - state.t_ms) < 1e-12,
          "элементов %d, модельное %s мс" % (len(state2.have), state2.t_ms))
    bad = [k for k in sorted(state.arrays)
           for f in sorted(state.arrays[k])
           if f in (state2.arrays.get(k) or {})
           and fp(state.arrays[k][f]) != fp(state2.arrays[k][f])]
    check("массивы снимка после диска побитово те же", not bad,
          "полей сверено %d, разошлось %d"
          % (sum(len(v) for v in state.arrays.values()), len(bad)))

    ref_recs = [rec_fp(r) for r in epA.records[CAPTURE_STEP + 1:]]
    ref_eng = eng_fp(engA.snapshot())
    ref_counts = fp(np.asarray(epA.counts, np.int64))
    ref_t = np.asarray(tapeA.tl.t_ms[CAPTURE_STEP + 1:], np.float64)
    rep["замеры"]["шагов для сверки"] = len(ref_recs)

    print("  эпизод B: продолжение из снимка, %d шагов" % TAIL_STEPS)
    got = _resume_arena(state2, neurons, weights, device, drop=(),
                        n_steps=TAIL_STEPS)
    same = first_diff(ref_recs, got["recs"])
    rep["замеры"]["продолжение: первое расхождение записи"] = same
    check("продолжение побитово равно непрерывному счёту: записи шагов",
          same is None and len(got["recs"]) == len(ref_recs),
          ("%d шагов из %d совпали побитово (%g с модельного времени)"
           % (len(got["recs"]), len(ref_recs),
              len(ref_recs) * CONTROL_MS / 1000.0)) if same is None else
          ("разошлось с шага %d из %d" % (same, len(ref_recs))))
    check("ось модельного времени продолжения побитово продолжает прежнюю",
          fp(ref_t) == fp(got["t_ms"][:len(ref_t)])
          and len(got["t_ms"]) == len(ref_t),
          "с %g по %g мс, отпечаток %s"
          % (float(got["t_ms"][0]), float(got["t_ms"][-1]),
             fp(got["t_ms"])))
    bad = [k for k, v in ref_eng.items() if got["eng"].get(k) != v]
    check("состояние движка на конце побитово то же", not bad,
          "полей %d, разошлось %d%s" % (len(ref_eng), len(bad),
                                        (": " + ", ".join(bad)) if bad else ""))
    check("накопленные счётчики спайков побитово те же",
          got["counts"] == ref_counts,
          "отпечаток %s против %s" % (got["counts"], ref_counts))
    check("вызовов движка на шаг управления ровно столько, сколько стоит шаг",
          got["calls_per_step"] == {RP.lif_per_step(CONTROL_MS)},
          "по записи: %s (ожидалось %d)"
          % (sorted(got["calls_per_step"]), RP.lif_per_step(CONTROL_MS)))

    # ---- выбрасывание по одному ----
    print("\n  — выбрасывание по одному (встречный замер на сравниватель) —")
    drops = [
        ("указатель кольца задержек", ("ptr",)),
        ("состояние генератора", ("gen",)),
        ("маски вмешательств", tuple(sim.Engine.MASK_TENSORS)),
        ("тензоры состояния движка", ("engine",)),
        ("накопитель сглаживания", ("motor",)),
        ("калибровка разогрева", ("calib",)),
        ("поза", ("pose",)),
        ("среда", ("env",)),
        ("счётчики сеанса", ("counts",)),
    ]
    for name, what in drops:
        cut = state2.drop(*what)
        try:
            g = _resume_arena(cut, neurons, weights, device, drop=what,
                              n_steps=TAIL_STEPS)
        except Exception as exc:                              # noqa: BLE001
            rep["выбрасывание"][name] = {"выброшено": list(what),
                                         "отказ": str(exc)[:300]}
            print("    %-32s отказ: %s" % (name, str(exc)[:80]))
            continue
        d = first_diff(ref_recs, g["recs"])
        bad_eng = [k for k, v in ref_eng.items() if g["eng"].get(k) != v]
        same_counts = g["counts"] == ref_counts
        # Элемент считается замеченным, если изменился хоть один байт
        # хоть где: запись шагов, состояние движка или накопленные
        # счётчики. Смотреть только на записи шагов нельзя — счётчики
        # сеанса в них не входят вовсе, и их потеря прошла бы мимо.
        seen = (d is not None) or bool(bad_eng) or (not same_counts)
        rep["выбрасывание"][name] = {
            "выброшено": list(what),
            "первое расхождение записи, шаг": d,
            "полей состояния движка разошлось": len(bad_eng),
            "какие": bad_eng,
            "счётчики совпали": same_counts,
            "замечено": bool(seen),
        }
        print("    %-32s расхождение с шага %-5s полей движка %-2d "
              "счётчики %s"
              % (name, "нет" if d is None else d, len(bad_eng),
                 "те же" if same_counts else "другие"))
        if not seen:
            rep["находки"].append(
                "выбрасывание «%s» (%s) НЕ изменило ни одного байта на %d "
                "шагах: по этому замеру элемент в состояние продолжения не "
                "входит" % (name, ", ".join(what), TAIL_STEPS))

    got_any = [v for v in rep["выбрасывание"].values()
               if v.get("замечено") or v.get("отказ")]
    check("встречный замер: сравниватель ловит потерю каждого элемента",
          len(got_any) == len(rep["выбрасывание"]),
          "из %d выбрасываний замечено %d"
          % (len(rep["выбрасывание"]), len(got_any)))


def _resume_arena(state, neurons, weights, device, drop, n_steps):
    """Продолжить эпизод из состояния и снять отпечатки."""
    from flybrain import loop, replay as RP

    engine, motor, fly, arena, params, counts, report = RP.restore_arena(
        state, neurons, weights=weights, device=device, drop=drop)
    cfg = loop.EpisodeConfig(duration_s=n_steps * CONTROL_MS / 1000.0,
                             control_ms=CONTROL_MS,
                             seed=int(state.meta.get("семя") or 0),
                             device=device, comment=False)
    tape = RP.Tape(kind="arena", control_ms=CONTROL_MS, keep_s=20.0,
                   mode="продолжение",
                   t_offset_ms=state.t_ms + CONTROL_MS,
                   meta={"семя": cfg.seed})
    ep = loop.run_episode(arena=arena, config=cfg, body_params=params,
                          neurons=neurons, weights=weights, progress=False,
                          start=fly, engine=engine, motor=motor,
                          counts0=counts, tape=tape)
    col = tape.tl.col.get("вызовов_движка")
    calls = set()
    if col is not None:
        calls = {int(v) for v in tape.tl.rows[:, col]}
    return {"recs": [rec_fp(r) for r in ep.records],
            "eng": eng_fp(engine.snapshot()),
            "counts": fp(np.asarray(ep.counts, np.int64)),
            "t_ms": np.asarray(tape.tl.t_ms, np.float64),
            "calls_per_step": calls,
            "отчёт": report}


# =============================================================== тело


def part_body(device, tail=25):
    """То же в теле: горизонт продолжения и выбрасывание фазы походки.

    Здесь требование «150 шагов побитово» заранее не выполняется, и это
    замер предыдущего исполнителя, который надо перепроверить своим:
    `replay.EXACT_STEPS['body']` = 21 шаг. Поэтому хвост короткий, а
    итог — число шагов, на которых совпало, и честное «дальше
    расходится».
    """
    from flybrain import body as B, data as D, replay as RP, sim

    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    cfg = B.BodyConfig(duration_s=(CAPTURE_STEP + 1 + tail) * CONTROL_MS
                       / 1000.0,
                       control_ms=CONTROL_MS, seed=0, device=device)
    rep["замеры"]["тело: шагов"] = CAPTURE_STEP + 1 + tail
    tapeA = RP.Tape(kind="body", control_ms=CONTROL_MS, keep_s=20.0,
                    mode="пересчёт",
                    capture_at_ms=CAPTURE_STEP * CONTROL_MS,
                    meta={"семя": 0})
    print("  сеанс A: непрерывно, снимок на шаге %d" % CAPTURE_STEP)
    t0 = time.perf_counter()
    runA = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                         progress=False, tape=tapeA)
    rep["замеры"]["тело: A, с"] = round(time.perf_counter() - t0, 2)
    state = runA.state
    check("тело: снимок снят на границе шага",
          state is not None
          and abs(state.t_ms - CAPTURE_STEP * CONTROL_MS) < 1e-9,
          "модельное время снимка %s мс" % (state and state.t_ms))
    if state is None:
        return
    check("тело: снимок полон", state.complete(),
          "элементов %d, не хватает: %s"
          % (len(state.have), ", ".join(state.missing()) or "ничего"))
    where = Path(probe._isolate.OUT) / "_resume_body"
    state.save(where)
    state2 = RP.State.load(where)

    ref = [rec_fp(r) for r in runA.records[CAPTURE_STEP + 1:]]
    print("  сеанс B: продолжение из снимка, %d шагов" % tail)
    got = _resume_body(state2, neurons, weights, device, (), tail)
    d = first_diff(ref, got["recs"])
    n_same = len(ref) if d is None else d
    rep["замеры"]["тело: шагов совпало побитово"] = n_same
    rep["замеры"]["тело: объявлено в replay.EXACT_STEPS"] = \
        RP.EXACT_STEPS["body"]
    check("тело: продолжение совпадает побитово не меньше объявленного "
          "горизонта",
          n_same >= RP.EXACT_STEPS["body"],
          "совпало %d шагов из %d проверенных, объявлено %d"
          % (n_same, len(ref), RP.EXACT_STEPS["body"]))
    if n_same < len(ref):
        rep["находки"].append(
            "тело: продолжение побитово держится %d шагов управления "
            "(%.0f мс) из %d проверенных, дальше расходится. Требование "
            "«150 шагов» в теле НЕ выполняется, и это свойство сеанса в "
            "MuJoCo, а не таймлайна: все названные поля состояния в миг "
            "восстановления совпадают." % (n_same, n_same * CONTROL_MS,
                                           len(ref)))

    print("\n  — выбрасывание по одному (тело) —")
    for name, what in (("фаза CPG", ("cpg",)),
                       ("физика MuJoCo", ("mujoco",)),
                       ("калибровка разогрева", ("calib",)),
                       ("накопитель сглаживания", ("smooth",)),
                       ("указатель кольца задержек", ("ptr",)),
                       ("состояние генератора", ("gen",)),
                       ("маски вмешательств", tuple(sim.Engine.MASK_TENSORS))):
        cut = state2.drop(*what)
        try:
            g = _resume_body(cut, neurons, weights, device, what, tail)
        except Exception as exc:                              # noqa: BLE001
            rep["выбрасывание"]["тело: " + name] = {
                "выброшено": list(what), "отказ": str(exc)[:300]}
            print("    %-28s отказ: %s" % (name, str(exc)[:80]))
            continue
        dd = first_diff(ref, g["recs"])
        same = len(ref) if dd is None else dd
        rep["выбрасывание"]["тело: " + name] = {
            "выброшено": list(what),
            "шагов совпало": same,
            "совпало с целым снимком": same == n_same,
        }
        print("    %-28s совпало %d шагов (с целым снимком %d)"
              % (name, same, n_same))
        if same >= n_same:
            rep["находки"].append(
                "тело: выбрасывание «%s» (%s) не укоротило побитовое "
                "совпадение (%d шагов против %d у целого снимка): по этому "
                "замеру элемент на первые шаги продолжения не влияет"
                % (name, ", ".join(what), same, n_same))


def _resume_body(state, neurons, weights, device, drop, n_steps):
    from flybrain import body as B, data as D, replay as RP, sim, world

    cfg = B.BodyConfig(duration_s=n_steps * CONTROL_MS / 1000.0,
                       control_ms=CONTROL_MS,
                       seed=int(state.meta.get("семя") or 0), device=device)
    # Те же три селектора, что собирает сам `body.run_episode`: запах с
    # двух антенн и вкус. Движок обязан быть собран с той же отменой
    # рефрактерности, иначе продолжение считало бы другую модель.
    driven = np.unique(np.concatenate([
        D.select(neurons, world.SENSORS[k])
        for k in ("odour_left", "odour_right", "taste")]))
    engine = sim.build_engine(weights=weights, neurons=neurons,
                              device=device, seed=cfg.seed,
                              no_refractory=driven)
    tape = RP.Tape(kind="body", control_ms=CONTROL_MS, keep_s=20.0,
                   mode="продолжение", t_offset_ms=state.t_ms + CONTROL_MS,
                   meta={"семя": cfg.seed})
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False, engine=engine, resume=state,
                        counts0=(state.arrays.get("counts") or {}).get(
                            "counts"),
                        tape=tape)
    # Выбрасывание внутри движка делается самим `restore_body` по `drop`,
    # поэтому здесь оно передаётся не сюда, а туда — через снимок с
    # выброшенным элементом. Параметр оставлен для отчёта.
    return {"recs": [rec_fp(r) for r in run.records], "drop": list(drop)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="arena",
                    choices=("arena", "body", "all"))
    ap.add_argument("--tail", type=int, default=25,
                    help="хвост сверки в теле, шагов управления")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep["замеры"]["устройство"] = device
    rep["замеры"]["шаг управления, мс"] = CONTROL_MS
    rep["замеры"]["снимок на шаге"] = CAPTURE_STEP
    rep["замеры"]["хвост сверки, шагов"] = TAIL_STEPS

    # Перечень полей движка в `replay` объявлен литерально, чтобы
    # просмотр записи обходился без torch. Разъехаться с `sim.Engine`
    # ему нельзя, и сверяется это здесь, а не доверием.
    from flybrain import replay as RP

    ok, bad = RP.engine_lists_agree()
    check("перечень полей состояния движка в replay совпадает с sim.Engine",
          ok, "состояние %d полей, маски %d полей%s"
          % (len(RP.ENGINE_STATE), len(RP.ENGINE_MASKS),
             ("; разошлось: " + "; ".join(bad)) if bad else ""))

    if args.part in ("arena", "all"):
        print("— арена: продолжение из состояния —")
        part_arena(device)
    if args.part in ("body", "all"):
        print("\n— тело: продолжение из состояния —")
        part_body(device, tail=args.tail)

    name = "resume_state" + ("" if args.part == "all" else "_" + args.part)
    (OUT / (name + ".json")).write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    for n in rep["находки"]:
        print("  замечание: " + n)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
